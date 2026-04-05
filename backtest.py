#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 回测验证引擎

评估历史推荐的实际表现，形成反馈闭环。
支持多数据源降级（DataEngine）和 per-model 绩效追踪。

用法:
  python backtest.py                # 回测最近一次推荐
  python backtest.py --days 5       # 回测5个交易日后的表现
  python backtest.py --date 20260310 # 回测指定日期的推荐
  python backtest.py --model-perf   # 输出各模型推荐绩效
"""

import argparse
import json
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional


class BacktestEngine:
    """回测验证引擎（支持多数据源 + per-model追踪）"""

    def __init__(self, data_dir="output/data", tushare_token=""):
        self.data_dir = data_dir
        self.tushare_token = tushare_token
        self._engine = None

    def _get_engine(self):
        """懒加载 DataEngine"""
        if self._engine is None:
            try:
                from data_engine import DataEngine
                self._engine = DataEngine(
                    tushare_token=self.tushare_token,
                )
            except ImportError:
                self._engine = None
        return self._engine

    def _fetch_close_price(self, code: str, target_date: str) -> Optional[float]:
        """
        获取某只股票在 target_date 当日或之后最近交易日的收盘价。
        优先用 DataEngine（多源降级），失败时 fallback 到 tushare。
        """
        engine = self._get_engine()
        if engine is not None:
            try:
                df = engine.fetch_kline(code, period="daily", n_bars=120)
                if df is not None and not df.empty:
                    if "date" in df.columns:
                        df["_date"] = df["date"].astype(str).str.replace("-", "")
                    elif "trade_date" in df.columns:
                        df["_date"] = df["trade_date"].astype(str).str.replace("-", "")
                    else:
                        df["_date"] = ""
                    df = df.sort_values("_date")
                    matches = df[df["_date"] >= target_date]
                    if not matches.empty:
                        return float(matches.iloc[0]["close"])
            except Exception:
                pass

        # fallback: tushare
        try:
            import tushare as ts
            pro = ts.pro_api(self.tushare_token)
            ts_code = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
            end_date = (datetime.strptime(target_date, "%Y%m%d") + timedelta(days=10)).strftime("%Y%m%d")
            df = pro.daily(ts_code=ts_code, start_date=target_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df = df.sort_values("trade_date")
                return float(df.iloc[0]["close"])
        except Exception:
            pass

        return None

    def find_previous_results(self, target_date: str = None) -> Optional[str]:
        """查找 risk_result 文件"""
        pattern = os.path.join(self.data_dir, "risk_result_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        # 排除 linkb
        files = [f for f in files if "linkb" not in os.path.basename(f)]
        if target_date:
            for f in files:
                if target_date in f:
                    return f
            return None
        return files[0] if files else None

    def _load_fusion_data(self, rec_date: str) -> List[Dict]:
        """加载 fusion_result 中的 recommended_by 信息"""
        fusion_path = os.path.join(self.data_dir, f"fusion_result_{rec_date}.json")
        if not os.path.exists(fusion_path):
            return []
        try:
            with open(fusion_path, "r", encoding="utf-8") as f:
                fusion = json.load(f)
            picks = fusion.get("final_top10", fusion.get("final_top25", []))
            return picks
        except Exception:
            return []

    def evaluate(self, result_file: str, eval_days: int = 5) -> Dict:
        """
        评估一次历��推荐。

        增强: 从 fusion_result 读��� recommended_by，支持 per-model 追踪。
        """
        with open(result_file, "r", encoding="utf-8") as f:
            risk_result = json.load(f)

        basename = os.path.basename(result_file)
        rec_date = basename.replace("risk_result_", "").replace(".json", "")

        # 计算评估目标日期
        dt = datetime.strptime(rec_date, "%Y%m%d")
        eval_date = (dt + timedelta(days=int(eval_days * 1.4) + 2)).strftime("%Y%m%d")
        today = datetime.now().strftime("%Y%m%d")
        if eval_date > today:
            eval_date = today

        # 获取所有推荐股票
        approved = risk_result.get("approved", [])
        soft_excluded = risk_result.get("soft_excluded", [])
        all_stocks = approved + soft_excluded
        if not all_stocks:
            print("  [回测] 无推荐股票")
            return {}

        # 加载 fusion 数据（per-model 信息）
        fusion_picks = self._load_fusion_data(rec_date)
        fusion_index = {p.get("code", ""): p for p in fusion_picks}

        # 获取价格并计算收益
        results = []
        soft_codes = {s.get("code", "") for s in soft_excluded}

        for stk in all_stocks:
            code = stk.get("code", "")
            name = stk.get("name", "")
            if not code:
                continue

            rec_close = self._fetch_close_price(code, rec_date)
            current_close = self._fetch_close_price(code, eval_date)

            if rec_close is None or current_close is None:
                continue
            if rec_close <= 0:
                continue

            return_pct = round((current_close - rec_close) / rec_close * 100, 2)

            entry = {
                "code": code,
                "name": name,
                "rec_close": rec_close,
                "current_close": current_close,
                "return_pct": return_pct,
                "risk_level": stk.get("risk_level", "⚠️" if code in soft_codes else ""),
                "is_excluded": code in soft_codes,
            }

            # 附加 per-model 信息
            fp = fusion_index.get(code, {})
            entry["recommended_by"] = fp.get("recommended_by", [])
            entry["borda_score"] = fp.get("borda_score", 0)

            results.append(entry)

        if not results:
            print("  [回测] 无法获取价格数据")
            return {}

        # 计算汇总
        returns = [r["return_pct"] for r in results]
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r < 0]

        eval_result = {
            "rec_date": rec_date,
            "eval_date": eval_date,
            "eval_days": eval_days,
            "stocks": sorted(results, key=lambda x: x["return_pct"], reverse=True),
            "summary": {
                "total": len(results),
                "winners": len(winners),
                "losers": len(losers),
                "win_rate": round(len(winners) / len(results) * 100, 1),
                "avg_return": round(sum(returns) / len(returns), 2),
                "max_gain": round(max(returns), 2) if returns else 0,
                "max_loss": round(min(returns), 2) if returns else 0,
                "total_return": round(sum(returns), 2),
            },
        }

        # per-model 绩效
        eval_result["model_perf"] = self._calc_model_perf(results)

        return eval_result

    def _calc_model_perf(self, results: List[Dict]) -> Dict:
        """计算各模型的推荐绩效"""
        model_stats = defaultdict(lambda: {"returns": [], "total": 0, "wins": 0})

        for stk in results:
            ret = stk["return_pct"]
            for rec in stk.get("recommended_by", []):
                model = rec.get("model", "unknown")
                model_stats[model]["returns"].append(ret)
                model_stats[model]["total"] += 1
                if ret > 0:
                    model_stats[model]["wins"] += 1

        perf = {}
        for model, data in model_stats.items():
            if data["total"] == 0:
                continue
            rets = data["returns"]
            perf[model] = {
                "total": data["total"],
                "winners": data["wins"],
                "win_rate": round(data["wins"] / data["total"] * 100, 1),
                "avg_return": round(sum(rets) / len(rets), 2),
                "max_gain": round(max(rets), 2),
                "max_loss": round(min(rets), 2),
            }

        return perf

    def print_report(self, eval_result: Dict):
        """打印回测报告"""
        if not eval_result:
            return

        s = eval_result["summary"]
        print(f"\n{'='*65}")
        print(f"  回测报告: {eval_result['rec_date']} → {eval_result['eval_date']} (T+{eval_result['eval_days']})")
        print(f"{'='*65}")
        print(f"  推荐: {s['total']}只 | 上涨: {s['winners']}只 | 下跌: {s['losers']}只")
        print(f"  胜率: {s['win_rate']:.1f}% | 平均收益: {s['avg_return']:+.2f}%")
        print(f"  最大盈利: {s['max_gain']:+.2f}% | 最大亏损: {s['max_loss']:+.2f}%")

        print(f"\n  {'#':<3} {'代码':<8} {'名称':<10} {'推荐价':>7} {'现价':>7} {'收益率':>8} {'Borda':>6} {'模型':>6}")
        print("  " + "─" * 65)

        for i, stk in enumerate(eval_result["stocks"], 1):
            ret_str = f"{stk['return_pct']:+.2f}%"
            borda = stk.get("borda_score", 0)
            models = len(stk.get("recommended_by", []))
            excluded = " ⚠" if stk.get("is_excluded") else ""
            print(f"  {i:<3} {stk['code']:<8} {stk['name']:<10} "
                  f"{stk['rec_close']:>7.2f} {stk['current_close']:>7.2f} "
                  f"{ret_str:>8} {borda:>6.0f} {models:>5}个{excluded}")

        # 模型绩效
        mp = eval_result.get("model_perf", {})
        if mp:
            print(f"\n  [各模型推荐绩效]")
            print(f"  {'模型':<12} {'推荐数':>6} {'胜率':>8} {'均收益':>8}")
            print("  " + "─" * 40)
            for model in sorted(mp.keys(), key=lambda m: -mp[m]["win_rate"]):
                p = mp[model]
                print(f"  {model:<12} {p['total']:>6} {p['win_rate']:>7.1f}% {p['avg_return']:>+7.2f}%")

    def get_reflection_text(self, eval_result: Dict) -> str:
        """生成可注入LLM提示词的反思文本"""
        if not eval_result:
            return ""

        s = eval_result["summary"]
        lines = [
            f"【历史回测参考】推荐日期:{eval_result['rec_date']} "
            f"胜率:{s['win_rate']:.0f}% 均收益:{s['avg_return']:+.1f}%",
        ]

        # 成功案例
        winners = [stk for stk in eval_result["stocks"] if stk["return_pct"] > 5]
        if winners:
            w_str = ", ".join(f"{w['code']}{w['name']}({w['return_pct']:+.1f}%)" for w in winners[:3])
            lines.append(f"  成功案例: {w_str}")

        # 失败案例
        losers = [stk for stk in eval_result["stocks"] if stk["return_pct"] < -5]
        if losers:
            l_str = ", ".join(f"{l['code']}{l['name']}({l['return_pct']:+.1f}%)" for l in losers[:3])
            lines.append(f"  失败案例: {l_str}")

        # 风控排除股表现
        excluded = [stk for stk in eval_result["stocks"] if stk.get("is_excluded")]
        if excluded:
            exc_avg = sum(e["return_pct"] for e in excluded) / len(excluded)
            lines.append(f"  风控排除股平均收益: {exc_avg:+.1f}%（验证风控是否有效）")

        # 模型绩效摘要
        mp = eval_result.get("model_perf", {})
        if mp:
            best = max(mp.items(), key=lambda x: x[1]["win_rate"])
            worst = min(mp.items(), key=lambda x: x[1]["win_rate"])
            lines.append(f"  最佳模型: {best[0]}(胜率{best[1]['win_rate']:.0f}%) "
                         f"最差模型: {worst[0]}(胜率{worst[1]['win_rate']:.0f}%)")

        return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股选股系统 - 回测验证")
    parser.add_argument("--days", type=int, default=5, help="回测天数（默认5天）")
    parser.add_argument("--date", type=str, default="", help="指定回测日期（如20260310）")
    parser.add_argument("--data-dir", type=str, default="output/data", help="数据目录")
    parser.add_argument("--model-perf", action="store_true", help="输出各模型推荐绩效")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN", "")

    engine = BacktestEngine(data_dir=args.data_dir, tushare_token=token)

    target_date = args.date if args.date else None
    result_file = engine.find_previous_results(target_date)

    if not result_file:
        print("  [回测] 未找到历史推荐记录")
    else:
        print(f"  [回测] 评估文件: {result_file}")
        eval_result = engine.evaluate(result_file, args.days)
        engine.print_report(eval_result)

        if eval_result:
            out_path = os.path.join(args.data_dir, f"backtest_{eval_result['rec_date']}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, ensure_ascii=False, indent=2)
            print(f"\n  [回测] 结果已保存: {out_path}")

            print(f"\n  [反思文本]")
            print(engine.get_reflection_text(eval_result))
