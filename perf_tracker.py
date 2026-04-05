#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 绩效追踪模块

功能:
  1. 追踪历史推荐的 T+3/T+5/T+8 实际涨幅
  2. 按模型维度统计推荐准确率（胜率/盈亏比）
  3. 按专家维度统计各 E1-E6 的贡献度
  4. 输出自适应权重建议（专家权重 + 模型Borda乘数）
  5. 绩效数据库持久化（JSON）

用法:
  python perf_tracker.py                    # 回测所有历史推荐
  python perf_tracker.py --date 20260320    # 回测指定日期
  python perf_tracker.py --report           # 输出绩效报告+权重建议
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ── 路径 ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "output", "data")
PERF_DIR = os.path.join(BASE_DIR, "output", "perf")
os.makedirs(PERF_DIR, exist_ok=True)

# 绩效数据库文件
PERF_DB_PATH = os.path.join(PERF_DIR, "perf_db.json")
# 自适应权重输出
ADAPTIVE_WEIGHTS_PATH = os.path.join(PERF_DIR, "adaptive_weights.json")

# 评估周期（交易日）
EVAL_PERIODS = [3, 5, 8]
# 自适应权重回看天数
LOOKBACK_DAYS = 30
# 默认专家权重（与 config.py 一致）
DEFAULT_EXPERT_WEIGHTS = {
    "E1": 0.10, "E2": 0.10, "E3": 0.25,
    "E4": 0.30, "E5": 0.15, "E6": 0.10,
}


class PerformanceTracker:
    """绩效追踪与自适应权重引擎"""

    def __init__(self, data_dir: str = DATA_DIR, perf_dir: str = PERF_DIR):
        self.data_dir = data_dir
        self.perf_dir = perf_dir
        self.perf_db = self._load_db()
        self._engine = None  # lazy init DataEngine

    # ── 数据库读写 ──────────────────────────────────────────────────

    def _load_db(self) -> Dict:
        """加载绩效数据库"""
        if os.path.exists(PERF_DB_PATH):
            try:
                with open(PERF_DB_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"records": {}, "updated_at": ""}

    def _save_db(self):
        """保存绩效数据库"""
        self.perf_db["updated_at"] = datetime.now().isoformat()
        with open(PERF_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(self.perf_db, f, ensure_ascii=False, indent=2)

    def _get_engine(self):
        """懒加载 DataEngine（避免无需数据时也初始化）"""
        if self._engine is None:
            from data_engine import DataEngine
            tushare_token = os.environ.get("TUSHARE_TOKEN", "")
            self._engine = DataEngine(tushare_token=tushare_token)
        return self._engine

    # ── 价格获取 ──────────────────────────────────────────────────

    def _fetch_price_on_date(self, code: str, target_date: str) -> Optional[float]:
        """
        获取某只股票在指定日期的收盘价。
        target_date 格式: YYYYMMDD
        如果该日非交易日，取之后最近一个交易日。
        """
        engine = self._get_engine()
        try:
            df = engine.fetch_kline(code, period="daily", n_bars=120)
            if df is None or df.empty:
                return None
            # 标准化日期列
            if "date" in df.columns:
                df["date"] = df["date"].astype(str).str.replace("-", "")
            elif "trade_date" in df.columns:
                df["date"] = df["trade_date"].astype(str).str.replace("-", "")
            else:
                return None

            df = df.sort_values("date").reset_index(drop=True)

            # 精确匹配或取之后最近日
            matches = df[df["date"] >= target_date]
            if matches.empty:
                return None
            return float(matches.iloc[0]["close"])
        except Exception:
            return None

    def _fetch_prices_batch(self, codes: List[str], target_date: str) -> Dict[str, Optional[float]]:
        """批量获取收盘价"""
        import concurrent.futures
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(self._fetch_price_on_date, c, target_date): c for c in codes}
            for fut in concurrent.futures.as_completed(futures):
                code = futures[fut]
                try:
                    results[code] = fut.result()
                except Exception:
                    results[code] = None
        return results

    # ── 交易日计算 ──────────────────────────────────────────────────

    @staticmethod
    def _add_trading_days(date_str: str, n_days: int) -> str:
        """
        粗略估算 T+N 交易日对应的自然日期。
        交易日≈自然日×5/7，这里简单用 n_days * 7/5 + 2 的自然日偏移。
        """
        dt = datetime.strptime(date_str, "%Y%m%d")
        # 1个交易日≈1.4自然日，多取2天余量
        natural_days = int(n_days * 1.4) + 2
        target = dt + timedelta(days=natural_days)
        return target.strftime("%Y%m%d")

    # ── 单日评估 ──────────────────────────────────────────────────

    def evaluate_date(self, rec_date: str, force: bool = False) -> Optional[Dict]:
        """
        评估某个推荐日期的绩效。

        读取 risk_result + fusion_result + model_results，
        获取 T+3/T+5/T+8 价格，计算收益率。

        返回: {
            "rec_date": str,
            "eval_time": str,
            "stocks": [{code, name, rec_price, prices_t3/t5/t8, returns_t3/t5/t8,
                        recommended_by: [{model, rank, score}]}],
            "summary_t3/t5/t8": {total, winners, win_rate, avg_return, ...},
            "model_perf": {model_name: {total, winners, win_rate, avg_return}},
        }
        """
        # 检查是否已有完整评估
        existing = self.perf_db.get("records", {}).get(rec_date)
        if existing and not force:
            # 检查是否所有周期都有数据
            if all(f"summary_t{p}" in existing for p in EVAL_PERIODS):
                return existing

        # 加载推荐数据
        risk_path = os.path.join(self.data_dir, f"risk_result_{rec_date}.json")
        fusion_path = os.path.join(self.data_dir, f"fusion_result_{rec_date}.json")

        if not os.path.exists(risk_path) and not os.path.exists(fusion_path):
            return None

        # 优先用 fusion_result（包含 recommended_by 信息）
        stocks_info = []
        if os.path.exists(fusion_path):
            with open(fusion_path, "r", encoding="utf-8") as f:
                fusion = json.load(f)
            picks = fusion.get("final_top10", fusion.get("final_top25", []))
            for p in picks:
                stocks_info.append({
                    "code": p.get("code", ""),
                    "name": p.get("name", ""),
                    "sector": p.get("sector", ""),
                    "borda_score": p.get("borda_score", 0),
                    "recommended_by": p.get("recommended_by", []),
                })

        # 如果 fusion 没有数据，用 risk_result
        if not stocks_info and os.path.exists(risk_path):
            with open(risk_path, "r", encoding="utf-8") as f:
                risk = json.load(f)
            for s in risk.get("approved", []) + risk.get("soft_excluded", []):
                stocks_info.append({
                    "code": s.get("code", ""),
                    "name": s.get("name", ""),
                    "sector": s.get("sector", ""),
                    "borda_score": 0,
                    "recommended_by": [],
                })

        if not stocks_info:
            return None

        codes = [s["code"] for s in stocks_info if s["code"]]

        # 获取推荐日收盘价
        print(f"  [绩效] 获取 {rec_date} 推荐日价格（{len(codes)}只）...")
        rec_prices = self._fetch_prices_batch(codes, rec_date)

        # 获取 T+3/T+5/T+8 价格
        period_prices = {}
        today_str = datetime.now().strftime("%Y%m%d")
        for period in EVAL_PERIODS:
            target_date = self._add_trading_days(rec_date, period)
            if target_date > today_str:
                print(f"  [绩效] T+{period}({target_date}) 尚未到期，跳过")
                continue
            print(f"  [绩效] 获取 T+{period} 价格({target_date})...")
            period_prices[period] = self._fetch_prices_batch(codes, target_date)

        # 组装结果
        stock_results = []
        for si in stocks_info:
            code = si["code"]
            if not code:
                continue
            rec_price = rec_prices.get(code)
            if rec_price is None or rec_price <= 0:
                continue

            entry = {
                "code": code,
                "name": si["name"],
                "sector": si["sector"],
                "borda_score": si["borda_score"],
                "rec_price": rec_price,
                "recommended_by": si["recommended_by"],
            }

            for period in EVAL_PERIODS:
                if period in period_prices:
                    p = period_prices[period].get(code)
                    if p is not None and p > 0:
                        ret = round((p - rec_price) / rec_price * 100, 2)
                        entry[f"price_t{period}"] = p
                        entry[f"return_t{period}"] = ret
                    else:
                        entry[f"price_t{period}"] = None
                        entry[f"return_t{period}"] = None

            stock_results.append(entry)

        if not stock_results:
            return None

        # 计算汇总
        record = {
            "rec_date": rec_date,
            "eval_time": datetime.now().isoformat(),
            "stock_count": len(stock_results),
            "stocks": stock_results,
        }

        for period in EVAL_PERIODS:
            key = f"return_t{period}"
            returns = [s[key] for s in stock_results if s.get(key) is not None]
            if returns:
                winners = [r for r in returns if r > 0]
                record[f"summary_t{period}"] = {
                    "total": len(returns),
                    "winners": len(winners),
                    "losers": len(returns) - len(winners),
                    "win_rate": round(len(winners) / len(returns) * 100, 1),
                    "avg_return": round(sum(returns) / len(returns), 2),
                    "max_gain": round(max(returns), 2),
                    "max_loss": round(min(returns), 2),
                    "total_return": round(sum(returns), 2),
                }

        # 计算 per-model 绩效
        record["model_perf"] = self._calc_model_perf(stock_results)

        # 存入数据库
        self.perf_db.setdefault("records", {})[rec_date] = record
        self._save_db()

        return record

    def _calc_model_perf(self, stock_results: List[Dict]) -> Dict:
        """计算每个模型的推荐绩效（以T+5为基准）"""
        model_stats = defaultdict(lambda: {"returns": [], "stocks": []})

        for stk in stock_results:
            ret_t5 = stk.get("return_t5")
            if ret_t5 is None:
                continue
            for rec in stk.get("recommended_by", []):
                model = rec.get("model", "unknown")
                rank = rec.get("rank", 99)
                model_stats[model]["returns"].append(ret_t5)
                model_stats[model]["stocks"].append({
                    "code": stk["code"],
                    "name": stk["name"],
                    "rank": rank,
                    "return_t5": ret_t5,
                })

        result = {}
        for model, data in model_stats.items():
            rets = data["returns"]
            if not rets:
                continue
            winners = [r for r in rets if r > 0]
            result[model] = {
                "total": len(rets),
                "winners": len(winners),
                "win_rate": round(len(winners) / len(rets) * 100, 1),
                "avg_return": round(sum(rets) / len(rets), 2),
                "max_gain": round(max(rets), 2),
                "max_loss": round(min(rets), 2),
            }

        return result

    # ── 批量评估 ──────────────────────────────────────────────────

    def evaluate_all(self, force: bool = False) -> List[Dict]:
        """评估所有历史推荐日期"""
        pattern = os.path.join(self.data_dir, "risk_result_*.json")
        files = sorted(glob.glob(pattern))
        results = []

        for f in files:
            basename = os.path.basename(f)
            # 排除 linkb 的结果
            if "linkb" in basename:
                continue
            date = basename.replace("risk_result_", "").replace(".json", "")
            try:
                r = self.evaluate_date(date, force=force)
                if r:
                    results.append(r)
            except Exception as e:
                print(f"  [绩效] {date} 评估失败: {e}")

        return results

    # ── 自适应权重计算 ──────────────────────────────────────────────

    def compute_adaptive_weights(self, lookback_days: int = LOOKBACK_DAYS) -> Dict:
        """
        基于近 N 天绩效数据，计算自适应权重。

        返回: {
            "expert_weights": {"E1": 0.xx, ...},
            "model_weights": {"grok": 1.xx, ...},
            "computed_at": str,
            "lookback_days": int,
            "data_points": int,
        }
        """
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        records = self.perf_db.get("records", {})

        # 收集近期有效记录
        recent = {d: r for d, r in records.items()
                  if d >= cutoff and r.get("summary_t5")}

        if len(recent) < 3:
            print(f"  [自适应] 有效数据不足（{len(recent)}<3），使用默认权重")
            return {
                "expert_weights": DEFAULT_EXPERT_WEIGHTS.copy(),
                "model_weights": {},
                "computed_at": datetime.now().isoformat(),
                "lookback_days": lookback_days,
                "data_points": len(recent),
                "source": "default",
            }

        # ── 模型权重 ──────────────────────────────────────────────
        model_agg = defaultdict(lambda: {"returns": [], "win_count": 0, "total": 0})
        for rec in recent.values():
            mp = rec.get("model_perf", {})
            for model, perf in mp.items():
                model_agg[model]["returns"].extend(
                    [perf["avg_return"]] * perf["total"]
                )
                model_agg[model]["win_count"] += perf["winners"]
                model_agg[model]["total"] += perf["total"]

        model_weights = {}
        if model_agg:
            # 计算每个模型的综合得分 = 胜率 × 0.6 + 归一化平均收益 × 0.4
            scores = {}
            for model, agg in model_agg.items():
                if agg["total"] < 5:
                    continue
                wr = agg["win_count"] / agg["total"]
                avg_ret = sum(agg["returns"]) / len(agg["returns"])
                # 归一化到 [0, 1]：avg_ret 在 [-10, +10] 范围映射
                norm_ret = max(0, min(1, (avg_ret + 10) / 20))
                scores[model] = wr * 0.6 + norm_ret * 0.4

            if scores:
                # 转换为乘数：平均分=1.0，好的>1.0，差的<1.0
                avg_score = sum(scores.values()) / len(scores)
                for model, s in scores.items():
                    multiplier = s / avg_score if avg_score > 0 else 1.0
                    # 限制范围 [0.5, 1.5]，避免极端
                    model_weights[model] = round(max(0.5, min(1.5, multiplier)), 3)

        # ── 专家权重（基于模型结果中各rank段的胜率）──────────────────
        # 由于当前没有 per-expert 独立追踪，用 Borda rank vs 收益的相关性
        # 后续可增加 per-expert 维度
        # 暂时保持默认专家权重，仅微调
        expert_weights = DEFAULT_EXPERT_WEIGHTS.copy()

        # 分析高 Borda 分 vs 低 Borda 分的胜率差异
        high_borda_wins = 0
        high_borda_total = 0
        low_borda_wins = 0
        low_borda_total = 0

        for rec in recent.values():
            stocks = rec.get("stocks", [])
            if not stocks:
                continue
            median_borda = sorted([s["borda_score"] for s in stocks])[len(stocks) // 2]
            for stk in stocks:
                ret = stk.get("return_t5")
                if ret is None:
                    continue
                if stk["borda_score"] >= median_borda:
                    high_borda_total += 1
                    if ret > 0:
                        high_borda_wins += 1
                else:
                    low_borda_total += 1
                    if ret > 0:
                        low_borda_wins += 1

        # 如果高 Borda 分股票胜率明显更高，说明当前融合逻辑有效
        borda_effectiveness = None
        if high_borda_total > 0 and low_borda_total > 0:
            high_wr = high_borda_wins / high_borda_total
            low_wr = low_borda_wins / low_borda_total
            borda_effectiveness = {
                "high_borda_win_rate": round(high_wr * 100, 1),
                "low_borda_win_rate": round(low_wr * 100, 1),
                "spread": round((high_wr - low_wr) * 100, 1),
            }

        result = {
            "expert_weights": expert_weights,
            "model_weights": model_weights,
            "borda_effectiveness": borda_effectiveness,
            "computed_at": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "data_points": len(recent),
            "source": "adaptive",
        }

        # 保存
        with open(ADAPTIVE_WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    # ── 报告 ──────────────────────────────────────────────────────

    def print_report(self):
        """打印完整绩效报告"""
        records = self.perf_db.get("records", {})
        if not records:
            print("  [绩效] 无历史绩效数据")
            return

        sorted_dates = sorted(records.keys(), reverse=True)

        print(f"\n{'='*70}")
        print(f"  绩效追踪报告 — 共 {len(sorted_dates)} 个推荐日")
        print(f"{'='*70}")

        # 总体统计
        all_t5_returns = []
        for d in sorted_dates:
            rec = records[d]
            for stk in rec.get("stocks", []):
                r = stk.get("return_t5")
                if r is not None:
                    all_t5_returns.append(r)

        if all_t5_returns:
            winners = [r for r in all_t5_returns if r > 0]
            print(f"\n  [总体T+5] 样本={len(all_t5_returns)}只 "
                  f"胜率={len(winners)/len(all_t5_returns)*100:.1f}% "
                  f"均收益={sum(all_t5_returns)/len(all_t5_returns):+.2f}% "
                  f"最大盈={max(all_t5_returns):+.2f}% "
                  f"最大亏={min(all_t5_returns):+.2f}%")

        # 每日详情
        print(f"\n  {'日期':<10} {'T+3胜率':>8} {'T+3均收':>8} {'T+5胜率':>8} {'T+5均收':>8} {'T+8胜率':>8} {'T+8均收':>8}")
        print("  " + "─" * 60)

        for d in sorted_dates[:20]:  # 最近20天
            rec = records[d]
            parts = [f"  {d:<10}"]
            for p in EVAL_PERIODS:
                s = rec.get(f"summary_t{p}")
                if s:
                    parts.append(f"{s['win_rate']:>7.1f}%")
                    parts.append(f"{s['avg_return']:>+7.2f}%")
                else:
                    parts.append(f"{'--':>8}")
                    parts.append(f"{'--':>8}")
            print(" ".join(parts))

        # 模型绩效
        print(f"\n  [模型绩效汇总（T+5）]")
        model_all = defaultdict(lambda: {"returns": [], "wins": 0, "total": 0})
        for d in sorted_dates:
            mp = records[d].get("model_perf", {})
            for model, perf in mp.items():
                model_all[model]["total"] += perf["total"]
                model_all[model]["wins"] += perf["winners"]
                rets = [perf["avg_return"]] * perf["total"]
                model_all[model]["returns"].extend(rets)

        if model_all:
            print(f"  {'模型':<12} {'推荐数':>6} {'胜率':>8} {'均收益':>8}")
            print("  " + "─" * 40)
            for model in sorted(model_all.keys()):
                data = model_all[model]
                if data["total"] == 0:
                    continue
                wr = data["wins"] / data["total"] * 100
                avg_r = sum(data["returns"]) / len(data["returns"])
                print(f"  {model:<12} {data['total']:>6} {wr:>7.1f}% {avg_r:>+7.2f}%")

        # 自适应权重建议
        print(f"\n  [自适应权重建议]")
        weights = self.compute_adaptive_weights()
        mw = weights.get("model_weights", {})
        if mw:
            print(f"  模型Borda乘数（基于近{weights['lookback_days']}天，{weights['data_points']}个数据点）:")
            for model, w in sorted(mw.items(), key=lambda x: -x[1]):
                bar = "█" * int(w * 10)
                print(f"    {model:<12} ×{w:.3f}  {bar}")
        else:
            print(f"  数据不足，暂无模型权重建议")

        be = weights.get("borda_effectiveness")
        if be:
            print(f"\n  Borda有效性检验:")
            print(f"    高Borda分股胜率: {be['high_borda_win_rate']:.1f}%")
            print(f"    低Borda分股胜率: {be['low_borda_win_rate']:.1f}%")
            print(f"    差值: {be['spread']:+.1f}pp {'✓ 融合有效' if be['spread'] > 0 else '⚠ 融合无效'}")

    # ── 加载自适应权重 ──────────────────────────────────────────────

    @staticmethod
    def load_adaptive_weights() -> Optional[Dict]:
        """加载已计算的自适应权重（供 main.py / config.py 使用）"""
        if os.path.exists(ADAPTIVE_WEIGHTS_PATH):
            try:
                with open(ADAPTIVE_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股选股系统 - 绩效追踪")
    parser.add_argument("--date", type=str, default="", help="评估指定推荐日期")
    parser.add_argument("--all", action="store_true", help="评估所有历史推荐")
    parser.add_argument("--report", action="store_true", help="输出绩效报告")
    parser.add_argument("--force", action="store_true", help="强制重新评估（忽略缓存）")
    parser.add_argument("--weights", action="store_true", help="仅计算并输出自适应权重")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    tracker = PerformanceTracker()

    if args.date:
        r = tracker.evaluate_date(args.date, force=args.force)
        if r:
            s = r.get("summary_t5", {})
            print(f"\n  {args.date}: T+5 胜率={s.get('win_rate',0):.1f}% "
                  f"均收益={s.get('avg_return',0):+.2f}%")
        else:
            print(f"  未找到 {args.date} 的推荐数据")

    elif args.all or args.report:
        print("  [绩效] 批量评估所有历史推荐...")
        tracker.evaluate_all(force=args.force)
        if args.report:
            tracker.print_report()

    elif args.weights:
        weights = tracker.compute_adaptive_weights()
        print(json.dumps(weights, ensure_ascii=False, indent=2))

    else:
        # 默认：评估所有 + 报告
        tracker.evaluate_all(force=args.force)
        tracker.print_report()
