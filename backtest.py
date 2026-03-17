#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 回测验证引擎

评估历史推荐的实际表现，形成反馈闭环。
用法:
  python backtest.py                # 回测最近一次推荐
  python backtest.py --days 5       # 回测5个交易日前的推荐
  python backtest.py --date 20260310 # 回测指定日期的推荐
"""

import argparse
import json
import os
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class BacktestEngine:
    """回测验证引擎"""

    def __init__(self, data_dir="output/data", tushare_token=""):
        self.data_dir = data_dir
        self.tushare_token = tushare_token

    def find_previous_results(self, target_date: str = None) -> Optional[str]:
        """
        Find risk_result_{date}.json files.
        If target_date provided, look for that specific date.
        Otherwise find the most recent one that is at least 1 day old.
        """
        pattern = os.path.join(self.data_dir, "risk_result_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        if target_date:
            for f in files:
                if target_date in f:
                    return f
            return None
        # Return most recent
        return files[0] if files else None

    def evaluate(self, result_file: str, eval_days: int = 5) -> Dict:
        """
        Evaluate a previous recommendation.

        1. Load risk_result JSON
        2. Extract recommended stock codes
        3. Fetch their price at recommendation date and current price
        4. Compute returns
        5. Generate summary stats

        Returns: {
            "rec_date": str,
            "eval_date": str,
            "eval_days": int,
            "stocks": [
                {"code", "name", "rec_close", "current_close", "return_pct", "risk_level"}
            ],
            "summary": {
                "total": int,
                "winners": int,
                "losers": int,
                "win_rate": float,
                "avg_return": float,
                "max_gain": float,
                "max_loss": float,
                "total_return": float
            }
        }
        """
        # Load result file
        with open(result_file, "r", encoding="utf-8") as f:
            risk_result = json.load(f)

        # Extract date from filename
        basename = os.path.basename(result_file)
        rec_date = basename.replace("risk_result_", "").replace(".json", "")

        # Get all recommended stocks (approved + soft_excluded)
        approved = risk_result.get("approved", [])
        soft_excluded = risk_result.get("soft_excluded", [])
        all_stocks = approved + soft_excluded

        if not all_stocks:
            print("  [回测] 无推荐股票")
            return {}

        # Fetch prices using tushare
        results = []
        try:
            import tushare as ts

            pro = ts.pro_api(self.tushare_token)

            for stk in all_stocks:
                code = stk.get("code", "")
                name = stk.get("name", "")
                if not code:
                    continue

                # Convert to tushare format
                if code.startswith("6"):
                    ts_code = f"{code}.SH"
                else:
                    ts_code = f"{code}.SZ"

                try:
                    # Get daily data from rec_date to now
                    df = pro.daily(
                        ts_code=ts_code,
                        start_date=rec_date,
                        end_date=datetime.now().strftime("%Y%m%d"),
                    )
                    if df is not None and len(df) >= 2:
                        df = df.sort_values("trade_date")
                        rec_close = float(df.iloc[0]["close"])
                        current_close = float(df.iloc[-1]["close"])
                        return_pct = (current_close - rec_close) / rec_close * 100

                        results.append({
                            "code": code,
                            "name": name,
                            "rec_close": rec_close,
                            "current_close": current_close,
                            "return_pct": round(return_pct, 2),
                            "risk_level": stk.get("risk_level", "⚠️" if stk in soft_excluded else ""),
                            "is_excluded": stk in soft_excluded,
                        })
                except Exception as e:
                    print(f"  [回测] {code} 数据获取失败: {e}")

        except ImportError:
            print("  [错误] 需要安装 tushare: pip install tushare")
            return {}

        if not results:
            print("  [回测] 无法获取价格数据")
            return {}

        # Compute summary
        returns = [r["return_pct"] for r in results]
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r < 0]

        eval_result = {
            "rec_date": rec_date,
            "eval_date": datetime.now().strftime("%Y%m%d"),
            "eval_days": eval_days,
            "stocks": sorted(results, key=lambda x: x["return_pct"], reverse=True),
            "summary": {
                "total": len(results),
                "winners": len(winners),
                "losers": len(losers),
                "win_rate": round(len(winners) / len(results) * 100, 1) if results else 0,
                "avg_return": round(sum(returns) / len(returns), 2) if returns else 0,
                "max_gain": round(max(returns), 2) if returns else 0,
                "max_loss": round(min(returns), 2) if returns else 0,
                "total_return": round(sum(returns), 2),
            },
        }

        return eval_result

    def print_report(self, eval_result: Dict):
        """Print backtest report to console"""
        if not eval_result:
            return

        s = eval_result["summary"]
        print(f"\n{'='*60}")
        print(f"  回测报告: {eval_result['rec_date']} → {eval_result['eval_date']}")
        print(f"{'='*60}")
        print(f"  推荐股票: {s['total']}只 | 上涨: {s['winners']}只 | 下跌: {s['losers']}只")
        print(f"  胜率: {s['win_rate']:.1f}% | 平均收益: {s['avg_return']:+.2f}%")
        print(f"  最大盈利: {s['max_gain']:+.2f}% | 最大亏损: {s['max_loss']:+.2f}%")

        print(f"\n  {'排名':<4} {'代码':<8} {'名称':<10} {'推荐价':<8} {'现价':<8} {'收益率':<8} {'风控':<6}")
        print("  " + "─" * 60)

        for i, stk in enumerate(eval_result["stocks"], 1):
            ret_str = f"{stk['return_pct']:+.2f}%"
            risk = "⚠️" if stk.get("is_excluded") else stk.get("risk_level", "")[:4]
            print(
                f"  {i:<4} {stk['code']:<8} {stk['name']:<10} "
                f"{stk['rec_close']:<8.2f} {stk['current_close']:<8.2f} "
                f"{ret_str:<8} {risk}"
            )

    def get_reflection_text(self, eval_result: Dict) -> str:
        """Generate reflection text that can be injected into LLM prompts"""
        if not eval_result:
            return ""

        s = eval_result["summary"]
        lines = [
            f"【历史回测参考】推荐日期:{eval_result['rec_date']} 胜率:{s['win_rate']:.0f}% 均收益:{s['avg_return']:+.1f}%",
        ]

        # Top winners
        winners = [stk for stk in eval_result["stocks"] if stk["return_pct"] > 5]
        if winners:
            w_str = ", ".join(f"{w['code']}{w['name']}({w['return_pct']:+.1f}%)" for w in winners[:3])
            lines.append(f"  成功案例: {w_str}")

        # Top losers
        losers = [stk for stk in eval_result["stocks"] if stk["return_pct"] < -5]
        if losers:
            l_str = ", ".join(f"{l['code']}{l['name']}({l['return_pct']:+.1f}%)" for l in losers[:3])
            lines.append(f"  失败案例: {l_str}")

        # Excluded stocks performance
        excluded = [stk for stk in eval_result["stocks"] if stk.get("is_excluded")]
        if excluded:
            exc_avg = sum(e["return_pct"] for e in excluded) / len(excluded)
            lines.append(f"  风控排除股平均收益: {exc_avg:+.1f}%（验证风控是否有效）")

        return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A股选股系统 - 回测验证")
    parser.add_argument("--days", type=int, default=5, help="回测天数（默认5天）")
    parser.add_argument("--date", type=str, default="", help="指定回测日期（如20260310）")
    parser.add_argument("--data-dir", type=str, default="output/data", help="数据目录")
    args = parser.parse_args()

    # Load tushare token from .env
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

        # Save result
        if eval_result:
            out_path = os.path.join(args.data_dir, f"backtest_{eval_result['rec_date']}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(eval_result, f, ensure_ascii=False, indent=2)
            print(f"\n  [回测] 结果已保存: {out_path}")

            print(f"\n  [反思文本（可注入LLM提示词）]")
            print(engine.get_reflection_text(eval_result))
