#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B 对比回测：旧版（固定权重+Top150/30） vs 新版v3（连续化四维评分+RPS60+风险维度）

从 2022~2026 年随机抽取若干时间段，每个时间段取 3 个交易日运行 L1+L2，
对比 T+3/T+5/T+8 表现。

用法: python backtest_ab_compare.py
"""

import json
import os
import sys
import time
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"

# Windows UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

# 导入回测基础设施
from historical_backtest import (
    TdxCache, _calc_sma, _compute_rps,
    measure_performance_tdx, _trading_day_offset,
)


# ═══════════════════════════════════════════════════════════════════════════
#  旧版 L1（固定权重，Top150）
# ═══════════════════════════════════════════════════════════════════════════

def l1_old(cache: TdxCache, codes: List[str],
           cutoff_date: str, top_n: int = 150) -> List[Dict]:
    """旧版L1：固定评分权重，输出固定Top150"""
    all_rps = _compute_rps(cache, codes, cutoff_date, lookback=20)
    candidates = []

    for code in codes:
        code_rps_ret = all_rps.get(code)
        if code_rps_ret is None:
            continue
        df = cache.get_bars(code, cutoff_date, days=65)
        if df is None:
            continue
        df = df.tail(65).reset_index(drop=True)

        c_col = "close" if "close" in df.columns else ("收盘" if "收盘" in df.columns else None)
        o_col = "open"  if "open"  in df.columns else ("开盘" if "开盘" in df.columns else None)
        h_col = "high"  if "high"  in df.columns else ("最高" if "最高" in df.columns else None)
        l_col = "low"   if "low"   in df.columns else ("最低" if "最低" in df.columns else None)
        v_col = "volume" if "volume" in df.columns else (
                "vol" if "vol" in df.columns else (
                "成交量" if "成交量" in df.columns else None))
        if not all([c_col, o_col, v_col, h_col, l_col]):
            continue

        close_arr = df[c_col].astype(float).values
        open_arr  = df[o_col].astype(float).values
        high_arr  = df[h_col].astype(float).values
        low_arr   = df[l_col].astype(float).values
        vol_arr   = df[v_col].astype(float).values
        n = len(close_arr)
        if n < 20:
            continue

        ma5_s  = _calc_sma(pd.Series(close_arr), 5)
        ma10_s = _calc_sma(pd.Series(close_arr), 10)
        ma20_s = _calc_sma(pd.Series(close_arr), 20)
        if ma5_s.iloc[-1] != ma5_s.iloc[-1]:
            continue
        ma5, ma10, ma20 = ma5_s.iloc[-1], ma10_s.iloc[-1], ma20_s.iloc[-1]

        # 硬性淘汰
        if ma5 < ma10 < ma20:
            continue
        if n >= 3:
            if (close_arr[-1] < open_arr[-1] and
                    close_arr[-2] < open_arr[-2] and
                    close_arr[-3] < open_arr[-3]):
                continue
        if n >= 3:
            if (close_arr[-1] < close_arr[-2] < close_arr[-3] and
                    vol_arr[-1] < vol_arr[-2] < vol_arr[-3]):
                continue
        if n >= 20:
            avg_amount_20 = float(np.mean(close_arr[-20:] * vol_arr[-20:] * 100) / 1e8)
            if avg_amount_20 < 2.5:
                continue
        else:
            avg_amount_20 = 0

        # 特征
        ret_20d = code_rps_ret
        ma20_slope = 0
        if len(ma20_s) >= 6:
            ma20_prev = ma20_s.iloc[-6]
            if ma20_prev == ma20_prev and abs(ma20_prev) > 0.01:
                ma20_slope = (ma20 - ma20_prev) / abs(ma20_prev)

        hl = high_arr[1:] - low_arr[1:]
        hc = np.abs(high_arr[1:] - close_arr[:-1])
        lc = np.abs(low_arr[1:] - close_arr[:-1])
        tr_arr = np.maximum(hl, np.maximum(hc, lc))
        atr5  = float(np.mean(tr_arr[-5:])) if len(tr_arr) >= 5 else 0
        atr20 = float(np.mean(tr_arr[-20:])) if len(tr_arr) >= 20 else 0

        high_20 = float(np.max(high_arr[-20:])) if n >= 20 else float(np.max(high_arr))
        near_high = close_arr[-1] / max(high_20, 0.01)
        is_yang_fang = (close_arr[-1] > open_arr[-1]) and (vol_arr[-1] > vol_arr[-2])
        vol_ma20 = float(np.mean(vol_arr[-20:])) if n >= 20 else float(np.mean(vol_arr))
        amount_ratio = vol_arr[-1] / max(vol_ma20, 1)

        candidates.append({
            "code": code, "close": float(round(close_arr[-1], 2)),
            "MA5": float(round(ma5, 2)), "MA10": float(round(ma10, 2)), "MA20": float(round(ma20, 2)),
            "ret_20d": ret_20d, "ma20_slope": ma20_slope,
            "atr5": atr5, "atr20": atr20, "near_high": round(near_high, 4),
            "is_yang_fang": is_yang_fang, "amount_ratio": round(amount_ratio, 3),
            "avg_amount_20": round(avg_amount_20, 2),
            "rps_20d": float(round(code_rps_ret * 100, 2)),
        })

    if not candidates:
        return []

    # 固定评分
    ret_values = np.array([c["ret_20d"] for c in candidates])
    rps_pcts = np.array([percentileofscore(ret_values, r, kind='rank') for r in ret_values])

    for i, c in enumerate(candidates):
        rps_pct = rps_pcts[i]
        score = 0.0
        score += rps_pct * 0.25
        if c["ma20_slope"] > 0: score += 20
        if c["atr20"] > 0 and c["atr5"] < c["atr20"]: score += 15
        if c["near_high"] > 0.8: score += 15
        if c["is_yang_fang"]: score += 15
        if c["amount_ratio"] > 1.2: score += 10
        c["rps20"] = round(rps_pct, 1)
        c["score"] = round(score, 1)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
#  旧版 L2（固定权重 40/30/30，Top30）
# ═══════════════════════════════════════════════════════════════════════════

def l2_old(cache: TdxCache, l1_candidates: List[Dict],
           cutoff_date: str, top_n: int = 30) -> List[Dict]:
    """旧版L2：固定权重 形态40%+量能30%+动量30%，输出固定Top30"""
    scored = []
    all_rps = [c.get("rps_20d", 0) for c in l1_candidates]
    rps_arr = np.array(all_rps) if all_rps else np.array([0])

    for item in l1_candidates:
        code = item["code"]
        df300 = cache.get_bars(code, cutoff_date, days=300)
        if df300 is None or len(df300) < 60:
            continue
        c_col = "close" if "close" in df300.columns else ("收盘" if "收盘" in df300.columns else None)
        o_col = "open"  if "open"  in df300.columns else ("开盘" if "开盘" in df300.columns else None)
        h_col = "high"  if "high"  in df300.columns else ("最高" if "最高" in df300.columns else None)
        l_col = "low"   if "low"   in df300.columns else ("最低" if "最低" in df300.columns else None)
        v_col = "volume" if "volume" in df300.columns else (
                "vol" if "vol" in df300.columns else (
                "成交量" if "成交量" in df300.columns else None))
        if not all([c_col, o_col, h_col, l_col]):
            continue

        close300 = df300[c_col].astype(float)
        close_v  = close300.values
        open300  = df300[o_col].astype(float).values
        high300  = df300[h_col].astype(float).values
        low300   = df300[l_col].astype(float).values
        n300 = len(close300)

        # 风控
        if n300 >= 6:
            drops = sum(1 for i in range(-5, 0) if close_v[i] < close_v[i - 1])
            if drops >= 5:
                continue
        if n300 >= 2 and close_v[-2] > 0:
            if (close_v[-1] - close_v[-2]) / close_v[-2] < -0.07:
                continue

        # 形态分
        ma5, ma10, ma20 = item.get("MA5", 0), item.get("MA10", 0), item.get("MA20", 0)
        if ma5 > ma10 > ma20: ma_score = 30
        elif ma5 > ma10: ma_score = 15
        else: ma_score = 0

        if n300 >= 20:
            tr = np.maximum(high300 - low300,
                            np.maximum(np.abs(high300 - np.roll(close_v, 1)),
                                       np.abs(low300 - np.roll(close_v, 1))))
            atr_5 = np.mean(tr[-5:])
            atr_20 = np.mean(tr[-20:])
            atr_ratio = atr_5 / max(atr_20, 0.001)
            atr_score = max(0, min(30, (1.0 - atr_ratio) * 60))
        else:
            atr_score = 0

        ma20_val = close300.rolling(20, min_periods=20).mean().iloc[-1]
        above_ma20 = 15 if (ma20_val == ma20_val and close300.iloc[-1] > ma20_val) else 0
        ma60_score = 0
        if n300 >= 60:
            ma60_val = close300.rolling(60, min_periods=60).mean().iloc[-1]
            if ma60_val == ma60_val and close300.iloc[-1] > ma60_val:
                ma60_score = 15

        ema12 = close300.ewm(span=12, adjust=False).mean()
        ema26 = close300.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bonus = 10 if dif.iloc[-1] > dea.iloc[-1] else 0

        form_score = min(ma_score + atr_score + above_ma20 + ma60_score + macd_bonus, 100)

        # 量能分
        vol300 = df300[v_col].astype(float).values if v_col else None
        vol_score = 0
        if vol300 is not None and n300 >= 20:
            vol_ma5 = np.mean(vol300[-5:])
            vol_ma20 = np.mean(vol300[-20:])
            if vol_ma5 < vol_ma20 * 0.8: vol_score += 40
            elif vol_ma5 < vol_ma20: vol_score += 20
            if len(vol300) >= 5:
                vol_std = np.std(vol300[-5:])
                vol_mean = np.mean(vol300[-5:])
                vol_cv = vol_std / max(vol_mean, 1)
                vol_score += max(0, min(30, (1.0 - vol_cv) * 40))
            if close_v[-1] > open300[-1] and vol300[-1] < vol_ma20 * 1.5:
                vol_score += 30
        vol_score = min(vol_score, 100)

        # 动量分
        rps_val = item.get("rps_20d", 0)
        rps_pct = float(np.searchsorted(np.sort(rps_arr), rps_val)) / max(len(rps_arr), 1)
        rps_score = rps_pct * 50
        ret_5d = (close300.iloc[-1] - close300.iloc[-5]) / close300.iloc[-5] * 100 if n300 >= 5 else 0
        excess_score = min(max(ret_5d * 5, 0), 50)
        momentum_score = rps_score + excess_score

        # 固定权重 40/30/30
        total = form_score * 0.40 + vol_score * 0.30 + momentum_score * 0.30

        scored.append({
            "code": code, "close": item["close"],
            "total_score": round(total, 1),
        })

    scored.sort(key=lambda x: -x["total_score"])
    return scored[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
#  新版 L1/L2（从 historical_backtest 导入）
# ═══════════════════════════════════════════════════════════════════════════

from historical_backtest import l1_filter_at_date as l1_new, l2_score_and_rank as l2_new


# ═══════════════════════════════════════════════════════════════════════════
#  对比回测主逻辑
# ═══════════════════════════════════════════════════════════════════════════

# 选取的对比时间段（覆盖不同市场环境）
TEST_PERIODS = [
    # (描述, 日期列表)  — 每个时间段取3个相邻交易日
    ("2022-04 熊市(上海封城)", ["20220411", "20220413", "20220415"]),
    ("2022-10 熊市底部",       ["20220310", "20221012", "20221014"]),
    ("2023-02 复苏初期",       ["20230206", "20230208", "20230210"]),
    ("2023-07 震荡期",         ["20230703", "20230705", "20230707"]),
    ("2024-01 小盘股暴跌",     ["20240115", "20240117", "20240119"]),
    ("2024-06 低迷期",         ["20240603", "20240605", "20240607"]),
    ("2024-10 大牛反转",       ["20240930", "20241009", "20241011"]),
    ("2025-01 震荡",           ["20250106", "20250108", "20250110"]),
    ("2025-05 近期",           ["20250505", "20250507", "20250509"]),
    ("2026-01 最近",           ["20260105", "20260107", "20260109"]),
]


def run_compare():
    print("=" * 70)
    print("  A/B 对比回测：旧版(固定权重) vs 新版v3(连续化四维+RPS60+风险)")
    print("=" * 70)

    # 初始化
    print("\n[1] 初始化 DataEngine + TDX 缓存...")
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    from data_engine import DataEngine

    tdx_dir = os.getenv("TDX_DIR", "d:/tdx")
    de = DataEngine(tushare_token=os.getenv("TUSHARE_TOKEN", ""), tdx_dir=tdx_dir)
    if de._tdx is None:
        print("  [错误] TDX 未连接")
        return

    # 股票代码
    sl_path = BASE_DIR / "output" / "stock_list.json"
    if sl_path.exists():
        sl = json.loads(sl_path.read_text(encoding="utf-8"))
        codes = list(sl.get("data", {}).keys())[:5000]
    else:
        codes = [f"{c:06d}" for c in range(600000, 602000)]
        codes += [f"{c:06d}" for c in range(1, 2000)]
        codes += [f"{c:06d}" for c in range(300001, 302000)]
    print(f"  股票代码: {len(codes)} 只")

    # 预加载
    cache = TdxCache(de)
    cache.preload(codes)

    # 对比结果收集
    all_results = []

    print(f"\n[2] 逐时间段对比（共 {len(TEST_PERIODS)} 个时间段）...\n")

    for period_name, dates in TEST_PERIODS:
        print(f"{'─'*60}")
        print(f"  时间段: {period_name}")
        print(f"{'─'*60}")

        period_old_rets = {"t3": [], "t5": [], "t8": []}
        period_new_rets = {"t3": [], "t5": [], "t8": []}
        period_detail = {
            "period": period_name,
            "dates": dates,
            "old": {"l1_counts": [], "l2_counts": []},
            "new": {"l1_counts": [], "l2_counts": [], "regimes": []},
        }

        for d in dates:
            t0 = time.time()

            # ── 旧版 ──
            old_l1 = l1_old(cache, codes, d)
            old_l2 = l2_old(cache, old_l1, d)
            old_codes_l2 = [x["code"] for x in old_l2]
            old_prices = {x["code"]: x["close"] for x in old_l2}

            # ── 新版 ──
            new_l1 = l1_new(cache, codes, d)
            new_l2 = l2_new(cache, new_l1, d)
            new_codes_l2 = [x["code"] for x in new_l2]
            new_prices = {x["code"]: x["close"] for x in new_l2}

            # 市场环境
            regime = new_l1[0].get("market_regime", "?") if new_l1 else "?"

            elapsed = time.time() - t0

            # 绩效
            old_perf = measure_performance_tdx(cache, old_codes_l2, d, old_prices)
            new_perf = measure_performance_tdx(cache, new_codes_l2, d, new_prices)

            # 收集收益率
            for p in (3, 5, 8):
                key = f"t{p}"
                for stk in old_perf.get("stocks", []):
                    v = stk.get(f"return_{key}")
                    if v is not None:
                        period_old_rets[key].append(v)
                for stk in new_perf.get("stocks", []):
                    v = stk.get(f"return_{key}")
                    if v is not None:
                        period_new_rets[key].append(v)

            period_detail["old"]["l1_counts"].append(len(old_l1))
            period_detail["old"]["l2_counts"].append(len(old_l2))
            period_detail["new"]["l1_counts"].append(len(new_l1))
            period_detail["new"]["l2_counts"].append(len(new_l2))
            period_detail["new"]["regimes"].append(regime)

            regime_cn = {"bull": "牛", "bear": "熊", "neutral": "震荡"}.get(regime, "?")
            print(f"  [{d}] {regime_cn} | "
                  f"旧L1={len(old_l1):>3} L2={len(old_l2):>2} | "
                  f"新L1={len(new_l1):>3} L2={len(new_l2):>2} | "
                  f"{elapsed:.1f}s")

        # 打印时间段汇总
        print()
        for key_label, key in [("T+3", "t3"), ("T+5", "t5"), ("T+8", "t8")]:
            old_r = period_old_rets[key]
            new_r = period_new_rets[key]
            if not old_r and not new_r:
                continue
            old_wr = len([r for r in old_r if r > 0]) / max(len(old_r), 1) * 100
            new_wr = len([r for r in new_r if r > 0]) / max(len(new_r), 1) * 100
            old_avg = sum(old_r) / max(len(old_r), 1)
            new_avg = sum(new_r) / max(len(new_r), 1)

            delta_wr = new_wr - old_wr
            delta_avg = new_avg - old_avg
            wr_mark = "✓" if delta_wr > 0 else "✗" if delta_wr < -1 else "="
            avg_mark = "✓" if delta_avg > 0 else "✗" if delta_avg < -0.1 else "="

            print(f"  {key_label}: 旧({len(old_r):>3}只) 胜率={old_wr:>5.1f}% 均收益={old_avg:>+6.2f}% | "
                  f"新({len(new_r):>3}只) 胜率={new_wr:>5.1f}% 均收益={new_avg:>+6.2f}% | "
                  f"Δ胜率={delta_wr:>+5.1f}pp{wr_mark} Δ收益={delta_avg:>+5.2f}%{avg_mark}")

        period_detail["old_rets"] = {k: v for k, v in period_old_rets.items()}
        period_detail["new_rets"] = {k: v for k, v in period_new_rets.items()}
        all_results.append(period_detail)
        print()

    # ═══════════════════════════════════════════════════════════════════════
    #  全局汇总
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  全局汇总")
    print("=" * 70)

    # 分牛/熊汇总
    bull_old = {"t3": [], "t5": [], "t8": []}
    bull_new = {"t3": [], "t5": [], "t8": []}
    bear_old = {"t3": [], "t5": [], "t8": []}
    bear_new = {"t3": [], "t5": [], "t8": []}
    all_old  = {"t3": [], "t5": [], "t8": []}
    all_new  = {"t3": [], "t5": [], "t8": []}

    for pd_item in all_results:
        regimes = pd_item["new"].get("regimes", [])
        is_bull_period = any(r in ("bull", "neutral") for r in regimes)
        is_bear_period = any(r == "bear" for r in regimes)

        for key in ("t3", "t5", "t8"):
            old_r = pd_item.get("old_rets", {}).get(key, [])
            new_r = pd_item.get("new_rets", {}).get(key, [])
            all_old[key].extend(old_r)
            all_new[key].extend(new_r)
            if is_bull_period:
                bull_old[key].extend(old_r)
                bull_new[key].extend(new_r)
            if is_bear_period:
                bear_old[key].extend(old_r)
                bear_new[key].extend(new_r)

    def _summary_line(label, old_rets, new_rets):
        if not old_rets and not new_rets:
            return f"  {label}: 无数据"
        old_n = len(old_rets)
        new_n = len(new_rets)
        old_wr = len([r for r in old_rets if r > 0]) / max(old_n, 1) * 100
        new_wr = len([r for r in new_rets if r > 0]) / max(new_n, 1) * 100
        old_avg = sum(old_rets) / max(old_n, 1)
        new_avg = sum(new_rets) / max(new_n, 1)
        d_wr = new_wr - old_wr
        d_avg = new_avg - old_avg
        winner = "新版胜" if (d_wr > 1 and d_avg > 0) else "旧版胜" if (d_wr < -1 and d_avg < 0) else "平局"
        return (f"  {label}: "
                f"旧({old_n:>4}只) 胜率={old_wr:.1f}% 均收益={old_avg:+.2f}% | "
                f"新({new_n:>4}只) 胜率={new_wr:.1f}% 均收益={new_avg:+.2f}% | "
                f"Δ={d_wr:+.1f}pp/{d_avg:+.2f}% → {winner}")

    print("\n[全部时间段]")
    for key_label, key in [("T+3", "t3"), ("T+5", "t5"), ("T+8", "t8")]:
        print(_summary_line(key_label, all_old[key], all_new[key]))

    print("\n[牛市/震荡时间段]")
    for key_label, key in [("T+3", "t3"), ("T+5", "t5"), ("T+8", "t8")]:
        print(_summary_line(key_label, bull_old[key], bull_new[key]))

    print("\n[熊市时间段]")
    for key_label, key in [("T+3", "t3"), ("T+5", "t5"), ("T+8", "t8")]:
        print(_summary_line(key_label, bear_old[key], bear_new[key]))

    # L1/L2 候选数量对比
    print("\n[候选数量对比]")
    old_l1_avg = np.mean([c for pd_item in all_results for c in pd_item["old"]["l1_counts"]])
    new_l1_avg = np.mean([c for pd_item in all_results for c in pd_item["new"]["l1_counts"]])
    old_l2_avg = np.mean([c for pd_item in all_results for c in pd_item["old"]["l2_counts"]])
    new_l2_avg = np.mean([c for pd_item in all_results for c in pd_item["new"]["l2_counts"]])
    print(f"  L1 均入选: 旧={old_l1_avg:.0f} 只  新={new_l1_avg:.0f} 只")
    print(f"  L2 均入选: 旧={old_l2_avg:.0f} 只  新={new_l2_avg:.0f} 只")

    # 保存结果
    save_path = OUTPUT_DIR / "backtest_ab_compare.json"
    save_data = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "periods": [],
    }
    for pd_item in all_results:
        save_data["periods"].append({
            "period": pd_item["period"],
            "dates": pd_item["dates"],
            "regimes": pd_item["new"]["regimes"],
            "old_l1_counts": pd_item["old"]["l1_counts"],
            "new_l1_counts": pd_item["new"]["l1_counts"],
            "old_l2_counts": pd_item["old"]["l2_counts"],
            "new_l2_counts": pd_item["new"]["l2_counts"],
            "old_t5_wr": len([r for r in pd_item["old_rets"]["t5"] if r > 0]) / max(len(pd_item["old_rets"]["t5"]), 1) * 100,
            "new_t5_wr": len([r for r in pd_item["new_rets"]["t5"] if r > 0]) / max(len(pd_item["new_rets"]["t5"]), 1) * 100,
            "old_t5_avg": sum(pd_item["old_rets"]["t5"]) / max(len(pd_item["old_rets"]["t5"]), 1),
            "new_t5_avg": sum(pd_item["new_rets"]["t5"]) / max(len(pd_item["new_rets"]["t5"]), 1),
        })
    save_path.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果已保存: {save_path}")
    print("\n" + "=" * 70)
    print("  回测完成")
    print("=" * 70)


if __name__ == "__main__":
    run_compare()
