#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史回测框架 — 管线 B 量化层（L1 + L2）

目标：
  评估过去1年内每周一、三、五的选股在 T+3/T+5/T+8 的实际表现，
  给出管线 A / B 的改进建议。

Point-in-time 保证：
  · 价格/量能：TDX 本地数据，按 cutoff_date 截断，严格不引入未来数据
  · 基本面（PE/市值）：历史快照缺失，本回测跳过该过滤（作为局限性说明）
  · 新闻/公告：L1/L2 为纯量价逻辑，不涉及新闻
  · 管线 A（LLM）：无法历史回测（LLM 知识截止时间 ≠ 历史时点信息），已排除

运行方式：
  python historical_backtest.py                  # 默认回测12个月
  python historical_backtest.py --months 6       # 回测6个月
  python historical_backtest.py --report-only    # 仅输出已有结果的报告
  python historical_backtest.py --suggest        # 额外调用LLM生成改进建议
"""

import argparse
import json
import os
import sys
import time
import concurrent.futures
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# ── 路径 ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
BT_DIR = OUTPUT_DIR / "backtest_history"
BT_DIR.mkdir(parents=True, exist_ok=True)

BT_RESULTS_PATH = BT_DIR / "results.json"
BT_REPORT_PATH  = BT_DIR / "report.md"


# ═══════════════════════════════════════════════════════════════════════════
#  交易日历
# ═══════════════════════════════════════════════════════════════════════════

def get_trading_calendar(months: int = 12) -> List[str]:
    """
    获取过去 N 个月内的交易日（YYYYMMDD），过滤出周一=0、周三=2、周五=4。
    数据来源: akshare tool_trade_date_hist_sina（缓存到本地）
    """
    cache_path = BT_DIR / "trading_calendar.json"

    # 尝试读缓存（当天内有效）
    today_str = datetime.now().strftime("%Y%m%d")
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("date") == today_str:
                all_dates = cached["dates"]
                print(f"  [日历] 使用缓存（{len(all_dates)} 个交易日）")
                return _filter_weekdays(all_dates, months)
        except Exception:
            pass

    # 从 akshare 获取
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        # 返回 DataFrame，列名 'trade_date'，值为 datetime.date 或 str
        dates = df.iloc[:, 0].astype(str).str.replace("-", "").tolist()
        cache_path.write_text(
            json.dumps({"date": today_str, "dates": dates}, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"  [日历] 从 akshare 获取 {len(dates)} 个交易日")
        return _filter_weekdays(dates, months)
    except Exception as e:
        print(f"  [日历] akshare 获取失败: {e}，使用粗略估算")
        return _fallback_calendar(months)


def _filter_weekdays(all_dates: List[str], months: int) -> List[str]:
    """保留过去 months 个月的周一(0)、周三(2)、周五(4)"""
    cutoff = (datetime.now() - timedelta(days=months * 31)).strftime("%Y%m%d")
    today_str = datetime.now().strftime("%Y%m%d")
    result = []
    for d in all_dates:
        if d < cutoff or d >= today_str:
            continue
        try:
            dt = datetime.strptime(d, "%Y%m%d")
            if dt.weekday() in (0, 2, 4):  # Mon, Wed, Fri
                result.append(d)
        except ValueError:
            continue
    result.sort()
    return result


def _fallback_calendar(months: int) -> List[str]:
    """akshare 不可用时的后备：按自然日生成（含少量非交易日，回测时会得到空结果）"""
    result = []
    today = datetime.now()
    start = today - timedelta(days=months * 31)
    cur = start
    while cur < today:
        if cur.weekday() in (0, 2, 4):
            result.append(cur.strftime("%Y%m%d"))
        cur += timedelta(days=1)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  TDX 数据预加载缓存
# ═══════════════════════════════════════════════════════════════════════════

class TdxCache:
    """
    一次性将所有股票的 TDX 日线数据加载到内存，避免逐日回测时重复磁盘读取。
    每只股票只读一次磁盘（约5000次），后续按日期过滤均在内存中完成。
    """

    def __init__(self, de):
        self._de = de
        self._data: Dict[str, pd.DataFrame] = {}  # code -> full df
        self._loaded = False

    def preload(self, codes: List[str], verbose: bool = True):
        """预加载所有代码的完整历史日线"""
        if self._loaded:
            return
        t0 = time.time()
        total = len(codes)
        print(f"  [缓存] 预加载 {total} 只股票的 TDX 日线数据…")

        # 顺序加载：TDX 本地读取约 5ms/只，5000只约30秒，不用多线程
        last_code = ""
        for i, code in enumerate(codes):
            last_code = code
            try:
                if self._de._tdx is not None:
                    df = self._de._tdx.daily(symbol=code)
                    if df is not None and len(df) > 0:
                        self._data[code] = df
            except Exception:
                pass
            if verbose and (i + 1) % 500 == 0:
                print(f"  [缓存] {i+1}/{total}… last={last_code}", flush=True)
        print(f"  [缓存] for循环已完成，last={last_code}", flush=True)
        print(f"  [缓存] self._data 大小: {len(self._data)}", flush=True)
        elapsed = time.time() - t0
        print(f"  [缓存] 完成：{len(self._data)}/{total} 只已加载，耗时 {elapsed:.1f}s", flush=True)
        self._loaded = True
        print(f"  [缓存] loaded=True 已设置", flush=True)

    def get_bars(self, code: str, cutoff_date: str, days: int = 35) -> Optional[pd.DataFrame]:
        """返回 cutoff_date 当天及之前的最后 days 根 K 线"""
        df = self._data.get(code)
        if df is None or len(df) == 0:
            return None
        try:
            if 'date' in df.columns:
                date_str = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
            else:
                date_str = pd.Series(df.index).apply(
                    lambda x: pd.to_datetime(x).strftime('%Y%m%d')
                )
            mask = date_str.values <= cutoff_date
            filtered = df[mask]
            if len(filtered) < 20:
                return None
            return filtered.tail(days + 10)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  L1 历史过滤（point-in-time，直接在内存数据上运行）
# ═══════════════════════════════════════════════════════════════════════════

def _calc_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _compute_rps(cache: TdxCache, codes: List[str],
                 cutoff_date: str, lookback: int = 20) -> Dict[str, float]:
    """
    计算全市场 RPS（相对强度百分位）。
    返回 {code: 20日涨幅百分比}，供 L1 过滤和 L2 打分使用。
    """
    returns = {}
    for code in codes:
        df = cache.get_bars(code, cutoff_date, days=lookback + 5)
        if df is None or len(df) < lookback:
            continue
        c_col = "close" if "close" in df.columns else ("收盘" if "收盘" in df.columns else None)
        if c_col is None:
            continue
        closes = df[c_col].astype(float).values
        if len(closes) >= lookback and closes[-lookback] > 0:
            ret = (closes[-1] - closes[-lookback]) / closes[-lookback]
            returns[code] = ret
    return returns


def l1_filter_at_date(cache: TdxCache, codes: List[str],
                      cutoff_date: str) -> List[Dict]:
    """
    L1 蓄势预过滤（v7 = v4的L1硬过滤 + v6的丰富因子传递）。

    硬性条件（沿用v4，已验证54.5%胜率）：
      1. RPS 30%~70%
      2. 收盘价 > MA20
      3. 看空形态淘汰
    蓄势条件（至少满足2条）：
      A. MA多头排列（MA5>MA10>MA20）
      B. ATR收窄
      C. 近5日无单日涨幅>5%
    新增：向L2传递丰富因子（MA20斜率、流动性、近高点、收益结构）用于评分。
    """
    all_rps = _compute_rps(cache, codes, cutoff_date, lookback=20)
    if all_rps:
        rps_values = sorted(all_rps.values())
        low_idx  = int(len(rps_values) * 0.30)
        high_idx = int(len(rps_values) * 0.70)
        rps_low  = rps_values[min(low_idx, len(rps_values) - 1)]
        rps_high = rps_values[min(high_idx, len(rps_values) - 1)]
    else:
        rps_low, rps_high = -999, 999

    results = []

    for code in codes:
        code_rps = all_rps.get(code)
        if code_rps is None or code_rps < rps_low or code_rps > rps_high:
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

        # ── 硬性过滤（v4 原样保留）────────────────────────────────────
        ma5_s  = _calc_sma(pd.Series(close_arr), 5)
        ma10_s = _calc_sma(pd.Series(close_arr), 10)
        ma20_s = _calc_sma(pd.Series(close_arr), 20)

        if ma5_s.iloc[-1] != ma5_s.iloc[-1]:
            continue

        ma5  = ma5_s.iloc[-1]
        ma10 = ma10_s.iloc[-1]
        ma20 = ma20_s.iloc[-1]

        if ma5 < ma10 < ma20:
            continue
        if close_arr[-1] <= ma20:
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

        # ── 蓄势条件（至少满足2条）────────────────────────────────────
        cond_a = (ma5 > ma10 > ma20)

        tr_arr = np.maximum(high_arr - low_arr,
                            np.maximum(np.abs(high_arr - np.roll(close_arr, 1)),
                                       np.abs(low_arr - np.roll(close_arr, 1))))
        if n >= 20:
            atr_5  = np.mean(tr_arr[-5:])
            atr_20 = np.mean(tr_arr[-20:])
            cond_b = (atr_5 < atr_20) if atr_20 > 0 else False
        else:
            cond_b = False

        if n >= 6:
            daily_returns = (close_arr[-5:] - close_arr[-6:-1]) / np.maximum(close_arr[-6:-1], 0.01)
            cond_c = bool(np.all(daily_returns < 0.05))
        else:
            cond_c = False

        cond_count = sum([cond_a, cond_b, cond_c])
        if cond_count < 2:
            continue

        # ── 计算丰富因子（传递给L2评分）───────────────────────────────
        vol_ma20 = np.mean(vol_arr[-20:]) if n >= 20 else vol_arr.mean()
        vol_ratio = vol_arr[-1] / max(vol_ma20, 1)

        # MA20斜率（软因子，不做硬过滤）
        ma20_slope = 0
        if len(ma20_s) >= 6:
            ma20_prev = ma20_s.iloc[-6]
            if ma20_prev == ma20_prev and abs(ma20_prev) > 0.01:
                ma20_slope = (ma20 - ma20_prev) / abs(ma20_prev)

        # 20日日均成交额（软因子）
        avg_amount_20 = np.mean(close_arr[-20:] * vol_arr[-20:] * 100)

        # 近60日高点距离
        if n >= 60:
            high_60 = np.max(high_arr[-60:])
        else:
            high_60 = np.max(high_arr)
        near_high_60 = close_arr[-1] / max(high_60, 0.01)

        # 短期收益结构
        ret5  = (close_arr[-1] / close_arr[-6]  - 1) * 100 if n >= 6  else 0
        ret10 = (close_arr[-1] / close_arr[-11] - 1) * 100 if n >= 11 else 0
        ret20 = (close_arr[-1] / close_arr[-21] - 1) * 100 if n >= 21 else 0

        # MA60
        ma60 = np.mean(close_arr[-60:]) if n >= 60 else None

        results.append({
            "code": code,
            "close": float(round(close_arr[-1], 2)),
            "MA5":   float(round(ma5, 2)),
            "MA10":  float(round(ma10, 2)),
            "MA20":  float(round(ma20, 2)),
            "MA60":  float(round(ma60, 2)) if ma60 else None,
            "ma20_slope": float(round(ma20_slope * 100, 3)),
            "vol_ratio": float(round(vol_ratio, 3)),
            "avg_amount_20": float(round(avg_amount_20 / 1e8, 2)),
            "near_high_60": float(round(near_high_60, 4)),
            "ret5":  float(round(ret5, 2)),
            "ret10": float(round(ret10, 2)),
            "ret20": float(round(ret20, 2)),
            "rps_20d": float(round(code_rps * 100, 2)),
            "cond_count": cond_count,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  L2 缠论结构确认 + 综合评分 Top30（融合版 v3）
# ═══════════════════════════════════════════════════════════════════════════

# 缠论库初始化（模块级，只加载一次）
_chan_available = False
try:
    import sys as _sys
    _chan_dir = str(BASE_DIR / "chan")
    if _chan_dir not in _sys.path:
        _sys.path.insert(0, _chan_dir)
    from chan.Chan import CChan
    from chan.ChanConfig import CChanConfig
    from chan.Common.CEnum import AUTYPE, BSP_TYPE, DATA_SRC, KL_TYPE, DATA_FIELD
    from chan.Common.CTime import CTime
    from chan.KLine.KLine_Unit import CKLine_Unit
    from chan.Common.func_util import str2float
    _chan_available = True
except Exception as _e:
    print(f"  [警告] 缠论库加载失败: {_e}，L2缠论将不可用")

# 通道反转指标初始化
_cr_available = False
try:
    _cr_dir = str(Path("D:/BaiduSyncdisk/aicoding/stockagent-analysis/src"))
    if _cr_dir not in sys.path:
        sys.path.insert(0, _cr_dir)
    from stockagent_analysis.channel_reversal import compute_channel, detect_phases
    _cr_available = True
except Exception as _e:
    print(f"  [警告] 通道反转指标加载失败: {_e}，L2将不含channel_reversal")


def _run_chan_analysis(df300: pd.DataFrame, code: str) -> Dict:
    """
    对单只股票运行缠论分析，返回买卖点信息。
    返回: {"has_buy": bool, "has_sell": bool, "buy_types": list, "detail": str}
    """
    result = {"has_buy": False, "has_sell": False, "buy_types": [], "detail": ""}

    if not _chan_available or df300 is None or len(df300) < 60:
        return result

    try:
        # 构造K线数据
        c_col = "close" if "close" in df300.columns else ("收盘" if "收盘" in df300.columns else None)
        o_col = "open"  if "open"  in df300.columns else ("开盘" if "开盘" in df300.columns else None)
        h_col = "high"  if "high"  in df300.columns else ("最高" if "最高" in df300.columns else None)
        l_col = "low"   if "low"   in df300.columns else ("最低" if "最低" in df300.columns else None)
        v_col = "volume" if "volume" in df300.columns else (
                "vol" if "vol" in df300.columns else (
                "成交量" if "成交量" in df300.columns else None))

        if not all([c_col, o_col, h_col, l_col]):
            return result

        df_kline = df300.tail(250)
        klu_list = []

        for idx in range(len(df_kline)):
            row = df_kline.iloc[idx]
            # 获取日期
            date_idx = df_kline.index[idx]
            if hasattr(date_idx, 'strftime'):
                date_val = date_idx.strftime("%Y-%m-%d")
            else:
                date_val = str(date_idx)[:10]
            if len(date_val) != 10:
                continue
            year, month, day = int(date_val[:4]), int(date_val[5:7]), int(date_val[8:10])

            o = str2float(row.get(o_col, 0))
            h = str2float(row.get(h_col, 0))
            l = str2float(row.get(l_col, 0))
            c = str2float(row.get(c_col, 0))
            h = max(o, c, h)
            l = min(o, c, l)

            item = {
                DATA_FIELD.FIELD_TIME: CTime(year, month, day, 0, 0, auto=False),
                DATA_FIELD.FIELD_OPEN: o,
                DATA_FIELD.FIELD_HIGH: h,
                DATA_FIELD.FIELD_LOW: l,
                DATA_FIELD.FIELD_CLOSE: c,
            }
            if v_col:
                vol_val = row.get(v_col, None)
                if vol_val is not None:
                    item[DATA_FIELD.FIELD_VOLUME] = str2float(vol_val)
            klu = CKLine_Unit(item)
            klu_list.append(klu)

        if len(klu_list) < 30:
            return result

        # 创建缠论实例
        config = CChanConfig()
        config.trigger_step = True
        config.print_warning = False
        config.print_err_time = False
        ch = CChan(
            code=code,
            begin_time=None,
            end_time=None,
            data_src=DATA_SRC.BAO_STOCK,
            lv_list=[KL_TYPE.K_DAY],
            config=config,
            autype=AUTYPE.QFQ,
        )
        ch.trigger_load({KL_TYPE.K_DAY: klu_list})

        # 获取买卖点（最近5个）
        bsp_list = ch.get_latest_bsp(number=5)
        details = []
        for bsp in bsp_list:
            for bt in bsp.type:
                if bsp.is_buy:
                    result["has_buy"] = True
                    result["buy_types"].append(bt)
                    details.append(f"买({bt.value})")
                else:
                    result["has_sell"] = True
                    details.append(f"卖({bt.value})")
        result["detail"] = ";".join(details)

    except Exception:
        pass

    return result


def _run_channel_reversal(df300: pd.DataFrame) -> Dict:
    """
    对单只股票运行通道反转分析。
    返回: {"cr_score": float, "phase": str, "phase_days": int, "is_bearish": bool}
    """
    result = {"cr_score": 0, "phase": "", "phase_days": 0, "is_bearish": False}
    if not _cr_available or df300 is None or len(df300) < 160:
        return result
    try:
        # 标准化列名
        col_map = {
            "收盘": "close", "开盘": "open", "最高": "high",
            "最低": "low", "成交量": "volume",
        }
        df = df300.rename(columns={k: v for k, v in col_map.items() if k in df300.columns}).copy()
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = compute_channel(df)
        df = detect_phases(df)

        last = df.iloc[-1]
        phase = str(last.get("phase", ""))
        cr_score = float(last.get("cr_score", 0))
        phase_days = int(last.get("phase_days", 0))

        result["cr_score"] = cr_score
        result["phase"] = phase
        result["phase_days"] = phase_days

        # 看空阶段标记（用于风控过滤器）
        # Phase 0D=空头区间, 3A=弱反弹(大概率破新低)
        if phase in ("0D", "3A"):
            result["is_bearish"] = True

    except Exception:
        pass
    return result


def _risk_filter(chan_result: Dict, cr_result: Dict,
                 close_arr: np.ndarray, vol_arr: np.ndarray) -> Optional[str]:
    """
    看空风控过滤器（仅极端条件）：返回拒绝原因字符串，None表示通过。

    规则（仅保留最极端的否决）：
      1. 缠论卖点（无买点时）一票否决
      2. 连续5日下跌 → 否决
      3. 当日跌幅>7% → 否决

    注：通道空头阶段(0D/3A)不否决 — 对蓄势策略来说，
        通道下半区恰是低位蓄势区，否决会排除好股票。
    """
    # 1. 缠论卖点否决
    if chan_result["has_sell"] and not chan_result["has_buy"]:
        return "缠论卖点"

    # 2. 连续5日下跌
    n = len(close_arr)
    if n >= 6:
        drops = sum(1 for i in range(-5, 0) if close_arr[i] < close_arr[i - 1])
        if drops >= 5:
            return "连续5日下跌"

    # 3. 当日暴跌>7%
    if n >= 2 and close_arr[-2] > 0:
        day_ret = (close_arr[-1] - close_arr[-2]) / close_arr[-2]
        if day_ret < -0.07:
            return "当日暴跌"

    return None  # 通过


def l2_score_and_rank(cache: TdxCache, l1_candidates: List[Dict],
                      cutoff_date: str, top_n: int = 30) -> List[Dict]:
    """
    L2 蓄势评分 + Top30 排名（最终版 — 回测验证最优的v3评分体系）。

    综合评分 = 蓄势形态分(40%) + 量能确认分(30%) + RPS动量分(30%)

    风控：连续5日下跌 / 当日暴跌>7% 否决（轻量）。
    """
    scored = []

    # 收集候选池RPS用于百分位
    all_rps = [c.get("rps_20d", 0) for c in l1_candidates]
    rps_arr = np.array(all_rps) if all_rps else np.array([0])

    for item in l1_candidates:
        code = item["code"]

        # ── 获取300根K线 ─────────────────────────────────────────────
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
        n300 = len(close300)

        # ── 轻量风控 ────────────────────────────────────────────────
        if n300 >= 6:
            drops = sum(1 for i in range(-5, 0) if close_v[i] < close_v[i - 1])
            if drops >= 5:
                continue
        if n300 >= 2 and close_v[-2] > 0:
            if (close_v[-1] - close_v[-2]) / close_v[-2] < -0.07:
                continue

        # ═══════ 蓄势形态分（0~100，权重40%）═════════════════════════
        ma5  = item.get("MA5", 0)
        ma10 = item.get("MA10", 0)
        ma20 = item.get("MA20", 0)
        if ma5 > ma10 > ma20:
            ma_score = 30
        elif ma5 > ma10:
            ma_score = 15
        else:
            ma_score = 0

        # ATR收窄
        high300 = df300[h_col].astype(float).values
        low300  = df300[l_col].astype(float).values
        if n300 >= 20:
            tr = np.maximum(high300 - low300,
                            np.maximum(np.abs(high300 - np.roll(close_v, 1)),
                                       np.abs(low300 - np.roll(close_v, 1))))
            atr_5  = np.mean(tr[-5:])
            atr_20 = np.mean(tr[-20:])
            atr_ratio = atr_5 / max(atr_20, 0.001)
            atr_score = max(0, min(30, (1.0 - atr_ratio) * 60))
        else:
            atr_score = 0

        # 站上MA20 + MA60
        ma20_val = close300.rolling(20, min_periods=20).mean().iloc[-1]
        above_ma20 = 15 if (ma20_val == ma20_val and close300.iloc[-1] > ma20_val) else 0
        ma60_score = 0
        if n300 >= 60:
            ma60_val = close300.rolling(60, min_periods=60).mean().iloc[-1]
            if ma60_val == ma60_val and close300.iloc[-1] > ma60_val:
                ma60_score = 15

        # MACD（DIF>DEA=多头排列加分，不要求DIF>0）
        ema12 = close300.ewm(span=12, adjust=False).mean()
        ema26 = close300.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_bonus = 10 if dif.iloc[-1] > dea.iloc[-1] else 0

        form_score = min(ma_score + atr_score + above_ma20 + ma60_score + macd_bonus, 100)

        # ═══════ 量能确认分（0~100，权重30%）═════════════════════════
        vol300 = df300[v_col].astype(float).values if v_col else None
        vol_score = 0
        if vol300 is not None and n300 >= 20:
            vol_ma5  = np.mean(vol300[-5:])
            vol_ma20 = np.mean(vol300[-20:])
            if vol_ma5 < vol_ma20 * 0.8:
                vol_score += 40  # 明显缩量（蓄势信号）
            elif vol_ma5 < vol_ma20:
                vol_score += 20

            if len(vol300) >= 5:
                vol_std = np.std(vol300[-5:])
                vol_mean = np.mean(vol300[-5:])
                vol_cv = vol_std / max(vol_mean, 1)
                vol_score += max(0, min(30, (1.0 - vol_cv) * 40))

            open300 = df300[o_col].astype(float).values
            if close_v[-1] > open300[-1] and vol300[-1] < vol_ma20 * 1.5:
                vol_score += 30
        vol_score = min(vol_score, 100)

        # ═══════ RPS动量分（0~100，权重30%）══════════════════════════
        rps_val = item.get("rps_20d", 0)
        rps_pct = float(np.searchsorted(np.sort(rps_arr), rps_val)) / max(len(rps_arr), 1)
        rps_score = rps_pct * 50

        if n300 >= 5:
            ret_5d = (close300.iloc[-1] - close300.iloc[-5]) / close300.iloc[-5] * 100
        else:
            ret_5d = 0
        excess_score = min(max(ret_5d * 5, 0), 50)

        momentum_score = rps_score + excess_score

        # ═══════ 综合评分 ════════════════════════════════════════════
        total = form_score * 0.40 + vol_score * 0.30 + momentum_score * 0.30

        scored.append({
            "code":       code,
            "close":      item["close"],
            "total_score": round(total, 1),
            "form_score": round(form_score, 1),
            "vol_score":  round(vol_score, 1),
            "mom_score":  round(momentum_score, 1),
            "rps_20d":    item.get("rps_20d", 0),
            "vol_ratio":  item.get("vol_ratio", 0),
        })

    scored.sort(key=lambda x: -x["total_score"])
    return scored[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
#  业绩评估
# ═══════════════════════════════════════════════════════════════════════════

def _trading_day_offset(base_date: str, n_trading_days: int) -> str:
    """粗略估算 T+N 交易日对应的自然日（1交易日≈1.4自然日）"""
    dt = datetime.strptime(base_date, "%Y%m%d")
    natural = int(n_trading_days * 1.4) + 2
    return (dt + timedelta(days=natural)).strftime("%Y%m%d")


def measure_performance_tdx(cache: 'TdxCache', codes: List[str], rec_date: str,
                            rec_prices: Dict[str, float]) -> Dict:
    """
    用 TDX 缓存数据测量 T+3/T+5/T+8 收益率（纯内存查表，秒级完成）。
    """
    today_str = datetime.now().strftime("%Y%m%d")
    perf = {"rec_date": rec_date, "stocks": []}

    for code in codes:
        rec_close = rec_prices.get(code)
        if not rec_close or rec_close <= 0:
            continue

        # 获取该股票完整数据（从缓存，不过滤日期）
        full_df = cache._data.get(code)
        if full_df is None or len(full_df) == 0:
            continue

        # 标准化日期列
        c_col = "close" if "close" in full_df.columns else ("收盘" if "收盘" in full_df.columns else None)
        if c_col is None:
            continue
        if 'date' in full_df.columns:
            dates = pd.to_datetime(full_df['date']).dt.strftime('%Y%m%d').values
        else:
            dates = pd.Series(full_df.index).apply(
                lambda x: pd.to_datetime(x).strftime('%Y%m%d')).values
        closes = full_df[c_col].astype(float).values

        entry = {"code": code, "rec_close": rec_close}
        for period in (3, 5, 8):
            tgt = _trading_day_offset(rec_date, period)
            if tgt > today_str:
                entry[f"return_t{period}"] = None
                continue
            # 找 >= tgt 的第一个交易日
            idx = np.searchsorted(dates, tgt, side='left')
            if idx < len(dates):
                p = float(closes[idx])
                if p > 0:
                    entry[f"return_t{period}"] = round((p - rec_close) / rec_close * 100, 2)
                else:
                    entry[f"return_t{period}"] = None
            else:
                entry[f"return_t{period}"] = None
        perf["stocks"].append(entry)

    # 汇总
    for period in (3, 5, 8):
        rets = [s[f"return_t{period}"] for s in perf["stocks"]
                if s.get(f"return_t{period}") is not None]
        if rets:
            wins = [r for r in rets if r > 0]
            perf[f"summary_t{period}"] = {
                "n":          len(rets),
                "win_rate":   round(len(wins) / len(rets) * 100, 1),
                "avg_return": round(sum(rets) / len(rets), 2),
                "max_gain":   round(max(rets), 2),
                "max_loss":   round(min(rets), 2),
            }

    return perf


# ═══════════════════════════════════════════════════════════════════════════
#  主回测引擎
# ═══════════════════════════════════════════════════════════════════════════

class HistoricalBacktest:
    """
    历史回测引擎：在过去 N 个月的每周一/三/五运行 L1+L2，
    评估后续 T+3/T+5/T+8 表现，生成管线改进建议。
    """

    def __init__(self, months: int = 12, verbose: bool = True):
        self.months  = months
        self.verbose = verbose
        self._de     = None
        self._cache  = None
        self._codes  = []

    # ── 初始化 ──────────────────────────────────────────────────────────

    def _init_engine(self):
        """初始化 DataEngine 和 TDX 缓存"""
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")

        from data_engine import DataEngine
        tdx_dir = os.getenv("TDX_DIR", "d:/tdx")
        self._de = DataEngine(
            tushare_token=os.getenv("TUSHARE_TOKEN", ""),
            tdx_dir=tdx_dir,
        )
        if self._de._tdx is None:
            raise RuntimeError("TDX 未连接，请检查 TDX_DIR 配置")

        # 加载股票代码列表
        sl_path = BASE_DIR / "output" / "stock_list.json"
        if sl_path.exists():
            sl = json.loads(sl_path.read_text(encoding="utf-8"))
            self._codes = list(sl.get("data", {}).keys())
        else:
            # 回退：生成代码范围
            codes = []
            for c in range(600000, 602000): codes.append(f"{c:06d}")
            for c in range(688000, 688300): codes.append(f"{c:06d}")
            for c in range(1, 2000):        codes.append(f"{c:06d}")
            for c in range(300001, 302000): codes.append(f"{c:06d}")
            self._codes = codes

        # 限制最多5000只，末尾部分通常是问题代码
        self._codes = self._codes[:5000]
        print(f"  [引擎] 股票代码: {len(self._codes)} 只（上限5000）")

        # 预加载 TDX 数据
        self._cache = TdxCache(self._de)
        self._cache.preload(self._codes, verbose=self.verbose)

    # ── 单日运行 ────────────────────────────────────────────────────────

    def run_date(self, cutoff_date: str) -> Dict:
        """
        在 cutoff_date 运行 L1 + L2，返回选股结果（不含未来数据）。
        cutoff_date 是该交易日的收盘时间点。
        """
        t0 = time.time()

        # L1
        l1 = l1_filter_at_date(self._cache, self._codes, cutoff_date)

        # L2（P1改进：综合评分 + Top30 排名）
        l2 = l2_score_and_rank(self._cache, l1, cutoff_date, top_n=30)

        elapsed = time.time() - t0

        # 记录推荐日收盘价（作为后续绩效评估的买入价）
        rec_prices_l1 = {item["code"]: item["close"] for item in l1}
        rec_prices_l2 = {item["code"]: item["close"] for item in l2}

        result = {
            "cutoff_date": cutoff_date,
            "weekday":     datetime.strptime(cutoff_date, "%Y%m%d").strftime("%A"),
            "l1_count":    len(l1),
            "l2_count":    len(l2),
            "l1_codes":    [i["code"] for i in l1],
            "l2_codes":    [i["code"] for i in l2],
            "rec_prices_l1": rec_prices_l1,
            "rec_prices_l2": rec_prices_l2,
            "scan_seconds": round(elapsed, 1),
            "perf_l1":  None,  # 填充于 evaluate() 阶段
            "perf_l2":  None,
        }

        if self.verbose:
            print(f"  [{cutoff_date}] L1={len(l1)} L2={len(l2)}  ({elapsed:.1f}s)")

        return result

    # ── 绩效评估阶段 ────────────────────────────────────────────────────

    def evaluate(self, result: Dict) -> Dict:
        """对已保存的选股结果补充 T+3/T+5/T+8 绩效（用 TDX 缓存，秒级完成）"""
        today_str = datetime.now().strftime("%Y%m%d")
        tgt_t3 = _trading_day_offset(result["cutoff_date"], 3)
        if tgt_t3 > today_str:
            return result  # T+3 尚未到期，跳过

        if result["l1_codes"]:
            result["perf_l1"] = measure_performance_tdx(
                self._cache, result["l1_codes"], result["cutoff_date"],
                result["rec_prices_l1"]
            )
        if result["l2_codes"]:
            result["perf_l2"] = measure_performance_tdx(
                self._cache, result["l2_codes"], result["cutoff_date"],
                result["rec_prices_l2"]
            )
        return result

    # ── 完整回测流程 ────────────────────────────────────────────────────

    def run(self) -> List[Dict]:
        """
        全量历史回测：
        1. 生成交易日历
        2. 预加载 TDX 数据
        3. 逐日运行 L1+L2 扫描
        4. 评估可评估日期的绩效
        5. 保存结果
        """
        print(f"\n{'='*65}")
        print(f"  历史回测 — 过去 {self.months} 个月每周一/三/五")
        print(f"  数据层：TDX 本地（point-in-time，无未来泄露）")
        print(f"  L1=蓄势预过滤(RPS中位+ATR收窄+MA多头)  L2=缠论买点确认+Top30")
        print(f"{'='*65}\n")

        # 初始化
        print("[步骤1] 初始化引擎…")
        self._init_engine()

        # 交易日历
        print("\n[步骤2] 生成交易日历…")
        dates = get_trading_calendar(self.months)
        print(f"  共 {len(dates)} 个回测日期（周一/三/五）")

        # 加载已有结果（增量模式：跳过已经扫描过的日期）
        existing = {}
        if BT_RESULTS_PATH.exists():
            try:
                raw = json.loads(BT_RESULTS_PATH.read_text(encoding="utf-8"))
                existing = {r["cutoff_date"]: r for r in raw}
                print(f"  已有 {len(existing)} 条缓存，跳过重复扫描")
            except Exception:
                pass

        # 逐日扫描
        print(f"\n[步骤3] 逐日 L1+L2 扫描…")
        results = []
        new_count = 0
        for d in dates:
            if d in existing:
                results.append(existing[d])
                continue
            r = self.run_date(d)
            results.append(r)
            existing[d] = r
            new_count += 1
            # 每10个日期保存一次中间结果
            if new_count % 10 == 0:
                self._save(list(existing.values()))

        print(f"  扫描完成：新增 {new_count} 个日期")

        # 绩效评估
        print(f"\n[步骤4] 获取 T+3/T+5/T+8 收盘价…")
        evaluated = 0
        for r in results:
            if r.get("perf_l1") is None and r.get("l1_codes"):
                r = self.evaluate(r)
                existing[r["cutoff_date"]] = r
                evaluated += 1
                if evaluated % 5 == 0:
                    print(f"  已评估 {evaluated} 个日期…")
                    self._save(list(existing.values()))
        print(f"  绩效评估完成：{evaluated} 个日期")

        # 保存
        self._save(list(existing.values()))
        print(f"\n  结果已保存: {BT_RESULTS_PATH}")

        return list(existing.values())

    def _save(self, results: List[Dict]):
        """保存回测结果（剔除 df 字段，写临时文件再重命名避免锁冲突）"""
        safe = []
        for r in results:
            safe.append({k: v for k, v in r.items() if k not in ("df",)})
        tmp = BT_RESULTS_PATH.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(safe, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8"
            )
            # 重命名（原子操作），如果目标被锁则重试
            for _ in range(3):
                try:
                    if BT_RESULTS_PATH.exists():
                        BT_RESULTS_PATH.unlink()
                    tmp.rename(BT_RESULTS_PATH)
                    break
                except PermissionError:
                    import time as _t; _t.sleep(1)
        except Exception as e:
            print(f"  [保存] 写入失败: {e}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  统计与报告生成
# ═══════════════════════════════════════════════════════════════════════════

def _collect_returns(results: List[Dict], layer: str,
                     period: int) -> List[float]:
    """收集指定层（l1/l2）指定周期（3/5/8）的所有收益率样本"""
    rets = []
    for r in results:
        perf = r.get(f"perf_{layer}")
        if not perf:
            continue
        s = perf.get(f"summary_t{period}")
        if not s:
            continue
        # 展开到个股维度（防止样本量被日期数压缩）
        for stk in perf.get("stocks", []):
            v = stk.get(f"return_t{period}")
            if v is not None:
                rets.append(v)
    return rets


def _win_rate(rets: List[float]) -> float:
    if not rets:
        return 0.0
    return round(len([r for r in rets if r > 0]) / len(rets) * 100, 1)


def _avg(rets: List[float]) -> float:
    return round(sum(rets) / len(rets), 2) if rets else 0.0


def compute_stats(results: List[Dict]) -> Dict:
    """计算回测统计摘要"""
    evaluated = [r for r in results
                 if r.get("perf_l1") and r["perf_l1"].get("summary_t5")]
    total_dates = len(results)

    stats = {
        "total_dates":     total_dates,
        "evaluated_dates": len(evaluated),
        "date_range":      (min(r["cutoff_date"] for r in results),
                            max(r["cutoff_date"] for r in results)) if results else ("", ""),
    }

    for layer in ("l1", "l2"):
        layer_stats = {}
        for period in (3, 5, 8):
            rets = _collect_returns(evaluated, layer, period)
            layer_stats[f"t{period}"] = {
                "n":          len(rets),
                "win_rate":   _win_rate(rets),
                "avg_return": _avg(rets),
                "max_gain":   round(max(rets), 2) if rets else 0,
                "max_loss":   round(min(rets), 2) if rets else 0,
            }

        # 每日平均入选数
        counts = [r[f"{layer}_count"] for r in results if r.get(f"{layer}_count") is not None]
        layer_stats["avg_selection_count"] = round(sum(counts) / len(counts), 1) if counts else 0

        # 按星期分析
        by_wd: Dict[str, List[float]] = defaultdict(list)
        for r in evaluated:
            wd = r.get("weekday", "")
            perf = r.get(f"perf_{layer}")
            if perf:
                for stk in perf.get("stocks", []):
                    v = stk.get("return_t5")
                    if v is not None:
                        by_wd[wd].append(v)
        layer_stats["by_weekday_t5"] = {
            wd: {"win_rate": _win_rate(v), "avg": _avg(v), "n": len(v)}
            for wd, v in by_wd.items()
        }

        stats[layer] = layer_stats

    # L2 是否在 L1 基础上提升了质量
    if stats.get("l1") and stats.get("l2"):
        l1_wr = stats["l1"]["t5"]["win_rate"]
        l2_wr = stats["l2"]["t5"]["win_rate"]
        stats["l2_vs_l1_t5_winrate_delta"] = round(l2_wr - l1_wr, 1)
        l1_avg = stats["l1"]["t5"]["avg_return"]
        l2_avg = stats["l2"]["t5"]["avg_return"]
        stats["l2_vs_l1_t5_return_delta"] = round(l2_avg - l1_avg, 2)

    return stats


def generate_report(results: List[Dict]) -> str:
    """生成 Markdown 报告 + 改进建议"""
    stats = compute_stats(results)

    lines = [
        "# 历史回测报告",
        f"\n> 回测区间：{stats['date_range'][0]} ~ {stats['date_range'][1]}",
        f"> 回测日期数：{stats['total_dates']}（已完成评估：{stats['evaluated_dates']}）",
        f"> 数据层：TDX 本地日线，point-in-time；PE/市值过滤已跳过（无历史快照）",
        f"> 管线A（LLM辩论）：未纳入历史回测（LLM 无法复现历史时点信息认知）\n",
    ]

    for layer_key, layer_name in [("l1", "管线B — L1 量化过滤"), ("l2", "管线B — L2 形态过滤")]:
        ls = stats.get(layer_key, {})
        lines.append(f"## {layer_name}")
        lines.append(f"\n**平均每日入选数**：{ls.get('avg_selection_count', 0)} 只\n")
        lines.append("| 评估周期 | 样本数 | 胜率 | 平均收益 | 最大盈 | 最大亏 |")
        lines.append("|---------|--------|------|---------|--------|--------|")
        for p in (3, 5, 8):
            s = ls.get(f"t{p}", {})
            lines.append(
                f"| T+{p} | {s.get('n',0)} | {s.get('win_rate',0)}% "
                f"| {s.get('avg_return',0):+.2f}% "
                f"| {s.get('max_gain',0):+.2f}% "
                f"| {s.get('max_loss',0):+.2f}% |"
            )
        # 星期分析
        by_wd = ls.get("by_weekday_t5", {})
        if by_wd:
            lines.append("\n**T+5 按星期分解**\n")
            lines.append("| 星期 | 样本数 | 胜率 | 均收益 |")
            lines.append("|------|--------|------|--------|")
            for wd in ("Monday", "Wednesday", "Friday"):
                d = by_wd.get(wd, {})
                cn = {"Monday": "周一", "Wednesday": "周三", "Friday": "周五"}.get(wd, wd)
                lines.append(
                    f"| {cn} | {d.get('n',0)} | {d.get('win_rate',0)}% | {d.get('avg',0):+.2f}% |"
                )
        lines.append("")

    # L2 vs L1
    d_wr  = stats.get("l2_vs_l1_t5_winrate_delta", 0)
    d_ret = stats.get("l2_vs_l1_t5_return_delta", 0)
    lines.append("## L2 过滤层有效性")
    lines.append(f"\nT+5 胜率变化：L1 → L2 = **{d_wr:+.1f}pp**")
    lines.append(f"T+5 均收益变化：L1 → L2 = **{d_ret:+.2f}%**")
    if d_wr > 5:
        lines.append("\n→ L2 过滤有效：胜率显著提升，建议保留并强化")
    elif d_wr > 0:
        lines.append("\n→ L2 过滤轻微正效果，可尝试放宽或改进条件")
    elif d_wr < -5:
        lines.append("\n→ L2 过滤产生负效果：筛掉了更多好股，建议检查代理信号是否匹配真实缠论买点")
    else:
        lines.append("\n→ L2 过滤效果中性，可考虑改用完整缠论（300根K线）替代代理信号")

    lines.append("\n---\n")

    # ── 改进建议 ──────────────────────────────────────────────────────
    lines.append("## 改进建议\n")

    l1_t5 = stats.get("l1", {}).get("t5", {})
    l1_wr = l1_t5.get("win_rate", 0)
    l1_avg = l1_t5.get("avg_return", 0)
    l2_t5 = stats.get("l2", {}).get("t5", {})
    l2_wr = l2_t5.get("win_rate", 0)
    avg_l1_count = stats.get("l1", {}).get("avg_selection_count", 0)

    suggestions_b = []

    # L1 建议
    if l1_wr < 45:
        suggestions_b.append(
            "**L1 胜率偏低（<45%）**：当前看空预筛条件不够严格，建议：\n"
            "  - 加入「连续2日阴线」淘汰条件（当前只有连续3日）\n"
            "  - 提高放量要求：vol_ratio > 80%（当前50%）\n"
            "  - 增加 RSP 相对强度 > 市场中位数的过滤"
        )
    elif l1_wr > 60:
        suggestions_b.append(
            "**L1 胜率良好（>60%）**：条件已相对严格，可考虑：\n"
            "  - 适当放宽放量阈值至 30%，扩大候选池"
        )

    if avg_l1_count > 200:
        suggestions_b.append(
            f"**L1 每日入选数过多（均值 {avg_l1_count:.0f} 只）**：需要更严格的前置过滤：\n"
            "  - 增加「收盘价站上20日均线」硬性要求\n"
            "  - 加入板块动量排名：只保留动量排名前30%的板块内个股"
        )
    elif avg_l1_count < 20:
        suggestions_b.append(
            f"**L1 每日入选数偏少（均值 {avg_l1_count:.0f} 只）**：条件可能过严：\n"
            "  - 将条件 A/B/C 改为「满足任一即通过」（当前OR逻辑已有此设计）\n"
            "  - 检查涨停淘汰是否过于激进"
        )

    if l2_wr < l1_wr - 3:
        suggestions_b.append(
            "**L2 代理信号（MA多头 + MACD + 突破）质量不佳**：\n"
            "  - 当前 L2 使用 35 根 K 线的代理指标，建议引入完整缠论（300根）\n"
            "  - 或改为并行打分模式：L1得分 + L2得分 = 综合评分，避免单一条件误杀"
        )

    # 星期分析建议
    by_wd_l1 = stats.get("l1", {}).get("by_weekday_t5", {})
    best_wd = max(by_wd_l1.items(), key=lambda x: x[1].get("win_rate", 0))[0] if by_wd_l1 else ""
    worst_wd = min(by_wd_l1.items(), key=lambda x: x[1].get("win_rate", 0))[0] if by_wd_l1 else ""
    if best_wd and worst_wd and best_wd != worst_wd:
        cn = {"Monday": "周一", "Wednesday": "周三", "Friday": "周五"}
        best_cn  = cn.get(best_wd, best_wd)
        worst_cn = cn.get(worst_wd, worst_wd)
        best_d  = by_wd_l1[best_wd]
        worst_d = by_wd_l1.get(worst_wd, {})
        if abs(best_d.get("win_rate", 0) - worst_d.get("win_rate", 0)) > 5:
            suggestions_b.append(
                f"**星期效应**：{best_cn} 胜率（{best_d.get('win_rate',0):.1f}%）"
                f" > {worst_cn}（{worst_d.get('win_rate',0):.1f}%），差值>5pp。\n"
                "  建议：管线B在不同星期使用差异化阈值，或减少在胜率低的交易日的推荐数量"
            )

    lines.append("### 管线B（量化层）建议\n")
    if suggestions_b:
        for i, s in enumerate(suggestions_b, 1):
            lines.append(f"{i}. {s}\n")
    else:
        lines.append("- 量化层整体表现良好，暂无强烈改进建议\n")

    lines.append("### 管线A（LLM辩论层）建议\n")
    lines.append(
        "以下建议基于量化回测结论推导，需结合 LLM 辩论的定性评估综合判断：\n"
    )
    suggestions_a = [
        f"**LLM幻觉校验（路线图 #6）**：量化回测中 L1 胜率为 {l1_wr:.1f}%，"
        "而 LLM 在没有实时数据的情况下可能编造更高的胜率信心。"
        "建议实现 PE/涨幅偏差校验（偏差>30% 标记存疑）",

        "**A/B管线交叉验证（路线图 #5）**："
        + ("L2 在 L1 基础上有正效果" if d_wr > 0 else "L2 代理信号有待改善")
        + "，建议在链路A中将链路B的L1通过股票作为硬约束（未通过L1的LLM推荐降权）",

        "**市场择时硬约束（路线图 #8）**：回测中"
        + (f"T+5 平均收益 {l1_avg:+.2f}%，" if l1_avg != 0 else "")
        + "建议在大盘处于下行阶段时（MR情绪温度<40）自动减少推荐数量",
    ]
    for i, s in enumerate(suggestions_a, 1):
        lines.append(f"{i}. {s}\n")

    lines.append("\n---")
    lines.append(f"\n*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    report_text = "\n".join(lines)
    BT_REPORT_PATH.write_text(report_text, encoding="utf-8")
    return report_text


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

    parser = argparse.ArgumentParser(description="A股选股系统 — 历史回测（管线B量化层）")
    parser.add_argument("--months",      type=int, default=12,
                        help="回测月数（默认12）")
    parser.add_argument("--report-only", action="store_true",
                        help="仅根据已有结果生成报告，不重新扫描")
    parser.add_argument("--quiet",       action="store_true",
                        help="减少输出")
    args = parser.parse_args()

    if args.report_only:
        if not BT_RESULTS_PATH.exists():
            print("  [错误] 未找到回测结果文件，请先运行回测")
            sys.exit(1)
        results = json.loads(BT_RESULTS_PATH.read_text(encoding="utf-8"))
        print(generate_report(results))
        print(f"\n  报告已保存: {BT_REPORT_PATH}")
    else:
        bt = HistoricalBacktest(months=args.months, verbose=not args.quiet)
        results = bt.run()
        print("\n" + "=" * 65)
        report = generate_report(results)
        print(report)
        print(f"\n  报告已保存: {BT_REPORT_PATH}")
