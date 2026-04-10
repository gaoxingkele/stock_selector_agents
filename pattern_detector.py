#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
形态识别模块 — 迁移自 stock-pattern (BennyThadikaran/stock-pattern)
适配本项目 TDX 日线/周线数据格式，提供统一的形态检测接口。

支持的形态：
  看多（加分）：双底、头肩底、VCP（波动收缩）看多、三角形向上突破、旗型看多
  看空（排除）：双顶、头肩顶、VCP 看空、三角形向下突破、旗型看空

用法：
    from pattern_detector import PatternDetector
    detector = PatternDetector(df_daily, df_weekly=df_weekly)
    result = detector.detect_all()
    # result = {
    #   "bullish_score": 55,     # 看多加分
    #   "bearish_exclude": True, # 是否应排除
    #   "patterns": [...]        # 检测到的形态列表
    # }
"""

import logging
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════

class Point(NamedTuple):
    x: int        # bar index (整数位置)
    y: float      # price


class Line(NamedTuple):
    slope: float
    y_int: float   # y-intercept


# ═══════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════

def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    """将 TDX 格式的 DataFrame 列名统一为 Open/High/Low/Close/Volume"""
    col_map = {}
    for src, dst in [("开盘", "Open"), ("最高", "High"), ("最低", "Low"),
                     ("收盘", "Close"), ("成交量", "Volume"),
                     ("open", "Open"), ("high", "High"), ("low", "Low"),
                     ("close", "Close"), ("volume", "Volume"), ("vol", "Volume")]:
        if src in df.columns and dst not in df.columns:
            col_map[src] = dst
    if col_map:
        df = df.rename(columns=col_map)
    return df


def daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """从日线 DataFrame 合成周线 OHLCV。
    输入必须已标准化（Open/High/Low/Close/Volume），index 为日期。
    """
    if df.empty:
        return df
    # 确保 index 是 datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    weekly = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna(subset=["Open"])
    return weekly


def get_atr(high: pd.Series, low: pd.Series, close: pd.Series,
            window: int = 15) -> pd.Series:
    """Average True Range"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def get_pivots(df: pd.DataFrame, bars_left: int = 6, bars_right: int = 6,
               pivot_type: str = "both") -> pd.DataFrame:
    """
    检测局部高低点（pivot points）。
    返回 DataFrame，列 = [P, V]，P=价格 V=成交量，index 同输入 df。
    """
    window = bars_left + 1 + bars_right
    local_max_idx = []
    local_min_idx = []

    high_arr = df["High"].values
    low_arr = df["Low"].values

    for i in range(bars_left, len(df) - bars_right):
        # 检查是否是窗口内最高点
        window_high = high_arr[i - bars_left: i + bars_right + 1]
        if high_arr[i] == window_high.max() and np.sum(window_high == high_arr[i]) == 1:
            local_max_idx.append(i)

        # 检查是否是窗口内最低点
        window_low = low_arr[i - bars_left: i + bars_right + 1]
        if low_arr[i] == window_low.min() and np.sum(window_low == low_arr[i]) == 1:
            local_min_idx.append(i)

    cols = ["P", "V"]
    maxima = pd.DataFrame(
        {"P": df["High"].iloc[local_max_idx].values,
         "V": df["Volume"].iloc[local_max_idx].values},
        index=df.index[local_max_idx]
    ) if local_max_idx else pd.DataFrame(columns=cols)

    minima = pd.DataFrame(
        {"P": df["Low"].iloc[local_min_idx].values,
         "V": df["Volume"].iloc[local_min_idx].values},
        index=df.index[local_min_idx]
    ) if local_min_idx else pd.DataFrame(columns=cols)

    if pivot_type == "high":
        return maxima
    if pivot_type == "low":
        return minima
    return pd.concat([maxima, minima], axis=0).sort_index()


def _generate_trend_line(series: pd.Series, idx1, idx2) -> Line:
    """基于两个日期点生成趋势线（斜率+截距）。"""
    p1 = float(series[idx1])
    p2 = float(series[idx2])
    d1 = series.index.get_loc(idx1)
    d2 = series.index.get_loc(idx2)
    if isinstance(d1, slice):
        d1 = d1.start
    if isinstance(d2, slice):
        d2 = d2.start
    if d1 == d2:
        return Line(slope=0, y_int=p1)
    m = (p2 - p1) / (d2 - d1)
    b = p1 - m * d1
    return Line(slope=m, y_int=b)


def _get_y(line: Line, x: int) -> float:
    """趋势线上某 x 位置对应的 y 值"""
    return line.slope * x + line.y_int


def _resolve_dup(pivots: pd.DataFrame, idx, col: str = "P",
                 mode: str = "max") -> float:
    """处理 pivot index 重复的情况"""
    val = pivots.at[idx, col]
    if isinstance(val, pd.Series):
        return val.max() if mode == "max" else val.min()
    return float(val)


# ═══════════════════════════════════════════════════════════════════════
# 形态判定函数（纯几何判断）
# ═══════════════════════════════════════════════════════════════════════

# A 股形态容差系数：原始 stock-pattern 默认为 1.0（基于美股低波动）
# A 股波动大、形态不规整，统一放宽到 2.0x
TOLERANCE = 2.0


def _is_hns(a, b, c, d, e, f, avg_bar) -> bool:
    """头肩顶: C 是头（最高），A/E 是肩，B/D 是颈线
    A 股放宽：颈线允许接近肩部（不要求严格低于肩部 0.5×avg_bar）
    """
    shoulder_thresh = round(avg_bar * 0.6, 2)
    neckline_tol = avg_bar * 0.5 * TOLERANCE
    return (c > max(a, e)
            and max(b, d) < min(a, e) + neckline_tol  # 颈线允许略高于肩
            and f < e
            and abs(b - d) < avg_bar * TOLERANCE
            and abs(c - e) > shoulder_thresh)


def _is_reverse_hns(a, b, c, d, e, f, avg_bar) -> bool:
    """头肩底: C 是头（最低），A/E 是肩，B/D 是颈线
    A 股放宽：颈线允许接近肩部（不要求严格高于肩部 0.5×avg_bar）
    """
    shoulder_thresh = round(avg_bar * 0.6, 2)
    neckline_tol = avg_bar * 0.5 * TOLERANCE
    return (c < min(a, e)
            and min(b, d) > max(a, e) - neckline_tol  # 颈线允许略低于肩
            and f > e
            and abs(b - d) < avg_bar * TOLERANCE
            and abs(c - e) > shoulder_thresh)


def _is_double_top(a, b, c, d, a_vol, c_vol, avg_bar, atr) -> bool:
    """双顶"""
    return (c - b < atr * 4
            and abs(a - c) <= avg_bar * 0.5 * TOLERANCE
            and c_vol < a_vol
            and b < min(a, c)
            and b < d < c)


def _is_double_bottom(a, b, c, d, a_vol, c_vol, avg_bar, atr) -> bool:
    """双底"""
    return (b - c < atr * 4
            and abs(a - c) <= avg_bar * 0.5 * TOLERANCE
            and c_vol < a_vol
            and b > max(a, c)
            and b > d > c)


def _is_triangle(a, b, c, d, e, f, avg_bar) -> Optional[str]:
    """三角形: Ascending / Descending / Symmetric
    A 股放宽：单调约束改为整体趋势（首末点差距足够大），允许中间噪音
    """
    ac_flat = abs(a - c) <= avg_bar * TOLERANCE
    ce_flat = abs(c - e) <= avg_bar * TOLERANCE
    bd_flat = abs(b - d) <= avg_bar * TOLERANCE
    noise = avg_bar * 0.5  # 允许的中间噪音容差

    # 上升三角：高点平 + 低点抬升（首末点 b<f 即可，中间允许小回调）
    if ac_flat and ce_flat and b < f - noise and d < f + noise and b < d + noise:
        return "Ascending"

    # 下降三角：低点平 + 高点下移
    if bd_flat and a > f + noise and c > f - noise and a > c - noise and e > f - noise:
        return "Descending"

    # 对称三角：高点下移 + 低点抬升
    if a > e + noise and b < f - noise and a > c - noise and c > e - noise and b < d + noise and d < f + noise:
        return "Symmetric"

    return None


def _is_bullish_vcp(a, b, c, d, e, avg_bar) -> bool:
    """VCP 看多（波动收缩底部抬升）"""
    if c > a and abs(a - c) >= avg_bar * 0.5 * TOLERANCE:
        return False
    return (abs(a - c) <= avg_bar * TOLERANCE
            and abs(b - d) >= avg_bar * 0.8
            and b < min(a, c, d, e)
            and d < min(a, c, e)
            and e < c)


def _is_bearish_vcp(a, b, c, d, e, avg_bar) -> bool:
    """VCP 看空（波动收缩顶部下移）"""
    if c < a and abs(a - c) >= avg_bar * 0.5 * TOLERANCE:
        return False
    return (abs(a - c) <= avg_bar * TOLERANCE
            and abs(b - d) >= avg_bar * 0.8
            and b > max(a, c, d, e)
            and d > max(a, c, e)
            and e > c)


# ═══════════════════════════════════════════════════════════════════════
# 完整形态搜索函数
# ═══════════════════════════════════════════════════════════════════════

def find_double_bottom(df: pd.DataFrame, pivots: pd.DataFrame) -> Optional[dict]:
    """搜索双底形态"""
    if len(pivots) < 3:
        return None

    pivot_len = len(pivots)
    a_idx = pivots["P"].idxmin()
    a = _resolve_dup(pivots, a_idx, mode="min")
    a_vol = _resolve_dup(pivots, a_idx, col="V", mode="min")

    d_idx = df.index[-1]
    d = float(df.at[d_idx, "Close"])
    atr_ser = get_atr(df["High"], df["Low"], df["Close"])
    _prev_a = None

    while True:
        if a_idx == _prev_a:
            break
        _prev_a = a_idx

        pos = pivots.index.get_loc(a_idx)
        if isinstance(pos, slice):
            pos = pos.stop
        next_pos = pos + 1
        if next_pos >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[next_pos]:, "P"].idxmin()
        c = _resolve_dup(pivots, c_idx, mode="min")
        c_vol = _resolve_dup(pivots, c_idx, col="V", mode="min")

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmax()
        b = _resolve_dup(pivots, b_idx, mode="max")

        atr_val = atr_ser.at[c_idx] if c_idx in atr_ser.index else atr_ser.iloc[-1]
        df_slice = df.loc[a_idx:c_idx]
        avg_bar = (df_slice["High"] - df_slice["Low"]).median()

        if _is_double_bottom(a, b, c, d, a_vol, c_vol, avg_bar, atr_val):
            if c_idx != df.loc[c_idx:, "Close"].idxmin():
                a_idx, a, a_vol = c_idx, c, c_vol
                continue
            if df.loc[c_idx:, "Close"].max() > b:
                a_idx, a, a_vol = c_idx, c, c_vol
                continue

            return {"pattern": "DBOT", "direction": "bullish",
                    "points": {"A": (a_idx, a), "B": (b_idx, b),
                               "C": (c_idx, c), "D": (d_idx, d)}}

        a_idx, a, a_vol = c_idx, c, c_vol
    return None


def find_double_top(df: pd.DataFrame, pivots: pd.DataFrame) -> Optional[dict]:
    """搜索双顶形态"""
    if len(pivots) < 3:
        return None

    pivot_len = len(pivots)
    a_idx = pivots["P"].idxmax()
    a = _resolve_dup(pivots, a_idx, mode="max")
    a_vol = _resolve_dup(pivots, a_idx, col="V", mode="max")

    d_idx = df.index[-1]
    d = float(df.at[d_idx, "Close"])
    atr_ser = get_atr(df["High"], df["Low"], df["Close"])
    _prev_a = None

    while True:
        if a_idx == _prev_a:
            break
        _prev_a = a_idx

        pos = pivots.index.get_loc(a_idx)
        if isinstance(pos, slice):
            pos = pos.stop
        next_pos = pos + 1
        if next_pos >= pivot_len:
            break

        c_idx = pivots.loc[pivots.index[next_pos]:, "P"].idxmax()
        c = _resolve_dup(pivots, c_idx, mode="max")
        c_vol = _resolve_dup(pivots, c_idx, col="V", mode="max")

        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmin()
        b = _resolve_dup(pivots, b_idx, mode="min")

        atr_val = atr_ser.at[c_idx] if c_idx in atr_ser.index else atr_ser.iloc[-1]
        df_slice = df.loc[a_idx:c_idx]
        avg_bar = (df_slice["High"] - df_slice["Low"]).median()

        if _is_double_top(a, b, c, d, a_vol, c_vol, avg_bar, atr_val):
            if c_idx != df.loc[c_idx:, "Close"].idxmax():
                a_idx, a, a_vol = c_idx, c, c_vol
                continue

            return {"pattern": "DTOP", "direction": "bearish",
                    "points": {"A": (a_idx, a), "B": (b_idx, b),
                               "C": (c_idx, c), "D": (d_idx, d)}}

        a_idx, a, a_vol = c_idx, c, c_vol
    return None


def find_hns(df: pd.DataFrame, pivots: pd.DataFrame) -> Optional[dict]:
    """搜索头肩顶（看空）"""
    if len(pivots) < 5:
        return None

    pivot_len = len(pivots)
    f_idx = df.index[-1]
    f = float(df.at[f_idx, "Close"])

    c_idx = pivots["P"].idxmax()
    c = _resolve_dup(pivots, c_idx, mode="max")
    _prev_c_idx = None

    while True:
        if c_idx == _prev_c_idx:
            break  # 防止死循环
        _prev_c_idx = c_idx

        pos = pivots.index.get_loc(c_idx)
        if isinstance(pos, slice):
            pos = pos.start
        if pos <= 0:
            break

        a_idx = pivots.loc[:pivots.index[pos - 1], "P"].idxmax()
        a = _resolve_dup(pivots, a_idx, mode="max")
        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmin()
        b = _resolve_dup(pivots, b_idx, mode="min")

        next_pos = pos + 1
        if next_pos >= pivot_len:
            break
        e_idx = pivots.loc[pivots.index[next_pos]:, "P"].idxmax()
        e = _resolve_dup(pivots, e_idx, mode="max")
        d_idx = pivots.loc[c_idx:e_idx, "P"].idxmin()
        d = _resolve_dup(pivots, d_idx, mode="min")

        df_slice = df.loc[b_idx:d_idx]
        avg_bar = (df_slice["High"] - df_slice["Low"]).median()

        if _is_hns(a, b, c, d, e, f, avg_bar):
            if (a == df.at[a_idx, "Low"] or c == df.at[c_idx, "Low"]
                    or e == df.at[e_idx, "Low"]):
                c_idx, c = e_idx, e
                continue

            neckline = min(b, d)
            # 头肩顶允许跌破颈线（跌破是看跌信号），只在远离>3×avg_bar 才认为已完成
            lowest_after_e = df.loc[e_idx:, "Low"].min()
            if lowest_after_e < neckline - avg_bar * 3:
                c_idx, c = e_idx, e
                continue

            return {"pattern": "HNSD", "direction": "bearish",
                    "neckline": neckline,
                    "points": {"A": (a_idx, a), "B": (b_idx, b), "C": (c_idx, c),
                               "D": (d_idx, d), "E": (e_idx, e), "F": (f_idx, f)}}

        c_idx, c = e_idx, e
    return None


def find_reverse_hns(df: pd.DataFrame, pivots: pd.DataFrame) -> Optional[dict]:
    """搜索头肩底（看多）"""
    if len(pivots) < 5:
        return None

    pivot_len = len(pivots)
    f_idx = df.index[-1]
    f = float(df.at[f_idx, "Close"])

    c_idx = pivots["P"].idxmin()
    c = _resolve_dup(pivots, c_idx, mode="min")
    _prev_c_idx = None

    while True:
        if c_idx == _prev_c_idx:
            break
        _prev_c_idx = c_idx

        pos = pivots.index.get_loc(c_idx)
        if isinstance(pos, slice):
            pos = pos.start
        if pos <= 0:
            break

        a_idx = pivots.loc[:pivots.index[pos - 1], "P"].idxmin()
        a = _resolve_dup(pivots, a_idx, mode="min")
        b_idx = pivots.loc[a_idx:c_idx, "P"].idxmax()
        b = _resolve_dup(pivots, b_idx, mode="max")

        next_pos = pos + 1
        if next_pos >= pivot_len:
            break
        e_idx = pivots.loc[pivots.index[next_pos]:, "P"].idxmin()
        e = _resolve_dup(pivots, e_idx, mode="min")
        d_idx = pivots.loc[c_idx:e_idx, "P"].idxmax()
        d = _resolve_dup(pivots, d_idx, mode="max")

        df_slice = df.loc[b_idx:d_idx]
        avg_bar = (df_slice["High"] - df_slice["Low"]).median()

        if _is_reverse_hns(a, b, c, d, e, f, avg_bar):
            if (a == df.at[a_idx, "High"] or c == df.at[c_idx, "High"]
                    or e == df.at[e_idx, "High"]):
                c_idx, c = e_idx, e
                continue

            neckline = max(b, d)
            # 头肩底允许突破颈线（突破是看涨信号），但要求 close 在颈线附近不能远离
            # 只在突破后又远走（>3×avg_bar）才认为形态已完成不再有效
            highest_after_e = df.loc[e_idx:, "High"].max()
            if highest_after_e > neckline + avg_bar * 3:
                c_idx, c = e_idx, e
                continue

            return {"pattern": "HNSU", "direction": "bullish",
                    "neckline": neckline,
                    "points": {"A": (a_idx, a), "B": (b_idx, b), "C": (c_idx, c),
                               "D": (d_idx, d), "E": (e_idx, e), "F": (f_idx, f)}}

        c_idx, c = e_idx, e
    return None


def find_triangles(df: pd.DataFrame, pivots: pd.DataFrame) -> Optional[dict]:
    """搜索三角形整理（对称/上升/下降），返回含突破方向"""
    if len(pivots) < 5:
        return None

    pivot_len = len(pivots)
    a_idx = pivots["P"].idxmax()
    a = _resolve_dup(pivots, a_idx, mode="max")
    f_idx = df.index[-1]
    f = float(df.at[f_idx, "Close"])
    _prev_a = None

    while True:
        if a_idx == _prev_a:
            break
        _prev_a = a_idx

        b_idx = pivots.loc[a_idx:, "P"].idxmin()
        b = _resolve_dup(pivots, b_idx, mode="min")

        if a_idx == b_idx:
            pos = pivots.index.get_loc(a_idx)
            if isinstance(pos, slice):
                pos = pos.stop
            next_pos = pos + 1
            if next_pos >= pivot_len:
                break
            a_idx = pivots.index[next_pos]
            a = _resolve_dup(pivots, a_idx, mode="max")
            continue

        # 找 C, D, E
        pos_b = pivots.index.get_loc(b_idx)
        if isinstance(pos_b, slice):
            pos_b = pos_b.stop
        if pos_b + 1 >= pivot_len:
            break
        c_idx = pivots.loc[pivots.index[pos_b + 1]:, "P"].idxmax()
        c = _resolve_dup(pivots, c_idx, mode="max")

        pos_c = pivots.index.get_loc(c_idx)
        if isinstance(pos_c, slice):
            pos_c = pos_c.stop
        if pos_c + 1 >= pivot_len:
            break
        d_idx = pivots.loc[pivots.index[pos_c + 1]:, "P"].idxmin()
        d = _resolve_dup(pivots, d_idx, mode="min")

        pos_d = pivots.index.get_loc(d_idx)
        if isinstance(pos_d, slice):
            pos_d = pos_d.stop
        if pos_d + 1 >= pivot_len:
            break
        e_idx = pivots.loc[pivots.index[pos_d + 1]:, "P"].idxmax()
        e = _resolve_dup(pivots, e_idx, mode="max")

        df_slice = df.loc[a_idx:d_idx]
        avg_bar = (df_slice["High"] - df_slice["Low"]).median()

        triangle_type = _is_triangle(a, b, c, d, e, f, avg_bar)

        if triangle_type is not None:
            upper = _generate_trend_line(df["High"], a_idx, c_idx)
            lower = _generate_trend_line(df["Low"], b_idx, d_idx)

            # 判断突破方向
            f_pos = df.index.get_loc(f_idx)
            if isinstance(f_pos, slice):
                f_pos = f_pos.stop
            upper_at_f = _get_y(upper, f_pos)
            lower_at_f = _get_y(lower, f_pos)

            if f > upper_at_f:
                breakout = "up"
            elif f < lower_at_f:
                breakout = "down"
            else:
                breakout = "inside"  # 尚未突破

            direction = "bullish" if breakout == "up" else ("bearish" if breakout == "down" else "neutral")

            return {"pattern": f"TRNG_{triangle_type}",
                    "direction": direction,
                    "breakout": breakout,
                    "triangle_type": triangle_type,
                    "slope_upper": upper.slope,
                    "slope_lower": lower.slope,
                    "points": {"A": (a_idx, a), "B": (b_idx, b), "C": (c_idx, c),
                               "D": (d_idx, d), "E": (e_idx, e), "F": (f_idx, f)}}

        # 继续搜索
        a_idx, a = c_idx, c
    return None


def find_bullish_vcp(df: pd.DataFrame, pivots: pd.DataFrame) -> Optional[dict]:
    """搜索 VCP 看多（波动收缩，底部抬升）"""
    if len(pivots) < 4:
        return None

    pivot_len = len(pivots)
    a_idx = pivots["P"].idxmax()
    a = _resolve_dup(pivots, a_idx, mode="max")
    e_idx = df.index[-1]
    e = float(df.at[e_idx, "Close"])
    _prev_a = None

    while True:
        if a_idx == _prev_a:
            break
        _prev_a = a_idx

        pos = pivots.index.get_loc(a_idx)
        if isinstance(pos, slice):
            pos = pos.stop
        if pos + 1 >= pivot_len:
            break

        b_idx = pivots.loc[pivots.index[pos + 1]:, "P"].idxmin()
        b = _resolve_dup(pivots, b_idx, mode="min")

        pos_b = pivots.index.get_loc(b_idx)
        if isinstance(pos_b, slice):
            pos_b = pos_b.stop
        if pos_b + 1 >= pivot_len:
            break

        d_idx = pivots.loc[pivots.index[pos_b + 1]:, "P"].idxmin()
        d = _resolve_dup(pivots, d_idx, mode="min")
        c_idx = pivots.loc[b_idx:d_idx, "P"].idxmax()
        c = _resolve_dup(pivots, c_idx, mode="max")

        df_slice = df.loc[a_idx:c_idx]
        avg_bar = (df_slice["High"] - df_slice["Low"]).median()

        if _is_bullish_vcp(a, b, c, d, e, avg_bar):
            if (c_idx != df.loc[c_idx:, "Close"].idxmax()
                    or d_idx != df.loc[d_idx:, "Close"].idxmin()):
                if pivots.index[-1] == c_idx or pivots.index[-1] == d_idx:
                    break
                a_idx, a = c_idx, c
                continue

            return {"pattern": "VCPU", "direction": "bullish",
                    "points": {"A": (a_idx, a), "B": (b_idx, b),
                               "C": (c_idx, c), "D": (d_idx, d), "E": (e_idx, e)}}

        a_idx, a = c_idx, c
    return None


def find_bearish_vcp(df: pd.DataFrame, pivots: pd.DataFrame) -> Optional[dict]:
    """搜索 VCP 看空（波动收缩，顶部下移）"""
    if len(pivots) < 4:
        return None

    pivot_len = len(pivots)
    a_idx = pivots["P"].idxmin()
    a = _resolve_dup(pivots, a_idx, mode="min")
    e_idx = df.index[-1]
    e = float(df.at[e_idx, "Close"])
    _prev_a = None

    while True:
        if a_idx == _prev_a:
            break
        _prev_a = a_idx

        pos = pivots.index.get_loc(a_idx)
        if isinstance(pos, slice):
            pos = pos.stop
        if pos + 1 >= pivot_len:
            break

        b_idx = pivots.loc[pivots.index[pos + 1]:, "P"].idxmax()
        b = _resolve_dup(pivots, b_idx, mode="max")

        pos_b = pivots.index.get_loc(b_idx)
        if isinstance(pos_b, slice):
            pos_b = pos_b.stop
        if pos_b + 1 >= pivot_len:
            break

        d_idx = pivots.loc[pivots.index[pos_b + 1]:, "P"].idxmax()
        d = _resolve_dup(pivots, d_idx, mode="max")
        c_idx = pivots.loc[b_idx:d_idx, "P"].idxmin()
        c = _resolve_dup(pivots, c_idx, mode="min")

        df_slice = df.loc[a_idx:c_idx]
        avg_bar = (df_slice["High"] - df_slice["Low"]).median()

        if _is_bearish_vcp(a, b, c, d, e, avg_bar):
            if (d_idx != df.loc[d_idx:, "Close"].idxmax()
                    or c_idx != df.loc[c_idx:, "Close"].idxmin()):
                if pivots.index[-1] == d_idx or pivots.index[-1] == c_idx:
                    break
                a_idx, a = c_idx, c
                continue

            return {"pattern": "VCPD", "direction": "bearish",
                    "points": {"A": (a_idx, a), "B": (b_idx, b),
                               "C": (c_idx, c), "D": (d_idx, d), "E": (e_idx, e)}}

        a_idx, a = c_idx, c
    return None


def find_bullish_flag(df: pd.DataFrame, pivots: pd.DataFrame) -> Optional[dict]:
    """搜索看多旗型（高杆+旗面整理）"""
    if len(df) < 50 or len(pivots) < 2:
        return None

    last_idx = df.index[-1]
    recent_high_idx = df["High"].iloc[-7:].idxmax()

    if recent_high_idx == last_idx:
        return None

    monthly_high = df["High"].iloc[-30:].max()
    three_month_high = df["High"].iloc[-90:].max() if len(df) >= 90 else monthly_high

    recent_high = df.at[recent_high_idx, "High"]
    recent_low = df.loc[recent_high_idx:, "Low"].min()

    if recent_high >= monthly_high and recent_high >= three_month_high:
        sma20 = df["Close"].rolling(20).mean().iloc[-1]
        sma50 = df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma20

        last_pivot_idx = pivots.index[-1]
        last_pivot = _resolve_dup(pivots, last_pivot_idx, mode="max")
        fib_50 = last_pivot + (recent_high - last_pivot) / 2

        if sma20 < sma50 * 1.08 or recent_low < fib_50:
            return None

        return {"pattern": "FLAGU", "direction": "bullish",
                "points": {"A": (last_pivot_idx, last_pivot),
                           "B": (recent_high_idx, recent_high)}}
    return None


def find_bearish_flag(df: pd.DataFrame, pivots: pd.DataFrame) -> Optional[dict]:
    """搜索看空旗型（低杆+旗面反弹）"""
    if len(df) < 50 or len(pivots) < 2:
        return None

    last_idx = df.index[-1]
    recent_low_idx = df["Low"].iloc[-7:].idxmin()

    if recent_low_idx == last_idx:
        return None

    monthly_low = df["Low"].iloc[-30:].min()
    three_month_low = df["Low"].iloc[-90:].min() if len(df) >= 90 else monthly_low

    recent_low = df.at[recent_low_idx, "Low"]
    recent_high = df.loc[recent_low_idx:, "High"].max()

    if recent_low <= monthly_low and recent_low <= three_month_low:
        sma20 = df["Close"].rolling(20).mean().iloc[-1]
        sma50 = df["Close"].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma20

        last_pivot_idx = pivots.index[-1]
        last_pivot = _resolve_dup(pivots, last_pivot_idx, mode="min")
        fib_50 = last_pivot - (last_pivot - recent_low) / 2

        if sma20 > sma50 * 0.92 or recent_high > fib_50:
            return None

        return {"pattern": "FLAGD", "direction": "bearish",
                "points": {"A": (last_pivot_idx, last_pivot),
                           "B": (recent_low_idx, recent_low)}}
    return None


# ═══════════════════════════════════════════════════════════════════════
# MACD 顶背离检测（TA-Lib 辅助）
# ═══════════════════════════════════════════════════════════════════════

def detect_macd_bearish_divergence(df: pd.DataFrame, pivots: pd.DataFrame) -> bool:
    """
    检测 MACD 零轴下方顶背离：
    价格创新高（或接近前高），但 MACD 柱/DIF 线走低。
    只在 DIF < 0 时检测（零轴下方）。
    """
    try:
        import talib
        close = df["Close"].values.astype(float)
        dif, dea, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    except ImportError:
        # fallback: 手算 MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        dif = (ema12 - ema26).values
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        macd_hist = (dif - dea) * 2

    if len(dif) < 60:
        return False

    # 只看最近120根K线内的 pivot high
    high_pivots = pivots[pivots["P"] >= df["High"].quantile(0.7)]
    if len(high_pivots) < 2:
        return False

    # 取最近两个高点
    recent_highs = high_pivots.tail(2)
    idx0, idx1 = recent_highs.index[0], recent_highs.index[1]

    pos0 = df.index.get_loc(idx0)
    pos1 = df.index.get_loc(idx1)
    if isinstance(pos0, slice):
        pos0 = pos0.start
    if isinstance(pos1, slice):
        pos1 = pos1.start

    price0, price1 = recent_highs.iloc[0]["P"], recent_highs.iloc[1]["P"]

    # DIF 必须在零轴下方（至少有一个）
    dif0 = dif[pos0] if pos0 < len(dif) else 0
    dif1 = dif[pos1] if pos1 < len(dif) else 0

    if dif0 >= 0 and dif1 >= 0:
        return False

    # 价格持平或走高，但 DIF 走低 → 顶背离
    if price1 >= price0 * 0.97 and dif1 < dif0:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════
# 破前低检测
# ═══════════════════════════════════════════════════════════════════════

def detect_break_below_support(df: pd.DataFrame, pivots: pd.DataFrame,
                                lookback_days: int = 5) -> bool:
    """
    检测5日内是否跌破前期有量支撑的低点（放量跌破）。
    前低：pivot low 中成交量较大的点。
    跌破确认：收盘价低于前低 + 当日成交量 > 20日均量。
    """
    low_pivots = pivots[pivots["P"] <= df["Low"].quantile(0.3)]
    if len(low_pivots) == 0:
        return False

    # 找有量支撑的前低（成交量 > 中位数）
    vol_median = df["Volume"].median()
    support_pivots = low_pivots[low_pivots["V"] > vol_median]
    if len(support_pivots) == 0:
        return False

    # 取最近的支撑低点
    support_price = support_pivots.iloc[-1]["P"]
    support_idx = support_pivots.index[-1]

    # 检查最近 lookback_days 是否跌破
    recent = df.tail(lookback_days)
    avg_vol_20 = df["Volume"].tail(20).mean()

    for idx, row in recent.iterrows():
        if idx <= support_idx:
            continue
        if row["Close"] < support_price and row["Volume"] > avg_vol_20:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════
# 均线排列检测
# ═══════════════════════════════════════════════════════════════════════

def detect_ma_bearish_alignment(df: pd.DataFrame) -> bool:
    """MA5 < MA20 < MA60 空头排列"""
    if len(df) < 60:
        return False
    close = df["Close"]
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    return ma5 < ma20 < ma60


def detect_ma_bullish_alignment(df: pd.DataFrame) -> bool:
    """MA5 > MA20 > MA60 多头排列"""
    if len(df) < 60:
        return False
    close = df["Close"]
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    return ma5 > ma20 > ma60


# ═══════════════════════════════════════════════════════════════════════
# PatternDetector 主类
# ═══════════════════════════════════════════════════════════════════════

class PatternDetector:
    """
    统一形态检测器，接受日线和（可选）周线数据，返回综合评分和排除标志。

    用法：
        detector = PatternDetector(df_daily, df_weekly=df_weekly)
        result = detector.detect_all()
    """

    # 加分权重（看多）
    BULL_SCORES = {
        "DBOT":  {"daily": 15, "weekly": 25},   # 双底
        "HNSU":  {"daily": 15, "weekly": 25},   # 头肩底
        "VCPU":  {"daily": 10, "weekly": 15},   # VCP 看多
        "TRNG_Ascending":  {"daily": 10, "weekly": 20},  # 上升三角突破
        "TRNG_Symmetric":  {"daily": 8,  "weekly": 15},  # 对称三角向上突破
        "FLAGU": {"daily": 10, "weekly": 15},   # 看多旗型
        "MA_BULL": {"daily": 10, "weekly": 0},  # 多头排列
    }

    # 看空强度分（按形态严重程度）
    # 决策：当 bear_score - bull_score > BEAR_VETO_THRESHOLD 时排除
    BEAR_SCORES = {
        "HNSD":          {"daily": 20, "weekly": 35},  # 头肩顶
        "DTOP":          {"daily": 15, "weekly": 25},  # 双顶
        "VCPD":          {"daily": 10, "weekly": 15},  # VCP 看空
        "TRNG_breakdown":{"daily": 10, "weekly": 15},  # 三角向下突破
        "FLAGD":         {"daily": 10, "weekly": 15},  # 看空旗型
        "BREAK_SUPPORT": {"daily": 15, "weekly": 0},   # 破前低放量（强信号）
        "MACD_DIV":      {"daily": 8,  "weekly": 12},  # MACD 顶背离（弱单独信号）
        "MA_BEAR":       {"daily": 5,  "weekly": 10},  # 均线空头（弱单独信号）
    }

    BEAR_VETO_THRESHOLD = 25  # 看空净分超过该阈值才一票否决（25 = 中等强度看空形态）

    def __init__(self, df_daily: pd.DataFrame,
                 df_weekly: pd.DataFrame = None,
                 pivot_bars_left: int = 5,
                 pivot_bars_right: int = 5,
                 weekly_pivot_bars_left: int = 2,
                 weekly_pivot_bars_right: int = 2):
        self.df_daily = _standardize_df(df_daily.copy())
        self.df_weekly = _standardize_df(df_weekly.copy()) if df_weekly is not None else None
        self.pivot_bl = pivot_bars_left
        self.pivot_br = pivot_bars_right
        self.weekly_bl = weekly_pivot_bars_left
        self.weekly_br = weekly_pivot_bars_right

    def _detect_on_timeframe(self, df: pd.DataFrame, tf: str,
                              bl: int, br: int) -> Tuple[List[dict], List[str]]:
        """在单个时间周期上运行所有形态检测。
        返回 (bullish_patterns, bearish_tags)
        """
        bullish = []
        bearish_tags = []

        if len(df) < 30:
            return bullish, bearish_tags

        pivots = get_pivots(df, bars_left=bl, bars_right=br)
        if len(pivots) < 3:
            return bullish, bearish_tags

        # ── 看多形态 ──
        for name, fn in [("DBOT", find_double_bottom),
                         ("HNSU", find_reverse_hns),
                         ("VCPU", find_bullish_vcp),
                         ("FLAGU", find_bullish_flag)]:
            try:
                result = fn(df, pivots)
                if result:
                    result["timeframe"] = tf
                    bullish.append(result)
            except Exception as e:
                logger.debug(f"[PatternDetector] {name} {tf} error: {e}")

        # 三角形
        try:
            tri = find_triangles(df, pivots)
            if tri:
                tri["timeframe"] = tf
                if tri["breakout"] == "up":
                    bullish.append(tri)
                elif tri["breakout"] == "down":
                    bearish_tags.append(f"TRNG_breakdown")
                # inside 不计分
        except Exception as e:
            logger.debug(f"[PatternDetector] TRNG {tf} error: {e}")

        # ── 看空形态 ──
        for name, fn in [("DTOP", find_double_top),
                         ("HNSD", find_hns),
                         ("VCPD", find_bearish_vcp),
                         ("FLAGD", find_bearish_flag)]:
            try:
                result = fn(df, pivots)
                if result:
                    result["timeframe"] = tf
                    bearish_tags.append(result["pattern"])
            except Exception as e:
                logger.debug(f"[PatternDetector] {name} {tf} error: {e}")

        # ── MACD 顶背离 ──
        try:
            if detect_macd_bearish_divergence(df, pivots):
                bearish_tags.append("MACD_DIV")
        except Exception as e:
            logger.debug(f"[PatternDetector] MACD_DIV {tf} error: {e}")

        # ── 破前低 ──
        try:
            if detect_break_below_support(df, pivots):
                bearish_tags.append("BREAK_SUPPORT")
        except Exception as e:
            logger.debug(f"[PatternDetector] BREAK_SUPPORT {tf} error: {e}")

        # ── 均线排列 ──
        if detect_ma_bearish_alignment(df):
            bearish_tags.append("MA_BEAR")
        if detect_ma_bullish_alignment(df):
            bullish.append({"pattern": "MA_BULL", "direction": "bullish", "timeframe": tf})

        return bullish, bearish_tags

    def detect_all(self) -> dict:
        """
        运行完整检测，返回：
        {
            "bullish_score": int,         # 看多总加分
            "bearish_score": int,         # 看空总扣分
            "bearish_exclude": bool,      # 是否应排除
            "bearish_tags": [str, ...],   # 触发的看空标签
            "patterns": [dict, ...],      # 所有检测到的形态详情
        }

        新版判定逻辑（看多 vs 看空强度对比）：
          1. 计算 bull_score 和 bear_score（区分日线/周线权重）
          2. 周线 HNSD/DTOP 直接排除（最强看空信号）
          3. bear_net = bear_score - bull_score
             - bear_net > BEAR_VETO_THRESHOLD → 排除
             - 否则保留，bullish_score 反映净加分
        """
        all_patterns = []
        all_bearish_tags = []
        bull_score = 0
        bear_score = 0

        # ── 日线检测 ──
        bull_d, bear_d = self._detect_on_timeframe(
            self.df_daily, "daily", self.pivot_bl, self.pivot_br)
        all_patterns.extend(bull_d)
        all_bearish_tags.extend([(t, "daily") for t in bear_d])

        for p in bull_d:
            ptn = p["pattern"]
            if ptn in self.BULL_SCORES:
                bull_score += self.BULL_SCORES[ptn]["daily"]
        for tag in bear_d:
            if tag in self.BEAR_SCORES:
                bear_score += self.BEAR_SCORES[tag]["daily"]

        # ── 周线检测 ──
        if self.df_weekly is not None and len(self.df_weekly) >= 20:
            bull_w, bear_w = self._detect_on_timeframe(
                self.df_weekly, "weekly", self.weekly_bl, self.weekly_br)
            all_patterns.extend(bull_w)
            all_bearish_tags.extend([(t, "weekly") for t in bear_w])

            for p in bull_w:
                ptn = p["pattern"]
                if ptn in self.BULL_SCORES:
                    bull_score += self.BULL_SCORES[ptn]["weekly"]
            for tag in bear_w:
                if tag in self.BEAR_SCORES:
                    bear_score += self.BEAR_SCORES[tag]["weekly"]

        # ── 排除判定 ──
        # 1) 周线 HNSD/DTOP 一票否决（最强看空形态）
        weekly_strong_bear = [t for t, tf in all_bearish_tags
                              if tf == "weekly" and t in ("HNSD", "DTOP")]
        if weekly_strong_bear:
            should_exclude = True
        else:
            # 2) 看空净分超阈值才排除
            bear_net = bear_score - bull_score
            should_exclude = bear_net > self.BEAR_VETO_THRESHOLD

        return {
            "bullish_score": bull_score,
            "bearish_score": bear_score,
            "bearish_exclude": should_exclude,
            "bearish_tags": list(set(t for t, _ in all_bearish_tags)),
            "patterns": all_patterns,
        }
