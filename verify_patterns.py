#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
形态识别可视化验证工具

用法:
    python verify_patterns.py 002460                  # 单只，最新日期
    python verify_patterns.py 002460 20260331         # 单只，指定 cutoff
    python verify_patterns.py --batch                 # 批量验证常用候选

为每只股票生成 PNG 图，标注：
  - K线 + MA20/MA60
  - pivot 高低点（红/绿三角）
  - 检出的形态点（带标号 ABCDEF）
  - 颈线/趋势线
  - 形态名称 + 评分

输出: output/pattern_verification/{code}_{cutoff}.png
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# Windows 中文字符
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output" / "pattern_verification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_stock_data(code: str, cutoff_date: Optional[str] = None,
                    days: int = 500) -> Optional[pd.DataFrame]:
    """加载股票日线数据"""
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    from data_engine import DataEngine
    de = DataEngine()
    df = de._tdx_daily_only(code, days=days, cutoff_date=cutoff_date)
    if df is None or len(df) < 60:
        return None
    return df.tail(days).copy()


def visualize_patterns(code: str, cutoff_date: Optional[str] = None,
                        timeframe: str = "daily") -> Optional[str]:
    """
    为单只股票生成形态验证图。
    timeframe: "daily" 或 "weekly"
    返回输出文件路径。
    """
    from pattern_detector import (
        PatternDetector, _standardize_df, daily_to_weekly,
        get_pivots, find_double_bottom, find_double_top, find_hns,
        find_reverse_hns, find_triangles, find_bullish_vcp, find_bearish_vcp,
        find_bullish_flag, find_bearish_flag,
    )

    df_raw = load_stock_data(code, cutoff_date, days=500)
    if df_raw is None:
        print(f"  [!] {code}: 无数据")
        return None

    std_df = _standardize_df(df_raw.copy())
    if not isinstance(std_df.index, pd.DatetimeIndex):
        if "date" in std_df.columns:
            std_df.index = pd.to_datetime(std_df["date"])
        else:
            std_df.index = pd.to_datetime(std_df.index)

    if timeframe == "weekly":
        df = daily_to_weekly(std_df)
        bl, br = 2, 2
    else:
        df = std_df
        bl, br = 5, 5

    if len(df) < 30:
        print(f"  [!] {code} {timeframe}: 数据不足 ({len(df)})")
        return None

    pivots = get_pivots(df, bars_left=bl, bars_right=br)

    # 跑所有形态检测
    detected = []
    fn_list = [
        ("DBOT", find_double_bottom),
        ("DTOP", find_double_top),
        ("HNSU", find_reverse_hns),
        ("HNSD", find_hns),
        ("VCPU", find_bullish_vcp),
        ("VCPD", find_bearish_vcp),
        ("TRNG", find_triangles),
        ("FLAGU", find_bullish_flag),
        ("FLAGD", find_bearish_flag),
    ]
    for name, fn in fn_list:
        try:
            r = fn(df, pivots)
            if r:
                r["short_name"] = name
                r["timeframe"] = timeframe
                detected.append(r)
        except Exception as e:
            print(f"  [!] {code} {name}: {e}")

    # 当主周期是日线时，额外跑周线检测，叠加显示在日线图上
    if timeframe == "daily" and len(std_df) >= 100:
        weekly_df = daily_to_weekly(std_df)
        if len(weekly_df) >= 30:
            weekly_pivots = get_pivots(weekly_df, bars_left=2, bars_right=2)
            for name, fn in fn_list:
                try:
                    r = fn(weekly_df, weekly_pivots)
                    if r:
                        r["short_name"] = name + "(W)"
                        r["timeframe"] = "weekly"
                        detected.append(r)
                except Exception as e:
                    print(f"  [!] {code} weekly {name}: {e}")

    # 跑完整 detect_all 拿评分
    weekly_for_full = daily_to_weekly(std_df) if timeframe == "daily" and len(std_df) >= 100 else None
    detector = PatternDetector(
        std_df, df_weekly=weekly_for_full,
        pivot_bars_left=5, pivot_bars_right=5,
        weekly_pivot_bars_left=2, weekly_pivot_bars_right=2,
    )
    full_result = detector.detect_all()

    # 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # 中文字体
    for fp in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]:
        if os.path.exists(fp):
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    # 多子图布局：顶部全局图 + 每个形态一个子图
    n_patterns = len(detected)
    if n_patterns > 0:
        n_zoom_rows = (n_patterns + 1) // 2  # 每行2个形态
        total_rows = 1 + n_zoom_rows
        fig = plt.figure(figsize=(20, 6 + 5 * n_zoom_rows))
        gs = fig.add_gridspec(total_rows, 2, height_ratios=[1.5] + [1] * n_zoom_rows, hspace=0.4, wspace=0.2)
        ax_full = fig.add_subplot(gs[0, :])  # 顶部跨两列
        zoom_axes = []
        for i in range(n_patterns):
            row = 1 + i // 2
            col = i % 2
            zoom_axes.append(fig.add_subplot(gs[row, col]))
    else:
        fig, ax_full = plt.subplots(figsize=(20, 8))
        zoom_axes = []

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    dates = df.index

    # ── 全局视图 ──
    ax_full.fill_between(dates, lows, highs, alpha=0.15, color="gray")
    ax_full.plot(dates, closes, color="black", linewidth=1.0, label="Close")

    ma20 = pd.Series(closes).rolling(20).mean().values
    ma60 = pd.Series(closes).rolling(60).mean().values if len(closes) >= 60 else None
    ax_full.plot(dates, ma20, color="orange", linewidth=1.2, alpha=0.7, label="MA20")
    if ma60 is not None:
        ax_full.plot(dates, ma60, color="purple", linewidth=1.2, alpha=0.7, label="MA60")

    # 普通 pivot 点用淡色小点（不抢戏）
    for idx in pivots.index:
        if idx not in df.index:
            continue
        p_val = pivots.at[idx, "P"]
        if isinstance(p_val, pd.Series):
            p_val = p_val.iloc[0]
        row_h = float(df.at[idx, "High"])
        row_l = float(df.at[idx, "Low"])
        if abs(p_val - row_h) < abs(p_val - row_l):
            ax_full.scatter([idx], [p_val], marker="v", color="lightcoral", s=20, alpha=0.5, zorder=3)
        else:
            ax_full.scatter([idx], [p_val], marker="^", color="lightgreen", s=20, alpha=0.5, zorder=3)

    # 形态点（突出标注）
    color_pool = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    pattern_legends = []
    for i, ptn in enumerate(detected):
        color = color_pool[i % len(color_pool)]
        points = ptn.get("points", {})
        ptn_name = ptn["short_name"]
        if ptn.get("alt_name"):
            ptn_name += f"({ptn['alt_name']})"
        elif ptn.get("triangle_type"):
            ptn_name += f"({ptn['triangle_type']})"

        # 形态点连线
        sorted_pts = sorted(points.items(), key=lambda kv: kv[1][0])
        xs = [p[1][0] for p in sorted_pts]
        ys = [p[1][1] for p in sorted_pts]
        ax_full.plot(xs, ys, color=color, linewidth=2, alpha=0.8, zorder=7)

        for label, (idx, price) in points.items():
            if idx in df.index or idx in dates:
                ax_full.scatter([idx], [price], marker="o", color=color, s=180,
                                edgecolors="black", linewidths=2, zorder=8)
                ax_full.annotate(label, (idx, price),
                                 xytext=(10, 10), textcoords="offset points",
                                 fontsize=14, fontweight="bold", color=color,
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8))

        # 颈线
        if "neckline" in ptn:
            ax_full.axhline(ptn["neckline"], color=color, linestyle="--", linewidth=2, alpha=0.7)
            pattern_legends.append(f'{ptn_name} 颈线={ptn["neckline"]:.2f}')
        else:
            pattern_legends.append(f'{ptn_name}')

    # ── 每个形态一个独立子图 ──
    from datetime import timedelta
    for i, ptn in enumerate(detected):
        if i >= len(zoom_axes):
            break
        ax_z = zoom_axes[i]
        color = color_pool[i % len(color_pool)]
        points = ptn.get("points", {})
        if not points:
            continue

        ptn_dates = [idx for _, (idx, _) in points.items()]
        min_d = min(ptn_dates)
        max_d = max(ptn_dates)
        d_range = (max_d - min_d).days
        pad = max(d_range * 0.4, 30)
        zoom_start = min_d - timedelta(days=pad)
        zoom_end = max_d + timedelta(days=pad)

        mask = (dates >= zoom_start) & (dates <= zoom_end)
        zoom_dates = dates[mask]
        zoom_closes = closes[mask]
        zoom_highs = highs[mask]
        zoom_lows = lows[mask]

        if len(zoom_dates) == 0:
            continue

        ax_z.fill_between(zoom_dates, zoom_lows, zoom_highs, alpha=0.2, color="gray")
        ax_z.plot(zoom_dates, zoom_closes, color="black", linewidth=1.5)

        # 局部 pivot
        for idx in pivots.index:
            if idx < zoom_start or idx > zoom_end or idx not in df.index:
                continue
            p_val = pivots.at[idx, "P"]
            if isinstance(p_val, pd.Series):
                p_val = p_val.iloc[0]
            row_h = float(df.at[idx, "High"])
            row_l = float(df.at[idx, "Low"])
            if abs(p_val - row_h) < abs(p_val - row_l):
                ax_z.scatter([idx], [p_val], marker="v", color="lightcoral", s=40, alpha=0.6)
            else:
                ax_z.scatter([idx], [p_val], marker="^", color="lightgreen", s=40, alpha=0.6)

        # 形态点
        sorted_pts = sorted(points.items(), key=lambda kv: kv[1][0])
        xs = [p[1][0] for p in sorted_pts]
        ys = [p[1][1] for p in sorted_pts]
        ax_z.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9, zorder=7)
        for label, (idx, price) in points.items():
            ax_z.scatter([idx], [price], marker="o", color=color, s=220,
                         edgecolors="black", linewidths=2, zorder=8)
            ax_z.annotate(label, (idx, price),
                          xytext=(12, 12), textcoords="offset points",
                          fontsize=14, fontweight="bold", color=color,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.9))
        if "neckline" in ptn:
            ax_z.axhline(ptn["neckline"], color=color, linestyle="--", linewidth=2, alpha=0.7,
                          label=f'颈线={ptn["neckline"]:.2f}')
            ax_z.legend(loc="upper left", fontsize=9)

        ptn_title = ptn["short_name"]
        if ptn.get("alt_name"):
            ptn_title += f" ({ptn['alt_name']})"
        elif ptn.get("triangle_type"):
            ptn_title += f" ({ptn['triangle_type']})"
        ptn_title += f" — {ptn.get('direction','?')}"
        ax_z.set_title(ptn_title, fontsize=12, color=color, fontweight="bold")

        ax_z.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_z.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.setp(ax_z.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
        ax_z.grid(True, alpha=0.3)

    # 标题（全局图）
    pname = ", ".join(p["pattern"] for p in full_result.get("patterns", [])) or "无形态"
    title = (f"{code} | {timeframe} | "
             f"看多={full_result['bullish_score']} 看空={full_result['bearish_score']} | "
             f"{'排除' if full_result['bearish_exclude'] else '保留'} | "
             f"形态={pname}")
    if full_result.get("bearish_tags"):
        title += f"\n看空标签: {', '.join(full_result['bearish_tags'])}"
    ax_full.set_title(title, fontsize=14)

    ax_full.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_full.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_full.xaxis.get_majorticklabels(), rotation=30)

    handles, labels = ax_full.get_legend_handles_labels()
    legend_lines = labels + pattern_legends
    legend_handles = handles + [plt.Line2D([], [], color=color_pool[i % len(color_pool)], linewidth=2)
                                 for i in range(len(pattern_legends))]
    ax_full.legend(legend_handles, legend_lines, loc="upper left", fontsize=10, ncol=2)
    ax_full.grid(True, alpha=0.3)
    ax_full.set_xlabel("日期")
    ax_full.set_ylabel("价格")

    # 保存
    suffix = f"_{cutoff_date}" if cutoff_date else ""
    out_path = OUTPUT_DIR / f"{code}{suffix}_{timeframe}.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()

    print(f"  [OK] {code} {timeframe} → {out_path.name}")
    print(f"       形态: {pname}")
    print(f"       看多分={full_result['bullish_score']} 看空分={full_result['bearish_score']} "
          f"{'排除' if full_result['bearish_exclude'] else '保留'}")
    return str(out_path)


def batch_verify():
    """批量验证常用形态候选股"""
    candidates = [
        # (code, cutoff_date, 备注)
        ("002460", "20260408", "TRNG_Ascending+FLAGU 组合分96.8"),
        ("000065", "20260408", "上升三角突破+百日新高"),
        ("000037", "20260408", "对称三角双周期"),
        ("002730", "20260408", "对称三角"),
        ("002432", "20260408", "对称三角"),
        ("600036", "20260331", "VCPU+DBOT 组合（招行）"),
        ("002281", "20260408", "DBOT 双底"),
        ("688127", "20260408", "DBOT 双底"),
        ("688601", "20260408", "DBOT 双底"),
    ]
    print(f"\n=== 批量验证 {len(candidates)} 只候选股 ===\n")
    for code, cutoff, note in candidates:
        print(f"\n--- {code} ({note}) cutoff={cutoff} ---")
        for tf in ["daily", "weekly"]:
            try:
                visualize_patterns(code, cutoff, tf)
            except Exception as e:
                print(f"  [ERR] {code} {tf}: {e}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="形态识别可视化验证")
    parser.add_argument("code", nargs="?", help="股票代码")
    parser.add_argument("cutoff", nargs="?", default=None,
                        help="截止日期 YYYYMMDD（默认最新）")
    parser.add_argument("--batch", action="store_true", help="批量验证常用候选")
    parser.add_argument("--tf", choices=["daily", "weekly", "both"], default="both",
                        help="时间框架")
    args = parser.parse_args()

    if args.batch:
        batch_verify()
        return

    if not args.code:
        parser.print_help()
        return

    print(f"\n=== 验证 {args.code} cutoff={args.cutoff or '最新'} ===")
    if args.tf in ("daily", "both"):
        visualize_patterns(args.code, args.cutoff, "daily")
    if args.tf in ("weekly", "both"):
        visualize_patterns(args.code, args.cutoff, "weekly")


if __name__ == "__main__":
    main()
