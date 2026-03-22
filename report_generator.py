#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - PDF投顾报告生成模块

生成专业二级市场投顾报告，包含：
  · 封面页（品牌封面+摘要指标）
  · 执行摘要（市场判断+板块概览）
  · 推荐总览表（全部标的一览）
  · 个股详细分析（K线图+专家评分+投资逻辑）
  · 风险提示与免责声明

依赖: reportlab >= 4.0, matplotlib, mplfinance, pandas, numpy
"""

import io
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image, KeepTogether,
    NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer,
    Table, TableStyle,
)

# ─────────────────────────────────────────────────────────────────────
#  颜色系统
# ─────────────────────────────────────────────────────────────────────
NAVY        = colors.HexColor("#1C2B4B")
NAVY2       = colors.HexColor("#2E5F8A")
NAVY3       = colors.HexColor("#4A7DB5")
GOLD        = colors.HexColor("#C5A028")
GOLD2       = colors.HexColor("#E8C84A")
BG_COVER    = colors.HexColor("#111E35")
BG_PAGE     = colors.HexColor("#F5F7FC")
BG_CARD     = colors.HexColor("#EBF0FA")
BG_ALT      = colors.HexColor("#EEF2F8")
BORDER      = colors.HexColor("#C8D3E5")
TEXT_DARK   = colors.HexColor("#1A1A2E")
TEXT_MID    = colors.HexColor("#4A5568")
TEXT_LIGHT  = colors.HexColor("#8896AB")
C_GREEN     = colors.HexColor("#16A34A")
C_RED       = colors.HexColor("#DC2626")
C_ORANGE    = colors.HexColor("#D97706")
C_WHITE     = colors.white

# matplotlib 同色系
MPL_UP      = "#E8312A"   # 阳线红
MPL_DOWN    = "#18A96A"   # 阴线绿
MPL_BG      = "#F5F7FC"
MPL_GRID    = "#E0E6F0"
MPL_TEXT    = "#4A5568"
MPL_MA5     = "#F39C12"
MPL_MA10    = "#3498DB"
MPL_MA20    = "#9B59B6"
MPL_MA60    = "#E67E22"

# ─────────────────────────────────────────────────────────────────────
#  字体注册
# ─────────────────────────────────────────────────────────────────────
_FONTS_READY = False
F_BODY = "SimHei"
F_BOLD = "SimHei"


def _register_fonts() -> None:
    global _FONTS_READY, F_BODY, F_BOLD

    if _FONTS_READY:
        return

    WIN_FONTS = "C:/Windows/Fonts"
    candidates = {
        "SimHei":    [f"{WIN_FONTS}/simhei.ttf"],
        "SimHeiB":   [f"{WIN_FONTS}/simhei.ttf"],
        "YaHei":     [f"{WIN_FONTS}/msyh.ttc"],
        "YaHeiB":    [f"{WIN_FONTS}/msyhbd.ttc"],
        "SimSun":    [f"{WIN_FONTS}/simsun.ttc"],
    }

    registered: Dict[str, bool] = {}
    for alias, paths in candidates.items():
        for path in paths:
            if os.path.exists(path):
                try:
                    if path.lower().endswith(".ttc"):
                        pdfmetrics.registerFont(TTFont(alias, path, subfontIndex=0))
                    else:
                        pdfmetrics.registerFont(TTFont(alias, path))
                    registered[alias] = True
                    break
                except Exception:
                    pass

    if "YaHei" in registered:
        F_BODY = "YaHei"
        F_BOLD = "YaHeiB" if "YaHeiB" in registered else "YaHei"
    elif "SimHei" in registered:
        F_BODY = "SimHei"
        F_BOLD = "SimHei"
    else:
        F_BODY = "Helvetica"
        F_BOLD = "Helvetica-Bold"
        print("[报告] 未找到中文字体，中文字符可能显示为方框")

    # matplotlib 中文字体
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans SC", "Arial"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    _FONTS_READY = True


# ─────────────────────────────────────────────────────────────────────
#  段落样式工厂
# ─────────────────────────────────────────────────────────────────────
def _styles() -> Dict[str, ParagraphStyle]:
    _register_fonts()
    s = {}

    def add(name, *, font=None, size=10, color=TEXT_DARK, align=TA_LEFT,
            leading=None, sb=0, sa=4, bold=False):
        f = font or (F_BOLD if bold else F_BODY)
        s[name] = ParagraphStyle(
            name, fontName=f, fontSize=size, textColor=color,
            alignment=align, leading=leading or size * 1.45,
            spaceBefore=sb, spaceAfter=sa,
        )

    # ── 封面 ──────────────────────────────────────────────────────
    add("cv_title",    size=30, color=C_WHITE, align=TA_CENTER, bold=True, leading=40, sa=6)
    add("cv_sub",      size=13, color=colors.HexColor("#AABDD8"), align=TA_CENTER, sa=4)
    add("cv_date",     size=11, color=GOLD2, align=TA_CENTER, sa=2)
    add("cv_metric_n", size=26, color=GOLD2, align=TA_CENTER, bold=True, sa=0)
    add("cv_metric_l", size=9,  color=colors.HexColor("#8AA0C0"), align=TA_CENTER, sa=2)
    add("cv_disc",     size=7.5,color=colors.HexColor("#607090"), align=TA_CENTER, leading=11)

    # ── 页面正文 ─────────────────────────────────────────────────
    add("h1",   size=17, color=NAVY, bold=True, sb=8, sa=5, leading=24)
    add("h2",   size=13, color=NAVY, bold=True, sb=6, sa=4, leading=18)
    add("h3",   size=11, color=NAVY2, bold=True, sb=4, sa=3)
    add("body", size=9.5, color=TEXT_DARK, align=TA_JUSTIFY, leading=14, sa=3)
    add("body_s", size=8.5, color=TEXT_MID, leading=12, sa=2)
    add("cap",  size=8,  color=TEXT_LIGHT, align=TA_CENTER, leading=10)
    add("disc", size=7.5, color=TEXT_LIGHT, align=TA_JUSTIFY, leading=11)

    # ── 表格内 ───────────────────────────────────────────────────
    add("th",   size=9, color=C_WHITE, align=TA_CENTER, bold=True)
    add("td",   size=9, color=TEXT_DARK, align=TA_CENTER)
    add("td_l", size=9, color=TEXT_DARK, align=TA_LEFT)
    add("td_g", size=9, color=C_GREEN, align=TA_CENTER, bold=True)
    add("td_r", size=9, color=C_RED,   align=TA_CENTER, bold=True)
    add("td_o", size=9, color=C_ORANGE,align=TA_CENTER, bold=True)

    # ── 股票详情页标题 ───────────────────────────────────────────
    add("stk_code",  size=22, color=NAVY, bold=True, sa=1)
    add("stk_name",  size=13, color=TEXT_DARK, bold=True, sa=2)
    add("met_val",   size=14, color=NAVY, bold=True, align=TA_CENTER, sa=0)
    add("met_lbl",   size=8,  color=TEXT_LIGHT, align=TA_CENTER, sa=2)

    # ── 页眉页脚 ─────────────────────────────────────────────────
    add("header", size=8,   color=TEXT_LIGHT, align=TA_RIGHT)
    add("footer", size=7.5, color=TEXT_LIGHT, align=TA_CENTER)

    return s


# ─────────────────────────────────────────────────────────────────────
#  图表生成器
# ─────────────────────────────────────────────────────────────────────
class ChartMaker:
    """生成股票分析图表（K线图、专家评分图、资金流图）"""

    TMP_DIR = tempfile.mkdtemp(prefix="stock_report_")

    @classmethod
    def _tmpfile(cls, suffix=".png") -> str:
        import uuid
        return os.path.join(cls.TMP_DIR, f"{uuid.uuid4().hex}{suffix}")

    # ── K 线图（日线）─────────────────────────────────────────────
    @classmethod
    def make_kline(
        cls,
        df: pd.DataFrame,
        title: str = "",
        show_weeks: int = 60,
        figsize: Tuple = (11, 5.5),
    ) -> Optional[str]:
        """
        生成日线 K 线蜡烛图，含 MA5/10/20/60、MACD、成交量子图。
        返回临时 PNG 文件路径。
        """
        if df is None or len(df) < 10:
            return None
        try:
            df = df.tail(show_weeks).reset_index(drop=True)
            n = len(df)
            close = df["close"]

            # ── 预计算 ──
            ma5  = close.rolling(5,  min_periods=1).mean()
            ma10 = close.rolling(10, min_periods=1).mean()
            ma20 = close.rolling(20, min_periods=1).mean()
            ma60 = close.rolling(60, min_periods=1).mean()

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            dif   = ema12 - ema26
            dea   = dif.ewm(span=9, adjust=False).mean()
            macd  = 2 * (dif - dea)

            # ── 布局 ──
            fig = plt.figure(figsize=figsize, facecolor=MPL_BG)
            gs  = gridspec.GridSpec(
                3, 1, height_ratios=[3.5, 1, 1.2],
                hspace=0.04, left=0.06, right=0.97, top=0.91, bottom=0.08,
            )
            ax_k   = fig.add_subplot(gs[0])
            ax_v   = fig.add_subplot(gs[1], sharex=ax_k)
            ax_m   = fig.add_subplot(gs[2], sharex=ax_k)

            for ax in (ax_k, ax_v, ax_m):
                ax.set_facecolor(MPL_BG)
                ax.grid(color=MPL_GRID, linewidth=0.5, linestyle="--", alpha=0.8)
                ax.tick_params(colors=MPL_TEXT, labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor(MPL_GRID)

            # ── K 线 ──
            for i in range(n):
                o = df["open"].iloc[i]
                c = df["close"].iloc[i]
                h = df["high"].iloc[i]
                lo = df["low"].iloc[i]
                color = MPL_UP if c >= o else MPL_DOWN
                ax_k.bar(i, abs(c - o), bottom=min(o, c),
                         color=color, width=0.6, linewidth=0, zorder=3)
                ax_k.plot([i, i], [lo, min(o, c)],  color=color, lw=0.7, zorder=2)
                ax_k.plot([i, i], [max(o, c), h],   color=color, lw=0.7, zorder=2)

            # ── 均线 ──
            for ma, col, lbl in [
                (ma5,  MPL_MA5,  "MA5"),
                (ma10, MPL_MA10, "MA10"),
                (ma20, MPL_MA20, "MA20"),
                (ma60, MPL_MA60, "MA60"),
            ]:
                ax_k.plot(range(n), ma.values, color=col, lw=1.2, label=lbl, zorder=4)

            ax_k.legend(loc="upper left", fontsize=7, framealpha=0.75,
                        ncol=4, handlelength=1.2, columnspacing=0.8)
            ax_k.set_title(title, fontsize=10, color="#1A1A2E", pad=6, fontweight="bold")
            ax_k.set_ylabel("价格 (元)", fontsize=7.5, color=MPL_TEXT)

            # ── 成交量 ──
            for i in range(n):
                c = df["close"].iloc[i]; o = df["open"].iloc[i]
                color = MPL_UP if c >= o else MPL_DOWN
                ax_v.bar(i, df["volume"].iloc[i], color=color, width=0.6,
                         alpha=0.75, linewidth=0)
            vol_ma5 = df["volume"].rolling(5, min_periods=1).mean()
            ax_v.plot(range(n), vol_ma5.values, color=MPL_MA5, lw=1.0)
            ax_v.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{x/1e8:.1f}亿" if x >= 1e8 else f"{x/1e4:.0f}万"
            ))
            ax_v.set_ylabel("成交量", fontsize=7, color=MPL_TEXT)

            # ── MACD ──
            colors_macd = [MPL_UP if v >= 0 else MPL_DOWN for v in macd.values]
            ax_m.bar(range(n), macd.values, color=colors_macd, alpha=0.7,
                     width=0.6, linewidth=0)
            ax_m.plot(range(n), dif.values, color="#E8C84A", lw=1.0, label="DIF")
            ax_m.plot(range(n), dea.values, color="#A78BFA", lw=1.0, label="DEA")
            ax_m.axhline(0, color=MPL_GRID, lw=0.8)
            ax_m.legend(loc="upper left", fontsize=6.5, framealpha=0.75,
                        ncol=2, handlelength=1.0)
            ax_m.set_ylabel("MACD", fontsize=7, color=MPL_TEXT)

            # ── X 轴日期 ──
            tick_step = max(1, n // 8)
            tick_pos  = list(range(0, n, tick_step))
            if "date" in df.columns:
                tick_labels = [str(df["date"].iloc[i])[:10] for i in tick_pos]
            else:
                tick_labels = [str(i) for i in tick_pos]
            ax_m.set_xticks(tick_pos)
            ax_m.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=7)
            plt.setp(ax_k.get_xticklabels(), visible=False)
            plt.setp(ax_v.get_xticklabels(), visible=False)

            path = cls._tmpfile()
            fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=MPL_BG)
            plt.close(fig)
            return path
        except Exception as e:
            print(f"  [图表] K线生成失败: {e}")
            return None

    # ── 专家评分条形图 ────────────────────────────────────────────
    @classmethod
    def make_expert_bar(
        cls,
        expert_stars: Dict[str, float],
        stock_name: str = "",
        figsize: Tuple = (7.5, 2.8),
    ) -> Optional[str]:
        """6位专家评分横向条形图"""
        try:
            order   = ["E1", "E2", "E3", "E4", "E5", "E6"]
            labels  = ["E1\n动量", "E2\n成长", "E3\n多因子", "E4\n技术", "E5\n资金", "E6\n催化"]
            weights = [0.15, 0.20, 0.20, 0.15, 0.15, 0.15]
            scores  = [float(expert_stars.get(eid, 0)) for eid in order]
            bar_colors = ["#F39C12","#E74C3C","#2ECC71","#3498DB","#9B59B6","#1ABC9C"]

            fig, ax = plt.subplots(figsize=figsize, facecolor=MPL_BG)
            ax.set_facecolor(MPL_BG)

            bars = ax.barh(labels, scores, color=bar_colors, alpha=0.85,
                           height=0.55, edgecolor="white", linewidth=0.8)

            # 数值标签
            for bar, score, w in zip(bars, scores, weights):
                if score > 0:
                    ax.text(score + 0.05, bar.get_y() + bar.get_height() / 2,
                            f"{score:.1f}★  (权重{int(w*100)}%)",
                            va="center", ha="left", fontsize=8.5, color=MPL_TEXT)

            ax.set_xlim(0, 6.5)
            ax.set_xlabel("评分（满分5星）", fontsize=8, color=MPL_TEXT)
            ax.set_title(f"专家评分汇总 — {stock_name}", fontsize=9.5,
                         color="#1C2B4B", fontweight="bold", pad=6)
            ax.axvline(x=3, color=MPL_GRID, linestyle="--", lw=1.0, alpha=0.8)
            ax.tick_params(colors=MPL_TEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(MPL_GRID)

            plt.tight_layout()
            path = cls._tmpfile()
            fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=MPL_BG)
            plt.close(fig)
            return path
        except Exception as e:
            print(f"  [图表] 专家评分图生成失败: {e}")
            return None

    # ── 月线/周线缩略趋势 ────────────────────────────────────────
    @classmethod
    def make_trend_mini(
        cls,
        df_m: Optional[pd.DataFrame],
        df_w: Optional[pd.DataFrame],
        figsize: Tuple = (7.5, 2.5),
    ) -> Optional[str]:
        """月线+周线趋势折线图（缩略）"""
        if df_m is None and df_w is None:
            return None
        try:
            fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=MPL_BG)
            data_pairs = [(df_m, "月线趋势", axes[0]), (df_w, "周线趋势", axes[1])]

            for df, lbl, ax in data_pairs:
                ax.set_facecolor(MPL_BG)
                ax.set_title(lbl, fontsize=9, color="#1A1A2E", fontweight="bold", pad=4)
                for spine in ax.spines.values():
                    spine.set_edgecolor(MPL_GRID)
                ax.grid(color=MPL_GRID, lw=0.5, linestyle="--", alpha=0.7)
                ax.tick_params(colors=MPL_TEXT, labelsize=7)

                if df is None or len(df) < 3:
                    ax.text(0.5, 0.5, "数据不足", transform=ax.transAxes,
                            ha="center", va="center", fontsize=9, color="#8896AB")
                    continue

                df = df.tail(24).reset_index(drop=True)
                c = df["close"]
                n = len(c)
                x = range(n)

                # 填充面积
                min_c = c.min()
                ax.fill_between(x, c.values, min_c, alpha=0.12, color="#3498DB")
                ax.plot(x, c.values, color="#3498DB", lw=1.8, zorder=3)

                # MA
                ma = c.rolling(6, min_periods=1).mean()
                ax.plot(x, ma.values, color=MPL_MA20, lw=1.0, linestyle="--", alpha=0.8)

                # 最新价标注
                ax.annotate(
                    f"{c.iloc[-1]:.2f}",
                    xy=(n - 1, c.iloc[-1]),
                    fontsize=7.5, color="#1C2B4B", fontweight="bold",
                    xytext=(4, 0), textcoords="offset points",
                )

                # X 轴
                if "date" in df.columns:
                    step = max(1, n // 5)
                    ax.set_xticks(range(0, n, step))
                    ax.set_xticklabels(
                        [str(df["date"].iloc[i])[:7] for i in range(0, n, step)],
                        rotation=20, ha="right", fontsize=6.5,
                    )

            plt.tight_layout(pad=0.8)
            path = cls._tmpfile()
            fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=MPL_BG)
            plt.close(fig)
            return path
        except Exception as e:
            print(f"  [图表] 趋势缩略图失败: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────
#  辅助：创建带圆角/颜色的标签单元
# ─────────────────────────────────────────────────────────────────────
def _risk_tag(risk_level: str, styles: dict) -> Paragraph:
    color_map = {"低风险": "#16A34A", "中风险": "#D97706", "高风险": "#DC2626"}
    bg = color_map.get(risk_level, "#6B7280")
    return Paragraph(
        f'<font color="white"><b>{risk_level}</b></font>',
        styles["td"],
    )


def _stars_str(n: float) -> str:
    full  = int(round(n))
    full  = max(0, min(5, full))
    return "★" * full + "☆" * (5 - full)


def _fmt_pct(v) -> str:
    try:
        f = float(v)
        sign = "+" if f >= 0 else ""
        return f"{sign}{f:.1f}%"
    except Exception:
        return str(v) if v else "—"


# ─────────────────────────────────────────────────────────────────────
#  主报告生成器
# ─────────────────────────────────────────────────────────────────────
class ReportGenerator:
    """PDF 投顾报告生成器"""

    PAGE_W, PAGE_H = A4                # 595.28 x 841.89 pt
    MARGIN_T = 1.8 * cm
    MARGIN_B = 1.8 * cm
    MARGIN_L = 1.8 * cm
    MARGIN_R = 1.8 * cm
    HEADER_H = 0.8 * cm

    def __init__(self, output_dir: str = "."):
        _register_fonts()
        self.output_dir = output_dir
        self.styles = _styles()
        self.today  = datetime.now().strftime("%Y年%m月%d日")
        self.dt_str = datetime.now().strftime("%Y%m%d_%H%M")

    # ── 页眉 / 页脚回调 ──────────────────────────────────────────
    def _on_cover(self, canvas, doc):
        """封面：绘制深蓝渐变背景 + 装饰线"""
        w, h = self.PAGE_W, self.PAGE_H
        canvas.saveState()

        # 深色背景
        canvas.setFillColor(BG_COVER)
        canvas.rect(0, 0, w, h, fill=1, stroke=0)

        # 顶部金色装饰横条
        canvas.setFillColor(GOLD)
        canvas.rect(0, h - 8, w, 8, fill=1, stroke=0)

        # 底部装饰横条
        canvas.setFillColor(GOLD)
        canvas.rect(0, 0, w, 5, fill=1, stroke=0)

        # 中间分割线
        canvas.setStrokeColor(colors.HexColor("#2A4070"))
        canvas.setLineWidth(0.5)
        canvas.line(2 * cm, h * 0.38, w - 2 * cm, h * 0.38)
        canvas.line(2 * cm, h * 0.36, w - 2 * cm, h * 0.36)

        canvas.restoreState()

    def _on_page(self, canvas, doc):
        """普通页：页眉线 + 页脚"""
        w, h = self.PAGE_W, self.PAGE_H
        canvas.saveState()

        # 页面背景（白色）
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, w, h, fill=1, stroke=0)

        # 顶部细线
        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 3, w, 3, fill=1, stroke=0)

        # 页眉文字
        canvas.setFont(F_BODY, 7.5)
        canvas.setFillColor(TEXT_LIGHT)
        canvas.drawRightString(
            w - self.MARGIN_R,
            h - self.MARGIN_T + 0.2 * cm,
            f"A股多智能体投资分析报告  |  {self.today}",
        )

        # 底部分割线
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(self.MARGIN_L, self.MARGIN_B - 0.4 * cm,
                    w - self.MARGIN_R, self.MARGIN_B - 0.4 * cm)

        # 页脚
        canvas.setFont(F_BODY, 7.5)
        canvas.setFillColor(TEXT_LIGHT)
        canvas.drawCentredString(
            w / 2,
            self.MARGIN_B - 0.75 * cm,
            f"第 {doc.page} 页  |  仅供参考，不构成投资建议  |  请阅读免责声明",
        )

        canvas.restoreState()

    # ── 封面页内容 ───────────────────────────────────────────────
    def _build_cover(
        self,
        approved: List[Dict],
        top_sectors: List[str],
        n_models: int,
    ) -> List:
        S = self.styles
        elems = []

        # 上部空间（留给背景装饰）
        elems.append(Spacer(1, 3.8 * cm))

        # ── 报告类型标签 ──
        tag_tbl = Table(
            [[Paragraph("AI 量化多智能体", S["cv_sub"])]],
            colWidths=[12 * cm],
        )
        tag_tbl.setStyle(TableStyle([
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#1E3A6E")),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elems.append(Table([[tag_tbl]], colWidths=[self.PAGE_W - 2 * self.MARGIN_L]))
        elems.append(Spacer(1, 0.4 * cm))

        # ── 主标题 ──
        elems.append(Paragraph("A 股 投 资 分 析 报 告", S["cv_title"]))
        elems.append(Spacer(1, 0.3 * cm))
        elems.append(Paragraph("Investment Research Report · A-Share Market", S["cv_sub"]))
        elems.append(Spacer(1, 0.5 * cm))
        elems.append(Paragraph(self.today, S["cv_date"]))

        elems.append(Spacer(1, 2.5 * cm))

        # ── 摘要指标卡 ──
        n_rec    = len(approved)
        n_sec    = len(top_sectors)
        strong   = sum(1 for s in approved if s.get("consensus_level", "") in ("强共识", "4专家+"))
        sec_str  = "、".join(top_sectors[:4]) if top_sectors else "—"

        metric_data = [
            [
                Paragraph(str(n_rec),    S["cv_metric_n"]),
                Paragraph(str(n_sec),    S["cv_metric_n"]),
                Paragraph(str(strong),   S["cv_metric_n"]),
                Paragraph(str(n_models), S["cv_metric_n"]),
            ],
            [
                Paragraph("推荐标的",    S["cv_metric_l"]),
                Paragraph("核心板块",    S["cv_metric_l"]),
                Paragraph("强共识标的",  S["cv_metric_l"]),
                Paragraph("参与模型",    S["cv_metric_l"]),
            ],
        ]
        cw = (self.PAGE_W - 2 * self.MARGIN_L) / 4
        mt = Table(metric_data, colWidths=[cw] * 4)
        mt.setStyle(TableStyle([
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND",  (0, 0), (-1, -1), colors.HexColor("#1A3060")),
            ("LINEAFTER",   (0, 0), (2, -1),  0.5, colors.HexColor("#2E4A80")),
            ("TOPPADDING",  (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        elems.append(mt)

        elems.append(Spacer(1, 1.0 * cm))

        # ── 核心板块 ──
        elems.append(Paragraph(f"核心板块：{sec_str}", S["cv_sub"]))

        elems.append(Spacer(1, 2.5 * cm))

        # ── 免责声明 ──
        disc = (
            "【重要提示】本报告由AI多智能体系统自动生成，仅供投资者参考，"
            "不构成任何投资建议或承诺。股市有风险，投资需谨慎。"
            "请在充分了解风险的基础上，结合自身财务状况和风险承受能力做出独立投资决策。"
        )
        elems.append(Paragraph(disc, S["cv_disc"]))

        return elems

    # ── 执行摘要页 ───────────────────────────────────────────────
    def _build_exec_summary(
        self,
        approved: List[Dict],
        top_sectors: List[str],
        market_view: str,
        portfolio_advice: str,
    ) -> List:
        S = self.styles
        elems = [Paragraph("执行摘要", S["h1"])]
        elems.append(HRFlowable(width="100%", thickness=2, color=GOLD, spaceAfter=10))

        # 市场观点
        if market_view:
            elems.append(Paragraph("市场环境判断", S["h2"]))
            elems.append(Paragraph(market_view, S["body"]))
            elems.append(Spacer(1, 0.3 * cm))

        # 核心板块
        if top_sectors:
            elems.append(Paragraph("今日核心板块", S["h2"]))
            sec_items = " · ".join(f"<b>{s}</b>" for s in top_sectors)
            elems.append(Paragraph(sec_items, S["body"]))
            elems.append(Spacer(1, 0.3 * cm))

        # 组合建议
        if portfolio_advice:
            elems.append(Paragraph("组合建议", S["h2"]))
            elems.append(Paragraph(portfolio_advice, S["body"]))
            elems.append(Spacer(1, 0.3 * cm))

        # 分析方法说明
        elems.append(Paragraph("分析方法论", S["h2"]))
        method_text = (
            "本报告采用 <b>AI多智能体联合辩论</b> 选股框架，系统由以下环节构成：<br/>"
            "①&nbsp;板块初筛（多模型投票，选出3-5个核心板块）→ "
            "②&nbsp;六专家并行分析（动量/成长/多因子/技术/资金/催化，各自独立→辩论→聚合）→ "
            "③&nbsp;加权投票仲裁（专家权重×共识奖励）→ "
            "④&nbsp;风险控制过滤（硬性排除+风险评级+仓位建议）。"
        )
        elems.append(Paragraph(method_text, S["body"]))
        elems.append(Spacer(1, 0.2 * cm))

        weight_data = [
            [Paragraph(k, S["td"]) for k in ["专家", "E1动量", "E2成长", "E3多因子", "E4技术", "E5资金", "E6催化"]],
            [Paragraph(k, S["td"]) for k in ["权重", "15%", "20%", "20%", "15%", "15%", "15%"]],
        ]
        wt = Table(weight_data, colWidths=[(self.PAGE_W - 2 * self.MARGIN_L) / 7] * 7)
        wt.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
            ("BACKGROUND",    (0, 1), (-1, 1), BG_CARD),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME",      (0, 0), (-1, 0),  F_BOLD),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
            ("FONTNAME",      (0, 1), (-1, -1), F_BODY),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ]))
        elems.append(wt)

        return elems

    # ── 推荐总览表 ───────────────────────────────────────────────
    def _build_recommendation_table(self, approved: List[Dict], soft_excluded: List[Dict] = None) -> List:
        S = self.styles
        elems = [PageBreak(), Paragraph("推荐股票总览（按Borda评分排序）", S["h1"])]
        elems.append(HRFlowable(width="100%", thickness=2, color=GOLD, spaceAfter=10))

        n_approved = len(approved)
        n_excluded = len(soft_excluded) if soft_excluded else 0
        elems.append(Paragraph(
            f"共 {n_approved + n_excluded} 只 · 风控通过 {n_approved} 只 · 风控提示 {n_excluded} 只（⚠ 标记）",
            S["body_s"],
        ))
        elems.append(Spacer(1, 0.2 * cm))

        # 合并并按 Borda/final_score 降序排列
        all_items = []
        for stk in approved:
            all_items.append({"data": stk, "is_excluded": False, "reason": ""})
        for se in (soft_excluded or []):
            all_items.append({"data": se, "is_excluded": True, "reason": se.get("reason", "")})
        all_items.sort(
            key=lambda x: float(x["data"].get("borda_score", x["data"].get("final_score", x["data"].get("weighted_score", 0))) or 0),
            reverse=True,
        )

        W = self.PAGE_W - self.MARGIN_L - self.MARGIN_R
        col_w = [0.5*cm, 1.8*cm, 2.5*cm, 2.8*cm, 2.5*cm, 2.0*cm, 2.2*cm, 2.5*cm, 2.2*cm]
        headers = ["#", "代码", "名称", "板块", "共识", "综合评分", "风险", "仓位建议", "持有周期"]

        rows = [[Paragraph(h, S["th"]) for h in headers]]
        excluded_row_indices = []

        for rank_i, item in enumerate(all_items, 1):
            stk = item["data"]
            code       = stk.get("code", "")
            name       = stk.get("name", "")
            sector     = stk.get("sector", "")
            score      = stk.get("borda_score", stk.get("final_score", stk.get("weighted_score", 0)))

            if item["is_excluded"]:
                consensus = "—"
                risk      = "⚠风控"
                position  = item["reason"][:18] if item["reason"] else "风控提示"
                hold      = "—"
                rs        = S["td_r"]
                excluded_row_indices.append(rank_i)  # 1-based data row index
            else:
                consensus  = stk.get("consensus_level", "—")
                risk       = stk.get("risk_level", "未评")
                position   = stk.get("position_advice", "—")
                hold       = stk.get("holding_period", stk.get("hold_period", "—"))
                if "低" in risk:
                    rs = S["td_g"]
                elif "高" in risk:
                    rs = S["td_r"]
                else:
                    rs = S["td_o"]

            rows.append([
                Paragraph(str(rank_i), S["td"]),
                Paragraph(code,        S["td"]),
                Paragraph(name,        S["td_l"]),
                Paragraph(sector,      S["td_l"]),
                Paragraph(consensus,   S["td"]),
                Paragraph(f"{float(score):.1f}分" if score else "—", S["td"]),
                Paragraph(risk,        rs),
                Paragraph(position,    S["body_s"]),
                Paragraph(hold,        S["td"]),
            ])

        tbl = Table(rows, colWidths=col_w, repeatRows=1)
        tbl.setStyle(TableStyle([
            # 表头
            ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
            ("FONTNAME",      (0, 0), (-1, 0), F_BOLD),
            ("TEXTCOLOR",     (0, 0), (-1, 0), C_WHITE),
            ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
            # 通用
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME",      (0, 1), (-1, -1), F_BODY),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_ALT]),
        ]))
        # 风控提示行用浅红色背景
        for row_i in excluded_row_indices:
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, row_i), (-1, row_i), colors.HexColor("#FFF0F0")),
            ]))
        elems.append(tbl)
        return elems

    # ── 个股分析页 ───────────────────────────────────────────────
    def _build_stock_page(
        self,
        stk: Dict,
        pkg: Optional[Dict],
        expert_summary: Dict,
        is_excluded: bool = False,
    ) -> List:
        S = self.styles
        elems = [PageBreak()]

        code    = stk.get("code", "")
        name    = stk.get("name", "")
        sector  = stk.get("sector", "")
        risk    = stk.get("risk_level", "未评")
        score   = stk.get("final_score", stk.get("weighted_score", 0))
        pos_adv = stk.get("position_advice", "—")
        stop_l  = stk.get("stop_loss", "—")
        entry   = stk.get("entry_point", "")
        logic   = stk.get("core_logic", "")
        hold    = stk.get("holding_period", "—")
        flags   = stk.get("risk_flags", [])
        consensus = stk.get("consensus_level", "")

        # 风控提示警告横幅
        if is_excluded:
            risk_warning = stk.get("risk_warning", "风控提示")
            warn_tbl = Table(
                [[Paragraph(f"⚠ 风控提示：{risk_warning}", S["td_r"])]],
                colWidths=[self.PAGE_W - self.MARGIN_L - self.MARGIN_R],
            )
            warn_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#FFF0F0")),
                ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                ("BOX",           (0, 0), (-1, -1), 1, C_RED),
            ]))
            elems.append(warn_tbl)
            elems.append(Spacer(1, 0.15 * cm))

        # ── 股票标题栏 ─────────────────────────────────────────
        rt = (pkg or {}).get("realtime", {})
        pe  = rt.get("市盈率-动态", rt.get("市盈率", "—"))
        pb  = rt.get("市净率", "—")
        cap = rt.get("总市值", "—")
        cur = rt.get("最新价", rt.get("收盘价", "—"))

        risk_color = C_GREEN if "低" in risk else (C_RED if "高" in risk else C_ORANGE)

        # 基本信息横幅
        banner_data = [[
            Paragraph(f"<b>{code}</b>", S["stk_code"]),
            Paragraph(name, S["stk_name"]),
            Paragraph(sector, S["body_s"]),
            Paragraph(f'<font color="{risk_color.hexval() if hasattr(risk_color,"hexval") else "#16A34A"}"><b>{risk}</b></font>', S["td"]),
            Paragraph(f"{float(score):.1f}分" if score else "—", S["td"]),
        ]]
        banner_w = [2.5*cm, 3.5*cm, 3.0*cm, 2.0*cm, 2.0*cm]
        banner_t = Table(banner_data, colWidths=banner_w)
        banner_t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), BG_CARD),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (0, -1),  10),
            ("LINEAFTER",     (0, 0), (3, -1),  0.5, BORDER),
            ("BOX",           (0, 0), (-1, -1), 1.5, NAVY),
        ]))

        # 指标行
        metric_data = [[
            Paragraph(str(cur),  S["met_val"]),
            Paragraph(str(pe),   S["met_val"]),
            Paragraph(str(pb),   S["met_val"]),
            Paragraph(str(cap),  S["met_val"]),
            Paragraph(pos_adv,   S["met_val"]),
            Paragraph(stop_l,    S["met_val"]),
        ], [
            Paragraph("最新价(元)", S["met_lbl"]),
            Paragraph("PE(TTM)",   S["met_lbl"]),
            Paragraph("PB",        S["met_lbl"]),
            Paragraph("总市值",    S["met_lbl"]),
            Paragraph("建议仓位",  S["met_lbl"]),
            Paragraph("止损参考",  S["met_lbl"]),
        ]]
        mw = (self.PAGE_W - self.MARGIN_L - self.MARGIN_R) / 6
        metric_t = Table(metric_data, colWidths=[mw] * 6)
        metric_t.setStyle(TableStyle([
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND",    (0, 0), (-1, -1), colors.white),
            ("LINEAFTER",     (0, 0), (4, -1),  0.5, BORDER),
            ("LINEBEFORE",    (0, 0), (0, -1),  0.5, BORDER),
            ("LINEBELOW",     (0, -1), (-1, -1), 0.5, BORDER),
            ("LINEABOVE",     (0, 0), (-1, 0),  0.5, BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))

        elems.append(banner_t)
        elems.append(Spacer(1, 0.15 * cm))
        elems.append(metric_t)
        elems.append(Spacer(1, 0.25 * cm))

        # ── K 线图 + 趋势图（双列布局）─────────────────────────
        df_d = (pkg or {}).get("daily")
        df_w = (pkg or {}).get("weekly")
        df_m = (pkg or {}).get("monthly")

        kline_img = ChartMaker.make_kline(df_d, title=f"{code} {name}  日线K线（近60日）")
        trend_img = ChartMaker.make_trend_mini(df_m, df_w)

        chart_row = []
        if kline_img and os.path.exists(kline_img):
            chart_row.append(Image(kline_img, width=11.2 * cm, height=5.5 * cm))
        else:
            chart_row.append(Paragraph("K线图数据不足", S["cap"]))

        if trend_img and os.path.exists(trend_img):
            chart_row.append(Image(trend_img, width=7.5 * cm, height=2.6 * cm))

        if len(chart_row) == 2:
            chart_tbl = Table([chart_row], colWidths=[11.5 * cm, 7.5 * cm])
            chart_tbl.setStyle(TableStyle([
                ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            elems.append(chart_tbl)
        elif chart_row:
            elems.append(chart_row[0])

        elems.append(Spacer(1, 0.2 * cm))

        # ── 专家评分图 ────────────────────────────────────────
        expert_stars: Dict[str, float] = {}
        for eid, edata in expert_summary.items():
            for pick in edata.get("picks", []):
                if str(pick.get("code", "")) == code:
                    expert_stars[eid] = float(pick.get("stars", 0))

        if expert_stars:
            bar_img = ChartMaker.make_expert_bar(expert_stars, name)
            if bar_img and os.path.exists(bar_img):
                elems.append(Image(bar_img, width=10.5 * cm, height=3.0 * cm))
                elems.append(Spacer(1, 0.15 * cm))

        # ── 专家共识标签 ─────────────────────────────────────
        voted_experts = [
            f"{eid}({_stars_str(s)})" for eid, s in sorted(expert_stars.items())
        ]
        voted_str = "  |  ".join(voted_experts) if voted_experts else "—"
        elems.append(Paragraph(
            f"专家投票：{voted_str}  ·  共识等级：<b>{consensus}</b>",
            S["body_s"],
        ))
        elems.append(Spacer(1, 0.15 * cm))

        # ── 投资逻辑 ─────────────────────────────────────────
        elems.append(Paragraph("投资逻辑", S["h3"]))
        if logic:
            elems.append(Paragraph(logic, S["body"]))
        else:
            reasonings = stk.get("all_reasonings", [])
            if reasonings:
                for r in reasonings[:3]:
                    elems.append(Paragraph(f"· {r}", S["body_s"]))

        # ── 买入建议 ─────────────────────────────────────────
        if entry:
            elems.append(Spacer(1, 0.1 * cm))
            elems.append(Paragraph("买入时机", S["h3"]))
            elems.append(Paragraph(entry, S["body"]))

        # ── 风险提示 ─────────────────────────────────────────
        elems.append(Spacer(1, 0.1 * cm))
        elems.append(Paragraph("风险提示", S["h3"]))
        if flags:
            for f in flags:
                elems.append(Paragraph(f"⚠ {f}", S["body_s"]))
        else:
            elems.append(Paragraph("无额外风险提示，请参考整体止损建议", S["body_s"]))

        # 止损 + 持有周期
        sl_entry = Table([[
            Paragraph(f"止损：{stop_l}", S["body_s"]),
            Paragraph(f"建议持有：{hold}", S["body_s"]),
        ]], colWidths=[(self.PAGE_W - self.MARGIN_L - self.MARGIN_R) / 2] * 2)
        sl_entry.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), BG_ALT),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ]))
        elems.append(Spacer(1, 0.1 * cm))
        elems.append(sl_entry)

        return elems

    # ── 风险声明页 ───────────────────────────────────────────────
    def _build_disclaimer(self) -> List:
        S = self.styles
        elems = [PageBreak(), Paragraph("风险提示与免责声明", S["h1"])]
        elems.append(HRFlowable(width="100%", thickness=2, color=C_RED, spaceAfter=10))

        items = [
            ("一、投资风险提示",
             "本报告涉及的股票仅为AI模型的分析输出，不代表任何机构对相关股票的投资建议。"
             "证券市场存在不可预知的风险，包括但不限于：宏观经济风险、政策法规变动风险、"
             "上市公司经营风险、市场波动风险等。过往表现不代表未来收益。"),

            ("二、不构成投资建议",
             "本报告内容仅供参考，不构成任何投资建议或承诺。投资者应当结合自身的财务状况、"
             "风险承受能力及投资目标，独立判断是否适合进行相关投资，并自行承担投资决策所带来的"
             "一切风险和损失。"),

            ("三、数据来源说明",
             "本报告使用的数据来源于AKShare、Tushare等公开数据接口及各大模型的分析能力，"
             "数据准确性以实际市场为准。如因数据延迟、接口异常等导致分析偏差，本系统不承担"
             "相应责任。"),

            ("四、AI分析局限性",
             "本报告由多个大语言模型（LLM）联合分析生成。AI模型的分析存在固有局限，"
             "包括但不限于：知识截止日期限制、对突发事件的感知延迟、对市场情绪的量化不足等。"
             "模型的输出结果仅反映基于历史数据的统计规律，不代表对未来市场走势的确定性判断。"),

            ("五、止损纪律",
             "建议所有投资者严格遵守止损纪律。当股票价格触及止损位时，应果断执行止损，"
             "以控制单笔损失。切勿因短期亏损而加仓摊薄，避免小亏变大亏。"),

            ("六、仓位管理",
             "单只股票的仓位建议仅供参考，实际仓位应根据整体投资组合风险及个人风险承受能力"
             "综合决定。建议单只标的不超过总资产的15%，全仓追入高风险标的具有极大风险。"),
        ]

        for title, content in items:
            elems.append(Paragraph(title, S["h3"]))
            elems.append(Paragraph(content, S["disc"]))
            elems.append(Spacer(1, 0.2 * cm))

        elems.append(Spacer(1, 0.5 * cm))
        elems.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
        elems.append(Spacer(1, 0.2 * cm))
        elems.append(Paragraph(
            f"本报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            "系统版本：A股多智能体选股系统 v1.0  |  "
            "本报告版权归属系统运营方，未经授权不得转载或用于商业用途。",
            S["disc"],
        ))

        return elems

    # ── 主生成入口 ───────────────────────────────────────────────
    def generate(
        self,
        risk_result: Dict,
        arb_result: Dict,
        expert_summary: Dict,
        stock_packages: Dict,
        top_sectors: List[str],
        n_models: int = 4,
        output_filename: Optional[str] = None,
    ) -> str:
        """
        生成完整 PDF 投顾报告。

        返回: 生成的 PDF 文件路径。
        """
        approved  = risk_result.get("approved", [])
        soft_excluded = risk_result.get("soft_excluded", [])
        market_view = risk_result.get("market_timing", "")
        portfolio   = risk_result.get("portfolio_advice", "")

        all_stocks = approved + soft_excluded
        if not all_stocks:
            print("  [报告] 无推荐股票，跳过PDF生成")
            return ""

        filename = output_filename or os.path.join(
            self.output_dir, f"投顾报告_{self.dt_str}.pdf"
        )

        print(f"\n  [报告] 开始生成PDF: {filename}")
        print(f"  [报告] 推荐股票 {len(approved)} 只，风控提示 {len(soft_excluded)} 只")

        # ── 文档设置 ──────────────────────────────────────────
        doc = BaseDocTemplate(
            filename,
            pagesize=A4,
            leftMargin=self.MARGIN_L,
            rightMargin=self.MARGIN_R,
            topMargin=self.MARGIN_T,
            bottomMargin=self.MARGIN_B,
        )

        # 页面模板：封面（无页眉页脚）、正文
        cover_frame = Frame(0, 0, self.PAGE_W, self.PAGE_H, id="cover_frame",
                            leftPadding=self.MARGIN_L, rightPadding=self.MARGIN_R,
                            topPadding=0, bottomPadding=self.MARGIN_B)
        body_frame  = Frame(
            self.MARGIN_L, self.MARGIN_B,
            self.PAGE_W - self.MARGIN_L - self.MARGIN_R,
            self.PAGE_H - self.MARGIN_T - self.MARGIN_B,
            id="body_frame",
        )

        cover_tpl = PageTemplate(id="Cover", frames=[cover_frame], onPage=self._on_cover)
        body_tpl  = PageTemplate(id="Body",  frames=[body_frame],  onPage=self._on_page)
        doc.addPageTemplates([cover_tpl, body_tpl])

        # ── 构建内容 ──────────────────────────────────────────
        story = []

        # 1. 封面（使用 Cover 模板）
        story.extend(self._build_cover(all_stocks, top_sectors, n_models))
        story.append(NextPageTemplate("Body"))
        story.append(PageBreak())

        # 2. 执行摘要
        story.extend(self._build_exec_summary(approved, top_sectors, market_view, portfolio))

        # 3. 推荐总览表（按Borda评分排序，含风控提示股票）
        story.extend(self._build_recommendation_table(approved, soft_excluded))

        # 4. 个股详情 — 按Borda评分排序，统一展示
        # 合并并排序
        all_stock_items = []
        for stk in approved:
            borda = float(stk.get("borda_score", stk.get("final_score", stk.get("weighted_score", 0))) or 0)
            all_stock_items.append({"data": stk, "is_excluded": False, "borda": borda})
        for se in soft_excluded:
            code = se.get("code", "")
            # 从 arb_result 中找到完整股票数据
            full_stk = None
            for pick in arb_result.get("final_picks", []):
                if str(pick.get("code", "")) == code:
                    full_stk = pick
                    break
            if full_stk:
                full_stk["risk_warning"] = se.get("reason", "")
                borda = float(full_stk.get("borda_score", full_stk.get("final_score", 0)) or 0)
                all_stock_items.append({"data": full_stk, "is_excluded": True, "borda": borda})
        all_stock_items.sort(key=lambda x: x["borda"], reverse=True)

        for item in all_stock_items:
            stk = item["data"]
            code = stk.get("code", "")
            pkg = stock_packages.get(code)
            tag = "风控提示" if item["is_excluded"] else "个股分析"
            print(f"  [报告] 生成{tag}页: {code} {stk.get('name','')}")
            story.extend(self._build_stock_page(stk, pkg, expert_summary, is_excluded=item["is_excluded"]))

        # 5. 免责声明
        story.extend(self._build_disclaimer())

        # ── 构建 PDF ──────────────────────────────────────────
        doc.build(story)
        size_kb = os.path.getsize(filename) / 1024
        print(f"  [报告] PDF生成完成: {filename}  ({size_kb:.0f} KB)")

        return filename


# ─────────────────────────────────────────────────────────────────────
#  便捷调用函数
# ─────────────────────────────────────────────────────────────────────
def generate_report(
    risk_result: Dict,
    arb_result: Dict,
    expert_summary: Dict,
    stock_packages: Dict,
    top_sectors: List[str],
    n_models: int = 4,
    output_dir: str = ".",
    output_filename: Optional[str] = None,
) -> str:
    """
    一键生成 PDF 投顾报告。

    参数:
        risk_result     风控审查结果（来自 RiskController.filter()）
        arb_result      投票仲裁结果（来自 VotingArbitrator.arbitrate()）
        expert_summary  专家分析摘要字典
        stock_packages  股票数据包字典（含K线DataFrame）
        top_sectors     初筛板块名称列表
        n_models        参与分析的模型数量
        output_dir      输出目录
        output_filename 指定PDF文件名（可选）

    返回: PDF文件路径
    """
    gen = ReportGenerator(output_dir=output_dir)
    return gen.generate(
        risk_result=risk_result,
        arb_result=arb_result,
        expert_summary=expert_summary,
        stock_packages=stock_packages,
        top_sectors=top_sectors,
        n_models=n_models,
        output_filename=output_filename,
    )


# ─────────────────────────────────────────────────────────────────────
#  推荐股 + 异动股 总览图片生成
# ─────────────────────────────────────────────────────────────────────

def _compute_support_resistance(engine, codes: List[str]) -> Dict:
    """为一组股票代码计算支撑/压力位"""
    results = {}
    for code in codes:
        try:
            df = engine.fetch_kline(code, period="daily", n_bars=120)
            if df is None or len(df) < 20:
                results[code] = {}
                continue
            closes = df["close"].astype(float).tolist()
            highs = df["high"].astype(float).tolist()
            lows = df["low"].astype(float).tolist()
            current = closes[-1]
            ma20 = sum(closes[-20:]) / len(closes[-20:])
            n60 = min(60, len(closes))
            ma60 = sum(closes[-n60:]) / n60
            h20 = max(highs[-20:])
            l20 = min(lows[-20:])
            n60h = min(60, len(highs))
            h60 = max(highs[-n60h:])
            l60 = min(lows[-n60h:])
            sup = sorted([v for v in [ma20, ma60, l20] if v < current], reverse=True)
            res = sorted([v for v in [h20, h60] if v > current * 1.005])
            results[code] = {
                "price": round(current, 2),
                "ma20": round(ma20, 2), "ma60": round(ma60, 2),
                "s1": round(sup[0], 2) if sup else round(l20, 2),
                "s2": round(sup[1], 2) if len(sup) > 1 else round(l60, 2),
                "r1": round(res[0], 2) if res else round(h20, 2),
                "r2": round(res[1], 2) if len(res) > 1 else round(h60, 2),
            }
        except Exception:
            results[code] = {}
    return results


def generate_overview_image(
    risk_result: Dict,
    fusion_result: Optional[List[Dict]] = None,
    stock_profiles: Optional[Dict] = None,
    breakout_result: Optional[Dict] = None,
    mr_result: Optional[Dict] = None,
    engine=None,
    active_models: Optional[List[str]] = None,
    llm_mode: str = "direct",
    output_dir: str = ".",
) -> str:
    """
    生成推荐股 + 异动股拼接总览图片（PNG）。

    参数:
        risk_result      风控审查结果
        fusion_result    Borda融合结果列表
        stock_profiles   股票画像字典 {code: profile_text}
        breakout_result  异动爆发扫描结果（可选）
        mr_result        市场雷达结果（含概念炒作预判+ETF）
        engine           DataEngine 实例
        active_models    本次使用的模型列表
        llm_mode         LLM 接入模式描述
        output_dir       输出目录

    返回: 图片文件路径
    """
    import re
    import warnings
    warnings.filterwarnings("ignore")
    from matplotlib.patches import Rectangle as Rect

    models_str = ", ".join(active_models) if active_models else "?"
    stock_profiles = stock_profiles or {}
    fusion_result = fusion_result or []
    mr_result = mr_result or {}

    # ── 构建板块→概念阶段+ETF映射（from MR hype_predictions） ──
    sector_concept_map = {}  # sector_name → {"stage": ..., "etfs": [...]}
    for hp in mr_result.get("hype_predictions", []):
        cname = hp.get("concept_name", "")
        stage = hp.get("hype_stage", "")
        etfs = hp.get("etf_alternatives", [])
        etf_str = "/".join(e.get("name", "") for e in etfs[:2]) if etfs else ""
        if cname:
            sector_concept_map[cname] = {"stage": stage, "etfs": etf_str}

    # ── 字体 ──
    font_path = "C:/Windows/Fonts/msyh.ttc"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/simhei.ttf"

    def mkfp(sz, bold=False):
        return fm.FontProperties(fname=font_path, size=sz, weight='bold' if bold else 'normal')

    fp9=mkfp(9); fp10=mkfp(10); fp10b=mkfp(10,True); fp11=mkfp(11); fp11b=mkfp(11,True)
    fp12b=mkfp(12,True); fp14b=mkfp(14,True); fp18b=mkfp(18,True)

    # ── 构建 fusion 索引（code → fusion item）──
    fusion_map = {}
    for item in (fusion_result or []):
        fusion_map[item.get("code", "")] = item

    # ── 提取推荐股数据 ──
    approved = risk_result.get("approved", [])
    soft_excluded = risk_result.get("soft_excluded", [])

    # 按 fusion 排名排序（fusion_result 已排好序）
    all_codes_ordered = [item.get("code", "") for item in (fusion_result or [])]
    code_set_approved = {s.get("code", "") for s in approved}
    code_set_excluded = {s.get("code", "") for s in soft_excluded}
    risk_map = {}
    for s in approved + soft_excluded:
        risk_map[s.get("code", "")] = s

    # 按 fusion 顺序排列，保留 risk 信息
    all_recs_ordered = []
    for code in all_codes_ordered:
        if code in risk_map:
            all_recs_ordered.append(risk_map[code])
    # 补充 fusion 中没有但 risk 有的
    for s in approved + soft_excluded:
        if s.get("code", "") not in {r.get("code","") for r in all_recs_ordered}:
            all_recs_ordered.append(s)

    rec_codes = [s.get("code", "") for s in all_recs_ordered]

    # 计算支撑压力
    sr_main = {}
    sr_anomaly = {}
    if engine:
        sr_main = _compute_support_resistance(engine, rec_codes)
        anomaly_codes = [p.get("code", "") for p in (breakout_result or {}).get("picks", [])]
        if anomaly_codes:
            sr_anomaly = _compute_support_resistance(engine, anomaly_codes)

    # 构建推荐股行数据
    def _extract_profile_field(code, field_re):
        p = stock_profiles.get(code, "")
        m = re.search(field_re, str(p))
        return m.group(1) if m else ""

    stocks_main = []
    for s in all_recs_ordered:
        code = s.get("code", "")
        name = s.get("name", "")
        # 从 fusion_result 获取准确的 borda/sector/model 数据
        fi = fusion_map.get(code, {})
        borda = fi.get("borda_score", s.get("borda_score", s.get("final_score", 0)))
        sector = fi.get("sector", s.get("sector", ""))
        model_count = fi.get("model_count", 0)
        total_models = len(active_models) if active_models else 3
        consensus = f"{model_count}/{total_models}"
        # 投票来源
        rec_by = fi.get("recommended_by", [])
        voters_str = ",".join(f"{r['model']}#{r['rank']}" for r in rec_by) if rec_by else ""

        # 推荐理由：从 fusion all_reasonings 或 risk core_logic 提取
        reasonings = fi.get("all_reasonings", [])
        if reasonings:
            # 取第一条理由，去掉 [model] 前缀
            reason_text = str(reasonings[0])
            if reason_text.startswith("[") and "]" in reason_text:
                reason_text = reason_text[reason_text.index("]")+1:].strip()
        else:
            reason_text = s.get("core_logic", "")
            if reason_text.startswith("[") and "]" in reason_text:
                reason_text = reason_text[reason_text.index("]")+1:].strip()
            # 截断到第一个分号（去掉后续模型理由）
            if ";" in reason_text:
                reason_text = reason_text[:reason_text.index(";")]

        # 风险提示：从 all_reasonings 中智能提取风险/看空句子
        _RISK_KW = [
            "风险", "回调", "追高", "高估", "泡沫", "减持", "质押", "商誉",
            "亏损", "空方", "看空", "下跌", "见顶", "超买", "过热", "利空",
            "不确定", "谨慎", "回撤", "波动", "拖累", "承压", "困境",
            "流出", "偏高", "透支", "兑现", "获利了结", "利好出尽",
        ]
        risk_parts = []
        # 1) 从 risk_flags 提取（如果 LLM 给了个性化内容）
        risk_flags = s.get("risk_flags", [])
        if risk_flags:
            risk_parts.extend(str(f) for f in risk_flags[:3])
        # 2) 从 all_reasonings 提取包含风险关键词的句段
        if not risk_parts:
            for reason_str in reasonings:
                r_clean = str(reason_str)
                if r_clean.startswith("[") and "]" in r_clean:
                    r_clean = r_clean[r_clean.index("]")+1:].strip()
                # 按句号/逗号分割，找包含风险关键词的片段
                for sep in ["。", "，", "；", "，但", "但"]:
                    r_clean = r_clean.replace(sep, "|")
                for frag in r_clean.split("|"):
                    frag = frag.strip()
                    if frag and any(kw in frag for kw in _RISK_KW):
                        if frag not in risk_parts and len(frag) > 4:
                            risk_parts.append(frag)
                if len(risk_parts) >= 2:
                    break
        # 3) 软排除原因
        if code in code_set_excluded:
            for ex in soft_excluded:
                if ex.get("code") == code:
                    reason = ex.get("reason", "")
                    if reason and reason not in risk_parts:
                        risk_parts.insert(0, reason)
                    break
        risk_text = "；".join(risk_parts[:3]) if risk_parts else ""

        # 板块+概念+ETF
        sector_info = sector
        cm = sector_concept_map.get(sector, {})
        if cm:
            stage = cm.get("stage", "")
            etf = cm.get("etfs", "")
            if stage:
                sector_info += f"|{stage}"
            if etf:
                sector_info += f"|{etf}"
        else:
            # 模糊匹配：板块名可能是概念名的子串
            for cname, cdata in sector_concept_map.items():
                if cname in sector or sector in cname:
                    stage = cdata.get("stage", "")
                    etf = cdata.get("etfs", "")
                    if stage:
                        sector_info += f"|{stage}"
                    if etf:
                        sector_info += f"|{etf}"
                    break

        # 位置
        chg20 = _extract_profile_field(code, r"近20日=([+\-]?[\d.]+%)")
        position = f"20日{chg20}" if chg20 else ""

        stocks_main.append((code, name, int(borda), consensus, sector_info, reason_text, risk_text, position))

    # 构建异动股行数据
    anomaly_rows = []
    for p in (breakout_result or {}).get("picks", []):
        code = p.get("code", "")
        name = p.get("name", "")
        score = p.get("score", 0)
        votes = p.get("votes", 0)
        theme = p.get("theme", "")
        position = p.get("position", "")
        anomaly_rows.append((code, name, score, votes, theme, position))

    n_main = len(stocks_main)
    n_anomaly = len(anomaly_rows)
    has_anomaly = n_anomaly > 0

    # ── 表格布局 ──
    # 列定义: (x, width, max_chars) — max_chars 控制每行文字截断字数
    # 总宽 18.0，充分利用空间
    cols_main_def = [
        (0.00, 0.30, 3),    # 0: #
        (0.30, 0.75, 6),    # 1: 代码
        (1.05, 0.90, 5),    # 2: 名称
        (1.95, 1.80, 12),   # 3: 板块/概念/ETF
        (3.75, 0.50, 4),    # 4: Borda
        (4.25, 0.45, 3),    # 5: 共识
        (4.70, 0.70, 6),    # 6: 现价
        (5.40, 0.65, 6),    # 7: 支撑1
        (6.05, 0.65, 6),    # 8: 支撑2
        (6.70, 0.65, 6),    # 9: 压力1
        (7.35, 0.65, 6),    # 10: 压力2
        (8.00, 1.00, 8),    # 11: 位置
        (9.00, 4.50, 30),   # 12: 推荐理由
        (13.50,4.50, 30),   # 13: 风险提示
    ]
    cols_main = [(x, w) for x, w, _ in cols_main_def]
    max_chars_main = [mc for _, _, mc in cols_main_def]
    hdrs_main = ["#","代码","名称","板块/概念/ETF","Borda","共识",
                 "现价","支撑1","支撑2","压力1","压力2","位置","推荐理由","风险提示"]

    cols_anom_def = [
        (0.00, 0.30, 3),    # 0: #
        (0.30, 0.75, 6),    # 1: 代码
        (1.05, 0.90, 5),    # 2: 名称
        (1.95, 0.50, 3),    # 3: 评分
        (2.45, 0.45, 3),    # 4: 票数
        (2.90, 3.00, 20),   # 5: 事件题材
        (5.90, 0.80, 7),    # 6: 现价
        (6.70, 0.75, 7),    # 7: 支撑1
        (7.45, 0.75, 7),    # 8: 支撑2
        (8.20, 0.75, 7),    # 9: 压力1
        (8.95, 0.75, 7),    # 10: 压力2
        (9.70, 1.30, 10),   # 11: 位置
    ]
    cols_anom = [(x, w) for x, w, _ in cols_anom_def]
    max_chars_anom = [mc for _, _, mc in cols_anom_def]
    hdrs_anom = ["#","代码","名称","评分","票数","事件题材",
                 "现价","支撑1","支撑2","压力1","压力2","位置"]

    total_w = 18.0
    ch = 0.80   # 行高（容纳推荐理由+风险提示多行）
    hh = 0.52   # 表头高
    gap = 0.7; title_h = 1.0; sub_h = 0.50

    fig_h = title_h + sub_h + hh + n_main * ch + 0.8
    if has_anomaly:
        fig_h += gap + sub_h + hh + n_anomaly * ch
    fig_w = total_w + 0.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.1, total_w + 0.3)
    ax.set_ylim(0, fig_h)
    ax.axis('off')
    ax.invert_yaxis()

    # ── 辅助函数 ──
    def _pos_color(txt):
        if any(k in txt for k in ['高位','超买','见顶','冲顶']):
            return '#c0392b', fp10b
        if any(k in txt for k in ['底部','低位','横盘','蓄势','整理','盘整']):
            return '#1e8449', fp10b
        return '#2471a3', fp10b

    def _style_main(j, val):
        if j in (7,8):  return '#1e8449', fp11b   # 支撑
        if j in (9,10): return '#c0392b', fp11b    # 压力
        if j == 6:  return '#2c3e50', fp11b        # 现价
        if j == 13 and val: return '#c0392b', mkfp(8)  # 风险提示
        if j == 12: return '#2c3e50', mkfp(8)      # 推荐理由
        if j == 11: return _pos_color(val)         # 位置
        if j == 3:  return '#2c3e50', mkfp(8.5)    # 板块/概念/ETF
        if j == 4:
            bv = int(val) if val.isdigit() else 0
            c = '#196f3d' if bv>=40 else '#1a5276' if bv>=20 else '#7f8c8d'
            return c, fp11b
        if j == 5:
            total_m = str(len(active_models)) if active_models else "3"
            is_full = val.startswith(total_m + "/")
            return ('#196f3d' if is_full else '#7f8c8d'), (fp11b if is_full else fp11)
        return '#2c3e50', fp11

    def _style_anom(j, val):
        if j in (7,8):  return '#1e8449', fp11b
        if j in (9,10): return '#c0392b', fp11b
        if j == 6:  return '#2c3e50', fp11b
        if j == 5:  return '#8e44ad', mkfp(9)   # 事件题材
        if j == 3:  return '#e67e22', fp11b
        if j == 4:  return '#2471a3', fp11b
        if j == 11: return _pos_color(val)
        return '#2c3e50', fp11

    def _draw_hdr(y, cols, hdrs, bg='#1a5276'):
        for j,(x,w) in enumerate(cols):
            r = Rect((x,y),w,hh, facecolor=bg, edgecolor='#0e3d5a', linewidth=0.5)
            ax.add_patch(r)
            ax.text(x+w/2, y+hh/2, hdrs[j], fontproperties=fp11b,
                    ha='center', va='center', color='white')

    def _fmt_sr(d, key):
        return f"{d[key]:.2f}" if key in d and d[key] else "-"

    def _wrap_text(txt, max_ch):
        """将长文本按 max_ch 字符截断并插入换行"""
        if not txt or len(txt) <= max_ch:
            return txt
        lines = []
        while txt:
            lines.append(txt[:max_ch])
            txt = txt[max_ch:]
            if len(lines) >= 3:  # 最多3行
                if txt:
                    lines[-1] = lines[-1][:-1] + ".."
                break
        return "\n".join(lines)

    def _draw_cell(x, y, w, h, text, color, fp_cell, bg, max_ch, ha='center'):
        """绘制单元格，文字自动换行"""
        r = Rect((x,y),w,h, facecolor=bg, edgecolor='#d5d8dc', linewidth=0.3)
        ax.add_patch(r)
        wrapped = _wrap_text(text, max_ch)
        # 左对齐时加 padding
        tx = x + 0.05 if ha == 'left' else x + w/2
        ax.text(tx, y+h/2, wrapped, fontproperties=fp_cell,
                ha=ha, va='center', color=color,
                linespacing=1.15, clip_on=True)

    # ── 标题 ──
    ax.text(total_w/2, 0.2, "A股多智能体选股系统 — 推荐股 + 异动股 完整分析",
            fontproperties=fp18b, ha='center', va='top', color='#1a5276')
    subtitle = (f"{datetime.now().strftime('%Y-%m-%d')}  |  {models_str}  "
                f"|  {llm_mode}  |  绿=支撑  红=压力")
    ax.text(total_w/2, 0.58, subtitle,
            fontproperties=fp10, ha='center', va='top', color='#7f8c8d')

    # ── 表1: 推荐股 ──
    y1 = title_h
    ax.text(0.0, y1+0.06, f"  Borda融合推荐 TOP{n_main}（多模型共识选股）",
            fontproperties=fp14b, color='#1a5276', va='top')
    y1 += sub_h
    _draw_hdr(y1, cols_main, hdrs_main)
    y1 += hh

    total_m = str(len(active_models)) if active_models else "3"
    for i,(code,name,borda,cons,sector,reason,risk,pos) in enumerate(stocks_main):
        y = y1 + i * ch
        d = sr_main.get(code, {})
        cells = [str(i+1),code,name,sector,str(borda),cons,
                 _fmt_sr(d,'price'),_fmt_sr(d,'s1'),_fmt_sr(d,'s2'),
                 _fmt_sr(d,'r1'),_fmt_sr(d,'r2'), pos, reason, risk]
        full_cons = cons == f"{total_m}/{total_m}"
        if risk and any(k in risk for k in ['排除','ST','退市']): bg='#fef5f5'
        elif full_cons: bg='#eafaf1'
        else: bg='#f8f9fa' if i%2==0 else '#ffffff'

        for j,(x,w) in enumerate(cols_main):
            color, f = _style_main(j, cells[j])
            ha = 'left' if j in (3,12,13) else 'center'  # 板块/投票/风控左对齐
            _draw_cell(x, y, w, ch, cells[j], color, f, bg, max_chars_main[j], ha)

    # ── 表2: 异动股 ──
    if has_anomaly:
        y2 = y1 + n_main * ch + gap * 0.3
        ax.plot([0, total_w], [y2, y2], color='#bdc3c7', linewidth=1.5, linestyle='--')
        y2 += gap * 0.2
        ax.text(0.0, y2+0.06,
                f"  异动爆发推荐 TOP{n_anomaly}（事件驱动短线）   仓位<=5%/只，严格止损",
                fontproperties=fp14b, color='#8e44ad', va='top')
        y2 += sub_h
        _draw_hdr(y2, cols_anom, hdrs_anom, bg='#6c3483')
        y2 += hh

        for i,(code,name,score,votes,theme,pos) in enumerate(anomaly_rows):
            y = y2 + i * ch
            d = sr_anomaly.get(code, {})
            cells = [str(i+1),code,name,str(score),f"{votes}/{total_m}",theme,
                     _fmt_sr(d,'price'),_fmt_sr(d,'s1'),_fmt_sr(d,'s2'),
                     _fmt_sr(d,'r1'),_fmt_sr(d,'r2'), pos]
            bg = '#f3e8ff' if votes >= 2 else ('#faf5ff' if i%2==0 else '#ffffff')

            for j,(x,w) in enumerate(cols_anom):
                color, f = _style_anom(j, cells[j])
                ha = 'left' if j == 5 else 'center'
                _draw_cell(x, y, w, ch, cells[j], color, f, bg, max_chars_anom[j], ha)
        footer_y = y2 + n_anomaly * ch + 0.15
    else:
        footer_y = y1 + n_main * ch + 0.15

    # ── 页脚 ──
    ax.text(0.0, footer_y,
        "  绿底=强共识  红底=风控预警  紫底=双模型确认异动  "
        "支撑1=MA20/近期低点  支撑2=MA60  压力1=20日高  压力2=60日高",
        fontproperties=fp10, color='#7f8c8d')
    ax.text(0.0, footer_y+0.3,
        "  本表仅为AI分析参考，不构成投资建议。股市有风险，投资需谨慎。",
        fontproperties=fp9, color='#aab7b8')

    # ── 保存 ──
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(output_dir, f"总览分析_{ts}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.3)
    plt.close()
    return out_path
