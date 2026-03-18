#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 主程序入口（重构版：模型纵向并行）

系统架构:
  数据采集 → 板块初筛 → N模型并行(每模型内E1→E6顺序+intra辩论) → Borda融合 → 风控 → 报告

运行方式:
  python main.py                              # 全流程运行（自动选板块）
  python main.py --sectors 人工智能,机器人    # 指定板块运行
  python main.py --stocks 000988,002506,000545 # 指定个股列表（跳过板块筛选）
  python main.py --panel 3                    # 使用3个模型并行
  python main.py --demo                       # 演示模式（模拟数据，不调用真实行情API）
"""

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

# Windows 控制台 UTF-8 输出支持
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 依赖检查
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("请安装依赖: pip install -r requirements.txt")
    sys.exit(1)

from config import load_config, Config
from data_engine import DataEngine
from llm_client import LLMClient
from stock_agents import (
    MarketRadar,
    SectorPicker,
    GlobalThemeAdvisor,
    SectorScreener,
    ModelTask,
    RiskController,
    EventAnalyst,
    BreakoutAnalyst,
)
from fusion import borda_fusion
from work_logger import WorkLogger
from report_generator import generate_report
from memory import StockMemory

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_BASE_DIR, "output")        # output/ 根
DATA_DIR    = os.path.join(OUTPUT_DIR, "data")          # output/data/  中间JSON
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")       # output/reports/ 报告
LOGS_DIR    = os.path.join(OUTPUT_DIR, "logs")          # output/logs/
for _d in (DATA_DIR, REPORTS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

TODAY = datetime.now().strftime("%Y%m%d")


# ===================================================================== #
#  工具函数                                                              #
# ===================================================================== #

def save_result(data: dict, filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


def print_banner():
    print("\n" + "=" * 70)
    print("   A股多智能体选股系统（模型纵向并行版）")
    print("   Multi-Model Parallel Stock Selector with Borda Fusion")
    print(f"   分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)


def print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _consensus_level(model_count: int) -> str:
    if model_count >= 3:
        return f"强共识({model_count}模型)"
    elif model_count >= 2:
        return f"弱共识({model_count}模型)"
    return "单一模型"


def run_chief_strategist(
    llm: LLMClient,
    fusion_top: List[Dict],
    model_results: List[Dict],
    mr_result: Optional[Dict] = None,
) -> Optional[List[Dict]]:
    """
    首席策略师综合判断：对Borda融合结果进行定性审视和调整。

    返回调整后的排名列表，失败时返回None（使用原始Borda排名）。
    """
    try:
        # ── 构建市场环境上下文 ──
        market_ctx = ""
        if mr_result:
            env = mr_result.get("market_environment", {})
            sent = mr_result.get("sentiment", {})
            market_ctx = (
                f"【市场环境】\n"
                f"大盘趋势: {env.get('trend', 'N/A')}\n"
                f"情绪温度: {sent.get('temperature', 'N/A')} | 周期: {sent.get('cycle_phase', '')}\n"
                f"策略提示: {sent.get('strategy_hint', '')}\n"
            )

        # ── 构建融合排名表 ──
        ranking_lines = []
        for item in fusion_top:
            sources = ", ".join(
                f"{r['model']}(#{r['rank']})" for r in item.get("recommended_by", [])
            )
            ranking_lines.append(
                f"  {item['rank']}. {item['code']} {item['name']}  "
                f"板块={item.get('sector', '')}  Borda={item['borda_score']:.0f}  "
                f"模型数={item['model_count']}  来源=[{sources}]"
            )
        ranking_table = "\n".join(ranking_lines)

        # ── 识别模型间分歧 ──
        disagreements = []
        code_rankings: Dict[str, List] = {}
        for mr in model_results:
            model_name = mr.get("model", "unknown")
            for pick in mr.get("picks", []):
                code = pick.get("code", "")
                rank = pick.get("rank", 99)
                if code:
                    code_rankings.setdefault(code, []).append((model_name, rank))

        for item in fusion_top:
            code = item["code"]
            ranks = code_rankings.get(code, [])
            if len(ranks) >= 2:
                rank_values = [r[1] for r in ranks]
                spread = max(rank_values) - min(rank_values)
                if spread >= 3:
                    rank_detail = ", ".join(f"{m}=#{r}" for m, r in ranks)
                    disagreements.append(
                        f"  {code} {item['name']}: 排名分歧大(极差={spread}) [{rank_detail}]"
                    )

        disagreement_text = ""
        if disagreements:
            disagreement_text = "\n【模型间主要分歧】\n" + "\n".join(disagreements[:8])

        # ── 构建 code 列表用于验证 ──
        valid_codes = {item["code"] for item in fusion_top}

        # ── 调用 LLM ──
        system_msg = (
            "你是首席投资策略师。你的任务是对量化融合排名进行定性审视，弥补纯投票机制的盲区。"
            "你可以调整排名顺序，但不能新增融合结果中没有的股票。"
        )

        user_msg = (
            f"{market_ctx}\n"
            f"【Borda融合排名（共{len(fusion_top)}只）】\n{ranking_table}\n"
            f"{disagreement_text}\n\n"
            f"请对以上融合排名进行定性审视，可根据市场环境、板块轮动、分歧信号等因素调整排序。\n"
            f"要求：\n"
            f"1. 只能重新排列上述股票，不能新增或删除\n"
            f"2. 返回全部{len(fusion_top)}只股票的调整后排名\n"
            f"3. 以纯JSON格式输出，不要包含markdown代码块标记：\n"
            f'{{\n'
            f'  "adjusted_ranking": [\n'
            f'    {{"code": "...", "name": "...", "adjustment": "升|降|维持", "reason": "调整理由"}}\n'
            f'  ],\n'
            f'  "strategic_commentary": "整体策略点评（50字以内）",\n'
            f'  "top_conviction": "最高信心标的代码及理由"\n'
            f'}}'
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        raw = llm.call_primary(messages, temperature=0.2)
        if not raw:
            return None

        # ── 解析 JSON ──
        # 去除可能的 markdown 代码块包裹
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]  # 去掉第一行 ```json
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        result = json.loads(text)
        adjusted_ranking = result.get("adjusted_ranking", [])
        if not adjusted_ranking:
            return None

        commentary = result.get("strategic_commentary", "")
        top_conviction = result.get("top_conviction", "")

        # ── 应用调整：按策略师返回的顺序重排 fusion_top ──
        fusion_lookup = {item["code"]: item for item in fusion_top}
        adjusted_list = []
        seen = set()

        for entry in adjusted_ranking:
            code = entry.get("code", "")
            if code in valid_codes and code not in seen:
                item = dict(fusion_lookup[code])  # 浅拷贝
                item["adjustment"] = entry.get("adjustment", "维持")
                item["adjustment_reason"] = entry.get("reason", "")
                item["strategic_commentary"] = commentary
                item["top_conviction"] = top_conviction
                adjusted_list.append(item)
                seen.add(code)

        # 补回策略师遗漏的股票（保持原顺序追加）
        for item in fusion_top:
            if item["code"] not in seen:
                adj_item = dict(item)
                adj_item["adjustment"] = "维持"
                adj_item["adjustment_reason"] = "策略师未提及"
                adj_item["strategic_commentary"] = commentary
                adj_item["top_conviction"] = top_conviction
                adjusted_list.append(adj_item)

        # 更新 rank
        for i, item in enumerate(adjusted_list, 1):
            item["rank"] = i

        if commentary:
            print(f"  策略点评: {commentary}")
        if top_conviction:
            print(f"  最高信心: {top_conviction}")

        return adjusted_list

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  [策略师] 解析失败: {e}")
        return None
    except Exception as e:
        print(f"  [策略师] 异常: {e}")
        return None


def fusion_to_arb_result(final_top: List[Dict]) -> Dict:
    """
    将 borda_fusion() 输出转换为 RiskController.filter() 期望的 arb_result 格式。
    """
    final_picks = []
    for pick in final_top:
        final_picks.append({
            "rank": pick["rank"],
            "code": pick["code"],
            "name": pick["name"],
            "sector": pick["sector"],
            "final_score": pick["borda_score"],
            "consensus_level": _consensus_level(pick["model_count"]),
            "expert_count": pick["model_count"],
            "warnings": [],
            "all_reasonings": pick.get("all_reasonings", []),
            # 保留融合专属字段供报告使用
            "borda_score": pick["borda_score"],
            "avg_score": pick["avg_score"],
            "model_count": pick["model_count"],
            "recommended_by": pick["recommended_by"],
        })
    return {
        "final_picks": final_picks,
        "candidate_pool": {},
        "total_candidates": len(final_top),
    }


def build_expert_summary(model_results: List[Dict]) -> Dict:
    """
    从 model_results 构建向后兼容的 expert_summary，供 PDF 报告使用。
    格式: {E1: {"name": ..., "picks": [...]}, ...}
    """
    expert_summary: Dict = {}
    for model_result in model_results:
        model_name = model_result.get("model", "unknown")
        for expert_id, expert_log in model_result.get("expert_logs", {}).items():
            if expert_id not in expert_summary:
                expert_summary[expert_id] = {"name": expert_id, "picks": []}
            for pick in expert_log.get("picks", []):
                expert_summary[expert_id]["picks"].append({
                    "code": pick.get("code", ""),
                    "name": pick.get("name", ""),
                    "stars": 3.0,
                    "score": pick.get("score", 70),
                    "reasoning": "",
                    "voters": [model_name],
                })
    return expert_summary


def format_final_report(
    risk_result: Dict,
    arb_result: Dict,
    fusion_result: Optional[List[Dict]] = None,
    mr_result: Optional[Dict] = None,
    etf_sector_match: Optional[Dict] = None,
    event_result: Optional[Dict] = None,
    breakout_result: Optional[Dict] = None,
) -> str:
    """生成最终报告文本（含 Borda 评分、推荐来源、市场雷达、ETF推荐）"""
    # 构建融合信息查找表
    fusion_lookup: Dict[str, Dict] = {}
    if fusion_result:
        for item in fusion_result:
            fusion_lookup[item["code"]] = item
    # 也从 arb_result 查
    for pick in arb_result.get("final_picks", []):
        code = pick.get("code", "")
        if code and code not in fusion_lookup:
            fusion_lookup[code] = pick

    lines = [
        "",
        "═" * 70,
        "   A股多智能体选股系统（模型并行版）— 最终推荐报告",
        f"   分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "═" * 70,
    ]

    # ── 市场雷达概览 ──────────────────────────────────────────────
    if mr_result:
        env = mr_result.get("market_environment", {})
        sent = mr_result.get("sentiment", {})
        lines.append("\n  【市场雷达】")
        lines.append(f"  大盘环境: {env.get('trend', 'N/A')}")
        idx_s = env.get("index_summary", "")
        if idx_s:
            lines.append(f"  指数概况: {idx_s}")
        lines.append(
            f"  情绪温度: {sent.get('temperature', 'N/A')} | "
            f"周期: {sent.get('cycle_phase', '')} | "
            f"分数: {sent.get('score', 'N/A')}"
        )
        hint = sent.get("strategy_hint", "")
        if hint:
            lines.append(f"  策略提示: {hint}")
        adv = mr_result.get("operation_advice", "")
        if adv:
            lines.append(f"  操作建议: {adv[:200]}")

        # 概念炒作预判
        hypes = mr_result.get("hype_predictions", [])
        if hypes:
            lines.append("\n  【概念炒作预判】")
            lines.append(
                f"  {'排名':<4} {'概念名称':<14} {'阶段':<10} "
                f"{'信心':<6} {'信号来源':<20} {'ETF替代':<20}"
            )
            lines.append("  " + "─" * 75)
            for h in hypes[:8]:
                etf_str = ""
                etf_alts = h.get("etf_alternatives", [])
                if etf_alts:
                    etf_str = ", ".join(
                        f"{e.get('name', '')}({e.get('code', '')})" for e in etf_alts[:2]
                    )
                signal = h.get("signal_sources", "")[:18]
                lines.append(
                    f"  {h['rank']:<4} {h['concept_name']:<14} {h['hype_stage']:<10} "
                    f"{h['confidence']:<6.0f} {signal:<20} {etf_str:<20}"
                )
        lines.append("")

    approved = risk_result.get("approved", [])
    soft_excluded = risk_result.get("soft_excluded", [])

    if not approved and not soft_excluded:
        lines.append("  [警告] 未产生有效推荐，请检查数据和模型配置")
        return "\n".join(lines)

    # 合并所有股票，统一按 Borda 分排序
    all_stocks = []
    for stock in approved:
        code = stock.get("code", "")
        fi = fusion_lookup.get(code, {})
        all_stocks.append({
            "code": code,
            "name": stock.get("name", ""),
            "sector": stock.get("sector", fi.get("sector", "")),
            "borda": fi.get("borda_score", stock.get("final_score", 0)),
            "model_count": fi.get("model_count", stock.get("model_count", 0)),
            "risk_level": stock.get("risk_level", ""),
            "position_advice": stock.get("position_advice", ""),
            "is_excluded": False,
            "reason": "",
            "stock_data": stock,
            "fusion_data": fi,
        })
    for exc in soft_excluded:
        code = exc.get("code", "")
        fi = fusion_lookup.get(code, {})
        all_stocks.append({
            "code": code,
            "name": exc.get("name", ""),
            "sector": fi.get("sector", ""),
            "borda": fi.get("borda_score", 0),
            "model_count": fi.get("model_count", 0),
            "risk_level": "⚠️",
            "position_advice": "",
            "is_excluded": True,
            "reason": exc.get("reason", "风控排除"),
            "stock_data": exc,
            "fusion_data": fi,
        })
    all_stocks.sort(key=lambda x: x["borda"], reverse=True)

    total = len(all_stocks)
    n_approved = len(approved)
    n_excluded = len(soft_excluded)
    lines.append(f"\n  共 {total} 只股票（按Borda评分排序 | 风控通过 {n_approved} 只，风控提示 {n_excluded} 只）\n")

    # ── 网格表格（处理中文宽度） ──────────────────────────────
    def _display_width(s: str) -> int:
        """计算字符串显示宽度（中文占2，ASCII占1）"""
        w = 0
        for ch in s:
            w += 2 if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f' or '\uff00' <= ch <= '\uffef' else 1
        return w

    def _pad(s: str, width: int, align: str = "left") -> str:
        """按显示宽度填充字符串"""
        dw = _display_width(s)
        pad = max(0, width - dw)
        if align == "center":
            left = pad // 2
            right = pad - left
            return " " * left + s + " " * right
        elif align == "right":
            return " " * pad + s
        return s + " " * pad

    # 列宽定义
    W = [4, 8, 10, 10, 6, 4, 8, 14, 30]  # 排名/代码/名称/板块/Borda/模型/风险/仓位/风控提示
    H = ["排名", "代码", "名称", "板块", "Borda", "模型", "风险", "仓位建议", "风控提示"]
    HA = ["center"] * 6 + ["center", "center", "center"]  # 表头对齐

    top_line  = "  ┌" + "┬".join("─" * w for w in W) + "┐"
    sep_line  = "  ├" + "┼".join("─" * w for w in W) + "┤"
    bot_line  = "  └" + "┴".join("─" * w for w in W) + "┘"

    def _row(cells, aligns=None):
        if aligns is None:
            aligns = ["center", "center", "left", "left", "center", "center", "left", "left", "left"]
        parts = []
        for i, (cell, w) in enumerate(zip(cells, W)):
            parts.append(_pad(str(cell), w, aligns[i] if i < len(aligns) else "left"))
        return "  │" + "│".join(parts) + "│"

    lines.append(top_line)
    lines.append(_row(H, HA))
    lines.append(sep_line)

    for rank_counter, item in enumerate(all_stocks, 1):
        code = item["code"]
        name = item["name"][:5]
        sector = item["sector"][:5]
        borda = f"{item['borda']:.0f}"
        model_count = str(item["model_count"])

        if item["is_excluded"]:
            risk_str = "⚠风控"
            position = "—"
            risk_note = item["reason"][:14]
        else:
            risk_str = item["risk_level"][:4]
            position = item["position_advice"][:7]
            stock = item["stock_data"]
            flags = stock.get("risk_flags", [])
            risk_note = "; ".join(flags)[:14] if flags else ""

        lines.append(_row(
            [str(rank_counter), code, name, sector, borda, model_count, risk_str, position, risk_note],
            ["center", "center", "left", "left", "center", "center", "left", "left", "left"],
        ))

    lines.append(bot_line)

    # 详情（按 Borda 分排序统一展示）
    lines.append("\n  【个股详情】")
    for item in all_stocks:
        code = item["code"]
        name = item["stock_data"].get("name", item["name"])
        fi = item["fusion_data"]
        lines.append(f"\n  ▸ {code} {name}" + ("  ⚠️风控提示" if item["is_excluded"] else ""))

        # Borda 来源信息
        recommended_by = fi.get("recommended_by", [])
        if recommended_by:
            by_str = "、".join(
                f"{r['model']}(#{r['rank']})"
                for r in recommended_by
            )
            lines.append(f"    推荐来源: {by_str}  Borda分={item['borda']:.0f}")

        if item["is_excluded"]:
            lines.append(f"    风控提示: {item['reason']}")
        else:
            stock = item["stock_data"]
            core = stock.get("core_logic", "")
            if core:
                lines.append(f"    核心逻辑: {core}")

            entry = stock.get("entry_point", "")
            if entry:
                lines.append(f"    买入时机: {entry}")

            stop = stock.get("stop_loss", "")
            if stop:
                lines.append(f"    止损参考: {stop}")

            flags = stock.get("risk_flags", [])
            if flags:
                lines.append(f"    风险提示: {'; '.join(flags)}")

    # 首席策略师点评
    strategist_commentary = ""
    strategist_conviction = ""
    has_adjustments = False
    for item in all_stocks:
        fi = item["fusion_data"]
        if fi.get("strategic_commentary"):
            strategist_commentary = fi["strategic_commentary"]
        if fi.get("top_conviction"):
            strategist_conviction = fi["top_conviction"]
        if fi.get("adjustment") and fi["adjustment"] != "维持":
            has_adjustments = True

    if strategist_commentary or has_adjustments:
        lines.append("\n" + "─" * 70)
        lines.append("  【首席策略师点评】")
        if strategist_commentary:
            lines.append(f"  策略总评: {strategist_commentary}")
        if strategist_conviction:
            lines.append(f"  最高信心: {strategist_conviction}")
        # 列出调整过的股票
        adj_items = [
            item for item in all_stocks
            if item["fusion_data"].get("adjustment") and item["fusion_data"]["adjustment"] != "维持"
        ]
        if adj_items:
            lines.append("  排名调整:")
            for item in adj_items:
                fi = item["fusion_data"]
                lines.append(
                    f"    {item['code']} {item['name']}  "
                    f"{fi['adjustment']}  {fi.get('adjustment_reason', '')}"
                )

    # 组合建议
    portfolio = risk_result.get("portfolio_advice", "")
    timing = risk_result.get("market_timing", "")
    if portfolio or timing:
        lines.append("\n" + "─" * 70)
        lines.append("  【组合与时机建议】")
        if timing:
            lines.append(f"  建仓时机: {timing}")
        if portfolio:
            lines.append(f"  组合建议: {portfolio}")

    # ── ETF 替代推荐 ─────────────────────────────────────────────
    if etf_sector_match:
        lines.append("\n" + "─" * 70)
        lines.append("  【ETF替代推荐（选不出个股时可先蹭板块涨幅）】")
        for sector, etfs in etf_sector_match.items():
            etf_str = " / ".join(f"{e['name']}({e['code']})" for e in etfs[:2])
            lines.append(f"    {sector} → {etf_str}")

    # ── 事件驱动推荐 ──
    if event_result and event_result.get("events"):
        lines.append("\n" + "═" * 70)
        lines.append("  【事件驱动推荐（前瞻预判）】")
        for evt in event_result["events"]:
            lines.append(f"\n  ■ {evt.get('event','')}")
            lines.append(f"    因果链: {evt.get('causal_chain','')[:80]}")
            lines.append(f"    时效: {evt.get('timeframe','')}  确定性: {evt.get('certainty',0)}%  模型数: {evt.get('model_votes',0)}")
            bens = evt.get("beneficiaries", [])
            if bens:
                for b in bens[:5]:
                    lines.append(f"    → {b.get('code','')} {b.get('name','')}  评分={b.get('score',0)}  {b.get('logic','')[:30]}")

    # ── 异动爆发推荐 ──
    if breakout_result and breakout_result.get("picks"):
        lines.append("\n" + "═" * 70)
        lines.append("  【异动爆发推荐（技术短线）】")
        lines.append("  ⚠ 短线激进策略，仓位≤5%/只，严格止损")
        for p in breakout_result["picks"][:10]:
            lines.append(
                f"    {p.get('code','')} {p.get('name',''):<8}  "
                f"评分={p.get('score',0)}  题材={p.get('theme','')}  "
                f"位置={p.get('position','')}  "
                f"票={p.get('votes',0)}"
            )

    lines.append("\n" + "═" * 70)
    lines.append("  [免责声明] 本报告仅为AI分析参考，不构成投资建议。")
    lines.append("             股市有风险，投资需谨慎。请结合自身情况判断。")
    lines.append("═" * 70 + "\n")

    return "\n".join(lines)


# ===================================================================== #
#  演示模式（模拟数据）                                                  #
# ===================================================================== #

def run_demo_mode(cfg: Config):
    """
    演示模式：使用模拟数据，测试单模型 ModelTask 流程（不调用真实行情API）。
    """
    print_banner()
    print("\n  [演示模式] 使用模拟数据，测试模型并行流程...")

    llm = LLMClient(cfg)
    available = list(cfg.providers.keys())
    print(f"\n  可用LLM: {available}")

    if not available:
        print("\n  [错误] 未配置任何LLM提供商，请检查 .env 文件")
        return

    # 初始化工作日志
    logger = WorkLogger(log_dir=LOGS_DIR)

    # 模拟股票数据
    demo_profiles = {
        "300750": (
            "═══ 股票画像: 300750 宁德时代 ═══\n"
            "所属板块: 新能源 | 分析日期: 2026-03-03\n"
            "PE(TTM)=28.5 | PB=3.2 | 总市值=6800亿 | 今日换手率=2.1%\n\n"
            "【日线详情（近60日）】\n"
            "当前价=230.5 | 5日: +6.2% | 10日: +12.5% | 20日: +18.3%\n"
            "日线均线: MA5=225 MA10=218 MA20=205 MA60=198 → 多头排列\n"
            "MACD: DIF=1.25 DEA=0.98 柱=0.54 [零轴上方多头，柱体扩张]\n"
            "RSI: RSI6=68.5 RSI14=62.3 [正常区间]\n"
            "成交量: 5日均量/20日均量=1.65x | 今日量比=2.1x\n"
            "【基本面摘要】\n"
            "营收增长=28.5% | 净利润增长=22.1% | ROE=16.3% | 毛利率=22.8%"
        ),
        "000858": (
            "═══ 股票画像: 000858 五粮液 ═══\n"
            "所属板块: 白酒 | 分析日期: 2026-03-03\n"
            "PE(TTM)=22.1 | PB=4.5 | 总市值=5200亿 | 今日换手率=0.8%\n\n"
            "【日线详情（近60日）】\n"
            "当前价=135.2 | 5日: +2.1% | 10日: +5.8% | 20日: +8.9%\n"
            "日线均线: MA5=133 MA10=130 MA20=125 MA60=120 → 多头排列\n"
            "MACD: DIF=0.85 DEA=0.72 柱=0.26 [零轴上方多头]\n"
            "RSI: RSI6=58.2 RSI14=55.1 [正常区间]\n"
            "成交量: 5日均量/20日均量=1.25x | 今日量比=1.3x\n"
            "【基本面摘要】\n"
            "营收增长=12.5% | 净利润增长=15.2% | ROE=28.5% | 毛利率=73.2%"
        ),
        "601318": (
            "═══ 股票画像: 601318 中国平安 ═══\n"
            "所属板块: 保险 | 分析日期: 2026-03-03\n"
            "PE(TTM)=9.2 | PB=1.1 | 总市值=8900亿 | 今日换手率=0.5%\n\n"
            "【日线详情（近60日）】\n"
            "当前价=48.5 | 5日: +3.5% | 10日: +7.2% | 20日: +11.5%\n"
            "日线均线: MA5=47 MA10=45 MA20=43 MA60=40 → 强多头排列\n"
            "MACD: DIF=0.55 DEA=0.42 柱=0.26 [零轴上方金叉]\n"
            "RSI: RSI6=65.1 RSI14=60.3 [正常偏强]\n"
            "成交量: 5日均量/20日均量=1.45x | 今日量比=1.8x\n"
            "【基本面摘要】\n"
            "营收增长=8.2% | 净利润增长=18.5% | ROE=12.8% | 毛利率=N/A"
        ),
    }

    # 演示：用主提供商运行 ModelTask
    primary = cfg.primary_provider
    print(f"\n  [演示] 使用 [{primary}] 模型运行 ModelTask（E1→E6 顺序执行+辩论）...")

    task = ModelTask(
        provider_name=primary,
        llm_client=llm,
        config=cfg,
        logger=logger,
        stock_profiles=demo_profiles,
        stock_packages={},
    )

    try:
        result = task.run()
        picks = result.get("picks", [])
        print(f"\n  [演示] ModelTask 完成: {len(picks)} 只推荐")
        for pick in picks[:5]:
            print(
                f"    #{pick.get('rank','')} {pick.get('code','')} "
                f"{pick.get('name','')}  score={pick.get('score',0)}"
                f"  {pick.get('reasoning','')[:60]}"
            )

        # 演示 Borda 融合（单模型时退化为直接排名）
        model_results = [result]
        top5 = borda_fusion(model_results, top_n=5)
        print(f"\n  [演示] Borda 融合 Top5:")
        for item in top5:
            models_str = ", ".join(
                f"{r['model']}(#{r['rank']})" for r in item["recommended_by"]
            )
            print(
                f"    {item['rank']}. {item['code']} {item['name']}"
                f"  Borda={item['borda_score']}"
                f"  [{models_str}]"
            )

    except Exception as e:
        print(f"\n  [错误] ModelTask 失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n  [演示模式完成] 日志已写入 logs/")
    print("  运行 'python main.py' 启动完整选股流程。")


# ===================================================================== #
#  完整流程                                                              #
# ===================================================================== #

def run_full_pipeline(
    cfg: Config,
    sectors: Optional[List[str]] = None,
    panel_size: int = 4,
    debate_rounds: int = 1,   # 保留参数（intra-model辩论已内置于ModelTask）
    max_stocks_per_sector: int = 25,
    parallel_models: bool = True,
    verbose: bool = True,
    models: Optional[List[str]] = None,
    stock_codes: Optional[List[str]] = None,
) -> Dict:
    """
    完整选股流程（模型纵向并行版）:
    S0(板块初筛) → 数据采集 → N模型并行(E1→E6+辩论) → Borda融合 → R(风控) → 报告
    """
    print_banner()
    start_time = time.time()

    # ── Step 0: 初始化 ───────────────────────────────────────────────
    print_section("初始化系统组件")
    llm = LLMClient(cfg)
    engine = DataEngine(tushare_token=cfg.tushare_token, tdx_dir=cfg.tdx_dir)
    logger = WorkLogger(log_dir=LOGS_DIR)

    available_providers = list(cfg.providers.keys())
    if models:
        valid = [m for m in models if m in cfg.providers]
        active_models = valid if valid else llm.get_panel(max_models=panel_size)
    else:
        active_models = llm.get_panel(max_models=panel_size)
    print(f"  可用LLM: {available_providers}")
    print(f"  本次并行模型: {active_models}（共{len(active_models)}个）")

    if not available_providers:
        print("\n  [错误] 未配置任何LLM提供商！请检查 .env 文件中的 API Key。")
        return {}

    # ── Step 0: 市场雷达（大盘+情绪+概念炒作预判+ETF）────────────
    # 非个股直选模式时执行，为后续板块优选提供市场环境上下文
    mr_result: Dict = {}
    etf_data: Dict = {}
    if not stock_codes and not sectors:
        print_section("Step 0: 市场雷达扫描（大盘+情绪+概念炒作预判+ETF）")
        t_step0 = time.time()

        # 并行采集市场雷达所需数据
        market_indices = engine.fetch_market_indices()
        market_sentiment = engine.fetch_market_sentiment()
        market_hot = engine.fetch_market_hot()
        concept_signals = engine.fetch_concept_hype_signals()
        etf_data = engine.fetch_etf_hot()

        # 运行市场雷达分析
        radar = MarketRadar(llm, cfg)
        mr_result = radar.scan(
            market_indices=market_indices,
            market_sentiment=market_sentiment,
            market_hot=market_hot,
            concept_signals=concept_signals,
            etf_data=etf_data,
            panel_size=len(active_models),
            verbose=verbose,
        )
        save_result(mr_result, f"market_radar_{TODAY}.json")

        # 打印市场雷达结果摘要
        env = mr_result.get("market_environment", {})
        sent = mr_result.get("sentiment", {})
        hypes = mr_result.get("hype_predictions", [])
        print(f"\n  [MR] 大盘环境: {env.get('trend', 'N/A')}")
        print(f"  [MR] {env.get('index_summary', '')}")
        print(f"  [MR] 情绪温度: {sent.get('temperature', 'N/A')} | 周期: {sent.get('cycle_phase', '')}")
        print(f"  [MR] {sent.get('strategy_hint', '')}")
        if hypes:
            print(f"\n  [MR] 概念炒作预判Top5:")
            for h in hypes[:5]:
                etf_str = ""
                etf_alts = h.get("etf_alternatives", [])
                if etf_alts:
                    etf_str = " → ETF: " + ", ".join(
                        f"{e.get('name', '')}({e.get('code', '')})" for e in etf_alts[:2]
                    )
                print(
                    f"    {h['rank']}. {h['concept_name']}  "
                    f"阶段={h['hype_stage']}  信心={h['confidence']:.0f}%  "
                    f"票={h['votes']}/{h.get('total_models', '?')}"
                    f"{etf_str}"
                )
        adv = mr_result.get("operation_advice", "")
        if adv:
            print(f"\n  [MR] 操作建议: {adv[:200]}")

        print(f"  ✓ Step 0 完成，耗时 {(time.time()-t_step0)/60:.1f} 分钟")

    # ── 个股直选模式（--stocks）：跳过板块筛选，直接采集指定股票 ────
    top_sectors = []  # 初始化，个股直选模式下为空列表
    if stock_codes:
        print_section("个股直选模式：跳过板块筛选")
        print(f"  指定股票: {len(stock_codes)} 只")
        print(f"  代码列表: {', '.join(stock_codes)}")

        # 直接采集指定股票数据
        print_section("Step 2: 个股数据采集（月线/周线/日线 + 基本面）")
        t_step2 = time.time()

        sector_map: Dict[str, str] = {code: "用户指定" for code in stock_codes}

        print(f"\n[数据] 待分析股票: {len(stock_codes)} 只")
        stock_packages = engine.build_stock_data_package(stock_codes, sector_map)

        if not stock_packages:
            print("\n  [错误] 未获取到股票数据，请检查网络和数据源")
            return {}

        print(f"\n  有效股票数据: {len(stock_packages)} 只")
        print(f"  ✓ Step 2 完成，耗时 {(time.time()-t_step2)/60:.1f} 分钟")

        # 通过 tushare 获取股票名称
        name_map: Dict[str, str] = {}
        try:
            import tushare as ts
            pro = ts.pro_api(cfg.tushare_token)
            basic_df = pro.stock_basic(
                exchange="", list_status="L",
                fields="ts_code,symbol,name",
            )
            for _, row in basic_df.iterrows():
                sym = str(row.get("symbol", "")).strip()
                nm = str(row.get("name", "")).strip()
                if sym and nm:
                    name_map[sym] = nm
            print(f"  [名称映射] 获取 {len(name_map)} 只股票名称")
        except Exception as e:
            print(f"  [警告] 股票名称获取失败: {e}")

        # 构建画像
        print_section("Step 3: 构建股票画像（极简坐标，LLM自主联网研究）")
        t_step3 = time.time()
        stock_profiles: Dict[str, str] = {}
        for code, pkg in stock_packages.items():
            name = name_map.get(code, pkg.get("realtime", {}).get("名称", code))
            profile = engine.build_stock_profile_slim(code, name, pkg)
            stock_profiles[code] = profile

        total_chars = sum(len(v) for v in stock_profiles.values())
        print(f"  完成 {len(stock_profiles)} 只股票画像")
        print(f"  画像总量: {total_chars} 字符 ≈ {total_chars//2} tokens（极简模式）")
        print(f"  ✓ Step 3 完成，耗时 {time.time()-t_step3:.1f}s")
        save_result(
            {k: v for k, v in stock_profiles.items()},
            f"stock_profiles_{TODAY}.json"
        )

    else:
        # ── 正常流程：板块筛选 → 数据采集 → 画像 ──

        # ── Step 1: 板块优选（全面板投票 Top5）────────────────────────
        if sectors:
            top_sectors = sectors
            sector_data = {}
            if not mr_result:
                market_hot = {}
            print(f"\n  [用户指定板块] {top_sectors}")
        else:
            print_section("Step 1: 板块数据采集与优选（SP 全面板投票）")
            t_step1 = time.time()
            sector_data = engine.fetch_sector_overview()
            # 如果 MR 已采集过 market_hot，复用之（避免重复网络请求）
            if not mr_result:
                market_hot = engine.fetch_market_hot()

            # ETF活跃板块信号（反向推导）
            etf_signals = engine.etf_active_sectors(etf_data) if etf_data else []
            if etf_signals:
                print(f"\n  [ETF信号] 活跃板块（ETF反推）:")
                for s in etf_signals[:5]:
                    print(
                        f"    {s['sector']}  "
                        f"{s['etf_name']}({s['etf_code']}) "
                        f"{s['change_pct']:+.2f}%  "
                        f"{s['amount_yi']:.1f}亿  "
                        f"{s['signal']}"
                    )

            picker = SectorPicker(llm, cfg)
            pick_result = picker.pick(
                sector_overview=sector_data,
                market_hot=market_hot,
                panel_size=len(active_models),   # 全部面板模型参与投票
                verbose=verbose,
                mr_result=mr_result if mr_result else None,
                etf_signals=etf_signals if etf_signals else None,
            )
            top_sectors = pick_result.get("sector_names", [])

            # 降级兜底：若 SP 未能产出足够板块，用 S0 补充
            if len(top_sectors) < 3:
                print("\n  [SP降级] 优选结果不足，启用 S0 补充...")
                screener = SectorScreener(llm, cfg)
                screen_result = screener.screen(
                    sector_overview=sector_data,
                    market_hot=market_hot,
                    panel_size=min(3, len(active_models)),
                    verbose=verbose,
                )
                fallback = screen_result.get("sector_names", [])
                for s in fallback:
                    if s not in top_sectors:
                        top_sectors.append(s)
                top_sectors = top_sectors[:5]

            if not top_sectors:
                print("\n  [警告] 板块优选未返回结果，使用默认热门板块")
                top_sectors = ["人工智能", "机器人", "新能源", "半导体", "电网设备"]

            save_result(pick_result, f"sector_screen_{TODAY}.json")
            print(f"  ✓ Step 1 完成，耗时 {(time.time()-t_step1)/60:.1f} 分钟")

        print(f"\n  目标板块（SP初选）: {top_sectors}")

        # ── Step 1.5: 跨市场主题补充（GX，美股/港股联动）────────────
        if not sectors:   # 仅自动流程触发，用户手动指定板块时跳过
            print_section("Step 1.5: 跨市场主题分析（GX 美股/港股联动）")
            t_step15 = time.time()
            global_hot = engine.fetch_global_hot()
            gx_advisor = GlobalThemeAdvisor(llm, cfg)
            gx_result = gx_advisor.advise(
                global_hot=global_hot,
                current_sectors=top_sectors,
                panel_size=len(active_models),
                verbose=verbose,
            )
            save_result(gx_result, f"global_theme_{TODAY}.json")

            extra = gx_result.get("extra_sector")
            if extra and extra not in top_sectors:
                vote_count = gx_result.get("vote_count", 0)
                total_v = gx_result.get("total_models", 1)
                conf = gx_result.get("extra_confidence", 0)
                print(
                    f"\n  [GX] 补充板块: {extra}"
                    f"  ({vote_count}/{total_v} 票，置信={conf:.0f}%)"
                    f"  来源={gx_result.get('source_market','')} · {gx_result.get('source_theme','')}"
                )
                top_sectors = top_sectors + [extra]   # 追加为第6个板块
            else:
                print(f"\n  [GX] 无补充板块（已选板块覆盖或未达多数同意）")
            print(f"  ✓ Step 1.5 完成，耗时 {(time.time()-t_step15)/60:.1f} 分钟")

        print(f"\n  最终目标板块: {top_sectors}")

        # ── Step 2: 数据采集 ─────────────────────────────────────────
        print_section("Step 2: 个股数据采集（月线/周线/日线 + 基本面）")
        t_step2 = time.time()
        data_result = engine.run_full_data_collection(
            sector_names=top_sectors,
            max_stocks_per_sector=max_stocks_per_sector,
        )

        stock_packages = data_result.get("stock_packages", {})
        if not stock_packages:
            print("\n  [错误] 未获取到股票数据，请检查网络和数据源")
            return {}

        print(f"\n  有效股票数据: {len(stock_packages)} 只")
        print(f"  ✓ Step 2 完成，耗时 {(time.time()-t_step2)/60:.1f} 分钟")

        # ── Step 3: 构建股票画像（极简版，供LLM坐标定位）────────────
        print_section("Step 3: 构建股票画像（极简坐标，LLM自主联网研究）")
        t_step3 = time.time()
        stock_profiles: Dict[str, str] = {}

        name_map: Dict[str, str] = {}
        for sname, stocks in data_result.get("sector_components", {}).items():
            for s in stocks:
                code = str(s.get("代码", s.get("stock_code", s.get("code", "")))).strip()
                name = str(s.get("名称", s.get("stock_name", s.get("name", "")))).strip()
                if code and name:
                    name_map[code] = name

        for code, pkg in stock_packages.items():
            name = name_map.get(code, pkg.get("realtime", {}).get("名称", code))
            profile = engine.build_stock_profile_slim(code, name, pkg)
            stock_profiles[code] = profile

        total_chars = sum(len(v) for v in stock_profiles.values())
        print(f"  完成 {len(stock_profiles)} 只股票画像")
        print(f"  画像总量: {total_chars} 字符 ≈ {total_chars//2} tokens（极简模式）")
        print(f"  ✓ Step 3 完成，耗时 {time.time()-t_step3:.1f}s")
        save_result(
            {k: v for k, v in stock_profiles.items()},
            f"stock_profiles_{TODAY}.json"
        )

    # ── 记忆检索：注入历史经验到系统提示词 ──────────────────────
    mem = StockMemory()
    memory_env = {}
    if mr_result:
        env = mr_result.get("market_environment", {})
        sent = mr_result.get("sentiment", {})
        memory_env = {
            "trend": env.get("trend", ""),
            "sentiment_temp": sent.get("temperature", ""),
            "cycle_phase": sent.get("cycle_phase", ""),
            "sentiment_score": sent.get("score", 0),
            "top_sectors": top_sectors,
        }
    reflection_text = mem.get_reflection_prompt(memory_env, mr_result)
    if reflection_text:
        print(f"\n  [记忆] 检索到历史决策参考，已注入提示词")

    # ── Step 4: N模型并行（每模型内E1→E7顺序执行 + intra-model辩论）──
    print_section(
        f"Step 4: {len(active_models)} 模型纵向并行分析"
        f"（每模型内部 E1→E7 顺序执行 + intra-model辩论）"
    )
    t_step4 = time.time()
    _model_start_times: Dict[str, float] = {}
    _model_states: Dict[str, str] = {}  # "running" | "done" | "failed"
    _monitor_stop = threading.Event()

    model_results: List[Dict] = []

    def _monitor_progress():
        """后台监控：每30s打印一次并行模型状态快照"""
        while not _monitor_stop.wait(30):
            now = time.time()
            running = [p for p, s in _model_states.items() if s == "running"]
            done    = [p for p, s in _model_states.items() if s == "done"]
            failed  = [p for p, s in _model_states.items() if s == "failed"]
            if running:
                running_str = ", ".join(
                    f"{p}({now - _model_start_times.get(p, now):.0f}s)"
                    for p in running
                )
                total_elapsed = (now - t_step4) / 60
                print(
                    f"\n  [进度快照 {total_elapsed:.1f}min] "
                    f"运行中: {running_str} | "
                    f"已完成: {len(done)}/{len(active_models)}"
                    + (f" | 失败: {len(failed)}" if failed else ""),
                    flush=True
                )

    def _run_model_task(provider: str) -> Dict:
        _model_states[provider] = "running"
        _model_start_times[provider] = time.time()
        print(f"\n  [{provider}] ▶ 开始分析（{len(stock_profiles)}只候选股）...", flush=True)
        try:
            result = ModelTask(
                provider_name=provider,
                llm_client=llm,
                config=cfg,
                logger=logger,
                stock_profiles=stock_profiles,
                stock_packages=stock_packages,
            ).run()
            elapsed_m = time.time() - _model_start_times.get(provider, t_step4)
            _model_states[provider] = "done"
            print(
                f"\n  [{provider}] ✓ 完成: "
                f"{len(result.get('picks', []))}只推荐，耗时{elapsed_m/60:.1f}分钟",
                flush=True
            )
            return result
        except Exception as e:
            elapsed_m = time.time() - _model_start_times.get(provider, t_step4)
            _model_states[provider] = "failed"
            print(f"\n  [{provider}] ✗ 任务失败 ({elapsed_m/60:.1f}分钟): {e}", flush=True)
            return {"model": provider, "picks": [], "expert_logs": {}, "debate_log": ""}

    if parallel_models and len(active_models) > 1:
        print(f"  [并发] 启动 {len(active_models)} 个模型并行分析...", flush=True)
        monitor_thread = threading.Thread(target=_monitor_progress, daemon=True)
        monitor_thread.start()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(active_models)
        ) as executor:
            futures = {
                executor.submit(_run_model_task, provider): provider
                for provider in active_models
            }
            for future in concurrent.futures.as_completed(futures):
                provider = futures[future]
                try:
                    result = future.result(timeout=1800)
                    model_results.append(result)
                except Exception as e:
                    print(f"\n  [{provider}] 超时/异常: {e}", flush=True)
        _monitor_stop.set()
        monitor_thread.join(timeout=2)
    else:
        for provider in active_models:
            result = _run_model_task(provider)
            model_results.append(result)

    print(f"\n  ✓ Step 4 完成，{len(model_results)}个模型，总耗时 {(time.time()-t_step4)/60:.1f} 分钟")
    save_result({"model_results": model_results}, f"model_results_{TODAY}.json")

    # ── Step 5: Borda Count 跨模型融合 ──────────────────────────────
    print_section("Step 5: Borda Count 跨模型融合（输出 Top25）")
    t_step5 = time.time()
    logger.log("fusion_start", detail={"model_count": len(model_results)})

    final_top10 = borda_fusion(model_results, top_n=25)

    logger.log("fusion_done", detail={"top_n": len(final_top10)})

    print(f"\n  融合结果 Top {len(final_top10)}:")
    for pick in final_top10:
        models_str = ", ".join(
            f"{r['model']}(#{r['rank']})" for r in pick["recommended_by"]
        )
        print(
            f"    {pick['rank']}. {pick['code']} {pick['name']}"
            f"  Borda={pick['borda_score']:.0f}"
            f"  [{models_str}]"
        )
    print(f"  ✓ Step 5 完成，耗时 {time.time()-t_step5:.1f}s")

    save_result({"final_top10": final_top10}, f"fusion_result_{TODAY}.json")

    # ── Step 5.5: 首席策略师综合判断（默认禁用，--strategist 开启）─────
    # 注意：首席策略师会用单一LLM主观调整Borda排名，可能干扰多模型投票的客观性
    if getattr(cfg, '_enable_strategist', False):
        print_section("Step 5.5: 首席策略师综合判断")
        adjusted = run_chief_strategist(llm, final_top10, model_results, mr_result if mr_result else None)
        if adjusted:
            final_top10 = adjusted
            print("  ✓ 策略师已调整排名")
        else:
            print("  [降级] 策略师调整失败，使用原始Borda排名")
    else:
        print("\n  [跳过] Step 5.5 首席策略师（默认禁用，--strategist 开启）")

    # 转换为 RiskController 兼容格式
    arb_result = fusion_to_arb_result(final_top10)

    # ── Step 6: 风控审查 ─────────────────────────────────────────────
    print_section("Step 6: 风险控制审查")
    t_step6 = time.time()
    risk_ctrl = RiskController(llm, cfg)
    risk_result = risk_ctrl.filter(
        arbitration_result=arb_result,
        stock_packages=stock_packages,
        verbose=verbose,
    )
    print(f"  ✓ Step 6 完成，耗时 {(time.time()-t_step6)/60:.1f} 分钟")
    save_result(risk_result, f"risk_result_{TODAY}.json")

    # ── Step 8: 生成 PDF 投顾报告 ────────────────────────────────────
    print_section("Step 8: 生成PDF投顾报告")
    expert_summary = build_expert_summary(model_results)
    pdf_path = ""
    try:
        pdf_path = generate_report(
            risk_result=risk_result,
            arb_result=arb_result,
            expert_summary=expert_summary,
            stock_packages=stock_packages,
            top_sectors=top_sectors,
            n_models=len(active_models),
            output_dir=REPORTS_DIR,
        )
    except Exception as e:
        print(f"  [警告] PDF生成失败: {e}")
        import traceback
        traceback.print_exc()

    # ── Step 9: 事件驱动预判 ──────────────────────────────────────
    print_section("Step 9: 事件驱动预判（新闻→因果链→受益标的）")
    event_result = {"events": []}
    try:
        t_step9 = time.time()
        news = engine.fetch_breaking_news()
        if news:
            print(f"  采集到 {len(news)} 条新闻")
            event_analyst = EventAnalyst(llm, cfg)
            event_result = event_analyst.analyze(news, panel_size=len(active_models))
            events = event_result.get("events", [])
            if events:
                print(f"\n  事件驱动推荐（{len(events)} 个事件）:")
                for evt in events:
                    print(f"    ■ {evt.get('event','')[:40]}")
                    print(f"      因果链: {evt.get('causal_chain','')[:60]}")
                    print(f"      确定性: {evt.get('certainty',0)}%  时效: {evt.get('timeframe','')}")
                    for b in evt.get("beneficiaries", [])[:3]:
                        print(f"      → {b.get('code','')} {b.get('name','')} ({b.get('logic','')[:30]})")
        else:
            print("  未采集到有效新闻")
        print(f"  ✓ Step 9 完成，耗时 {(time.time()-t_step9)/60:.1f} 分钟")
    except Exception as e:
        print(f"  [警告] Step 9 事件驱动分析失败: {e}")

    # ── Step 10: 异动爆发股扫描 ───────────────────────────────────
    print_section("Step 10: 异动爆发股扫描（涨停/放量突破）")
    breakout_result = {"picks": []}
    try:
        t_step10 = time.time()
        breakout_stocks = engine.scan_breakout_stocks()
        if breakout_stocks:
            print(f"  扫描到 {len(breakout_stocks)} 只异动股")
            breakout_analyst = BreakoutAnalyst(llm, cfg)
            breakout_result = breakout_analyst.analyze(breakout_stocks, event_result, panel_size=len(active_models))
            picks = breakout_result.get("picks", [])
            if picks:
                print(f"\n  异动推荐 Top{len(picks)}:")
                for p in picks[:10]:
                    print(f"    {p.get('code','')} {p.get('name','')} "
                          f"评分={p.get('score',0)} 题材={p.get('theme','')[:15]} "
                          f"票={p.get('votes',0)}")
        else:
            print("  未扫描到异动股")
        print(f"  ✓ Step 10 完成，耗时 {(time.time()-t_step10)/60:.1f} 分钟")
    except Exception as e:
        print(f"  [警告] Step 10 异动扫描失败: {e}")

    # ── Step 7: 最终报告（文本）──────────────────────────────────────
    print_section("Step 7: 最终报告（含事件驱动+异动推荐）")
    # 匹配选出板块对应的ETF
    etf_sector_match = engine.match_sector_etfs(top_sectors) if top_sectors else {}
    report = format_final_report(
        risk_result, arb_result,
        fusion_result=final_top10,
        mr_result=mr_result if mr_result else None,
        etf_sector_match=etf_sector_match if etf_sector_match else None,
        event_result=event_result if event_result.get("events") else None,
        breakout_result=breakout_result if breakout_result.get("picks") else None,
    )
    print(report)

    report_path = os.path.join(REPORTS_DIR, f"final_report_{TODAY}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ── 存储记忆（供下次运行检索）──────────────────────────────
    try:
        approved = risk_result.get("approved", [])
        soft_excluded = risk_result.get("soft_excluded", [])
        all_recs = []
        for s in approved:
            all_recs.append({"code": s.get("code",""), "name": s.get("name",""),
                            "borda_score": s.get("borda_score", s.get("final_score",0)),
                            "risk_level": s.get("risk_level","")})
        for s in soft_excluded:
            all_recs.append({"code": s.get("code",""), "name": s.get("name",""),
                            "borda_score": 0, "risk_level": "⚠️"})
        mem.store(TODAY, memory_env, all_recs, mr_result)
        print(f"  [记忆] 已存储本次决策（{len(all_recs)}只）")
    except Exception as e:
        print(f"  [记忆] 存储失败: {e}")

    logger.log("pipeline_done", detail={"message": "完整流程结束"})

    elapsed = time.time() - start_time
    print(f"\n  [完成] 总耗时: {elapsed/60:.1f} 分钟")
    print(f"  [文本报告] {report_path}")
    if pdf_path:
        print(f"  [PDF报告]  {pdf_path}")
    print(f"  [工作日志] {logger.log_path}")

    return {
        "sector_screen": top_sectors,
        "model_results": model_results,
        "fusion_top10": final_top10,
        "arbitration": arb_result,
        "risk_result": risk_result,
        "report": report,
        "pdf_path": pdf_path,
        "event_result": event_result,
        "breakout_result": breakout_result,
    }


# ===================================================================== #
#  命令行入口                                                            #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="A股多智能体选股系统 — 模型纵向并行 + Borda 融合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                              # 全自动流程（自动选板块）
  python main.py --sectors 人工智能,机器人    # 指定板块
  python main.py --stocks 000988,002506       # 指定个股（跳过板块筛选）
  python main.py --panel 3                    # 使用3个模型并行
  python main.py --demo                       # 演示模式（测试用）
  python main.py --config                     # 显示当前配置
        """,
    )

    parser.add_argument(
        "--sectors", "-s",
        type=str, default="",
        help="指定分析板块，用逗号分隔（如: 人工智能,机器人）",
    )
    parser.add_argument(
        "--stocks",
        type=str, default="",
        help="指定个股代码列表，逗号分隔（如: 000988,002506,000545）。跳过板块筛选，直接对这些股票分析排序",
    )
    parser.add_argument(
        "--panel", "-p",
        type=int, default=3,
        help="并行模型数量（默认4）",
    )
    parser.add_argument(
        "--models",
        type=str, default="",
        help="指定模型列表，逗号分隔（如: deepseek,kimi,grok）",
    )
    parser.add_argument(
        "--debate", "-d",
        type=int, default=1,
        help="(保留参数，intra-model辩论已内置于ModelTask，此参数不影响行为)",
    )
    parser.add_argument(
        "--max-stocks", "-m",
        type=int, default=25,
        help="每个板块最多分析的股票数量（默认25）",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="关闭模型并行执行（顺序执行）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="演示模式：使用模拟数据测试LLM调用",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="显示当前配置信息",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="详细输出（默认开启）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="安静模式（减少输出）",
    )
    parser.add_argument(
        "--strategist",
        action="store_true",
        help="启用首席策略师（Step 5.5，默认禁用，会用单一LLM主观调整Borda排名）",
    )

    args = parser.parse_args()

    # 加载配置
    cfg = load_config()
    cfg._enable_strategist = args.strategist

    # 显示配置
    if args.config:
        print("\n  [配置信息]")
        print(f"  主提供商: {cfg.primary_provider}")
        print(f"  已配置提供商: {list(cfg.providers.keys())}")
        print(f"  LLM代理: {cfg.llm_proxy or '未设置'}")
        print(f"  Tushare: {'已配置' if cfg.tushare_token else '未配置'}")
        print(f"\n  各提供商模型:")
        for name, prov in cfg.providers.items():
            vision_str = f" | vision={prov.vision_model}" if prov.vision_model else ""
            vision_native = " [原生视觉]" if prov.supports_vision else ""
            print(f"    {name}: {prov.model} ({prov.base_url}){vision_str}{vision_native}")
        return

    # 演示模式
    if args.demo:
        run_demo_mode(cfg)
        return

    # 解析板块
    sectors = None
    if args.sectors:
        sectors = [s.strip() for s in args.sectors.split(",") if s.strip()]

    verbose = not args.quiet

    # 解析模型列表
    models_list = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None

    # 解析个股列表
    stock_codes = [c.strip() for c in args.stocks.split(",") if c.strip()] if args.stocks else None

    # 运行完整流程
    run_full_pipeline(
        cfg=cfg,
        sectors=sectors,
        panel_size=args.panel,
        debate_rounds=args.debate,
        max_stocks_per_sector=args.max_stocks,
        parallel_models=not args.no_parallel,
        verbose=verbose,
        models=models_list,
        stock_codes=stock_codes,
    )


if __name__ == "__main__":
    main()
