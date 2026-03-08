#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 主程序入口（重构版：模型纵向并行）

系统架构:
  数据采集 → 板块初筛 → N模型并行(每模型内E1→E6顺序+intra辩论) → Borda融合 → 风控 → 报告

运行方式:
  python main.py                              # 全流程运行（自动选板块）
  python main.py --sectors 人工智能,机器人    # 指定板块运行
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
    SectorPicker,
    GlobalThemeAdvisor,
    SectorScreener,
    ModelTask,
    RiskController,
)
from fusion import borda_fusion
from work_logger import WorkLogger
from report_generator import generate_report

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
) -> str:
    """生成最终报告文本（含 Borda 评分和推荐来源）"""
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

    approved = risk_result.get("approved", [])
    if not approved:
        lines.append("  [警告] 未产生有效推荐，请检查数据和模型配置")
        return "\n".join(lines)

    lines.append(f"\n  共推荐 {len(approved)} 只股票\n")

    # 表头
    lines.append(
        f"  {'排名':<4} {'代码':<8} {'名称':<10} {'板块':<12} "
        f"{'Borda分':<8} {'推荐模型数':<8} {'风险':<6} {'仓位建议':<12}"
    )
    lines.append("  " + "─" * 75)

    for stock in approved:
        rank = stock.get("rank", "")
        code = stock.get("code", "")
        name = stock.get("name", "")[:8]
        sector = stock.get("sector", "")[:10]
        risk = stock.get("risk_level", "")[:5]
        position = stock.get("position_advice", "")[:12]

        fi = fusion_lookup.get(code, {})
        borda = fi.get("borda_score", stock.get("final_score", 0))
        model_count = fi.get("model_count", stock.get("model_count", 0))

        lines.append(
            f"  {rank:<4} {code:<8} {name:<10} {sector:<12} "
            f"{borda:<8.0f} {model_count:<8} {risk:<6} {position:<12}"
        )

    lines.append("\n" + "─" * 70)

    # 详情
    lines.append("\n  【个股详情】")
    for stock in approved:
        code = stock.get("code", "")
        name = stock.get("name", "")
        lines.append(f"\n  ▸ {code} {name}")

        # Borda 来源信息
        fi = fusion_lookup.get(code, {})
        recommended_by = fi.get("recommended_by", [])
        if recommended_by:
            by_str = "、".join(
                f"{r['model']}(#{r['rank']})"
                for r in recommended_by
            )
            lines.append(f"    推荐来源: {by_str}  Borda分={fi.get('borda_score', 0):.0f}")

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

    # ── Step 1: 板块优选（全面板投票 Top5）────────────────────────────
    if sectors:
        top_sectors = sectors
        sector_data = {}
        market_hot = {}
        print(f"\n  [用户指定板块] {top_sectors}")
    else:
        print_section("Step 1: 板块数据采集与优选（SP 全面板投票）")
        t_step1 = time.time()
        sector_data = engine.fetch_sector_overview()
        market_hot = engine.fetch_market_hot()

        picker = SectorPicker(llm, cfg)
        pick_result = picker.pick(
            sector_overview=sector_data,
            market_hot=market_hot,
            panel_size=len(active_models),   # 全部面板模型参与投票
            verbose=verbose,
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

    # ── Step 1.5: 跨市场主题补充（GX，美股/港股联动）────────────────
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

    # ── Step 2: 数据采集 ─────────────────────────────────────────────
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

    # ── Step 3: 构建股票画像（极简版，供LLM坐标定位）────────────────
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
        # 使用极简画像：每股约30-40 token（代码/名称/板块/PE/PB/市值/近20日涨幅）
        # LLM通过联网搜索和云端知识补全深度分析，不再本地提交原始K线序列
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
    print_section("Step 5: Borda Count 跨模型融合（输出 Top10）")
    t_step5 = time.time()
    logger.log("fusion_start", detail={"model_count": len(model_results)})

    final_top10 = borda_fusion(model_results, top_n=10)

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

    # ── Step 7: 最终报告（文本）──────────────────────────────────────
    print_section("Step 7: 最终报告")
    report = format_final_report(risk_result, arb_result, fusion_result=final_top10)
    print(report)

    report_path = os.path.join(REPORTS_DIR, f"final_report_{TODAY}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

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

    args = parser.parse_args()

    # 加载配置
    cfg = load_config()

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
    )


if __name__ == "__main__":
    main()
