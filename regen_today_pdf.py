#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从当日缓存 JSON 重新生成 PDF 报告（不重跑 LLM）。
用于修复报告字段补全 bug 后快速重建当天的 PDF。

用法:
    python regen_today_pdf.py            # 默认使用今日日期
    python regen_today_pdf.py 20260412   # 指定日期
"""
import json
import os
import sys
from datetime import datetime

# 复用 main.py 的助手
from main import (
    DATA_DIR, REPORTS_DIR,
    fusion_to_arb_result,
    build_expert_summary,
    _build_industry_map,
    _enrich_risk_result_for_report,
)
from data_engine import DataEngine
from report_generator import generate_report


def _load_json(name: str) -> dict:
    path = os.path.join(DATA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
    print(f"[regen] 日期: {date}")

    # ── 载入缓存 ────────────────────────────────
    fusion = _load_json(f"fusion_result_{date}.json")
    final_top = fusion.get("final_top10") or fusion.get("final_top25") or []
    print(f"[regen] fusion final_top = {len(final_top)} 只")

    risk_result = _load_json(f"risk_result_{date}.json")
    print(f"[regen] approved={len(risk_result.get('approved', []))} "
          f"soft_excluded={len(risk_result.get('soft_excluded', []))}")

    mr_result = _load_json(f"market_radar_{date}.json")
    sector_screen = _load_json(f"sector_screen_{date}.json")
    top_sectors = sector_screen.get("sector_names") or []
    if not top_sectors:
        picks = sector_screen.get("picks", [])
        top_sectors = [p.get("sector", "") for p in picks if p.get("sector")][:5]

    try:
        model_results_raw = _load_json(f"model_results_{date}.json")
        model_results = model_results_raw.get("model_results", model_results_raw)
    except Exception:
        model_results = []

    l3_result = None
    try:
        l3_raw = _load_json(f"L3_result_{date}.json")
        l3_result = l3_raw.get("L3_result", l3_raw)
    except Exception:
        pass

    # ── 重建 arb_result + 补全字段 ──────────────
    arb_result = fusion_to_arb_result(final_top)

    engine = DataEngine()
    industry_map = _build_industry_map(engine)
    print(f"[regen] 行业映射: {len(industry_map)} 条")

    _enrich_risk_result_for_report(risk_result, arb_result, industry_map)

    # ── 构建 stock_packages（25 只，3-5 分钟）──
    all_codes = []
    for s in risk_result.get("approved", []) + risk_result.get("soft_excluded", []):
        code = s.get("code", "")
        if code and code not in all_codes:
            all_codes.append(code)
    print(f"[regen] 构建数据包 {len(all_codes)} 只...")

    sector_map = {
        s.get("code", ""): s.get("sector", "")
        for s in risk_result.get("approved", []) + risk_result.get("soft_excluded", [])
        if s.get("sector")
    }
    stock_packages = engine.build_stock_data_package(all_codes, sector_map=sector_map)
    print(f"[regen] 数据包完成: {len(stock_packages)} 只")

    expert_summary = build_expert_summary(model_results) if model_results else {}

    # ── 生成 PDF ────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    pdf_path = generate_report(
        risk_result=risk_result,
        arb_result=arb_result,
        expert_summary=expert_summary,
        stock_packages=stock_packages,
        top_sectors=top_sectors,
        n_models=3,
        output_dir=REPORTS_DIR,
        l3_result=l3_result,
    )
    print(f"\n[regen] ✓ PDF 已生成: {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
