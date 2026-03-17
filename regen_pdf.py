#!/usr/bin/env python3
"""从保存的数据重新生成PDF报告（含风控提示股票）"""
import json
import os
import sys

# 加载数据
DATA_DIR = os.path.join(os.path.dirname(__file__), "output", "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "output", "reports")
DATE = "20260314"

def load_json(name):
    path = os.path.join(DATA_DIR, f"{name}_{DATE}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _consensus_level(model_count: int) -> str:
    if model_count >= 3:
        return f"强共识({model_count}模型)"
    elif model_count >= 2:
        return f"弱共识({model_count}模型)"
    return "单一模型"

def fusion_to_arb_result(final_top):
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

def build_expert_summary(model_results):
    expert_summary = {}
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


if __name__ == "__main__":
    print("加载数据...")
    fusion_result = load_json("fusion_result")
    risk_result = load_json("risk_result")
    model_results_raw = load_json("model_results")
    model_results = model_results_raw.get("model_results", model_results_raw) if isinstance(model_results_raw, dict) else model_results_raw
    sector_screen = load_json("sector_screen")

    final_top = fusion_result["final_top10"]
    arb_result = fusion_to_arb_result(final_top)
    expert_summary = build_expert_summary(model_results)

    # 提取板块
    top_sectors = sector_screen.get("sector_names", [])
    if not top_sectors:
        sectors = []
        for pick in final_top:
            s = pick.get("sector", "")
            if s and s not in sectors:
                sectors.append(s)
        top_sectors = sectors[:5]

    # 构建stock_packages（需要K线数据）
    print("构建股票数据包（获取K线）...")
    from data_engine import DataEngine
    engine = DataEngine()

    # 所有15只股票的代码
    all_codes = [s.get("code") for s in risk_result.get("approved", [])]
    all_codes += [s.get("code") for s in risk_result.get("soft_excluded", [])]

    stock_packages = engine.build_stock_data_package(all_codes)

    print(f"获取到 {len(stock_packages)} 只股票数据包")

    # 生成PDF
    from report_generator import generate_report
    os.makedirs(REPORTS_DIR, exist_ok=True)

    pdf_path = generate_report(
        risk_result=risk_result,
        arb_result=arb_result,
        expert_summary=expert_summary,
        stock_packages=stock_packages,
        top_sectors=top_sectors,
        n_models=4,
        output_dir=REPORTS_DIR,
    )

    if pdf_path:
        print(f"\nPDF生成成功: {pdf_path}")
    else:
        print("\nPDF生成失败")
