#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1 改进项集成测试 — 串联跑 4 个新函数验证完整流程

不调用任何 LLM，模拟链路A最后阶段：
  Borda 融合结果 → A/B 交叉验证 → 幻觉校验 → 组合风控 → 市场择时截断
"""

import sys
import os
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

from data_engine import DataEngine
from main import (
    cross_validate_with_l1,
    verify_llm_hallucination,
    portfolio_risk_check,
    compute_timing_limits,
    fusion_to_arb_result,
)


def make_fake_packages(codes_with_meta):
    """构造 fake stock_packages，用真实 TDX 日线数据"""
    de = DataEngine()
    packages = {}
    for code, meta in codes_with_meta.items():
        df = de._tdx_daily_only(code, days=100)
        if df is None or len(df) < 60:
            continue
        # 计算真实 PE 模拟（实际从 basics 拿，这里 mock）
        packages[code] = {
            "daily": df,
            "realtime": {
                "市盈率-动态": meta.get("real_pe", 15.0),
                "市净率": 1.5,
                "总市值": f"{meta.get('real_mktcap', 200):.0f}亿",
            },
        }
    return packages


def make_fake_borda(packages):
    """构造 fake Borda 融合结果，故意混入一些 LLM 幻觉"""
    final = []

    # 1. 数据真实，应该通过所有检查
    if "002460" in packages:
        final.append({
            "rank": 1, "code": "002460", "name": "赣锋锂业", "sector": "有色金属",
            "borda_score": 80, "avg_score": 80, "model_count": 3,
            "recommended_by": [{"model": "openai", "rank": 1}, {"model": "claude", "rank": 2}],
            "all_reasonings": ["龙头股，技术形态强势，建议关注"],
            "extra_warnings": [],
            "position_advice": "8-10%",
            "core_logic": "上升三角突破",
            "final_score": 80,
        })

    # 2. PE 编造（实际 PE 是 15.5，LLM 说 50）
    if "000001" in packages:
        final.append({
            "rank": 2, "code": "000001", "name": "平安银行", "sector": "金融",
            "borda_score": 75, "avg_score": 75, "model_count": 2,
            "recommended_by": [{"model": "claude", "rank": 1}, {"model": "deepseek", "rank": 3}],
            "all_reasonings": ["优质银行股，PE=50倍偏高但成长性好"],
            "extra_warnings": [],
            "position_advice": "6-8%",
            "core_logic": "估值修复",
            "final_score": 75,
        })

    # 3. 同板块第二只（金融，应该被同板块限制保留 - 排序靠前的两只）
    if "601318" in packages:
        final.append({
            "rank": 3, "code": "601318", "name": "中国平安", "sector": "金融",
            "borda_score": 70, "avg_score": 70, "model_count": 2,
            "recommended_by": [{"model": "openai", "rank": 2}],
            "all_reasonings": ["保险龙头，估值合理"],
            "extra_warnings": [],
            "position_advice": "8-10%",
            "core_logic": "底部反转",
            "final_score": 70,
        })

    # 4. 同板块第三只（金融，应该被同板块限制截断）
    if "600036" in packages:
        final.append({
            "rank": 4, "code": "600036", "name": "招商银行", "sector": "金融",
            "borda_score": 65, "avg_score": 65, "model_count": 2,
            "recommended_by": [{"model": "deepseek", "rank": 1}],
            "all_reasonings": ["银行龙头，分红稳定"],
            "extra_warnings": [],
            "position_advice": "10-12%",
            "core_logic": "高股息防御",
            "final_score": 65,
        })

    # 5. 数据完全编造（虚拟代码）
    final.append({
        "rank": 5, "code": "999999", "name": "虚拟股票", "sector": "概念",
        "borda_score": 60, "avg_score": 60, "model_count": 1,
        "recommended_by": [{"model": "openai", "rank": 5}],
        "all_reasonings": ["热门概念，可关注"],
        "extra_warnings": [],
        "position_advice": "5-7%",
        "core_logic": "概念炒作",
        "final_score": 60,
    })

    return final


def main():
    print("=" * 70)
    print("  P1 改进项集成测试")
    print("=" * 70)

    de = DataEngine()

    # 构造数据
    print("\n[Step 0] 加载数据...")
    codes_meta = {
        "002460": {"real_pe": 25, "real_mktcap": 1500},
        "000001": {"real_pe": 5.5, "real_mktcap": 2200},  # 真实 PE 5.5，LLM 说 50
        "601318": {"real_pe": 12, "real_mktcap": 9000},
        "600036": {"real_pe": 6, "real_mktcap": 9500},
    }
    packages = make_fake_packages(codes_meta)
    print(f"  加载了 {len(packages)} 只股票数据")

    # 构造 Borda 融合结果
    final_top = make_fake_borda(packages)
    print(f"\n[Step 1] Borda 融合: {len(final_top)} 只候选")
    for p in final_top:
        print(f"  #{p['rank']} {p['code']} {p['name']} ({p['sector']}) borda={p['borda_score']}")

    # ── Step 5.4: A/B 交叉验证 ──
    print("\n" + "─" * 70)
    print("[Step 5.4] A/B 交叉验证")
    final_top, l1 = cross_validate_with_l1(final_top, de, l1_candidates=None, verbose=True)

    # ── Step 5.45: 幻觉校验 ──
    print("\n" + "─" * 70)
    print("[Step 5.45] LLM 幻觉校验")
    final_top = verify_llm_hallucination(final_top, packages, verbose=True)

    # 转换为 arb_result，模拟风控通过
    arb_result = fusion_to_arb_result(final_top)
    risk_result = {
        "approved": [
            {**p, "name": p.get("name", ""), "sector": p.get("sector", ""),
             "risk_level": "低", "risk_flags": list(p.get("warnings", []))}
            for p in arb_result["final_picks"]
        ],
        "soft_excluded": [],
    }

    # ── 模拟 Step 6 风控（假设全部通过）──
    print("\n" + "─" * 70)
    print(f"[Step 6 模拟] 风控通过 {len(risk_result['approved'])} 只")

    # ── Step 6.3: 组合风控 ──
    print("\n" + "─" * 70)
    print("[Step 6.3] 组合层面风控")
    timing_limits = compute_timing_limits({"sentiment": {"score": 60}, "market_environment": {"trend": "偏多震荡"}})
    print(f"  市场环境: {timing_limits['regime']} (score={timing_limits['score']})")
    risk_result = portfolio_risk_check(risk_result, timing_limits, packages, verbose=True)

    # ── Step 6.5: 市场择时截断 ──
    print("\n" + "─" * 70)
    print(f"[Step 6.5] 市场择时硬截断 (max={timing_limits['max_recommendations']})")
    approved = risk_result["approved"]
    if timing_limits["max_recommendations"] < len(approved):
        approved = sorted(approved, key=lambda s: s.get("final_score", 0), reverse=True)
        risk_result["approved"] = approved[:timing_limits["max_recommendations"]]
        print(f"  截断到 {len(risk_result['approved'])} 只")
    else:
        print(f"  无需截断 ({len(approved)} ≤ {timing_limits['max_recommendations']})")

    # ── 最终结果展示 ──
    print("\n" + "=" * 70)
    print("[最终] 通过所有检查的推荐:")
    print("=" * 70)
    for s in risk_result["approved"]:
        print(f"\n  ▸ {s['code']} {s['name']} ({s['sector']})")
        print(f"    Borda={s.get('final_score',0):.1f} | 仓位={s.get('position_advice','-')}")
        print(f"    pass_l1={s.get('pass_l1', False)} | l1_source={s.get('l1_source','none')}")
        if s.get("hallucination_warnings"):
            print(f"    幻觉警告:")
            for w in s["hallucination_warnings"]:
                print(f"      - {w}")
        if s.get("risk_flags"):
            print(f"    风险标记:")
            for f in s["risk_flags"]:
                print(f"      - {f}")

    print(f"\n[组合摘要] {risk_result.get('portfolio', {})}")

    # 软过滤
    soft = risk_result.get("soft_excluded", [])
    if soft:
        print(f"\n[软过滤 {len(soft)} 只]")
        for s in soft:
            print(f"  {s['code']} {s.get('name','?')}: {'; '.join(s.get('risk_flags',[]))}")

    print("\n" + "=" * 70)
    print("  测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
