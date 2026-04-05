#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 跨模型融合模块

使用 Borda Count 算法将多个模型的选股结果融合，
并为每支推荐股票标注推荐来源模型。
"""

from typing import Dict, List, Optional


def borda_fusion(
    model_results: List[Dict],
    top_n: int = 10,
    model_weights: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Borda Count 跨模型融合（支持自适应模型权重）。

    输入:
        model_results : [
            {
                "model": str,
                "picks": [{"rank": int, "code": str, "name": str,
                           "sector": str, "score": float, ...}]
            }
        ]
        model_weights : 可选，模型Borda乘数 {"grok": 1.2, "claude": 0.8, ...}
                        准确率高的模型乘数>1，差的<1。None时所有模型等权。

    算法:
        每模型排名第 i 的股票得 (21 - i) × model_weight 分
        （第1名=20分，第20名=1分，第21名及以后=0分）

    输出:
        前 top_n 名列表，每支股票包含:
            rank          : 最终排名
            code          : 股票代码
            name          : 股票名称
            sector        : 所属板块
            borda_score   : 总 Borda 分（加权后）
            avg_score     : 各模型给出的平均 score
            model_count   : 推荐该股票的模型数量
            recommended_by: [{"model": str, "rank": int, "score": float}]
            all_reasonings: 各模型给出的理由（最多3条）
    """
    if model_weights is None:
        model_weights = {}

    stock_data: Dict[str, Dict] = {}

    for model_result in model_results:
        model_name = model_result.get("model", "unknown")
        picks = model_result.get("picks", [])
        weight = model_weights.get(model_name, 1.0)

        for pick in picks:
            if not isinstance(pick, dict):
                continue
            code = str(pick.get("code", "")).strip()
            if not code:
                continue

            rank = int(pick.get("rank", 99))
            borda_pts = max(0, 21 - rank) * weight  # 加权 Borda 分
            score = float(pick.get("score", 70))

            if code not in stock_data:
                stock_data[code] = {
                    "code": code,
                    "name": pick.get("name", ""),
                    "sector": pick.get("sector", ""),
                    "borda_score": 0.0,
                    "total_score": 0.0,
                    "score_count": 0,
                    "recommended_by": [],
                    "all_reasonings": [],
                }

            stock_data[code]["borda_score"] += borda_pts
            stock_data[code]["total_score"] += score
            stock_data[code]["score_count"] += 1
            stock_data[code]["recommended_by"].append({
                "model": model_name,
                "rank": rank,
                "score": score,
            })

            # 补全名称 / 板块（取首个非空值）
            if pick.get("name") and not stock_data[code]["name"]:
                stock_data[code]["name"] = pick["name"]
            if pick.get("sector") and not stock_data[code]["sector"]:
                stock_data[code]["sector"] = pick["sector"]
            if pick.get("reasoning"):
                stock_data[code]["all_reasonings"].append(
                    f"[{model_name}] {pick['reasoning']}"
                )

    # 统计实际参与模型数（有有效推荐的模型）
    num_models = len([m for m in model_results if len(m.get("picks", [])) > 0])
    num_models = max(num_models, 1)  # 防除零

    # 计算均分 + 归一化 Borda 分
    for data in stock_data.values():
        n = data["score_count"]
        data["avg_score"] = round(data["total_score"] / n, 1) if n > 0 else 0.0
        data["model_count"] = len(data["recommended_by"])
        data["normalized_borda"] = round(data["borda_score"] / num_models, 2)

    # 多级排序：归一化Borda → 推荐模型数 → 平均分 → 股票代码（确定性兜底）
    sorted_stocks = sorted(
        stock_data.values(),
        key=lambda x: (x["normalized_borda"], x["model_count"], x["avg_score"], x["code"]),
        reverse=True,
    )

    result = []
    for rank, stock in enumerate(sorted_stocks[:top_n], 1):
        result.append({
            "rank": rank,
            "code": stock["code"],
            "name": stock["name"],
            "sector": stock["sector"],
            "borda_score": stock["borda_score"],
            "normalized_borda": stock["normalized_borda"],
            "avg_score": stock["avg_score"],
            "model_count": stock["model_count"],
            "recommended_by": stock["recommended_by"],
            "all_reasonings": stock["all_reasonings"][:3],
        })

    return result
