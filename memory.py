#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 记忆学习模块

功能:
  1. 存储每次选股结果（市场环境+推荐列表+评分）
  2. 语义检索相似历史环境（ChromaDB向量库）
  3. 回测后存储实际涨跌结果
  4. 生成经验反思文本注入LLM提示词

降级: ChromaDB不可用时，回退到JSON文件按日期倒序匹配
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Optional

MEMORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False


class StockMemory:
    """选股记忆系统"""

    def __init__(self, memory_dir: str = MEMORY_DIR):
        self.memory_dir = memory_dir
        self._collection = None

        if _CHROMA_AVAILABLE:
            try:
                self._client = chromadb.PersistentClient(path=os.path.join(memory_dir, "chromadb"))
                self._collection = self._client.get_or_create_collection(
                    name="stock_decisions",
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                print(f"  [记忆] ChromaDB初始化失败: {e}，回退到JSON模式")
                self._collection = None

    def store(
        self,
        date: str,
        market_env: Dict,
        recommendations: List[Dict],
        mr_result: Optional[Dict] = None,
    ) -> None:
        """
        存储一次选股决策。

        Args:
            date: 日期 YYYYMMDD
            market_env: {trend, sentiment_score, sentiment_temp, cycle_phase, top_sectors}
            recommendations: [{code, name, borda_score, risk_level, ...}]
            mr_result: 市场雷达完整结果
        """
        record = {
            "date": date,
            "market_env": market_env,
            "recommendations": recommendations[:25],
            "mr_summary": "",
            "stored_at": datetime.now().isoformat(),
        }

        # 构建市场环境描述文本（用于语义检索）
        env_text = self._build_env_text(market_env, mr_result)
        record["env_text"] = env_text

        if mr_result:
            env = mr_result.get("market_environment", {})
            sent = mr_result.get("sentiment", {})
            record["mr_summary"] = (
                f"大盘:{env.get('trend','')} 情绪:{sent.get('temperature','')} "
                f"周期:{sent.get('cycle_phase','')} 分数:{sent.get('score','')}"
            )

        # 存入JSON
        json_path = os.path.join(self.memory_dir, f"decision_{date}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)

        # 存入ChromaDB
        if self._collection is not None:
            try:
                rec_summary = ", ".join(
                    f"{r.get('code','')}{r.get('name','')}"
                    for r in recommendations[:10]
                )
                self._collection.upsert(
                    ids=[date],
                    documents=[env_text],
                    metadatas=[{
                        "date": date,
                        "trend": market_env.get("trend", ""),
                        "sentiment": market_env.get("sentiment_temp", ""),
                        "score": str(market_env.get("sentiment_score", "")),
                        "top_picks": rec_summary[:500],
                    }],
                )
            except Exception as e:
                print(f"  [记忆] ChromaDB存储失败: {e}")

    def store_outcome(self, date: str, outcomes: List[Dict]) -> None:
        """
        存储回测结果（推荐N日后的实际涨跌）。

        Args:
            date: 原推荐日期
            outcomes: [{code, name, return_pct, ...}]
        """
        json_path = os.path.join(self.memory_dir, f"decision_{date}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            record["outcomes"] = outcomes
            record["outcome_date"] = datetime.now().strftime("%Y%m%d")

            # 计算总结
            if outcomes:
                returns = [o.get("return_pct", 0) for o in outcomes if o.get("return_pct") is not None]
                if returns:
                    record["outcome_summary"] = {
                        "win_rate": round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
                        "avg_return": round(sum(returns) / len(returns), 2),
                        "max_gain": round(max(returns), 2),
                        "max_loss": round(min(returns), 2),
                    }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2, default=str)

            # 更新ChromaDB的metadata
            if self._collection is not None and record.get("outcome_summary"):
                try:
                    s = record["outcome_summary"]
                    self._collection.update(
                        ids=[date],
                        metadatas=[{
                            "date": date,
                            "trend": record.get("market_env", {}).get("trend", ""),
                            "sentiment": record.get("market_env", {}).get("sentiment_temp", ""),
                            "score": str(record.get("market_env", {}).get("sentiment_score", "")),
                            "win_rate": str(s["win_rate"]),
                            "avg_return": str(s["avg_return"]),
                        }],
                    )
                except Exception:
                    pass

    def recall(self, current_env: Dict, mr_result: Optional[Dict] = None, n_matches: int = 3) -> List[Dict]:
        """
        检索与当前市场环境最相似的历史决策。

        Returns: [{date, env_text, mr_summary, recommendations, outcomes, outcome_summary}]
        """
        env_text = self._build_env_text(current_env, mr_result)

        # 优先用ChromaDB语义检索
        if self._collection is not None:
            try:
                count = self._collection.count()
                if count > 0:
                    results = self._collection.query(
                        query_texts=[env_text],
                        n_results=min(n_matches, count),
                    )
                    matches = []
                    for i, doc_id in enumerate(results["ids"][0]):
                        json_path = os.path.join(self.memory_dir, f"decision_{doc_id}.json")
                        if os.path.exists(json_path):
                            with open(json_path, "r", encoding="utf-8") as f:
                                record = json.load(f)
                            record["similarity"] = 1 - (results["distances"][0][i] if results.get("distances") else 0)
                            matches.append(record)
                    if matches:
                        return matches
            except Exception as e:
                print(f"  [记忆] ChromaDB检索失败: {e}，回退JSON")

        # 降级: JSON文件按日期倒序
        return self._recall_from_json(n_matches)

    def get_reflection_prompt(self, current_env: Dict, mr_result: Optional[Dict] = None) -> str:
        """
        生成可注入LLM提示词的历史反思文本。
        """
        matches = self.recall(current_env, mr_result, n_matches=3)
        if not matches:
            return ""

        lines = ["【历史决策参考（系统记忆）】"]
        for m in matches:
            date = m.get("date", "?")
            mr_sum = m.get("mr_summary", "")
            recs = m.get("recommendations", [])
            outcomes = m.get("outcome_summary")

            rec_str = ", ".join(f"{r.get('code','')}{r.get('name','')}" for r in recs[:5])
            lines.append(f"\n  ■ {date}: {mr_sum}")
            lines.append(f"    推荐: {rec_str}...")

            if outcomes:
                lines.append(
                    f"    结果: 胜率{outcomes['win_rate']:.0f}% "
                    f"均收益{outcomes['avg_return']:+.1f}% "
                    f"最大盈{outcomes['max_gain']:+.1f}% "
                    f"最大亏{outcomes['max_loss']:+.1f}%"
                )
                # 教训
                if outcomes["avg_return"] < -2:
                    lines.append(f"    ⚠ 教训: 该环境下推荐效果差，需更保守")
                elif outcomes["win_rate"] > 70:
                    lines.append(f"    ✓ 经验: 该环境下策略有效，可延续")
            else:
                lines.append(f"    结果: 尚未回测")

        return "\n".join(lines)

    def _build_env_text(self, env: Dict, mr_result: Optional[Dict] = None) -> str:
        """构建市场环境描述文本"""
        parts = []
        parts.append(f"大盘趋势:{env.get('trend', 'N/A')}")
        parts.append(f"情绪温度:{env.get('sentiment_temp', 'N/A')}")
        parts.append(f"情绪周期:{env.get('cycle_phase', 'N/A')}")
        parts.append(f"情绪分数:{env.get('sentiment_score', 'N/A')}")

        sectors = env.get("top_sectors", [])
        if sectors:
            parts.append(f"热门板块:{','.join(sectors[:5])}")

        if mr_result:
            hypes = mr_result.get("hype_predictions", [])
            if hypes:
                hype_str = ", ".join(h.get("concept_name", "") for h in hypes[:5])
                parts.append(f"概念炒作:{hype_str}")

            adv = mr_result.get("operation_advice", "")
            if adv:
                parts.append(f"操作建议:{adv[:100]}")

        return " | ".join(parts)

    def _recall_from_json(self, n_matches: int) -> List[Dict]:
        """JSON降级检索：按日期倒序返回最近N条"""
        pattern = os.path.join(self.memory_dir, "decision_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        matches = []
        for f in files[:n_matches]:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    record = json.load(fh)
                matches.append(record)
            except Exception:
                continue
        return matches

    def list_records(self) -> List[str]:
        """列出所有记忆记录日期"""
        pattern = os.path.join(self.memory_dir, "decision_*.json")
        files = sorted(glob.glob(pattern), reverse=True)
        dates = []
        for f in files:
            basename = os.path.basename(f)
            date = basename.replace("decision_", "").replace(".json", "")
            dates.append(date)
        return dates
