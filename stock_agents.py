#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 智能体模块

包含全部12个智能体：
  SP. 板块优选专家 (SectorPicker)        ← 全面板多模型投票选出 Top5 板块
  GX. 跨市场主题专家 (GlobalThemeAdvisor) ← 参考美股/港股热门，补充至多1个A股板块
  S0. 板块初筛 (SectorScreener)          ← 轻量3模型快速初筛（降级备用）
  E1. 动量换手型专家
  E2. 成长估值型专家
  E3. 多因子平衡型专家
  E4. 技术形态型专家
  E5. 资金流向型专家
  E6. 事件催化型专家
  E7. 板块优股专家 (SectorOutperformer)
  V.  投票仲裁者 (VotingArbitrator)
  R.  风控审查 (RiskController)

每个专家角色使用多个大模型进行辩论，最终聚合投票给出选股结论。
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llm_client import LLMClient, ConversationSession
from config import Config
from work_logger import WorkLogger


# ===================================================================== #
#  数据结构                                                              #
# ===================================================================== #

@dataclass
class StockPick:
    """单只股票选股结论"""
    code: str
    name: str
    sector: str
    expert_id: str
    stars: float = 3.0
    score: float = 70.0
    reasoning: str = ""
    hold_period: str = "5-10日"
    stop_loss: str = "跌破MA5"
    warning: str = ""
    voters: List[str] = field(default_factory=list)
    vote_count: int = 1


@dataclass
class ExpertResult:
    """单位专家的分析结论"""
    expert_id: str
    expert_name: str
    picks: List[StockPick] = field(default_factory=list)
    reasoning: str = ""
    debate_log: Dict[str, str] = field(default_factory=dict)


# ===================================================================== #
#  基础智能体                                                            #
# ===================================================================== #

# ── 联网研究指令（嵌入每位专家的 system prompt 末尾）────────────────
_RESEARCH_INSTRUCTION = """\

【数据获取方式】
候选清单只提供：股票代码、名称、板块、PE/PB/市值、近20日涨幅等坐标信息。
请主动运用以下能力获取分析依据：
1. 联网搜索（若模型支持）：查询各股近期K线走势、公告、机构评级、主力资金动向、新闻
2. 云端知识：调用你对A股上市公司基本面、行业地位、历史估值区间的已有认知
3. 推理判断：基于上述信息，结合本专家的分析框架，给出有据可查的选股结论

不要因数据看起来"不够"而拒绝分析——坐标数据配合你的知识与搜索即可完成专业判断。"""

# JSON 输出格式说明（所有专家共用）
_PICK_FORMAT = """\
请以如下 JSON 格式返回，不要包含其他文字：
```json
{
  "expert_id": "<专家代号>",
  "picks": [
    {
      "code": "股票代码(6位)",
      "name": "股票名称",
      "sector": "所属板块",
      "stars": <1-5的整数>,
      "score": <0-100的整数>,
      "reasoning": "选择理由（200字以内）",
      "hold_period": "持有周期，如5-10日",
      "stop_loss": "止损参考，如跌破MA5或-7%",
      "warning": "风险提示（可为空字符串）"
    }
  ],
  "sector_view": "对本板块的整体看法（100字以内）"
}
```
picks 数量：每个板块选 2-5 只，总数不超过 10 只。
stars 含义：5=极强 4=强 3=中等 2=弱 1=极弱"""


class BaseAgent:
    """智能体基类"""

    EXPERT_ID = "BASE"
    EXPERT_NAME = "基础专家"

    def __init__(self, llm_client: LLMClient, config: Config):
        self.llm = llm_client
        self.config = config

    def _system_prompt(self) -> str:
        raise NotImplementedError

    def _build_messages(self, data_text: str, provider_name: str) -> List[Dict]:
        """构建 LLM 消息列表"""
        return [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    f"【候选股票数据】\n\n{data_text}\n\n"
                    "---\n"
                    f"请以 {self.EXPERT_NAME}（{self.EXPERT_ID}）的视角，"
                    "对以上候选股票进行专业分析，给出你的选股推荐。\n\n"
                    f"{_PICK_FORMAT}"
                ),
            },
        ]

    def _summarize_results(self, results: Dict[str, str]) -> str:
        """将多模型结果汇总为辩论上下文"""
        return LLMClient.build_debate_summary(results)

    def analyze(
        self,
        stock_profiles: Dict[str, str],  # {code: profile_text}
        panel_size: Optional[int] = None,
        debate_rounds: int = 1,
        verbose: bool = True,
    ) -> ExpertResult:
        """
        运行专家分析（含多模型辩论）。
        stock_profiles: 由 DataEngine.build_stock_profile() 生成的文本字典
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  专家 [{self.EXPERT_ID}] {self.EXPERT_NAME} 启动分析")
            print(f"{'='*60}")

        # 合并所有股票画像文本
        data_text = "\n\n".join(
            f"--- {code} ---\n{profile}"
            for code, profile in stock_profiles.items()
        )

        debate_result = self.llm.run_debate(
            build_prompt_fn=lambda name: self._build_messages(data_text, name),
            summarize_fn=self._summarize_results,
            max_models=panel_size,
            rounds=debate_rounds,
            verbose=verbose,
        )

        final_results = debate_result.get("final", debate_result.get("round1", {}))
        aggregated = LLMClient.aggregate_picks(final_results)

        # 转换为 StockPick 列表
        picks = []
        for code, agg in sorted(
            aggregated.items(),
            key=lambda x: x[1]["consensus_score"],
            reverse=True,
        ):
            # 取票数最多的 reasoning
            main_voter = agg["voters"][0] if agg["voters"] else ""
            reasoning = agg["reasonings"].get(main_voter, "")
            # 合并所有理由
            all_reasons = [
                f"[{v}] {r}" for v, r in agg["reasonings"].items() if r
            ]
            merged_reason = reasoning if len(all_reasons) <= 1 else "; ".join(all_reasons[:3])

            # 取第一个 pick_detail 的额外字段
            detail = {}
            for v_detail in agg.get("pick_details", {}).values():
                detail = v_detail
                break

            picks.append(
                StockPick(
                    code=code,
                    name=agg.get("name", ""),
                    sector=agg.get("sector", ""),
                    expert_id=self.EXPERT_ID,
                    stars=agg.get("avg_stars", 3.0),
                    score=agg.get("avg_score", 70.0),
                    reasoning=merged_reason[:300],
                    hold_period=detail.get("hold_period", "5-10日"),
                    stop_loss=detail.get("stop_loss", "跌破MA5"),
                    warning=detail.get("warning", ""),
                    voters=agg["voters"],
                    vote_count=agg["votes"],
                )
            )

        # 只保留得票≥1 的 top-10
        picks = picks[:10]

        return ExpertResult(
            expert_id=self.EXPERT_ID,
            expert_name=self.EXPERT_NAME,
            picks=picks,
            reasoning=f"{self.EXPERT_NAME}完成分析，共选出{len(picks)}只股票",
            debate_log=debate_result,
        )

    def run_in_session(
        self,
        session: ConversationSession,
        stock_profiles: Dict[str, str],
        stock_packages: Optional[Dict] = None,
        include_profiles: bool = True,
    ) -> ExpertResult:
        """
        在共享 ConversationSession 中运行该专家分析。
        每次调用追加 user/assistant 消息，保留完整对话上下文。
        include_profiles=True  (仅E1): 将所有股票画像嵌入消息
        include_profiles=False (E2-E6): 画像已在历史中，只发送指令，避免重复叠加
        """
        if include_profiles:
            data_text = "\n\n".join(
                f"--- {code} ---\n{profile}"
                for code, profile in stock_profiles.items()
            )
            content = (
                f"【候选股票数据】\n\n{data_text}\n\n"
                "---\n"
                f"请以 {self.EXPERT_NAME}（{self.EXPERT_ID}）的视角，"
                "对以上候选股票进行专业分析，给出你的选股推荐。\n\n"
                f"{_PICK_FORMAT}"
            )
        else:
            content = (
                f"请以 {self.EXPERT_NAME}（{self.EXPERT_ID}）的视角，"
                "基于对话中已提供的所有候选股票数据，进行独立专业分析，"
                "给出你的选股推荐（不受前面专家观点影响，独立判断）。\n\n"
                f"{_PICK_FORMAT}"
            )

        resp = session.say(content)

        picks: List[StockPick] = []
        if resp:
            parsed = LLMClient.parse_json(resp)
            if parsed and "picks" in parsed:
                for pick in parsed["picks"]:
                    if not isinstance(pick, dict):
                        continue
                    code = str(pick.get("code", "")).strip()
                    if not code:
                        continue
                    picks.append(StockPick(
                        code=code,
                        name=str(pick.get("name", "")),
                        sector=str(pick.get("sector", "")),
                        expert_id=self.EXPERT_ID,
                        stars=float(pick.get("stars", 3)),
                        score=float(pick.get("score", 70)),
                        reasoning=str(pick.get("reasoning", ""))[:300],
                        hold_period=str(pick.get("hold_period", "5-10日")),
                        stop_loss=str(pick.get("stop_loss", "跌破MA5")),
                        warning=str(pick.get("warning", "")),
                    ))

        return ExpertResult(
            expert_id=self.EXPERT_ID,
            expert_name=self.EXPERT_NAME,
            picks=picks[:10],
            reasoning=f"{self.EXPERT_NAME}完成Session分析，共选出{len(picks)}只股票",
            debate_log={"session_response": resp or ""},
        )


# ===================================================================== #
#  SP: 板块优选智能体（全面板投票，输出 Top5）                          #
# ===================================================================== #

class SectorPicker(BaseAgent):
    """
    板块优选专家（代号 SP）。

    使用全部并行模型对当前市场板块进行独立打分与投票，
    按票数（+平均排名）聚合出得票最高的 Top5 板块，
    作为后续个股分析的前提板块范围。

    评估重点：近期热度 × 资金流入 × 成交量活跃度
    """

    EXPERT_ID = "SP"
    EXPERT_NAME = "板块优选专家"

    # ------------------------------------------------------------------ #
    #  系统提示词                                                         #
    # ------------------------------------------------------------------ #
    def _system_prompt(self) -> str:
        return """你是A股板块优选专家（代号SP）。

你的目标是从当前市场的所有板块中，选出近期综合表现最优、最值得重点关注的5个板块，
供后续在这些板块内精选个股使用。

【核心评估维度（三维聚焦）】

1. 近期热度（权重35%）
   - 近5日板块平均涨幅（绝对涨幅 + 相对大盘超额）
   - 涨停股数量及集中度（同一板块涨停越多越热）
   - 龙虎榜席位集中情况（游资/机构争抢信号）
   - 板块在媒体/市场的讨论热度

2. 资金流入（权重35%）
   - 近5日行业/概念主力净流入（绝对值 + 加速度）
   - 超大单净买入方向（机构性主动买入）
   - 北向资金配置方向
   - 融资余额变化趋势（近5日增减）

3. 成交量与换手活跃度（权重30%）
   - 板块整体换手率对比60日均值（>1.3倍为活跃）
   - 成交量放大倍数（近5日成交额/60日日均）
   - 板块内个股同步放量比例
   - 量价配合质量（放量上涨优于缩量或阴量）

【排除条件】
- 近5日板块涨幅>20%（过热，短期风险高）
- 资金流向出现明显逆转（近2日连续净流出）
- 主要成分股已连续涨停>3日（追高风险）
- 成分股数量<8只（流动性不足）

【输出格式】
请返回如下 JSON，不要包含其他文字：
```json
{
  "top5": [
    {
      "rank": 1,
      "sector_name": "板块名称",
      "sector_type": "行业/概念",
      "heat_score": 85,
      "fund_flow_score": 90,
      "volume_score": 80,
      "total_score": 88,
      "reasoning": "选择理由（120字以内，突出热度/资金/量能三个维度）",
      "key_evidence": "最关键的1-2条数据支撑",
      "risk_note": "主要风险（可为空）"
    }
  ],
  "excluded": ["排除板块（可为空列表）"],
  "market_summary": "当前市场资金风格简述（60字以内）"
}
```
top5 数量恰好为5个，按综合评分从高到低排列。"""

    # ------------------------------------------------------------------ #
    #  全面板投票选板块                                                   #
    # ------------------------------------------------------------------ #
    def pick(
        self,
        sector_overview: Dict,
        market_hot: Dict,
        panel_size: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        全面板模型投票，返回得票最高的 Top5 板块。

        返回:
        {
          "top5_sectors": [{"sector_name", "votes", "avg_rank", "total_score", ...}],
          "sector_names": ["板块1", "板块2", ...],   # 仅名称列表
          "vote_detail":  {sector_name: {"votes", "avg_rank", "models": [...]}},
          "raw_results":  {provider: raw_text},
        }
        """
        if verbose:
            print(f"\n{'='*60}")
            print("  SP: 板块优选专家启动（全面板投票）")
            print(f"{'='*60}")

        data_text = self._build_data_text(sector_overview, market_hot)

        def _build_msgs(_: str) -> List[Dict]:
            return [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"【当前市场板块数据】\n\n{data_text}\n\n"
                        "---\n"
                        "请分析以上数据，从热度、资金流入、成交量三个维度，"
                        "选出最值得重点关注的5个板块。\n"
                        "必须以 JSON 格式返回，top5 恰好5个。"
                    ),
                },
            ]

        # 调用全部面板模型
        n_models = panel_size or len(self.config.providers)
        raw_results = self.llm.call_panel(
            messages=_build_msgs(""),
            max_models=n_models,
            verbose=verbose,
        )

        # ── 聚合投票 ─────────────────────────────────────────────────
        vote_map: Dict[str, Dict] = {}   # sector_name -> aggregated info

        for provider, text in raw_results.items():
            parsed = LLMClient.parse_json(text)
            if not parsed:
                continue
            top5_list = parsed.get("top5", [])
            if not isinstance(top5_list, list):
                continue
            for item in top5_list:
                name = str(item.get("sector_name", "")).strip()
                if not name:
                    continue
                rank = int(item.get("rank", 5))
                score = float(item.get("total_score", 70))

                if name not in vote_map:
                    vote_map[name] = {
                        "sector_name": name,
                        "sector_type": item.get("sector_type", ""),
                        "votes": 0,
                        "rank_sum": 0,
                        "score_sum": 0.0,
                        "models": [],
                        "reasonings": [],
                        "key_evidences": [],
                    }

                vote_map[name]["votes"] += 1
                vote_map[name]["rank_sum"] += rank
                vote_map[name]["score_sum"] += score
                vote_map[name]["models"].append(provider)

                r = item.get("reasoning", "")
                if r:
                    vote_map[name]["reasonings"].append(f"[{provider}] {r}")
                e = item.get("key_evidence", "")
                if e:
                    vote_map[name]["key_evidences"].append(f"[{provider}] {e}")

        # ── 排序：票数优先，票数相同时按平均排名（升序）排 ──────────
        for v in vote_map.values():
            n = max(v["votes"], 1)
            v["avg_rank"] = round(v["rank_sum"] / n, 2)
            v["avg_score"] = round(v["score_sum"] / n, 1)

        ranked = sorted(
            vote_map.values(),
            key=lambda x: (-x["votes"], x["avg_rank"]),
        )

        top5 = ranked[:5]
        sector_names = [s["sector_name"] for s in top5]

        if verbose:
            total_models = len(raw_results)
            print(f"\n  [SP] 参与投票模型: {total_models} 个")
            print(f"  [SP] 优选结果 Top5:")
            for i, s in enumerate(top5, 1):
                print(
                    f"    {i}. {s['sector_name']}"
                    f"  票数={s['votes']}/{total_models}"
                    f"  均排={s['avg_rank']:.1f}"
                    f"  均分={s['avg_score']:.0f}"
                    f"  模型=[{', '.join(s['models'])}]"
                )

        return {
            "top5_sectors": top5,
            "sector_names": sector_names,
            "vote_detail": vote_map,
            "raw_results": raw_results,
            "total_models_voted": len(raw_results),
        }

    # ------------------------------------------------------------------ #
    #  数据文本构建（独立出来便于复用）                                  #
    # ------------------------------------------------------------------ #
    def _build_data_text(self, sector_overview: Dict, market_hot: Dict) -> str:
        parts: List[str] = []

        # 申万行业行情（涨跌幅前30）
        sw = sector_overview.get("sw_industry", [])[:30]
        if sw:
            parts.append("【申万行业板块行情（近期涨跌幅前30）】")
            for r in sw[:30]:
                fields = {k: r[k] for k in ["板块名称", "涨跌幅", "换手率", "成交额"] if k in r}
                if fields:
                    parts.append("  " + " | ".join(f"{k}={v}" for k, v in fields.items()))

        # 概念板块行情
        concept = sector_overview.get("concept", [])[:30]
        if concept:
            parts.append("\n【概念板块行情（涨跌幅前30）】")
            for r in concept[:30]:
                fields = {k: r[k] for k in ["板块名称", "涨跌幅", "换手率", "成交额"] if k in r}
                if fields:
                    parts.append("  " + " | ".join(f"{k}={v}" for k, v in fields.items()))

        # 行业主力资金流向（前20，最核心的维度）
        iflow = sector_overview.get("industry_fund_flow", [])[:20]
        if iflow:
            parts.append("\n【行业主力资金流向（今日，前20）】")
            for r in iflow[:20]:
                row_items = list(r.items())[:6]
                parts.append("  " + " | ".join(f"{k}={v}" for k, v in row_items))

        # 概念资金流向（前20）
        cflow = sector_overview.get("concept_fund_flow", [])[:20]
        if cflow:
            parts.append("\n【概念主力资金流向（今日，前20）】")
            for r in cflow[:20]:
                row_items = list(r.items())[:6]
                parts.append("  " + " | ".join(f"{k}={v}" for k, v in row_items))

        # 涨停股板块分布（热度最直接信号）
        zt_pool = market_hot.get("zt_pool", [])
        if zt_pool:
            sector_zt: Dict[str, int] = {}
            for s in zt_pool[:100]:
                sn = s.get("所属行业", s.get("行业", ""))
                if sn:
                    sector_zt[sn] = sector_zt.get(sn, 0) + 1
            top_zt = sorted(sector_zt.items(), key=lambda x: x[1], reverse=True)[:12]
            parts.append(f"\n【今日涨停股 {len(zt_pool)} 只 — 板块分布（前12）】")
            parts.append("  " + " | ".join(f"{k}({v}只)" for k, v in top_zt))

        # 热门股票（东财人气前50）
        hot = market_hot.get("hot_stocks", [])
        if hot:
            hot_sectors: Dict[str, int] = {}
            for s in hot[:50]:
                sn = s.get("所属行业", s.get("行业", ""))
                if sn:
                    hot_sectors[sn] = hot_sectors.get(sn, 0) + 1
            if hot_sectors:
                top_hot = sorted(hot_sectors.items(), key=lambda x: x[1], reverse=True)[:10]
                parts.append(f"\n【东财人气热门股板块分布（前10）】")
                parts.append("  " + " | ".join(f"{k}({v}只)" for k, v in top_hot))

        # 龙虎榜板块集中
        dt = market_hot.get("dragon_tiger", [])
        if dt:
            dt_sectors: Dict[str, int] = {}
            for s in dt[:100]:
                sn = s.get("所属行业", s.get("行业", ""))
                if sn:
                    dt_sectors[sn] = dt_sectors.get(sn, 0) + 1
            if dt_sectors:
                top_dt = sorted(dt_sectors.items(), key=lambda x: x[1], reverse=True)[:8]
                parts.append(f"\n【龙虎榜 {len(dt)} 条 — 板块集中度（前8）】")
                parts.append("  " + " | ".join(f"{k}({v}只)" for k, v in top_dt))

        # 北向资金（近10日趋势）
        north = market_hot.get("north_flow", [])
        if north:
            parts.append(f"\n【北向资金（近{min(10, len(north))}日）】")
            for r in north[-10:]:
                row_items = list(r.items())[:3]
                parts.append("  " + " | ".join(f"{k}={v}" for k, v in row_items))

        return "\n".join(parts)


# ===================================================================== #
#  GX: 跨市场主题专家（美股/港股联动，补充1个A股板块）               #
# ===================================================================== #

class GlobalThemeAdvisor(BaseAgent):
    """
    跨市场主题专家（代号 GX）。

    参考美股、港股当前热门股票与板块趋势，寻找与A股联动共振的主题。
    若A股存在对应板块且不在已选 Top5 中，全面板投票后按多数同意原则
    额外推荐至多 1 个补充板块。
    """

    EXPERT_ID = "GX"
    EXPERT_NAME = "跨市场主题专家"

    # 常见跨市场板块映射参考（提示词中嵌入，辅助LLM判断）
    _CROSS_MARKET_HINTS = """
【常见跨市场板块对应参考（仅供参考，以实际数据为准）】
美股 → A股对应：
  科技/AI大模型(NVDA/META/MSFT) → 人工智能、算力基础设施、光模块
  半导体(NVDA/AMD/QCOM)         → 半导体、集成电路、芯片设计
  新能源车(TSLA)                → 新能源汽车、动力电池、汽车电子
  生物医药(LLY/NVO/MRNA)        → 创新药、CXO、医疗器械
  云计算/SaaS                   → 云计算、信创、软件国产化
  黄金/大宗(GLD/XLE)            → 黄金珠宝、能源、有色金属
  消费(AMZN/WMT)                → 消费电子、跨境电商、商贸零售
港股 → A股对应：
  腾讯/阿里/美团                → 互联网平台、游戏、电商
  中芯国际/华虹                 → 半导体、晶圆代工
  石油(中石油HK/中海油HK)       → 油气开采、炼化及贸
  汽车(吉利/比亚迪H)            → 新能源汽车、整车制造
  房地产/物业                   → 房地产、物业管理
  金融(汇丰/中资银行H股)        → 银行、保险"""

    # ------------------------------------------------------------------ #
    #  系统提示词                                                         #
    # ------------------------------------------------------------------ #
    def _system_prompt(self) -> str:
        return f"""你是A股跨市场主题专家（代号GX），擅长识别美股、港股热门趋势与A股板块的联动共振机会。

【任务】
1. 分析当日美股、港股热门股票和涨跌主题
2. 找出与A股存在联动关系的板块主题
3. 若有A股对应板块且不在当前已选板块中，推荐最多1个补充板块

【联动逻辑三层判断】
① 直接映射：全球同一产业链（如NVDA → A股半导体/算力）
② 资金联动：外资ETF持仓调整带动A股同类标的
③ 情绪传导：全球风险偏好变化（美联储、地缘政治）→ A股防御/进攻切换

【A股补充板块的必要条件（同时满足）】
✓ 美股/港股有明确热门信号（涨幅显著或成交量异常放大）
✓ A股存在对应或高度相似的板块（产业链关联）
✓ 该板块不在当前已选的Top5板块中
✓ 近期A股该板块有跟随迹象（非冷门板块）

{self._CROSS_MARKET_HINTS}

【输出格式】
请返回如下 JSON，不包含其他文字：
```json
{{
  "global_overview": "美股/港股今日市场情绪一句话总结（40字以内）",
  "theme_resonances": [
    {{
      "global_market": "美股/港股",
      "global_theme": "热门主题/股票名",
      "a_share_sector": "对应A股板块",
      "already_selected": true或false,
      "signal_strength": 75,
      "reasoning": "联动逻辑（60字以内）"
    }}
  ],
  "extra_sector": {{
    "sector_name": "板块名称（若无补充填null）",
    "source_market": "美股/港股",
    "source_theme": "触发的海外主题",
    "a_share_match": "A股对应关系说明（40字以内）",
    "reasoning": "推荐理由（80字以内）",
    "confidence": 80
  }}
}}
```
注意：extra_sector.sector_name 填 null 表示无补充推荐。
若数据不足，可结合你对当前全球市场的认知进行判断。"""

    # ------------------------------------------------------------------ #
    #  跨市场分析主方法                                                  #
    # ------------------------------------------------------------------ #
    def advise(
        self,
        global_hot: Dict,
        current_sectors: List[str],
        panel_size: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        分析美股/港股热门主题，多模型投票决定是否补充1个A股板块。

        参数:
            global_hot      : DataEngine.fetch_global_hot() 的返回值
            current_sectors : SP 已选出的 Top5 板块名称列表
            panel_size      : 参与投票的模型数（默认全部）

        返回:
        {
          "extra_sector":      str | None,   # 补充板块名，None=无
          "extra_confidence":  int,          # 平均置信度
          "vote_count":        int,          # 赞成票数
          "total_models":      int,          # 参与模型数
          "source_market":     str,          # 美股/港股
          "source_theme":      str,          # 触发主题
          "a_share_match":     str,          # 对应关系说明
          "reasoning":         str,          # 综合理由
          "global_overview":   str,          # 市场情绪摘要
          "theme_resonances":  list,         # 所有共振主题
          "raw_results":       dict,         # 各模型原始响应
        }
        """
        if verbose:
            print(f"\n{'='*60}")
            print("  GX: 跨市场主题专家启动（美股/港股联动分析）")
            print(f"{'='*60}")

        data_text = self._build_global_text(global_hot, current_sectors)

        def _build_msgs(_: str) -> List[Dict]:
            return [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"【当前已选A股板块（Top5，无需重复推荐）】\n"
                        f"{', '.join(current_sectors) if current_sectors else '（无）'}\n\n"
                        f"【美股/港股热门数据】\n{data_text}\n\n"
                        "---\n请分析海外热门主题与A股的联动，判断是否需要额外推荐1个A股补充板块。"
                    ),
                },
            ]

        n_models = panel_size or len(self.config.providers)
        raw_results = self.llm.call_panel(
            messages=_build_msgs(""),
            max_models=n_models,
            verbose=verbose,
        )

        # ── 聚合投票 ─────────────────────────────────────────────────
        extra_votes: Dict[str, Dict] = {}   # sector_name -> vote info
        null_votes = 0
        global_overviews: List[str] = []
        all_resonances: List[Dict] = []

        for provider, text in raw_results.items():
            parsed = LLMClient.parse_json(text)
            if not parsed:
                null_votes += 1
                continue

            ov = parsed.get("global_overview", "")
            if ov:
                global_overviews.append(f"[{provider}] {ov}")

            resonances = parsed.get("theme_resonances", [])
            if isinstance(resonances, list):
                all_resonances.extend(resonances)

            extra = parsed.get("extra_sector", {})
            if not isinstance(extra, dict):
                null_votes += 1
                continue

            sector_name = extra.get("sector_name")
            if not sector_name or str(sector_name).lower() in ("null", "none", ""):
                null_votes += 1
                continue

            sector_name = str(sector_name).strip()
            # 校验：不能与已选板块重复（LLM有时会重复推荐）
            if sector_name in current_sectors:
                null_votes += 1
                continue

            if sector_name not in extra_votes:
                extra_votes[sector_name] = {
                    "sector_name": sector_name,
                    "votes": 0,
                    "conf_sum": 0,
                    "source_market": extra.get("source_market", ""),
                    "source_theme": extra.get("source_theme", ""),
                    "a_share_match": extra.get("a_share_match", ""),
                    "reasonings": [],
                    "models": [],
                }

            extra_votes[sector_name]["votes"] += 1
            extra_votes[sector_name]["conf_sum"] += float(extra.get("confidence", 70))
            reason = extra.get("reasoning", "")
            if reason:
                extra_votes[sector_name]["reasonings"].append(f"[{provider}] {reason}")
            extra_votes[sector_name]["models"].append(provider)

        total_models = len(raw_results)

        # ── 判断是否达到多数同意（> 50% 参与模型）────────────────────
        result_sector: Optional[str] = None
        result_info: Dict = {}

        if extra_votes:
            best = max(extra_votes.values(), key=lambda x: x["votes"])
            threshold = max(1, total_models // 2 + (1 if total_models % 2 else 0))  # 严格过半
            if best["votes"] >= threshold:
                result_sector = best["sector_name"]
                result_info = best

        if verbose:
            print(f"\n  [GX] 参与模型: {total_models} 个 | 无推荐票: {null_votes}")
            if extra_votes:
                print("  [GX] 补充板块投票:")
                for name, v in sorted(extra_votes.items(), key=lambda x: -x[1]["votes"]):
                    print(f"    {name}: {v['votes']}/{total_models} 票  [{', '.join(v['models'])}]")
            if result_sector:
                conf = round(result_info["conf_sum"] / result_info["votes"], 1)
                print(f"  [GX] 补充推荐: {result_sector}  置信={conf:.0f}%  通过（{result_info['votes']}/{total_models})")
            else:
                print("  [GX] 无补充推荐（未达多数同意或无差异性提案）")

        conf_val = 0
        if result_info.get("votes", 0) > 0:
            conf_val = round(result_info["conf_sum"] / result_info["votes"], 1)

        return {
            "extra_sector": result_sector,
            "extra_confidence": conf_val,
            "vote_count": result_info.get("votes", 0),
            "total_models": total_models,
            "source_market": result_info.get("source_market", ""),
            "source_theme": result_info.get("source_theme", ""),
            "a_share_match": result_info.get("a_share_match", ""),
            "reasoning": "; ".join(result_info.get("reasonings", [])[:3]),
            "global_overview": " | ".join(global_overviews[:3]),
            "theme_resonances": all_resonances[:10],
            "raw_results": raw_results,
        }

    # ------------------------------------------------------------------ #
    #  数据文本构建                                                       #
    # ------------------------------------------------------------------ #
    def _build_global_text(self, global_hot: Dict, current_sectors: List[str]) -> str:
        parts: List[str] = []
        data_available = global_hot.get("data_available", False)

        # ── 美股指数 ────────────────────────────────────────────────────
        us_indices = global_hot.get("us_indices", {})
        if us_indices:
            idx_str = " | ".join(f"{k}={v}%" for k, v in us_indices.items())
            parts.append(f"【美股主要指数】{idx_str}")

        # ── 美股热门 ────────────────────────────────────────────────────
        us_hot = global_hot.get("us_hot", [])
        if us_hot:
            parts.append(f"\n【美股热门股票 Top{min(20, len(us_hot))}】")
            for r in us_hot[:20]:
                fields = []
                for k in ["名称", "代码", "涨跌幅", "所属行业", "板块", "行业"]:
                    if k in r and r[k]:
                        fields.append(f"{k}={r[k]}")
                if fields:
                    parts.append("  " + " | ".join(fields[:5]))

        # ── 港股热门 ────────────────────────────────────────────────────
        hk_hot = global_hot.get("hk_hot", [])
        if hk_hot:
            parts.append(f"\n【港股热门股票 Top{min(20, len(hk_hot))}】")
            for r in hk_hot[:20]:
                fields = []
                for k in ["名称", "代码", "涨跌幅", "所属行业", "板块", "行业"]:
                    if k in r and r[k]:
                        fields.append(f"{k}={r[k]}")
                if fields:
                    parts.append("  " + " | ".join(fields[:5]))

        if not data_available:
            parts.append(
                "\n【注意】实时行情数据获取失败，请结合你对当前全球市场（美股/港股）"
                "最新动态的知识进行分析，重点关注近期影响较大的行业主题。"
            )

        return "\n".join(parts) if parts else "（数据暂不可用，请依赖LLM自身知识）"


# ===================================================================== #
#  S0: 板块初筛智能体                                                   #
# ===================================================================== #

class SectorScreener(BaseAgent):
    """板块初筛：从全市场板块中选出 3-5 个核心候选板块"""

    EXPERT_ID = "S0"
    EXPERT_NAME = "板块初筛"

    def _system_prompt(self) -> str:
        return """你是A股板块初筛专家（代号S0）。

你的任务是从全市场板块数据中，筛选出 3-5 个最具短期投资价值的核心候选板块。

【筛选维度】
1. 资金流向（权重30%）
   - 近5日行业资金净流入排名前20%
   - 主力资金持续涌入信号
2. 近期动量（权重25%）
   - 5日板块平均涨幅在前30%
   - 近期持续强于大盘
3. 量能配合（权重25%）
   - 成交量放大，换手活跃
   - 板块换手率≥60日均值的1.3倍以上
4. 热点匹配（权重20%）
   - 与市场热点主题高度契合
   - 有政策催化或事件驱动

【排除信号】
- 资金流向出现明显逆转（近2日连续净流出）
- 板块5日涨幅>20%（过热警报）
- 成分股数量<10只（流动性不足）
- 已连续上涨>15个交易日

【输出格式】
请以 JSON 格式返回，不要包含其他文字：
```json
{
  "top_sectors": [
    {
      "sector_name": "板块名称",
      "sector_type": "行业/概念",
      "rank": 1,
      "score": 85,
      "fund_flow": "资金流向描述",
      "momentum": "动量描述",
      "reasoning": "选择理由（150字以内）",
      "key_stocks": ["龙头股1", "龙头股2"]
    }
  ],
  "market_overview": "当前市场环境总体判断（100字以内）",
  "excluded_sectors": ["排除的板块（可为空）"]
}
```
top_sectors 数量：3-5个，按综合评分从高到低排列。"""

    def screen(
        self,
        sector_overview: Dict,
        market_hot: Dict,
        panel_size: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        执行板块初筛。
        返回: {"top_sectors": [...], "sector_names": [...], ...}
        """
        if verbose:
            print(f"\n{'='*60}")
            print("  S0: 板块初筛智能体启动")
            print(f"{'='*60}")

        # 构建数据文本
        data_parts = []

        # 申万行业前30
        sw = sector_overview.get("sw_industry", [])[:30]
        if sw:
            data_parts.append("【申万行业板块行情（涨跌幅前30）】")
            cols_of_interest = ["板块名称", "涨跌幅", "换手率", "成交额"]
            for r in sw[:30]:
                parts = []
                for c in cols_of_interest:
                    if c in r:
                        parts.append(f"{c}={r[c]}")
                if parts:
                    data_parts.append(" | ".join(parts))

        # 概念板块前30
        concept = sector_overview.get("concept", [])[:30]
        if concept:
            data_parts.append("\n【概念板块行情（涨跌幅前30）】")
            for r in concept[:30]:
                parts = []
                for c in ["板块名称", "涨跌幅", "换手率", "成交额"]:
                    if c in r:
                        parts.append(f"{c}={r[c]}")
                if parts:
                    data_parts.append(" | ".join(parts))

        # 行业资金流向前20
        iflow = sector_overview.get("industry_fund_flow", [])[:20]
        if iflow:
            data_parts.append("\n【行业资金流向（前20）】")
            for r in iflow[:20]:
                parts = []
                for c in list(r.keys())[:6]:
                    parts.append(f"{c}={r[c]}")
                data_parts.append(" | ".join(parts))

        # 概念资金流向前20
        cflow = sector_overview.get("concept_fund_flow", [])[:20]
        if cflow:
            data_parts.append("\n【概念资金流向（前20）】")
            for r in cflow[:20]:
                parts = []
                for c in list(r.keys())[:6]:
                    parts.append(f"{c}={r[c]}")
                data_parts.append(" | ".join(parts))

        # 市场热点
        zt_pool = market_hot.get("zt_pool", [])
        if zt_pool:
            data_parts.append(f"\n【今日涨停股数量: {len(zt_pool)}只】")
            # 统计涨停板块分布
            sectors_in_zt = {}
            for s in zt_pool[:50]:
                sn = s.get("所属行业", s.get("行业", ""))
                if sn:
                    sectors_in_zt[sn] = sectors_in_zt.get(sn, 0) + 1
            top_zt_sectors = sorted(sectors_in_zt.items(), key=lambda x: x[1], reverse=True)[:8]
            if top_zt_sectors:
                data_parts.append("涨停集中板块: " + " | ".join(f"{k}({v}只)" for k, v in top_zt_sectors))

        north_flow = market_hot.get("north_flow", [])
        if north_flow:
            data_parts.append(f"\n【北向资金（近10日）】")
            for r in north_flow[-10:]:
                parts = [f"{k}={v}" for k, v in list(r.items())[:3]]
                data_parts.append(" | ".join(parts))

        data_text = "\n".join(data_parts)

        # 多模型投票
        def build_msgs(name: str) -> List[Dict]:
            return [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"【市场数据】\n\n{data_text}\n\n"
                        "---\n请分析以上数据，筛选出最具投资价值的 3-5 个板块。\n"
                        "注意：请务必以 JSON 格式返回。"
                    ),
                },
            ]

        results = self.llm.call_panel(
            messages=build_msgs(""),
            max_models=panel_size or min(3, len(self.config.providers)),
            verbose=verbose,
        )

        # 聚合板块投票
        sector_votes: Dict[str, Dict] = {}
        for provider, text in results.items():
            parsed = LLMClient.parse_json(text)
            if not parsed:
                continue
            top = parsed.get("top_sectors", [])
            if not isinstance(top, list):
                continue
            for s in top:
                name = str(s.get("sector_name", "")).strip()
                if not name:
                    continue
                if name not in sector_votes:
                    sector_votes[name] = {
                        "sector_name": name,
                        "sector_type": s.get("sector_type", ""),
                        "votes": 0,
                        "total_score": 0,
                        "reasonings": [],
                        "key_stocks": s.get("key_stocks", []),
                    }
                sector_votes[name]["votes"] += 1
                sector_votes[name]["total_score"] += s.get("score", 70)
                r = s.get("reasoning", "")
                if r:
                    sector_votes[name]["reasonings"].append(f"[{provider}] {r}")

        # 排序：票数 > 平均分
        ranked = sorted(
            sector_votes.values(),
            key=lambda x: (x["votes"], x["total_score"] / max(x["votes"], 1)),
            reverse=True,
        )

        top5 = ranked[:5]
        sector_names = [s["sector_name"] for s in top5]

        if verbose:
            print(f"\n  [S0] 初筛结果: {sector_names}")

        return {
            "top_sectors": top5,
            "sector_names": sector_names,
            "raw_results": results,
        }


# ===================================================================== #
#  E1: 动量换手型专家                                                   #
# ===================================================================== #

class MomentumExpert(BaseAgent):
    """动量换手型专家：短期动量 + 量能放大"""

    EXPERT_ID = "E1"
    EXPERT_NAME = "动量换手型专家"

    def _system_prompt(self) -> str:
        return """你是A股动量换手型选股专家（代号E1），专注于识别短期强势动量股。

【核心逻辑】："强动量 + 高换手 = 资金共识形成，趋势处于初中段"

【分析框架】
1. 价格动量（权重40%）
   - 5日涨幅 > 3%，且在20日均线以上（优选突破20日均线）
   - 5日涨幅排名靠前，但未连续5日涨停（过热警戒）
   - 均线呈多头排列：MA5 > MA10 > MA20

2. 量能换手（权重35%）
   - 近3日均换手率 > 60日均值的1.5倍（放量）
   - 成交量递增趋势（3日>5日>10日均量）
   - 量比 > 1.5（今日成交活跃）

3. 情绪信号（权重25%）
   - 近期有涨停或连续上涨记录
   - 近5日主力净流入为正
   - MACD在零轴上方且DIF>DEA（多头信号）
   - KDJ未严重超买（J值<100）

【加分项】
- 板块龙头地位（市值居中，非最大）
- 突破60日/52周高点
- 近3日出现缩量回调后再放量

【扣分/排除】
- 5日涨幅>25%（短期涨幅过大）
- 换手率连续5日>5%（过热）
- MA5死叉MA10
- 主力连续净流出

【持仓策略】
- 持有周期：3-10个交易日
- 止损：跌破MA5均线或最高点回撤-7%""" + _RESEARCH_INSTRUCTION

    # analyze() 直接继承自 BaseAgent，无需重写


# ===================================================================== #
#  E2: 成长估值型专家                                                   #
# ===================================================================== #

class GrowthExpert(BaseAgent):
    """成长估值型专家：基本面成长 + 合理估值 + 技术确认"""

    EXPERT_ID = "E2"
    EXPERT_NAME = "成长估值型专家"

    def _system_prompt(self) -> str:
        return """你是A股成长估值型选股专家（代号E2），从中线视角寻找兼具成长性和安全边际的股票。

【核心逻辑】："好赛道 + 好业绩 + 合理估值 = 安全边际 + 弹性空间"

【分析框架】
1. 成长指标（权重40%）
   - 最近一季度营收同比增长 > 15%（核心指标）
   - 净利润增长 > 10%，且方向向上
   - ROE > 12%（盈利质量好）
   - 毛利率稳定或提升（护城河信号）

2. 估值弹性（权重30%）
   - 总市值在50-500亿（弹性区间，非巨无霸）
   - PE处于行业合理区间（非极端高估）
   - PEG < 1.5（成长匹配估值）
   - PB < 5（非恶意溢价）

3. 技术确认（权重30%）
   - 日线上升趋势（MA20向上，价格>MA20）
   - 量能适度放大（非暴量）
   - 月线和周线同向（三线共振更佳）
   - 近20日主力净流入为正或转正

【排除信号】
- 营收连续2期下滑
- 净利润由正转负
- 商誉占净资产>30%（商誉雷风险）
- 资产负债率>70%（高杠杆）
- 大股东质押率>80%
- 近期有减持公告（重要股东大规模）

【持仓策略】
- 持有周期：10-30个交易日
- 止损：跌破MA20均线或最高点回撤-10%""" + _RESEARCH_INSTRUCTION


# ===================================================================== #
#  E3: 多因子平衡型专家                                                 #
# ===================================================================== #

class MultiFactorExpert(BaseAgent):
    """多因子平衡型专家：质量 × 动量 × 流动性的综合评分"""

    EXPERT_ID = "E3"
    EXPERT_NAME = "多因子平衡型专家"

    def _system_prompt(self) -> str:
        return """你是A股多因子平衡型选股专家（代号E3），通过综合打分模型选出最稳健的强势股。

【核心逻辑】："不选最强，选最稳的强——质量×动量×流动性三维平衡"

【综合评分模型（满分100分）】
1. 质量因子（35分）
   - ROE ≥ 15%：+10分；10%-15%：+6分；<10%：+2分
   - 毛利率 ≥ 30%：+8分；20%-30%：+5分；<20%：+2分
   - 净利率 ≥ 10%：+8分；5%-10%：+5分；<5%：+2分
   - 资产负债率 < 50%：+5分；50%-70%：+3分；>70%：0分
   - 近两期净利润持续增长：+4分

2. 动量因子（35分）
   - 20日涨幅排名前10%：+12分；前30%：+8分；前50%：+4分
   - 5日涨幅>3%且<15%：+8分（动量好但未过热）
   - 日线均线多头排列（MA5>MA10>MA20）：+8分
   - MACD零轴上方金叉：+5分；金叉在零轴下方：+3分
   - ⚠️ 5日涨幅>20%：扣10分（过热惩罚）

3. 流动性因子（30分）
   - 20日日均成交额>1亿：+10分；>5000万：+6分
   - 近5日换手率/60日均值：1.2-2.5倍得+10分；2.5-4倍得+5分；>4倍得0分
   - 量价配合良好（放量上涨/缩量回调）：+10分

【特殊规则】
- S级（90+分）：重点推荐
- A级（80-89）：优先推荐
- B级（70-79）：正常推荐
- C级及以下：不推荐

【持仓策略】
- 持有周期：5-20个交易日
- 止损：跌破MA10均线或-8%""" + _RESEARCH_INSTRUCTION


# ===================================================================== #
#  E4: 技术形态型专家                                                   #
# ===================================================================== #

class TechnicalExpert(BaseAgent):
    """技术形态型专家：纯技术分析，多信号共振（支持K线图视觉输入）"""

    EXPERT_ID = "E4"
    EXPERT_NAME = "技术形态型专家"

    def run_in_session(
        self,
        session: ConversationSession,
        stock_profiles: Dict[str, str],
        stock_packages: Optional[Dict] = None,
        include_profiles: bool = True,
    ) -> ExpertResult:
        """覆盖基类：调用 say_with_vision() 以支持 K 线图输入"""
        if include_profiles:
            data_text = "\n\n".join(
                f"--- {code} ---\n{profile}"
                for code, profile in stock_profiles.items()
            )
            content = (
                f"【候选股票数据（含技术指标）】\n\n{data_text}\n\n"
                "---\n"
                f"请以 {self.EXPERT_NAME}（{self.EXPERT_ID}）的视角，"
                "重点关注K线形态、均线系统、MACD/RSI/KDJ等技术指标共振，"
                "给出你的技术面选股推荐。\n\n"
                f"{_PICK_FORMAT}"
            )
        else:
            content = (
                f"请以 {self.EXPERT_NAME}（{self.EXPERT_ID}）的视角，"
                "基于对话中已提供的股票数据，重点关注K线形态、均线系统、MACD/RSI/KDJ等技术指标共振，"
                "独立给出你的技术面选股推荐。\n\n"
                f"{_PICK_FORMAT}"
            )

        # 从 stock_packages 获取K线图路径（最多5张）
        image_paths: List[str] = []
        if stock_packages:
            for _c in list(stock_profiles.keys())[:5]:
                _cp = (stock_packages.get(_c) or {}).get("chart_path")
                if _cp and os.path.exists(_cp):
                    image_paths.append(_cp)
        resp = session.say_with_vision(content, image_paths=image_paths)

        picks: List[StockPick] = []
        if resp:
            parsed = LLMClient.parse_json(resp)
            if parsed and "picks" in parsed:
                for pick in parsed["picks"]:
                    if not isinstance(pick, dict):
                        continue
                    code = str(pick.get("code", "")).strip()
                    if not code:
                        continue
                    picks.append(StockPick(
                        code=code,
                        name=str(pick.get("name", "")),
                        sector=str(pick.get("sector", "")),
                        expert_id=self.EXPERT_ID,
                        stars=float(pick.get("stars", 3)),
                        score=float(pick.get("score", 70)),
                        reasoning=str(pick.get("reasoning", ""))[:300],
                        hold_period=str(pick.get("hold_period", "3-15日")),
                        stop_loss=str(pick.get("stop_loss", "跌破买入日最低价")),
                        warning=str(pick.get("warning", "")),
                    ))

        return ExpertResult(
            expert_id=self.EXPERT_ID,
            expert_name=self.EXPERT_NAME,
            picks=picks[:10],
            reasoning=f"{self.EXPERT_NAME}完成Session分析，共选出{len(picks)}只股票",
            debate_log={"session_response": resp or ""},
        )

    def _system_prompt(self) -> str:
        return """你是A股技术形态分析专家（代号E4），通过多维技术指标共振识别最佳买入时机。

【核心逻辑】："价格包含一切信息。多个技术信号共振 = 趋势启动前夕"

【技术信号评分体系】
1. 均线系统（权重30%）
   - MA5/MA10/MA20/MA60完全多头排列：满分
   - MA均线同向向上+均线间距扩张：高分
   - MA20上穿MA60（长期金叉）：重要加分
   - 价格上穿MA60（半年线突破）：重要加分

2. K线形态（权重25%）
   - 突破形态：放量突破前期平台/箱体/颈线位：高分
   - 反转形态：底部锤子线、晨星、吞没形态：中高分
   - 持续形态：旗形、矩形整理后的方向选择：中分
   - 量能确认：突破时成交量≥前5日均量的1.5倍

3. 技术指标共振（权重30%）
   - MACD：DIF上穿DEA（金叉）+柱体由负转正：+高分
   - RSI(6)：从超卖(<30)回升或从50附近上穿：+中分
   - KDJ：J值金叉且K值在30-70区间：+中分
   - 多指标同时给出买入信号：额外加分

4. 量价关系（权重15%）
   - 健康放量上涨：每日成交量递增配合价格上升
   - 缩量回调后放量突破：最优形态
   - 地量后异常放量：关注方向

【关键形态识别（多周期共振最优）】
- 月线：处于上升趋势（多头排列）
- 周线：从低位向上启动或突破整理区
- 日线：买入信号出现，量能配合

【持仓策略】
- 持有周期：3-15个交易日
- 止损：跌破买入日最低价或前期支撑位""" + _RESEARCH_INSTRUCTION


# ===================================================================== #
#  E5: 资金流向型专家                                                   #
# ===================================================================== #

class CapitalFlowExpert(BaseAgent):
    """资金流向型专家：追踪主力资金行为（支持资金流图视觉输入）"""

    EXPERT_ID = "E5"
    EXPERT_NAME = "资金流向型专家"

    def run_in_session(
        self,
        session: ConversationSession,
        stock_profiles: Dict[str, str],
        stock_packages: Optional[Dict] = None,
        include_profiles: bool = True,
    ) -> ExpertResult:
        """覆盖基类：调用 say_with_vision() 以支持资金流图输入"""
        if include_profiles:
            data_text = "\n\n".join(
                f"--- {code} ---\n{profile}"
                for code, profile in stock_profiles.items()
            )
            content = (
                f"【候选股票数据（含资金流向）】\n\n{data_text}\n\n"
                "---\n"
                f"请以 {self.EXPERT_NAME}（{self.EXPERT_ID}）的视角，"
                "重点关注主力资金净流入、超大单/大单净买入、北向资金及机构动向，"
                "给出你的资金面选股推荐。\n\n"
                f"{_PICK_FORMAT}"
            )
        else:
            content = (
                f"请以 {self.EXPERT_NAME}（{self.EXPERT_ID}）的视角，"
                "基于对话中已提供的股票数据，重点关注主力资金净流入、超大单/大单净买入、北向资金及机构动向，"
                "独立给出你的资金面选股推荐。\n\n"
                f"{_PICK_FORMAT}"
            )

        # image_paths 留空（当前数据引擎不生成图表）
        image_paths: List[str] = []
        resp = session.say_with_vision(content, image_paths=image_paths)

        picks: List[StockPick] = []
        if resp:
            parsed = LLMClient.parse_json(resp)
            if parsed and "picks" in parsed:
                for pick in parsed["picks"]:
                    if not isinstance(pick, dict):
                        continue
                    code = str(pick.get("code", "")).strip()
                    if not code:
                        continue
                    picks.append(StockPick(
                        code=code,
                        name=str(pick.get("name", "")),
                        sector=str(pick.get("sector", "")),
                        expert_id=self.EXPERT_ID,
                        stars=float(pick.get("stars", 3)),
                        score=float(pick.get("score", 70)),
                        reasoning=str(pick.get("reasoning", ""))[:300],
                        hold_period=str(pick.get("hold_period", "5-20日")),
                        stop_loss=str(pick.get("stop_loss", "主力连续净流出-8%")),
                        warning=str(pick.get("warning", "")),
                    ))

        return ExpertResult(
            expert_id=self.EXPERT_ID,
            expert_name=self.EXPERT_NAME,
            picks=picks[:10],
            reasoning=f"{self.EXPERT_NAME}完成Session分析，共选出{len(picks)}只股票",
            debate_log={"session_response": resp or ""},
        )

    def _system_prompt(self) -> str:
        return """你是A股资金流向追踪专家（代号E5），专注于识别主力资金流入的股票。

【核心逻辑】："跟随聪明钱。持续净流入 = 主力建仓，是股价上涨的核心驱动力"

【分析维度】
1. 主力资金净流入（权重35%）
   - 近5日主力净流入持续为正：强烈买入信号
   - 净流入加速（每日流入量递增）：极强信号
   - 单日超大单净流入>5000万：机构性买入

2. 筹码结构（权重25%）
   - 近期换手率适度（1%-3%/日），筹码逐渐集中
   - 大单持续买入，散单持续卖出（筹码交换中）
   - 融资余额持续增加（杠杆资金看多）

3. 北向资金（权重20%）
   - 沪深股通持续净买入该股
   - 外资持仓比例上升
   - 近5日北向合计净流入为正

4. 机构动向（权重20%）
   - 最近一期机构调研频次增加
   - 融资买入占比提升（>30%）
   - 龙虎榜机构席位净买入

【排除信号（一票否决）】
- 近5日主力连续净流出（资金撤离）
- 超大单连续卖出>3日
- 北向连续净卖出且持仓比例下降
- 融券余量快速上升（空头增仓）

【关注模式】
- 吸筹模式：长时间低换手+主力净流入
- 启动前模式：换手率开始放大+净流入加速
- 拉升模式：高换手+净流入+价格快速上涨

【持仓策略】
- 持有周期：5-20个交易日
- 止损：主力连续2日净流出或-8%""" + _RESEARCH_INSTRUCTION


# ===================================================================== #
#  E6: 事件催化型专家                                                   #
# ===================================================================== #

class CatalystExpert(BaseAgent):
    """事件催化型专家：识别催化剂和异动信号"""

    EXPERT_ID = "E6"
    EXPERT_NAME = "事件催化型专家"

    def _system_prompt(self) -> str:
        return """你是A股事件催化型选股专家（代号E6），专注于识别可能引发股价重估的催化事件和异动信号。

【核心逻辑】："催化剂 = 短期Alpha的来源。找到重新定价事件，赚取事件窗口收益"

【催化剂类型评分】
1. 宏观政策催化（权重25%）
   - 国务院/财政部/央行重大政策利好：高分
   - 行业主管部门新政策：中高分
   - 地方重大专项支持：中分

2. 行业事件（权重25%）
   - 行业重大技术突破/产品发布
   - 需求端爆发（订单超预期）
   - 价格上涨（大宗商品/产品提价）
   - 竞争格局改善（行业整合/龙头确立）

3. 公司催化（权重30%）
   - 业绩超预期（业绩预告/快报明显好于预期）
   - 重大合同/订单签订
   - 并购重组预期（资产注入）
   - 回购增持（显示公司对低估值的自信）
   - 股权激励（绑定管理层利益）

4. 市场情绪催化（权重20%）
   - 分析师密集上调评级
   - 媒体/论坛热度骤升
   - 概念板块异动龙头带动
   - 北向资金集中涌入该股

【异动信号识别（无需公告也可判断）】
- 突然放量但价格平稳（内部消息吸筹）
- 盘后大宗交易折价成交（机构换手）
- 融资余额短期急剧上升（杠杆资金追入）
- 龙虎榜连续出现知名游资/机构席位

【定价评估】
- 未定价/低估：事件刚发生，市场反应慢→买入机会
- 部分定价：已上涨5-15%，仍有上涨空间
- 充分定价：已反映大部分利好→谨慎
- 过度定价：明显透支预期→回避

【持仓策略】
- 持有周期：1-15个交易日（事件驱动窗口期）
- 止损：逻辑证伪（事件反转）或-7%""" + _RESEARCH_INSTRUCTION


# ===================================================================== #
#  E7: 板块优股专家                                                     #
# ===================================================================== #

class SectorOutperformer(BaseAgent):
    """板块优股专家：识别近一个月内走势结构、量能、财务均优于本板块均值的个股"""

    EXPERT_ID = "E7"
    EXPERT_NAME = "板块优股专家"

    # ------------------------------------------------------------------ #
    #  内部工具：解析市值字符串为浮点数（亿元）                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_mktcap(val) -> float:
        """将各种格式的市值转换为亿元浮点数"""
        if val is None:
            return 0.0
        s = str(val).replace(",", "").strip()
        if "亿" in s:
            try:
                return float(s.replace("亿", ""))
            except ValueError:
                return 0.0
        if "万" in s:
            try:
                return float(s.replace("万", "")) / 10000
            except ValueError:
                return 0.0
        try:
            v = float(s)
            # 原始数字：>1e8 按元计，否则按亿计
            return v / 1e8 if v > 1e6 else v
        except ValueError:
            return 0.0

    # ------------------------------------------------------------------ #
    #  板块基准计算                                                       #
    # ------------------------------------------------------------------ #
    def _compute_sector_benchmarks(
        self, stock_packages: Dict
    ) -> Dict[str, Dict]:
        """
        按板块分组，计算近20日相对表现基准。
        返回: {sector_name: {
            avg_d20, avg_vol_ratio, avg_turnover5,
            leaders: [code, name, ...],  stock_count
        }}
        """
        # 分组
        sector_stocks: Dict[str, List] = {}
        for code, pkg in stock_packages.items():
            sname = pkg.get("sector", "未知") or "未知"
            sector_stocks.setdefault(sname, []).append((code, pkg))

        benchmarks: Dict[str, Dict] = {}
        for sname, stocks in sector_stocks.items():
            d20_list: List[float] = []
            vol_ratio_list: List[float] = []
            turnover_list: List[float] = []
            cap_return_map: Dict[str, Tuple[float, float]] = {}  # code -> (cap, d20)

            for code, pkg in stocks:
                df_d = pkg.get("daily")
                d20_val = 0.0
                if df_d is not None and len(df_d) >= 2:
                    try:
                        c = df_d["close"]
                        n_back = min(20, len(c) - 1)
                        d20_val = (float(c.iloc[-1]) - float(c.iloc[-(n_back+1)])) / max(float(c.iloc[-(n_back+1)]), 0.01) * 100
                        d20_list.append(d20_val)
                        v5 = float(df_d["volume"].tail(5).mean())
                        v20 = float(df_d["volume"].tail(20).mean())
                        if v20 > 0:
                            vol_ratio_list.append(v5 / v20)
                        if "turnover" in df_d.columns:
                            to5 = float(df_d["turnover"].tail(5).mean())
                            turnover_list.append(to5)
                    except Exception:
                        pass

                cap = self._parse_mktcap(pkg.get("realtime", {}).get("总市值"))
                cap_return_map[code] = (cap, d20_val)

            # 板块龙头：先按市值排序，市值=0时按20日收益率补充
            valid_caps = [(c, v[0], v[1]) for c, v in cap_return_map.items() if v[0] > 0]
            if valid_caps:
                valid_caps.sort(key=lambda x: x[1], reverse=True)
            else:
                valid_caps = [(c, 0.0, v[1]) for c, v in cap_return_map.items()]
                valid_caps.sort(key=lambda x: x[2], reverse=True)

            leaders = [c for c, _, _ in valid_caps[:2]]

            benchmarks[sname] = {
                "avg_d20": round(sum(d20_list) / len(d20_list), 2) if d20_list else 0.0,
                "avg_vol_ratio": round(sum(vol_ratio_list) / len(vol_ratio_list), 2) if vol_ratio_list else 1.0,
                "avg_turnover5": round(sum(turnover_list) / len(turnover_list), 2) if turnover_list else 0.0,
                "leaders": leaders,
                "stock_count": len(stocks),
            }

        return benchmarks

    # ------------------------------------------------------------------ #
    #  Session 调用（覆盖基类）                                          #
    # ------------------------------------------------------------------ #
    def run_in_session(
        self,
        session: ConversationSession,
        stock_profiles: Dict[str, str],
        stock_packages: Optional[Dict] = None,
        include_profiles: bool = True,
    ) -> ExpertResult:
        """在 ConversationSession 中运行板块优股分析，注入板块基准数据"""
        # ── 计算板块基准 ──────────────────────────────────────────────
        benchmarks: Dict[str, Dict] = {}
        if stock_packages:
            benchmarks = self._compute_sector_benchmarks(stock_packages)

        bench_lines = ["【板块基准数据（近20日均值，用于个股对比）】"]
        for sname, bm in benchmarks.items():
            leader_str = "、".join(bm["leaders"][:2]) if bm["leaders"] else "N/A"
            bench_lines.append(
                f"  {sname}（{bm['stock_count']}只）| "
                f"板块均涨幅={bm['avg_d20']:+.1f}% | "
                f"均量比={bm['avg_vol_ratio']:.2f}x | "
                f"均换手(5日)={bm['avg_turnover5']:.2f}% | "
                f"龙头1/2={leader_str}"
            )
        bench_text = "\n".join(bench_lines)

        # ── 构建 prompt ───────────────────────────────────────────────
        if include_profiles:
            data_text = "\n\n".join(
                f"--- {code} ---\n{profile}"
                for code, profile in stock_profiles.items()
            )
            content = (
                f"【候选股票数据】\n\n{data_text}\n\n"
                f"{bench_text}\n\n"
                "---\n"
                f"请以 {self.EXPERT_NAME}（{self.EXPERT_ID}）的视角，"
                "对比上方板块基准数据，识别近一个月走势结构、量能、财务均优于本板块的个股。\n\n"
                f"{_PICK_FORMAT}"
            )
        else:
            content = (
                f"{bench_text}\n\n"
                "---\n"
                f"请以 {self.EXPERT_NAME}（{self.EXPERT_ID}）的视角，"
                "基于对话中已有的股票数据，并结合以上板块基准，"
                "识别近一个月走势、量能、财务均显著优于本板块均值的个股。\n"
                "若该股本身是龙头1/2，可豁免板块对比，直接按绝对强势判断。\n\n"
                f"{_PICK_FORMAT}"
            )

        resp = session.say(content)

        picks: List[StockPick] = []
        if resp:
            parsed = LLMClient.parse_json(resp)
            if parsed and "picks" in parsed:
                for pick in parsed["picks"]:
                    if not isinstance(pick, dict):
                        continue
                    code = str(pick.get("code", "")).strip()
                    if not code:
                        continue
                    picks.append(StockPick(
                        code=code,
                        name=str(pick.get("name", "")),
                        sector=str(pick.get("sector", "")),
                        expert_id=self.EXPERT_ID,
                        stars=float(pick.get("stars", 3)),
                        score=float(pick.get("score", 70)),
                        reasoning=str(pick.get("reasoning", ""))[:300],
                        hold_period=str(pick.get("hold_period", "10-20日")),
                        stop_loss=str(pick.get("stop_loss", "跌破MA20")),
                        warning=str(pick.get("warning", "")),
                    ))

        return ExpertResult(
            expert_id=self.EXPERT_ID,
            expert_name=self.EXPERT_NAME,
            picks=picks[:10],
            reasoning=f"{self.EXPERT_NAME}完成Session分析，共选出{len(picks)}只股票",
            debate_log={"session_response": resp or ""},
        )

    def _system_prompt(self) -> str:
        return """你是A股板块优股专家（代号E7），专注于识别近一个月内相对本板块整体走势更强的个股。

【核心逻辑】："强板块中的更强个股 = 资金精选 + 超额Alpha来源"

【评判维度（对比板块基准数据）】
1. 近一个月价格走势（权重35%）
   - 近20日涨幅高于板块均涨幅（超额>3%为优，>8%为强）
   - K线结构更健康：均线多头排列、无大幅破位
   - 板块整体回调时该股跌幅更小（抗跌性好）
   - 周线/月线趋势方向更顺畅

2. 量能与换手（权重25%）
   - 近5日量比高于板块均值（均量比 > 板块均量比×1.2为优）
   - 近5日换手率活跃度优于板块均换手
   - 上涨日放量、下跌日缩量（量价配合优于板块均水平）

3. 基本面与财务优势（权重25%）
   - 营收增速、净利润增速高于板块同类均值
   - ROE、毛利率优于板块平均水平
   - 资产负债率低于板块均值（财务结构更健康）

4. 龙头对比（权重15%）
   - 若该股非龙头1/2：其走势、量能、财务是否接近甚至超越龙头
   - 若该股本身是龙头1/2：直接按绝对强势判断，无需与其他股对比

【优选特征】
- 同一板块横向对比，该股20日超额收益在前30%
- 量能活跃度、换手率在板块前30%
- 基本面数据在板块中属于头部20%
- 资金流向优于板块均值

【排除特征】
- 近20日涨幅显著落后板块均值（负超额>5%）
- 量能萎缩，换手长期低迷
- 财务数据在板块中属于后30%
- 股价趋势破位（跌破MA20或MA60）

【持仓策略】
- 持有周期：10-20个交易日
- 止损：跌破MA20或相对板块开始明显落后""" + _RESEARCH_INSTRUCTION


# ===================================================================== #
#  V: 投票仲裁者                                                        #
# ===================================================================== #

class VotingArbitrator:
    """
    投票仲裁者：
    1. 汇总7位专家的选股结果
    2. 加权投票计算综合得分
    3. 多模型辩论审视（可选）
    4. 输出最终排序候选列表
    """

    EXPERT_WEIGHTS = {
        "E1": 0.13,
        "E2": 0.17,
        "E3": 0.17,
        "E4": 0.13,
        "E5": 0.12,
        "E6": 0.13,
        "E7": 0.15,
    }

    def __init__(self, llm_client: LLMClient, config: Config):
        self.llm = llm_client
        self.config = config
        weights = config.expert_weights or {}
        if weights:
            self.EXPERT_WEIGHTS = {k: float(v) for k, v in weights.items()}

    def _aggregate(self, expert_results: List[ExpertResult]) -> Dict[str, Dict]:
        """加权聚合专家结论"""
        pool: Dict[str, Dict] = {}

        for er in expert_results:
            weight = self.EXPERT_WEIGHTS.get(er.expert_id, 0.15)
            for pick in er.picks:
                code = pick.code
                if code not in pool:
                    pool[code] = {
                        "code": code,
                        "name": pick.name,
                        "sector": pick.sector,
                        "expert_stars": {},
                        "weighted_score": 0.0,
                        "expert_count": 0,
                        "pick_details": {},
                        "all_reasonings": [],
                        "warnings": [],
                        "hold_periods": [],
                    }

                pool[code]["expert_stars"][er.expert_id] = pick.stars
                pool[code]["weighted_score"] += pick.stars * weight * 20  # 转化为百分制
                pool[code]["expert_count"] += 1
                pool[code]["pick_details"][er.expert_id] = pick
                if pick.reasoning:
                    pool[code]["all_reasonings"].append(f"[{er.expert_id}] {pick.reasoning}")
                if pick.warning:
                    pool[code]["warnings"].append(pick.warning)
                if pick.hold_period:
                    pool[code]["hold_periods"].append(pick.hold_period)

                # 更新名称
                if pick.name and not pool[code]["name"]:
                    pool[code]["name"] = pick.name

        # 添加共识奖励
        for code, data in pool.items():
            n = data["expert_count"]
            if n >= 4:
                data["consensus_bonus"] = 1.5
                data["consensus_level"] = "强共识"
            elif n >= 3:
                data["consensus_bonus"] = 1.0
                data["consensus_level"] = "中共识"
            elif n >= 2:
                data["consensus_bonus"] = 0.5
                data["consensus_level"] = "弱共识"
            else:
                data["consensus_bonus"] = 0.0
                data["consensus_level"] = "单一支持"

            data["final_score"] = round(
                data["weighted_score"] + data["consensus_bonus"] * 5, 2
            )

        return pool

    def _build_arbitration_prompt(self, pool: Dict[str, Dict], provider_name: str) -> List[Dict]:
        """构建仲裁提示词"""
        # 构建候选股票摘要
        candidates_text = []
        for code, data in sorted(
            pool.items(),
            key=lambda x: x[1]["final_score"],
            reverse=True,
        )[:20]:
            expert_stars_str = " ".join(
                f"{eid}:{stars}★" for eid, stars in data["expert_stars"].items()
            )
            reason_summary = "; ".join(data["all_reasonings"][:3])[:200]
            candidates_text.append(
                f"• {code} {data['name']} [{data['sector']}]"
                f" 共识={data['consensus_level']}({data['expert_count']}/6专家)"
                f" 加权分={data['final_score']:.1f}"
                f" 各专家评星: {expert_stars_str}"
                f"\n  综合理由: {reason_summary}"
            )

        system_prompt = """你是A股多智能体选股系统的投票仲裁者。

你的职责是：
1. 审视7位专家的选股结论
2. 识别多专家共识的优质标的
3. 对存在分歧的标的给出裁决意见
4. 输出最终推荐的股票排名（5-15只）

【评判标准】
优先推荐：
- 4个以上专家认可（强共识）
- 加权得分>80分
- 多种投资逻辑共振（如动量+基本面+资金共振）

次优推荐：
- 3个专家认可+至少1个高分（4-5星）专家
- 加权得分75-80分

可选推荐：
- 2个专家认可但信号特别强（如E5资金+E4技术双确认）
- 单一5星推荐（特殊情况补充）

板块集中度：同一板块不超过3只。"""

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"【7位专家的综合选股结果】\n\n"
                    + "\n\n".join(candidates_text)
                    + "\n\n---\n"
                    "请基于以上专家共识，给出最终推荐排名。\n"
                    "输出 JSON 格式：\n"
                    "```json\n"
                    "{\n"
                    '  "final_picks": [\n'
                    "    {\n"
                    '      "rank": 1,\n'
                    '      "code": "股票代码",\n'
                    '      "name": "股票名称",\n'
                    '      "sector": "板块",\n'
                    '      "consensus_level": "共识等级",\n'
                    '      "final_score": 数值,\n'
                    '      "recommendation": "重点关注/积极关注/观察关注",\n'
                    '      "core_logic": "核心投资逻辑（150字以内）",\n'
                    '      "hold_period": "建议持有周期",\n'
                    '      "key_risk": "主要风险点"\n'
                    "    }\n"
                    "  ],\n"
                    '  "debate_note": "辩论/分歧说明（如有）",\n'
                    '  "market_view": "当前市场环境综合判断"\n'
                    "}\n```\n"
                    "final_picks 数量：5-15只，按推荐强度从高到低排列。"
                ),
            },
        ]

    def arbitrate(
        self,
        expert_results: List[ExpertResult],
        run_debate: bool = True,
        panel_size: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        执行投票仲裁。
        返回最终排序的候选股票列表。
        """
        if verbose:
            print(f"\n{'='*60}")
            print("  V: 投票仲裁者启动")
            print(f"{'='*60}")

        # Step 1: 聚合所有专家结论
        pool = self._aggregate(expert_results)

        if verbose:
            print(f"  候选池: {len(pool)} 只股票")
            for code, data in sorted(pool.items(), key=lambda x: x[1]["final_score"], reverse=True)[:10]:
                print(
                    f"    {code} {data['name']}: "
                    f"共识={data['consensus_level']} "
                    f"综合分={data['final_score']:.1f} "
                    f"专家数={data['expert_count']}"
                )

        # Step 2: 多模型辩论仲裁
        if run_debate and len(self.config.providers) > 0:
            if verbose:
                print("\n  [仲裁] 多模型辩论仲裁...")

            arb_results = self.llm.call_panel(
                messages=self._build_arbitration_prompt(pool, ""),
                max_models=panel_size or min(3, len(self.config.providers)),
                verbose=verbose,
            )

            # 从仲裁结果中提取最终排名
            final_picks = []
            score_map: Dict[str, float] = {}

            for provider, text in arb_results.items():
                parsed = LLMClient.parse_json(text)
                if not parsed:
                    continue
                picks = parsed.get("final_picks", [])
                for pick in picks:
                    code = str(pick.get("code", "")).strip()
                    if not code:
                        continue
                    rank = pick.get("rank", 99)
                    if code not in score_map:
                        score_map[code] = 0
                    score_map[code] += (20 - min(rank, 20))  # 排名越高得分越高

            # 按得分排序，取前15
            sorted_codes = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:15]

            for rank, (code, _) in enumerate(sorted_codes, 1):
                if code in pool:
                    pd_data = pool[code]
                    final_picks.append({
                        "rank": rank,
                        "code": code,
                        "name": pd_data["name"],
                        "sector": pd_data["sector"],
                        "consensus_level": pd_data["consensus_level"],
                        "final_score": pd_data["final_score"],
                        "expert_stars": pd_data["expert_stars"],
                        "expert_count": pd_data["expert_count"],
                        "all_reasonings": pd_data["all_reasonings"],
                        "warnings": pd_data["warnings"],
                    })

        else:
            # 无辩论：直接按综合分排序
            sorted_pool = sorted(pool.items(), key=lambda x: x[1]["final_score"], reverse=True)
            final_picks = []
            for rank, (code, data) in enumerate(sorted_pool[:15], 1):
                final_picks.append({
                    "rank": rank,
                    "code": code,
                    **data,
                })

        if verbose:
            print(f"\n  [仲裁] 最终候选: {len(final_picks)} 只")
            for p in final_picks[:5]:
                print(f"    {p['rank']}. {p['code']} {p.get('name', '')} 共识={p.get('consensus_level', '')}")

        return {
            "final_picks": final_picks,
            "candidate_pool": pool,
            "total_candidates": len(pool),
        }


# ===================================================================== #
#  R: 风控审查                                                           #
# ===================================================================== #

class RiskController:
    """
    风控审查：
    1. 硬性排除（一票否决）
    2. 风险信号识别（黄灯/红灯）
    3. 仓位建议
    4. 止损参考
    """

    HARD_EXCLUSION_KEYWORDS = [
        "ST", "*ST", "退市", "暂停上市", "证监会立案",
        "财务造假", "强制退市", "连续跌停",
    ]

    def __init__(self, llm_client: LLMClient, config: Config):
        self.llm = llm_client
        self.config = config

    def _build_risk_prompt(self, candidates: List[Dict], stock_packages: Dict) -> List[Dict]:
        """构建风控审查提示词"""
        system = """你是A股风险控制专家，负责对候选股票进行最终风险审查。

【硬性排除（一票否决）】
- ST/*ST 或退市警示股
- 证监会立案调查中
- 大股东质押率>90%（爆仓风险）
- 近30日出现连续跌停（≥3次）
- 净利润连续3期为负（持续亏损）
- 财务数据严重异常

【风险信号】
黄灯⚠️（提示仓位减半）:
- 近60日涨幅>30%（短期涨幅较大）
- 日换手率>5%（连续3日，过热）
- 大股东减持公告（小规模）
- 近期商誉占净资产20-30%

红灯🔴（建议排除或极小仓位<5%）:
- 近60日涨幅>50%
- 换手率>8%（连续3日）
- 大股东大规模减持
- 商誉占净资产>30%
- 现金流连续4期为负

【仓位建议】
- 低风险（无信号）：标准仓位 10-15%
- 中风险（1-2个黄灯）：轻仓 5-10%
- 高风险（3+黄灯或1个红灯）：极轻仓 <5% 或排除

【止损建议】
- 短线（3-10日）：跌破MA5或最高点回落7%
- 中线（10-30日）：跌破MA20或最高点回落10%
- 事件驱动：逻辑证伪即出"""

        candidates_text = []
        for c in candidates[:15]:
            code = c.get("code", "")
            name = c.get("name", "")
            sector = c.get("sector", "")
            score = c.get("final_score", 0)
            consensus = c.get("consensus_level", "")
            warnings = c.get("warnings", [])
            warn_str = "；".join(warnings[:3]) if warnings else "暂无专家风险提示"

            # 从 stock_packages 取关键指标
            pkg = stock_packages.get(code, {})
            rt = pkg.get("realtime", {})
            pe = rt.get("市盈率-动态", rt.get("市盈率", "N/A"))
            pb = rt.get("市净率", "N/A")
            mktcap = rt.get("总市值", "N/A")

            d60_gain = "N/A"
            df_d = pkg.get("daily")
            if df_d is not None and len(df_d) >= 2:
                try:
                    n_back = min(60, len(df_d) - 1)
                    old = float(df_d["close"].iloc[-(n_back+1)])
                    new = float(df_d["close"].iloc[-1])
                    d60_gain = f"{(new-old)/old*100:.1f}%({n_back}日)"
                except Exception:
                    pass

            candidates_text.append(
                f"• {code} {name} [{sector}]\n"
                f"  共识={consensus} 综合分={score:.1f}\n"
                f"  PE={pe} PB={pb} 总市值={mktcap}\n"
                f"  近60日涨幅={d60_gain}\n"
                f"  专家风险提示: {warn_str}"
            )

        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "【候选股票列表】\n\n"
                    + "\n\n".join(candidates_text)
                    + "\n\n---\n"
                    "请对以上股票进行风险审查，输出最终推荐结果。\n"
                    "JSON 格式：\n"
                    "```json\n"
                    "{\n"
                    '  "approved": [\n'
                    "    {\n"
                    '      "rank": 1,\n'
                    '      "code": "代码",\n'
                    '      "name": "名称",\n'
                    '      "sector": "板块",\n'
                    '      "risk_level": "低风险/中风险/高风险",\n'
                    '      "risk_flags": ["风险信号列表"],\n'
                    '      "position_advice": "仓位建议，如标准仓位10-15%",\n'
                    '      "stop_loss": "止损建议",\n'
                    '      "entry_point": "建议买入时机描述",\n'
                    '      "core_logic": "核心投资逻辑（100字以内）",\n'
                    '      "holding_period": "建议持有周期"\n'
                    "    }\n"
                    "  ],\n"
                    '  "excluded": [\n'
                    '    {"code": "代码", "name": "名称", "reason": "排除原因"}\n'
                    "  ],\n"
                    '  "portfolio_advice": "整体组合建议（100字以内）",\n'
                    '  "market_timing": "当前建仓时机判断"\n'
                    "}\n```"
                ),
            },
        ]

    def filter(
        self,
        arbitration_result: Dict,
        stock_packages: Dict,
        verbose: bool = True,
    ) -> Dict:
        """
        执行风控过滤。
        返回最终审核通过的股票列表。
        """
        if verbose:
            print(f"\n{'='*60}")
            print("  R: 风控审查启动")
            print(f"{'='*60}")

        candidates = arbitration_result.get("final_picks", [])

        # 硬性排除（基于名称关键词）
        hard_excluded = []
        to_review = []
        for c in candidates:
            name = c.get("name", "")
            code = c.get("code", "")
            excluded = False
            for kw in self.HARD_EXCLUSION_KEYWORDS:
                if kw in name or kw in code:
                    hard_excluded.append({"code": code, "name": name, "reason": f"命中硬性排除: {kw}"})
                    excluded = True
                    break
            if not excluded:
                to_review.append(c)

        if verbose and hard_excluded:
            print(f"  [硬性排除] {len(hard_excluded)} 只: {[e['code'] for e in hard_excluded]}")

        # LLM 风控审查
        msgs = self._build_risk_prompt(to_review, stock_packages)
        resp = self.llm.call_primary(msgs, temperature=0.2)

        if resp:
            parsed = LLMClient.parse_json(resp)
            if parsed:
                approved = parsed.get("approved", [])
                soft_excluded = parsed.get("excluded", [])
                portfolio_advice = parsed.get("portfolio_advice", "")
                market_timing = parsed.get("market_timing", "")

                if verbose:
                    print(f"\n  [风控] 通过: {len(approved)} 只 | 软性排除: {len(soft_excluded)} 只")
                    for a in approved:
                        print(
                            f"    {a.get('rank','')}.{a.get('code','')} {a.get('name','')} "
                            f"[{a.get('risk_level','')}] {a.get('position_advice','')}"
                        )

                return {
                    "approved": approved,
                    "hard_excluded": hard_excluded,
                    "soft_excluded": soft_excluded,
                    "portfolio_advice": portfolio_advice,
                    "market_timing": market_timing,
                    "total_approved": len(approved),
                }

        # 降级处理：直接返回 to_review 并加默认风控标注
        if verbose:
            print("  [警告] LLM风控审查失败，使用默认结果")

        approved = []
        for i, c in enumerate(to_review[:10], 1):
            approved.append({
                "rank": i,
                "code": c.get("code", ""),
                "name": c.get("name", ""),
                "sector": c.get("sector", ""),
                "risk_level": "未知",
                "risk_flags": [],
                "position_advice": "谨慎仓位 5-10%",
                "stop_loss": "跌破MA5或-7%",
                "entry_point": "结合实盘判断",
                "core_logic": "; ".join((c.get("all_reasonings") or [])[:2]),
                "holding_period": "5-10日",
            })

        return {
            "approved": approved,
            "hard_excluded": hard_excluded,
            "soft_excluded": [],
            "portfolio_advice": "请参考各股票风险提示自行判断仓位",
            "market_timing": "请结合当日盘面判断",
            "total_approved": len(approved),
        }


# ===================================================================== #
#  ModelTask: 模型纵向并行任务                                           #
# ===================================================================== #

class ModelTask:
    """
    单个 LLM 模型作为独立子任务，内部顺序执行 E1→E2→E3→E4→E5→E6，
    完成 intra-model 辩论后选出 20 支股票。

    被 main.py 中的 ThreadPoolExecutor 并发调用，每个 provider 独立运行。
    """

    SYSTEM_PROMPT = (
        "你是A股量化投研系统的首席分析师。\n"
        "你将依次扮演7位专家角色，对候选股票池进行全面的多维度分析：\n"
        "  E1 动量换手型专家：识别短期强势动量股\n"
        "  E2 成长估值型专家：寻找兼具成长性和安全边际的股票\n"
        "  E3 多因子平衡型专家：综合质量×动量×流动性评分\n"
        "  E4 技术形态型专家：多维技术指标共振分析\n"
        "  E5 资金流向型专家：追踪主力资金行为\n"
        "  E6 事件催化型专家：识别催化剂和异动信号\n"
        "  E7 板块优股专家：识别近一个月内相对板块走势更强的个股\n\n"
        "【重要工作方式】\n"
        "候选清单只提供股票代码、名称、板块、PE/PB/市值、近20日涨幅等基础坐标。\n"
        "你应当主动运用联网搜索（若支持）和云端知识获取每只股票的深度信息，\n"
        "包括K线走势、基本面数据、最新公告、机构评级、主力资金动向等，\n"
        "而不是依赖本地提交的原始数据。坐标信息配合你的知识即可完成专业分析。\n\n"
        "每次分析时你只扮演当前指定的专家，严格遵守该专家的分析框架。\n"
        "最终你将综合7位专家的意见，作为首席分析师给出最终选股决策。"
    )

    INTRA_MODEL_DEBATE_PROMPT = """\
现在你作为首席分析师，综合以上7位专家的分析意见，进行辩论与取舍，\
从候选股票池中精选出最具投资价值的20支股票，按信心度从高到低排名。

请返回如下 JSON 格式（不要包含其他文字）：
```json
{
  "picks": [
    {
      "rank": 1,
      "code": "股票代码(6位)",
      "name": "股票名称",
      "sector": "所属板块",
      "score": 85,
      "reasoning": "选择理由（100字以内）",
      "recommended_by_experts": ["E1", "E3"]
    }
  ]
}
```
picks 数量：选出10-20支，按信心度从高到低排名。如候选池不足10支，则全部列出。"""

    def __init__(
        self,
        provider_name: str,
        llm_client: LLMClient,
        config: Config,
        logger: WorkLogger,
        stock_profiles: Dict,
        stock_packages: Dict,
    ):
        self.provider_name = provider_name
        self.llm = llm_client
        self.config = config
        self.logger = logger
        self.stock_profiles = stock_profiles
        self.stock_packages = stock_packages

    def run(self) -> Dict:
        """
        执行模型纵向并行任务。

        流程:
          1. 创建共享 ConversationSession（首席分析师角色）
          2. 顺序调用 E1→E2→E3→E4→E5→E6，每个专家调用 run_in_session()
          3. Intra-model 辩论：session.say(辩论提示) → 精选20支
          4. 解析 JSON，返回结构化结果

        返回:
          {
            "model"      : str,
            "picks"      : [{"rank", "code", "name", "sector", "score", ...}],
            "expert_logs": {E1: {...}, ...},
            "debate_log" : str,
          }
        """
        self.logger.log(
            "session_start",
            model=self.provider_name,
            detail={"message": f"开始分析，共{len(self.stock_profiles)}只候选股票"},
        )

        session = ConversationSession(
            client=self.llm,
            provider_name=self.provider_name,
            system_prompt=self.SYSTEM_PROMPT,
        )

        experts: List[BaseAgent] = [
            MomentumExpert(self.llm, self.config),
            GrowthExpert(self.llm, self.config),
            MultiFactorExpert(self.llm, self.config),
            TechnicalExpert(self.llm, self.config),
            CapitalFlowExpert(self.llm, self.config),
            CatalystExpert(self.llm, self.config),
            SectorOutperformer(self.llm, self.config),
        ]

        expert_logs: Dict[str, Dict] = {}

        for i, expert in enumerate(experts):
            self.logger.log(
                "expert_start",
                model=self.provider_name,
                detail={"expert_id": expert.EXPERT_ID, "expert_name": expert.EXPERT_NAME},
            )
            try:
                er = expert.run_in_session(
                    session=session,
                    stock_profiles=self.stock_profiles,
                    stock_packages=self.stock_packages,
                    include_profiles=(i == 0),  # Only E1 embeds all profiles
                )
                expert_logs[expert.EXPERT_ID] = {
                    "picks_count": len(er.picks),
                    "picks": [
                        {"code": p.code, "name": p.name, "score": p.score}
                        for p in er.picks
                    ],
                }
                self.logger.log(
                    "expert_done",
                    model=self.provider_name,
                    detail={"expert_id": expert.EXPERT_ID, "picks_count": len(er.picks)},
                )
            except Exception as exc:
                self.logger.log(
                    "error",
                    model=self.provider_name,
                    detail={"expert_id": expert.EXPERT_ID, "error": str(exc)},
                )
                expert_logs[expert.EXPERT_ID] = {
                    "picks_count": 0,
                    "picks": [],
                    "error": str(exc),
                }

        # ── Intra-model 辩论 ──────────────────────────────────────────
        self.logger.log("debate_start", model=self.provider_name)
        debate_resp = session.say(self.INTRA_MODEL_DEBATE_PROMPT)

        picks = self._parse_debate_picks(debate_resp)

        # 重试一次：解析失败或 picks 不足预期（考虑候选池大小）
        min_expected = min(5, len(self.stock_profiles))
        if len(picks) < min_expected:
            self.logger.log(
                "debate_retry",
                model=self.provider_name,
                detail={"reason": f"首次解析仅得到{len(picks)}支（预期≥{min_expected}），重试"},
            )
            retry_resp = session.say(
                "请重新输出JSON。只需返回 {\"picks\": [...]} 格式，"
                "picks每项含code/name/sector/score/rank/reasoning字段。选10-20支。"
            )
            retry_picks = self._parse_debate_picks(retry_resp)
            if len(retry_picks) > len(picks):
                picks = retry_picks
                debate_resp = retry_resp

        # 降级兜底：从专家结果聚合
        if len(picks) < min_expected:
            self.logger.log(
                "debate_fallback",
                model=self.provider_name,
                detail={"reason": f"辩论解析仍仅{len(picks)}支，降级聚合专家推荐"},
            )
            fallback = self._fallback_from_experts(expert_logs)
            if len(fallback) > len(picks):
                picks = fallback

        self.logger.log(
            "debate_done",
            model=self.provider_name,
            detail={"picks_count": len(picks)},
        )
        self.logger.log(
            "model_done",
            model=self.provider_name,
            detail={"picks_count": len(picks), "message": f"选出{len(picks)}支股票"},
        )

        return {
            "model": self.provider_name,
            "picks": picks,
            "expert_logs": expert_logs,
            "debate_log": debate_resp or "",
        }

    @staticmethod
    def _parse_debate_picks(resp: Optional[str]) -> List[Dict]:
        """解析辩论结果，带类型验证和安全转换"""
        if not resp:
            return []
        parsed = LLMClient.parse_json(resp)
        if not parsed or "picks" not in parsed:
            return []
        raw_picks = parsed["picks"]
        if not isinstance(raw_picks, list):
            return []
        picks: List[Dict] = []
        for pick in raw_picks:
            if not isinstance(pick, dict):
                continue
            code = str(pick.get("code", "")).strip()
            if not code:
                continue
            try:
                rank = int(pick.get("rank", 99))
            except (ValueError, TypeError):
                rank = 99
            try:
                score = float(pick.get("score", 70))
            except (ValueError, TypeError):
                score = 70.0
            picks.append({
                "rank": rank,
                "code": code,
                "name": str(pick.get("name", "")),
                "sector": str(pick.get("sector", "")),
                "score": score,
                "reasoning": str(pick.get("reasoning", ""))[:300],
                "recommended_by_experts": pick.get("recommended_by_experts", []),
            })
        return sorted(picks, key=lambda x: x["rank"])[:20]

    @staticmethod
    def _fallback_from_experts(expert_logs: Dict[str, Dict]) -> List[Dict]:
        """辩论失败时从专家结果聚合，按出现次数和平均分排序"""
        pool: Dict[str, Dict] = {}
        for expert_id, log in expert_logs.items():
            for p in log.get("picks", []):
                code = str(p.get("code", "")).strip()
                if not code:
                    continue
                if code not in pool:
                    pool[code] = {
                        "code": code,
                        "name": p.get("name", ""),
                        "sector": "",
                        "count": 0,
                        "total_score": 0.0,
                        "experts": [],
                    }
                pool[code]["count"] += 1
                pool[code]["total_score"] += float(p.get("score", 70))
                pool[code]["experts"].append(expert_id)
                if p.get("name") and not pool[code]["name"]:
                    pool[code]["name"] = p["name"]
        sorted_pool = sorted(
            pool.values(),
            key=lambda x: (x["count"], x["total_score"] / max(x["count"], 1)),
            reverse=True,
        )
        return [
            {
                "rank": i + 1,
                "code": s["code"],
                "name": s["name"],
                "sector": s["sector"],
                "score": round(s["total_score"] / max(s["count"], 1), 1),
                "reasoning": f"由{s['count']}位专家({','.join(s['experts'])})共同推荐",
                "recommended_by_experts": s["experts"],
            }
            for i, s in enumerate(sorted_pool[:20])
        ]
