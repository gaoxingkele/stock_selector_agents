# 开源项目选股逻辑借鉴参考

> 分析对象：`daily_stock_analysis` 和 `TradingAgents-CN`
> 分析日期：2026-03-16
> 目的：提炼可借鉴的选股逻辑、架构模式和代码，用于改进 stock_selector_agents

---

## 一、daily_stock_analysis（个股AI分析系统）

### 项目概况
- 定位：AI驱动的A股/港股/美股智能分析系统，支持定时自动分析+多渠道推送
- 技术栈：FastAPI + SQLAlchemy + LiteLLM + 多数据源
- 规模：146个Python文件，功能完备

### 1. 多因子信号评分系统 ⭐⭐⭐⭐⭐

**文件**：`src/stock_analyzer.py`（850行）

当前我们的系统让LLM自主打分，缺乏硬性量化约束。该项目实现了一套**100分制6因子加权评分**，可作为LLM分析前的预筛选层：

| 因子 | 权重 | 评分规则 |
|------|------|---------|
| 趋势（Trend） | 30分 | STRONG_BULL=30, BULL=26, WEAK_BULL=18, CONSOLIDATION=12 |
| 乖离率（Bias） | 20分 | 价格<MA5下方3%=20, 接近MA5(0-2%)=18, 高于MA5但<阈值=14, 过度偏离=4 |
| 量能（Volume） | 15分 | 缩量回调=15, 放量上涨=12, 正常=10, 缩量上涨=6, 放量下跌=0 |
| 支撑（Support） | 10分 | MA5支撑=5, MA10支撑=5 |
| MACD | 15分 | 零轴上金叉=15, 金叉=12, 上穿中=10, 多头=8 |
| RSI | 10分 | 超卖=10, 强势=8, 中性=5, 弱势=3, 超买=0 |

**买入信号映射**：
- ≥75分 + 多头排列 → STRONG_BUY
- ≥60分 + 牛势趋势 → BUY
- ≥45分 → HOLD
- ≥30分 → WAIT
- 空头趋势 → SELL/STRONG_SELL

**借鉴方式**：在 `data_engine.py` 中新增 `compute_signal_score()` 方法，在LLM分析前先算出量化分数，注入到专家提示词中，让LLM的判断有硬数据支撑，而非纯主观分析。

### 2. 乖离率防追高逻辑 ⭐⭐⭐⭐

**文件**：`src/stock_analyzer.py`（617-663行）

```python
# 核心规则：价格偏离MA5超过阈值则降低评分
bias_pct = (close - ma5) / ma5 * 100
base_threshold = 5.0  # 默认5%

# 强趋势补偿：趋势很强时放宽到7.5%
if trend == STRONG_BULL and trend_strength >= 70:
    effective_threshold = base_threshold * 1.5

# 超过阈值 → 追高风险警告
if bias_pct > effective_threshold:
    risk_factors.append(f"乖离率{bias_pct:.1f}%偏高，追高风险")
```

**借鉴方式**：当前风控只检查60日涨幅>50%的红灯规则。可在风控中加入乖离率检查（MA5/MA10/MA20多级），对短期涨幅过大的个股自动降级或加风险标签。

### 3. 数据源策略模式（多源自动降级） ⭐⭐⭐⭐⭐

**文件**：`data_provider/base.py`

```python
class DataFetcherManager:
    """按优先级尝试多个数据源，失败自动切换下一个"""
    fetchers = [
        AkShareFetcher(priority=1),
        TuShareFetcher(priority=2),
        EFinanceFetcher(priority=3),
        PyTDXFetcher(priority=4),
        BaoStockFetcher(priority=5),
    ]

    def fetch_daily(self, code, start, end):
        for fetcher in sorted(self.fetchers, key=lambda f: f.priority):
            try:
                df = fetcher.fetch_daily(code, start, end)
                if df is not None and len(df) > 0:
                    return self._normalize(df)  # 统一列名
            except Exception:
                continue
        return None

    def _normalize(self, df):
        """统一列名为标准格式"""
        # → ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
```

**借鉴方式**：当前 `data_engine.py` 的 akshare 调用经常因连接断开失败（运行日志中大量 `RemoteDisconnected`）。应引入此模式，akshare 失败后自动降级到 tushare/efinance/baostock，大幅提升数据采集成功率。

### 4. 回测验证引擎 ⭐⭐⭐⭐

**文件**：`src/core/backtest_engine.py`

```python
class BacktestEngine:
    eval_window_days = 10    # 推荐后10个交易日评估
    neutral_band_pct = 2.0   # ±2%内算中性（不扣分）

    def evaluate(self, prediction, actual_returns):
        """
        评估维度：
        - 方向准确率：预测涨/跌是否正确
        - 止损触发率：是否触及止损位
        - 目标达成率：是否达到目标价
        - 策略胜率：按策略分类统计
        - 风险收益比：目标距离 / 止损距离
        """
```

**借鉴方式**：当前系统没有回测能力，无法评估推荐质量。可新增一个 `backtest.py` 模块，每次运行时回测上一次推荐的表现，输出胜率报告，并将结果注入LLM提示词（"上次你推荐的XX涨了/跌了，请反思"）。

### 5. 11种YAML策略模板 ⭐⭐⭐

**目录**：`strategies/`

| 策略文件 | 策略名称 | 核心逻辑 |
|---------|---------|---------|
| `bull_trend.yaml` | 多头趋势 | MA5>MA10>MA20 + 量价配合 |
| `ma_golden_cross.yaml` | 均线金叉 | MA(5,10,20)金叉确认 |
| `shrink_pullback.yaml` | 缩量回调 | 上涨趋势中缩量回踩MA支撑 |
| `box_oscillation.yaml` | 箱体震荡 | 箱体下沿买入上沿卖出 |
| `chan_theory.yaml` | 缠论 | 缠中说禅技术分析体系 |
| `wave_theory.yaml` | 波浪理论 | 艾略特波浪5-3结构 |
| `bottom_volume.yaml` | 底部放量 | 低位放量突破 |
| `volume_breakout.yaml` | 放量突破 | 关键位放量突破确认 |
| `dragon_head.yaml` | 龙头战法 | 板块龙头首板/二板 |
| `emotion_cycle.yaml` | 情绪周期 | 市场情绪阶段交替 |
| `one_yang_three_yin.yaml` | 一阳吞三阴 | K线形态识别 |

**借鉴方式**：当前专家（E1-E7）的策略硬编码在提示词中。可将策略抽象为YAML配置，方便新增/调整策略而无需改代码。同时可参考"龙头战法"和"情绪周期"策略的逻辑，增强现有的MR市场雷达和E6事件催化专家。

### 6. LiteLLM统一接口 + 多Key负载均衡 ⭐⭐⭐

**文件**：`src/analyzer.py`

```python
# LiteLLM Router：统一接口，多key轮询，自动降级
from litellm import Router

model_list = [
    {"model_name": "main", "litellm_params": {"model": "gemini/gemini-3-pro", "api_key": key1}},
    {"model_name": "main", "litellm_params": {"model": "gemini/gemini-3-pro", "api_key": key2}},
    {"model_name": "fallback", "litellm_params": {"model": "openai/gpt-4o", "api_key": key3}},
]
router = Router(model_list=model_list, fallbacks=[{"main": ["fallback"]}])
```

**借鉴方式**：如果后续不用Cloubic统一网关，可考虑用LiteLLM替代当前的手动OpenAI客户端管理，自动处理多key轮询和降级。

### 7. 筹码分布分析 ⭐⭐⭐

**文件**：`data_provider/efinance_fetcher.py` + `data_provider/realtime_types.py`

```python
@dataclass
class ChipDistribution:
    profit_ratio: float      # 获利比例
    avg_cost: float          # 平均成本
    concentration_90: float  # 90%筹码集中度
    concentration_70: float  # 70%筹码集中度
    chip_peak_price: float   # 筹码峰值价格
```

**借鉴方式**：当前系统未使用筹码分布数据。可通过 efinance 库获取筹码集中度，作为额外的选股因子（筹码集中 = 主力控盘 = 更容易拉升）。

---

## 二、TradingAgents-CN（多智能体交易系统）

### 项目概况
- 定位：基于LangGraph的多智能体协作交易决策系统
- 架构：4阶段流水线（数据采集→投资辩论→风险评估→交易执行）
- 特色：结构化辩论 + 记忆学习 + 多市场支持

### 1. 多阶段辩论架构 ⭐⭐⭐⭐⭐

**核心流程**：

```
阶段1：并行数据采集
  ├─ 市场分析师（技术面）
  ├─ 社交媒体分析师（情绪面）
  ├─ 新闻分析师（事件面）
  └─ 基本面分析师（财务面）

阶段2：投资辩论（多空对抗）
  ├─ 看多研究员（构建买入论据）
  ├─ 看空研究员（构建卖出论据）
  └─ 裁判/研究经理（综合判决→投资计划）

阶段3：风险辩论（三方博弈）
  ├─ 激进派（强调高收益机会）
  ├─ 中立派（平衡视角）
  ├─ 保守派（强调风险控制）
  └─ 风险经理（最终风险评估）

阶段4：交易执行
  ├─ 交易员（最终决策）
  └─ 信号处理（结构化输出）
```

**与当前系统对比**：
- 当前：E1-E7专家独立分析 → 辩论（看到彼此结论后修改） → Borda融合
- TradingAgents-CN：数据采集 → 多空对抗辩论 → 风险三方辩论 → 综合决策

**借鉴方式**：当前的辩论是"各模型看到彼此结论后修改"，缺乏结构化对抗。可借鉴**多空对抗模式**：
1. 让部分模型专门扮演"看多方"，部分扮演"看空方"
2. 由"裁判模型"根据双方论据质量（而非简单投票）做出判决
3. 再经过风险辩论层（激进/中立/保守三方），得出最终推荐

### 2. 看多/看空研究员提示词设计 ⭐⭐⭐⭐

**文件**：`tradingagents/agents/researchers/bull_researcher.py`

```python
# 看多研究员 prompt（精华摘要）
prompt = """你是一位看涨分析师，负责为股票{company_name}构建强有力的买入论证。

你的任务：
- 增长潜力：市场机会、收入预测、可扩展性
- 竞争优势：独特产品、强势品牌、市场地位
- 积极指标：财务健康、行业趋势、积极消息
- 反驳看跌：用具体数据批判性分析对方论点

要求：
1. 必须基于事实和数据，不能空泛
2. 必须直接回应看空方的论点并反驳
3. 给出具体的上涨目标和时间框架"""
```

**文件**：`tradingagents/agents/researchers/bear_researcher.py`

```python
# 看空研究员 prompt
prompt = """你是一位看跌分析师，负责为股票{company_name}构建风险论证。

你的任务：
- 识别弱点和脆弱性
- 质疑乐观假设
- 呈现下行情景
- 挑战看多方的论点"""
```

**借鉴方式**：当前辩论轮次中，各模型只是"审视并辩论"，角色定位不够鲜明。可以在辩论阶段明确分配"看多"和"看空"角色，让对抗更有深度。

### 3. 研究经理（裁判）综合决策 ⭐⭐⭐⭐⭐

**文件**：`tradingagents/agents/managers/research_manager.py`

```python
prompt = """你是投资决策裁判。

规则：
1. 不要默认给出"持有"——当双方都有道理时，必须做出倾向性判断
2. 基于论据质量而非数量做决策
3. 生成详细投资计划：
   - 保守情景目标价（下行）
   - 基准情景目标价（最可能）
   - 乐观情景目标价（上行）
   - 时间框架：1个月、3个月、6个月
4. 参考历史记忆中类似情况的决策结果，避免重蹈覆辙"""
```

**关键设计**：**强制做出倾向性判断，不允许和稀泥**。

**借鉴方式**：当前Borda融合是纯机械的票数/排名加权，缺乏"裁判综合判断"环节。可在Borda融合后增加一个**"首席策略师"Agent**，输入融合结果和各模型的详细分析，输出最终的综合判断和调仓建议，特别是处理模型之间严重分歧的情况。

### 4. 三层风险辩论 ⭐⭐⭐⭐

**文件**：`tradingagents/agents/risk_mgmt/`

```
激进派（aggresive_debator.py）:
  → 强调上行潜力，质疑保守派的过度谨慎
  → 论点：变革性机遇、高回报场景

保守派（conservative_debator.py）:
  → 强调资本保全，提出最坏情景
  → 论点：隐藏风险、过度乐观预期、低风险替代方案

中立派（neutral_debator.py）:
  → 平衡视角，寻找折中方案

风险经理（risk_manager.py）:
  → 综合三方意见，输出最终风控参数
```

**借鉴方式**：当前风控是单模型一次性审查。可改为**三视角风控**：
1. 激进视角：哪些股票虽然有风险但值得冒险（避免错杀好股）
2. 保守视角：哪些股票风险被低估（避免踩雷）
3. 风险经理综合：最终仓位建议和止损设定

### 5. ChromaDB记忆学习系统 ⭐⭐⭐⭐

**文件**：`tradingagents/agents/utils/memory.py`

```python
class FinancialSituationMemory:
    """基于向量相似度的交易记忆"""

    def store_memory(self, situation, recommendation, outcome, lessons):
        """存储：当时的市场情况 + 推荐 + 实际结果 + 经验教训"""
        self.collection.add(
            documents=[situation],
            metadatas=[{"recommendation": recommendation,
                       "outcome": outcome,
                       "lessons": lessons}],
        )

    def get_memories(self, curr_situation, n_matches=2):
        """语义检索：找到与当前情况最相似的历史决策"""
        results = self.collection.query(
            query_texts=[curr_situation],
            n_results=n_matches,
        )
        return results  # 返回相似历史案例供Agent参考
```

**文件**：`tradingagents/graph/reflection.py`

```python
class Reflector:
    def reflect(self, state, actual_returns):
        """
        复盘：
        1. 当时推荐了什么？实际涨跌如何？
        2. 哪些因素判断正确？哪些判断失误？
        3. 下次遇到类似情况应该怎么改进？
        4. 将经验教训存入记忆库
        """
```

**借鉴方式**：当前系统没有历史学习能力。可新增：
1. 每次运行后存储 `{市场环境, 推荐列表, N日后实际涨跌}` 到本地向量库
2. 下次运行时检索相似的历史市场环境，将历史经验注入提示词
3. 实现"越用越准"的学习闭环

### 6. 结构化信号输出 ⭐⭐⭐

**文件**：`tradingagents/graph/signal_processing.py`

```python
class SignalProcessor:
    def process_signal(self, full_signal, stock_symbol=None):
        """从LLM自由文本中提取标准化交易信号"""
        return {
            'action': '买入/持有/卖出',
            'target_price': float,       # 目标价
            'confidence': 0.0-1.0,       # 置信度
            'risk_score': 0.0-1.0,       # 风险分
            'reasoning': str,            # 推理过程
            'scenarios': {
                'conservative': {...},   # 保守情景
                'baseline': {...},       # 基准情景
                'optimistic': {...},     # 乐观情景
            }
        }
```

**借鉴方式**：当前输出只有picks数组（code/name/score/stars/reasoning）。可增加**多情景目标价**（保守/基准/乐观）和**置信度评分**，让最终报告更具参考价值。

### 7. 死循环防护 ⭐⭐⭐

**文件**：`tradingagents/agents/analysts/market_analyst.py`

```python
# 防止Agent陷入无限工具调用循环
tool_call_count = state.get("market_tool_call_count", 0)
MAX_TOOL_CALLS = 3

if tool_call_count >= MAX_TOOL_CALLS:
    return "END_ANALYSIS"  # 强制终止

state["market_tool_call_count"] = tool_call_count + 1
```

**借鉴方式**：当前kimi辩论阶段超时55分钟的问题，部分原因是缺乏强制终止机制。应在ConversationSession中加入调用计数和时间上限。

### 8. 状态管理架构 ⭐⭐⭐

**文件**：`tradingagents/agents/utils/agent_states.py`

```python
class AgentState(TypedDict):
    company_of_interest: str
    trade_date: str

    # 四路并行分析报告
    market_report: str         # 技术面
    sentiment_report: str      # 情绪面
    news_report: str           # 新闻面
    fundamentals_report: str   # 基本面

    # 辩论状态
    investment_debate_state: InvestDebateState
    risk_debate_state: RiskDebateState

    # 最终输出
    investment_plan: str
    trader_investment_plan: str
    final_signal: dict

class InvestDebateState(TypedDict):
    bull_history: str      # 看多方全部论点
    bear_history: str      # 看空方全部论点
    judge_decision: str    # 裁判判决
    count: int             # 辩论轮次计数器
```

**借鉴方式**：当前系统的状态散落在 `main.py` 的局部变量中。可定义正式的 `AnalysisState` dataclass，在各步骤间传递，便于调试和持久化。

---

## 三、综合借鉴优先级排序

按**投入产出比**排序，推荐的改进方向：

### 第一优先级（高价值 + 低改动量）

| # | 改进项 | 来源 | 改动文件 | 预期效果 |
|---|--------|------|---------|---------|
| 1 | **量化信号评分预筛** | daily_stock | `data_engine.py` | LLM分析前先算100分制硬分数，减少主观偏差 |
| 2 | **数据源自动降级** | daily_stock | `data_engine.py` | 解决akshare频繁断连问题，数据采集成功率↑ |
| 3 | **乖离率防追高** | daily_stock | `stock_agents.py`(风控) | 补充短期过热检测，减少追高推荐 |
| 4 | **调用超时强制终止** | TradingAgents | `llm_client.py` | 防止单模型超时拖垮整体（kimi 55分钟问题） |

### 第二优先级（高价值 + 中等改动量）

| # | 改进项 | 来源 | 改动范围 | 预期效果 |
|---|--------|------|---------|---------|
| 5 | **多空对抗辩论** | TradingAgents | `stock_agents.py` | 辩论质量↑，从"各自修改"变为"结构化对抗" |
| 6 | **裁判综合决策** | TradingAgents | `main.py` + 新Agent | Borda融合后增加"首席策略师"综合判断 |
| 7 | **回测验证引擎** | daily_stock | 新文件 `backtest.py` | 评估推荐质量，形成反馈闭环 |
| 8 | **多情景目标价** | TradingAgents | `stock_agents.py` | 输出保守/基准/乐观三档目标价 |

### 第三优先级（高价值 + 大改动量）

| # | 改进项 | 来源 | 改动范围 | 预期效果 |
|---|--------|------|---------|---------|
| 9 | **记忆学习系统** | TradingAgents | 新模块 | 历史经验注入，越用越准 |
| 10 | **三视角风控辩论** | TradingAgents | `stock_agents.py` | 风控从单模型变为三方辩论 |
| 11 | **筹码分布因子** | daily_stock | `data_engine.py` | 增加筹码集中度选股维度 |
| 12 | **YAML策略模板化** | daily_stock | 架构重构 | 策略可配置，新增策略不改代码 |

---

## 四、关键代码片段速查

### 信号评分（可直接移植）

```python
# 来源：daily_stock_analysis/src/stock_analyzer.py
def compute_signal_score(df):
    """
    输入：DataFrame (含 close, volume, ma5, ma10, ma20, dif, dea, macd_bar, rsi6, rsi12, rsi24)
    输出：0-100 分数 + 买入信号等级
    """
    score = 0
    latest = df.iloc[-1]

    # 1. 趋势 (30分)
    if latest['ma5'] > latest['ma10'] > latest['ma20']:
        score += 30  # STRONG_BULL
    elif latest['ma5'] > latest['ma10']:
        score += 26  # BULL
    elif latest['ma5'] > latest['ma20']:
        score += 18  # WEAK_BULL
    else:
        score += 12  # CONSOLIDATION or worse

    # 2. 乖离率 (20分)
    bias = (latest['close'] - latest['ma5']) / latest['ma5'] * 100
    if bias < -3:
        score += 20
    elif -1 <= bias <= 2:
        score += 18
    elif 2 < bias <= 5:
        score += 14
    else:
        score += 4

    # 3. 量能 (15分) — 需要5日均量
    vol_ratio = latest['volume'] / df['volume'].tail(5).mean()
    prev_close = df.iloc[-2]['close']
    if latest['close'] < prev_close and vol_ratio < 0.8:
        score += 15  # 缩量回调（最优）
    elif latest['close'] > prev_close and vol_ratio > 1.2:
        score += 12  # 放量上涨
    elif 0.8 <= vol_ratio <= 1.2:
        score += 10  # 正常量能
    elif latest['close'] > prev_close and vol_ratio < 0.8:
        score += 6   # 缩量上涨（量价背离）
    else:
        score += 0   # 放量下跌

    # 4. 支撑 (10分)
    if latest['close'] >= latest['ma5'] >= latest['close'] * 0.98:
        score += 5
    if latest['close'] >= latest['ma10'] >= latest['close'] * 0.97:
        score += 5

    # 5. MACD (15分)
    if latest['dif'] > 0 and latest['dea'] > 0 and latest['macd_bar'] > 0:
        score += 15  # 零轴上金叉
    elif latest['macd_bar'] > 0 and df.iloc[-2]['macd_bar'] <= 0:
        score += 12  # 金叉
    elif latest['macd_bar'] > 0:
        score += 8   # 多头
    else:
        score += 3

    # 6. RSI (10分)
    rsi = latest['rsi6']
    if rsi < 30:
        score += 10  # 超卖
    elif 60 <= rsi < 70:
        score += 8   # 强势
    elif 40 <= rsi < 60:
        score += 5   # 中性
    elif 30 <= rsi < 40:
        score += 3   # 弱势
    else:
        score += 0   # 超买

    return score
```

### 多空辩论 prompt 模板（可直接采用）

```python
# 来源：TradingAgents-CN 简化版
BULL_PROMPT = """你是看多研究员。基于以下分析数据，为{stock_name}构建强有力的买入论证：
{analysis_data}

要求：
1. 列出3-5个核心买入理由（每个有数据支撑）
2. 预判看空方可能的攻击点并提前反驳
3. 给出保守/基准/乐观三档目标价
4. 以JSON返回"""

BEAR_PROMPT = """你是看空研究员。基于以下分析数据，为{stock_name}构建风险论证：
{analysis_data}

要求：
1. 列出3-5个核心风险点（每个有数据支撑）
2. 质疑看多方可能的乐观假设
3. 给出下行风险目标和止损位
4. 以JSON返回"""

JUDGE_PROMPT = """你是投资决策裁判。以下是多空双方的论证：
看多方：{bull_case}
看空方：{bear_case}

规则：
1. 不允许给出模糊的"持有"——必须倾向买入或回避
2. 基于论据质量而非数量做判断
3. 给出最终评分(0-100)和仓位建议
4. 以JSON返回"""
```

---

## 五、两个项目的关键文件索引

### daily_stock_analysis

| 文件 | 内容 | 借鉴价值 |
|------|------|---------|
| `src/stock_analyzer.py` | 趋势分析 + 信号评分 + MACD/RSI | ⭐⭐⭐⭐⭐ |
| `data_provider/base.py` | 多数据源策略模式 | ⭐⭐⭐⭐⭐ |
| `src/core/pipeline.py` | 分析流水线编排 | ⭐⭐⭐⭐ |
| `src/core/backtest_engine.py` | 回测验证引擎 | ⭐⭐⭐⭐ |
| `src/analyzer.py` | LiteLLM多模型集成 | ⭐⭐⭐ |
| `src/market_analyzer.py` | 市场环境分析 | ⭐⭐⭐ |
| `src/search_service.py` | 新闻搜索集成 | ⭐⭐⭐ |
| `strategies/*.yaml` | 11种策略模板 | ⭐⭐⭐ |
| `data_provider/efinance_fetcher.py` | 筹码分布数据 | ⭐⭐⭐ |

### TradingAgents-CN

| 文件 | 内容 | 借鉴价值 |
|------|------|---------|
| `agents/researchers/bull_researcher.py` | 看多研究员 | ⭐⭐⭐⭐ |
| `agents/researchers/bear_researcher.py` | 看空研究员 | ⭐⭐⭐⭐ |
| `agents/managers/research_manager.py` | 裁判综合决策 | ⭐⭐⭐⭐⭐ |
| `agents/risk_mgmt/*.py` | 三视角风控辩论 | ⭐⭐⭐⭐ |
| `agents/utils/memory.py` | ChromaDB记忆系统 | ⭐⭐⭐⭐ |
| `graph/reflection.py` | 决策复盘反思 | ⭐⭐⭐⭐ |
| `graph/signal_processing.py` | 结构化信号输出 | ⭐⭐⭐ |
| `agents/utils/agent_states.py` | 状态管理 | ⭐⭐⭐ |
| `graph/trading_graph.py` | LangGraph工作流 | ⭐⭐⭐ |
| `tools/analysis/indicators.py` | 技术指标计算 | ⭐⭐⭐ |
