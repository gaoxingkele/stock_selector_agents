# A股多智能体选股系统

**Multi-Model Parallel Stock Selector with Borda Fusion**

多个大语言模型（LLM）并行分析，6位风格互补的专家智能体协同选股，Borda Count跨模型融合，自动生成PDF投研报告。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        初始化 & 模型面板                             │
│  加载 .env + .env.cloubic → 按 PANEL_PRIORITY 选取 N 个模型         │
│  Cloubic 桥接（openai/claude/gemini/glm）+ 直连（其余）              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 0: 市场雷达 (MR)                                               │
│  全面板投票 → 大盘趋势 + 情绪温度 + 概念炒作预判 + ETF活跃板块        │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 1: 板块优选 (SP)                                               │
│  全面板多模型投票 → Top 3-5 板块                                      │
│  降级兜底: SP 结果不足时 S0 快速补充                                   │
├──────────────────────────────────────────────────────────────────────┤
│  Step 1.5: 跨市场主题 (GX)                                           │
│  美股/港股热点联动 → 多数同意则补充 +1 板块                            │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 2: 个股数据采集                                                │
│  月线/周线/日线 K线 + 财务 + 资金流 + 实时行情                        │
│  数据源三级降级: TDX本地 → AKShare → Tushare                         │
├──────────────────────────────────────────────────────────────────────┤
│  Step 3: 构建股票画像 (Slim Profile)                                 │
│  极简坐标文本（PE/PB/市值/涨幅/业绩预告/股东户数/商誉/质押等）         │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 4: N模型纵向并行 (ThreadPoolExecutor)                          │
│                                                                      │
│  ┌─ Model A ──────────┐  ┌─ Model B ──────────┐  ┌─ Model C ─────┐ │
│  │ ConversationSession │  │ ConversationSession │  │ ...           │ │
│  │ E1 动量换手 ──────→ │  │ E1 ──────→         │  │               │ │
│  │ E2 成长估值 ──────→ │  │ E2 ──────→         │  │               │ │
│  │ E3 多因子   ──────→ │  │ E3 ──────→         │  │               │ │
│  │ E4 技术形态 ──────→ │  │ E4 ──────→         │  │               │ │
│  │ E5 资金流向 ──────→ │  │ E5 ──────→         │  │               │ │
│  │ E6 事件催化 ──────→ │  │ E6 ──────→         │  │               │ │
│  │    ↓ Intra辩论      │  │    ↓ Intra辩论     │  │               │ │
│  │ → Top20 推荐        │  │ → Top20 推荐       │  │               │ │
│  └─────────────────────┘  └────────────────────┘  └───────────────┘ │
│  后台线程每30秒打印进度快照                                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 5: Borda Count 跨模型融合 → Top25                              │
│  第1名=20分, 第2名=19分, ..., 第20名=1分 → 汇总排序                   │
├──────────────────────────────────────────────────────────────────────┤
│  Step 5.5: 首席策略师（可选，--strategist 开启）                      │
│  单一LLM对Borda排名做定性调整（默认禁用，避免干扰客观投票）            │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 6: 风控审查 (R)                                                │
│  硬性排除（ST/退市/涨停锁死）+ 软性过滤 + 仓位建议（低/中/高风险）    │
│  三视角辩论（保守/中性/激进）→ 综合裁决                                │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│  Step 8: PDF 投顾报告                                                │
│  reportlab + matplotlib → 表格/K线图/评分卡                          │
├──────────────────────────────────────────────────────────────────────┤
│  Step 9: 事件驱动预判（独立推荐线）                                   │
│  新闻采集 → 全面板投票 → 因果链推理 → 受益标的                        │
├──────────────────────────────────────────────────────────────────────┤
│  Step 10: 异动爆发股扫描（独立推荐线）                                │
│  涨停/放量突破 → 全面板评估 → 题材共振评分                            │
├──────────────────────────────────────────────────────────────────────┤
│  Step 7: 最终文本报告 + 总览图片                                      │
│  融合推荐 + 事件驱动 + 异动推荐 + ETF匹配 → TXT + PNG                 │
├──────────────────────────────────────────────────────────────────────┤
│  记忆存储：本次决策写入 memory/，供下次运行检索历史经验                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 智能体说明

### 市场环境层

| 代号 | 名称 | 职责 |
|------|------|------|
| **MR** | 市场雷达 (MarketRadar) | 全面板投票，扫描大盘趋势+情绪温度+概念炒作预判+ETF活跃信号 |

### 板块选择层

| 代号 | 名称 | 职责 |
|------|------|------|
| **SP** | 板块优选专家 (SectorPicker) | 全面板多模型投票，选出 Top 3-5 板块 |
| **GX** | 跨市场主题专家 (GlobalThemeAdvisor) | 参考美股/港股热点，补充A股联动板块（最多+1个） |
| **S0** | 板块初筛 (SectorScreener) | SP降级备用：3模型快速投票 |

### 个股分析层（每模型内顺序执行，共享会话历史）

| 代号 | 名称 | 风格 | 持有期 | 核心分析维度 |
|------|------|------|--------|-------------|
| **E1** | 动量换手型专家 | 短线激进 | 1-10日 | 量价动量、换手率、涨停板效应、短期超跌反弹 |
| **E2** | 成长估值型专家 | 中线稳健 | 5-30日 | 业绩成长性、PEG估值弹性、ROE趋势、机构持仓 |
| **E3** | 多因子平衡型专家 | 均衡攻守 | 3-20日 | 质量因子 + 动量因子 + 流动性因子综合评分 |
| **E4** | 技术形态型专家 | 短中结合 | 1-15日 | K线形态、MACD/RSI/布林带共振，**支持K线图视觉分析** |
| **E5** | 资金流向型专家 | 中线跟庄 | 3-20日 | 主力净流入、大单买卖、融资余额变化、北向资金 |
| **E6** | 事件催化型专家 | 事件驱动 | 1-15日 | 政策催化、公告事件、行业景气度拐点 |

### 独立推荐线

| 代号 | 名称 | 职责 |
|------|------|------|
| **EA** | 事件驱动分析师 (EventAnalyst) | 新闻→因果链→受益标的推理，全面板投票 |
| **BA** | 异动爆发分析师 (BreakoutAnalyst) | 涨停/放量突破股扫描→题材共振评分 |

### 裁决与风控层

| 代号 | 名称 | 职责 |
|------|------|------|
| **V** | 投票仲裁者 (VotingArbitrator) | 加权投票聚合，计算共识度 |
| **R** | 风控审查 (RiskController) | ST/涨停/退市排除，三视角辩论，仓位建议 |

### 融合算法

**Borda Count**：每个模型对各股排名，第1名得20分，第2名得19分，……第20名得1分，第21名以后得0分。各模型得分汇总后排序，输出 Top 25，并附带：
- `recommended_by`：推荐该股的所有模型及其排名/分数
- `borda_score`：总 Borda 分
- `model_count`：推荐模型数量
- `all_reasonings`：各模型给出的核心理由（最多3条）

---

## LLM 模型面板

### 面板优先级

```
minimax → doubao → openai → claude → kimi → qwen → grok → gemini → glm → deepseek → perplexity
```

默认 `panel_size=4`，取前4个有 API Key 的 provider。

### 接入模式

| 路由方式 | Provider | 默认模型 | 说明 |
|----------|----------|---------|------|
| **Cloubic 桥接** | openai | gpt-5.4 | 国内直连，无需代理 |
| **Cloubic 桥接** | claude | claude-opus-4-6 | 国内直连，无需代理 |
| **Cloubic 桥接** | gemini | gemini-3.1-pro-preview | 国内直连，无需代理 |
| **Cloubic 桥接** | glm | glm-5 | 国内直连，无需代理 |
| **直连** | minimax | MiniMax-M2.7 | 国内直连 |
| **直连** | kimi | kimi-k2.5 | 国内直连（硬编码不走Cloubic） |
| **直连** | doubao | doubao-seed-2-0-pro-260215 | 国内直连 |
| **直连** | qwen | qwen3.5-plus | 国内直连 |
| **直连** | deepseek | deepseek-reasoner | 国内直连 |
| **直连(代理)** | grok | grok-4.20 | 需 LLM_PROXY |
| **直连(代理)** | perplexity | sonar-reasoning-pro | 需 LLM_PROXY |

### Cloubic 路由

[Cloubic](https://cloubic.com) 是 AI 多模型统一调度平台，一个 API Key 即可访问 100+ 大模型。

- `--cloubic`：强制启用 Cloubic 路由
- `--direct`：强制全部直连（禁用 Cloubic）
- 白名单配置：`.env.cloubic` 中 `CLOUBIC_ROUTED_PROVIDERS`
- kimi 始终直连（代码硬编码），不走 Cloubic

---

## 数据来源

| 数据类型 | 优先级1 | 优先级2 | 优先级3 |
|----------|---------|---------|---------|
| K线数据（月/周/日） | 通达信TDX(本地) | AKShare | Tushare |
| 板块成分股 | AKShare板块成分 | Tushare申万行业索引 | 关键字兜底 |
| 实时行情/资金流 | AKShare | — | — |
| 财务数据 | AKShare(同花顺) | Tushare | — |
| 业绩预告 | AKShare | — | — |
| 股东户数 | AKShare | — | — |
| 大宗交易 | AKShare | — | — |
| 融资融券 | AKShare | Tushare margin_detail | — |
| 商誉/质押率 | Tushare | — | — |
| 大盘指数/情绪面 | AKShare | — | — |
| ETF行情 | AKShare | — | — |
| 新闻/公告 | AKShare | — | — |
| 美股/港股热点 | AKShare | — | — |

> 数据采集（AKShare/Tushare）设置 `trust_env=False`，不走系统代理。

---

## Slim Profile 数据字段

每只候选股的画像字符串包含以下信息（供专家分析）：

```
{代码} {名称} | 板块:{板块} | PE={PE} PB={PB} 市值={市值} 现价={价格} 近20日={涨幅}
 | 业绩预告:{变动类型}{变动幅度}
 | 股东户数:{增加/减少}{变化幅度}%
 | 板块RPS={百分位}
 | 商誉率={商誉率}% 质押率={质押率}%
 | 大宗交易:{笔数}笔{折价率}%
 | 融资余额5日:{增/减}{变化幅度}%
```

---

## 记忆学习系统

系统内置决策记忆（`memory/` 目录），跨运行积累经验：

- **存储**：每次运行后保存推荐股票、市场环境、Borda评分
- **检索**：下次运行时匹配相似市场环境，注入历史决策参考到提示词
- **反思**：回顾历史推荐的后续涨跌表现，提炼经验教训

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 `.env`

```bash
cp .env.example .env   # 或直接编辑 .env
```

必填项：
```ini
TUSHARE_TOKEN=你的token
LLM_PROXY=http://127.0.0.1:7890     # LLM代理（可选，grok/perplexity需要）
GROK_API_KEY=xai-xxx                 # 至少一个LLM的API Key
```

Cloubic 模式（推荐）：
```ini
# .env.cloubic
CLOUBIC_ENABLED=true
CLOUBIC_API_KEY=sk-xxx               # 一个Key访问openai/claude/gemini/glm
CLOUBIC_ROUTED_PROVIDERS=openai,gemini,claude,glm
```

### 3. 运行

```bash
# 全自动流程（自动选板块，默认4个模型并行）
python main.py

# 指定板块
python main.py --sectors 人工智能,机器人,半导体

# 指定个股（跳过板块筛选）
python main.py --stocks 000988,002506,000545

# 指定使用的模型
python main.py --models minimax,doubao,openai,claude

# 指定板块 + 指定模型
python main.py --sectors 新能源 --models grok,kimi --panel 2

# 强制 Cloubic / 强制直连
python main.py --cloubic
python main.py --direct

# 启用首席策略师（对Borda结果做定性调整）
python main.py --strategist

# 演示模式（模拟数据，无需行情API）
python main.py --demo

# 查看当前配置
python main.py --config
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sectors` / `-s` | 自动选 | 指定分析板块，逗号分隔 |
| `--stocks` | — | 指定个股代码，逗号分隔（跳过板块筛选） |
| `--models` | 自动选 | 指定使用的模型，逗号分隔 |
| `--panel` / `-p` | `4` | 并行模型数量 |
| `--max-stocks` / `-m` | `25` | 每板块最多分析股票数 |
| `--no-parallel` | 关闭 | 关闭模型并行，改为顺序执行 |
| `--strategist` | 禁用 | 启用首席策略师（Step 5.5） |
| `--cloubic` | — | 强制启用 Cloubic 路由 |
| `--direct` | — | 强制全部直连 |
| `--demo` | — | 演示模式 |
| `--config` | — | 显示配置信息后退出 |
| `--quiet` / `-q` | — | 减少控制台输出 |

---

## 项目结构

```
stock_selector_agents/
├── main.py               # 主程序入口，完整流程编排（Step 0-10）
├── config.py             # .env 加载，Cloubic路由，LLM配置结构
├── llm_client.py         # 多LLM客户端（ConversationSession / call_panel / 代理策略）
├── data_engine.py        # 数据引擎（K线/财务/资金流/情绪面/概念信号/ETF/新闻）
├── stock_agents.py       # 所有智能体（MR/SP/GX/S0/E1-E6/EA/BA/V/R）
├── fusion.py             # Borda Count 跨模型融合（归一化 + 多级排序）
├── work_logger.py        # JSONL结构化日志 + 控制台输出
├── report_generator.py   # PDF投研报告 + 总览图片生成（reportlab + matplotlib）
├── memory.py             # 决策记忆系统（存储/检索/反思）
├── requirements.txt      # Python依赖
├── .env                  # API密钥配置（不提交到Git）
├── .env.cloubic          # Cloubic 专用配置（模型映射/白名单）
│
├── prompts/              # 各智能体提示词规格文档
│   ├── 00_总调度_Master_Selector.md
│   ├── 01_板块初筛_Sector_Screener.md
│   ├── 02-07_专家E1-E6提示词
│   ├── 08_投票仲裁_Voting_Arbitrator.md
│   └── 09_风控审查_Risk_Control.md
│
├── memory/               # 决策记忆数据（不提交到Git）
│
└── output/               # 运行输出（不提交到Git）
    ├── data/             # 中间JSON数据
    ├── reports/          # PDF + TXT + PNG 投研报告
    ├── charts/           # K线图表PNG（供E4视觉分析）
    └── logs/             # work_log + llm_calls JSONL
```

---

## 输出说明

每次运行在 `output/` 下生成：

| 文件 | 说明 |
|------|------|
| `data/market_radar_YYYYMMDD.json` | 市场雷达结果（大盘/情绪/概念预判） |
| `data/sector_screen_YYYYMMDD.json` | 板块优选结果 |
| `data/global_theme_YYYYMMDD.json` | 跨市场主题分析结果 |
| `data/stock_profiles_YYYYMMDD.json` | 个股画像数据 |
| `data/model_results_YYYYMMDD.json` | 各模型选股结果（含排名/理由） |
| `data/fusion_result_YYYYMMDD.json` | Borda融合结果（Top25） |
| `data/risk_result_YYYYMMDD.json` | 风控审查结果 |
| `data/breakout_result_YYYYMMDD.json` | 异动爆发股结果 |
| `reports/final_report_YYYYMMDD.txt` | 文本版最终报告 |
| `reports/investment_report_YYYYMMDD.pdf` | PDF投研报告 |
| `reports/overview_YYYYMMDD.png` | 总览分析图片 |
| `logs/work_log_YYYYMMDD_HHMM.log` | JSONL结构化运行日志 |
| `logs/llm_calls_YYYYMMDD.jsonl` | LLM调用详情（token用量/耗时） |
| `charts/{code}.png` | 各股K线图表 |

---

## 技术特性

- **ConversationSession 上下文累积**：E1 发送完整个股画像（~53k tokens），E2-E6 仅发指令，共享历史上下文，避免重复传输
- **模型纵向并行**：每个 LLM 独立运行完整 E1→E6 流程，互不干扰，最后 Borda 融合
- **Cloubic 透明路由**：按白名单决定每个 provider 走 Cloubic 还是直连，运行时动态切换
- **视觉分析路由**：`supports_vision=true` → 主模型直接处理图片；`vision_model` 非空 → 路由到专用视觉模型；否则降级为文字描述
- **三级数据降级**：TDX本地（最快）→ AKShare → Tushare，任一失败自动降级
- **三视角风控辩论**：保守/中性/激进三个视角独立审查，综合裁决
- **记忆学习系统**：跨运行积累决策经验，匹配相似市场环境注入参考
- **事件驱动独立线**：新闻因果链推理 + 异动爆发扫描，与主推荐线互补
- **全 try/except 保护**：新数据字段获取失败不影响主流程
- **Queue-based 异步日志**：写文件在独立后台线程，不阻塞分析流程
- **并行进度监控**：后台线程每30秒打印各模型进度快照

---

## 风险提示

> 本系统仅供投资研究参考，不构成任何投资建议。
> 股市有风险，投资需谨慎。所有选股结果均为AI模型基于历史数据的推断，不代表未来收益。
