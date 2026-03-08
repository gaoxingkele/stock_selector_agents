# A股多智能体选股系统

**Multi-Model Parallel Stock Selector with Borda Fusion**

多个大语言模型（LLM）并行分析，6位风格互补的专家智能体协同选股，Borda Count跨模型融合，自动生成PDF投研报告。

---

## 系统架构

```
数据采集层（TDX/AKShare/Tushare 三级降级）
    ↓
板块优选（SP全面板投票 → GX跨市场主题补充）
    ↓
个股数据采集（月线/周线/日线 + 财务 + 资金流 + K线图表）
    ↓
N模型并行（ThreadPoolExecutor）
  ┌─────────────────────────────────────────────────────┐
  │  每个模型内部（ConversationSession 共享上下文）        │
  │  E1动量 → E2成长 → E3多因子 → E4技术 → E5资金 → E6事件 │
  │              ↓ Intra-model 辩论                      │
  │         精选 Top20 股票                               │
  └─────────────────────────────────────────────────────┘
    ↓
Borda Count 跨模型融合（→ Top10）
    ↓
R风控审查（硬性排除 + 软性过滤 + 仓位建议）
    ↓
输出：文本报告 + PDF投研报告 + 结构化JSON
```

---

## 智能体说明

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

### 裁决与风控层

| 代号 | 名称 | 职责 |
|------|------|------|
| **V** | 投票仲裁者 (VotingArbitrator) | 加权投票聚合，计算共识度 |
| **R** | 风控审查 (RiskController) | ST/涨停/退市排除，仓位建议（低/中/高风险） |

### 融合算法

**Borda Count**：每个模型对各股排名，第1名得20分，第2名得19分，……第20名得1分，第21名以后得0分。各模型得分汇总后排序，输出 Top 10，并附带：
- `recommended_by`：推荐该股的所有模型及其排名/分数
- `borda_score`：总 Borda 分
- `model_count`：推荐模型数量
- `all_reasonings`：各模型给出的核心理由（最多3条）

---

## 数据来源

| 数据类型 | 优先级1 | 优先级2 | 优先级3 |
|----------|---------|---------|---------|
| K线数据（月/周/日） | 通达信TDX(本地) | AKShare | Tushare |
| 板块成分股 | AKShare板块成分 | Tushare申万行业索引 | 关键字兜底 |
| 实时行情/资金流 | AKShare | — | — |
| 财务数据 | AKShare(同花顺) | Tushare | — |
| 业绩预告 | AKShare (stock_performance_forecast_em) | — | — |
| 股东户数 | AKShare (stock_holder_num_em) | — | — |
| 大宗交易 | AKShare (stock_dzjy_detail_em) | — | — |
| 融资融券 | AKShare (市场接口) | Tushare margin_detail | — |
| 商誉/质押率 | Tushare balancesheet / pledge_stat | — | — |

> 所有 LLM 调用走代理（`.env` 中 `LLM_PROXY`），数据采集（AKShare/Tushare）不走代理。

---

## slim profile 数据字段

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

## 支持的 LLM 提供商

| 提供商 | 模型 | 视觉支持 |
|--------|------|---------|
| **Grok** (xAI) | grok-4-1-fast-reasoning | ✅ 原生视觉（主模型即支持） |
| **GLM** (智谱) | glm-4.7 | ✅ 视觉模型 glm-4v-plus |
| **DeepSeek** | deepseek-reasoner | ❌ 官方API暂不支持图片 |
| **Kimi** (月之暗面) | kimi-k2-turbo-preview | ✅ 视觉模型 moonshot-v1-8k-vision-preview |
| **Gemini** (Google) | gemini-3.1-pro-preview | — |
| **Perplexity** | sonar-pro | ✅ 原生视觉 |
| **Doubao** (火山引擎) | doubao-seed-2-0-pro-260215 | ✅ 同模型支持视觉 |
| **Qwen** (阿里云) | qwen3.5-plus | ✅ 视觉模型 qwen-vl-max-latest |
| **MiniMax** | MiniMax-M1 | ✅ 视觉模型 MiniMax-VL-01 |

默认面板优先级：`grok → glm → deepseek → kimi → gemini → doubao → qwen → minimax → perplexity`

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
LLM_PROXY=http://127.0.0.1:7890     # LLM代理（可选）
LLM_PROVIDER=grok                    # 主提供商
GROK_API_KEY=xai-xxx                 # 至少一个LLM的API Key
```

### 3. 运行

```bash
# 全自动流程（自动选板块，默认3个模型并行）
python main.py

# 指定板块
python main.py --sectors 人工智能,机器人,半导体

# 指定使用的模型
python main.py --models grok,glm,deepseek

# 指定板块 + 指定模型
python main.py --sectors 新能源 --models grok,kimi --panel 2

# 演示模式（模拟数据，无需行情API）
python main.py --demo

# 查看当前配置（含视觉模型信息）
python main.py --config
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sectors` / `-s` | 自动选 | 指定分析板块，逗号分隔 |
| `--models` | 自动选 | 指定使用的模型，逗号分隔（如 `grok,glm,deepseek`） |
| `--panel` / `-p` | `3` | 并行模型数量 |
| `--max-stocks` / `-m` | `25` | 每板块最多分析股票数 |
| `--no-parallel` | 关闭 | 关闭模型并行，改为顺序执行 |
| `--demo` | — | 演示模式 |
| `--config` | — | 显示配置信息后退出 |
| `--quiet` / `-q` | — | 减少控制台输出 |

---

## 项目结构

```
stock_selector_agents/
├── main.py               # 主程序入口，完整流程编排
├── config.py             # .env 加载，LLM配置结构
├── llm_client.py         # 多LLM客户端（ConversationSession / call_with_image）
├── data_engine.py        # 数据引擎（采集/清洗/指标/画像生成/K线图表）
├── stock_agents.py       # 所有智能体（SP/GX/S0/E1-E6/V/R）
├── fusion.py             # Borda Count 跨模型融合
├── work_logger.py        # JSONL结构化日志 + 控制台输出
├── report_generator.py   # PDF投研报告生成（reportlab + matplotlib）
├── data_fetcher.py       # 独立数据采集脚本（可单独运行）
├── requirements.txt      # Python依赖
├── .env                  # API密钥配置（不提交到Git）
│
├── prompts/              # 各智能体提示词规格文档
│   ├── 00_总调度_Master_Selector.md
│   ├── 01_板块初筛_Sector_Screener.md
│   ├── 02_专家一_动量换手型_Momentum_Expert.md
│   ├── 03_专家二_成长估值型_Growth_Value_Expert.md
│   ├── 04_专家三_多因子平衡型_Multi_Factor_Expert.md
│   ├── 05_专家四_技术形态型_Technical_Expert.md
│   ├── 06_专家五_资金流向型_Capital_Flow_Expert.md
│   ├── 07_专家六_事件催化型_Event_Catalyst_Expert.md
│   ├── 08_投票仲裁_Voting_Arbitrator.md
│   └── 09_风控审查_Risk_Control.md
│
└── output/               # 运行输出（不提交到Git）
    ├── data/             # 中间JSON数据
    ├── reports/          # PDF + TXT 投研报告
    ├── charts/           # K线图表PNG（供E4视觉分析）
    └── logs/             # work_log_*.log + llm_calls_*.jsonl
```

---

## 输出说明

每次运行在 `output/` 下生成：

| 文件 | 说明 |
|------|------|
| `data/sector_screen_YYYYMMDD.json` | 板块初筛结果 |
| `data/stock_packages_YYYYMMDD.json` | 个股数据包（K线+财务+资金流） |
| `data/model_results_YYYYMMDD.json` | 各模型选股结果（含排名/理由） |
| `data/fusion_result_YYYYMMDD.json` | Borda融合结果（Top10） |
| `reports/final_report_YYYYMMDD.txt` | 文本版最终报告 |
| `reports/investment_report_YYYYMMDD.pdf` | PDF投研报告（含K线图/表格） |
| `logs/work_log_YYYYMMDD_HHMM.log` | JSONL结构化运行日志 |
| `logs/llm_calls_YYYYMMDD.jsonl` | LLM调用详情（token用量/耗时） |
| `charts/{code}.png` | 各股K线图表 |

---

## 技术特性

- **ConversationSession 上下文累积**：E1 发送完整个股画像（~53k tokens），E2-E6 仅发指令，共享历史上下文，避免重复传输
- **模型纵向并行**：每个 LLM 独立运行完整 E1→E6 流程，互不干扰，最后 Borda 融合
- **视觉分析路由**：`supports_vision=true` → 主模型直接处理图片；`vision_model` 非空 → 路由到专用视觉模型；否则降级为文字描述
- **三级数据降级**：TDX本地（最快）→ AKShare → Tushare，任一失败自动降级
- **全 try/except 保护**：新数据字段获取失败不影响主流程
- **Queue-based 异步日志**：写文件在独立后台线程，不阻塞分析流程
- **并行进度监控**：后台线程每30秒打印各模型进度快照

---

## 风险提示

> ⚠️ **本系统仅供投资研究参考，不构成任何投资建议。**
> 股市有风险，投资需谨慎。所有选股结果均为AI模型基于历史数据的推断，不代表未来收益。
