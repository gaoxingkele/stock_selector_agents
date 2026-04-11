# A股多智能体选股系统 — Claude Code 项目指南

## 项目概述
双管线 A股选股系统：链路A (LLM 辩论) + 链路B (量化技术分析)，最终 Borda 融合 → 风控 → PDF 报告。

## 技术栈
- **Python 3.14**, pandas, numpy, scipy, matplotlib, mplfinance
- **数据源**：TDX 本地（D:/tdx），akshare（必须关闭代理），adata
- **形态识别**：pattern_detector.py（迁移自 BennyThadikaran/stock-pattern）
- **缠论**：内置 chan/ 模块
- **LLM**：多 Provider 支持（OpenAI/Claude/Gemini/Doubao/Deepseek/GLM/Kimi/Qwen/Grok 等）
  - Cloubic 路由 + 直连双模式
  - 通过 PANEL_PRIORITY 排序，paneL_size 控制并发数

## 关键文件

| 文件 | 说明 |
|---|---|
| `main.py` | 链路A/B 主入口、命令行参数、所有 P1/P2/P3 逻辑函数 |
| `data_engine.py` | DataEngine：数据采集、L1_quant_filter、L2_score_and_rank、L3_generate_charts |
| `stock_agents.py` | LLM 智能体：MarketRadar、SectorPicker、ModelTask、ChartAnalyst、RiskController 等 |
| `llm_client.py` | LLMClient：多 Provider 调用、call_panel、run_debate |
| `config.py` | Config + ProviderConfig + Cloubic 路由 |
| `pattern_detector.py` | 形态识别模块（双底/头肩底/三角/旗型/VCP + 颈线检测） |
| `historical_backtest.py` | 历史回测（与 L1/L2 严格对齐 v6） |
| `verify_patterns.py` | 形态可视化验证工具 |
| `scheduler.py` | 自动化调度（pre_market/post_market/weekly_report） |
| `test_p1_integration.py` | P1 改进项集成测试 |

## 核心架构（v6）

### 链路A 完整流程
```
Step 0   MR 市场雷达 → compute_timing_limits（极度弱势直接观望）
Step 1   板块筛选 SP（cheap_models 可选）
Step 2   股票数据采集
Step 3   L3 看图判定
Step 4   N模型并行 E1-E6 + intra-model 辩论
Step 5   Borda 融合（top_n 受 timing_limits 控制）
Step 5.3 E1-E7 专家辩论（n_top 受 timing_limits）
Step 5.4 A/B 交叉验证（cross_validate_with_l1）
Step 5.45 LLM 幻觉校验（verify_llm_hallucination）
Step 6   个股风控
Step 6.2 交易维度补全（enrich_trade_levels）
Step 6.3 组合风控（portfolio_risk_check）
Step 6.4 多日一致性追踪（track_consecutive_recommendations）
Step 6.5 市场择时硬截断
Step 7   最终报告
```

### L1 v6 双轮并行架构
```
全市场预筛（PE/市值/流动性/连续大阴）
  ↓
  ┌── 并行 ──┐
量化强势轮      形态潜力轮
(RPS/MA/ATR    (双底/头肩底/三角/旗型/
 +筹码集中度    VCP+日线+周线)
 +龙虎榜)
  └── 合并 ──┘
       ↓
   双轮共识+10分
   综合排序 → Top 5%
```

## 常用命令

```bash
# 链路 A（默认）
python main.py

# 链路 B（量化）
python main.py --link b

# 双链路 + 增量缓存复用
python main.py --link both --incremental

# LLM 成本优化（早期阶段用便宜模型）
python main.py --models claude,openai --cheap-models deepseek,glm

# 历史回测
python historical_backtest.py --months 6

# 形态验证
python verify_patterns.py 002460 20260408
python verify_patterns.py --batch

# P1 集成测试
python test_p1_integration.py

# 自动化调度
python scheduler.py --task pre_market   # cron 模式
python scheduler.py --daemon            # 内置循环
```

## 关键约定

- **TDX 数据路径**：D:/tdx
- **akshare 必须关闭代理**：`os.environ['http_proxy']=''` `os.environ['https_proxy']=''`
- **windows shell 用 bash**：本项目用 git bash，不用 cmd
- **L1 数据窗口**：500 日（2 年+），支持周线形态检测
- **回测对齐**：historical_backtest.py 的 l1_filter_at_date / l2_score_and_rank 必须与 data_engine 同步

## 重要历史决策

- **L1 v3 → v4 → v5**：从单轮量化打分演进到双轮并行（量化轮+形态轮），解决"底部反转股进不了 Top"的问题
- **L2 v5 重构**：从独立打分改为继承 L1 的 quant_score+pattern_score，避免重复计算
- **形态参数 TOLERANCE=2.0**：A 股波动大，原 stock-pattern 美股标准（1.0）太严
- **HNSD/HNSU 颈线检查**：当前价偏离颈线 > 1.5×avg_bar 直接失效，防止已突破的形态被误判
- **DBOT/DTOP 改用相对 4% 容差**：原 avg_bar 在长期数据上太宽松（10% 误差也通过）
- **BEAR_VETO_THRESHOLD=10 + BEAR_DOMINANT_FLOOR=20**：双层阈值防止下跌中的小整理误判

## 6 个月回测对比（v3 vs v5，34 共同日期）
- T+3 胜率: v3 39.7% → v5 42.4% (+2.8pp)
- T+3 收益: v3 -1.26% → v5 -0.59% (+0.67%)
- 反弹期最佳：2025-12 49.1%/+1.05%，2026-02 53.1%/+1.50%
