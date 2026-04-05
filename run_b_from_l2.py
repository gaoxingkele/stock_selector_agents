"""
从 L2 结果继续运行 B 管线：L3 → Borda → 风控 → 报告
用法: python run_b_from_l2.py
"""
import json, os, sys, time
from datetime import date
from pathlib import Path

# UTF-8 输出
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), 'w', encoding='utf-8', buffering=1)
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = open(sys.stderr.fileno(), 'w', encoding='utf-8', buffering=1)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TODAY = str(date.today())
NOW = time.strftime("%H%M%S")
sys.path.insert(0, str(BASE_DIR))

# ─── 加载 L2 结果 ───────────────────────────────────────────────
print("=" * 70)
print("L3 续跑：从 L2 结果继续（B管线后半段）")
print("=" * 70)

l2_data = json.loads((OUTPUT_DIR / "l2_candidates.json").read_text(encoding="utf-8"))
l2_candidates = l2_data["data"]
print(f"\n[L2] 已加载 {len(l2_candidates)} 只候选股")

# 注入 300 根日线
import pandas as pd
l1_300day = {}
raw_300 = json.loads((OUTPUT_DIR / "l1_300day.json").read_text(encoding="utf-8"))
for code, info in raw_300.get("data", {}).items():
    if info.get("data") and info["bars"] > 0:
        l1_300day[code] = pd.DataFrame(info["data"])

for item in l2_candidates:
    code = item["code"]
    if code in l1_300day:
        item["df"] = l1_300day[code]
print(f"[300日线] 已注入 {sum(1 for i in l2_candidates if i.get('df') is not None)} 只")

# ─── L3.1: 生成 K 线图 ─────────────────────────────────────────
print(f"\n[L3.1] K线图生成（{len(l2_candidates)} 只）...")

from data_engine import DataEngine
engine = DataEngine(tdx_dir="d:/tdx")

chart_dir = OUTPUT_DIR / "charts" / TODAY
chart_dir.mkdir(parents=True, exist_ok=True)
t_chart = time.time()
chart_paths = engine.L3_generate_charts(l2_candidates, output_dir=str(chart_dir), verbose=True)
print(f"[L3.1] 图表生成完成，耗时 {time.time()-t_chart:.1f}秒")
ok_charts = sum(1 for v in chart_paths.values() if v)
print(f"[L3.1] 成功: {ok_charts}/{len(l2_candidates)} 只")

# ─── L3.2: 多模型看图判定 ─────────────────────────────────────
print(f"\n[L3.2] 多模型看图判定...")

from config import load_config
from llm_client import LLMClient

cfg = load_config()
llm = LLMClient(cfg)
ok_providers = [r["name"] for r in llm.preflight_check(None) if r["ok"]]
panel_size = min(4, len(ok_providers))
active_models = ok_providers[:panel_size]
print(f"[L3.2] 可用模型: {active_models}，使用 {panel_size} 个并行")

from stock_agents import ChartAnalyst

chart_analyst = ChartAnalyst(llm, cfg)
t_analyze = time.time()
l3_result = chart_analyst.analyze(
    candidates=l2_candidates,
    chart_paths=chart_paths,
    providers=active_models,
    panel_size=panel_size,
)
l3_picks = l3_result.get("picks", [])
print(f"\n[L3.2] 模型判定完成，耗时 {time.time()-t_analyze:.1f}秒")
print(f"[L3] 推荐 {len(l3_picks)} 只（TOP10）：")
for p in l3_picks[:10]:
    print(f"    {p['code']} {p.get('name',''):<8} 信号={p['signal']} 评分={p['score']} 票={p.get('model_count',1)}模型")

# 保存 L3 结果
chart_paths_str = {k: {kk: str(vv) for kk, vv in v.items()} for k, v in chart_paths.items()}
l3_save = {"L3_result": {**l3_result, "chart_paths": chart_paths_str}}
(OUTPUT_DIR / f"L3_result_{TODAY}.json").write_text(
    json.dumps(l3_save, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[L3] 结果已写入: output/L3_result_{TODAY}.json")

# ─── Borda 融合 ────────────────────────────────────────────────
print(f"\n[Borda] 融合...")

# 构建 model_results_for_borda（与 run_link_b_pipeline 一致）
model_results_for_borda = []
for res in l3_result.get("model_results", []):
    picks = res.get("picks", [])
    ranked_picks = []
    for i, p in enumerate(picks):
        ranked_picks.append({
            "rank": p.get("rank", i + 1),
            "code": p.get("code", ""),
            "name": p.get("name", ""),
            "sector": p.get("sector", "L3量化"),
            "score": float(p.get("score", p.get("signal_score", 70))),
        })
    model_results_for_borda.append({
        "model": res.get("model", "unknown"),
        "picks": ranked_picks,
    })

if not model_results_for_borda:
    for p in l3_picks:
        model_results_for_borda.append({
            "model": "l3_fallback",
            "picks": [{"rank": i+1, "code": p["code"], "name": p.get("name",""),
                       "sector": "L3量化", "score": float(p.get("score", 70))} for i, p in enumerate(l3_picks)]
        })

from fusion import borda_fusion
borda_top = borda_fusion(model_results_for_borda, top_n=30)
print(f"[Borda] 融合完成，共 {len(borda_top)} 只")
for i, p in enumerate(borda_top[:10]):
    print(f"    {i+1}. {p['code']} {p.get('name',''):<8} Borda={p['borda_score']:.1f} 票={p.get('model_count',0)}")

# ─── 转换为 arb_result 格式（供风控使用）─────────────────────
def _consensus_level(model_count):
    if model_count >= 4: return "强共识"
    if model_count >= 2: return "中共识"
    return "弱共识"

def fusion_to_arb_result(final_top):
    final_picks = []
    for pick in final_top:
        final_picks.append({
            "rank": pick["rank"],
            "code": pick["code"],
            "name": pick["name"],
            "sector": pick.get("sector", "L3量化"),
            "final_score": pick["borda_score"],
            "consensus_level": _consensus_level(pick["model_count"]),
            "expert_count": pick["model_count"],
            "warnings": [],
            "all_reasonings": pick.get("all_reasonings", []),
            "borda_score": pick["borda_score"],
            "avg_score": pick.get("avg_score"),
            "model_count": pick["model_count"],
            "recommended_by": pick.get("recommended_by", []),
        })
    return {
        "final_picks": final_picks,
        "candidate_pool": {},
        "total_candidates": len(final_top),
    }

arb_result = fusion_to_arb_result(borda_top)

# ─── E1-E7 专家辩论（对 Borda Top15 做最终逻辑验证）──────────
print(f"\n[E1-E7] 专家辩论（Borda Top15）...")
from main import run_expert_debate_after_borda

# 构建 Pipeline B 的 minimal stock_packages（供专家辩论用）
# 从 l2_candidates 提取已有数据
borda_codes = {p["code"] for p in borda_top[:15]}
sp_b = {}
for cand in l2_candidates:
    code = cand["code"]
    if code in borda_codes:
        sp_b[code] = {
            "realtime": {"名称": cand.get("name", code)},
            "daily": cand.get("df"),        # 300日线 DataFrame
            "weekly": cand.get("df"),       # 用日线近似（实际需要周线）
            "monthly": cand.get("df"),      # 用日线近似
        }

enriched = run_expert_debate_after_borda(
    borda_top=borda_top[:15],
    stock_packages=sp_b,
    llm=llm,
    cfg=cfg,
    active_models=ok_providers[:panel_size],
    n_top=15,
    verbose=True,
)
# 用辩论结果覆盖 arb_result 中的对应股票
if enriched:
    enriched_map = {p["code"]: p for p in enriched}
    for fp in arb_result["final_picks"]:
        code = fp["code"]
        if code in enriched_map:
            fp["expert_reasoning"] = enriched_map[code].get("expert_reasoning", "")
            fp["expert_logs"] = enriched_map[code].get("expert_logs", {})
            fp["extra_warnings"] = enriched_map[code].get("extra_warnings", [])

# ─── 风控过滤 ────────────────────────────────────────────────
print(f"\n[风控] 过滤...")
from stock_agents import RiskController

rc = RiskController(llm, cfg)
risk_result = rc.filter(
    arbitration_result=arb_result,
    stock_packages={},   # 链路B不采集完整数据包
    verbose=True,
)
risk_approved = risk_result.get("approved", [])
risk_soft = risk_result.get("soft_excluded", [])
print(f"\n[风控] 通过 {len(risk_approved)} 只，软过滤 {len(risk_soft)} 只")

# ─── 生成报告 ────────────────────────────────────────────────
print(f"\n[报告] 生成 PDF...")

# 简化 expert_summary（链路B无专家链）
expert_summary = {}

from report_generator import generate_report
report_path = generate_report(
    risk_result=risk_result,
    arb_result=arb_result,
    expert_summary=expert_summary,
    stock_packages={},
    top_sectors=[],
    n_models=panel_size,
    output_dir=str(OUTPUT_DIR / "reports"),
    output_filename=str(OUTPUT_DIR / "reports" / f"选股报告_{TODAY}_{NOW}.pdf"),
    l3_result=l3_result,
)
print(f"[报告] 已生成: {report_path}")

# 保存所有结果
(OUTPUT_DIR / f"risk_result_linkb_{TODAY}.json").write_text(
    json.dumps(risk_result, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\n{'='*70}")
print(f"B 管线后半段完成！")
print(f"  L2候选: {len(l2_candidates)} → L3推荐: {len(l3_picks)} → "
      f"Borda: {len(borda_top)} → 风控: {len(risk_approved)}")
print(f"{'='*70}")
