"""
单独运行 L1 量化过滤，显示结果（含PE/市值）。
用法: python run_l1.py
"""
import json
import time
import requests
import re
import pandas as pd
from pathlib import Path
from datetime import date

# ========== 路径 ==========
BASE_DIR = Path(__file__).parent
BASICS_FILE = BASE_DIR / "output" / "stock_basics.json"


# ========== 1. 加载股票清单 + 本地基础数据 ==========
def load_stock_list_data():
    """从 stock_list.json 加载股票名称数据"""
    f = BASE_DIR / "output" / "stock_list.json"
    if not f.exists():
        return {}
    data = json.loads(f.read_text(encoding="utf-8"))
    return data.get("data", {})

def load_basics():
    if not BASICS_FILE.exists():
        return {}, ""
    data = json.loads(BASICS_FILE.read_text(encoding="utf-8"))
    return data.get("data", {}), data.get("date", "")


# ========== 2. 腾讯API补充（缺失的PE/市值） ==========
def fetch_tencent_missing(codes_needed):
    """对缺失PE/市值的股票，批量从腾讯API补充"""
    results = {}
    if not codes_needed:
        return results

    BATCH = 50
    for i in range(0, len(codes_needed), BATCH):
        batch = codes_needed[i:i + BATCH]
        # 加前缀
        prefixed = []
        for c in batch:
            if c.startswith("6") or c.startswith("5"):
                prefixed.append(f"sh{c}")
            else:
                prefixed.append(f"sz{c}")

        try:
            url = f"https://qt.gtimg.cn/q={','.join(prefixed)}"
            r = requests.get(url, timeout=10)
            r.encoding = "gbk"
            for line in r.text.strip().split("\n"):
                if "=" not in line:
                    continue
                try:
                    _, val_part = line.split("=", 1)
                    val_part = val_part.strip().strip('"')
                    if not val_part:
                        continue
                    fields = val_part.split("~")
                    if len(fields) < 45:
                        continue
                    code = fields[2]
                    results[code] = {
                        "name": fields[1],
                        "pe": float(fields[39]) if fields[39] and fields[39] not in ("", "-", "None") else None,
                        "float_shares": int(float(fields[37])) if fields[37] and fields[37] not in ("", "-", "None") else None,
                        "mktcap": float(fields[44]) if fields[44] and fields[44] not in ("", "-", "None") else None,
                    }
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            print(f"\n  [腾讯] 批次失败: {e}")
        time.sleep(0.3)

    return results


# ========== 3. 加载 DataEngine 并运行 L1 ==========
def run_l1():
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from data_engine import DataEngine

    print("=" * 60)
    print("L1 量化预筛（仅TDX）")
    print("=" * 60)

    # 加载本地基础数据
    print("\n[1/4] 加载本地 PE/市值数据...")
    basics, basics_date = load_basics()
    if basics_date:
        print(f"  本地数据日期: {basics_date}，共 {len(basics)} 只")
    else:
        print("  本地数据为空")

    # 加载股票清单（用于名称和ST过滤）
    print("\n[1b/4] 加载股票清单...")
    stock_list_data = load_stock_list_data()
    print(f"  股票清单: {len(stock_list_data)} 只")

    # 初始化 DataEngine（只加载TDX）
    print("\n[2/4] 初始化 DataEngine（TDX）...")
    de = DataEngine(tdx_dir="d:/tdx")
    if de._tdx is None:
        print("  [错误] TDX 未连接，退出")
        return

    # 运行 L1
    print("\n[3/4] 运行 L1 量化过滤...")
    t0 = time.time()
    candidates = de.L1_quant_filter(verbose=True)
    elapsed = time.time() - t0
    print(f"  L1 耗时: {elapsed:.1f} 秒")

    if not candidates:
        print("  L1 无候选股，退出")
        return

    # ========== 4. 补充 PE/市值 ==========
    print(f"\n[4/4] 补充 PE/市值...")

    # 用本地数据填充
    for item in candidates:
        code = item["code"]
        # 优先用股票清单的名称，其次用basics的名称
        if code in stock_list_data:
            item["name"] = stock_list_data[code].get("name", code)
        elif code in basics:
            item["name"] = basics[code].get("name", code)
        else:
            item["name"] = code
        if code in basics:
            b = basics[code]
            item["pe"] = b.get("pe")
            item["float_shares"] = b.get("float_shares")  # 万股
            item["mktcap"] = b.get("mktcap")  # 亿元
        else:
            item["pe"] = None
            item["float_shares"] = None
            item["mktcap"] = None

    # 找出缺失的，上腾讯API补充
    missing_codes = [item["code"] for item in candidates if item["pe"] is None]
    if missing_codes:
        print(f"  本地缺失 {len(missing_codes)} 只，上腾讯API查询...")
        fetched = fetch_tencent_missing(missing_codes)
        for item in candidates:
            if item["pe"] is None and item["code"] in fetched:
                f = fetched[item["code"]]
                item["name"] = f.get("name", item["code"])
                item["pe"] = f.get("pe")
                item["float_shares"] = f.get("float_shares")
                item["mktcap"] = f.get("mktcap")
    else:
        print("  全部命中本地数据")

    # ========== 5. 过滤ST股 ==========
    st_eliminated = 0
    candidates_filtered = []
    for item in candidates:
        name = item.get("name", "")
        # ST股包括：ST、*ST、SST、S*ST等
        if "ST" in name.upper():
            st_eliminated += 1
            continue
        candidates_filtered.append(item)
    candidates = candidates_filtered
    print(f"  ST股淘汰: {st_eliminated} 只")

    # ========== 6. 展示结果 ==========
    print(f"\n{'='*80}")
    print(f"L1 筛选结果：{len(candidates)} 只")
    print(f"{'='*80}")

    # 表头
    print(f"{'代码':<8} {'名称':<10} {'得分':>6} {'RPS20':>6} {'near_h':>7} {'量比':>6} "
          f"{'MA20斜率':>8} {'ATR比':>6} {'放量阳':>6} {'PE':>7} {'总市值亿':>11}")
    print("-" * 110)

    for item in candidates:
        pe_str = f"{item.get('pe', 0) or 0:.1f}" if item.get('pe') is not None else "N/A"
        mktcap_str = f"{item['mktcap']:.0f}" if item.get('mktcap') is not None else "N/A"
        atr_ratio = f"{item.get('atr5', 0) / item.get('atr20', 1):.2f}" if item.get('atr20', 0) > 0 else "N/A"
        yang = "Y" if item.get("is_yang_fang") else "-"

        print(f"{item['code']:<8} {item.get('name', '?'):<10} "
              f"{item.get('score', 0):>6.1f} {item.get('rps20', 0):>6.1f} "
              f"{item.get('near_high', 0):>7.3f} {item.get('amount_ratio', 0):>6.2f} "
              f"{item.get('ma20_slope', 0):>8.4f} {atr_ratio:>6} {yang:>6} "
              f"{pe_str:>7} {mktcap_str:>11}")

    print("-" * 110)
    regime = candidates[0].get("market_regime", "?") if candidates else "?"
    regime_cn = {"bull": "🐂牛市", "bear": "🐻熊市", "neutral": "⚖️震荡偏多"}.get(regime, regime)
    print(f"共 {len(candidates)} 只（Top 5%，{regime_cn}模式，按得分降序排列）")
    print()
    if regime in ("bull", "neutral"):
        print("牛市评分: RPS20×30 + near_high>0.85=20 + 放量收阳=15 + MA20斜率=15 + ATR收敛=10 + 量比>1.5=10")
    else:
        print("熊市评分: MA20斜率=25 + ATR收敛=25 + RPS20×15 + near_high>0.75=15 + 缩量企稳=15 + 站上MA20=5")

    # ========== 6b. 注入300根日线（用于L2缠论分析）==========
    print("\n[6b] 加载300根日线数据（用于L2缠论分析）...")
    l1_300day_file = BASE_DIR / "output" / "l1_300day.json"
    l1_300day = {}
    if l1_300day_file.exists():
        try:
            raw = json.loads(l1_300day_file.read_text(encoding="utf-8"))
            for code, info in raw.get("data", {}).items():
                if info.get("data") and info["bars"] > 0:
                    df300 = pd.DataFrame(info["data"])
                    l1_300day[code] = df300
            print(f"  已加载 {len(l1_300day)} 只的300根日线")
        except Exception as e:
            print(f"  加载300日线失败: {e}")
    else:
        print(f"  文件不存在: {l1_300day_file}，L2将使用35根日线")

    # 将300根日线注入到各候选股的 df 字段中
    injected = 0
    for item in candidates:
        code = item["code"]
        if code in l1_300day:
            item["df"] = l1_300day[code]
            injected += 1
    print(f"  已将 {injected}/{len(candidates)} 只的df替换为300根版本")

    # ========== 7. 保存L1结果到本地文件 ==========
    l1_file = BASE_DIR / "output" / "l1_candidates.json"

    def _to_json_safe(v):
        if isinstance(v, (bool, int, float, str, type(None))):
            return v
        try:
            return bool(v)
        except (ValueError, TypeError):
            pass
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
        try:
            return float(v)
        except (ValueError, TypeError):
            pass
        return str(v)

    l1_save = []
    for item in candidates:
        d = {_k: _to_json_safe(_v) for _k, _v in item.items() if _k != "df"}
        l1_save.append(d)
    l1_file.write_text(json.dumps({"date": str(date.today()), "count": len(l1_save), "data": l1_save}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[L1] 结果已写入: {l1_file}（{len(l1_save)} 只）")

    # ========== 8. 运行 L2 缠论+波浪分析 ==========
    print("\n" + "=" * 80)
    print("L2 缠论 + 波浪形态分析")
    print("=" * 80)
    l2_candidates = de.L2_chan_wave_filter(candidates, verbose=True)

    if l2_candidates:
        # 补充名称
        for item in l2_candidates:
            code = item["code"]
            if code in stock_list_data:
                item["name"] = stock_list_data[code].get("name", code)
            elif code in basics:
                item["name"] = basics[code].get("name", code)
            else:
                item["name"] = code

        # ── 排序：两者都有 > 1p > 量价齐升 > 1类买点 ──
        def _l2_sort_key(it):
            has_both = bool(it.get('chan_buy') and it.get('wave_bull'))
            has_1p   = '1p' in (it.get('chan_buy_detail') or '')
            has_vol  = '量价齐升' in (it.get('wave_bull_detail') or '')
            has_1    = bool(it.get('chan_buy') and not has_1p)
            if has_both: return 0
            if has_1p:   return 1
            if has_vol:  return 2
            if has_1:    return 3
            return 4
        l2_candidates.sort(key=_l2_sort_key)

        print(f"\n{'='*140}")
        print(f"L2 筛选结果：{len(l2_candidates)} 只（排序：两者都有 > 1p强力 > 量价齐升 > 1类买点）")
        print(f"{'='*140}")
        print(f"{'代码':<8} {'名称':<10} {'缠论信号':<25} {'波浪信号':<30} {'L1':<6} {'现价':<8} {'PE':<8} {'总市值亿'}")
        print("-" * 140)
        for item in l2_candidates:
            cond = ""
            if item.get("cond_a"): cond += "A"
            if item.get("cond_b"): cond += "B"
            if item.get("cond_c"): cond += "C"
            chan_str = item.get("chan_buy_detail", "") or ("有" if item.get("chan_buy") else "—")
            wave_str = item.get("wave_bull_detail", "") or "—"
            pe = item.get("pe")
            pe_str = f"{pe:.1f}" if pe else "N/A"
            close = item.get("close", 0)
            mktcap = item.get("mktcap")
            mc_str = f"{mktcap:.0f}" if mktcap else "N/A"
            print(f"{item['code']:<8} {item.get('name', item['code']):<10} "
                  f"{chan_str:<25} {wave_str:<30} {cond:<6} {close:<8.2f} {pe_str:<8} {mc_str}")
        print("-" * 140)
        print(f"共 {len(l2_candidates)} 只")
    else:
        print("\n[L2] 无候选股通过形态分析")
        l2_candidates = []

    # 保存L2结果（排序后）
    if l2_candidates:
        l2_file = BASE_DIR / "output" / "l2_candidates.json"
        l2_save = []
        for item in l2_candidates:
            d = {_k: _to_json_safe(_v) for _k, _v in item.items() if _k != "df"}
            l2_save.append(d)
        l2_file.write_text(json.dumps({"date": str(date.today()), "count": len(l2_save), "data": l2_save}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[L2] 结果已写入: {l2_file}（{len(l2_save)} 只）")

        # 生成PDF报告
        print("\n[L2] 生成PDF报告...")
        try:
            import report_l2
            report_l2.main()
        except Exception as e:
            print(f"PDF生成失败: {e}")


if __name__ == "__main__":
    import sys, io
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr.encoding.lower() != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    run_l1()
