"""
为 L1 候选股票补充 300 根日线数据（TDX 本地）。
用法: python fetch_l1_300day.py
"""
import json
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
L1_FILE = BASE_DIR / "output" / "l1_candidates.json"
OUT_FILE = BASE_DIR / "output" / "l1_300day.json"


def main():
    # 加载 L1 结果
    l1 = json.loads(L1_FILE.read_text(encoding="utf-8"))
    items = l1["data"]
    codes = [it["code"] for it in items]
    print(f"L1 候选股票: {len(codes)} 只")

    # 初始化 TDX
    from data_engine import DataEngine
    de = DataEngine(tdx_dir="d:/tdx")
    if de._tdx is None:
        print("[错误] TDX 未连接，退出")
        return

    # 逐只获取 300 根日线
    results = {}
    t0 = time.time()
    for i, code in enumerate(codes):
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(codes) - i - 1) / rate if rate > 0 else 0
            print(f"  进度 {i+1}/{len(codes)} ({rate:.0f}只/秒, 剩余约{eta:.0f}秒)", flush=True)

        df = de._tdx_daily_only(code, days=300)
        if df is None or len(df) < 50:
            results[code] = {"error": "数据不足", "bars": 0}
            continue

        # 转换为可序列化格式
        records = []
        for idx, row in df.iterrows():
            rec = {}
            # 日期在索引中
            date_val = idx
            if hasattr(date_val, "strftime"):
                rec["date"] = date_val.strftime("%Y-%m-%d")
            else:
                rec["date"] = str(date_val)[:10]
            for k, v in row.items():
                if hasattr(v, "item"):  # numpy type
                    v = v.item()
                if isinstance(v, (int, float)):
                    rec[k] = float(v) if isinstance(v, (float,)) else int(v)
                elif hasattr(v, "strftime"):  # datetime
                    rec[k] = v.strftime("%Y-%m-%d")
                else:
                    rec[k] = str(v) if v is not None else None
            records.append(rec)

        results[code] = {
            "bars": len(records),
            "data": records
        }

    elapsed = time.time() - t0
    ok_count = sum(1 for v in results.values() if v.get("bars", 0) > 0)
    print(f"\n完成：{ok_count}/{len(codes)} 只成功，耗时 {elapsed:.1f}秒")

    # 保存
    OUT_FILE.write_text(
        json.dumps({"date": datetime.now().strftime("%Y-%m-%d"), "count": ok_count, "data": results}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"已写入: {OUT_FILE}")


if __name__ == "__main__":
    import sys, io
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr.encoding.lower() != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    main()
