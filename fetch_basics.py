"""
手动抓取全量A股 PE/流通股本/市值 数据
保存到 output/stock_basics.json

用法: python fetch_basics.py
"""
import json
import re
import time
import requests
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent / "output" / "stock_basics.json"
BATCH_SIZE = 50  # 腾讯API每批50个代码


def get_tdx_codes():
    """从TDX获取全量A股代码"""
    try:
        from mootdx.reader import Reader
        reader = Reader.factory(market='std', tdxdir='d:/tdx')
        codes = []
        # 沪市
        for code in range(600000, 602000):
            codes.append(f"{code:06d}")
        for code in range(688000, 688400):
            codes.append(f"{code:06d}")
        # 深市主板
        for code in range(1, 1000):
            codes.append(f"{code:06d}")
        for code in range(1000, 2000):
            codes.append(f"{code:06d}")
        # 创业板
        for code in range(300001, 302000):
            codes.append(f"{code:06d}")
        # 北交所
        for code in range(430001, 430200):
            codes.append(f"bj{code}")
        return codes
    except Exception as e:
        print(f"[TDX] 获取代码失败: {e}")
        return []


def add_prefix(codes):
    """给代码加上 sh/sz/bj 前缀"""
    result = []
    for c in codes:
        if c.startswith('bj'):
            result.append(c)  # 北交所已经是 bj 开头
        elif c.startswith('4') or c.startswith('8'):
            result.append(f'bj{c}')
        elif int(c) >= 600000 or (len(c) == 6 and c.startswith('5')):
            result.append(f'sh{c}')
        else:
            result.append(f'sz{c}')
    return result


def fetch_tencent_batch(codes_with_prefix):
    """批量从腾讯API获取实时行情"""
    codes_str = ','.join(codes_with_prefix)
    url = f'https://qt.gtimg.cn/q={codes_str}'
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[腾讯API] 请求失败: {e}")
        return {}

    results = {}
    for line in r.text.strip().split('\n'):
        m = re.match(r'v_(\w+)=\"(.+)\"', line)
        if not m:
            continue
        raw_code = m.group(1)
        fields = m.group(2).split('~')
        if len(fields) < 45:
            continue

        # 解析字段
        # [1] 名称, [3] 现价, [36] 成交量(万股), [37] 流通股本(万股)
        # [39] PE, [44] 总市值(亿元)
        try:
            pe_str = fields[39].strip()
            pe = float(pe_str) if pe_str and pe_str not in ('', '-', 'None', '0') else None

            float_str = fields[37].strip()
            float_shares = int(float_str) if float_str and float_str not in ('', '-', 'None') else None

            mktcap_str = fields[44].strip()
            mktcap = float(mktcap_str) if mktcap_str and mktcap_str not in ('', '-', 'None') else None

            name = fields[1].strip()

            # 去掉前缀得到纯代码
            code = raw_code[2:] if raw_code.startswith(('sh', 'sz')) else raw_code

            results[code] = {
                "name": name,
                "pe": pe,
                "float_shares": float_shares,
                "mktcap": mktcap
            }
        except (ValueError, IndexError):
            continue

    return results


def fetch_all(basic_codes, verbose=True):
    """抓取全量A股数据"""
    total = len(basic_codes)
    all_data = {}
    batch_time = 0

    for i in range(0, total, BATCH_SIZE):
        batch = basic_codes[i:i + BATCH_SIZE]
        prefixed = add_prefix(batch)

        t0 = time.time()
        data = fetch_tencent_batch(prefixed)
        elapsed = time.time() - t0
        batch_time += elapsed

        all_data.update(data)

        done = min(i + BATCH_SIZE, total)
        rate = (i + BATCH_SIZE) / batch_time if batch_time > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0

        if verbose:
            print(f"\r  [{done:5d}/{total}] ({rate:.0f}只/秒, 剩余约{eta:.0f}秒)", end="", flush=True)

        # 腾讯API有频率限制，间隔一下
        if i + BATCH_SIZE < total:
            time.sleep(0.3)

    if verbose:
        print()
    return all_data


def main():
    print("=" * 60)
    print("全量A股 PE/市值 数据抓取工具")
    print("=" * 60)

    # 1. 获取股票代码列表
    print("\n[1/3] 获取A股代码列表（来自TDX）...")
    codes = get_tdx_codes()
    print(f"  共 {len(codes)} 只股票")

    if not codes:
        print("[错误] 无法获取股票列表，退出")
        return

    # 2. 抓取数据
    print(f"\n[2/3] 从腾讯API抓取实时行情（{len(codes)} 只）...")
    t0 = time.time()
    data = fetch_all(codes)
    elapsed = time.time() - t0
    print(f"  抓取完成，有效数据 {len(data)} 只，耗时 {elapsed:.1f} 秒")

    # 3. 保存
    print(f"\n[3/3] 保存到 {OUTPUT_FILE}...")
    from datetime import date
    result = {
        "date": str(date.today()),
        "data": data
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"  已保存 {len(data)} 条数据，日期 {result['date']}")

    # 统计
    valid_pe = sum(1 for v in data.values() if v['pe'] is not None)
    valid_float = sum(1 for v in data.values() if v['float_shares'] is not None)
    valid_mktcap = sum(1 for v in data.values() if v['mktcap'] is not None)
    print(f"\n  数据完整度: PE={valid_pe}/{len(data)}  流通股本={valid_float}/{len(data)}  市值={valid_mktcap}/{len(data)}")


if __name__ == "__main__":
    main()
