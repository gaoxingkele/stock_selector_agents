"""
下载全量A股股票清单，保存到 output/stock_list.json
用法: python fetch_stock_list.py
"""
import json
import requests
import time
from pathlib import Path
from datetime import date

OUTPUT_FILE = Path(__file__).parent / "output" / "stock_list.json"


def fetch_sina_stock_list():
    """从新浪获取全量A股股票列表"""
    all_stocks = []
    for page in range(1, 50):
        url = (
            f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
            f"/Market_Center.getHQNodeDataSimple"
            f"?page={page}&num=1000&sort=symbol&asc=1&node=hs_a&_s_r_a=page"
        )
        try:
            r = requests.get(url, timeout=15)
            data = json.loads(r.text)
            if not data:
                break
            all_stocks.extend(data)
            print(f"\r  第{page}页: +{len(data)} = {len(all_stocks)} 只", end="", flush=True)
        except Exception as e:
            print(f"\n  page {page} error: {e}")
            break
        time.sleep(0.1)
    return all_stocks


def main():
    print("=" * 60)
    print("下载全量A股股票清单（来源：新浪）")
    print("=" * 60)

    print("\n[1/2] 从新浪API获取股票列表...")
    stocks = fetch_sina_stock_list()
    print(f"\n  共获取 {len(stocks)} 只股票")

    print("\n[2/2] 保存到文件...")
    # 整理数据：去掉前缀，转为纯代码
    result = {
        "date": str(date.today()),
        "source": "sina",
        "total": len(stocks),
        "data": {}
    }

    for s in stocks:
        raw = s["symbol"]  # e.g. 'sh600519', 'sz000001', 'bj920000'
        code = raw[2:]  # 去掉 sh/sz/bj 前缀
        market = raw[:2]  # sh, sz, bj

        # 转换PE格式
        pe_str = s.get("pricechange", "")
        pe = None
        if pe_str and pe_str not in ("", "-", "None"):
            try:
                pe = float(pe_str)
            except ValueError:
                pass

        result["data"][code] = {
            "name": s.get("name", ""),
            "market": market,
            "close": float(s["trade"]) if s.get("trade") not in ("", "-") else None,
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存到: {OUTPUT_FILE}")
    print(f"  日期: {result['date']}")

    # 统计
    sh = sum(1 for v in result["data"].values() if v["market"] == "sh")
    sz = sum(1 for v in result["data"].values() if v["market"] == "sz")
    bj = sum(1 for v in result["data"].values() if v["market"] == "bj")
    print(f"\n  沪市: {sh} | 深市: {sz} | 北交所: {bj}")


if __name__ == "__main__":
    main()
