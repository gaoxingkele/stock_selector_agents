#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 — 数据采集模块
====================================
为板块初筛和6位选股专家提供所需的市场数据。
数据来源：akshare（免费开源A股数据接口）

使用方法：
    pip install akshare pandas --break-system-packages
    python data_fetcher.py

输出：在当前目录生成 JSON 文件，供AI智能体使用。
"""

import json
import os
import sys
from datetime import datetime, timedelta

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    print("请先安装依赖: pip install akshare pandas --break-system-packages")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TODAY = datetime.now().strftime("%Y%m%d")
DATE_60D_AGO = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")  # 预留buffer


def save_json(data, filename):
    """保存数据为JSON文件"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✓ 已保存: {filepath}")
    return filepath


# ============================================================
# 模块一：板块数据采集
# ============================================================
def fetch_sector_data():
    """
    获取申万行业板块和概念板块数据
    包括：涨跌幅、成交额、换手率、资金流向
    """
    print("\n" + "=" * 60)
    print("模块一：板块数据采集")
    print("=" * 60)

    sector_data = {}

    # 1. 申万行业板块行情
    print("\n[1/4] 获取申万行业板块行情...")
    try:
        df_sw = ak.stock_board_industry_name_em()
        if df_sw is not None and len(df_sw) > 0:
            sector_data["sw_industry"] = df_sw.to_dict(orient="records")
            print(f"  获取到 {len(df_sw)} 个申万行业板块")
        else:
            print("  ⚠ 申万行业板块数据为空")
    except Exception as e:
        print(f"  ✗ 申万行业板块获取失败: {e}")

    # 2. 概念板块行情
    print("\n[2/4] 获取概念板块行情...")
    try:
        df_concept = ak.stock_board_concept_name_em()
        if df_concept is not None and len(df_concept) > 0:
            sector_data["concept"] = df_concept.to_dict(orient="records")
            print(f"  获取到 {len(df_concept)} 个概念板块")
        else:
            print("  ⚠ 概念板块数据为空")
    except Exception as e:
        print(f"  ✗ 概念板块获取失败: {e}")

    # 3. 行业板块资金流向
    print("\n[3/4] 获取行业板块资金流向...")
    try:
        df_flow = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
        if df_flow is not None and len(df_flow) > 0:
            sector_data["industry_fund_flow"] = df_flow.to_dict(orient="records")
            print(f"  获取到 {len(df_flow)} 条行业资金流向数据")
    except Exception as e:
        print(f"  ✗ 行业资金流向获取失败: {e}")
        # 尝试备用接口
        try:
            df_flow = ak.stock_fund_flow_industry(symbol="即时")
            if df_flow is not None:
                sector_data["industry_fund_flow_alt"] = df_flow.to_dict(orient="records")
                print(f"  (备用接口) 获取到资金流向数据")
        except Exception as e2:
            print(f"  ✗ 备用接口也失败: {e2}")

    # 4. 概念板块资金流向
    print("\n[4/4] 获取概念板块资金流向...")
    try:
        df_cflow = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流")
        if df_cflow is not None and len(df_cflow) > 0:
            sector_data["concept_fund_flow"] = df_cflow.to_dict(orient="records")
            print(f"  获取到 {len(df_cflow)} 条概念资金流向数据")
    except Exception as e:
        print(f"  ✗ 概念资金流向获取失败: {e}")

    save_json(sector_data, f"sector_data_{TODAY}.json")
    return sector_data


# ============================================================
# 模块二：板块成分股获取
# ============================================================
def fetch_sector_components(sector_names: list):
    """
    获取指定板块的成分股列表
    sector_names: 板块名称列表
    """
    print("\n" + "=" * 60)
    print("模块二：板块成分股采集")
    print("=" * 60)

    components = {}
    for name in sector_names:
        print(f"\n  获取板块 [{name}] 的成分股...")
        try:
            # 尝试行业板块
            df = ak.stock_board_industry_cons_em(symbol=name)
            if df is not None and len(df) > 0:
                components[name] = df.to_dict(orient="records")
                print(f"  ✓ {name}: {len(df)} 只成分股")
                continue
        except Exception:
            pass

        try:
            # 尝试概念板块
            df = ak.stock_board_concept_cons_em(symbol=name)
            if df is not None and len(df) > 0:
                components[name] = df.to_dict(orient="records")
                print(f"  ✓ {name}: {len(df)} 只成分股")
                continue
        except Exception:
            pass

        print(f"  ✗ {name}: 未找到成分股数据")

    save_json(components, f"sector_components_{TODAY}.json")
    return components


# ============================================================
# 模块三：个股行情数据
# ============================================================
def fetch_stock_data(stock_codes: list, period="daily"):
    """
    获取个股历史行情（最近60个交易日）
    """
    print("\n" + "=" * 60)
    print(f"模块三：个股行情数据采集 ({len(stock_codes)}只)")
    print("=" * 60)

    all_stock_data = {}
    for i, code in enumerate(stock_codes):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(stock_codes)}")
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period=period,
                start_date=DATE_60D_AGO,
                end_date=TODAY,
                adjust="qfq"  # 前复权
            )
            if df is not None and len(df) > 0:
                all_stock_data[code] = df.to_dict(orient="records")
        except Exception as e:
            print(f"  ✗ {code}: {e}")

    print(f"  共获取 {len(all_stock_data)}/{len(stock_codes)} 只个股行情")
    save_json(all_stock_data, f"stock_daily_{TODAY}.json")
    return all_stock_data


# ============================================================
# 模块四：个股资金流向
# ============================================================
def fetch_stock_fund_flow(stock_codes: list):
    """
    获取个股主力资金流向数据
    """
    print("\n" + "=" * 60)
    print(f"模块四：个股资金流向采集 ({len(stock_codes)}只)")
    print("=" * 60)

    fund_flow_data = {}
    for i, code in enumerate(stock_codes):
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(stock_codes)}")
        try:
            df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith("6") else "sz")
            if df is not None and len(df) > 0:
                fund_flow_data[code] = df.head(20).to_dict(orient="records")
        except Exception:
            pass  # 静默处理，部分股票可能无此数据

    print(f"  共获取 {len(fund_flow_data)}/{len(stock_codes)} 只个股资金流向")
    save_json(fund_flow_data, f"stock_fund_flow_{TODAY}.json")
    return fund_flow_data


# ============================================================
# 模块五：个股财务数据
# ============================================================
def fetch_financial_data(stock_codes: list):
    """
    获取个股基本财务指标（用于成长估值型和多因子型专家）
    """
    print("\n" + "=" * 60)
    print(f"模块五：个股财务数据采集 ({len(stock_codes)}只)")
    print("=" * 60)

    financial_data = {}
    for i, code in enumerate(stock_codes):
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(stock_codes)}")
        try:
            # 获取财务指标
            df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
            if df is not None and len(df) > 0:
                financial_data[code] = df.head(8).to_dict(orient="records")
        except Exception:
            pass

    print(f"  共获取 {len(financial_data)}/{len(stock_codes)} 只个股财务数据")
    save_json(financial_data, f"stock_financial_{TODAY}.json")
    return financial_data


# ============================================================
# 模块六：市场热点和新闻
# ============================================================
def fetch_market_hot():
    """
    获取当日市场热点信息
    """
    print("\n" + "=" * 60)
    print("模块六：市场热点采集")
    print("=" * 60)

    hot_data = {}

    # 1. 涨停股池
    print("\n[1/4] 获取涨停股池...")
    try:
        df = ak.stock_zt_pool_em(date=TODAY)
        if df is not None and len(df) > 0:
            hot_data["zt_pool"] = df.to_dict(orient="records")
            print(f"  获取到 {len(df)} 只涨停股")
    except Exception as e:
        print(f"  ✗ 涨停股池获取失败: {e}")

    # 2. 热门股票排行
    print("\n[2/4] 获取热门股票...")
    try:
        df = ak.stock_hot_rank_em()
        if df is not None and len(df) > 0:
            hot_data["hot_rank"] = df.head(50).to_dict(orient="records")
            print(f"  获取到热门股票排行")
    except Exception as e:
        print(f"  ✗ 热门股票获取失败: {e}")

    # 3. 龙虎榜
    print("\n[3/4] 获取龙虎榜数据...")
    try:
        df = ak.stock_lhb_detail_em(
            start_date=(datetime.now() - timedelta(days=5)).strftime("%Y%m%d"),
            end_date=TODAY
        )
        if df is not None and len(df) > 0:
            hot_data["lhb"] = df.to_dict(orient="records")
            print(f"  获取到 {len(df)} 条龙虎榜数据")
    except Exception as e:
        print(f"  ✗ 龙虎榜获取失败: {e}")

    # 4. 北向资金
    print("\n[4/4] 获取北向资金数据...")
    try:
        df_hgt = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        if df_hgt is not None and len(df_hgt) > 0:
            hot_data["north_flow"] = df_hgt.tail(20).to_dict(orient="records")
            print(f"  获取到北向资金数据")
    except Exception as e:
        print(f"  ✗ 北向资金获取失败: {e}")

    save_json(hot_data, f"market_hot_{TODAY}.json")
    return hot_data


# ============================================================
# 模块七：融资融券数据
# ============================================================
def fetch_margin_data():
    """
    获取融资融券数据
    """
    print("\n" + "=" * 60)
    print("模块七：融资融券数据采集")
    print("=" * 60)

    margin_data = {}
    try:
        df = ak.stock_margin_underlying_info_sz(date=TODAY.replace("-", ""))
        if df is not None and len(df) > 0:
            margin_data["sz_margin"] = df.to_dict(orient="records")
            print(f"  获取到深市融资融券标的 {len(df)} 只")
    except Exception as e:
        print(f"  ✗ 深市融资融券获取失败: {e}")

    save_json(margin_data, f"margin_data_{TODAY}.json")
    return margin_data


# ============================================================
# 主流程
# ============================================================
def run_full_pipeline():
    """
    完整数据采集流程
    """
    print("=" * 60)
    print(f"A股多智能体选股系统 — 数据采集")
    print(f"采集时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: 板块数据
    sector_data = fetch_sector_data()

    # Step 2: 市场热点
    hot_data = fetch_market_hot()

    # Step 3: 融资融券
    margin_data = fetch_margin_data()

    print("\n" + "=" * 60)
    print("基础数据采集完成！")
    print("=" * 60)
    print(f"\n后续步骤:")
    print(f"1. 将板块数据提供给AI执行板块初筛")
    print(f"2. 确定核心候选板块后，运行以下命令获取成分股数据:")
    print(f'   python data_fetcher.py --sectors "板块1,板块2,板块3"')
    print(f"3. 将所有数据提供给AI执行6位专家选股")

    return {
        "sector_data": sector_data,
        "hot_data": hot_data,
        "margin_data": margin_data,
    }


def run_sector_deep(sector_names: list):
    """
    针对指定板块进行深度数据采集
    """
    print(f"\n对指定板块进行深度采集: {sector_names}")

    # 获取成分股
    components = fetch_sector_components(sector_names)

    # 汇总所有成分股代码
    all_codes = []
    for name, stocks in components.items():
        for s in stocks:
            code = s.get("代码", s.get("stock_code", ""))
            if code:
                all_codes.append(code)
    all_codes = list(set(all_codes))
    print(f"\n共 {len(all_codes)} 只成分股需要采集详细数据")

    if len(all_codes) == 0:
        print("未获取到成分股，退出深度采集")
        return

    # 获取行情
    stock_data = fetch_stock_data(all_codes)

    # 获取资金流向
    fund_flow = fetch_stock_fund_flow(all_codes)

    # 获取财务数据
    financial = fetch_financial_data(all_codes)

    print("\n" + "=" * 60)
    print("深度数据采集完成！所有数据已保存为JSON文件。")
    print("请将这些文件提供给AI智能体执行选股分析。")
    print("=" * 60)


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sectors":
        # 深度采集模式
        if len(sys.argv) > 2:
            sectors = [s.strip() for s in sys.argv[2].split(",")]
            run_sector_deep(sectors)
        else:
            print("请指定板块名称，用逗号分隔")
            print('示例: python data_fetcher.py --sectors "算力,机器人,创新药"')
    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("A股多智能体选股系统 — 数据采集模块")
        print()
        print("用法:")
        print("  python data_fetcher.py              # 基础数据采集（板块+热点+融资融券）")
        print('  python data_fetcher.py --sectors "算力,机器人"  # 指定板块深度采集')
        print("  python data_fetcher.py --help        # 显示帮助")
    else:
        # 基础采集模式
        run_full_pipeline()
