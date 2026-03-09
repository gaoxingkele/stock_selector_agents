#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 数据引擎模块

功能:
  1. 板块数据采集（资金流向、涨跌幅、换手率）
  2. 多周期 K 线采集（月线/周线/日线）
  3. 技术指标计算（MA、MACD、RSI、KDJ、布林带）
  4. 个股资金流向数据
  5. 财务基本面数据
  6. 市场热点（涨停、龙虎榜、北向资金）
  7. 融资融券数据
  8. 综合股票画像生成（供LLM分析用）

数据源: TDX本地通达信（主）/ akshare（备）/ tushare（末）
"""

import concurrent.futures
import json
import os
import threading
import time
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 屏蔽 mootdx 的 FutureWarning（pandas fillna method= 参数已废弃，不影响功能）
warnings.filterwarnings("ignore", category=FutureWarning, module="mootdx")

try:
    import akshare as ak
    # Fix: bypass Windows system proxy for all akshare calls
    # (Windows system proxy is set for LLM APIs but breaks domestic Chinese data sources)
    try:
        import requests as _requests
        import random as _rand
        import time as _time
        from requests.adapters import HTTPAdapter as _HTTPAdapter
        import akshare.utils.request as _ak_req

        def _request_no_proxy(
            url, params=None, timeout=15, max_retries=3,
            base_delay=1.0, random_delay_range=(0.5, 1.5),
        ):
            last_ex = None
            for attempt in range(max_retries):
                try:
                    with _requests.Session() as s:
                        s.trust_env = False  # bypass Windows system proxy
                        adapter = _HTTPAdapter(pool_connections=1, pool_maxsize=1)
                        s.mount("http://", adapter)
                        s.mount("https://", adapter)
                        resp = s.get(url, params=params, timeout=timeout)
                        resp.raise_for_status()
                        return resp
                except (_requests.RequestException, ValueError) as e:
                    last_ex = e
                    if attempt < max_retries - 1:
                        _time.sleep(base_delay * (2 ** attempt) + _rand.uniform(*random_delay_range))
            raise last_ex

        # Patch both the source module and the func module's local reference
        _ak_req.request_with_retry = _request_no_proxy
        import akshare.utils.func as _ak_func
        _ak_func.request_with_retry = _request_no_proxy
    except Exception:
        pass  # patch failed silently, akshare still works normally
except ImportError:
    print("请先安装: pip install akshare")
    ak = None

try:
    import tushare as ts
except ImportError:
    ts = None

try:
    from mootdx.reader import Reader as TdxReader
    _MOOTDX_AVAILABLE = True
except ImportError:
    _MOOTDX_AVAILABLE = False


# ===================================================================== #
#  全局常量                                                              #
# ===================================================================== #

TODAY = datetime.now().strftime("%Y%m%d")
DATE_1Y_AGO = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")   # 月线
DATE_2Y_AGO = (datetime.now() - timedelta(days=760)).strftime("%Y%m%d")   # 备用
DATE_120D_AGO = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d") # 日线/周线
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE_DIR, "output", "data")   # output/data/ 中间JSON
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_json(data: Any, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


def _load_json(filename: str) -> Optional[Any]:
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _pct_change(series: pd.Series, periods: int) -> Optional[float]:
    if len(series) <= periods:
        return None
    try:
        old = float(series.iloc[-(periods + 1)])
        new = float(series.iloc[-1])
        return round((new - old) / old * 100, 2) if old != 0 else None
    except Exception:
        return None


# ===================================================================== #
#  技术指标计算                                                          #
# ===================================================================== #

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 OHLCV 数据计算常用技术指标。
    要求列名: open, high, low, close, volume
    """
    df = df.copy()
    close = df["close"]
    vol = df["volume"]

    # ── 均线 ──────────────────────────────────────────────
    for n in [5, 10, 20, 60]:
        df[f"ma{n}"] = close.rolling(n, min_periods=1).mean().round(3)

    # ── MACD (12, 26, 9) ──────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["dif"] = (ema12 - ema26).round(4)
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean().round(4)
    df["macd_hist"] = (2 * (df["dif"] - df["dea"])).round(4)

    # ── RSI (6, 14) ────────────────────────────────────────
    delta = close.diff()
    for span in [6, 14]:
        gain = delta.clip(lower=0).ewm(com=span - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=span - 1, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-9)
        df[f"rsi{span}"] = (100 - 100 / (1 + rs)).round(2)

    # ── KDJ (9, 3, 3) ─────────────────────────────────────
    low9 = df["low"].rolling(9, min_periods=1).min()
    high9 = df["high"].rolling(9, min_periods=1).max()
    rsv = ((close - low9) / (high9 - low9 + 1e-9) * 100).fillna(50)
    df["kdj_k"] = rsv.ewm(com=2, adjust=False).mean().round(2)
    df["kdj_d"] = df["kdj_k"].ewm(com=2, adjust=False).mean().round(2)
    df["kdj_j"] = (3 * df["kdj_k"] - 2 * df["kdj_d"]).round(2)

    # ── 布林带 (20, 2) ────────────────────────────────────
    df["bb_mid"] = close.rolling(20, min_periods=1).mean().round(3)
    bb_std = close.rolling(20, min_periods=1).std().fillna(0)
    df["bb_upper"] = (df["bb_mid"] + 2 * bb_std).round(3)
    df["bb_lower"] = (df["bb_mid"] - 2 * bb_std).round(3)
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_pos"] = ((close - df["bb_lower"]) / bb_range.replace(0, 1e-9) * 100).round(1)
    df["bb_width"] = (bb_range / df["bb_mid"] * 100).round(2)

    # ── 成交量指标 ─────────────────────────────────────────
    df["vol_ma5"] = vol.rolling(5, min_periods=1).mean()
    df["vol_ma20"] = vol.rolling(20, min_periods=1).mean()
    df["vol_ratio"] = (vol / df["vol_ma20"].replace(0, 1e-9)).round(2)

    return df


def _ma_arrange(df: pd.DataFrame) -> str:
    """判断日线均线排列状态"""
    try:
        row = df.iloc[-1]
        ma5, ma10, ma20, ma60 = (
            row.get("ma5", 0), row.get("ma10", 0),
            row.get("ma20", 0), row.get("ma60", 0),
        )
        if ma5 > ma10 > ma20 > ma60:
            return "强多头排列"
        if ma5 > ma10 > ma20:
            return "多头排列"
        if ma5 < ma10 < ma20 < ma60:
            return "强空头排列"
        if ma5 < ma10 < ma20:
            return "空头排列"
        return "均线纠缠/震荡"
    except Exception:
        return "数据不足"


def _macd_status(df: pd.DataFrame) -> str:
    try:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        dif, dea, hist = last.get("dif", 0), last.get("dea", 0), last.get("macd_hist", 0)
        prev_hist = prev.get("macd_hist", 0)

        above_zero = dif > 0 and dea > 0
        cross_up = last.get("dif", 0) > last.get("dea", 0) and prev.get("dif", 0) <= prev.get("dea", 0)
        cross_dn = last.get("dif", 0) < last.get("dea", 0) and prev.get("dif", 0) >= prev.get("dea", 0)
        hist_grow = hist > prev_hist

        if cross_up:
            zone = "零轴上方金叉" if above_zero else "零轴下方金叉"
        elif cross_dn:
            zone = "零轴上方死叉" if above_zero else "零轴下方死叉"
        elif above_zero:
            zone = "零轴上方多头" + ("，柱体扩张" if hist_grow else "，柱体收缩")
        else:
            zone = "零轴下方空头" + ("，柱体扩张" if not hist_grow else "，柱体收缩")

        return f"DIF={dif:.3f} DEA={dea:.3f} 柱={hist:.3f} [{zone}]"
    except Exception:
        return "数据不足"


# ===================================================================== #
#  K线图表生成（模块级工具函数）                                         #
# ===================================================================== #

def generate_kline_chart(code: str, df_daily: "pd.DataFrame", chart_dir: str) -> Optional[str]:
    """生成K线+成交量图表，返回图片路径；失败返回None"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        os.makedirs(chart_dir, exist_ok=True)
        df = df_daily.tail(60).copy().reset_index(drop=True)
        if len(df) < 10:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]})
        fig.suptitle(f"{code}", fontsize=10)

        # 尝试用 mplfinance
        try:
            import mplfinance as mpf
            plt.close(fig)
            df_mpf = df_daily.tail(60).copy()
            df_mpf.index = pd.to_datetime(df_mpf.index) if not isinstance(df_mpf.index, pd.DatetimeIndex) else df_mpf.index
            for col_map in [{"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}]:
                df_mpf = df_mpf.rename(columns={v: k for k, v in col_map.items() if v in df_mpf.columns})
            save_path = os.path.join(chart_dir, f"{code}.png")
            mpf.plot(df_mpf, type="candle", volume=True, savefig=save_path, style="charles", figsize=(10, 6))
            return save_path if os.path.exists(save_path) else None
        except Exception:
            pass

        # 降级：用 matplotlib.patches 手动绘蜡烛
        for i, row in df.iterrows():
            o = float(row.get("open", row.get("close", 0)))
            h = float(row.get("high", 0))
            l = float(row.get("low", 0))
            c = float(row.get("close", 0))
            color = "red" if c >= o else "green"
            ax1.plot([i, i], [l, h], color=color, linewidth=0.8)
            ax1.add_patch(mpatches.Rectangle((i - 0.3, min(o, c)), 0.6, abs(c - o) or 0.001, color=color))

        if "volume" in df.columns:
            vols = df["volume"].astype(float)
            colors = ["red" if df["close"].iloc[i] >= df["open"].iloc[i] else "green" for i in range(len(df))]
            ax2.bar(range(len(df)), vols, color=colors, alpha=0.7)

        plt.tight_layout()
        save_path = os.path.join(chart_dir, f"{code}.png")
        plt.savefig(save_path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        return save_path
    except Exception:
        return None


# ===================================================================== #
#  板块数据                                                              #
# ===================================================================== #

class DataEngine:
    """A股数据引擎：采集、处理、生成股票画像"""

    def __init__(self, tushare_token: str = "", tdx_dir: str = "d:/tdx"):
        self.tushare_token = tushare_token
        self._pro = None
        if tushare_token and ts:
            try:
                ts.set_token(tushare_token)
                self._pro = ts.pro_api()
            except Exception as e:
                print(f"  [tushare] 初始化失败: {e}")

        # 初始化本地 TDX 读取器
        self._tdx = None
        if _MOOTDX_AVAILABLE and tdx_dir and os.path.isdir(tdx_dir):
            try:
                self._tdx = TdxReader.factory(market='std', tdxdir=tdx_dir)
                print(f"  [TDX] 本地数据源已加载: {tdx_dir}")
            except Exception as e:
                print(f"  [TDX] 初始化失败: {e}")
        else:
            if not _MOOTDX_AVAILABLE:
                print("  [TDX] mootdx 未安装（pip install mootdx），跳过本地数据")
            elif tdx_dir:
                print(f"  [TDX] 目录不存在: {tdx_dir}，跳过本地数据")

    # ------------------------------------------------------------------ #
    #  板块总览                                                            #
    # ------------------------------------------------------------------ #

    def fetch_sector_overview(self) -> Dict:
        """
        获取板块行情数据（申万行业 + 概念板块）及资金流向。
        返回聚合字典，供板块初筛智能体使用。
        """
        print("[数据] 采集板块总览数据...")
        data: Dict = {}

        # 申万行业板块行情
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and len(df) > 0:
                data["sw_industry"] = df.to_dict("records")
                print(f"  申万行业: {len(df)} 个板块")
        except Exception as e:
            print(f"  [警告] 申万行业获取失败: {e}")

        # 概念板块行情
        try:
            df = ak.stock_board_concept_name_em()
            if df is not None and len(df) > 0:
                data["concept"] = df.to_dict("records")
                print(f"  概念板块: {len(df)} 个板块")
        except Exception as e:
            print(f"  [警告] 概念板块获取失败: {e}")

        # 行业资金流向
        try:
            df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
            if df is not None and len(df) > 0:
                data["industry_fund_flow"] = df.to_dict("records")
                print(f"  行业资金流向: {len(df)} 条")
        except Exception as e:
            print(f"  [警告] 行业资金流向失败: {e}")
            # 备用
            try:
                df = ak.stock_fund_flow_industry(symbol="即时")
                if df is not None:
                    data["industry_fund_flow_alt"] = df.to_dict("records")
            except Exception:
                pass

        # 概念资金流向
        try:
            df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流")
            if df is not None and len(df) > 0:
                data["concept_fund_flow"] = df.to_dict("records")
                print(f"  概念资金流向: {len(df)} 条")
        except Exception as e:
            print(f"  [警告] 概念资金流向失败: {e}")

        _save_json(data, f"sector_overview_{TODAY}.json")
        return data

    # ------------------------------------------------------------------ #
    #  板块成分股                                                          #
    # ------------------------------------------------------------------ #

    def _get_sw_index_map(self) -> Dict[str, str]:
        """
        获取申万二级行业指数名→代码映射（带缓存）。
        返回 {"油服工程": "801962.SI", ...}
        """
        if hasattr(self, "_sw_index_cache"):
            return self._sw_index_cache
        if not self._pro:
            self._sw_index_cache = {}
            return {}
        try:
            df = self._pro.index_basic(market="SW")
            df2 = df[df["category"].str.contains("二级", na=False)]
            mapping: Dict[str, str] = {}
            for _, row in df2.iterrows():
                raw_name: str = str(row["name"])
                # 去掉 "(申万)"、"Ⅱ" 等后缀变体供模糊匹配用
                clean = raw_name.replace("(申万)", "").replace("（申万）", "").strip()
                mapping[clean] = str(row["ts_code"])
                mapping[raw_name] = str(row["ts_code"])
            self._sw_index_cache = mapping
        except Exception:
            self._sw_index_cache = {}
        return self._sw_index_cache

    def _tushare_sector_stocks(self, sector_name: str) -> List[Dict]:
        """
        通过 tushare 申万行业索引获取板块成分股。
        支持模糊名称匹配（去掉 'Ⅱ'、'(申万)' 等变体）。
        """
        if not self._pro:
            return []
        sw_map = self._get_sw_index_map()
        # 构建候选匹配键
        candidates = [
            sector_name,
            sector_name + "Ⅱ",
            sector_name.rstrip("Ⅱ"),
            sector_name + "(申万)",
        ]
        index_code = None
        for cand in candidates:
            if cand in sw_map:
                index_code = sw_map[cand]
                break
        # 还没找到，做子串匹配
        if not index_code:
            for key, code in sw_map.items():
                if sector_name in key or key in sector_name:
                    index_code = code
                    break
        if not index_code:
            return []
        try:
            df = self._pro.index_member(index_code=index_code, fields="con_code,con_name")
            if df is not None and not df.empty:
                stocks = []
                for _, row in df.iterrows():
                    code = str(row["con_code"]).split(".")[0]
                    stocks.append({"代码": code, "名称": str(row["con_name"])})
                return stocks
        except Exception:
            pass

        # 最终降级：stock_basic 行业关键字匹配
        # 申万→tushare行业关键字别名（常见映射）
        _SW_TO_TUSHARE_KEYWORD = {
            "油气开采": ["石油开采"],
            "炼化及贸": ["石油加工", "石油贸易"],
            "油服工程": ["石油开采"],
            "煤炭开采": ["煤炭开采"],
            "焦炭": ["焦炭加工"],
            "燃气": ["供气供热"],
            "航运港口": ["水运", "港口"],
            "航运": ["水运"],
            "港口": ["港口"],
            "半导体": ["半导体"],
            "软件开发": ["软件服务"],
            "人工智能": ["互联网", "IT设备", "软件服务"],
            "机器人": ["专用机械", "工程机械"],
            "新能源": ["新型电力", "电气设备"],
            "医疗器械": ["医疗保健"],
            "白酒": ["白酒"],
            "银行": ["银行"],
            "证券": ["多元金融"],
            "保险": ["保险"],
            "电力": ["火力发电", "水力发电", "新型电力"],
            "光伏": ["新型电力"],
            "风电": ["新型电力"],
            "电池": ["元器件", "电气设备"],
        }
        try:
            # 查找别名
            keywords: List[str] = []
            for sw_key, ts_industries in _SW_TO_TUSHARE_KEYWORD.items():
                if sw_key in sector_name or sector_name in sw_key:
                    keywords.extend(ts_industries)
            if not keywords:
                # 原始关键词（去掉 Ⅱ/Ⅲ）
                kw = sector_name.replace("Ⅱ", "").replace("Ⅲ", "").strip()
                keywords = [kw]

            all_df = self._pro.stock_basic(
                exchange="", list_status="L",
                fields="ts_code,symbol,name,industry",
            )
            if all_df is not None and not all_df.empty:
                mask = all_df["industry"].str.contains("|".join(keywords), na=False)
                matched = all_df[mask]
                if not matched.empty:
                    stocks = []
                    for _, row in matched.iterrows():
                        code = str(row["ts_code"]).split(".")[0]
                        stocks.append({"代码": code, "名称": str(row["name"])})
                    return stocks
        except Exception:
            pass
        return []

    def fetch_sector_components(self, sector_names: List[str]) -> Dict[str, List[Dict]]:
        """获取指定板块的成分股列表（akshare→tushare申万降级）"""
        print(f"[数据] 获取板块成分股: {sector_names}")
        components: Dict[str, List[Dict]] = {}

        for name in sector_names:
            stocks = []
            # 尝试 akshare 行业板块
            try:
                df = ak.stock_board_industry_cons_em(symbol=name)
                if df is not None and len(df) > 0:
                    stocks = df.to_dict("records")
            except Exception:
                pass

            # 尝试 akshare 概念板块
            if not stocks:
                try:
                    df = ak.stock_board_concept_cons_em(symbol=name)
                    if df is not None and len(df) > 0:
                        stocks = df.to_dict("records")
                except Exception:
                    pass

            # 降级：tushare 申万行业索引
            if not stocks:
                stocks = self._tushare_sector_stocks(name)
                if stocks:
                    print(f"  {name}: {len(stocks)} 只成分股 [tushare SW]")

            if stocks:
                components[name] = stocks
            else:
                print(f"  [警告] {name}: 未获取到成分股")

        return components

    # ------------------------------------------------------------------ #
    #  K 线数据（月/周/日）                                               #
    # ------------------------------------------------------------------ #

    def _tdx_kline(self, code: str, period: str, n_bars: int) -> Optional[pd.DataFrame]:
        """
        从本地 TDX 读取 K 线。
        period: "daily" | "weekly" | "monthly"
        返回标准化 DataFrame（列：date, open, high, low, close, volume, [amount]）
        失败或数据不足时返回 None。
        """
        if not self._tdx:
            return None
        try:
            df = self._tdx.daily(symbol=code, adjust='qfq')
            if df is None or df.empty or len(df) < 5:
                return None

            # 周线 / 月线：从日线重采样
            if period == 'weekly':
                df = df.resample('W').agg(
                    open=('open', 'first'), high=('high', 'max'),
                    low=('low', 'min'),   close=('close', 'last'),
                    volume=('volume', 'sum'), amount=('amount', 'sum'),
                ).dropna(subset=['close'])
            elif period == 'monthly':
                df = df.resample('ME').agg(
                    open=('open', 'first'), high=('high', 'max'),
                    low=('low', 'min'),   close=('close', 'last'),
                    volume=('volume', 'sum'), amount=('amount', 'sum'),
                ).dropna(subset=['close'])

            # 统一格式：DatetimeIndex → date 列，与 akshare 结果一致
            df = df.reset_index()                            # index 'date' → 列
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            return df.tail(n_bars) if len(df) >= 5 else None

        except Exception as e:
            print(f"  [TDX] {code} {period} 读取失败: {e}")
            return None

    def fetch_kline(
        self,
        code: str,
        period: str = "daily",
        n_bars: int = 120,
    ) -> Optional[pd.DataFrame]:
        """
        获取单只股票 K 线数据。
        period: "monthly" | "weekly" | "daily"
        n_bars: 目标 K 线条数
        数据源优先级: TDX本地通达信（Priority 1）→ akshare（Priority 2）→ tushare（Priority 3）
        """
        period_map = {"monthly": "monthly", "weekly": "weekly", "daily": "daily"}
        ak_period = period_map.get(period, "daily")

        # 计算开始日期
        if period == "monthly":
            start = (datetime.now() - timedelta(days=n_bars * 35)).strftime("%Y%m%d")
        elif period == "weekly":
            start = (datetime.now() - timedelta(days=n_bars * 8)).strftime("%Y%m%d")
        else:
            start = (datetime.now() - timedelta(days=n_bars * 2)).strftime("%Y%m%d")

        # ── Priority 1: 本地 TDX ─────────────────────────────────────
        min_bars = 20 if period == "daily" else 5
        df = self._tdx_kline(code, period, n_bars)
        if df is not None and len(df) >= min_bars:
            return df

        # ── Priority 2: akshare 主接口 ───────────────────────────────
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period=ak_period,
                start_date=start,
                end_date=TODAY,
                adjust="qfq",
            )
            if df is not None and len(df) > 0:
                df = self._normalize_kline(df)
                return df.tail(n_bars)
        except Exception as e:
            print(f"  [akshare] {code} {period} K线失败: {e}")

        # ── Priority 3: tushare 备用接口 ────────────────────────────
        if self._pro:
            try:
                ts_code = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
                freq_map = {"monthly": "M", "weekly": "W", "daily": "D"}
                df = ts.pro_bar(
                    ts_code=ts_code,
                    adj="qfq",
                    freq=freq_map.get(period, "D"),
                    start_date=start,
                    end_date=TODAY,
                )
                if df is not None and len(df) > 0:
                    df = self._normalize_tushare_kline(df)
                    return df.tail(n_bars)
            except Exception as e:
                print(f"  [tushare] {code} {period} K线失败: {e}")

        return None

    @staticmethod
    def _normalize_kline(df: pd.DataFrame) -> pd.DataFrame:
        """统一 akshare K 线列名"""
        rename = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude",
            "涨跌幅": "pct_chg", "涨跌额": "change",
            "换手率": "turnover",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _normalize_tushare_kline(df: pd.DataFrame) -> pd.DataFrame:
        """统一 tushare K 线列名"""
        rename = {
            "trade_date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "vol": "volume",
            "amount": "amount", "pct_chg": "pct_chg",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    #  资金流向                                                            #
    # ------------------------------------------------------------------ #

    def fetch_fund_flow(self, code: str) -> Optional[pd.DataFrame]:
        """获取个股主力资金流向（近20日）"""
        market = "sh" if code.startswith("6") else "sz"
        try:
            df = ak.stock_individual_fund_flow(stock=code, market=market)
            if df is not None and len(df) > 0:
                return df.head(20)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    #  财务数据                                                            #
    # ------------------------------------------------------------------ #

    def fetch_financial(self, code: str) -> Optional[Dict]:
        """获取个股财务摘要指标"""
        # 同花顺财务摘要
        try:
            df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
            if df is not None and len(df) > 0:
                return {"ths_abstract": df.head(6).to_dict("records")}
        except Exception:
            pass

        # 备用: 股票基本信息
        try:
            df = ak.stock_individual_info_em(symbol=code)
            if df is not None and len(df) > 0:
                return {"basic_info": df.to_dict("records")}
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------ #
    #  实时行情（PE/PB/市值）                                              #
    # ------------------------------------------------------------------ #

    def fetch_realtime_info(self, codes: List[str]) -> Dict[str, Dict]:
        """批量获取实时行情（用于获取PE、PB、市值等）"""
        result: Dict[str, Dict] = {}
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and len(df) > 0:
                # 标准化列名
                if "代码" in df.columns:
                    df = df.set_index("代码")
                    for code in codes:
                        if code in df.index:
                            row = df.loc[code]
                            result[code] = row.to_dict()
        except Exception as e:
            print(f"  [警告] 实时行情获取失败: {e}")
        return result

    # ------------------------------------------------------------------ #
    #  市场热点                                                            #
    # ------------------------------------------------------------------ #

    def fetch_market_hot(self) -> Dict:
        """获取涨停股、热门股、龙虎榜、北向资金"""
        print("[数据] 采集市场热点数据...")
        data: Dict = {}

        # 涨停股池
        try:
            df = ak.stock_zt_pool_em(date=TODAY)
            if df is not None and len(df) > 0:
                data["zt_pool"] = df.to_dict("records")
                print(f"  涨停股: {len(df)} 只")
        except Exception as e:
            print(f"  [警告] 涨停股池: {e}")

        # 热门股票
        try:
            df = ak.stock_hot_rank_em()
            if df is not None and len(df) > 0:
                data["hot_rank"] = df.head(50).to_dict("records")
                print(f"  热门股票: Top50")
        except Exception as e:
            print(f"  [警告] 热门股票: {e}")

        # 龙虎榜（近5日）
        try:
            start5 = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")
            df = ak.stock_lhb_detail_em(start_date=start5, end_date=TODAY)
            if df is not None and len(df) > 0:
                data["lhb"] = df.to_dict("records")
                print(f"  龙虎榜: {len(df)} 条")
        except Exception as e:
            print(f"  [警告] 龙虎榜: {e}")

        # 北向资金
        try:
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            if df is not None and len(df) > 0:
                data["north_flow"] = df.tail(20).to_dict("records")
                print(f"  北向资金: 近20日")
        except Exception as e:
            print(f"  [警告] 北向资金: {e}")

        _save_json(data, f"market_hot_{TODAY}.json")
        return data

    # ------------------------------------------------------------------ #
    #  美股 / 港股热门数据                                                #
    # ------------------------------------------------------------------ #

    def fetch_global_hot(self) -> Dict:
        """
        获取美股、港股热门股票及板块信息，供跨市场主题分析使用。

        返回:
        {
          "us_hot":  [{"名称", "代码", "涨跌幅", "所属行业", ...}],
          "hk_hot":  [{"名称", "代码", "涨跌幅", "所属行业", ...}],
          "us_indices": {"纳斯达克": xx%, "道琼斯": xx%, "标普500": xx%},
          "data_available": bool,  # False 表示数据获取全部失败
        }
        """
        print("[数据] 采集美股/港股热门数据...")
        data: Dict = {"us_hot": [], "hk_hot": [], "us_indices": {}, "data_available": False}

        # ── 美股热门股票（东财） ─────────────────────────────────────
        try:
            df = ak.stock_us_hot_rank_em()
            if df is not None and len(df) > 0:
                data["us_hot"] = df.head(30).to_dict("records")
                print(f"  美股热门: {len(data['us_hot'])} 只")
                data["data_available"] = True
        except Exception as e:
            print(f"  [警告] 美股热门获取失败: {e}")

        # ── 港股热门股票（东财） ─────────────────────────────────────
        try:
            df = ak.stock_hk_hot_rank_em()
            if df is not None and len(df) > 0:
                data["hk_hot"] = df.head(30).to_dict("records")
                print(f"  港股热门: {len(data['hk_hot'])} 只")
                data["data_available"] = True
        except Exception as e:
            print(f"  [警告] 港股热门获取失败: {e}")

        # ── 美股主要指数涨跌 ─────────────────────────────────────────
        us_index_map = {
            "纳斯达克": "NASDAQ",
            "道琼斯":   "DJIA",
            "标普500":  "SPX",
        }
        for cn_name, symbol in us_index_map.items():
            try:
                df = ak.index_us_stock_sina(symbol=symbol)
                if df is not None and len(df) > 0:
                    last = df.iloc[-1]
                    chg = last.get("涨跌幅", last.get("change_pct", "N/A"))
                    data["us_indices"][cn_name] = chg
                    data["data_available"] = True
            except Exception:
                pass

        if not data["us_indices"]:
            # 退路：尝试东财美股指数
            try:
                df = ak.stock_us_index_spot_em()
                if df is not None and len(df) > 0:
                    for _, row in df.iterrows():
                        name = str(row.get("名称", row.get("index_name", "")))
                        chg = row.get("涨跌幅", row.get("change_pct", ""))
                        if name and chg:
                            data["us_indices"][name] = chg
                    data["data_available"] = True
            except Exception:
                pass

        if data["data_available"]:
            _save_json(data, f"global_hot_{TODAY}.json")
        else:
            print("  [警告] 美股/港股数据均获取失败，GX将依赖LLM自身知识")

        return data

    # ------------------------------------------------------------------ #
    #  融资融券                                                            #
    # ------------------------------------------------------------------ #

    def fetch_margin(self, code: Optional[str] = None) -> Dict:
        """获取融资融券数据"""
        data: Dict = {}
        try:
            df = ak.stock_margin_underlying_info_sz()
            if df is not None and len(df) > 0:
                if code:
                    filtered = df[df.get("证券代码", df.columns[0]).eq(code)] if "证券代码" in df.columns else df
                    data["margin_info"] = filtered.to_dict("records")
                else:
                    data["sz_margin_list"] = df.head(100).to_dict("records")
        except Exception:
            pass
        return data

    # ------------------------------------------------------------------ #
    #  业绩预告                                                           #
    # ------------------------------------------------------------------ #

    def fetch_profit_forecast(self, code: str) -> Optional[Dict]:
        """业绩预告数据（akshare）"""
        try:
            df = ak.stock_performance_forecast_em(symbol=code)
            if df is not None and len(df) > 0:
                latest = df.iloc[0]
                return {
                    "type": str(latest.get("业绩变动类型", "")),
                    "range": str(latest.get("业绩变动幅度", "")),
                    "reason": str(latest.get("业绩变动原因", ""))[:80],
                    "period": str(latest.get("预告期间", "")),
                }
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    #  股东户数                                                           #
    # ------------------------------------------------------------------ #

    def fetch_holder_num(self, code: str) -> Optional[Dict]:
        """股东户数变化（akshare，筹码集中度代理指标）"""
        try:
            df = ak.stock_holder_num_em(symbol=code)
            if df is not None and len(df) >= 2:
                latest = float(df.iloc[0].get("股东户数", 0))
                prev   = float(df.iloc[1].get("股东户数", 0))
                if prev > 0:
                    chg = (latest - prev) / prev * 100
                    return {"latest": latest, "chg_pct": round(chg, 1),
                            "trend": "减少" if chg < 0 else "增加"}
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    #  商誉+质押率                                                        #
    # ------------------------------------------------------------------ #

    def fetch_goodwill_pledge(self, code: str) -> Optional[Dict]:
        """商誉率+质押率（tushare）"""
        if not self._pro:
            return None
        ts_code = code + (".SH" if code.startswith("6") else ".SZ")
        result = {}
        try:
            df_bs = self._pro.balancesheet(ts_code=ts_code, fields="goodwill,total_hldr_eqy_exc_min_int", limit=2)
            if df_bs is not None and len(df_bs) > 0:
                gw = float(df_bs.iloc[0].get("goodwill", 0) or 0)
                eq = float(df_bs.iloc[0].get("total_hldr_eqy_exc_min_int", 1) or 1)
                result["goodwill_ratio"] = round(gw / max(eq, 1) * 100, 1)
        except Exception:
            pass
        try:
            df_pl = self._pro.pledge_stat(ts_code=ts_code)
            if df_pl is not None and len(df_pl) > 0:
                result["pledge_ratio"] = float(df_pl.iloc[0].get("pledge_ratio", 0) or 0)
        except Exception:
            pass
        return result or None

    # ------------------------------------------------------------------ #
    #  大宗交易                                                           #
    # ------------------------------------------------------------------ #

    def fetch_block_trade(self, code: str) -> Optional[Dict]:
        """大宗交易数据（akshare）"""
        try:
            df = ak.stock_dzjy_detail_em(symbol=code)
            if df is not None and len(df) > 0:
                count = len(df.head(30))
                discount_col = [c for c in df.columns if "折" in str(c) or "溢" in str(c)]
                avg_disc = None
                if discount_col:
                    try:
                        avg_disc = round(df.head(30)[discount_col[0]].astype(float).mean(), 2)
                    except Exception:
                        pass
                return {"count_30d": count, "avg_discount": avg_disc}
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    #  融资融券历史                                                        #
    # ------------------------------------------------------------------ #

    def fetch_margin_history(self, code: str) -> Optional[Dict]:
        """融资融券历史（akshare优先，tushare备用）"""
        try:
            market = "sh" if code.startswith("6") else "sz"
            df = ak.stock_margin_detail_szse(symbol=code) if market == "sz" \
                 else ak.stock_margin_detail_sse(symbol=code)
            if df is not None and len(df) >= 2:
                bal_col = [c for c in df.columns if "融资余额" in str(c)]
                if bal_col:
                    recent = float(df.iloc[0][bal_col[0]] or 0)
                    prev5  = float(df.iloc[min(4, len(df)-1)][bal_col[0]] or 0)
                    if prev5 > 0:
                        chg = (recent - prev5) / prev5 * 100
                        return {"balance": recent, "chg_5d_pct": round(chg, 1)}
        except Exception:
            pass
        if self._pro:
            try:
                ts_code = code + (".SH" if code.startswith("6") else ".SZ")
                df = self._pro.margin_detail(ts_code=ts_code, limit=5)
                if df is not None and len(df) >= 2:
                    recent = float(df.iloc[0].get("rzye", 0) or 0)
                    prev   = float(df.iloc[-1].get("rzye", 0) or 0)
                    if prev > 0:
                        chg = (recent - prev) / prev * 100
                        return {"balance": recent, "chg_5d_pct": round(chg, 1)}
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------ #
    #  全量股票数据包（供专家分析用）                                      #
    # ------------------------------------------------------------------ #

    def build_stock_data_package(
        self,
        codes: List[str],
        sector_map: Optional[Dict[str, str]] = None,
        max_workers: int = 6,
    ) -> Dict[str, Dict]:
        """
        为指定股票列表并行构建完整数据包（K线+指标+资金流+财务）。

        数据源优先级（每个 fetch_kline 调用内部已处理）:
          1. 通达信本地 TDX（最快，无网络，无限速）
          2. AKShare（国内接口，中速）
          3. Tushare（备用）
          4. 若以上全失败：数据包跳过该股票

        并发控制：
          - 若 TDX 可用，max_workers=6（本地读取快，并发无压力）
          - 若仅远程 API，max_workers 降至 3（限制并发防触发限流）
          - AKShare/Tushare 调用共享 Semaphore(3)，同时最多3个远程 API 请求

        sector_map: {code: sector_name}
        """
        total = len(codes)

        # 若只有远程 API 可用，限制并发避免触发限流
        if not self._tdx:
            max_workers = min(max_workers, 3)

        print(f"\n[数据] 并行构建 {total} 只股票数据包（{max_workers} 线程）...")

        # 批量获取实时行情（PE/PB/市值），单次全量拉取效率最高
        print("  获取实时行情（PE/PB/市值）...", end="", flush=True)
        t_rt = time.time()
        realtime = self.fetch_realtime_info(codes)
        print(f" ✓ {time.time()-t_rt:.1f}s，获取 {len(realtime)} 只")

        packages: Dict[str, Dict] = {}
        _pkgs_lock = threading.Lock()   # 保护 packages dict（写入）
        _done = [0]                     # 已完成（含跳过）计数
        _skip = [0]
        _cnt_lock = threading.Lock()

        # 远程 API 并发信号量：TDX本地读取不计入
        # 使用信号量而非降低 max_workers，这样 TDX 命中时仍能全速并行
        _api_sem = threading.Semaphore(3)

        def _fetch_kline_with_sem(code: str, period: str, n_bars: int):
            """先尝试 TDX（不占信号量），TDX失败再用远程 API（占信号量）"""
            df = self._tdx_kline(code, period, n_bars)
            if df is not None and len(df) > 0:
                return df
            # TDX 未命中，走远程（AKShare → Tushare）
            with _api_sem:
                return self.fetch_kline(code, period, n_bars)

        def _fetch_one(code: str) -> None:
            rt_info = realtime.get(code, {})
            name = rt_info.get("名称", rt_info.get("name", code))
            sector = (sector_map or {}).get(code, "")
            t_s = time.time()

            pkg: Dict = {"code": code, "sector": sector, "realtime": rt_info}

            # ── 日线（有多少拿多少，新股K线少也保留）───────────────
            df_d = _fetch_kline_with_sem(code, "daily", 120)
            if df_d is None or len(df_d) == 0:
                with _cnt_lock:
                    _skip[0] += 1
                    n = _done[0] + _skip[0]
                print(
                    f"  [{n:3d}/{total}] {code} {name:<6} ✗ 无日线数据，跳过",
                    flush=True,
                )
                return
            pkg["daily"] = compute_indicators(df_d)
            pkg["daily_bars"] = len(df_d)  # 记录实际K线根数，供画像参考

            # ── K线图表（用于E4视觉分析）────────────────────────────
            _chart_dir = os.path.join(_BASE_DIR, "output", "charts")
            cp = generate_kline_chart(code, df_d, _chart_dir)
            if cp:
                pkg["chart_path"] = cp

            # ── 周线（有多少拿多少）─────────────────────────────────
            df_w = _fetch_kline_with_sem(code, "weekly", 52)
            if df_w is not None and len(df_w) > 0:
                pkg["weekly"] = compute_indicators(df_w)

            # ── 月线（有多少拿多少）─────────────────────────────────
            df_m = _fetch_kline_with_sem(code, "monthly", 24)
            if df_m is not None and len(df_m) > 0:
                pkg["monthly"] = compute_indicators(df_m)

            # ── 资金流向 + 财务（远程 API，受信号量控速）────────────
            with _api_sem:
                pkg["fund_flow"] = self.fetch_fund_flow(code)
            with _api_sem:
                pkg["financial"] = self.fetch_financial(code)
            with _api_sem:
                pkg["profit_forecast"] = self.fetch_profit_forecast(code)
            with _api_sem:
                pkg["holder_num"] = self.fetch_holder_num(code)
            with _api_sem:
                pkg["goodwill_pledge"] = self.fetch_goodwill_pledge(code)
            with _api_sem:
                pkg["block_trade"] = self.fetch_block_trade(code)
            with _api_sem:
                pkg["margin_history"] = self.fetch_margin_history(code)

            elapsed_s = round(time.time() - t_s, 1)
            with _cnt_lock:
                _done[0] += 1
                n = _done[0] + _skip[0]

            has_w = "W" if "weekly" in pkg else "-"
            has_m = "M" if "monthly" in pkg else "-"
            print(
                f"  [{n:3d}/{total}] {code} {name:<6} {sector:<8}"
                f" {has_w}{has_m} ✓ {elapsed_s}s",
                flush=True,
            )
            with _pkgs_lock:
                packages[code] = pkg

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="DataFetch"
        ) as executor:
            futures = [executor.submit(_fetch_one, code) for code in codes]
            # 等待所有完成，传播未捕获异常
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  [数据] worker异常: {e}", flush=True)

        # ── 板块内RPS（近20日涨幅百分位）────────────────────────────
        sector_groups: Dict[str, List[str]] = {}
        for code, pkg in packages.items():
            sector_groups.setdefault(pkg.get("sector", ""), []).append(code)
        for _sector, codes_in_sector in sector_groups.items():
            perf = {}
            for c in codes_in_sector:
                df_d = packages[c].get("daily")
                if df_d is not None and len(df_d) >= 2:
                    try:
                        cl = df_d["close"]
                        n_back = min(20, len(cl) - 1)
                        perf[c] = (float(cl.iloc[-1]) - float(cl.iloc[-(n_back+1)])) / max(float(cl.iloc[-(n_back+1)]), 0.01) * 100
                    except Exception:
                        perf[c] = 0.0
                else:
                    perf[c] = 0.0
            sorted_codes = sorted(perf, key=lambda x: perf[x])
            n = len(sorted_codes)
            for rank, c in enumerate(sorted_codes):
                packages[c]["rps"] = round((rank + 1) / n * 100) if n > 0 else 50

        print(
            f"[数据] 完成: 有效 {len(packages)}/{total} 只"
            f"（跳过 {_skip[0]} 只日线不足）"
        )
        return packages

    # ------------------------------------------------------------------ #
    #  股票画像文本生成（供LLM分析用）                                    #
    # ------------------------------------------------------------------ #

    def build_stock_profile_slim(
        self,
        code: str,
        name: str,
        pkg: Dict,
    ) -> str:
        """
        极简股票画像：仅传标识信息 + 3-4 个关键数字，约 30-40 tokens/只。

        设计原则：
          不传 K 线序列、技术指标原始值、逐日成交量等冗余数据。
          LLM 通过自身云端知识与联网搜索获取深度信息；
          本地数据只作"坐标定位"用途。
        """
        sector = pkg.get("sector", "未知")
        rt = pkg.get("realtime", {})

        pe  = rt.get("市盈率-动态", rt.get("市盈率", "N/A"))
        pb  = rt.get("市净率", "N/A")
        cap = rt.get("总市值", "N/A")
        price = rt.get("最新价", rt.get("当前价", "N/A"))

        # 近N日涨幅（有多少K线算多少，新股可能不足20日）
        d20 = "N/A"
        df_d = pkg.get("daily")
        if df_d is not None and len(df_d) >= 2:
            try:
                c = df_d["close"]
                n_back = min(20, len(c) - 1)
                d20 = f"{(float(c.iloc[-1]) - float(c.iloc[-(n_back+1)])) / max(float(c.iloc[-(n_back+1)]), 0.01) * 100:+.1f}%({n_back}日)"
            except Exception:
                pass

        extra = ""
        # 业绩预告
        pf = pkg.get("profit_forecast")
        if pf and pf.get("type"):
            extra += f" | 业绩预告:{pf['type']}{pf.get('range','')}"
        # 股东户数变化
        hn = pkg.get("holder_num")
        if hn and "chg_pct" in hn:
            extra += f" | 股东户数:{hn['trend']}{abs(hn['chg_pct']):.1f}%"
        # 板块RPS
        rps = pkg.get("rps")
        if rps is not None:
            extra += f" | 板块RPS={rps}"
        # 商誉+质押率
        gp = pkg.get("goodwill_pledge")
        if gp:
            parts = []
            if "goodwill_ratio" in gp: parts.append(f"商誉率={gp['goodwill_ratio']}%")
            if "pledge_ratio" in gp: parts.append(f"质押率={gp['pledge_ratio']:.1f}%")
            if parts: extra += " | " + " ".join(parts)
        # 大宗交易
        bt = pkg.get("block_trade")
        if bt and bt.get("count_30d", 0) > 0:
            disc_str = f"折价{bt['avg_discount']}%" if bt.get("avg_discount") else ""
            extra += f" | 大宗交易:{bt['count_30d']}笔{disc_str}"
        # 融资融券
        mh = pkg.get("margin_history")
        if mh and "chg_5d_pct" in mh:
            direction = "增" if mh["chg_5d_pct"] > 0 else "减"
            extra += f" | 融资余额5日:{direction}{abs(mh['chg_5d_pct']):.1f}%"

        return (
            f"{code} {name} | 板块:{sector} | "
            f"PE={pe} PB={pb} 市值={cap} 现价={price} 近20日={d20}"
            f"{extra}"
        )

    def build_stock_profile(
        self,
        code: str,
        name: str,
        pkg: Dict,
    ) -> str:
        """
        将股票数据包转化为结构化文本摘要，供LLM分析。
        约 600-900 tokens。
        """
        lines = [
            f"═══ 股票画像: {code} {name} ═══",
            f"所属板块: {pkg.get('sector', '未知')} | 分析日期: {TODAY}",
        ]

        # ── 实时数据 ──────────────────────────────────────────
        rt = pkg.get("realtime", {})
        if rt:
            pe = rt.get("市盈率-动态", rt.get("市盈率", "N/A"))
            pb = rt.get("市净率", "N/A")
            mktcap = rt.get("总市值", "N/A")
            turnover_today = rt.get("换手率", "N/A")
            lines.append(
                f"PE(TTM)={pe} | PB={pb} | 总市值={mktcap} | 今日换手率={turnover_today}%"
            )

        # ── 月线趋势 ──────────────────────────────────────────
        df_m: Optional[pd.DataFrame] = pkg.get("monthly")
        if df_m is not None and len(df_m) >= 3:
            lines.append("\n【月线趋势（近12月）】")
            c = df_m["close"]
            m1 = _pct_change(c, 1)
            m3 = _pct_change(c, 3)
            m6 = _pct_change(c, 6) if len(df_m) > 6 else None
            m12 = _pct_change(c, 12) if len(df_m) > 12 else None

            pcts = []
            if m1 is not None: pcts.append(f"1月: {m1:+.1f}%")
            if m3 is not None: pcts.append(f"3月: {m3:+.1f}%")
            if m6 is not None: pcts.append(f"6月: {m6:+.1f}%")
            if m12 is not None: pcts.append(f"12月: {m12:+.1f}%")
            if pcts:
                lines.append("涨幅: " + " | ".join(pcts))

            # 月线均线
            last_m = df_m.iloc[-1]
            ma6 = round(df_m["close"].rolling(6, min_periods=1).mean().iloc[-1], 2)
            ma12 = round(df_m["close"].rolling(12, min_periods=1).mean().iloc[-1], 2)
            ma24 = round(df_m["close"].rolling(24, min_periods=1).mean().iloc[-1], 2) if len(df_m) >= 24 else "N/A"
            lines.append(f"月线MA: MA6={ma6} | MA12={ma12} | MA24={ma24}")
            # 月线量能比
            v_recent3 = df_m["volume"].tail(3).mean()
            v_all = df_m["volume"].mean()
            lines.append(f"近3月均量/全期均量 = {v_recent3/max(v_all,1):.2f}x")

        # ── 周线趋势 ──────────────────────────────────────────
        df_w: Optional[pd.DataFrame] = pkg.get("weekly")
        if df_w is not None and len(df_w) >= 5:
            lines.append("\n【周线趋势（近20周）】")
            c = df_w["close"]
            w4 = _pct_change(c, 4)
            w8 = _pct_change(c, 8) if len(df_w) > 8 else None
            w13 = _pct_change(c, 13) if len(df_w) > 13 else None
            pcts = []
            if w4 is not None: pcts.append(f"4周: {w4:+.1f}%")
            if w8 is not None: pcts.append(f"8周: {w8:+.1f}%")
            if w13 is not None: pcts.append(f"13周: {w13:+.1f}%")
            if pcts:
                lines.append("涨幅: " + " | ".join(pcts))

            last_w = df_w.iloc[-1]
            wma5 = round(df_w["close"].rolling(5, min_periods=1).mean().iloc[-1], 2)
            wma10 = round(df_w["close"].rolling(10, min_periods=1).mean().iloc[-1], 2)
            wma20 = round(df_w["close"].rolling(20, min_periods=1).mean().iloc[-1], 2) if len(df_w) >= 20 else "N/A"
            lines.append(f"周线MA: MA5={wma5} | MA10={wma10} | MA20={wma20}")
            v_r4 = df_w["volume"].tail(4).mean()
            v_all_w = df_w["volume"].mean()
            lines.append(f"近4周均量/均值 = {v_r4/max(v_all_w,1):.2f}x")

        # ── 日线详情 ──────────────────────────────────────────
        df_d: Optional[pd.DataFrame] = pkg.get("daily")
        if df_d is not None and len(df_d) >= 2:
            lines.append("\n【日线详情（近60日）】")
            c = df_d["close"]
            cur = float(c.iloc[-1])
            d5 = _pct_change(c, 5)
            d10 = _pct_change(c, 10)
            d20 = _pct_change(c, 20)
            d60 = _pct_change(c, min(60, len(c)-1))

            pcts = [f"当前价={cur}"]
            if d5 is not None: pcts.append(f"5日: {d5:+.1f}%")
            if d10 is not None: pcts.append(f"10日: {d10:+.1f}%")
            if d20 is not None: pcts.append(f"20日: {d20:+.1f}%")
            if d60 is not None: pcts.append(f"60日: {d60:+.1f}%")
            lines.append(" | ".join(pcts))

            last = df_d.iloc[-1]
            ma5 = round(last.get("ma5", 0), 2)
            ma10 = round(last.get("ma10", 0), 2)
            ma20 = round(last.get("ma20", 0), 2)
            ma60 = round(last.get("ma60", 0), 2)
            lines.append(
                f"日线均线: MA5={ma5} MA10={ma10} MA20={ma20} MA60={ma60} → {_ma_arrange(df_d)}"
            )
            lines.append(f"MACD: {_macd_status(df_d)}")

            rsi6 = round(last.get("rsi6", 0), 1)
            rsi14 = round(last.get("rsi14", 0), 1)
            rsi_desc = "超买" if rsi6 > 80 else ("超卖" if rsi6 < 20 else "正常区间")
            lines.append(f"RSI: RSI6={rsi6} RSI14={rsi14} [{rsi_desc}]")

            kdj_k = round(last.get("kdj_k", 0), 1)
            kdj_d = round(last.get("kdj_d", 0), 1)
            kdj_j = round(last.get("kdj_j", 0), 1)
            kdj_desc = "超买" if kdj_j > 100 else ("超卖" if kdj_j < 0 else "正常")
            lines.append(f"KDJ: K={kdj_k} D={kdj_d} J={kdj_j} [{kdj_desc}]")

            bb_pos = round(last.get("bb_pos", 50), 1)
            bb_width = round(last.get("bb_width", 0), 2)
            lines.append(f"布林带: 价格位于通道 {bb_pos}% | 带宽={bb_width}%")

            # 量能分析
            lines.append("\n【成交量分析】")
            v_ma5 = float(df_d["volume"].tail(5).mean())
            v_ma20 = float(df_d["volume"].tail(20).mean())
            v_ma60 = float(df_d["volume"].tail(60).mean()) if len(df_d) >= 60 else v_ma20
            v_today = float(df_d["volume"].iloc[-1])
            vol_ratio = round(v_today / max(v_ma20, 1), 2)
            lines.append(
                f"5日均量/20日均量 = {v_ma5/max(v_ma20,1):.2f}x | "
                f"今日量比(vs20日) = {vol_ratio:.2f}x"
            )

            if "turnover" in df_d.columns:
                to_5 = df_d["turnover"].tail(5).mean()
                to_20 = df_d["turnover"].tail(20).mean()
                to_60 = df_d["turnover"].tail(60).mean() if len(df_d) >= 60 else to_20
                lines.append(
                    f"换手率: 5日均={to_5:.2f}% | 20日均={to_20:.2f}% | "
                    f"60日均={to_60:.2f}% | 5日vs60日={to_5/max(to_60,0.01):.2f}x"
                )

        # ── 资金流向 ──────────────────────────────────────────
        ff: Optional[pd.DataFrame] = pkg.get("fund_flow")
        if ff is not None and len(ff) > 0:
            lines.append("\n【主力资金流向（近5日）】")
            ff_cols = ff.columns.tolist()
            # 寻找净流入列
            net_cols = [c for c in ff_cols if "净" in str(c) and ("亿" in str(c) or "额" in str(c))]
            big_cols = [c for c in ff_cols if "大单" in str(c) and "净" in str(c)]
            main_cols = [c for c in ff_cols if "主力" in str(c) and "净" in str(c)]

            recent5 = ff.head(5)
            if main_cols:
                mc = main_cols[0]
                try:
                    val = recent5[mc].astype(float).sum()
                    lines.append(f"主力净流入(5日合计): {val:.2f} 元")
                except Exception:
                    pass
            if big_cols:
                bc = big_cols[0]
                try:
                    val = recent5[bc].astype(float).sum()
                    lines.append(f"大单净流入(5日合计): {val:.2f} 元")
                except Exception:
                    pass
            if not main_cols and not big_cols and net_cols:
                for nc in net_cols[:2]:
                    try:
                        val = recent5[nc].astype(float).sum()
                        lines.append(f"{nc}(5日合计): {val:.2f}")
                    except Exception:
                        pass

        # ── 财务数据 ──────────────────────────────────────────
        fin = pkg.get("financial")
        if fin:
            lines.append("\n【基本面摘要（近2期报告）】")
            ths = fin.get("ths_abstract", [])
            if ths:
                for record in ths[:2]:
                    period_str = str(record.get("报告期", ""))
                    rev_growth = record.get("营业总收入同比增长率", record.get("营收增长率", "N/A"))
                    profit_growth = record.get("净利润同比增长率", record.get("净利润增速", "N/A"))
                    roe = record.get("净资产收益率", "N/A")
                    gross_margin = record.get("毛利率", "N/A")
                    lines.append(
                        f"报告期: {period_str} | 营收增长={rev_growth}% | "
                        f"净利润增长={profit_growth}% | ROE={roe}% | 毛利率={gross_margin}%"
                    )
            basic = fin.get("basic_info", [])
            if basic and not ths:
                for item in basic[:5]:
                    k = item.get("item", "")
                    v = item.get("value", "")
                    if k:
                        lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  全流程数据采集                                                      #
    # ------------------------------------------------------------------ #

    def run_full_data_collection(
        self,
        sector_names: Optional[List[str]] = None,
        max_stocks_per_sector: int = 30,
    ) -> Dict:
        """
        完整数据采集流程，返回:
        {
          "sector_overview": {...},
          "market_hot": {...},
          "sector_components": {"板块名": [stock,...]},
          "stock_packages": {"code": {...}},
          "sector_map": {"code": "板块名"},
        }
        """
        result: Dict = {}

        # 板块总览
        result["sector_overview"] = self.fetch_sector_overview()

        # 市场热点
        result["market_hot"] = self.fetch_market_hot()

        # 板块成分股（若指定）
        if sector_names:
            components = self.fetch_sector_components(sector_names)
            result["sector_components"] = components

            # 提取所有成分股
            all_codes = []
            sector_map: Dict[str, str] = {}
            for sname, stocks in components.items():
                for s in stocks:
                    code = str(s.get("代码", s.get("stock_code", s.get("code", "")))).strip()
                    if code and code not in sector_map:
                        all_codes.append(code)
                        sector_map[code] = sname

            # 每板块限制股票数量
            if len(all_codes) > len(sector_names) * max_stocks_per_sector:
                # 按板块均匀采样
                sampled = []
                for sname, stocks in components.items():
                    s_codes = [
                        str(s.get("代码", s.get("stock_code", s.get("code", "")))).strip()
                        for s in stocks
                        if str(s.get("代码", s.get("stock_code", s.get("code", "")))).strip()
                    ]
                    sampled.extend(s_codes[:max_stocks_per_sector])
                all_codes = list(dict.fromkeys(sampled))  # 去重保序

            result["sector_map"] = sector_map
            result["all_codes"] = all_codes

            print(f"\n[数据] 待分析股票: {len(all_codes)} 只")
            result["stock_packages"] = self.build_stock_data_package(all_codes, sector_map)

        return result
