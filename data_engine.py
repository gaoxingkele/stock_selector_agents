#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股多智能体选股系统 - 数据引擎模块

功能:
  1. 板块数据采集（资金流向、涨跌幅、换手率）
  2. 多周期 K 线采集（月线/周线/日线）
  3. 技术指标计算（MA、MACD、RSI、KDJ、布林带）
  4. 个股资金流向数据
"""

import concurrent.futures
import json
import os
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 屏蔽 mootdx 的 FutureWarning（pandas fillna method= 参数已废弃，不影响功能）
warnings.filterwarnings("ignore", category=FutureWarning, module="mootdx")

# ===================================================================== #
#  进程级代理屏蔽：所有国内数据接口（akshare/tushare/adata/baostock等）      #
#  必须直连，不经过 LLM_PROXY。在导入任何数据库之前清除代理环境变量，         #
#  然后用 requests 猴子补丁确保底层 HTTP 也不走代理。                       #
# ===================================================================== #

import requests as _requests

# 1) 保存 LLM 代理配置，然后从环境变量中移除（防止 requests/urllib3 自动读取）
_SAVED_PROXY = {}
for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                    "ALL_PROXY", "all_proxy"):
    if _proxy_key in os.environ:
        _SAVED_PROXY[_proxy_key] = os.environ.pop(_proxy_key)

# 2) 猴子补丁 requests.Session：所有 Session 默认 trust_env=False + 清空 proxies
_OrigSession = _requests.Session

class _NoProxySession(_OrigSession):
    """所有国内数据采集用的 Session，强制不走代理"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_env = False
        self.proxies = {"http": None, "https": None}

_requests.Session = _NoProxySession

# 3) 猴子补丁 requests.get/post 等快捷方法
_orig_request = _requests.api.request

def _no_proxy_request(method, url, **kwargs):
    kwargs.setdefault("proxies", {"http": None, "https": None})
    return _orig_request(method, url, **kwargs)

_requests.api.request = _no_proxy_request
_requests.get = lambda url, **kw: _no_proxy_request("GET", url, **kw)
_requests.post = lambda url, **kw: _no_proxy_request("POST", url, **kw)
_requests.head = lambda url, **kw: _no_proxy_request("HEAD", url, **kw)

# 4) 猴子补丁 urllib3（pandas.read_html / lxml 底层用的）
try:
    import urllib3
    _orig_urlopen = urllib3.HTTPConnectionPool.urlopen

    def _no_proxy_urlopen(self, method, url, **kwargs):
        # 确保不使用 CONNECT 隧道代理
        self.proxy = None
        self.proxy_headers = {}
        return _orig_urlopen(self, method, url, **kwargs)

    urllib3.HTTPConnectionPool.urlopen = _no_proxy_urlopen
except Exception:
    pass

# 5) 设置 NO_PROXY 环境变量覆盖所有国内域名
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# 6) DNS 修复：代理软件可能劫持国内域名解析到虚假 IP（如 198.18.x.x），
#    导致 akshare/东方财富等国内接口连不上。用公共DNS(114.114.114.114)
#    解析国内域名，替换 getaddrinfo 返回真实 IP 但保留原始地址族信息。
import socket as _socket

_PUBLIC_DNS = "114.114.114.114"
_DOMESTIC_SUFFIXES = (
    ".eastmoney.com", ".sina.com.cn", ".sinajs.cn", ".mairui.club",
    ".10jqka.com.cn", ".ssec.com.cn", ".szse.cn", ".sse.com.cn",
    ".cninfo.com.cn", ".csrc.gov.cn", ".csindex.com.cn",
    ".tushare.pro", ".waditu.com", ".akshare.xyz", ".gtimg.cn",
)
_dns_cache: dict = {}
_orig_getaddrinfo = _socket.getaddrinfo

def _resolve_via_public_dns(host: str) -> str:
    """用公共DNS解析域名，返回真实IP"""
    if host in _dns_cache:
        return _dns_cache[host]
    try:
        import dns.resolver
        resolver = dns.resolver.Resolver(configure=False)
        resolver.nameservers = [_PUBLIC_DNS]
        resolver.lifetime = 5
        answers = resolver.resolve(host, "A")
        real_ip = str(answers[0])
        _dns_cache[host] = real_ip
        return real_ip
    except Exception:
        return ""

def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if isinstance(host, str) and any(host.endswith(s) for s in _DOMESTIC_SUFFIXES):
        real_ip = _resolve_via_public_dns(host)
        if real_ip:
            # 返回 IPv4 地址信息，保留端口和其他参数
            return [(_socket.AF_INET, _socket.SOCK_STREAM, 6, '', (real_ip, port))]
    return _orig_getaddrinfo(host, port, family, type, proto, flags)

_socket.getaddrinfo = _patched_getaddrinfo

_socket.getaddrinfo = _patched_getaddrinfo

# ── 导入数据库 ───────────────────────────────────────────────────────

try:
    import akshare as ak
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

try:
    import adata
    _ADATA_AVAILABLE = True
except ImportError:
    _ADATA_AVAILABLE = False

try:
    import baostock as bs
    _BAOSTOCK_AVAILABLE = True
except ImportError:
    _BAOSTOCK_AVAILABLE = False

try:
    from Ashare import get_price as ashare_get_price
    _ASHARE_AVAILABLE = True
except ImportError:
    _ASHARE_AVAILABLE = False

# 6) 恢复 LLM 代理环境变量（供 llm_client.py 的 httpx 使用）
for _k, _v in _SAVED_PROXY.items():
    os.environ[_k] = _v


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


def _is_trading_session() -> bool:
    """
    判断当前是否在A股交易时段（工作日 9:15 - 15:05）。
    用于决定是否启用盘中虚拟K线模式。
    """
    now = datetime.now()
    if now.weekday() >= 5:   # 周六日
        return False
    t = now.hour * 100 + now.minute
    return 915 <= t <= 1505


def _make_virtual_bar(rt: Dict) -> Optional[Dict]:
    """
    从实时行情快照构造今日虚拟日K线bar。
    字段来源: ak.stock_zh_a_spot_em() 返回的行情字典。
    返回 None 表示数据不足（如停牌/未开盘/非交易日）。
    """
    v_open = float(rt.get("今开", 0) or 0)
    v_high = float(rt.get("最高", 0) or 0)
    v_low = float(rt.get("最低", 0) or 0)
    v_close = float(rt.get("最新价", rt.get("close", 0)) or 0)
    v_vol = float(rt.get("成交量", rt.get("volume", 0)) or 0)
    v_amount = float(rt.get("成交额", rt.get("amount", 0)) or 0)
    v_turnover = float(rt.get("换手率", rt.get("turnover", 0)) or 0)

    # 必须有有效的开盘价、现价和成交量（排除停牌/未开盘）
    if v_open <= 0 or v_close <= 0 or v_vol <= 0:
        return None

    return {
        "date": pd.Timestamp(TODAY),
        "open": v_open,
        "high": v_high,
        "low": v_low,
        "close": v_close,
        "volume": v_vol,
        "amount": v_amount,
        "turnover": v_turnover,
    }


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


def compute_signal_score(df: pd.DataFrame) -> Dict:
    """
    量化信号评分（100分制，6因子），基于 compute_indicators() 产出的 DataFrame。
    返回: {"total_score": int, "signal": str, "bias_pct": float, "breakdown": {...}}
    """
    default = {
        "total_score": 0, "signal": "SELL", "bias_pct": 0.0,
        "breakdown": {"trend": 0, "bias": 0, "volume": 0, "support": 0, "macd": 0, "rsi": 0},
    }
    if df is None or len(df) < 5:
        return default

    try:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        ma5 = float(last.get("ma5", 0))
        ma10 = float(last.get("ma10", 0))
        ma20 = float(last.get("ma20", 0))
        ma60 = float(last.get("ma60", 0))
        close = float(last.get("close", 0))
        prev_close = float(prev.get("close", 0))
        volume = float(last.get("volume", 0))
        vol_ma5 = float(last.get("vol_ma5", 1))
        dif = float(last.get("dif", 0))
        dea = float(last.get("dea", 0))
        macd_hist = float(last.get("macd_hist", 0))
        prev_macd_hist = float(prev.get("macd_hist", 0))
        rsi6 = float(last.get("rsi6", 50))

        if close <= 0 or ma5 <= 0:
            return default

        # ── 1. Trend (30pts) ──
        if ma5 > ma10 > ma20:
            trend_score = 30  # STRONG_BULL
            trend_label = "STRONG_BULL"
        elif ma5 > ma10:
            trend_score = 26  # BULL
            trend_label = "BULL"
        elif ma5 > ma20:
            trend_score = 18  # WEAK_BULL
            trend_label = "WEAK_BULL"
        else:
            trend_score = 12
            trend_label = "BEAR"

        # ── 2. Bias rate (20pts) ──
        bias = (close - ma5) / ma5 * 100
        if bias < -3:
            bias_score = 20
        elif -1 <= bias <= 2:
            bias_score = 18
        elif 2 < bias <= 5:
            bias_score = 14
        else:
            bias_score = 4

        # ── 3. Volume (15pts) ──
        vol_ratio = volume / max(vol_ma5, 1e-9)
        if close < prev_close and vol_ratio < 0.8:
            vol_score = 15  # shrink pullback
        elif close > prev_close and vol_ratio > 1.2:
            vol_score = 12
        elif 0.8 <= vol_ratio <= 1.2:
            vol_score = 10
        elif close > prev_close and vol_ratio < 0.8:
            vol_score = 6
        else:
            vol_score = 0

        # ── 4. Support (10pts) ──
        support_score = 0
        if close >= ma5 >= close * 0.98:
            support_score += 5
        if close >= ma10 >= close * 0.97:
            support_score += 5

        # ── 5. MACD (15pts) ──
        if dif > 0 and dea > 0 and macd_hist > 0:
            macd_score = 15
        elif macd_hist > 0 and prev_macd_hist <= 0:
            macd_score = 12
        elif macd_hist > 0:
            macd_score = 8
        else:
            macd_score = 3

        # ── 6. RSI (10pts) ──
        if rsi6 < 30:
            rsi_score = 10
        elif 60 <= rsi6 < 70:
            rsi_score = 8
        elif 40 <= rsi6 < 60:
            rsi_score = 5
        elif 30 <= rsi6 < 40:
            rsi_score = 3
        else:
            rsi_score = 0

        total = trend_score + bias_score + vol_score + support_score + macd_score + rsi_score

        # ── Signal mapping ──
        multi_head = ma5 > ma10 > ma20 > ma60
        bull = trend_label in ("STRONG_BULL", "BULL")
        if total >= 75 and multi_head:
            signal = "STRONG_BUY"
        elif total >= 60 and bull:
            signal = "BUY"
        elif total >= 45:
            signal = "HOLD"
        elif total >= 30:
            signal = "WAIT"
        else:
            signal = "SELL"

        return {
            "total_score": total,
            "signal": signal,
            "bias_pct": round(bias, 2),
            "breakdown": {
                "trend": trend_score,
                "bias": bias_score,
                "volume": vol_score,
                "support": support_score,
                "macd": macd_score,
                "rsi": rsi_score,
            },
        }
    except Exception:
        return default


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
    #  数据源自动降级（连接异常兜底）                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _call_with_fallback(primary_fn, fallback_fn=None, label=""):
        """
        尝试 primary_fn()，遇连接类异常时降级到 fallback_fn()。
        两者都失败则返回 None。
        """
        from http.client import RemoteDisconnected
        from requests.exceptions import ReadTimeout
        _conn_errors = (RemoteDisconnected, ConnectionError, ConnectionAbortedError,
                        ConnectionResetError, ReadTimeout, OSError,
                        IndexError, KeyError, ValueError)
        try:
            return primary_fn()
        except _conn_errors as e:
            if label:
                print(f"  [降级] {label} 连接异常: {type(e).__name__}, ", end="")
            if fallback_fn is not None:
                try:
                    if label:
                        print("尝试备用数据源...", flush=True)
                    return fallback_fn()
                except Exception as e2:
                    if label:
                        print(f"备用也失败: {e2}")
                    return None
            else:
                if label:
                    print("跳过", flush=True)
                return None

    # ------------------------------------------------------------------ #
    #  板块列表降级数据源                                                   #
    # ------------------------------------------------------------------ #

    def _tushare_industry_list(self) -> Optional[pd.DataFrame]:
        """tushare 获取申万行业板块列表（降级数据源）"""
        if not self._pro:
            return None
        df = self._pro.index_basic(market="SW")
        if df is None or df.empty:
            return None
        # 取一级行业（与东财申万行业对齐）
        df1 = df[df["category"].str.contains("一级", na=False)].copy()
        if df1.empty:
            df1 = df.copy()
        # 统一列名，与 akshare stock_board_industry_name_em() 对齐
        df1 = df1.rename(columns={"name": "板块名称", "ts_code": "板块代码"})
        # 去掉 "(申万)" 后缀
        df1["板块名称"] = df1["板块名称"].str.replace(r"\(申万\)|（申万）", "", regex=True).str.strip()
        return df1

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

        # 申万行业板块行情（akshare → tushare → adata 三级降级）
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_board_industry_name_em(),
                fallback_fn=lambda: self._tushare_industry_list(),
                label="申万行业")
            if df is not None and len(df) > 0:
                data["sw_industry"] = df.to_dict("records")
                print(f"  申万行业: {len(df)} 个板块")
        except Exception as e:
            print(f"  [警告] 申万行业获取失败: {e}")

        # 概念板块行情
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_board_concept_name_em(), label="概念板块")
            if df is not None and len(df) > 0:
                data["concept"] = df.to_dict("records")
                print(f"  概念板块: {len(df)} 个板块")
        except Exception as e:
            print(f"  [警告] 概念板块获取失败: {e}")

        # 行业资金流向
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流"),
                fallback_fn=lambda: ak.stock_fund_flow_industry(symbol="即时"),
                label="行业资金流向")
            if df is not None and len(df) > 0:
                data["industry_fund_flow"] = df.to_dict("records")
                print(f"  行业资金流向: {len(df)} 条")
        except Exception as e:
            print(f"  [警告] 行业资金流向失败: {e}")

        # 概念资金流向
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流"),
                label="概念资金流向")
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
        """
        获取指定板块的成分股列表（5级降级链）：
        Level 1: akshare 行业板块
        Level 2: akshare 概念板块
        Level 3: adata 同花顺概念 / 东财概念
        Level 4: tushare 申万行业（精确 + 概念映射）
        Level 5: 东财HTTP直调（绕过库）
        """
        print(f"[数据] 获取板块成分股: {sector_names}")
        components: Dict[str, List[Dict]] = {}

        # 概念板块→申万行业的模糊映射
        _CONCEPT_TO_SW = {
            "智能电网": "电力设备", "电网设备": "电力设备",
            "中字头": "建筑装饰", "国资云": "计算机",
            "人工智能": "计算机", "机器人": "专用设备",
            "新能源": "电力设备", "新能源汽车": "汽车",
            "光伏": "光伏设备", "储能": "电力设备",
            "芯片": "半导体", "消费电子": "消费电子",
            "军工": "国防军工", "医药": "医药生物",
            "创新药": "医药生物", "白酒": "食品饮料",
            "食品饮料": "食品饮料", "房地产": "房地产",
            "数字经济": "计算机", "网络安全": "计算机",
            "工业母机": "通用设备", "锂电池": "电力设备",
            "钠电池": "电力设备", "算力": "通信", "CPO": "通信",
        }

        for name in sector_names:
            stocks = []

            # ── Level 1: akshare 行业板块 ──
            df = self._call_with_fallback(
                lambda n=name: ak.stock_board_industry_cons_em(symbol=n),
                label=f"{name}行业成分股",
            )
            if df is not None and len(df) > 0:
                stocks = df.to_dict("records")
                print(f"  {name}: {len(stocks)} 只成分股 [akshare 行业]")

            # ── Level 2: akshare 概念板块 ──
            if not stocks:
                df = self._call_with_fallback(
                    lambda n=name: ak.stock_board_concept_cons_em(symbol=n),
                    label=f"{name}概念成分股",
                )
                if df is not None and len(df) > 0:
                    stocks = df.to_dict("records")
                    print(f"  {name}: {len(stocks)} 只成分股 [akshare 概念]")

            # ── Level 3: adata 同花顺/东财概念 ──
            if not stocks and _ADATA_AVAILABLE:
                for src, fn in [
                    ("adata THS", lambda n=name: adata.stock.info.concept_constituent_ths(concept_name=n)),
                    ("adata East", lambda n=name: adata.stock.info.concept_constituent_east(concept_name=n)),
                ]:
                    try:
                        df = fn()
                        if df is not None and len(df) > 0:
                            stocks = df.to_dict("records")
                            print(f"  {name}: {len(stocks)} 只成分股 [{src}]")
                            break
                    except Exception as e:
                        pass

            # ── Level 4: tushare 申万（精确 + 概念映射） ──
            if not stocks:
                stocks = self._tushare_sector_stocks(name)
                if stocks:
                    print(f"  {name}: {len(stocks)} 只成分股 [tushare SW]")

            if not stocks and name in _CONCEPT_TO_SW:
                sw_name = _CONCEPT_TO_SW[name]
                stocks = self._tushare_sector_stocks(sw_name)
                if stocks:
                    print(f"  {name}: {len(stocks)} 只成分股 [tushare SW 映射→{sw_name}]")

            # ── Level 5: 东财HTTP直调（绕过所有库） ──
            if not stocks:
                stocks = self._eastmoney_concept_stocks_direct(name)
                if stocks:
                    print(f"  {name}: {len(stocks)} 只成分股 [东财HTTP直调]")

            if stocks:
                components[name] = stocks
            else:
                print(f"  [警告] {name}: 未获取到成分股（全部数据源失败）")

        return components

    def _eastmoney_concept_stocks_direct(self, concept_name: str) -> List[Dict]:
        """
        东财HTTP直调：绕过akshare/adata，直接请求东方财富API获取概念板块成分股。
        先查板块代码，再查成分股。
        使用多个东财子域名做 fallback（push2 经常不可用）。
        """
        import requests as _req

        # 可用的东财 API 子域名（按优先级）
        api_hosts = [
            "https://push2.eastmoney.com/api/qt/clist/get",
            "https://datacenter-web.eastmoney.com/api/qt/clist/get",
        ]

        def _try_get(params: dict) -> Optional[dict]:
            """尝试多个子域名请求"""
            for url in api_hosts:
                try:
                    with _req.Session() as s:
                        s.trust_env = False
                        resp = s.get(url, params=params, timeout=10)
                        data = resp.json()
                        if data and data.get("data"):
                            return data
                except Exception:
                    continue
            return None

        try:
            # Step 1: 获取概念板块列表，找到匹配的板块代码
            params_boards = {
                "pn": 1, "pz": 500, "po": 1, "np": 1,
                "fltt": 2, "invt": 2,
                "fs": "m:90+t:3",  # 概念板块
                "fields": "f2,f3,f12,f14",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            }
            data = _try_get(params_boards)

            board_code = None
            if data:
                for item in (data.get("data", {}) or {}).get("diff", []):
                    if item.get("f14", "") == concept_name:
                        board_code = item.get("f12", "")
                        break

                # 也尝试模糊匹配
                if not board_code:
                    for item in (data.get("data", {}) or {}).get("diff", []):
                        if concept_name in item.get("f14", ""):
                            board_code = item.get("f12", "")
                            break

            if not board_code:
                # 再试行业板块 m:90+t:2
                params_boards["fs"] = "m:90+t:2"
                data = _try_get(params_boards)
                if data:
                    for item in (data.get("data", {}) or {}).get("diff", []):
                        if item.get("f14", "") == concept_name or concept_name in item.get("f14", ""):
                            board_code = item.get("f12", "")
                            break

            if not board_code:
                return []

            # Step 2: 获取板块成分股
            params_stocks = {
                "pn": 1, "pz": 500, "po": 1, "np": 1,
                "fltt": 2, "invt": 2,
                "fs": f"b:{board_code}",
                "fields": "f12,f14,f2,f3,f6,f7,f15,f16,f17,f18",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            }
            data = _try_get(params_stocks)

            stocks = []
            if data:
                for item in (data.get("data", {}) or {}).get("diff", []):
                    code = item.get("f12", "")
                    name = item.get("f14", "")
                    if code and len(code) == 6:
                        stocks.append({"代码": code, "名称": name, "code": code, "name": name})
            return stocks

        except Exception:
            return []

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
        数据源优先级: TDX本地（1）→ akshare（2）→ tushare（3）→ Ashare（3.5）→ 腾讯K线（3.75）→ adata（4）→ baostock（5）
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

        # ── Priority 3.5: Ashare（新浪+腾讯双源）───────────────────
        if _ASHARE_AVAILABLE:
            try:
                freq_map_ashare = {"daily": "1d", "weekly": "1w", "monthly": "1M"}
                ashare_freq = freq_map_ashare.get(period, "1d")
                # 上海: 6开头; 深圳: 0/3开头
                ashare_code = f"sh{code}" if code.startswith("6") else f"sz{code}"
                df = ashare_get_price(ashare_code, count=n_bars, frequency=ashare_freq)
                if df is not None and len(df) > 0:
                    df = df.reset_index()
                    df.columns = [c.lower() if c != '' else 'date' for c in df.columns]
                    # Ashare返回的index是日期，reset后第一列为日期
                    first_col = df.columns[0]
                    if first_col != 'date':
                        df = df.rename(columns={first_col: 'date'})
                    df['date'] = pd.to_datetime(df['date'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(subset=['close'])
                    df = df.sort_values('date').reset_index(drop=True)
                    return df.tail(n_bars)
            except Exception as e:
                print(f"  [Ashare] {code} {period} K线失败: {e}")

        # ── Priority 3.75: 腾讯K线（push2.eastmoney.com不可用时的备用）──
        if period == "daily":
            df = self._tencent_kline(code, period, n_bars)
            if df is not None and len(df) >= min_bars:
                return df

        # ── Priority 4: adata ──────────────────────────────────────
        if _ADATA_AVAILABLE and period == "daily":
            try:
                df = adata.stock.market.get_market(stock_code=code, k_type=1, start_date=start)
                if df is not None and len(df) > 0:
                    df = self._normalize_adata_kline(df)
                    return df.tail(n_bars)
            except Exception:
                pass

        # ── Priority 5: baostock ───────────────────────────────────
        if _BAOSTOCK_AVAILABLE and period == "daily":
            try:
                bs.login()
                bs_code = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,volume,amount",
                    start_date=start[:4] + "-" + start[4:6] + "-" + start[6:],
                    end_date=TODAY[:4] + "-" + TODAY[4:6] + "-" + TODAY[6:],
                    frequency="d", adjustflag="2",  # 前复权
                )
                rows = []
                while (rs.error_code == '0') and rs.next():
                    rows.append(rs.get_row_data())
                bs.logout()
                if rows:
                    df = pd.DataFrame(rows, columns=rs.fields)
                    df = self._normalize_baostock_kline(df)
                    return df.tail(n_bars)
            except Exception:
                try:
                    bs.logout()
                except Exception:
                    pass

        return None

    @staticmethod
    def _normalize_adata_kline(df: pd.DataFrame) -> pd.DataFrame:
        """统一 adata K 线列名"""
        rename = {
            "trade_date": "date", "trade_time": "date",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _normalize_baostock_kline(df: pd.DataFrame) -> pd.DataFrame:
        """统一 baostock K 线列名"""
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

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
    #  腾讯/新浪 fallback 数据源（push2.eastmoney.com 不可用时）            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _code_with_market(code: str) -> str:
        """将纯数字代码转为带市场前缀: sh600519, sz000001"""
        if code.startswith("6") or code.startswith("9"):
            return f"sh{code}"
        return f"sz{code}"

    def _tencent_realtime_quotes(self, codes: List[str]) -> Dict[str, Dict]:
        """
        腾讯实时行情（替代 ak.stock_zh_a_spot_em）。
        Endpoint: https://qt.gtimg.cn/q={code_list}
        """
        import requests as _req
        result: Dict[str, Dict] = {}
        if not codes:
            return result

        # 批量请求，每批最多50个
        market_codes = [self._code_with_market(c) for c in codes]
        for i in range(0, len(market_codes), 50):
            batch = market_codes[i:i + 50]
            code_list = ",".join(batch)
            try:
                resp = _req.get(f"https://qt.gtimg.cn/q={code_list}", timeout=8)
                resp.encoding = "gbk"
                text = resp.text
                for line in text.strip().split(";"):
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    # v_sh600519="1~贵州茅台~600519~1850.00~..."
                    try:
                        var_part, val_part = line.split("=", 1)
                        val_part = val_part.strip().strip('"')
                        if not val_part:
                            continue
                        fields = val_part.split("~")
                        if len(fields) < 47:
                            continue
                        code = fields[2]
                        # 腾讯行情字段索引（经验证）:
                        # [1]名称 [2]代码 [3]现价 [4]昨收 [5]今开
                        # [6]买入价(非最高) [7]卖出价(非最低)
                        # [33]最高 [34]最低  (注: 部分股票这两个位置为空)
                        # [36]成交量(手) [37]成交额(万) [38]换手率
                        # [39]PE(动态) [44]流通市值(亿) [45]总市值(亿) [46]PB
                        # [41]最高(备用) [42]最低(备用)
                        high_val = _safe_float(fields[41]) if len(fields) > 41 and fields[41] else _safe_float(fields[33]) if len(fields) > 33 and fields[33] else 0.0
                        low_val = _safe_float(fields[42]) if len(fields) > 42 and fields[42] else _safe_float(fields[34]) if len(fields) > 34 and fields[34] else 0.0
                        result[code] = {
                            "名称": fields[1],
                            "最新价": _safe_float(fields[3]),
                            "昨收": _safe_float(fields[4]),
                            "今开": _safe_float(fields[5]),
                            "最高": high_val,
                            "最低": low_val,
                            "换手率": _safe_float(fields[38]) if len(fields) > 38 else 0.0,
                            "市盈率-动态": _safe_float(fields[39]) if len(fields) > 39 else 0.0,
                            "振幅": _safe_float(fields[43]) if len(fields) > 43 else 0.0,
                            "总市值": _safe_float(fields[45]) * 1e8 if len(fields) > 45 and fields[45] else 0.0,
                            "流通市值": _safe_float(fields[44]) * 1e8 if len(fields) > 44 and fields[44] else 0.0,
                            "市净率": _safe_float(fields[46]) if len(fields) > 46 else 0.0,
                            "成交量": _safe_float(fields[36]) if len(fields) > 36 else 0.0,
                            "成交额": _safe_float(fields[37]) if len(fields) > 37 else 0.0,
                        }
                        # 计算涨跌幅
                        yesterday = _safe_float(fields[4])
                        current = _safe_float(fields[3])
                        if yesterday > 0:
                            result[code]["涨跌幅"] = round((current - yesterday) / yesterday * 100, 2)
                    except Exception:
                        continue
            except Exception as e:
                print(f"  [腾讯行情] 批次请求失败: {e}")
        return result

    def _sina_realtime_quotes(self, codes: List[str]) -> Dict[str, Dict]:
        """
        新浪实时行情（备用）。
        Endpoint: https://hq.sinajs.cn/list={code_list}
        """
        import requests as _req
        result: Dict[str, Dict] = {}
        if not codes:
            return result

        market_codes = [self._code_with_market(c) for c in codes]
        for i in range(0, len(market_codes), 50):
            batch = market_codes[i:i + 50]
            code_list = ",".join(batch)
            try:
                resp = _req.get(
                    f"https://hq.sinajs.cn/list={code_list}",
                    headers={"Referer": "https://finance.sina.com.cn"},
                    timeout=10,
                )
                resp.encoding = "gbk"
                for line in resp.text.strip().split("\n"):
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    try:
                        # var hq_str_sh600519="贵州茅台,1838.00,..."
                        var_part, val_part = line.split("=", 1)
                        val_part = val_part.strip().strip('"').rstrip(";").strip('"')
                        if not val_part:
                            continue
                        # 从变量名提取代码
                        mc = var_part.split("_")[-1]  # sh600519
                        code = mc[2:]  # 600519
                        fields = val_part.split(",")
                        if len(fields) < 10:
                            continue
                        current = _safe_float(fields[3])
                        yesterday = _safe_float(fields[2])
                        result[code] = {
                            "名称": fields[0],
                            "今开": _safe_float(fields[1]),
                            "昨收": yesterday,
                            "最新价": current,
                            "最高": _safe_float(fields[4]),
                            "最低": _safe_float(fields[5]),
                            "成交量": _safe_float(fields[8]),
                            "成交额": _safe_float(fields[9]),
                        }
                        if yesterday > 0:
                            result[code]["涨跌幅"] = round((current - yesterday) / yesterday * 100, 2)
                    except Exception:
                        continue
            except Exception as e:
                print(f"  [新浪行情] 批次请求失败: {e}")
        return result

    def _tencent_kline(self, code: str, period: str, n_bars: int = 120) -> Optional[pd.DataFrame]:
        """
        腾讯K线数据（替代 akshare 的 stock_zh_a_hist）。
        Endpoint: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
        仅支持日线。
        """
        import requests as _req
        if period != "daily":
            return None
        try:
            market_code = self._code_with_market(code)
            start_date = (datetime.now() - timedelta(days=n_bars * 2)).strftime("%Y-%m-%d")
            url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            params = {
                "param": f"{market_code},day,{start_date},,{n_bars},qfq",
                "_var": "kline_dayqfq",
                "r": str(time.time()),
            }
            resp = _req.get(url, params=params, timeout=10)
            resp.encoding = "utf-8"
            text = resp.text
            # 响应可能是 kline_dayqfq={json} 格式
            if "=" in text:
                text = text.split("=", 1)[1]
            data = json.loads(text)

            kdata = data.get("data", {}).get(market_code, {})
            # 尝试前复权数据，降级到普通日线
            bars = kdata.get("qfqday") or kdata.get("day") or []
            if not bars:
                return None

            rows = []
            for bar in bars:
                # [date, open, close, high, low, volume, ...]
                if len(bar) < 6:
                    continue
                rows.append({
                    "date": bar[0],
                    "open": _safe_float(bar[1]),
                    "close": _safe_float(bar[2]),
                    "high": _safe_float(bar[3]),
                    "low": _safe_float(bar[4]),
                    "volume": _safe_float(bar[5]),
                })

            if not rows:
                return None

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            return df.tail(n_bars) if len(df) >= 5 else None

        except Exception as e:
            print(f"  [腾讯K线] {code} 获取失败: {e}")
            return None

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
        # Priority 1: akshare (东方财富 push2.eastmoney.com)
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
            print(f"  [警告] akshare实时行情获取失败: {e}")

        # Priority 2: 腾讯实时行情（push2.eastmoney.com 不可用时）
        missing = [c for c in codes if c not in result]
        if missing:
            try:
                tencent_data = self._tencent_realtime_quotes(missing)
                if tencent_data:
                    result.update(tencent_data)
                    if len(tencent_data) > 0:
                        print(f"  [腾讯行情] 补充 {len(tencent_data)} 只实时行情")
            except Exception as e:
                print(f"  [警告] 腾讯实时行情获取失败: {e}")

        # Priority 3: 新浪实时行情（最终备用）
        missing = [c for c in codes if c not in result]
        if missing:
            try:
                sina_data = self._sina_realtime_quotes(missing)
                if sina_data:
                    result.update(sina_data)
                    if len(sina_data) > 0:
                        print(f"  [新浪行情] 补充 {len(sina_data)} 只实时行情")
            except Exception as e:
                print(f"  [警告] 新浪实时行情获取失败: {e}")

        # Priority 4: tushare daily_basic 补充/修正 PE/PB/市值
        # tushare 的 PE(TTM)/PB 数据最权威，用于修正腾讯/新浪可能的解析偏差
        if self._pro and result:
            try:
                need_fix = [c for c in result if not result[c].get("市盈率-动态") or result[c].get("市盈率-动态", 0) > 2000]
                if need_fix:
                    ts_codes = []
                    for c in need_fix:
                        ts_codes.append(f"{c}.SH" if c.startswith("6") else f"{c}.SZ")
                    # 批量获取
                    with self._api_sem:
                        df = self._pro.daily_basic(
                            trade_date="",
                            ts_code=",".join(ts_codes[:50]),
                            fields="ts_code,pe_ttm,pb,total_mv,circ_mv,turnover_rate",
                        )
                    if df is not None and len(df) > 0:
                        fixed = 0
                        for _, row in df.iterrows():
                            ts_code = row.get("ts_code", "")
                            code = ts_code.split(".")[0] if "." in ts_code else ""
                            if code in result:
                                pe = row.get("pe_ttm")
                                pb = row.get("pb")
                                if pe and not pd.isna(pe):
                                    result[code]["市盈率-动态"] = round(float(pe), 2)
                                    result[code]["PE"] = round(float(pe), 2)
                                    fixed += 1
                                if pb and not pd.isna(pb):
                                    result[code]["市净率"] = round(float(pb), 2)
                                    result[code]["PB"] = round(float(pb), 2)
                                tmv = row.get("total_mv")
                                if tmv and not pd.isna(tmv):
                                    result[code]["总市值"] = float(tmv) * 10000  # 万→元
                        if fixed:
                            print(f"  [tushare] 修正 {fixed} 只PE/PB数据")
            except Exception as e:
                pass  # tushare补充失败不影响主流程

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
            df = self._call_with_fallback(
                lambda: ak.stock_zt_pool_em(date=TODAY), label="涨停股池")
            if df is not None and len(df) > 0:
                data["zt_pool"] = df.to_dict("records")
                print(f"  涨停股: {len(df)} 只")
        except Exception as e:
            print(f"  [警告] 涨停股池: {e}")

        # 热门股票
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_hot_rank_em(), label="热门股票")
            if df is not None and len(df) > 0:
                data["hot_rank"] = df.head(50).to_dict("records")
                print(f"  热门股票: Top50")
        except Exception as e:
            print(f"  [警告] 热门股票: {e}")

        # 龙虎榜（近5日）
        try:
            start5 = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")
            df = self._call_with_fallback(
                lambda: ak.stock_lhb_detail_em(start_date=start5, end_date=TODAY), label="龙虎榜")
            if df is not None and len(df) > 0:
                data["lhb"] = df.to_dict("records")
                print(f"  龙虎榜: {len(df)} 条")
        except Exception as e:
            print(f"  [警告] 龙虎榜: {e}")

        # 北向资金（akshare接口已失效，优先用adata）
        try:
            north_ok = False
            # Level 1: adata 北向资金
            if _ADATA_AVAILABLE and not north_ok:
                try:
                    df = adata.sentiment.north.north_flow()
                    if df is not None and len(df) > 0:
                        data["north_flow"] = df.tail(20).to_dict("records")
                        print(f"  北向资金: 近20日 [adata]")
                        north_ok = True
                except Exception:
                    pass
            # Level 2: akshare（可能已失效）
            if not north_ok:
                df = self._call_with_fallback(
                    lambda: ak.stock_hsgt_north_net_flow_in_em(symbol="北上"), label="北向资金")
                if df is not None and len(df) > 0:
                    data["north_flow"] = df.tail(20).to_dict("records")
                    print(f"  北向资金: 近20日 [akshare]")
                    north_ok = True
            if not north_ok:
                print(f"  [警告] 北向资金: 全部数据源失败")
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
    #  大盘指数行情                                                        #
    # ------------------------------------------------------------------ #

    def fetch_market_indices(self) -> Dict:
        """
        获取A股大盘三大指数日K行情（上证/深证/创业板）及趋势判断。
        返回各指数的当前价、涨跌幅、均线状态、趋势判断、量能比。
        """
        intraday = _is_trading_session()
        if intraday:
            print("[数据] 采集大盘指数行情（盘中模式：含实时数据）...")
        else:
            print("[数据] 采集大盘指数行情...")

        indices_config = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
        }
        data: Dict = {}

        # 盘中模式：获取指数实时行情（用于构造今日虚拟K线）
        index_realtime: Dict[str, Dict] = {}
        if intraday:
            try:
                spot_df = ak.stock_zh_index_spot()
                if spot_df is not None and len(spot_df) > 0:
                    for _, row in spot_df.iterrows():
                        code = str(row.get("代码", row.get("symbol", "")))
                        idx_rt = row.to_dict()
                        index_realtime[code] = idx_rt
            except Exception:
                pass
            # 备用：东财接口
            if not index_realtime:
                try:
                    spot_df = ak.stock_zh_index_spot_em()
                    if spot_df is not None and len(spot_df) > 0:
                        for _, row in spot_df.iterrows():
                            code = str(row.get("代码", row.get("symbol", "")))
                            idx_rt = row.to_dict()
                            index_realtime[code] = idx_rt
                except Exception:
                    pass

        for symbol, name in indices_config.items():
            try:
                start = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
                df = ak.stock_zh_index_daily(symbol=symbol)
                if df is None or len(df) < 10:
                    continue
                # 统一列名
                col_map = {"date": "date", "open": "open", "high": "high",
                           "low": "low", "close": "close", "volume": "volume"}
                for old_col in list(df.columns):
                    lc = str(old_col).lower()
                    if lc in col_map:
                        df = df.rename(columns={old_col: col_map[lc]})
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                df = df.dropna(subset=["close"])

                # 盘中：追加今日虚拟K线
                if intraday:
                    last_date_str = str(df["date"].iloc[-1])[:10].replace("-", "")
                    if last_date_str != TODAY:
                        # 尝试从实时行情获取今日数据
                        pure_code = symbol.replace("sh", "").replace("sz", "")
                        idx_rt = index_realtime.get(pure_code, index_realtime.get(symbol, {}))
                        v_open = float(idx_rt.get("今开", idx_rt.get("open", 0)) or 0)
                        v_close = float(idx_rt.get("最新价", idx_rt.get("close", idx_rt.get("current", 0))) or 0)
                        v_high = float(idx_rt.get("最高", idx_rt.get("high", 0)) or 0)
                        v_low = float(idx_rt.get("最低", idx_rt.get("low", 0)) or 0)
                        v_vol = float(idx_rt.get("成交量", idx_rt.get("volume", 0)) or 0)
                        if v_open > 0 and v_close > 0:
                            vbar = pd.DataFrame([{
                                "date": pd.Timestamp(TODAY),
                                "open": v_open, "high": v_high,
                                "low": v_low, "close": v_close,
                                "volume": v_vol,
                            }])
                            df = pd.concat([df, vbar], ignore_index=True)

                df = df.tail(60)
                close = df["close"]
                current = float(close.iloc[-1])

                # 涨跌幅
                changes = {}
                for n, label in [(1, "1日"), (5, "5日"), (20, "20日")]:
                    if len(close) > n:
                        prev = float(close.iloc[-(n + 1)])
                        if prev > 0:
                            changes[label] = round((current - prev) / prev * 100, 2)

                # 均线
                ma5 = round(float(close.tail(5).mean()), 2)
                ma10 = round(float(close.tail(10).mean()), 2)
                ma20 = round(float(close.tail(20).mean()), 2)
                ma60 = round(float(close.mean()), 2)

                if current > ma5 > ma10 > ma20:
                    trend = "多头排列"
                elif current < ma5 < ma10 < ma20:
                    trend = "空头排列"
                else:
                    trend = "震荡整理"

                # 量能比
                vol = df["volume"].astype(float)
                vol_5 = float(vol.tail(5).mean())
                vol_20 = float(vol.tail(20).mean())
                vol_ratio = round(vol_5 / max(vol_20, 1), 2)

                data[name] = {
                    "code": symbol,
                    "current": current,
                    "changes": changes,
                    "ma": {"MA5": ma5, "MA10": ma10, "MA20": ma20, "MA60": ma60},
                    "trend": trend,
                    "vol_ratio_5d_vs_20d": vol_ratio,
                }
                chg_1d = changes.get("1日", "N/A")
                chg_5d = changes.get("5日", "N/A")
                print(f"  {name}: {current} {trend} 1日={chg_1d}% 5日={chg_5d}% 量比={vol_ratio}")
            except Exception as e:
                print(f"  [警告] {name}获取失败: {e}")

        _save_json(data, f"market_indices_{TODAY}.json")
        return data

    # ------------------------------------------------------------------ #
    #  市场情绪指标                                                        #
    # ------------------------------------------------------------------ #

    def fetch_market_sentiment(self) -> Dict:
        """
        获取市场情绪面指标：连板股高度、炸板率、昨涨停今日表现、盘口异动。
        用于判断市场情绪周期（冰点→修复→升温→亢奋→过热→退潮）。
        """
        print("[数据] 采集市场情绪指标...")
        data: Dict = {}

        # 强势连板股（连板高度 = 题材空间天花板）
        try:
            df = ak.stock_zt_pool_strong_em(date=TODAY)
            if df is not None and len(df) > 0:
                data["strong_zt"] = df.to_dict("records")
                board_col = [c for c in df.columns if "连板" in str(c)]
                if board_col:
                    df[board_col[0]] = pd.to_numeric(df[board_col[0]], errors="coerce")
                    max_board = int(df[board_col[0]].max())
                    board_dist = df[board_col[0]].value_counts().sort_index(ascending=False)
                    data["max_consecutive_zt"] = max_board
                    data["board_distribution"] = {str(int(k)): int(v) for k, v in board_dist.items()}
                    print(f"  连板股: {len(df)}只, 最高{max_board}板")
                else:
                    print(f"  连板股: {len(df)}只")
        except Exception as e:
            print(f"  [警告] 强势连板: {e}")

        # 炸板股（情绪温度计：炸板率高=情绪差）
        try:
            df = ak.stock_zt_pool_zbgc_em(date=TODAY)
            if df is not None and len(df) > 0:
                data["failed_zt"] = df.to_dict("records")
                data["failed_zt_count"] = len(df)
                print(f"  炸板股: {len(df)}只")
        except Exception as e:
            print(f"  [警告] 炸板股: {e}")

        # 昨日涨停今日表现（接力情绪/溢价率）
        try:
            df = ak.stock_zt_pool_previous_em(date=TODAY)
            if df is not None and len(df) > 0:
                data["prev_zt_today"] = df.to_dict("records")
                data["prev_zt_count"] = len(df)
                print(f"  昨日涨停: {len(df)}只")
        except Exception as e:
            print(f"  [警告] 昨日涨停: {e}")

        # 炸板率计算
        zt_count = len(data.get("strong_zt", []))
        failed_count = data.get("failed_zt_count", 0)
        if zt_count + failed_count > 0:
            data["zt_success_rate"] = round(zt_count / (zt_count + failed_count) * 100, 1)
            print(f"  封板成功率: {data['zt_success_rate']}%")

        _save_json(data, f"market_sentiment_{TODAY}.json")
        return data

    # ------------------------------------------------------------------ #
    #  概念炒作信号                                                        #
    # ------------------------------------------------------------------ #

    def fetch_concept_hype_signals(self) -> Dict:
        """
        概念炒作信号检测：
        1. 概念板块涨幅异动（涨幅>3%的概念）
        2. 概念/行业板块涨幅排名Top30
        3. 概念/行业5日资金流向排名
        """
        print("[数据] 采集概念炒作信号...")
        data: Dict = {}

        # 概念板块涨幅排名（全量，用于异动检测）
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_board_concept_name_em(), label="概念涨幅排名")
            if df is not None and len(df) > 0:
                if "涨跌幅" in df.columns:
                    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
                    hot = df[df["涨跌幅"] > 3.0]
                    data["hot_concepts"] = hot.to_dict("records") if len(hot) > 0 else []
                    data["concept_ranking"] = df.nlargest(30, "涨跌幅").to_dict("records")
                    print(f"  概念异动(>3%): {len(data['hot_concepts'])}个")
        except Exception as e:
            print(f"  [警告] 概念涨幅排名: {e}")

        # 行业板块涨幅排名
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_board_industry_name_em(), label="行业涨幅排名")
            if df is not None and len(df) > 0:
                if "涨跌幅" in df.columns:
                    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
                    data["industry_ranking"] = df.nlargest(30, "涨跌幅").to_dict("records")
        except Exception as e:
            print(f"  [警告] 行业涨幅排名: {e}")

        # 概念板块5日资金流向
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_sector_fund_flow_rank(indicator="5日", sector_type="概念资金流"),
                label="概念5日资金流")
            if df is not None and len(df) > 0:
                data["concept_fund_flow_5d"] = df.head(20).to_dict("records")
                print(f"  概念5日资金流Top20已采集")
        except Exception as e:
            print(f"  [警告] 概念5日资金流: {e}")

        # 行业板块5日资金流向
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_sector_fund_flow_rank(indicator="5日", sector_type="行业资金流"),
                label="行业5日资金流")
            if df is not None and len(df) > 0:
                data["industry_fund_flow_5d"] = df.head(20).to_dict("records")
                print(f"  行业5日资金流Top20已采集")
        except Exception as e:
            print(f"  [警告] 行业5日资金流: {e}")

        _save_json(data, f"concept_hype_{TODAY}.json")
        return data

    # ------------------------------------------------------------------ #
    #  ETF 行情与板块匹配                                                  #
    # ------------------------------------------------------------------ #

    # 常见板块→ETF映射表（覆盖主要热门板块）
    _SECTOR_ETF_MAP = {
        "半导体": [("512480", "半导体ETF"), ("159995", "芯片ETF")],
        "芯片": [("159995", "芯片ETF"), ("512480", "半导体ETF")],
        "集成电路": [("159546", "集成电路ETF")],
        "人工智能": [("515070", "人工智能ETF"), ("159819", "AI ETF")],
        "算力": [("159515", "算力ETF")],
        "机器人": [("562500", "机器人ETF"), ("159770", "机器人ETF")],
        "新能源": [("516160", "新能源ETF"), ("159875", "新能源ETF")],
        "光伏": [("515790", "光伏ETF"), ("159857", "光伏ETF")],
        "新能源汽车": [("515030", "新能源车ETF"), ("159806", "新能源车ETF")],
        "汽车": [("516110", "汽车ETF")],
        "动力电池": [("159755", "电池ETF")],
        "储能": [("159566", "储能ETF")],
        "电力设备": [("516390", "电力装备ETF")],
        "电网设备": [("562350", "电力ETF")],
        "医药": [("512010", "医药ETF"), ("159929", "医药ETF")],
        "创新药": [("159992", "创新药ETF")],
        "医疗器械": [("159883", "医疗器械ETF")],
        "白酒": [("512690", "白酒ETF"), ("159870", "白酒ETF")],
        "消费": [("159928", "消费ETF")],
        "食品饮料": [("515170", "食品饮料ETF")],
        "银行": [("512800", "银行ETF"), ("159887", "银行ETF")],
        "证券": [("512880", "证券ETF"), ("512000", "券商ETF")],
        "保险": [("512070", "保险ETF")],
        "房地产": [("159768", "房地产ETF")],
        "军工": [("512660", "军工ETF"), ("512810", "军工ETF")],
        "国防军工": [("512660", "军工ETF")],
        "电力": [("159611", "电力ETF")],
        "信创": [("159839", "信创ETF")],
        "游戏": [("516010", "游戏ETF")],
        "传媒": [("512980", "传媒ETF")],
        "有色金属": [("512400", "有色ETF")],
        "黄金": [("518880", "黄金ETF"), ("159834", "黄金ETF")],
        "煤炭": [("515220", "煤炭ETF")],
        "钢铁": [("515210", "钢铁ETF")],
        "农业": [("159825", "农业ETF")],
        "旅游": [("159766", "旅游ETF")],
        "科技": [("515000", "科技ETF")],
        "5G": [("515050", "5G ETF")],
        "云计算": [("516510", "云计算ETF")],
        "大数据": [("515400", "大数据ETF")],
        "软件": [("515230", "软件ETF")],
        "IT服务": [("515230", "软件ETF")],
        "通信": [("515880", "通信ETF")],
        "物联网": [("159719", "物联网ETF")],
        "区块链": [("516160", "区块链ETF")],
        "网络安全": [("159613", "网络安全ETF")],
        "化工": [("516020", "化工ETF")],
        "建材": [("159745", "建材ETF")],
        "家电": [("159996", "家电ETF")],
        "港股": [("159920", "恒生ETF"), ("513060", "恒生科技ETF")],
    }

    def fetch_etf_hot(self) -> Dict:
        """
        获取热门ETF行情（成交额Top50 + 涨幅Top20），
        并提供板块→ETF映射表供报告使用。
        """
        print("[数据] 采集ETF行情数据...")
        data: Dict = {"etf_list": [], "top_gainers": [], "sector_etf_map": self._SECTOR_ETF_MAP}

        try:
            df = self._call_with_fallback(
                lambda: ak.fund_etf_spot_em(), label="ETF行情")
            if df is not None and len(df) > 0:
                # 按成交额排序 Top50
                if "成交额" in df.columns:
                    df["成交额"] = pd.to_numeric(df["成交额"], errors="coerce")
                    df_sorted = df.dropna(subset=["成交额"]).sort_values("成交额", ascending=False)
                    data["etf_list"] = df_sorted.head(50).to_dict("records")
                    print(f"  ETF成交额Top50已采集")

                # 涨幅Top20
                if "涨跌幅" in df.columns:
                    df["涨跌幅"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
                    top_gainers = df.dropna(subset=["涨跌幅"]).nlargest(20, "涨跌幅")
                    data["top_gainers"] = top_gainers.to_dict("records")
                    print(f"  ETF涨幅Top20已采集")
        except Exception as e:
            print(f"  [警告] ETF行情获取失败: {e}")

        _save_json(data, f"etf_hot_{TODAY}.json")
        return data

    def etf_active_sectors(self, etf_data: Dict) -> List[Dict]:
        """
        从ETF行情数据中反向推导活跃板块信号。
        将涨幅Top20和成交额异动的ETF，通过 _SECTOR_ETF_MAP 反向映射回板块名。

        返回: [{"sector": "半导体", "etf_code": "512480", "etf_name": "半导体ETF",
                "change_pct": 3.5, "amount": 12.3亿, "signal": "涨幅+量能"}, ...]
        """
        # 构建反向映射: ETF代码 → 板块名
        etf_to_sector: Dict[str, str] = {}
        for sector, etfs in self._SECTOR_ETF_MAP.items():
            for code, _ in etfs:
                etf_to_sector[code] = sector

        seen_sectors: Dict[str, Dict] = {}  # 去重，每个板块取最强ETF

        # 从涨幅Top20中提取
        for rec in etf_data.get("top_gainers", []):
            code = str(rec.get("代码", "")).strip()
            if code in etf_to_sector:
                sector = etf_to_sector[code]
                chg = float(rec.get("涨跌幅", 0) or 0)
                amt = float(rec.get("成交额", 0) or 0)
                if sector not in seen_sectors or chg > seen_sectors[sector].get("change_pct", 0):
                    seen_sectors[sector] = {
                        "sector": sector,
                        "etf_code": code,
                        "etf_name": str(rec.get("名称", "")),
                        "change_pct": round(chg, 2),
                        "amount_yi": round(amt / 1e8, 2) if amt > 0 else 0,
                        "signal": "涨幅靠前",
                    }

        # 从成交额Top50中提取量能异动
        for rec in etf_data.get("etf_list", []):
            code = str(rec.get("代码", "")).strip()
            if code in etf_to_sector:
                sector = etf_to_sector[code]
                chg = float(rec.get("涨跌幅", 0) or 0)
                amt = float(rec.get("成交额", 0) or 0)
                if sector in seen_sectors:
                    if amt / 1e8 > 5:  # 成交额>5亿才标记量能
                        seen_sectors[sector]["signal"] += "+量能活跃"
                        seen_sectors[sector]["amount_yi"] = round(amt / 1e8, 2)
                elif amt / 1e8 > 10:  # 不在涨幅榜但成交额>10亿
                    seen_sectors[sector] = {
                        "sector": sector,
                        "etf_code": code,
                        "etf_name": str(rec.get("名称", "")),
                        "change_pct": round(chg, 2),
                        "amount_yi": round(amt / 1e8, 2),
                        "signal": "量能异动",
                    }

        # 按涨幅排序
        result = sorted(seen_sectors.values(), key=lambda x: x["change_pct"], reverse=True)
        return result

    def match_sector_etfs(self, sector_names: List[str]) -> Dict[str, List[Dict]]:
        """
        根据板块名称匹配对应ETF。
        返回: {板块名: [{"code": "512480", "name": "半导体ETF"}, ...]}
        支持模糊匹配（板块名包含映射表key或反向包含）。
        """
        result: Dict[str, List[Dict]] = {}
        for sector in sector_names:
            matched = []
            # 精确匹配
            if sector in self._SECTOR_ETF_MAP:
                matched = [{"code": c, "name": n} for c, n in self._SECTOR_ETF_MAP[sector]]
            else:
                # 模糊匹配：板块名包含映射key，或映射key包含板块名
                for key, etfs in self._SECTOR_ETF_MAP.items():
                    if key in sector or sector in key:
                        matched = [{"code": c, "name": n} for c, n in etfs]
                        break
            if matched:
                result[sector] = matched
        return result

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

    def fetch_chip_distribution(self, code: str, daily_df=None) -> Optional[Dict]:
        """
        获取筹码分布数据。

        Returns: {
            "avg_cost": float,          # 60日成交量加权均价（近似平均成本）
            "profit_ratio": float,      # 获利比例(0-1)
            "concentration": float,     # 筹码集中度(越小越集中)
            "main_net_inflow_5d": float,# 近5日主力净流入(万)
            "super_net_inflow_5d": float,# 近5日超大单净流入(万)
        }
        """
        result = {}

        # 从日线计算筹码指标
        df = daily_df
        if df is not None and len(df) >= 20:
            try:
                recent = df.tail(60)
                close = float(recent["close"].iloc[-1])

                # VWAP as average cost
                if "volume" in recent.columns and "close" in recent.columns:
                    vol = recent["volume"].astype(float)
                    prices = recent["close"].astype(float)
                    total_vol = vol.sum()
                    if total_vol > 0:
                        vwap = (prices * vol).sum() / total_vol
                        result["avg_cost"] = round(float(vwap), 2)

                        # Profit ratio: volume bought below current price / total volume
                        profit_vol = vol[prices <= close].sum()
                        result["profit_ratio"] = round(float(profit_vol / total_vol), 3)

                # Concentration: coefficient of variation of close prices
                if len(recent) >= 10:
                    std = float(recent["close"].std())
                    mean = float(recent["close"].mean())
                    if mean > 0:
                        result["concentration"] = round(std / mean, 4)
            except Exception:
                pass

        # efinance fund flow (主力资金)
        try:
            import efinance as ef
            bill_df = ef.stock.get_history_bill(code)
            if bill_df is not None and len(bill_df) >= 5:
                recent_bill = bill_df.tail(5)
                # Column names may vary, try common ones
                for col_main in ["主力净流入", "主力净额"]:
                    if col_main in recent_bill.columns:
                        result["main_net_inflow_5d"] = round(float(recent_bill[col_main].astype(float).sum()), 2)
                        break
                for col_super in ["超大单净流入", "超大单净额"]:
                    if col_super in recent_bill.columns:
                        result["super_net_inflow_5d"] = round(float(recent_bill[col_super].astype(float).sum()), 2)
                        break
        except Exception:
            pass

        return result if result else None

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

        # 检测盘中模式
        _intraday_mode = _is_trading_session()
        if _intraday_mode:
            print(f"\n[盘中模式] 检测到交易时段，将用实时行情构造今日虚拟K线追加到日线末尾")
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
        _PER_STOCK_TIMEOUT = 60  # 单只股票数据采集超时（秒）

        def _call_with_timeout(fn, *args, timeout_sec=30, default=None):
            """在子线程中执行 fn，超时返回 default（防止底层I/O卡死）"""
            with concurrent.futures.ThreadPoolExecutor(1) as _ex:
                _f = _ex.submit(fn, *args)
                try:
                    return _f.result(timeout=timeout_sec)
                except (concurrent.futures.TimeoutError, Exception):
                    return default

        def _fetch_kline_with_sem(code: str, period: str, n_bars: int):
            """先尝试 TDX（不占信号量），TDX失败再用远程 API（占信号量）"""
            df = _call_with_timeout(self._tdx_kline, code, period, n_bars, timeout_sec=15)
            if df is not None and len(df) > 0:
                return df
            # TDX 未命中，走远程（AKShare → Tushare）
            with _api_sem:
                return _call_with_timeout(self.fetch_kline, code, period, n_bars, timeout_sec=30)

        def _fetch_one(code: str) -> None:
            rt_info = realtime.get(code, {})
            name = rt_info.get("名称", rt_info.get("name", code))
            sector = (sector_map or {}).get(code, "")
            t_s = time.time()

            def _check_timeout(stage: str = "") -> bool:
                """检查单只股票是否超时（60s），超时则跳过"""
                if time.time() - t_s > _PER_STOCK_TIMEOUT:
                    with _cnt_lock:
                        _skip[0] += 1
                        n = _done[0] + _skip[0]
                    print(
                        f"  [{n:3d}/{total}] {code} {name:<6} ✗ 超时{_PER_STOCK_TIMEOUT}s({stage})，跳过",
                        flush=True,
                    )
                    return True
                return False

            pkg: Dict = {"code": code, "sector": sector, "realtime": rt_info}

            # ── 日线（有多少拿多少，新股K线少也保留）───────────────
            df_d = _fetch_kline_with_sem(code, "daily", 120)
            if _check_timeout("日线"):
                return
            if df_d is None or len(df_d) == 0:
                with _cnt_lock:
                    _skip[0] += 1
                    n = _done[0] + _skip[0]
                print(
                    f"  [{n:3d}/{total}] {code} {name:<6} ✗ 无日线数据，跳过",
                    flush=True,
                )
                return

            # ── 盘中模式：追加今日虚拟K线 ────────────────────────
            _vbar_appended = False
            if _intraday_mode and len(df_d) > 0:
                last_date_str = str(df_d["date"].iloc[-1])[:10].replace("-", "")
                if last_date_str != TODAY:
                    vbar = _make_virtual_bar(rt_info)
                    if vbar:
                        vbar_df = pd.DataFrame([vbar])
                        df_d = pd.concat([df_d, vbar_df], ignore_index=True)
                        _vbar_appended = True

            pkg["daily"] = compute_indicators(df_d)
            pkg["daily_bars"] = len(df_d)  # 记录实际K线根数，供画像参考
            pkg["intraday_virtual"] = _vbar_appended  # 标记是否含盘中虚拟K线
            pkg["signal_score"] = compute_signal_score(pkg["daily"])

            # ── K线图表（用于E4视觉分析）────────────────────────────
            _chart_dir = os.path.join(_BASE_DIR, "output", "charts")
            cp = generate_kline_chart(code, df_d, _chart_dir)
            if cp:
                pkg["chart_path"] = cp

            # ── 周线（有多少拿多少）─────────────────────────────────
            df_w = _fetch_kline_with_sem(code, "weekly", 52)
            if _check_timeout("周线"):
                return
            if df_w is not None and len(df_w) > 0:
                pkg["weekly"] = compute_indicators(df_w)

            # ── 月线（有多少拿多少）─────────────────────────────────
            df_m = _fetch_kline_with_sem(code, "monthly", 24)
            if _check_timeout("月线"):
                return
            if df_m is not None and len(df_m) > 0:
                pkg["monthly"] = compute_indicators(df_m)

            # ── 资金流向 + 财务（远程 API，受信号量控速）────────────
            for fetch_name, fetch_fn, pkg_key in [
                ("资金流", self.fetch_fund_flow, "fund_flow"),
                ("财务", self.fetch_financial, "financial"),
                ("盈利预测", self.fetch_profit_forecast, "profit_forecast"),
                ("股东数", self.fetch_holder_num, "holder_num"),
                ("商誉质押", self.fetch_goodwill_pledge, "goodwill_pledge"),
                ("大宗交易", self.fetch_block_trade, "block_trade"),
                ("融资融券", self.fetch_margin_history, "margin_history"),
            ]:
                if _check_timeout(fetch_name):
                    return
                with _api_sem:
                    pkg[pkg_key] = fetch_fn(code)

            # ── 筹码分布（日线计算 + efinance资金流）─────────────────
            chip = self.fetch_chip_distribution(code, pkg.get("daily"))
            if chip:
                pkg["chip"] = chip

            elapsed_s = round(time.time() - t_s, 1)
            with _cnt_lock:
                _done[0] += 1
                n = _done[0] + _skip[0]

            has_w = "W" if "weekly" in pkg else "-"
            has_m = "M" if "monthly" in pkg else "-"
            has_v = "V" if _vbar_appended else "-"
            print(
                f"  [{n:3d}/{total}] {code} {name:<6} {sector:<8}"
                f" {has_w}{has_m}{has_v} ✓ {elapsed_s}s",
                flush=True,
            )
            with _pkgs_lock:
                packages[code] = pkg

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="DataFetch"
        ) as executor:
            futures = [executor.submit(_fetch_one, code) for code in codes]
            # _fetch_one 内部有 _PER_STOCK_TIMEOUT 超时自退出，这里只需等全部返回
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

        # 量化信号评分
        sig = pkg.get("signal_score")
        signal_tag = ""
        if sig and sig.get("total_score", 0) > 0:
            signal_tag = f" | 量化评分={sig['total_score']}分({sig['signal']}) 乖离率(MA5)={sig['bias_pct']:+.1f}%"

        # 筹码分布
        chip = pkg.get("chip", {})
        if chip:
            chip_parts = []
            if "avg_cost" in chip:
                chip_parts.append(f"均成本={chip['avg_cost']}")
            if "profit_ratio" in chip:
                chip_parts.append(f"获利比={chip['profit_ratio']:.0%}")
            if "concentration" in chip:
                chip_parts.append(f"集中度={chip['concentration']:.3f}")
            if "main_net_inflow_5d" in chip:
                chip_parts.append(f"5日主力净流入={chip['main_net_inflow_5d']:.0f}万")
            if chip_parts:
                extra += " | 筹码: " + " ".join(chip_parts)

        # 盘中标记
        intraday_tag = ""
        if pkg.get("intraday_virtual"):
            intraday_tag = " [盘中实时]"

        return (
            f"{code} {name} | 板块:{sector} | "
            f"PE={pe} PB={pb} 市值={cap} 现价={price} 近20日={d20}"
            f"{extra}{signal_tag}{intraday_tag}"
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
    #  新闻采集（事件驱动分析用）                                           #
    # ------------------------------------------------------------------ #

    def fetch_breaking_news(self) -> List[Dict]:
        """
        采集近期财经新闻/快讯，用于事件驱动因果分析。
        Level 1: 东方财富快讯 API
        Level 2: 新浪财经滚动新闻
        返回: [{"title": str, "time": str, "source": str, "summary": str}]
        """
        import requests as _req
        print("[数据] 采集财经新闻快讯...")
        news_list: List[Dict] = []

        # Level 1: 东方财富快讯
        try:
            url = (
                "https://np-listapi.eastmoney.com/comm/wap/getListInfo"
                "?cb=&client=wap&type=0&mession=&pageSize=30&pageIndex=1"
                "&keyword=&sitelibId=Information&source="
            )
            resp = _req.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://wap.eastmoney.com/",
            })
            if resp.status_code == 200:
                text = resp.text.strip()
                # 去掉可能的 JSONP callback 包裹
                if text.startswith("(") and text.endswith(")"):
                    text = text[1:-1]
                data = json.loads(text) if text else {}
                items = data.get("data", {}).get("list", []) if isinstance(data.get("data"), dict) else []
                if not items and isinstance(data.get("data"), list):
                    items = data["data"]
                for item in items[:30]:
                    title = item.get("title", item.get("Title", ""))
                    summary = item.get("digest", item.get("content", item.get("Art_Content", "")))
                    t = item.get("showtime", item.get("time", item.get("Art_ShowTime", "")))
                    if title:
                        news_list.append({
                            "title": title,
                            "time": str(t),
                            "source": "东方财富",
                            "summary": str(summary)[:200] if summary else "",
                        })
                if news_list:
                    print(f"  东方财富快讯: {len(news_list)} 条")
        except Exception as e:
            print(f"  [警告] 东方财富快讯: {e}")

        # Level 2: 新浪财经滚动新闻
        try:
            url = (
                "https://feed.mix.sina.com.cn/api/roll/get"
                "?pageid=153&lid=2516&k=&num=30&page=1&r=0.1"
            )
            resp = _req.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://finance.sina.com.cn/",
            })
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("result", {}).get("data", [])
                existing_titles = {n["title"] for n in news_list}
                count_before = len(news_list)
                for item in items[:30]:
                    title = item.get("title", "")
                    if title and title not in existing_titles:
                        news_list.append({
                            "title": title,
                            "time": item.get("ctime", item.get("intime", "")),
                            "source": "新浪财经",
                            "summary": item.get("intro", item.get("summary", ""))[:200],
                        })
                        existing_titles.add(title)
                added = len(news_list) - count_before
                if added > 0:
                    print(f"  新浪财经: +{added} 条")
        except Exception as e:
            print(f"  [警告] 新浪财经: {e}")

        print(f"  新闻采集完成: 共 {len(news_list)} 条")
        return news_list

    # ------------------------------------------------------------------ #
    #  异动爆发股扫描                                                      #
    # ------------------------------------------------------------------ #

    def scan_breakout_stocks(self) -> List[Dict]:
        """
        扫描异动爆发股（涨停池 + 昨涨停溢价股）。
        返回: [{"code", "name", "trigger", "pct_chg", "close", "turnover", "sector"}]
        """
        print("[数据] 扫描异动爆发股...")
        candidates: Dict[str, Dict] = {}  # code -> info

        # A: 今日涨停池
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_zt_pool_em(date=TODAY), label="涨停池(breakout)")
            if df is not None and len(df) > 0:
                code_col = [c for c in df.columns if "代码" in str(c)]
                name_col = [c for c in df.columns if "名称" in str(c)]
                pct_col = [c for c in df.columns if "涨跌幅" in str(c)]
                price_col = [c for c in df.columns if "最新价" in str(c)]
                turnover_col = [c for c in df.columns if "换手率" in str(c)]
                for _, row in df.iterrows():
                    code = str(row[code_col[0]]) if code_col else ""
                    if not code:
                        continue
                    candidates[code] = {
                        "code": code,
                        "name": str(row[name_col[0]]) if name_col else "",
                        "trigger": "涨停",
                        "pct_chg": float(row[pct_col[0]]) if pct_col else 0.0,
                        "close": float(row[price_col[0]]) if price_col else 0.0,
                        "turnover": str(round(float(row[turnover_col[0]]), 2)) if turnover_col else "N/A",
                        "sector": "",
                    }
                print(f"  涨停池: {len(candidates)} 只")
        except Exception as e:
            print(f"  [警告] 涨停池扫描: {e}")

        # B: 昨涨停今表现（溢价率>3%视为涨停接力）
        try:
            df = self._call_with_fallback(
                lambda: ak.stock_zt_pool_previous_em(date=TODAY), label="昨涨停(breakout)")
            if df is not None and len(df) > 0:
                code_col = [c for c in df.columns if "代码" in str(c)]
                name_col = [c for c in df.columns if "名称" in str(c)]
                pct_col = [c for c in df.columns if "涨跌幅" in str(c)]
                price_col = [c for c in df.columns if "最新价" in str(c) or "现价" in str(c)]
                turnover_col = [c for c in df.columns if "换手率" in str(c)]
                added_count = 0
                for _, row in df.iterrows():
                    code = str(row[code_col[0]]) if code_col else ""
                    if not code or code in candidates:
                        continue
                    pct = float(row[pct_col[0]]) if pct_col else 0.0
                    # 溢价>3%算涨停接力
                    if pct > 3.0:
                        candidates[code] = {
                            "code": code,
                            "name": str(row[name_col[0]]) if name_col else "",
                            "trigger": "涨停溢价",
                            "pct_chg": pct,
                            "close": float(row[price_col[0]]) if price_col else 0.0,
                            "turnover": str(round(float(row[turnover_col[0]]), 2)) if turnover_col else "N/A",
                            "sector": "",
                        }
                        added_count += 1
                if added_count:
                    print(f"  涨停溢价: +{added_count} 只")
        except Exception as e:
            print(f"  [警告] 昨涨停扫描: {e}")

        result = list(candidates.values())
        # 按涨跌幅降序，取 Top30
        result.sort(key=lambda x: x.get("pct_chg", 0), reverse=True)
        result = result[:30]
        print(f"  异动扫描完成: 共 {len(result)} 只候选")
        return result

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

    # ------------------------------------------------------------------ #
    #  启动前数据源连通性预检                                               #
    # ------------------------------------------------------------------ #

    def preflight_data_check(self) -> List[Dict]:
        """
        并行测试所有数据源的连通性和延迟。
        测试顺序/优先级: TDX本地 → akshare → tushare → 其他(adata/baostock)

        返回:
        [{"name", "source", "latency", "ok", "error", "detail"}, ...]
        """
        results: List[Dict] = []

        # 计算最近交易日（回退到最近的工作日，避免非交易日空数据误报）
        def _last_trade_date() -> str:
            from datetime import datetime, timedelta
            d = datetime.now()
            # 如果当前时间在15:05之前，用前一个工作日
            if d.hour < 15 or (d.hour == 15 and d.minute < 5):
                d -= timedelta(days=1)
            # 跳过周末
            while d.weekday() >= 5:
                d -= timedelta(days=1)
            return d.strftime("%Y%m%d")

        last_td = _last_trade_date()
        last_td_dash = f"{last_td[:4]}-{last_td[4:6]}-{last_td[6:]}"  # baostock用

        _TEST_TIMEOUT = 15  # 每个测试最多15秒

        def _test(name: str, source: str, fn) -> Dict:
            t = time.time()
            try:
                _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                fut = _ex.submit(fn)
                try:
                    r = fut.result(timeout=_TEST_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    # 不等待底层线程结束，立即返回超时结果
                    _ex.shutdown(wait=False, cancel_futures=True)
                    latency = round(time.time() - t, 2)
                    return {"name": name, "source": source, "latency": latency,
                            "ok": False, "error": f"超时({_TEST_TIMEOUT}s)", "detail": ""}
                finally:
                    _ex.shutdown(wait=False, cancel_futures=True)
                latency = round(time.time() - t, 2)
                ok = r is not None and (not isinstance(r, pd.DataFrame) or not r.empty)
                detail = ""
                if isinstance(r, pd.DataFrame):
                    detail = f"{len(r)}行"
                elif isinstance(r, (list, dict)):
                    detail = f"{len(r)}条" if isinstance(r, list) else f"{len(r)}字段"
                return {"name": name, "source": source, "latency": latency,
                        "ok": ok, "error": "" if ok else "返回空数据", "detail": detail}
            except Exception as e:
                latency = round(time.time() - t, 2)
                return {"name": name, "source": source, "latency": latency,
                        "ok": False, "error": str(e)[:80], "detail": ""}

        # ── 定义测试用例 ──
        tests = []

        # 1) TDX 本地数据
        if self._tdx:
            tests.append(("TDX日线(000001)", "TDX本地",
                          lambda: self._tdx.daily(symbol='000001')))
            tests.append(("TDX日线(600519)", "TDX本地",
                          lambda: self._tdx.daily(symbol='600519')))
        else:
            results.append({"name": "TDX本地数据", "source": "TDX本地",
                            "latency": 0, "ok": False,
                            "error": "未安装mootdx或TDX目录不存在", "detail": ""})

        # 2) akshare 接口
        if ak:
            tests.append(("行业板块列表", "akshare",
                          lambda: ak.stock_board_industry_name_em()))
        # 行业板块列表降级源（独立测试，不管 akshare 是否可用）
        if self._pro:
            tests.append(("行业列表(降级)", "tushare",
                          lambda: self._tushare_industry_list()))
            tests.append(("A股实时行情", "akshare",
                          lambda: ak.stock_zh_a_spot_em()))
            tests.append(("大盘指数快照", "akshare",
                          lambda: ak.stock_zh_index_spot_em()))
            tests.append(("涨停股池", "akshare",
                          lambda: ak.stock_zt_pool_em(date=last_td)))
            tests.append(("个股日K(000001)", "akshare",
                          lambda: ak.stock_zh_a_hist(symbol="000001", period="daily",
                                                     start_date=last_td, end_date=last_td,
                                                     adjust="qfq")))
        else:
            results.append({"name": "akshare", "source": "akshare",
                            "latency": 0, "ok": False,
                            "error": "未安装akshare", "detail": ""})

        # 3) tushare 接口
        if self._pro:
            tests.append(("交易日历", "tushare",
                          lambda: self._pro.trade_cal(exchange='SSE', start_date=TODAY,
                                                      end_date=TODAY)))
            tests.append(("指数日线(000001.SH)", "tushare",
                          lambda: self._pro.index_daily(ts_code='000001.SH',
                                                        start_date=last_td, end_date=last_td)))
        else:
            results.append({"name": "tushare", "source": "tushare",
                            "latency": 0, "ok": False,
                            "error": "未配置token或未安装", "detail": ""})

        # 4) 其他数据源
        if _ADATA_AVAILABLE:
            tests.append(("adata行情", "adata",
                          lambda: adata.stock.market.get_market(stock_code='000001')))
        else:
            results.append({"name": "adata", "source": "adata",
                            "latency": 0, "ok": False,
                            "error": "未安装adata", "detail": ""})

        if _BAOSTOCK_AVAILABLE:
            def _test_baostock():
                lg = bs.login()
                if lg.error_code != '0':
                    raise RuntimeError(f"login失败: {lg.error_msg}")
                rs = bs.query_history_k_data_plus(
                    "sh.000001", "date,close",
                    start_date=last_td_dash, end_date=last_td_dash)
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                bs.logout()
                return data_list or None
            tests.append(("baostock行情", "baostock", _test_baostock))
        else:
            results.append({"name": "baostock", "source": "baostock",
                            "latency": 0, "ok": False,
                            "error": "未安装baostock", "detail": ""})

        # ── 并行执行所有测试 ──
        if tests:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tests), 8)) as executor:
                futures = {
                    executor.submit(_test, name, source, fn): (name, source)
                    for name, source, fn in tests
                }
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

        # ── 按数据源优先级排序：TDX → akshare → tushare → 其他 ──
        source_order = {"TDX本地": 0, "akshare": 1, "tushare": 2, "adata": 3, "baostock": 4}
        results.sort(key=lambda r: (source_order.get(r["source"], 99), r["name"]))

        return results

    # =================================================================== #
    #  L1: 量化预筛层 — 全市场扫描                                         #
    # =================================================================== #

    def L1_quant_filter(self, verbose: bool = True,
                        cutoff_date: str = None,
                        top_pct: float = 0.05) -> List[Dict]:
        """
        L1 量化过滤：全市场扫描，多时间框架形态+量化打分选股。

        v4: 形态识别驱动（日线+周线），看多加分，看空排除。
          数据窗口: 500日（2年+），支持周线形态检测。
          排除层（硬淘汰）：MA5/20/60空头、MACD零轴下顶背离、破前低放量、
                          双顶/头肩顶/VCP看空、三角breakdown、旗型看空
          加分层：双底/头肩底/VCP看多/三角突破/旗型看多/多头排列（周线>日线）
                 + RPS20/60动量 + 技术指标（MA斜率/ATR/near_high/量比）
        """
        if verbose:
            print("\n[数据] L1: 全市场量化扫描...")
            print("  ① 仅使用TDX本地数据（PE/市值过滤在个股日线中检查）...")

        # Step 1: 生成A股代码列表
        # 优先从 stock_list.json 读取真实股票清单（新浪接口获取的公开数据）
        # 若文件不存在则回退到代码范围扫描
        # 加载本地PE/市值缓存（stock_basics.json）
        basics_file = Path(__file__).parent / "output" / "stock_basics.json"
        basics_cache: Dict[str, Dict] = {}
        if basics_file.exists():
            try:
                basics_data = json.loads(basics_file.read_text(encoding="utf-8"))
                basics_cache = basics_data.get("data", {})
                if verbose:
                    print(f"  basics缓存: {len(basics_cache)} 只（日期: {basics_data.get('date', '未知')}）")
            except Exception:
                pass

        stock_list_file = Path(__file__).parent / "output" / "stock_list.json"
        if stock_list_file.exists():
            try:
                sl_data = json.loads(stock_list_file.read_text(encoding="utf-8"))
                all_codes = list(sl_data.get("data", {}).keys())
                if verbose:
                    sl_date = sl_data.get("date", "未知")
                    print(f"  股票清单: {len(all_codes)} 只（来源: stock_list.json, 日期: {sl_date}）")
            except Exception:
                all_codes = None
        else:
            all_codes = None

        if all_codes is None:
            # 回退：按代码范围生成
            all_codes = []
            for code in range(600000, 602000):
                all_codes.append(f"{code:06d}")
            for code in range(688000, 688300):
                all_codes.append(f"{code:06d}")
            for code in range(1, 2000):
                all_codes.append(f"{code:06d}")
            for code in range(300001, 302000):
                all_codes.append(f"{code:06d}")
            if verbose:
                print(f"  代码范围: 约{len(all_codes)}只（沪市主板+科创板+深市主板+创业板，无stock_list.json）")
        elif verbose:
            print(f"  代码范围: 约{len(all_codes)}只")

        # Step 2: 直接进入日线扫描，在个股循环内做所有过滤
        codes = all_codes
        candidate_results = []
        checked = 0
        failed = 0
        eliminated_bear = 0  # 被预筛淘汰的看空形态
        eliminated_pe_mcap = 0  # PE/市值不合格
        eliminated_limitup = 0  # 涨停股
        eliminated_liquidity = 0  # 流动性枯竭
        eliminated空头 = 0  # 均线空头排列
        eliminated无量空跌 = 0  # 无量空跌
        # 市场广度统计（用于牛熊判断）
        _breadth_above_ma20 = 0  # 站上MA20的股票数
        _breadth_total = 0       # 有效统计的股票数
        _breadth_ret20_sum = 0.0 # 20日收益率总和
        _l1_start = time.time()

        for code in codes:
            checked += 1
            if verbose and checked % 100 == 0:
                elapsed = time.time() - _l1_start
                rate = checked / elapsed if elapsed > 0 else 0
                eta = (len(codes) - checked) / rate if rate > 0 else 0
                print(f"\r  [L1] 进度 {checked}/{len(codes)} 只 ({rate:.0f}只/秒, 剩余约{eta:.0f}秒)", end="", flush=True)

            df = self._tdx_daily_only(code, days=500, cutoff_date=cutoff_date)
            if df is None or len(df) < 60:
                failed += 1
                continue

            df = df.tail(500).reset_index(drop=True)

            _c = "close" if "close" in df.columns else "收盘"
            _o = "open" if "open" in df.columns else "开盘"
            _h = "high" if "high" in df.columns else "最高"
            _l = "low" if "low" in df.columns else "最低"
            _v = "volume" if "volume" in df.columns else ("vol" if "vol" in df.columns else "成交量")

            close_arr = df[_c].astype(float).values
            open_arr = df[_o].astype(float).values
            high_arr = df[_h].astype(float).values
            low_arr = df[_l].astype(float).values
            vol_arr = df[_v].astype(float).values

            # ── 市场广度统计（所有有效股票都计入，不论后续是否被淘汰）──
            n = len(close_arr)
            if n >= 20:
                _breadth_total += 1
                _ma20_val = float(np.mean(close_arr[-20:]))
                if close_arr[-1] > _ma20_val:
                    _breadth_above_ma20 += 1
                _r20 = (close_arr[-1] - close_arr[-20]) / close_arr[-20] if close_arr[-20] > 0 else 0
                _breadth_ret20_sum += _r20

            # ── 预筛：淘汰明显崩溃形态（仅排除极端情况，形态判断交给 pattern_detector）──
            if n < 3:
                failed += 1
                continue

            y_close = close_arr[-2]
            d_close = close_arr[-3]

            # ① 连续2日大阴（跌幅>3%）→ 排除（极端崩溃）
            if n >= 3 and d_close > 0 and y_close > 0:
                decline_d1 = (close_arr[-2] - close_arr[-3]) / close_arr[-3]
                decline_d2 = (close_arr[-3] - close_arr[-4]) / close_arr[-4] if n >= 4 and close_arr[-4] > 0 else 0
                if decline_d1 < -0.03 and decline_d2 < -0.03:
                    eliminated_bear += 1
                    continue

            # ── 预筛：PE/市值/换手率/涨停/均线/无量空跌 ───────────────
            # 先计算均线（后续多个过滤器共享）
            ma5_series = self._calc_sma(pd.Series(close_arr), 5)
            ma20_series = self._calc_sma(pd.Series(close_arr), 20)
            ma60_series = self._calc_sma(pd.Series(close_arr), 60) if n >= 60 else None
            ma5 = ma5_series.iloc[-1]
            ma20 = ma20_series.iloc[-1]
            ma60 = ma60_series.iloc[-1] if ma60_series is not None else 0

            # ① 市值 40亿~1500亿（PFH门槛），PE>60 OR PE<10 → 淘汰
            if code in basics_cache:
                b = basics_cache[code]
                pe = b.get("pe")
                mktcap = b.get("mktcap")
                if mktcap is not None and (mktcap < 40 or mktcap > 1500):
                    eliminated_pe_mcap += 1
                    continue
                if pe is not None and (pe > 60 or pe < 10):
                    eliminated_pe_mcap += 1
                    continue

            # ② 今日涨停（沪深10%，科创/创业板20%，ST 5%）→ 淘汰
            # 今日数据（index -1）若为涨停则排除（最后结果不含涨停股）
            today_close = close_arr[-1]
            today_high = high_arr[-1]
            y_close = close_arr[-2]   # 昨日收盘（已验证为已完成）
            if y_close > 0:
                # 判断涨停：收盘≈最高价 且 涨幅≈±10%（普通股）
                gain_today = (today_close - y_close) / y_close
                is_limitup = (abs(today_close - today_high) < 0.01) and (gain_today > 0.099)
                if is_limitup:
                    eliminated_limitup += 1
                    continue

            # ③ 20日平均成交额 < 2.5亿 → 淘汰（PFH流动性门槛）
            # 成交额 ≈ 成交量(手)×收盘价×100 → 亿元
            if len(close_arr) >= 20 and y_close > 0:
                recent_amounts = vol_arr[-20:] * close_arr[-20:] * 100 / 1e8
                avg_amount_20 = float(np.mean(recent_amounts))
                if avg_amount_20 < 2.5:
                    eliminated_liquidity += 1
                    continue
            else:
                avg_amount_20 = 0

            # ④ 均线空头排列 MA5<MA20<MA60 → 淘汰（形态排除层也会检测，这里做快速预筛）
            if ma60 > 0 and ma5 < ma20 < ma60:
                eliminated空头 += 1
                continue

            # ⑤ 无量空跌：近3日（已完成）连续下跌且成交量萎缩
            # 使用 index -2(昨日), -3(前日), -4(大前日) —— 不含今日未完成数据
            if n >= 4:
                c1 = close_arr[-2]   # 昨日收盘
                c2 = close_arr[-3]   # 前日收盘
                c3 = close_arr[-4]   # 大前日收盘
                v1 = vol_arr[-2]     # 昨日成交量
                v2 = vol_arr[-3]     # 前日成交量
                v3 = vol_arr[-4]     # 大前日成交量
                # 连续3天下跌：c1<c2<c3（指数越小越低）
                down_3 = (c1 < c2) and (c2 < c3)
                # 成交量萎缩：v1<v2 and v2<v3
                vol_shrink = (v1 < v2) and (v2 < v3)
                if down_3 and vol_shrink:
                    eliminated无量空跌 += 1
                    continue

            # ── 收集打分特征（不再用 OR 逻辑，全部纳入评分体系）─────
            # 1) 20日收益率（用于 RPS 排名，后续计算）
            ret_20d = (close_arr[-1] - close_arr[-20]) / close_arr[-20] if len(close_arr) >= 20 and close_arr[-20] > 0 else 0.0
            # 1b) 60日收益率（中期动量确认）
            ret_60d = (close_arr[-1] - close_arr[-60]) / close_arr[-60] if len(close_arr) >= 60 and close_arr[-60] > 0 else ret_20d

            # 2) MA20 斜率：5日变化率
            ma20_prev = ma20_series.iloc[-6] if len(ma20_series) >= 6 else ma20_series.iloc[0]
            ma20_slope = (ma20 - ma20_prev) / max(ma20_prev, 1)

            # 3) ATR5 vs ATR20（波动收敛）
            # True Range = max(H-L, |H-prevC|, |L-prevC|)
            hl = high_arr[1:] - low_arr[1:]
            hc = np.abs(high_arr[1:] - close_arr[:-1])
            lc = np.abs(low_arr[1:] - close_arr[:-1])
            tr_tail = np.maximum(hl, np.maximum(hc, lc))
            tr_arr = np.concatenate([[high_arr[0] - low_arr[0]], tr_tail])
            atr5 = float(np.mean(tr_arr[-5:])) if len(tr_arr) >= 5 else float(np.mean(tr_arr))
            atr20 = float(np.mean(tr_arr[-20:])) if len(tr_arr) >= 20 else float(np.mean(tr_arr))

            # 4) near_high：当前价 / 近20日最高价
            high_20 = float(np.max(high_arr[-20:])) if len(high_arr) >= 20 else float(np.max(high_arr))
            near_high = close_arr[-1] / high_20 if high_20 > 0 else 0

            # 5) 放量收阳：当日收阳 + 当日成交量 > 前日
            is_yang = close_arr[-1] > open_arr[-1]
            vol_up = vol_arr[-1] > vol_arr[-2]
            is_yang_fang = is_yang and vol_up

            # 6) 量比：当日成交量 / 20日均量
            avg_vol_20 = float(np.mean(vol_arr[-20:])) if len(vol_arr) >= 20 else float(np.mean(vol_arr))
            amount_ratio = vol_arr[-1] / avg_vol_20 if avg_vol_20 > 0 else 0

            # 7) 放量幅度（保留兼容）
            v1, v2, v3 = vol_arr[-3], vol_arr[-2], vol_arr[-1]
            ma_vol_10 = vol_arr[-10:-3].mean() if len(vol_arr) >= 10 else vol_arr.mean()
            vol_increase = ((v3 - v1) / max(ma_vol_10, 1)) if ma_vol_10 > 0 else 0

            candidate_results.append({
                "code": code,
                "name": code,
                "close": close_arr[-1],
                "MA5": round(ma5, 2),
                "MA20": round(ma20, 2),
                "MA60": round(ma60, 2),
                "放量幅度": round(vol_increase * 100, 1),
                "ret_20d": ret_20d,
                "ret_60d": ret_60d,
                "ma20_slope": ma20_slope,
                "atr5": atr5,
                "atr20": atr20,
                "near_high": round(near_high, 4),
                "is_yang_fang": is_yang_fang,
                "amount_ratio": round(amount_ratio, 2),
                "avg_amount_20": round(avg_amount_20, 2),
                "df": df,
                "_df_raw": df.copy(),  # 保留原始 df（含列名），用于 pattern_detector
            })

        if verbose:
            elapsed = time.time() - _l1_start
            print(f"\n  [L1] 预筛完成：{len(candidate_results)} 只通过硬性门槛，耗时 {elapsed:.1f}秒")
            print(f"       检查 {checked} 只 | 看空淘汰 {eliminated_bear} 只 | 数据不足 {failed} 只")
            print(f"       PE/市值淘汰 {eliminated_pe_mcap} | 涨停淘汰 {eliminated_limitup} | "
                  f"流动性淘汰 {eliminated_liquidity} | 空头排列 {eliminated空头} | 无量空跌 {eliminated无量空跌}")

        if not candidate_results:
            return []

        # ── 牛熊判断：基于市场广度 ────────────────────────────────────
        breadth_ratio = _breadth_above_ma20 / max(_breadth_total, 1)
        avg_ret20 = _breadth_ret20_sum / max(_breadth_total, 1)
        if breadth_ratio > 0.55 and avg_ret20 > 0.01:
            market_regime = "bull"
        elif breadth_ratio < 0.40 or avg_ret20 < -0.02:
            market_regime = "bear"
        else:
            market_regime = "neutral"
        # neutral 按牛市处理（偏进攻）
        is_bull = market_regime in ("bull", "neutral")

        if verbose:
            regime_cn = {"bull": "🐂 牛市", "bear": "🐻 熊市", "neutral": "⚖️ 震荡偏多"}[market_regime]
            print(f"\n  [市场环境] {regime_cn}（广度={breadth_ratio:.1%} 站上MA20, 均20日收益={avg_ret20:.2%}）")

        # ── Phase 2: 计算 RPS 百分位 + 连续化多维打分 ──────────────────
        if verbose:
            print(f"  [L1] Phase 2: 连续化打分-{'牛市模式' if is_bull else '熊市模式'}（{len(candidate_results)} 只）...")

        # RPS20: 按 ret_20d 排名，计算百分位
        ret_values = np.array([c["ret_20d"] for c in candidate_results])
        from scipy.stats import percentileofscore
        rps_pcts = np.array([percentileofscore(ret_values, r, kind='rank') for r in ret_values])
        # RPS60: 按 ret_60d 排名，中期趋势确认
        ret60_values = np.array([c["ret_60d"] for c in candidate_results])
        rps60_pcts = np.array([percentileofscore(ret60_values, r, kind='rank') for r in ret60_values])

        for i, c in enumerate(candidate_results):
            rps_pct = rps_pcts[i]    # 0~100
            rps60_pct = rps60_pcts[i]  # 0~100
            score = 0.0
            atr_r = c["atr5"] / max(c["atr20"], 0.001)

            if is_bull:
                # ══ 牛市评分：二值基础+渐变加成（max≈100）══
                # 1) RPS20 × 0.20 → 最高20分（短期动量）
                score += rps_pct * 0.20
                # 2) RPS60 × 0.10 → 最高10分（中期趋势确认）
                score += rps60_pct * 0.10
                # 3) MA20斜率：基础10分(>0) + 渐变加成5分 → 最高15分
                if c["ma20_slope"] > 0:
                    score += 10 + min(5, c["ma20_slope"] * 200)
                # 4) ATR收敛：基础5分(ratio<1) + 渐变加成5分 → 最高10分
                if atr_r < 1.0:
                    score += 5 + min(5, (1.0 - atr_r) * 15)
                # 5) near_high渐变 → 最高20分（0.6→0分, 1.0→20分）
                score += max(0, min(20, (c["near_high"] - 0.6) / 0.4 * 20))
                # 6) 放量收阳 → 10分
                if c["is_yang_fang"]:
                    score += 10
                # 7) 量比渐变：最优1.0-2.0区间 → 最高10分
                ar = c["amount_ratio"]
                if 1.0 <= ar <= 2.0:
                    score += 10
                elif 0.7 <= ar < 1.0:
                    score += (ar - 0.7) / 0.3 * 10
                elif 2.0 < ar <= 3.0:
                    score += (3.0 - ar) / 1.0 * 10
                # 8) 极致波动率收敛奖励 → 最高5分
                if atr_r < 0.7:
                    score += 5
            else:
                # ══ 熊市评分：二值基础+渐变加成（max≈100）══
                # 1) MA20斜率：基础15分(>0) + 渐变加成10分 → 最高25分
                if c["ma20_slope"] > 0:
                    score += 15 + min(10, c["ma20_slope"] * 300)
                # 2) ATR收敛：基础15分(ratio<1) + 渐变加成10分 → 最高25分
                if atr_r < 1.0:
                    score += 15 + min(10, (1.0 - atr_r) * 30)
                # 3) RPS20 × 0.10 → 最高10分
                score += rps_pct * 0.10
                # 4) RPS60 × 0.05 → 最高5分（中期趋势弱确认）
                score += rps60_pct * 0.05
                # 5) near_high：基础10分(>0.75) + 渐变5分 → 最高15分
                if c["near_high"] > 0.75:
                    score += 10 + min(5, (c["near_high"] - 0.75) / 0.25 * 5)
                elif c["near_high"] > 0.6:
                    score += (c["near_high"] - 0.6) / 0.15 * 10
                # 6) 缩量企稳（0.5~1.2）→ 15分（保持二值，熊市核心信号）
                if 0.5 <= c["amount_ratio"] <= 1.2:
                    score += 15
                # 7) 站上MA20 → 5分
                if c["close"] > c["MA20"] > 0:
                    score += 5

            c["rps20"] = round(rps_pct, 1)
            c["rps60"] = round(rps60_pct, 1)
            c["score"] = round(score, 1)

        # ── Phase 3: 形态识别（日线+周线）──────────────────────────────
        # 先按量化分排序，取 Top 30% 候选跑形态（控制耗时）
        from pattern_detector import PatternDetector, _standardize_df, daily_to_weekly

        candidate_results.sort(key=lambda x: x["score"], reverse=True)
        pattern_pool_n = max(50, int(len(candidate_results) * 0.30))
        pattern_pool = candidate_results[:pattern_pool_n]

        if verbose:
            print(f"\n  [L1] Phase 3: 形态识别（日线+周线）对 Top {pattern_pool_n} 只运行...")

        _pattern_start = time.time()
        eliminated_pattern = 0  # 被形态排除的数量

        for idx_c, c in enumerate(pattern_pool):
            if verbose and (idx_c + 1) % 50 == 0:
                print(f"\r  [形态] {idx_c + 1}/{pattern_pool_n}", end="", flush=True)

            try:
                raw_df = c.get("_df_raw")
                if raw_df is None or len(raw_df) < 60:
                    continue

                # 标准化列名 + 设置日期 index
                std_df = _standardize_df(raw_df.copy())
                if not isinstance(std_df.index, pd.DatetimeIndex):
                    if "date" in std_df.columns:
                        std_df.index = pd.to_datetime(std_df["date"])
                    elif "日期" in std_df.columns:
                        std_df.index = pd.to_datetime(std_df["日期"])
                    else:
                        std_df.index = pd.to_datetime(std_df.index)

                # 合成周线
                weekly_df = daily_to_weekly(std_df) if len(std_df) >= 100 else None

                detector = PatternDetector(
                    std_df, df_weekly=weekly_df,
                    pivot_bars_left=6, pivot_bars_right=6,
                    weekly_pivot_bars_left=4, weekly_pivot_bars_right=4
                )
                result = detector.detect_all()

                c["pattern_score"] = result["bullish_score"]
                c["pattern_exclude"] = result["bearish_exclude"]
                c["pattern_tags"] = result.get("bearish_tags", [])
                c["pattern_list"] = [p["pattern"] for p in result.get("patterns", [])]

                # 看空排除 → 标记（不从列表删除，但打分设为-1）
                if result["bearish_exclude"]:
                    c["score"] = -1
                    eliminated_pattern += 1
                else:
                    # 形态加分叠加到总分
                    c["score"] = round(c["score"] + result["bullish_score"], 1)

            except Exception as e:
                logger.debug(f"[L1] 形态检测异常 {c['code']}: {e}")
                c["pattern_score"] = 0
                c["pattern_exclude"] = False
                c["pattern_tags"] = []
                c["pattern_list"] = []

        # 未进入形态检测的候选，补充默认值
        for c in candidate_results[pattern_pool_n:]:
            c["pattern_score"] = 0
            c["pattern_exclude"] = False
            c["pattern_tags"] = []
            c["pattern_list"] = []

        if verbose:
            _pattern_elapsed = time.time() - _pattern_start
            print(f"\n  [L1] 形态检测完成：{eliminated_pattern} 只被排除，耗时 {_pattern_elapsed:.1f}秒")

        # ── 排序 + 截断：排除看空标记的候选，按综合分排序 ──
        candidate_results = [c for c in candidate_results if c["score"] > 0]
        candidate_results.sort(key=lambda x: x["score"], reverse=True)
        pct = 0.15 if is_bull else top_pct
        top_n = max(20, min(200, int(len(candidate_results) * pct)))
        top_results = candidate_results[:top_n]

        if verbose:
            if top_results:
                scores = [c["score"] for c in top_results]
                print(f"  [L1] 最终选出 {len(top_results)} 只（共{len(candidate_results)}只存活）")
                print(f"       得分范围: {min(scores):.1f} ~ {max(scores):.1f}")
                # 统计有形态加分的候选
                with_pattern = sum(1 for c in top_results if c.get("pattern_score", 0) > 0)
                print(f"       有形态加分: {with_pattern} 只")
            else:
                print(f"  [L1] 无候选存活")

        # 附带市场环境信息到每个候选
        for c in top_results:
            c["market_regime"] = market_regime

        # 清理内部字段
        for c in top_results:
            c.pop("_df_raw", None)

        return top_results

    # ------------------------------------------------------------------ #
    #  L2: 综合评分 + Top30 排名                                           #
    # ------------------------------------------------------------------ #

    def L2_score_and_rank(self, candidates: List[Dict], verbose: bool = True,
                          top_pct: float = 0.05) -> List[Dict]:
        """
        L2 综合评分：形态+量能+动量（三维）。

        v3: 动量维度新增RPS60中期确认，其余保持二值评分。
          - 牛市：形态35% + 量能25% + 动量40%
          - 熊市：形态45% + 量能35% + 动量20%
        """
        # 继承L1的市场环境
        market_regime = "bull"
        if candidates:
            market_regime = candidates[0].get("market_regime", "bull")
        is_bull = market_regime in ("bull", "neutral")

        if verbose:
            regime_cn = {"bull": "牛市", "bear": "熊市", "neutral": "震荡偏多"}.get(market_regime, market_regime)
            print(f"\n[数据] L2: 综合评分排名（{regime_cn}模式）...")
            print(f"  输入: {len(candidates)} 只")

        if not candidates:
            return []

        scored = []
        risk_rejected = 0
        data_insufficient = 0
        _l2_start = time.time()

        # 收集所有候选的5日收益率（用于超额收益百分位）
        ret5_list = []
        for stock in candidates:
            df = stock.get("df")
            if df is not None and len(df) >= 6:
                _c = "close" if "close" in df.columns else "收盘"
                c_arr = df[_c].astype(float).values
                ret5_list.append((c_arr[-1] / c_arr[-6] - 1) * 100)
            else:
                ret5_list.append(0)
        ret5_arr = np.array(ret5_list)

        for idx, stock in enumerate(candidates):
            if verbose and (idx + 1) % 50 == 0:
                elapsed = time.time() - _l2_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (len(candidates) - idx - 1) / rate if rate > 0 else 0
                print(f"\r  [L2] 进度 {idx+1}/{len(candidates)} 只 ({rate:.0f}只/秒, 剩余约{eta:.0f}秒)", end="", flush=True)

            code = stock["code"]

            # ── 获取300根K线（用于EMA60/MACD等长期指标）──────────────
            df300 = self._tdx_daily_only(code, days=300)
            if df300 is None or len(df300) < 60:
                data_insufficient += 1
                continue

            _c = "close" if "close" in df300.columns else "收盘"
            _o = "open" if "open" in df300.columns else "开盘"
            _h = "high" if "high" in df300.columns else "最高"
            _l = "low" if "low" in df300.columns else "最低"
            _v = "volume" if "volume" in df300.columns else ("vol" if "vol" in df300.columns else "成交量")

            close300 = df300[_c].astype(float).values
            open300 = df300[_o].astype(float).values
            high300 = df300[_h].astype(float).values
            low300 = df300[_l].astype(float).values
            vol300 = df300[_v].astype(float).values
            n300 = len(close300)

            # ── 轻量风控否决 ──────────────────────────────────────────
            # 连续5日下跌
            if n300 >= 6:
                drops = sum(1 for i in range(-5, 0) if close300[i] < close300[i - 1])
                if drops >= 5:
                    risk_rejected += 1
                    continue
            # 当日暴跌>7%
            if n300 >= 2 and close300[-2] > 0:
                if (close300[-1] - close300[-2]) / close300[-2] < -0.07:
                    risk_rejected += 1
                    continue

            # ═══════ 形态分（0~100）═══════════════════════════════════════
            ma5 = stock.get("MA5", 0)
            ma20 = stock.get("MA20", 0)
            ma60 = stock.get("MA60", 0)

            # MA多头排列（二值）
            if ma60 > 0 and ma5 > ma20 > ma60:
                ma_score = 30
            elif ma5 > ma20:
                ma_score = 15
            else:
                ma_score = 0

            # ATR收窄
            tr300 = np.maximum(high300[1:] - low300[1:],
                               np.maximum(np.abs(high300[1:] - close300[:-1]),
                                          np.abs(low300[1:] - close300[:-1])))
            if len(tr300) >= 20:
                atr_5 = float(np.mean(tr300[-5:]))
                atr_20 = float(np.mean(tr300[-20:]))
                atr_ratio = atr_5 / max(atr_20, 0.001)
                atr_score = max(0, min(30, (1.0 - atr_ratio) * 60))
            else:
                atr_score = 0

            # 站上MA20
            close_s = pd.Series(close300)
            ma20_val = close_s.rolling(20, min_periods=20).mean().iloc[-1]
            above_ma20 = 15 if (ma20_val == ma20_val and close300[-1] > ma20_val) else 0

            # 站上MA60
            ma60_score = 0
            if n300 >= 60:
                ma60_val = close_s.rolling(60, min_periods=60).mean().iloc[-1]
                if ma60_val == ma60_val and close300[-1] > ma60_val:
                    ma60_score = 15

            # MACD多头排列(DIF>DEA)
            ema12 = close_s.ewm(span=12, adjust=False).mean()
            ema26 = close_s.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            macd_bonus = 10 if dif.iloc[-1] > dea.iloc[-1] else 0

            form_score = min(ma_score + atr_score + above_ma20 + ma60_score + macd_bonus, 100)

            # ═══════ 量能确认分（0~100）══════════════════════════════════
            vol_score_val = 0
            if n300 >= 20:
                vol_ma5 = float(np.mean(vol300[-5:]))
                vol_ma20 = float(np.mean(vol300[-20:]))

                # 缩量蓄势
                if vol_ma5 < vol_ma20 * 0.8:
                    vol_score_val += 40
                elif vol_ma5 < vol_ma20:
                    vol_score_val += 20

                # 量能稳定（低变异系数）
                vol_std = float(np.std(vol300[-5:]))
                vol_cv = vol_std / max(vol_ma5, 1)
                vol_score_val += max(0, min(30, (1.0 - vol_cv) * 40))

                # 温和收阳（收阳 + 量不超均量1.5倍）
                if close300[-1] > open300[-1] and vol300[-1] < vol_ma20 * 1.5:
                    vol_score_val += 30

            vol_score_val = min(vol_score_val, 100)

            # ═══════ 动量分（0~100）：RPS20+RPS60+5日超额 ════════════════
            from scipy.stats import percentileofscore
            # RPS20 × 30
            rps_pct = stock.get("rps20", 50) / 100.0
            rps20_score = rps_pct * 30
            # RPS60 × 20（中期趋势确认，v3新增）
            rps60_pct = stock.get("rps60", 50) / 100.0
            rps60_score = rps60_pct * 20
            # 5日超额收益百分位 × 50
            ret5_val = ret5_list[idx]
            excess_pct = percentileofscore(ret5_arr, ret5_val, kind='rank') / 100.0
            excess_score = excess_pct * 50

            momentum_score = min(rps20_score + rps60_score + excess_score, 100)

            # ═══════ 综合评分（牛熊差异化权重）══════════════════════════
            if is_bull:
                total = form_score * 0.35 + vol_score_val * 0.25 + momentum_score * 0.40
            else:
                total = form_score * 0.45 + vol_score_val * 0.35 + momentum_score * 0.20

            scored.append({
                **stock,
                "l2_score": round(total, 1),
                "form_score": round(form_score, 1),
                "vol_score": round(vol_score_val, 1),
                "mom_score": round(momentum_score, 1),
                "l2_detail": f"形态={form_score:.0f} 量能={vol_score_val:.0f} 动量={momentum_score:.0f}",
                "market_regime": market_regime,
            })

        # 按 l2_score 降序排列，牛市取 Top 15%，熊市取 Top 5%
        scored.sort(key=lambda x: x["l2_score"], reverse=True)
        pct = 0.15 if is_bull else top_pct
        top_n = max(5, min(30, int(len(scored) * pct)))
        top_results = scored[:top_n]

        if verbose:
            elapsed = time.time() - _l2_start
            regime_cn = {"bull": "牛市", "bear": "熊市", "neutral": "震荡偏多"}.get(market_regime, market_regime)
            print(f"\n  [L2] 评分完成（{regime_cn}）：{len(scored)} 只已评分，取 Top {top_pct:.0%}={top_n} 只，耗时 {elapsed:.1f}秒")
            print(f"       风控否决: {risk_rejected} | 数据不足: {data_insufficient}")
            if is_bull:
                print(f"       权重: 形态35% + 量能25% + 动量40%")
            else:
                print(f"       权重: 形态45% + 量能35% + 动量20%")
            if top_results:
                scores = [c["l2_score"] for c in top_results]
                print(f"       Top{min(top_n, len(top_results))} 得分范围: {min(scores):.1f} ~ {max(scores):.1f}")

        return top_results

    # ------------------------------------------------------------------ #
    #  L3: K线图 + 指标生成                                               #
    # ------------------------------------------------------------------ #

    def L3_generate_charts(self, candidates: List[Dict],
                           output_dir: str = "output/charts",
                           verbose: bool = True) -> Dict[str, str]:
        """
        L3: 为候选股票生成 K线图 + MACD + KDJ + RSI 截图。

        返回: {code: {"kline": path, "macd": path, "kdj": path, "rsi": path}}
        """
        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print(f"\n[数据] L3: 生成K线图+指标图...")
            print(f"  生成 {len(candidates)} 只股票的图表...")

        chart_paths: Dict[str, Dict[str, str]] = {}

        for i, stock in enumerate(candidates):
            code = stock["code"]
            name = stock["name"]
            df = stock.get("df")

            if df is None or len(df) < 20:
                continue

            stock_dir = os.path.join(output_dir, code)
            os.makedirs(stock_dir, exist_ok=True)
            chart_paths[code] = {}

            try:
                import matplotlib
                matplotlib.use("Agg")  # 无头模式，不弹出窗口
                import matplotlib.pyplot as plt
                import matplotlib.gridspec as gridspec
                from matplotlib import font_manager

                # 设置中文字体
                font_paths = [
                    "C:/Windows/Fonts/simhei.ttf",
                    "C:/Windows/Fonts/msyh.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                ]
                for fp in font_paths:
                    if os.path.exists(fp):
                        plt.rcParams["font.sans-serif"] = [
                            font_manager.FontProperties(fname=fp).get_name()
                        ]
                        break
                plt.rcParams["axes.unicode_minus"] = False

                _c = "close" if "close" in df.columns else "收盘"
                _h = "high" if "high" in df.columns else "最高"
                _l = "low" if "low" in df.columns else "最低"
                _o = "open" if "open" in df.columns else "开盘"
                close = df[_c].astype(float).values
                high = df[_h].astype(float).values
                low = df[_l].astype(float).values
                open_col = df[_o].astype(float).values
                volume = None
                for vc in ["成交量", "vol", "volume"]:
                    if vc in df.columns:
                        volume = df[vc].astype(float).values
                        break

                # 计算指标
                import talib
                ma5 = talib.SMA(np.array(close), timeperiod=5)
                ma10 = talib.SMA(np.array(close), timeperiod=10)
                ma20 = talib.SMA(np.array(close), timeperiod=20)
                macd, macd_sig, macd_hist = talib.MACD(np.array(close))
                k, d = talib.STOCH(high, low, close)
                rsi = talib.RSI(np.array(close), timeperiod=14)

                # 绘图：K线 + 均线（主图）
                n = min(60, len(df))  # 最近60根K线
                x = range(n)
                closes_n = close[-n:]
                highs_n = high[-n:]
                lows_n = low[-n:]
                opens_n = open_col[-n:]

                fig = plt.figure(figsize=(14, 10))
                gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.1)

                # 1. K线 + 均线
                ax1 = fig.add_subplot(gs[0])
                for j in range(n):
                    c = closes_n[j]
                    o = opens_n[j]
                    h = highs_n[j]
                    l = lows_n[j]
                    color = "#e74c3c" if c >= o else "#27ae60"
                    ax1.plot([j, j], [l, h], color=color, linewidth=0.8)
                    ax1.plot([j-0.3, j+0.3], [c, c], color=color, linewidth=0.8)
                    ax1.plot([j-0.3, j+0.3], [o, o], color=color, linewidth=0.8)
                    ax1.add_patch(plt.Rectangle((j-0.35, min(o, c)), 0.7, abs(c-o),
                                                 color=color, alpha=0.3))

                ma5_n = ma5[-n:]
                ma10_n = ma10[-n:]
                ma20_n = ma20[-n:]
                ax1.plot(x, ma5_n, label="MA5", linewidth=1, color="#e74c3c")
                ax1.plot(x, ma10_n, label="MA10", linewidth=1, color="#3498db")
                ax1.plot(x, ma20_n, label="MA20", linewidth=1, color="#f39c12")
                ax1.set_title(f"{code} {name}", fontsize=14, fontweight="bold")
                ax1.legend(loc="upper left")
                ax1.set_xticklabels([])
                ax1.grid(True, alpha=0.3)

                # 2. MACD
                ax2 = fig.add_subplot(gs[1], sharex=ax1)
                macd_n = macd[-n:]
                sig_n = macd_sig[-n:]
                hist_n = macd_hist[-n:]
                ax2.bar(x, hist_n, color=["#e74c3c" if h > 0 else "#27ae60" for h in hist_n], alpha=0.6)
                ax2.plot(x, macd_n, color="#3498db", linewidth=1, label="MACD")
                ax2.plot(x, sig_n, color="#f39c12", linewidth=1, label="Signal")
                ax2.axhline(y=0, color="gray", linewidth=0.5)
                ax2.legend(loc="upper left", fontsize=8)
                ax2.set_xticklabels([])
                ax2.grid(True, alpha=0.3)

                # 3. KDJ
                ax3 = fig.add_subplot(gs[2], sharex=ax1)
                k_n = k[-n:]
                d_n = d[-n:]
                ax3.plot(x, k_n, color="#e74c3c", linewidth=1, label="K")
                ax3.plot(x, d_n, color="#3498db", linewidth=1, label="D")
                ax3.axhline(y=80, color="gray", linewidth=0.5, linestyle="--")
                ax3.axhline(y=20, color="gray", linewidth=0.5, linestyle="--")
                ax3.legend(loc="upper left", fontsize=8)
                ax3.set_xticklabels([])
                ax3.grid(True, alpha=0.3)

                # 4. RSI
                ax4 = fig.add_subplot(gs[3], sharex=ax1)
                rsi_n = rsi[-n:]
                ax4.plot(x, rsi_n, color="#9b59b6", linewidth=1.5, label="RSI14")
                ax4.axhline(y=70, color="gray", linewidth=0.5, linestyle="--")
                ax4.axhline(y=30, color="gray", linewidth=0.5, linestyle="--")
                ax4.fill_between(x, rsi_n, 70, where=(rsi_n >= 70), color="#e74c3c", alpha=0.2)
                ax4.fill_between(x, rsi_n, 30, where=(rsi_n <= 30), color="#27ae60", alpha=0.2)
                ax4.set_xlim(-1, n)
                ax4.legend(loc="upper left", fontsize=8)
                ax4.set_xticklabels([])
                ax4.grid(True, alpha=0.3)

                # 保存
                out_path = os.path.join(stock_dir, f"{code}_{name}.png")
                fig.savefig(out_path, dpi=80, bbox_inches="tight",
                           facecolor="white", edgecolor="none")
                plt.close(fig)
                chart_paths[code] = {"chart": out_path}

            except ImportError as e:
                if verbose:
                    print(f"  [L3] 缺少依赖: {e}，跳过图表生成")
                chart_paths[code] = {}
            except Exception as e:
                if verbose:
                    print(f"  [L3] {code} 图表生成失败: {e}")
                chart_paths[code] = {}

        if verbose:
            success = sum(1 for v in chart_paths.values() if v)
            print(f"  [L3] ✓ 完成：{success}/{len(candidates)} 只图表生成成功")

        return chart_paths

    # ------------------------------------------------------------------ #
    #  辅助方法                                                           #
    # ------------------------------------------------------------------ #

    def _tdx_daily_only(self, code: str, days: int = 60,
                        cutoff_date: str = None) -> Optional[pd.DataFrame]:
        """仅从TDX本地获取日线数据（不调用任何远程API）

        cutoff_date: 'YYYYMMDD'，设置后只返回该日期及之前的数据（回测point-in-time用）
        """
        if not self._tdx:
            return None
        try:
            df = self._tdx.daily(symbol=code)
            if df is not None and len(df) > 0:
                if cutoff_date:
                    # 过滤到 cutoff_date（含），严格 point-in-time
                    if 'date' in df.columns:
                        date_series = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
                    else:
                        date_series = pd.Series(df.index).apply(
                            lambda x: pd.to_datetime(x).strftime('%Y%m%d')
                        )
                    df = df[date_series.values <= cutoff_date]
                    if len(df) == 0:
                        return None
                return df.tail(days + 10)
        except Exception:
            pass
        return None

    def _fetch_daily_data(self, code: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取个股日线数据（优先TDX本地 -> akshare -> tushare）"""
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        # TDX 本地
        if self._tdx:
            try:
                df = self._tdx.daily(symbol=code)
                if df is not None and len(df) > 0:
                    # TDX 列名: date, open, high, low, close, turnover
                    df = df.tail(days + 10)
                    return df
            except Exception:
                pass

        # akshare
        if ak:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start_date, end_date=end_date, adjust="qfq"
                )
                if df is not None and len(df) > 0:
                    # 统一列名：支持中文（日期/开盘/收盘/最高/最低/成交量）和英文（date/open/close/high/low/volume）
                    rename = {}
                    for c in df.columns:
                        cl = c.lower()
                        if "日期" in c or cl == "date":
                            rename[c] = "date"
                        elif "开盘" in c or cl == "open":
                            rename[c] = "open"
                        elif "收盘" in c or cl == "close":
                            rename[c] = "close"
                        elif "最高" in c or cl == "high":
                            rename[c] = "high"
                        elif "最低" in c or cl == "low":
                            rename[c] = "low"
                        elif "成交量" in c or cl in ("volume", "vol"):
                            rename[c] = "volume"
                    if rename:
                        df = df.rename(columns=rename)
                    return df.tail(days + 10)
            except Exception:
                pass

        # tushare
        if self._pro:
            try:
                df = self._pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
                if df is not None and len(df) > 0:
                    return df.tail(days + 10)
            except Exception:
                pass

        return None

    @staticmethod
    def _calc_sma(series: pd.Series, period: int) -> pd.Series:
        """计算简单移动平均"""
        return series.rolling(window=period, min_periods=1).mean()
