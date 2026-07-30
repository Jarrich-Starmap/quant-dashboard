"""行情采集：AkShare → adapter 标准化 → 新鲜度检查 → 换月检测 → K线缓存。"""

import warnings
from datetime import datetime
from typing import Optional

import pandas as pd

from data.adapter import normalize_akshare, resolve_contract_code, _call_with_timeout
from db.models import (
    get_cached_klines,
    upsert_klines,
    clear_klines,
    get_latest_close,
    reset_ewma_state,
)

warnings.filterwarnings("ignore")


def fetch_and_cache(symbol: str, config: dict) -> tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], bool, bool]:
    """
    拉取 AkShare 分钟 K 线，经 adapter 标准化后合并缓存。

    返回: (closes, highs, lows, is_stale, is_rollover)
    """
    kline_count = config.get("kline_cache_count", 100)
    freshness_threshold = config.get("data_freshness_threshold", 600)
    rollover_pct = config.get("contract_rollover_gap_pct", 0.03)

    akshare_code = resolve_contract_code(symbol)

    records = []
    try:
        import akshare as ak
        df = _call_with_timeout(ak.futures_zh_minute_sina, symbol=akshare_code, timeout=15)
        records = normalize_akshare(df)
    except Exception:
        pass

    if not records:
        cached = get_cached_klines(symbol, limit=200)
        if not cached:
            return None, None, None, True, False
        closes = pd.Series([r["close"] for r in cached])
        highs = pd.Series([r["high"] for r in cached])
        lows = pd.Series([r["low"] for r in cached])
        is_stale = _check_staleness(closes, freshness_threshold)
        return closes, highs, lows, is_stale, False

    # 丢弃最新一根未闭合的分钟K线（Sina 在分钟进行中实时更新 OHLCV）
    records = records[:-1]

    records = records[-kline_count:] if len(records) > kline_count else records

    is_rollover = False
    latest_close_old = get_latest_close(symbol)
    latest_close_new = records[-1]["close"]
    if latest_close_old is not None and latest_close_old > 0:
        gap = abs(latest_close_new - latest_close_old) / latest_close_old
        if gap > rollover_pct:
            is_rollover = True
            clear_klines(symbol, keep_last=20)
            reset_ewma_state(symbol)

    upsert_klines(symbol, records)

    is_stale = _check_staleness_by_latest_ts(records[-1]["timestamp"], freshness_threshold)

    cached = get_cached_klines(symbol, limit=200)
    if not cached:
        return None, None, None, True, is_rollover

    closes = pd.Series([r["close"] for r in cached])
    highs = pd.Series([r["high"] for r in cached])
    lows = pd.Series([r["low"] for r in cached])
    return closes, highs, lows, is_stale, is_rollover


def _check_staleness(closes: pd.Series, threshold: int) -> bool:
    """检查缓存数据是否陈旧。closes 为空或最新 Bar 距今超过 threshold 秒则视为陈旧。"""
    if len(closes) < 40:
        return True
    last_close = closes.index[-1]
    try:
        last_ts = pd.Timestamp(last_close).to_pydatetime()
        return (datetime.now() - last_ts).total_seconds() > threshold
    except Exception:
        return True


def _check_staleness_by_latest_ts(latest_ts_str: str, threshold: int) -> bool:
    try:
        latest = pd.Timestamp(latest_ts_str).to_pydatetime()
        return (datetime.now() - latest).total_seconds() > threshold
    except Exception:
        return False
