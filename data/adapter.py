"""数据源适配层。将 AkShare 原始返回统一转换为标准 OHLCV 格式。"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests as _requests


# ---- 全局 requests 默认超时（杜绝 AkShare 底层 socket 永久阻塞）----
def _install_requests_timeout(default_timeout: int = 15):
    """给 requests 的所有调用加默认超时。
    只覆盖调用方未显式传入 timeout 的请求（AkShare 内部几乎都不传）。
    若某次请求在 default_timeout 内无响应，requests 会抛 Timeout 异常，
    由上层 _call_with_timeout 捕获并降级，而非永久挂起。"""
    _orig = _requests.Session.request

    def _patched(self, method, url, *args, **kwargs):
        kwargs.setdefault("timeout", default_timeout)
        return _orig(self, method, url, *args, **kwargs)

    _requests.Session.request = _patched


_install_requests_timeout()


STANDARD_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def normalize_akshare(df: pd.DataFrame) -> list[dict]:
    """
    AkShare futures_zh_minute_sina 返回的 DataFrame → 标准 dict 列表。

    输入列名可能为中文（时间/开盘价/最高价/最低价/收盘价/成交量）或英文。
    输出每行: {timestamp, open, high, low, close, volume}
    """
    if df is None or df.empty:
        return []

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 列名中文→英文映射
    col_map = {}
    for col in df.columns:
        c = col.lower()
        if any(k in c for k in ("时间", "time", "date", "日期")):
            col_map["timestamp"] = col
        elif any(k in c for k in ("开", "open")):
            col_map["open"] = col
        elif any(k in c for k in ("高", "high")):
            col_map["high"] = col
        elif any(k in c for k in ("低", "low")):
            col_map["low"] = col
        elif any(k in c for k in ("收", "close")):
            col_map["close"] = col
        elif any(k in c for k in ("量", "vol")):
            col_map["volume"] = col

    records = []
    for _, row in df.iterrows():
        ts_raw = row.get(col_map.get("timestamp", df.columns[0]))
        if ts_raw is None or pd.isna(ts_raw):
            continue
        try:
            ts = pd.Timestamp(ts_raw).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = str(ts_raw)

        records.append({
            "timestamp": ts,
            "open": float(row.get(col_map.get("open", 0)) or 0),
            "high": float(row.get(col_map.get("high", 0)) or 0),
            "low": float(row.get(col_map.get("low", 0)) or 0),
            "close": float(row.get(col_map.get("close", 0)) or 0),
            "volume": float(row.get(col_map.get("volume", 0)) or 0),
        })

    return records


# ---- 文件缓存（跨进程持久化）+ 超时保护 ----
_CACHE_FILE = Path(__file__).resolve().parent.parent / "contract_cache.json"
_CACHE_TTL = 3600  # 缓存1小时
_API_TIMEOUT = 15  # 单次 AkShare API 调用超时秒数


def _load_cache() -> dict:
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict):
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass


def _call_with_timeout(func, *args, timeout: int = _API_TIMEOUT, **kwargs):
    """在子线程中调用 func，超时返回 None。

    注意：Python 线程无法被强制终止。超时后若底层 socket 仍阻塞，
    子线程会泄漏并在服务端关闭连接后自行退出；这里 **不** 用
    `with` 语句（其退出时会 shutdown(wait=True) 死等泄漏线程），
    而是显式 shutdown(wait=False)，避免主线程被连带挂起。
    配合 _install_requests_timeout 的底层超时，绝大多数情况下
    子线程会在 timeout 内正常抛异常退出，不会泄漏。
    """
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(func, *args, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeout:
            return None
        except Exception:
            return None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def resolve_contract_code(symbol: str, fallback: str = "") -> str:
    """
    通过交易所合约信息 + 日K线持仓量确定当前主力合约。

    AU/AG → SHFE (futures_contract_info_shfe)
    SC    → INE  (futures_contract_info_ine)

    策略:
    1. 获取合约列表，筛选已上市且未到期的合约
    2. 取最近到期的4个合约作为候选
    3. 逐一查询日K线，选择持仓量最大的合约作为主力
    4. API失败时回退到最近到期合约

    结果通过文件缓存1小时（跨进程持久化），每次API调用有15秒超时保护。
    symbol: "AU"/"AG"/"SC"，返回如 "au2608"
    """
    # 检查文件缓存
    cache = _load_cache()
    cached = cache.get(symbol)
    if cached and (time.time() - cached.get("ts", 0)) < _CACHE_TTL:
        return cached["code"]

    try:
        import akshare as ak

        # 交易所合约信息接口在非交易日可能返回空，最多往前试3天
        df = None
        for days_back in range(4):
            d = date.today() - timedelta(days=days_back)
            try:
                if symbol.upper() in ("AU", "AG"):
                    df = _call_with_timeout(ak.futures_contract_info_shfe, date=d.strftime("%Y%m%d"))
                elif symbol.upper() == "SC":
                    df = _call_with_timeout(ak.futures_contract_info_ine, date=d.strftime("%Y%m%d"))
                elif symbol.upper() in ("IC", "IM"):
                    df = _call_with_timeout(ak.futures_contract_info_cffex, date=d.strftime("%Y%m%d"))
                else:
                    raise ValueError(f"Unsupported symbol: {symbol}")
                if df is not None and not df.empty:
                    break
            except Exception:
                continue

        if df is None or df.empty:
            raise ValueError(f"Cannot fetch contract info for {symbol}")

        # CFFEX(中金所) 使用"最后交易日"而非"到期日"，重命名以复用后续通用过滤逻辑
        if symbol.upper() in ("IC", "IM") and "最后交易日" in df.columns and "到期日" not in df.columns:
            df = df.rename(columns={"最后交易日": "到期日"})

        # 按合约代码前缀过滤（如 au2608、ag2608、sc2608、ic2608）
        prefix = symbol.lower()
        matching = df[df["合约代码"].str.match(rf"^{prefix}\d{{4}}$", case=False)].copy()
        if matching.empty:
            raise ValueError(f"No contracts found for {symbol}")

        # 筛选已上市且未到期的合约
        today = date.today()
        matching["到期日_parsed"] = pd.to_datetime(matching["到期日"], errors="coerce")
        matching = matching[matching["到期日_parsed"] > pd.Timestamp(today)]

        if "上市日" in matching.columns:
            matching["上市日_parsed"] = pd.to_datetime(matching["上市日"], errors="coerce")
            matching = matching[matching["上市日_parsed"] <= pd.Timestamp(today)]

        if matching.empty:
            raise ValueError(f"No active contracts for {symbol}")

        # 取最近到期的4个合约作为候选
        matching = matching.sort_values("到期日_parsed")
        candidates = matching.head(4)

        # 逐一查询日K线持仓量，选持仓量最大的合约
        best_contract = None
        best_hold = -1.0

        for _, row in candidates.iterrows():
            code = str(row["合约代码"]).strip()
            daily = _call_with_timeout(ak.futures_zh_daily_sina, symbol=code)
            if daily is not None and not daily.empty:
                hold = float(daily.tail(1)["hold"].values[0])
                if hold > best_hold:
                    best_hold = hold
                    best_contract = code

        if best_contract:
            cache[symbol] = {"code": best_contract, "ts": time.time()}
            _save_cache(cache)
            return best_contract

        # 回退: 取最近到期合约
        fallback_contract = str(matching.iloc[0]["合约代码"]).strip()
        cache[symbol] = {"code": fallback_contract, "ts": time.time()}
        _save_cache(cache)
        return fallback_contract

    except Exception:
        pass

    # 尝试使用过期缓存（比 fallback 更好）
    if cached:
        return cached["code"]

    return fallback
