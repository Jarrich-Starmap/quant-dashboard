"""技术指标计算与信号映射。
RSI — ADX 自适应 S 型分段映射
MACD — 非对称零轴因子
布林带 — 带宽因子
动量 — 加速判断
ADX — Wilder 趋势强度（RSI 调制器，非交易信号）
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class RSISignal:
    value: float
    signal: float


@dataclass
class MACDSignal:
    line: float
    signal_line: float
    hist: float
    zero_factor: float
    signal: float


@dataclass
class BBSignal:
    upper: float
    lower: float
    middle: float
    bandwidth_ratio: float
    signal: float


@dataclass
class MomentumSignal:
    value: float
    acceleration: float
    signal: float


@dataclass
class ADXSignal:
    value: float
    plus_di: float
    minus_di: float


def compute_adx(highs: pd.Series, lows: pd.Series, closes: pd.Series, period: int = 14) -> ADXSignal:
    """ADX 趋势强度指标。Wilder 平滑。"""
    # True Range
    tr1 = highs - lows
    tr2 = (highs - closes.shift(1)).abs()
    tr3 = (lows - closes.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = highs.diff()
    down_move = lows.shift(1) - lows  # prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Wilder smoothing (EWM alpha=1/period)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    adx_val = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
    pdi_val = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
    mdi_val = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0

    return ADXSignal(value=round(adx_val, 2), plus_di=round(pdi_val, 2), minus_di=round(mdi_val, 2))


def compute_rsi(closes: pd.Series, period: int = 14,
                adx_val: float = 0.0,
                adx_trend: float = 25.0,
                adx_range: float = 20.0) -> RSISignal:
    """RSI 计算 + ADX 自适应 S 型分段映射。

    ADX >= adx_trend (强趋势): 阈值放宽至 80/20, 过渡区 70-80 / 20-30
    ADX <= adx_range (震荡):   阈值 70/30, 过渡区 60-70 / 30-40
    adx_range < ADX < adx_trend: 线性插值平滑过渡
    """
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_val = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    # ADX 自适应阈值
    if adx_val >= adx_trend:
        ob_thresh, ob_trans = 80.0, 70.0
        os_thresh, os_trans = 20.0, 30.0
    elif adx_val <= adx_range:
        ob_thresh, ob_trans = 70.0, 60.0
        os_thresh, os_trans = 30.0, 40.0
    else:
        t = (adx_val - adx_range) / (adx_trend - adx_range)
        ob_thresh = 70.0 + 10.0 * t
        ob_trans = 60.0 + 10.0 * t
        os_thresh = 30.0 - 10.0 * t
        os_trans = 40.0 - 10.0 * t

    if rsi_val >= ob_thresh:
        signal = -1.0
    elif rsi_val <= os_thresh:
        signal = 1.0
    elif ob_trans <= rsi_val < ob_thresh:
        signal = -(rsi_val - ob_trans) / (ob_thresh - ob_trans)
    elif os_thresh < rsi_val <= os_trans:
        signal = (os_trans - rsi_val) / (os_trans - os_thresh)
    else:
        signal = 0.0

    return RSISignal(value=round(rsi_val, 2), signal=round(signal, 4))


def compute_macd(closes: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> MACDSignal:
    """MACD + 非对称零轴因子。"""
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=sig, adjust=False).mean()
    hist = macd_line - signal_line

    ml = float(macd_line.iloc[-1])
    sl = float(signal_line.iloc[-1])
    h = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2]) if len(hist) >= 2 else h

    # raw_signal
    if h > 0 and h > h_prev:
        raw = 1.0
    elif h < 0 and h < h_prev:
        raw = -1.0
    else:
        raw = 0.0

    # 非对称零轴因子
    factor = 0.7 + 0.3 * np.sign(ml) * raw
    factor = max(0.4, min(1.0, factor))

    return MACDSignal(
        line=round(ml, 6),
        signal_line=round(sl, 6),
        hist=round(h, 6),
        zero_factor=round(factor, 4),
        signal=round(raw * factor, 4),
    )


def compute_bb(closes: pd.Series, period: int = 20, stddev: float = 2.0) -> BBSignal:
    """布林带 + 带宽因子。"""
    middle = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    upper = middle + stddev * std
    lower = middle - stddev * std

    price = float(closes.iloc[-1])
    bb_up = float(upper.iloc[-1])
    bb_lo = float(lower.iloc[-1])
    bb_mid = float(middle.iloc[-1])
    bandwidth = bb_up - bb_lo

    # 带宽比 = 当前带宽 / SMA(带宽, 20)
    bandwidth_series = upper - lower
    bandwidth_ma = bandwidth_series.rolling(20).mean()
    bw_ma_val = float(bandwidth_ma.iloc[-1])
    bandwidth_ratio = bandwidth / bw_ma_val if bw_ma_val and bw_ma_val > 0 else 1.0

    # raw_signal
    if price <= bb_lo:
        raw = 1.0
    elif price >= bb_up:
        raw = -1.0
    else:
        span = bb_up - bb_mid
        if span == 0:
            raw = 0.0
        else:
            raw = -(price - bb_mid) / span

    # 带宽乘数
    multiplier = 1.0 + 0.3 * (1.0 - min(bandwidth_ratio, 2.0))
    multiplier = max(0.5, min(1.5, multiplier))

    return BBSignal(
        upper=round(bb_up, 4),
        lower=round(bb_lo, 4),
        middle=round(bb_mid, 4),
        bandwidth_ratio=round(bandwidth_ratio, 4),
        signal=round(max(-1.0, min(1.0, raw * multiplier)), 4),
    )


def compute_momentum(closes: pd.Series, period: int = 10) -> MomentumSignal:
    """动量 + 加速判断。"""
    mom = closes.pct_change(period)
    mom_prev = closes.shift(1).pct_change(period)
    accel = mom - mom_prev

    m = float(mom.iloc[-1]) if not pd.isna(mom.iloc[-1]) else 0.0
    a = float(accel.iloc[-1]) if not pd.isna(accel.iloc[-1]) else 0.0

    # weight: 加速满分 1.0, 减速折半 0.5
    if m > 0 and a > 0:
        weight = 1.0
    elif m > 0 and a <= 0:
        weight = 0.5
    elif m < 0 and a < 0:
        weight = 1.0
    elif m < 0 and a >= 0:
        weight = 0.5
    else:
        weight = 0.0

    sig = float(np.tanh(m * 50) * weight)
    sig = max(-1.0, min(1.0, sig))

    return MomentumSignal(value=round(m, 6), acceleration=round(a, 6), signal=round(sig, 4))


def compute_all(closes: pd.Series, params: dict,
                highs: Optional[pd.Series] = None,
                lows: Optional[pd.Series] = None) -> dict:
    """计算全部技术指标，返回完整结果字典。

    highs/lows 可选，用于计算 ADX 趋势强度。
    ADX 作为 RSI 的调制器，不直接参与 softmax 加权。
    """
    adx = None
    adx_val = 0.0
    if highs is not None and lows is not None:
        adx = compute_adx(highs, lows, closes, params.get("adx_period", 14))
        adx_val = adx.value

    return {
        "rsi": compute_rsi(
            closes, params["rsi_period"], adx_val,
            params.get("adx_trend_threshold", 25.0),
            params.get("adx_range_threshold", 20.0),
        ),
        "macd": compute_macd(closes, params["macd_fast"], params["macd_slow"], params["macd_signal"]),
        "bb": compute_bb(closes, params["bb_period"], params["bb_stddev"]),
        "momentum": compute_momentum(closes, params["momentum_period"]),
        "adx": adx,
    }
