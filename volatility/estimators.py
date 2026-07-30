"""
Volatility estimators — Yang-Zhang (primary) + ATR (utility).

Design decision (方案B'):
    Only Yang-Zhang is used as the volatility estimator.
    YZ is a strict superset of ATR information-wise:
      - sigma_overnight  →  captures the gap component of TR
      - sigma_rs         →  captures the intraday range (H-L) component
      - sigma_close      →  adds directional info ATR doesn't have

    ATR is retained as a UTILITY function only — it provides absolute
    price-unit volatility for stop-loss calculations (YZ outputs a
    dimensionless log-return fraction, not a price value).

Input: a pandas DataFrame with columns [open, high, low, close, volume].
Computational complexity: O(n) — same as single ATR.
"""

import numpy as np
import pandas as pd


# ============================================================
# Primary estimator: Yang-Zhang
# ============================================================

def calc_yang_zhang(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Yang-Zhang volatility estimator.

    Separates overnight (open-to-open) and intraday (close-to-close + RS)
    variance. Best for futures with frequent overnight gaps (night session).

    sigma^2_YZ = sigma^2_overnight + k * sigma^2_close + (1-k) * sigma^2_RS
    where k = 0.34 / (1.34 + (n+1)/(n-1))

    Output: volatility as a fraction of price (dimensionless).
    """
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)

    # Overnight variance: log(open_t / close_{t-1})
    log_overnight = np.log(open_ / prev_close.replace(0, np.nan))
    sigma_overnight_sq = log_overnight.rolling(period, min_periods=2).var()

    # Rogers-Satchell intraday variance
    log_ho = np.log(high / open_.replace(0, np.nan))
    log_lo = np.log(low / open_.replace(0, np.nan))
    log_co = np.log(close / open_.replace(0, np.nan))

    rs_var = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co))
    rs_var = rs_var.rolling(period, min_periods=1).mean().clip(lower=0)

    # Close-to-close variance
    log_cc = np.log(close / prev_close.replace(0, np.nan))
    sigma_close_sq = log_cc.rolling(period, min_periods=2).var()

    # Yang-Zhang weighting factor
    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    sigma_yz_sq = sigma_overnight_sq + k * sigma_close_sq + (1 - k) * rs_var
    sigma_yz_sq = sigma_yz_sq.clip(lower=0)

    return np.sqrt(sigma_yz_sq)


# ============================================================
# Utility: ATR (for stop-loss price units only)
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder's original).

    Retained as a UTILITY — provides absolute price units for stop-loss
    calculations. Not used as a volatility estimator; YZ is the sole
    estimator. ATR is kept because:
      1. Stop-loss needs price-unit distance (e.g. "18.5 points")
      2. YZ outputs a dimensionless fraction; converting back requires
         a sqrt(bar_period) factor that needs calibration
      3. ATR is a single EMA — computational cost is negligible

    TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
    ATR = EMA(TR, period)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing (equivalent to EMA with alpha = 1/period)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()
