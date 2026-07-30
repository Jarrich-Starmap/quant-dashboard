"""
Adaptive Volatility Indicator (AVI) — 方案B' 精简版.

Single estimator (Yang-Zhang) + z-score state assessment.

Key changes from previous version:
  1. Only Yang-Zhang estimator (removed ATR/Parkinson/GK fusion + adaptive weights)
  2. z-score replaces percentile rank (EWMA-based mean/std, O(n) not O(n log n))
  3. Expansion rate uses fast/slow EMA diff (not single-period ROC)
  4. Squeeze alert removed (BB bandwidth already covers it)
  5. Warmup protection: early z-scores fillna(0) to avoid NaN/extreme values

Outputs:
  1. yz         — raw YZ volatility (fraction of price, dimensionless)
  2. z_score    — current volatility vs EWMA baseline (statistical, not percentile)
  3. state      — CALM / NORMAL / HIGH / EXTREME (informational only; actual
                  adjustments use continuous z-score mapping, not thresholds)
  4. expansion_rate — fast/slow EMA differential (robust, not noisy single-step)
  5. atr        — ATR in absolute price units (for stop-loss layer)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimators import calc_yang_zhang, calc_atr


@dataclass
class VolatilityState:
    """
    Immutable snapshot of current volatility state.

    The z_score is the PRIMARY signal for all downstream layers.
    State string is for logging/DB only — never use it for if-elif branching.
    """

    yz: float                # Raw YZ volatility (fraction of price)
    z_score: float           # z-score: (yz - ewm_mean) / ewm_std
    state: str               # CALM / NORMAL / HIGH / EXTREME (informational)
    expansion_rate: float    # Fast/slow EMA diff (positive = expanding)
    atr: float               # ATR in absolute price units (for stop-loss)

    @property
    def effective_atr(self) -> float:
        """Alias for stop-loss compatibility."""
        return self.atr

    def to_dict(self) -> dict:
        return {
            "yz": self.yz,
            "z_score": self.z_score,
            "vol_state": self.state,
            "expansion_rate": self.expansion_rate,
            "atr": self.atr,
        }


class AVI:
    """
    Volatility state calculator — YZ + z-score.

    Usage:
        avi = AVI(config)
        vol_state = avi.calculate(df)  # df = OHLCV DataFrame
    """

    def __init__(self, config: dict):
        self.period = config.get("period", 14)
        self.state_window = config.get("state_window", 60)
        self.trend_fast = config.get("trend_fast", 3)
        self.trend_slow = config.get("trend_slow", 10)
        self.calm_z = config.get("calm_z", -1.0)
        self.high_z = config.get("high_z", 1.0)
        self.extreme_z = config.get("extreme_z", 2.0)

    def calculate(self, df: pd.DataFrame) -> VolatilityState:
        """
        Calculate volatility state from OHLCV data.

        Args:
            df: DataFrame with [open, high, low, close, volume]
                Needs at least `period + 1` rows for YZ.

        Returns:
            VolatilityState snapshot for the latest bar.
        """
        if len(df) < self.period + 1:
            return self._fallback_state(df)

        # 1. Yang-Zhang volatility (dimensionless fraction)
        yz = calc_yang_zhang(df, self.period)
        current_yz = yz.iloc[-1]

        # 2. z-score via EWMA mean/std (aligned with existing EWMA system)
        yz_mean = yz.ewm(span=self.state_window, adjust=False).mean()
        yz_std = yz.ewm(span=self.state_window, adjust=False).std()

        z_score = (yz - yz_mean) / yz_std.replace(0, np.nan)

        # Warmup protection: early values may be NaN or unstable
        current_z = z_score.iloc[-1]
        if pd.isna(current_z) or not np.isfinite(current_z):
            current_z = 0.0  # Neutral during warmup

        # 3. Expansion rate: fast/slow EMA differential (robust)
        yz_fast = yz.ewm(span=self.trend_fast, adjust=False).mean()
        yz_slow = yz.ewm(span=self.trend_slow, adjust=False).mean()
        expansion = (yz_fast - yz_slow) / yz_slow.replace(0, np.nan)

        current_expansion = expansion.iloc[-1]
        if pd.isna(current_expansion) or not np.isfinite(current_expansion):
            current_expansion = 0.0

        # 4. ATR for stop-loss (utility, not an estimator)
        atr_series = calc_atr(df, self.period)
        current_atr = atr_series.iloc[-1]
        if pd.isna(current_atr):
            current_atr = 0.0

        # 5. State classification (informational only)
        state = self._classify_state(current_z)

        return VolatilityState(
            yz=float(current_yz) if not pd.isna(current_yz) else 0.0,
            z_score=float(current_z),
            state=state,
            expansion_rate=float(current_expansion),
            atr=float(current_atr),
        )

    def _classify_state(self, z: float) -> str:
        """
        Classify volatility regime from z-score.

        NOTE: This is for LOGGING/DB only. All downstream layers must use
        the continuous z_score value, NOT this string. Using if-elif on
        state would reintroduce the cliff-effect problem.
        """
        if z >= self.extreme_z:
            return "EXTREME"
        elif z >= self.high_z:
            return "HIGH"
        elif z <= self.calm_z:
            return "CALM"
        else:
            return "NORMAL"

    def _fallback_state(self, df: pd.DataFrame) -> VolatilityState:
        """Return neutral state when insufficient data."""
        price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        atr_val = 0.0
        if len(df) > 2:
            atr_series = calc_atr(df, min(len(df) - 1, 14))
            atr_val = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

        return VolatilityState(
            yz=atr_val / price if price > 0 else 0.0,
            z_score=0.0,
            state="NORMAL",
            expansion_rate=0.0,
            atr=atr_val,
        )

    def reset(self):
        """Reset persistent state (call on contract rollover)."""
        pass  # No persistent state in z-score version
