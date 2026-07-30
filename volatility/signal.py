"""
Volatility directional signal — 方案B' 精简版.

Only TWO modes (squeeze removed — BB bandwidth already covers it):

  1. Expansion follow (expansion_rate > threshold):
     Volatility expanding rapidly → momentum building.
     Direction = price trend (tanh of price vs EMA, ATR-scaled).
     Base confidence = 0.20 (lowered: expansion often mid-trend).
     Volume confirmation boosts up to ~0.30.

  2. Extreme reversal (z_score > reversal_z):
     Volatility at extreme → panic/climax → mean reversion likely.
     Direction = FADE price trend.
     Confidence = 0.25 (lowered from 0.40: let EWMA learn the weight).

  3. Normal (everything else):
     No strong volatility-based opinion.
     vol_score ≈ 0 (defer to tech/sent)

Why squeeze was removed:
  BB bandwidth ratio already does "compression → breakout signal enhancement".
  AVI squeeze would be signal overlap, not information gain.
  Reversal is AVI's unique alpha — BB says "low vol", AVI says "extreme vol → reverse".

All transitions are continuous sigmoid functions — no if-elif cliffs.
The two modes can overlap and blend smoothly.

Usage:
    vol_signal = VolatilitySignal(config)
    score = vol_signal.calculate(df, vol_state)
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

from .avi import VolatilityState


# ============================================================
# Helpers
# ============================================================

def _sigmoid(x: float, center: float = 0.5, steepness: float = 6.0) -> float:
    """Smooth sigmoid. Returns [0, 1]."""
    z = steepness * (x - center)
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================
# VolatilitySignal
# ============================================================

class VolatilitySignal:
    """Produces a directional score [-1, 1] from volatility state + price data."""

    def __init__(self, config: Optional[dict] = None):
        c = config or {}

        # Direction
        self.direction_ma_period = c.get("direction_ma_period", 20)
        self.direction_scale_atr = c.get("direction_scale_atr", 2.0)

        # Expansion follow
        self.expansion_strength = c.get("expansion_strength", 0.20)
        self.expansion_vol_boost = c.get("expansion_vol_boost", 0.10)
        self.expansion_center = c.get("expansion_center", 0.12)
        self.expansion_steepness = c.get("expansion_steepness", 20.0)
        self.volume_ma_period = c.get("volume_ma_period", 20)
        self.volume_confirm_ratio = c.get("volume_confirm_ratio", 1.3)

        # Extreme reversal
        self.reversal_strength = c.get("reversal_strength", 0.25)
        self.reversal_z = c.get("reversal_z", 2.0)
        self.reversal_steepness = c.get("reversal_steepness", 3.0)

        self.dead_zone = c.get("dead_zone", 0.05)

    # ---- Volume confirmation for expansion mode ----

    def _volume_boost(self, df: pd.DataFrame) -> float:
        """
        Compute volume-based confidence boost for expansion mode.
        Returns a value in [0, expansion_vol_boost].
        Zero when volume data is unavailable or below confirmation threshold.
        """
        if "volume" not in df.columns:
            return 0.0
        if len(df) < self.volume_ma_period:
            return 0.0

        vol_ma = float(df["volume"].iloc[-self.volume_ma_period:].mean())
        if vol_ma <= 0:
            return 0.0

        current_vol = float(df["volume"].iloc[-1])
        vol_ratio = current_vol / vol_ma

        boost = self.expansion_vol_boost * _sigmoid(
            vol_ratio,
            center=self.volume_confirm_ratio,
            steepness=5.0,
        )
        return boost

    # ---- Main calculation ----

    def calculate(self, df: pd.DataFrame, vol: VolatilityState) -> float:
        """
        Calculate volatility-based directional score.

        Args:
            df: OHLCV DataFrame, needs at least direction_ma_period rows
            vol: VolatilityState from AVI.calculate()

        Returns:
            vol_score in [-1, 1]. Near zero in normal volatility.
        """
        z_score = vol.z_score
        expansion_rate = vol.expansion_rate
        atr = vol.atr if vol.atr > 0 else 1.0

        # ---- 1. EMA-based direction ----
        price = float(df["close"].iloc[-1])
        ema = float(
            df["close"]
            .ewm(span=self.direction_ma_period, adjust=False)
            .mean()
            .iloc[-1]
        )
        direction = math.tanh((price - ema) / (atr * self.direction_scale_atr))

        # ---- 2. Expansion follow confidence ----
        vol_boost = self._volume_boost(df)
        expansion_conf = (self.expansion_strength + vol_boost) * _sigmoid(
            expansion_rate,
            center=self.expansion_center,
            steepness=self.expansion_steepness,
        )

        # ---- 3. Extreme reversal confidence ----
        reversal_conf = self.reversal_strength * _sigmoid(
            z_score,
            center=self.reversal_z,
            steepness=self.reversal_steepness,
        )

        # ---- 4. Combine ----
        # Expansion: follow price direction
        # Reversal: fade price direction (opposite)
        vol_score = direction * (expansion_conf - reversal_conf)

        # ---- 5. Dead zone ----
        if abs(vol_score) < self.dead_zone:
            return 0.0

        return _clamp(vol_score, -1.0, 1.0)

    def explain(self, df: pd.DataFrame, vol: VolatilityState) -> dict:
        """Return detailed breakdown for logging/debugging."""
        z_score = vol.z_score
        expansion_rate = vol.expansion_rate
        atr = vol.atr if vol.atr > 0 else 1.0

        price = float(df["close"].iloc[-1])
        ema = float(
            df["close"]
            .ewm(span=self.direction_ma_period, adjust=False)
            .mean()
            .iloc[-1]
        )
        direction = math.tanh((price - ema) / (atr * self.direction_scale_atr))

        vol_boost = self._volume_boost(df)
        expansion_conf = (self.expansion_strength + vol_boost) * _sigmoid(
            expansion_rate,
            center=self.expansion_center,
            steepness=self.expansion_steepness,
        )
        reversal_conf = self.reversal_strength * _sigmoid(
            z_score,
            center=self.reversal_z,
            steepness=self.reversal_steepness,
        )

        vol_score = direction * (expansion_conf - reversal_conf)
        if abs(vol_score) < self.dead_zone:
            vol_score = 0.0
        vol_score = _clamp(vol_score, -1.0, 1.0)

        if vol_score == 0.0:
            mode = "NEUTRAL"
        elif reversal_conf > expansion_conf:
            mode = "REVERSAL"
        elif expansion_conf > 0.01:
            mode = "EXPANSION_FOLLOW"
        else:
            mode = "NEUTRAL"

        if "volume" in df.columns and len(df) >= self.volume_ma_period:
            vol_ma = float(df["volume"].iloc[-self.volume_ma_period:].mean())
            current_vol = float(df["volume"].iloc[-1])
            vol_ratio = round(current_vol / vol_ma, 2) if vol_ma > 0 else None
        else:
            vol_ratio = None

        return {
            "vol_score": round(vol_score, 4),
            "mode": mode,
            "direction": round(direction, 4),
            "price": round(price, 2),
            "ema": round(ema, 2),
            "atr": round(atr, 2),
            "expansion_conf": round(expansion_conf, 4),
            "expansion_base": round(self.expansion_strength, 4),
            "vol_boost": round(vol_boost, 4),
            "vol_ratio": vol_ratio,
            "reversal_conf": round(reversal_conf, 4),
            "z_score": round(z_score, 4),
            "expansion_rate": round(expansion_rate, 4),
        }
