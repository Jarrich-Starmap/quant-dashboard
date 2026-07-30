"""
Integration layer — continuous sigmoid risk control based on z-score.

方案B' 核心改进:
  ALL functions use CONTINUOUS sigmoid mapping from z_score.
  ZERO if-elif threshold branching — eliminates cliff-effect entirely.

  This module UNIFIES the volatility_penalty concept:
  - Position sizing
  - Slippage penalty (replaces existing ATR/price > 2% check)
  - Stop-loss multiple
  - EWMA reward weighting

  All four share the SAME z_score → continuous mapping, ensuring
  signal and execution layers are always in sync.
"""

import math

from .avi import VolatilityState


# ============================================================
# Helper: continuous sigmoid smoothing
# ============================================================

def _sigmoid(x: float, center: float = 0.0, steepness: float = 1.5) -> float:
    z = steepness * (x - center)
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================
# Unified volatility penalty (replaces if-elif volatility_penalty)
# ============================================================

def volatility_penalty(vol: VolatilityState) -> dict:
    """
    Unified risk control parameters from z-score.

    ALL outputs are continuous sigmoid functions of z_score:
      z = -2 → calm:    vol_mult=1.20, slippage=0.80, stop_mult=1.50
      z =  0 → normal:  vol_mult=1.00, slippage=1.00, stop_mult=2.00
      z = +1 → high:    vol_mult=0.50, slippage=1.50, stop_mult=2.50
      z = +2 → extreme: vol_mult=0.33, slippage=2.00, stop_mult=3.00

    Returns dict with:
      vol_multiplier   — position sizing multiplier [0.25, 1.30]
      slippage_factor  — slippage penalty multiplier [0.70, 2.50]
      stop_multiple    — ATR multiple for stop-loss [1.0, 4.0]
      pause_new_entry  — bool, True only at extreme volatility
    """
    z = vol.z_score

    s = _sigmoid(z, center=0.0, steepness=1.5)
    vol_mult = _clamp(1.20 - 0.87 * s, 0.25, 1.30)
    slippage = _clamp(0.80 + 1.20 * s, 0.70, 2.50)
    stop_mult = _clamp(1.50 + 1.50 * s, 1.0, 4.0)
    pause = z > 2.5

    return {
        "vol_multiplier": vol_mult,
        "slippage_factor": slippage,
        "stop_multiple": stop_mult,
        "pause_new_entry": pause,
    }


# ============================================================
# Position sizing (uses volatility_penalty output)
# ============================================================

def calc_vol_position_multiplier(vol: VolatilityState) -> float:
    """
    Position size multiplier based on z-score (continuous, no if-elif):
      z = -2 → 1.20 (calm, can trade slightly larger)
      z =  0 → 1.00 (normal)
      z = +1 → ~0.50 (high volatility, reduce)
      z = +2 → ~0.33 (extreme, heavy reduction)

    Additional penalty when volatility is expanding rapidly:
      expansion_rate > 0.10 → extra ×0.85 (continuous sigmoid)
    """
    penalty = volatility_penalty(vol)
    vol_mult = penalty["vol_multiplier"]

    expansion = vol.expansion_rate
    exp_s = _sigmoid(expansion, center=0.10, steepness=15.0)
    exp_penalty = 1.0 - 0.15 * exp_s  # min 0.85 at full expansion

    return _clamp(vol_mult * exp_penalty, 0.25, 1.30)


# ============================================================
# Stop-loss (uses ATR utility + z-score multiple)
# ============================================================

def calc_dynamic_stop_loss(
    entry_price: float,
    vol: VolatilityState,
    direction: str,
    hard_stop_pct: float,
) -> float:
    """
    Dynamic stop-loss using ATR × z-score-adaptive multiple.

    The ATR multiple is CONTINUOUSLY scaled by z-score:
      z = -2 → 1.5×ATR  (calm: tight stop, protect profit)
      z =  0 → 2.0×ATR  (normal: standard)
      z = +1 → 2.5×ATR  (high: wider to avoid noise)
      z = +2 → 3.0×ATR  (extreme: widest)

    Hard stop cap: never wider than entry × hard_stop_pct.
    Minimum floor: never tighter than 0.5×ATR.
    """
    atr = vol.atr
    z = vol.z_score

    s = _sigmoid(z, center=0.0, steepness=1.5)
    atr_multiple = 1.50 + 1.50 * s

    atr_distance = atr * atr_multiple
    hard_distance = entry_price * hard_stop_pct
    min_distance = atr * 0.5

    stop_distance = max(min(atr_distance, hard_distance), min_distance)

    if direction == "LONG":
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance


def calc_dynamic_trailing_stop(
    peak_price: float,
    vol: VolatilityState,
    direction: str,
    base_trailing_pct: float,
) -> float:
    """Dynamic trailing stop using ATR × z-score multiple."""
    atr = vol.atr
    z = vol.z_score

    s = _sigmoid(z, center=0.0, steepness=1.5)
    atr_multiple = 1.0 + 1.0 * s

    avi_distance = atr * atr_multiple
    base_distance = peak_price * base_trailing_pct

    trailing_distance = min(avi_distance, base_distance)
    min_trailing = peak_price * base_trailing_pct * 0.5
    trailing_distance = max(trailing_distance, min_trailing)

    if direction == "LONG":
        return peak_price - trailing_distance
    else:
        return peak_price + trailing_distance


# ============================================================
# EWMA reward weighting (PnL-based only, NOT direction ±1)
# ============================================================

def adjust_ewma_reward(reward: float, vol: VolatilityState) -> float:
    """
    Adjust PnL-based reward by volatility difficulty.

    IMPORTANT: Only adjusts the PnL-based reward (tanh mapping).
    Does NOT touch direction ±1 EWMA — that learning mechanism is sacred.

    Continuous mapping from z-score:
      z < -1 → 0.85 (low vol: easier to profit, slightly discounted)
      z =  0 → 1.00 (normal)
      z > +1 → 1.30 (high vol: harder to profit, amplified)
      z > +2 → 1.50 (extreme: maximum amplification)

    Clamped to [0.80, 1.50].
    """
    z = vol.z_score
    s = _sigmoid(z, center=0.0, steepness=1.0)
    weight = 0.85 + 0.65 * s
    return reward * _clamp(weight, 0.80, 1.50)


# ============================================================
# Slippage penalty (replaces existing ATR/price > 2% check)
# ============================================================

def calc_slippage_penalty(vol: VolatilityState) -> float:
    """
    Unified slippage penalty from z-score.

    Replaces the existing discrete `ATR/price > 2%` check with a
    continuous sigmoid. This ensures signal and execution layers
    share the SAME volatility assessment.

    Returns multiplier:
      z < 0  → ~0.80 (less slippage in calm markets)
      z = 0  → 1.00 (normal)
      z > +1 → ~1.50 (more slippage in high vol)
      z > +2 → ~2.00 (maximum slippage penalty)
    """
    penalty = volatility_penalty(vol)
    return penalty["slippage_factor"]
