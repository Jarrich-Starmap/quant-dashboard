"""
Volatility module — Adaptive Volatility Indicator (AVI), 方案B'.

Multi-estimator design was simplified to a SINGLE estimator (Yang-Zhang)
+ z-score state assessment + continuous-sigmoid risk control.

This module is designed to be ADDITIVE to the existing quant-trader system:
  - It is a standalone third signal source (alongside tech + sentiment).
  - All integration points in the existing code are config-gated, so when
    volatility is disabled the system behaves EXACTLY as before.

Public API:
  AVI, VolatilityState        — volatility state calculator
  VolatilitySignal            — vol_score in [-1, 1] (Reversal + Expansion)
  calc_vol_position_multiplier — position sizing multiplier from z-score
  calc_slippage_penalty       — slippage multiplier from z-score
  calc_dynamic_stop_loss      — ATR × z-score dynamic stop
  adjust_ewma_reward          — volatility-weighted PnL reward
  calc_yang_zhang, calc_atr    — low-level estimators
"""

from .estimators import calc_yang_zhang, calc_atr
from .avi import AVI, VolatilityState
from .signal import VolatilitySignal
from .integration import (
    volatility_penalty,
    calc_vol_position_multiplier,
    calc_dynamic_stop_loss,
    calc_dynamic_trailing_stop,
    adjust_ewma_reward,
    calc_slippage_penalty,
)

__all__ = [
    "calc_yang_zhang",
    "calc_atr",
    "AVI",
    "VolatilityState",
    "VolatilitySignal",
    "volatility_penalty",
    "calc_vol_position_multiplier",
    "calc_dynamic_stop_loss",
    "calc_dynamic_trailing_stop",
    "adjust_ewma_reward",
    "calc_slippage_penalty",
]
