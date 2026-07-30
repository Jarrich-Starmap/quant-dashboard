"""反馈优化器：滑动窗口冷却期管理 + reward 计算。"""

import math
from typing import Optional

import numpy as np


class FeedbackOptimizer:
    """单品种反馈层。"""

    def __init__(self, symbol: str, params: dict):
        self.symbol = symbol
        self.base_alpha = params["base_alpha"]
        self.max_errors = params["max_errors_before_cooldown"]
        self.cooldown_periods = params["cooldown_periods"]
        self.recovery_scale = params["recovery_position_scale"]
        self.risk_per_trade = params["risk_per_trade"]
        self.decay = params["decay_factor"]

        self.error_count = 0
        self.cool_remaining = 0
        self.recovery_correct_streak = 0

    def load_state(self, state: dict):
        self.error_count = state.get("error_count", 0)
        self.cool_remaining = state.get("cool_remaining", 0)
        self.recovery_correct_streak = state.get("recovery_correct_streak", 0)

    def to_state_dict(self) -> dict:
        return {
            "error_count": self.error_count,
            "cool_remaining": self.cool_remaining,
            "recovery_correct_streak": self.recovery_correct_streak,
        }

    def is_in_cooldown(self) -> bool:
        return self.cool_remaining > 0

    def get_position_multiplier(self) -> float:
        """获取当前仓位乘数。"""
        if self.cool_remaining > 0:
            return 0.0
        if self.recovery_correct_streak >= 3:
            return 1.0
        if self.error_count >= self.max_errors and self.recovery_correct_streak < 3:
            return self.recovery_scale
        return 1.0

    def advance_cycle(self):
        """推进冷却计数（即使不交易也调用）。"""
        if self.cool_remaining > 0:
            self.cool_remaining -= 1

    def update(self, reward: float):
        """交易完成后更新冷却/错误状态。EWMA 更新由 main.py 调用 ewma_tracker.update 完成。"""
        if reward > 0:
            self.error_count = 0
            if self.cool_remaining <= 0:
                self.recovery_correct_streak = min(3, self.recovery_correct_streak + 1)
        else:
            self.error_count += 1
            self.recovery_correct_streak = 0
            if self.error_count >= self.max_errors:
                self.cool_remaining = self.cooldown_periods


def compute_reward(pnl: float, entry_price: float, contract_multiplier: float,
                   position_size: float, risk_per_trade: float = 0.02) -> float:
    """
    reward = tanh(pnl / (risk_capital × risk_per_trade))
    risk_capital = entry_price × contract_multiplier × position_size
    risk_per_trade 使 reward 对盈亏百分比更敏感（2% 盈利 → tanh(1.0) ≈ 0.76）。
    """
    risk_capital = entry_price * contract_multiplier * position_size
    if risk_capital <= 0:
        return 0.0
    pnl_pct = pnl / risk_capital
    return math.tanh(pnl_pct / risk_per_trade)
