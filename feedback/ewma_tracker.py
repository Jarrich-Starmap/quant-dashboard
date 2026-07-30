"""EWMA 胜率追踪器。维护各指标 + 技术/情绪的 EWMA 值。"""

import numpy as np


class EwmaTracker:
    """单个品种的 EWMA 胜率追踪器。"""

    def __init__(self, decay_factor: float = 0.9):
        self.decay = decay_factor
        self.ewma_rsi = 0.0
        self.ewma_macd = 0.0
        self.ewma_bb = 0.0
        self.ewma_mom = 0.0
        self.ewma_tech = 0.0
        self.ewma_sent = 0.0
        self.ewma_vol = 0.5  # 波动率信号路（中性起点）
        self.ewma_sent_frozen = False

    def load_from_dict(self, state: dict):
        self.ewma_rsi = state.get("ewma_rsi", 0.0)
        self.ewma_macd = state.get("ewma_macd", 0.0)
        self.ewma_bb = state.get("ewma_bb", 0.0)
        self.ewma_mom = state.get("ewma_mom", 0.0)
        self.ewma_tech = state.get("ewma_tech", 0.0)
        self.ewma_sent = state.get("ewma_sent", 0.0)
        self.ewma_vol = state.get("ewma_vol", 0.5)
        self.ewma_sent_frozen = bool(state.get("ewma_sent_frozen", 0))

    def to_dict(self, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "ewma_rsi": self.ewma_rsi,
            "ewma_macd": self.ewma_macd,
            "ewma_bb": self.ewma_bb,
            "ewma_mom": self.ewma_mom,
            "ewma_tech": self.ewma_tech,
            "ewma_sent": self.ewma_sent,
            "ewma_vol": self.ewma_vol,
            "ewma_sent_frozen": int(self.ewma_sent_frozen),
        }

    def update_indicators(self, rsi_reward: float, macd_reward: float,
                          bb_reward: float, mom_reward: float):
        """基于下一根K线的实际方向，用 ±1 奖励更新四个子指标 EWMA。"""
        self.ewma_rsi = self.ewma_rsi * self.decay + rsi_reward * (1 - self.decay)
        self.ewma_macd = self.ewma_macd * self.decay + macd_reward * (1 - self.decay)
        self.ewma_bb = self.ewma_bb * self.decay + bb_reward * (1 - self.decay)
        self.ewma_mom = self.ewma_mom * self.decay + mom_reward * (1 - self.decay)

    def update_tech(self, tech_score: float, actual_direction: int):
        """技术侧 EWMA 独立更新。tech_score 符号为技术预测方向，actual_direction +1/-1。"""
        predicted = 1 if tech_score > 0 else -1 if tech_score < 0 else 0
        r = 1.0 if predicted == actual_direction else -1.0
        self.ewma_tech = self.ewma_tech * self.decay + r * (1 - self.decay)

    def update_sent(self, sentiment_score: float, actual_direction: int):
        """情绪侧 EWMA 独立更新。sentiment_score 符号为情绪预测方向。"""
        if self.ewma_sent_frozen:
            return
        predicted = 1 if sentiment_score > 0 else -1 if sentiment_score < 0 else 0
        r = 1.0 if predicted == actual_direction else -1.0
        self.ewma_sent = self.ewma_sent * self.decay + r * (1 - self.decay)

    def update_vol(self, vol_score: float, actual_direction: int):
        """波动率信号侧 EWMA 独立更新（方案B' 第三路）。vol_score 符号为预测方向。"""
        predicted = 1 if vol_score > 0 else -1 if vol_score < 0 else 0
        r = 1.0 if predicted == actual_direction else -1.0
        self.ewma_vol = self.ewma_vol * self.decay + r * (1 - self.decay)

