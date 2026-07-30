"""综合评分引擎。动态技术内部权重 + 技术/情绪融合。"""

import numpy as np


def softmax(x: list[float], temperature: float = 0.5) -> list[float]:
    x = np.array(x, dtype=float)
    x_adj = (x - np.max(x)) / temperature  # 数值稳定
    e = np.exp(x_adj)
    return (e / e.sum()).tolist()


def compute_score(
    rsi_signal: float, macd_signal: float, bb_signal: float, mom_signal: float,
    sentiment_score: float,
    ewma_rsi: float, ewma_macd: float, ewma_bb: float, ewma_mom: float,
    ewma_tech: float, ewma_sent: float,
    sentiment_valid: bool,
    vol_score: float = 0.0,
    ewma_vol: float = 0.5,
    volatility_enabled: bool = False,
    vol_epsilon: float = 0.02,
) -> tuple[float, float, float, list[float]]:
    """
    计算综合评分。

    返回: (final_score, tech_score, alpha, [w_rsi, w_macd, w_bb, w_mom])

    波动率融合（方案B'）：
      当 volatility_enabled=True 且 vol_score 显著时，使用三路 softmax 融合
      weights = softmax([ewma_tech, ewma_sent, ewma_vol])，final = Σ w·score。
      - 情绪无效时自动剔除 sent 路（仅 tech + vol 两路）。
      - vol_score 接近 0（弃权）时自动剔除 vol 路，退化为 tech/sent 两路，
        但用 softmax 而非原 alpha，权重更平滑。
      - volatility_enabled=False 时，行为与改动前完全一致（alpha 二路融合）。
    """
    # 技术指标内部 softmax 动态权重（不变）
    ewma_list = [ewma_rsi, ewma_macd, ewma_bb, ewma_mom]
    min_ewma = min(ewma_list)
    shifted = [e - min_ewma + 1e-6 for e in ewma_list]
    weights = softmax(shifted, temperature=0.5)

    tech_score = (
        weights[0] * rsi_signal
        + weights[1] * macd_signal
        + weights[2] * bb_signal
        + weights[3] * mom_signal
    )
    tech_score = max(-1.0, min(1.0, tech_score))

    if not volatility_enabled:
        # 兼容路径：与改动前完全一致
        if not sentiment_valid:
            alpha = 1.0
        else:
            alpha = 1.0 / (1.0 + np.exp(-(ewma_tech - ewma_sent)))
            alpha = max(0.2, min(0.8, alpha))

        final_score = alpha * tech_score + (1.0 - alpha) * sentiment_score
        final_score = max(-1.0, min(1.0, final_score))
        return final_score, tech_score, alpha, weights

    # ---- 波动率融合（方案B'）----
    vol_active = abs(vol_score) >= vol_epsilon

    if not vol_active:
        # vol 弃权（波动率平静）：完全退化为原 alpha 二路融合。
        # 即使启用了波动率模块，只要 vol_score≈0，行为与原系统逐位一致。
        if not sentiment_valid:
            alpha = 1.0
        else:
            alpha = 1.0 / (1.0 + np.exp(-(ewma_tech - ewma_sent)))
            alpha = max(0.2, min(0.8, alpha))
        final_score = alpha * tech_score + (1.0 - alpha) * sentiment_score
        final_score = max(-1.0, min(1.0, final_score))
        return final_score, tech_score, alpha, weights

    # vol 生效：三路 softmax（tech + sent + vol），弃权即剔除避免稀释
    active_ewma: list[float] = []
    active_score: list[float] = []
    if sentiment_valid:
        active_ewma.append(ewma_sent)
        active_score.append(sentiment_score)
    active_ewma.append(ewma_tech)
    active_score.append(tech_score)
    active_ewma.append(ewma_vol)
    active_score.append(vol_score)

    min_a = min(active_ewma)
    a_shifted = [e - min_a + 1e-6 for e in active_ewma]
    a_weights = softmax(a_shifted, temperature=0.5)

    final_score = float(sum(w * s for w, s in zip(a_weights, active_score)))
    final_score = max(-1.0, min(1.0, final_score))

    # alpha 等价量（用于日志/快照）：tech 在 active 集合中的权重
    tech_idx = 1 if sentiment_valid else 0
    alpha = a_weights[tech_idx]

    return final_score, tech_score, alpha, weights
