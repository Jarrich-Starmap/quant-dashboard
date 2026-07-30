"""仓位计算与保证金检查。"""


def compute_position_size(final_score: float, base_lot: int,
                          position_multiplier: float,
                          entry_threshold: float = 0.2) -> float:
    """
    final_score 绝对值 ≤ entry_threshold 时不持仓。

    返回 0 表示不持仓。
    """
    if abs(final_score) <= entry_threshold:
        return 0.0

    size = base_lot * abs(final_score) * position_multiplier
    return max(0.0, size)


def get_direction(final_score: float,
                  entry_threshold: float = 0.2) -> str:
    """
    final_score > entry_threshold → LONG
    final_score < -entry_threshold → SHORT
    else → NEUTRAL
    """
    if final_score > entry_threshold:
        return "LONG"
    elif final_score < -entry_threshold:
        return "SHORT"
    return "NEUTRAL"


def calc_required_margin(symbol: str, price: float, lots: float, symbol_cfg: dict) -> float:
    """计算开仓所需保证金。"""
    contract_value = price * symbol_cfg["contract_multiplier"] * lots
    margin_rate = symbol_cfg.get("margin_rate", 0.10)
    return contract_value * margin_rate


def calc_used_margin(current_positions: dict[str, dict], config_symbols: dict) -> float:
    """计算当前所有持仓已占用的保证金总额。"""
    total = 0.0
    for sym, pos in current_positions.items():
        if pos.get("size", 0) > 0 and sym in config_symbols:
            cfg = config_symbols[sym]
            contract_value = pos["entry_price"] * cfg["contract_multiplier"] * pos["size"]
            total += contract_value * cfg.get("margin_rate", 0.10)
    return total


def can_open_position(symbol: str, price: float, lots: float,
                      account_equity: float,
                      current_positions: dict[str, dict],
                      config_symbols: dict) -> tuple[bool, str]:
    """
    检查账户是否有足够保证金开仓。

    返回: (can_open, reason)
    """
    if lots <= 0:
        return True, ""

    cfg = config_symbols.get(symbol, {})
    required = calc_required_margin(symbol, price, lots, cfg)
    used = calc_used_margin(current_positions, config_symbols)
    available = account_equity - used

    if required > available:
        return False, (
            f"保证金不足: 需 {required:.2f} / 可用 {available:.2f} "
            f"(已占用 {used:.2f}, 总权益 {account_equity:.2f})"
        )
    return True, ""
