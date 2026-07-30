"""模拟交易执行引擎。"""

import logging
from datetime import datetime
from typing import Optional

from db.models import insert_trade, update_trade_close, get_conn
from volatility import calc_dynamic_stop_loss

logger = logging.getLogger(__name__)


def calc_commission(symbol_cfg: dict, price: float, lots: float) -> float:
    """计算单边手续费。"""
    if symbol_cfg.get("commission_mode") == "fixed":
        return symbol_cfg.get("commission_fixed", 0) * lots
    elif symbol_cfg.get("commission_mode") == "ratio":
        contract_value = price * symbol_cfg["contract_multiplier"] * lots
        return contract_value * symbol_cfg.get("commission_ratio", 0)
    return 0.0


def apply_slippage(signal_price: float, direction: str, symbol_cfg: dict,
                   atr: float = None, vol_slip_factor: float = None) -> float:
    """对成交价施加不利方向的滑点。"""
    rate = symbol_cfg.get("slippage_rate", 0)
    if rate == 0:
        return signal_price

    base = signal_price * rate

    # 高波动惩罚（离散 ATR 检查，已不再由外部触发，保留兼容）
    if atr:
        norm_vol = atr / signal_price
        if norm_vol > 0.02:
            penalty = symbol_cfg.get("volatility_penalty", 0.5)
            base *= (1 + penalty * (norm_vol / 0.02 - 1))

    # 方案B'：统一的波动率滑点惩罚（连续 sigmoid），由 main 计算后传入
    if vol_slip_factor is not None:
        base *= vol_slip_factor

    if direction == "LONG":
        return round(signal_price + base, 2)
    else:  # SHORT
        return round(signal_price - base, 2)


class SimTrader:
    """模拟交易器。同一品种同一时间仅持有一个方向的仓位。"""

    def __init__(self, position_states: dict[str, dict] | None = None):
        """
        position_states: {symbol: {pos_direction, pos_entry, pos_size}} from ewma_state
        """
        self.positions: dict[str, dict] = {}
        if position_states:
            for sym, state in position_states.items():
                if state.get("pos_direction") and state.get("pos_size", 0) > 0:
                    entry = state["pos_entry"] or 0.0
                    self.positions[sym] = {
                        "direction": state["pos_direction"],
                        "entry_price": entry,
                        "size": state["pos_size"],
                        "open_time": state.get("pos_open_time"),
                        "peak_price": entry,  # 开仓时峰值 = 入场价
                    }
        self.prev_positions: dict[str, dict] = {}

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions and self.positions[symbol]["size"] > 0

    def update_peak(self, symbol: str, current_price: float):
        """更新持仓期间的峰值价格（用于移动止损）。"""
        if not self.has_position(symbol):
            return
        pos = self.positions[symbol]
        if pos["direction"] == "LONG":
            pos["peak_price"] = max(pos.get("peak_price", current_price), current_price)
        else:  # SHORT
            pos["peak_price"] = min(pos.get("peak_price", current_price), current_price)

    def check_stop_loss(self, symbol: str, current_price: float,
                        symbol_cfg: dict,
                        vol_state=None) -> tuple[Optional[str], Optional[float]]:
        """
        检查止损条件。在每次信号计算前调用。

        返回: (trigger_type, exit_price) 或 (None, None)
        trigger_type: "HARD_STOP" / "BREAKEVEN" / "TRAILING"

        vol_state: 方案B' 波动率状态对象。当配置了 volatility.dynamic_stop
                  且传入 vol_state 时，硬止损距离改用 AVI 动态止损
                  （min(2×ATR, 百分比硬止损)），其余逻辑不变。
        """
        if not self.has_position(symbol):
            return None, None

        pos = self.positions[symbol]
        entry = pos["entry_price"]
        hard_stop = symbol_cfg.get("hard_stop_loss", 0.04)
        breakeven_trigger = symbol_cfg.get("breakeven_trigger", 0.04)
        trailing_pct = symbol_cfg.get("trailing_stop_pct", 0.02)

        if pos["direction"] == "LONG":
            loss_pct = (entry - current_price) / entry
            profit_pct = (current_price - entry) / entry
        else:  # SHORT
            loss_pct = (current_price - entry) / entry
            profit_pct = (entry - current_price) / entry

        # 方案B'：动态硬止损（可选，默认关闭，保持原百分比止损不变）
        dynamic_cfg = symbol_cfg.get("volatility", {})
        use_dynamic = vol_state is not None and dynamic_cfg.get("dynamic_stop", False)
        if use_dynamic:
            dyn_price = calc_dynamic_stop_loss(
                entry, vol_state, pos["direction"], hard_stop
            )
            if pos["direction"] == "LONG" and current_price <= dyn_price:
                return "HARD_STOP", current_price
            if pos["direction"] == "SHORT" and current_price >= dyn_price:
                return "HARD_STOP", current_price
        else:
            # 硬止损（原逻辑）
            if loss_pct >= hard_stop:
                return "HARD_STOP", current_price

        # 保本 / 移动止损（仅当浮盈达到触发阈值后生效）
        if profit_pct >= breakeven_trigger:
            # 保本：回撤到入场价
            if pos["direction"] == "LONG" and current_price <= entry:
                return "BREAKEVEN", entry
            if pos["direction"] == "SHORT" and current_price >= entry:
                return "BREAKEVEN", entry

            # 移动止损：从峰值回撤超过 trailing_stop_pct
            peak = pos.get("peak_price", entry)
            if pos["direction"] == "LONG":
                drawdown = (peak - current_price) / peak
            else:
                drawdown = (current_price - peak) / peak
            if drawdown >= trailing_pct:
                return "TRAILING", current_price

        return None, None

    def force_close(self, symbol: str, exit_price: float, cycle_time: datetime,
                    contract_multiplier: float, symbol_cfg: dict,
                    contract_code: str = "", reason: str = "") -> Optional[dict]:
        """
        强制平仓（止损触发时调用）。扣除双边手续费。
        """
        if not self.has_position(symbol):
            return None

        pos = self.positions[symbol]

        # 计算理论盈亏
        if pos["direction"] == "LONG":
            pnl = (exit_price - pos["entry_price"]) * pos["size"] * contract_multiplier
        else:  # SHORT
            pnl = (pos["entry_price"] - exit_price) * pos["size"] * contract_multiplier

        # 扣除双边手续费
        open_commission = calc_commission(symbol_cfg, pos["entry_price"], pos["size"])
        close_commission = calc_commission(symbol_cfg, exit_price, pos["size"])
        total_commission = open_commission + close_commission
        pnl -= total_commission

        risk_capital = pos["entry_price"] * contract_multiplier * pos["size"]
        pnl_pct = pnl / risk_capital if risk_capital > 0 else 0.0

        update_trade_close(
            symbol=symbol, direction=pos["direction"],
            exit_price=exit_price,
            position_size=pos["size"], pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 6), close_time=cycle_time,
            contract_code=contract_code,
        )

        logger.info(
            f"  [{symbol}] 止损触发 [{reason}] | "
            f"方向={pos['direction']} 入场={pos['entry_price']} 离场={exit_price} | "
            f"盈亏={pnl:.4f} (手续费={total_commission:.4f})"
        )

        result = {"pnl": pnl, "pnl_pct": pnl_pct, "direction": pos["direction"],
                   "stop_type": reason}
        del self.positions[symbol]
        return result

    def execute(self, symbol: str, direction: str, position_size: float,
                entry_price: float, exit_price: float, cycle_time: datetime,
                contract_multiplier: float, symbol_cfg: dict = None,
                contract_code: str = "", vol_slip_factor: float = None) -> Optional[dict]:
        """
        执行模拟交易。

        返回: {"pnl": float, "pnl_pct": float, "direction": str} 或 None（无交易）
        """
        # 平仓
        if self.has_position(symbol):
            pos = self.positions[symbol]
            # 方向相反或 NEUTRAL → 平仓
            if direction != pos["direction"]:
                if pos["direction"] == "LONG":
                    pnl = (exit_price - pos["entry_price"]) * pos["size"] * contract_multiplier
                else:
                    pnl = (pos["entry_price"] - exit_price) * pos["size"] * contract_multiplier

                # 扣除双边手续费
                if symbol_cfg:
                    open_comm = calc_commission(symbol_cfg, pos["entry_price"], pos["size"])
                    close_comm = calc_commission(symbol_cfg, exit_price, pos["size"])
                    pnl -= (open_comm + close_comm)

                risk_capital = pos["entry_price"] * contract_multiplier * pos["size"]
                pnl_pct = pnl / risk_capital if risk_capital > 0 else 0.0

                update_trade_close(
                    symbol=symbol, direction=pos["direction"],
                    exit_price=exit_price,
                    position_size=pos["size"], pnl=round(pnl, 4),
                    pnl_pct=round(pnl_pct, 6), close_time=cycle_time,
                    contract_code=contract_code,
                )
                result = {"pnl": pnl, "pnl_pct": pnl_pct, "direction": pos["direction"]}
                del self.positions[symbol]

                # 开新仓（若需要，应用滑点）
                if direction != "NEUTRAL" and position_size > 0:
                    slipped_entry = apply_slippage(entry_price, direction, symbol_cfg,
                                                  vol_slip_factor=vol_slip_factor) if symbol_cfg else entry_price
                    self.positions[symbol] = {
                        "direction": direction,
                        "entry_price": slipped_entry,
                        "size": position_size,
                        "open_time": cycle_time,
                        "peak_price": slipped_entry,
                    }
                    insert_trade(
                        symbol=symbol, direction=direction,
                        entry_price=slipped_entry, exit_price=0,
                        position_size=position_size, pnl=0,
                        pnl_pct=0, cycle_time=cycle_time,
                        open_time=cycle_time, close_time=None,
                        contract_code=contract_code,
                    )
                    get_conn().execute(
                        "UPDATE ewma_state SET pos_direction=?, pos_entry=?, pos_size=?, pos_open_time=? WHERE symbol=?",
                        (direction, slipped_entry, position_size, cycle_time, symbol),
                    )
                    get_conn().commit()
                return result

            # 方向相同 → 持仓不动，更新峰值
            self.update_peak(symbol, exit_price)
            return None

        # 无持仓 → 开仓
        if direction != "NEUTRAL" and position_size > 0:
            slipped_entry = apply_slippage(entry_price, direction, symbol_cfg,
                                          vol_slip_factor=vol_slip_factor) if symbol_cfg else entry_price
            self.positions[symbol] = {
                "direction": direction,
                "entry_price": slipped_entry,
                "size": position_size,
                "open_time": cycle_time,
                "peak_price": slipped_entry,
            }
            insert_trade(
                symbol=symbol, direction=direction,
                entry_price=slipped_entry, exit_price=0,
                position_size=position_size, pnl=0,
                pnl_pct=0, cycle_time=cycle_time,
                open_time=cycle_time, close_time=None,
                contract_code=contract_code,
            )
            get_conn().execute(
                "UPDATE ewma_state SET pos_direction=?, pos_entry=?, pos_size=?, pos_open_time=? WHERE symbol=?",
                (direction, slipped_entry, position_size, cycle_time, symbol),
            )
            get_conn().commit()
            return None

        return None

    def get_position_state(self) -> dict:
        """Export current positions for DB persistence."""
        result = {}
        for sym, pos in self.positions.items():
            if pos["size"] > 0:
                result[sym] = {
                    "pos_direction": pos["direction"],
                    "pos_entry": pos["entry_price"],
                    "pos_size": pos["size"],
                    "pos_open_time": pos.get("open_time"),
                }
        return result

    def close_all(self, symbol: str, exit_price: float, cycle_time: datetime,
                  contract_multiplier: float, symbol_cfg: dict = None) -> Optional[dict]:
        """强制平仓（换月时调用）。"""
        if not self.has_position(symbol):
            return None
        return self.execute(symbol, "NEUTRAL", 0, exit_price, exit_price,
                            cycle_time, contract_multiplier, symbol_cfg)
