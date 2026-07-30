"""量化模拟交易系统主入口。四层流水线：Data → Signal → Execution → Feedback。"""

import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from db.models import (
    init_db,
    load_ewma_state,
    save_ewma_state,
    insert_snapshot,
    get_cached_klines,
    get_open_trades,
    close_trade_by_id,
    get_conn,
)
from data.fetcher import fetch_and_cache
from data.sentiment import load_sentiment
from data.adapter import resolve_contract_code
from signal.indicators import compute_all
from signal.scorer import compute_score
from feedback.ewma_tracker import EwmaTracker
from feedback.optimizer import FeedbackOptimizer, compute_reward
from executor.position import compute_position_size, get_direction
from executor.trader import SimTrader, apply_slippage, calc_commission
from volatility import (
    AVI,
    VolatilitySignal,
    calc_vol_position_multiplier,
    calc_slippage_penalty,
)
from volatility.integration import adjust_ewma_reward
import pandas as pd
from notify.wecom_notifier import notify_before_trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("quant_trader.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
WARMUP_CYCLES = 20

# 交易时段（北京时间）
TRADING_SESSIONS = [
    ("09:00", "10:15"),
    ("10:30", "11:30"),
    ("13:30", "15:00"),
    ("21:00", "23:59"),   # 夜盘前半段
    ("00:00", "02:30"),   # 夜盘后半段（跨日）
]


def is_trading_time(now: datetime) -> bool:
    """判断当前是否在交易时段内。支持夜盘跨日。"""
    now_str = now.strftime("%H:%M")
    weekday = now.weekday()  # 0=周一, 6=周日
    if weekday == 5 and now.hour >= 3:
        # 周六 03:00 后休市（周五夜盘延续到周六 02:30，由 TRADING_SESSIONS 上限约束）
        return False
    if weekday == 6:
        # 周日全天休市（国内期货无周日夜盘）
        return False
    if weekday == 0 and now.hour < 9:
        # 周一凌晨无夜盘延续（周日晚无夜盘）
        return False
    for start, end in TRADING_SESSIONS:
        if start == "00:00":
            if "00:00" <= now_str <= end:
                return True
        elif start <= now_str <= end:
            return True
    return False


def is_session_end_cycle(now) -> bool:
    """判断当前周期是否为交易时段的最后一分钟（下一分钟不在任何交易时段内）。"""
    from datetime import timedelta
    next_cycle = now + timedelta(minutes=1)
    return is_trading_time(now) and not is_trading_time(next_cycle)


def load_config() -> dict:
    import yaml
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 三级合并：symbol > global > 代码默认
    # global 块仅含 A 类指标/波动率内部常数（全品种相同），symbol 可覆盖任意键
    g = cfg.get("global", {}) or {}
    g_vol = g.get("volatility", {}) or {}
    for sym, scfg in cfg.get("symbols", {}).items():
        scfg_vol = scfg.get("volatility", {}) or {}
        merged = {k: v for k, v in g.items() if k != "volatility"}
        merged.update({k: v for k, v in scfg.items() if k != "volatility"})
        vol = dict(g_vol)
        vol.update(scfg_vol)
        merged["volatility"] = vol
        cfg["symbols"][sym] = merged
    return cfg


def reconcile_and_load_positions(config: dict) -> dict:
    """
    启动自愈 + 以 trades 表为持仓唯一权威来源。

    修复：ewma_state 与 trades 两套账本失同步导致孤儿仓（多空同时持有）。
    - 遍历每个品种，查出所有未平仓记录；
    - 若 >1 笔，保留最新一笔（最大 id），其余按最新入场价平仓并写入 trades；
    - 同步 ewma_state 为保留的那笔（不再以 ewma_state.pos_* 为权威）；
    - 返回 SimTrader 初始化所需的 pos_states。
    """
    pos_states: dict = {}
    for sym, sym_cfg in config["symbols"].items():
        mult = sym_cfg["contract_multiplier"]
        opens = get_open_trades(sym)
        if not opens:
            # 全部已平仓：将 ewma_state 冗余持仓清零，避免 dashboard 显示幽灵持仓
            get_conn().execute(
                "UPDATE ewma_state SET pos_direction='', pos_entry=0, pos_size=0, pos_open_time=NULL WHERE symbol=?",
                (sym,),
            )
            get_conn().commit()
            pos_states[sym] = {"pos_direction": "", "pos_entry": 0.0, "pos_size": 0.0, "pos_open_time": None}
            continue
        if len(opens) > 1:
            latest = opens[-1]
            logger.warning(
                f"[{sym}] 检测到 {len(opens)} 笔未平仓（孤儿仓），启动自愈："
                f"保留 id={latest['id']}，平掉其余 {len(opens) - 1} 笔"
            )
            for t in opens[:-1]:
                size = t["position_size"] or 0.0
                entry = t["entry_price"] or 0.0
                exit_px = latest["entry_price"] or entry
                if exit_px == 0:
                    exit_px = entry
                raw = (exit_px - entry) * size * mult if t["direction"] == "LONG" \
                    else (entry - exit_px) * size * mult
                comm = calc_commission(sym_cfg, entry, size) + calc_commission(sym_cfg, exit_px, size)
                pnl = raw - comm
                risk = entry * mult * size
                pnl_pct = pnl / risk if risk > 0 else 0.0
                close_trade_by_id(
                    t["id"], exit_px, round(pnl, 4), round(pnl_pct, 6),
                    contract_code=t["contract_code"] or "",
                )
                logger.info(
                    f"  [{sym}] 自愈平掉孤儿 id={t['id']} {t['direction']} "
                    f"入场={entry} 离场={exit_px} 盈亏={pnl:.4f}"
                )
        kept = opens[-1]
        # 始终同步 ewma_state 为权威持仓（冗余记录；启动恢复以 trades 表为准），
        # 杜绝两套账本再次分歧。
        get_conn().execute(
            "UPDATE ewma_state SET pos_direction=?, pos_entry=?, pos_size=?, pos_open_time=? WHERE symbol=?",
            (kept["direction"], kept["entry_price"], kept["position_size"],
             kept["open_time"], sym),
        )
        get_conn().commit()
        pos_states[sym] = {
            "pos_direction": kept["direction"],
            "pos_entry": kept["entry_price"],
            "pos_size": kept["position_size"],
            "pos_open_time": kept["open_time"],
        }
    return pos_states


def process_symbol(symbol: str, sym_cfg: dict, global_cfg: dict,
                   sentiment_scores: dict, sentiment_valid: bool,
                   trader: SimTrader, cycle_time: datetime, config: dict) -> dict:
    result = {"symbol": symbol, "traded": False, "pnl": 0.0, "direction": "NEUTRAL", "notes": ""}

    # ---- Data Layer ----
    closes, highs, lows, is_stale, is_rollover = fetch_and_cache(symbol, {**global_cfg, **sym_cfg})
    if closes is None or len(closes) < 40:
        result["notes"] = "insufficient_data"
        return result

    price = float(closes.iloc[-1])

    if is_rollover:
        if trader.has_position(symbol):
            _pos = trader.positions.get(symbol, {})
            _entry = _pos.get("entry_price")
            _dir = _pos.get("direction", "")
            _size = _pos.get("size")
            _close_res = trader.close_all(symbol, price, cycle_time, sym_cfg["contract_multiplier"], sym_cfg)
            notify_before_trade(
                config,
                plan={
                    "event": "FORCE_CLOSE",
                    "symbol": symbol,
                    "direction": _dir,
                    "entry_price": _entry,
                    "exit_price": round(price, 2),
                    "position_size": _size,
                    "pnl": round(_close_res["pnl"], 2) if _close_res else None,
                    "cycle_time": cycle_time.strftime("%Y-%m-%d %H:%M"),
                },
                block=True,
            )
        result["notes"] = "rollover_detected"
        return result

    # ---- Volatility module (方案B', config-gated) ----
    vol_cfg = sym_cfg.get("volatility", {})
    vol_enabled = bool(vol_cfg.get("enabled", False))
    vol_state = None
    vol_score = 0.0
    vol_multiplier = 1.0
    vol_slip_factor = 1.0
    if vol_enabled:
        try:
            _vrows = get_cached_klines(
                symbol,
                limit=max(int(vol_cfg.get("state_window", 60)) + int(vol_cfg.get("period", 14)), 100),
            )
            if _vrows and len(_vrows) >= int(vol_cfg.get("period", 14)) + 1:
                df = pd.DataFrame(_vrows)
                _vstate = AVI(vol_cfg).calculate(df)
                vol_state = _vstate
                vol_score = VolatilitySignal(vol_cfg).calculate(df, _vstate)
                vol_multiplier = calc_vol_position_multiplier(_vstate)
                vol_slip_factor = calc_slippage_penalty(_vstate)
        except Exception as _e:
            logger.warning(f"{symbol} 波动率计算异常，降级为原逻辑: {_e}")
            vol_enabled = False
            vol_state = None

    # vol 快照字段（供 signal_snapshots 记录）
    if vol_state is not None:
        _vol_atr = vol_state.atr
        _vol_yz = vol_state.yz
        _vol_zscore = vol_state.z_score
        _vol_state_str = vol_state.state
    else:
        _vol_atr = _vol_yz = _vol_zscore = 0.0
        _vol_state_str = ""
    _vol_mult = vol_multiplier
    _vol_score = vol_score

    # ---- Session End Force Close (收盘前最后一分钟强制平仓) ----
    if is_session_end_cycle(cycle_time):
        if trader.has_position(symbol):
            contract_code = resolve_contract_code(symbol)
            _pos = trader.positions.get(symbol, {})
            _entry = _pos.get("entry_price")
            _dir = _pos.get("direction", "")
            _size = _pos.get("size")
            close_result = trader.force_close(
                symbol, price, cycle_time,
                sym_cfg["contract_multiplier"], sym_cfg,
                contract_code=contract_code, reason="session_end",
            )
            notify_before_trade(
                config,
                plan={
                    "event": "FORCE_CLOSE",
                    "symbol": symbol,
                    "direction": _dir,
                    "entry_price": _entry,
                    "exit_price": round(price, 2),
                    "position_size": _size,
                    "pnl": round(close_result["pnl"], 2) if close_result else None,
                    "cycle_time": cycle_time.strftime("%Y-%m-%d %H:%M"),
                },
                block=True,
            )
            if close_result:
                result["traded"] = True
                result["pnl"] = round(close_result["pnl"], 4)
                result["direction"] = close_result["direction"]
                result["notes"] = "session_end_close"
        # 收盘平仓后记录 snapshot
        insert_snapshot(
            symbol=symbol, cycle_time=cycle_time, price=price,
            rsi_value=0, rsi_signal=0,
            macd_line=0, macd_signal_line=0, macd_hist=0,
            macd_zero_factor=0, macd_signal=0,
            bb_upper=0, bb_lower=0, bb_middle=0,
            bb_bandwidth_ratio=0, bb_signal=0,
            momentum_value=0, momentum_accel=0, momentum_signal=0,
            tech_score=0, sentiment_score=0,
            alpha=0, final_score=0,
            direction="NEUTRAL", position_size=0,
            position_multiplier=0,
            is_stale=1 if is_stale else 0, is_rollover=0,
            adx_value=0,
            vol_atr=_vol_atr, vol_yz=_vol_yz, vol_zscore=_vol_zscore,
            vol_state=_vol_state_str, vol_multiplier=_vol_mult, vol_score=_vol_score,
        )
        return result

    # ---- Stop Loss Check (最高优先级，先于信号) ----
    trader.update_peak(symbol, price)
    stop_type, stop_price = trader.check_stop_loss(
        symbol, price, sym_cfg, vol_state=vol_state if vol_enabled else None)
    if stop_type:
        contract_code = resolve_contract_code(symbol)
        _pos = trader.positions.get(symbol, {})
        _entry = _pos.get("entry_price")
        _dir = _pos.get("direction", "")
        _size = _pos.get("size")
        stop_result = trader.force_close(
            symbol, stop_price, cycle_time,
            sym_cfg["contract_multiplier"], sym_cfg,
            contract_code=contract_code, reason=stop_type,
        )
        notify_before_trade(
            config,
            plan={
                "event": "STOP",
                "symbol": symbol,
                "direction": _dir,
                "entry_price": _entry,
                "exit_price": round(stop_price, 2),
                "position_size": _size,
                "pnl": round(stop_result["pnl"], 2) if stop_result else None,
                "cycle_time": cycle_time.strftime("%Y-%m-%d %H:%M"),
            },
            block=True,
        )
        if stop_result:
            result["traded"] = True
            result["pnl"] = round(stop_result["pnl"], 4)
            result["direction"] = stop_result["direction"]
            result["notes"] = f"stop_loss:{stop_type}"
            # 止损触发后仍继续记录 snapshot，但不发新信号
            insert_snapshot(
                symbol=symbol, cycle_time=cycle_time, price=price,
                rsi_value=0, rsi_signal=0,
                macd_line=0, macd_signal_line=0, macd_hist=0,
                macd_zero_factor=0, macd_signal=0,
                bb_upper=0, bb_lower=0, bb_middle=0,
                bb_bandwidth_ratio=0, bb_signal=0,
                momentum_value=0, momentum_accel=0, momentum_signal=0,
                tech_score=0, sentiment_score=0,
                alpha=0, final_score=0,
                direction="NEUTRAL", position_size=0,
            position_multiplier=0,
            is_stale=0, is_rollover=0,
            adx_value=0,
            vol_atr=_vol_atr, vol_yz=_vol_yz, vol_zscore=_vol_zscore,
            vol_state=_vol_state_str, vol_multiplier=_vol_mult, vol_score=_vol_score,
        )
        return result

    # ---- Load State ----
    ewma = EwmaTracker(decay_factor=sym_cfg["decay_factor"])
    state = load_ewma_state(symbol)
    ewma.load_from_dict(state)

    opt = FeedbackOptimizer(symbol, sym_cfg)
    opt.load_state(state)

    warmup_count = state.get("warmup_count", 0)

    # ---- Indicator EWMA Feedback (use prev-cycle signals vs actual price move) ----
    prev_rsi = state.get("prev_rsi_signal") or 0.0
    prev_macd = state.get("prev_macd_signal") or 0.0
    prev_bb = state.get("prev_bb_signal") or 0.0
    prev_mom = state.get("prev_mom_signal") or 0.0
    prev_price = state.get("prev_price") or 0.0

    if prev_price > 0:
        actual_dir = 1 if price > prev_price else -1 if price < prev_price else 0
        if actual_dir != 0:
            def _reward(sig):
                d = 1 if sig > 0 else -1 if sig < 0 else 0
                return 1.0 if d == actual_dir else (-1.0 if d != 0 else 0.0)
            ewma.update_indicators(
                _reward(prev_rsi), _reward(prev_macd),
                _reward(prev_bb), _reward(prev_mom),
            )

    # ---- Signal Layer ----
    indicators = compute_all(closes, sym_cfg, highs=highs, lows=lows)
    rsi = indicators["rsi"]
    macd = indicators["macd"]
    bb = indicators["bb"]
    mom = indicators["momentum"]
    adx = indicators["adx"]
    adx_val = adx.value if adx else 0.0

    sentiment_raw = sentiment_scores.get(symbol, 0.0)
    if not sentiment_valid:
        sentiment_raw = 0.0

    # ---- Scoring ----
    final_score, tech_score, alpha, weights = compute_score(
        rsi_signal=rsi.signal,
        macd_signal=macd.signal,
        bb_signal=bb.signal,
        mom_signal=mom.signal,
        sentiment_score=sentiment_raw,
        ewma_rsi=ewma.ewma_rsi,
        ewma_macd=ewma.ewma_macd,
        ewma_bb=ewma.ewma_bb,
        ewma_mom=ewma.ewma_mom,
        ewma_tech=ewma.ewma_tech,
        ewma_sent=ewma.ewma_sent,
        sentiment_valid=sentiment_valid,
        vol_score=vol_score,
        ewma_vol=ewma.ewma_vol,
        volatility_enabled=vol_enabled,
    )

    if not sentiment_valid:
        ewma.ewma_sent_frozen = True
        # 波动率融合开启时保留三路融合结果（tech + vol 两路）；
        # 关闭时回退为原行为（final = tech_score, alpha = 1.0）
        if not vol_enabled:
            alpha = 1.0
            final_score = tech_score

    # ---- Cold Start: warmup < 20 只拉数据+记录快照，不交易 ----
    if warmup_count < WARMUP_CYCLES:
        warmup_count += 1
        position_multiplier = opt.get_position_multiplier()
        insert_snapshot(
            symbol=symbol, cycle_time=cycle_time, price=price,
            rsi_value=rsi.value, rsi_signal=rsi.signal,
            macd_line=macd.line, macd_signal_line=macd.signal_line,
            macd_hist=macd.hist, macd_zero_factor=macd.zero_factor,
            macd_signal=macd.signal,
            bb_upper=bb.upper, bb_lower=bb.lower, bb_middle=bb.middle,
            bb_bandwidth_ratio=bb.bandwidth_ratio, bb_signal=bb.signal,
            momentum_value=mom.value, momentum_accel=mom.acceleration,
            momentum_signal=mom.signal,
            tech_score=tech_score, sentiment_score=sentiment_raw,
            alpha=alpha, final_score=final_score,
            direction="NEUTRAL", position_size=0,
            position_multiplier=position_multiplier,
            is_stale=1 if is_stale else 0, is_rollover=0,
            adx_value=adx_val,
            vol_atr=_vol_atr, vol_yz=_vol_yz, vol_zscore=_vol_zscore,
            vol_state=_vol_state_str, vol_multiplier=_vol_mult, vol_score=_vol_score,
        )
        # 保存当前周期信号供下周期指标 EWMA 反馈
        state_data = ewma.to_dict(symbol)
        state_data.update(opt.to_state_dict())
        state_data["alpha"] = alpha
        state_data["warmup_count"] = warmup_count
        state_data["prev_rsi_signal"] = rsi.signal
        state_data["prev_macd_signal"] = macd.signal
        state_data["prev_bb_signal"] = bb.signal
        state_data["prev_mom_signal"] = mom.signal
        state_data["prev_price"] = price
        save_ewma_state(**state_data)
        result["notes"] = f"warmup {warmup_count}/{WARMUP_CYCLES}"
        result["final_score"] = round(final_score, 4)
        result["alpha"] = round(alpha, 4)
        result["direction"] = "NEUTRAL"
        result["position_size"] = 0
        return result

    # ---- Position & Execution ----
    position_multiplier = opt.get_position_multiplier()
    if vol_enabled and vol_state is not None:
        position_multiplier = position_multiplier * vol_multiplier
    entry_threshold = sym_cfg.get("entry_threshold", 0.2)
    direction = get_direction(final_score, entry_threshold)
    position_size = compute_position_size(final_score, sym_cfg["base_lot"], position_multiplier, entry_threshold)

    if position_size == 0:
        direction = "NEUTRAL"

    trade_result = None
    if not is_stale and not opt.is_in_cooldown():
        # ---- 交易执行后：企业微信通知（同步发送；超时/失败不影响交易）----
        _will_open = direction != "NEUTRAL" and position_size > 0 and (
            not trader.has_position(symbol) or trader.positions[symbol]["direction"] != direction)
        _will_close = trader.has_position(symbol) and direction != trader.positions[symbol]["direction"]
        _before = trader.positions.get(symbol, {}) if trader.has_position(symbol) else {}
        _before_entry = _before.get("entry_price")
        _before_dir = _before.get("direction", "")
        _before_size = _before.get("size")
        trade_result = trader.execute(
            symbol, direction, position_size,
            price, price, cycle_time,
            sym_cfg["contract_multiplier"],
            symbol_cfg=sym_cfg,
            contract_code=resolve_contract_code(symbol),
            vol_slip_factor=vol_slip_factor if vol_enabled else None,
        )
        if _will_open or _will_close:
            if trade_result is not None:
                notify_before_trade(
                    config,
                    plan={
                        "event": "CLOSE",
                        "symbol": symbol,
                        "direction": _before_dir,
                        "entry_price": _before_entry,
                        "exit_price": round(price, 2),
                        "position_size": _before_size,
                        "pnl": round(trade_result["pnl"], 2),
                        "cycle_time": cycle_time.strftime("%Y-%m-%d %H:%M"),
                    },
                    block=True,
                )
            if _will_open:
                notify_before_trade(
                    config,
                    plan={
                        "event": "OPEN",
                        "symbol": symbol,
                        "direction": direction,
                        "entry_price": round(price, 2),
                        "position_size": round(position_size, 2),
                        "final_score": round(final_score, 4),
                        "alpha": round(alpha, 4),
                        "vol_score": round(_vol_score, 4),
                        "cycle_time": cycle_time.strftime("%Y-%m-%d %H:%M"),
                    },
                    block=True,
                )

    # ---- Snapshot ----
    insert_snapshot(
        symbol=symbol, cycle_time=cycle_time, price=price,
        rsi_value=rsi.value, rsi_signal=rsi.signal,
        macd_line=macd.line, macd_signal_line=macd.signal_line,
        macd_hist=macd.hist, macd_zero_factor=macd.zero_factor,
        macd_signal=macd.signal,
        bb_upper=bb.upper, bb_lower=bb.lower, bb_middle=bb.middle,
        bb_bandwidth_ratio=bb.bandwidth_ratio, bb_signal=bb.signal,
        momentum_value=mom.value, momentum_accel=mom.acceleration,
        momentum_signal=mom.signal,
        tech_score=tech_score, sentiment_score=sentiment_raw,
        alpha=alpha, final_score=final_score,
        direction=direction, position_size=position_size,
        position_multiplier=position_multiplier,
        is_stale=1 if is_stale else 0, is_rollover=0,
        adx_value=adx_val,
        vol_atr=_vol_atr, vol_yz=_vol_yz, vol_zscore=_vol_zscore,
        vol_state=_vol_state_str, vol_multiplier=_vol_mult, vol_score=_vol_score,
    )

    # ---- Feedback ----
    if trade_result and trade_result["pnl"] != 0:
        reward = compute_reward(
            trade_result["pnl"],
            price,
            sym_cfg["contract_multiplier"],
            position_size if position_size > 0 else sym_cfg["base_lot"],
            sym_cfg["risk_per_trade"],
        )
        # 技术/情绪 EWMA 独立更新
        actual_dir = 1 if trade_result["pnl"] > 0 else -1
        ewma.update_tech(tech_score, actual_dir)
        if sentiment_valid:
            ewma.update_sent(sentiment_raw, actual_dir)
        if vol_enabled and vol_state is not None and abs(vol_score) >= 0.02:
            ewma.update_vol(vol_score, actual_dir)
        # 方案B'：波动率加权 reward（可选，默认关闭，避免初期扰动反馈优化器）
        if vol_enabled and vol_state is not None and vol_cfg.get("adjust_reward", False):
            reward = adjust_ewma_reward(reward, vol_state)
        opt.update(reward)
        result["traded"] = True
        result["pnl"] = round(trade_result["pnl"], 4)
        result["direction"] = trade_result["direction"]

    if not trade_result or (trade_result and trade_result["pnl"] == 0):
        opt.advance_cycle()

    # ---- Persist ----
    state_data = ewma.to_dict(symbol)
    state_data.update(opt.to_state_dict())
    state_data["alpha"] = alpha
    state_data["warmup_count"] = warmup_count
    # 附加当前持仓状态（为空时使用默认值）
    pos = trader.positions.get(symbol, {})
    state_data["pos_direction"] = pos.get("direction", "")
    state_data["pos_entry"] = pos.get("entry_price", 0.0)
    state_data["pos_size"] = pos.get("size", 0.0)
    state_data["pos_open_time"] = pos.get("open_time")
    # 保存当前周期信号供下周期指标 EWMA 反馈
    state_data["prev_rsi_signal"] = rsi.signal
    state_data["prev_macd_signal"] = macd.signal
    state_data["prev_bb_signal"] = bb.signal
    state_data["prev_mom_signal"] = mom.signal
    state_data["prev_price"] = price
    save_ewma_state(**state_data)

    result["final_score"] = round(final_score, 4)
    result["alpha"] = round(alpha, 4)
    result["direction"] = direction
    result["position_size"] = round(position_size, 4)
    return result


def main():
    logger.info("=== 量化模拟交易系统启动 ===")
    config = load_config()
    global_cfg = {k: v for k, v in config.items() if k in {"data_freshness_threshold", "kline_cache_count", "contract_rollover_gap_pct"}}
    sentiment_cfg = config.get("sentiment", {})

    init_db()
    # 启动自愈：以 trades 表为持仓唯一权威来源，关闭孤儿仓并恢复持仓
    pos_states = reconcile_and_load_positions(config)
    trader = SimTrader(pos_states)

    now = datetime.now()
    if not is_trading_time(now):
        logger.info("非交易时段，跳过信号计算（持仓自愈已完成）")
        return

    sentiment_scores, sentiment_valid = load_sentiment(
        sentiment_cfg.get("json_path", ""),
        sentiment_cfg.get("max_age_hours", 2.0),
        symbols=list(config["symbols"].keys()),
    )

    cycle_time = datetime.now()
    symbols_cfg = config["symbols"]
    summaries = []

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(
                process_symbol, symbol, sym_cfg, global_cfg,
                sentiment_scores, sentiment_valid, trader, cycle_time, config,
            ): symbol
            for symbol, sym_cfg in symbols_cfg.items()
        }
        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                summary = fut.result()
                summaries.append(summary)
            except Exception as e:
                logger.error(f"{symbol} 处理异常: {e}", exc_info=True)
                summaries.append({"symbol": symbol, "traded": False, "pnl": 0, "notes": f"error: {e}"})

    logger.info("=== 周期汇总 ===")
    total_pnl = 0.0
    for s in summaries:
        total_pnl += s.get("pnl", 0.0)
        logger.info(
            f"  {s['symbol']}: dir={s.get('direction','?')}, "
            f"score={s.get('final_score',0):.4f}, alpha={s.get('alpha',0):.4f}, "
            f"size={s.get('position_size',0)}, traded={s.get('traded',False)}, "
            f"pnl={s.get('pnl',0):.4f}, note={s.get('notes','')}"
        )
    logger.info(f"  总盈亏: {total_pnl:.4f}")
    logger.info("=== 周期结束 ===")


if __name__ == "__main__":
    main()
