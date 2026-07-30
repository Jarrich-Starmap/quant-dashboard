"""量化交易 Dashboard 后端 API。FastAPI + SQLite 只读查询 + 配置读写。"""

from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import yaml
import copy

# 复用 db/models 的只读查询
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.models import get_conn

import yaml

def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def _calc_commission(symbol_cfg, price, lots):
    if symbol_cfg.get("commission_mode") == "fixed":
        return symbol_cfg.get("commission_fixed", 0) * lots
    elif symbol_cfg.get("commission_mode") == "ratio":
        return price * symbol_cfg["contract_multiplier"] * lots * symbol_cfg.get("commission_ratio", 0)
    return 0.0

app = FastAPI(title="Quant Trader Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DB = Path(__file__).resolve().parent.parent / "quant_trader.db"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _is_residual_zero(snap: dict) -> bool:
    """判断是否为休市/收盘残留的零快照（无真实信号）。

    收盘平仓与止损分支会写入一条全零快照（所有指标=0、adx_value=0）。
    真实交易快照必有非零的 adx_value / 指标值，因此「四项全为0」可稳健地
    识别残留零快照，即使历史数据在引入 ADX 前列的 adx_value 默认也是 0，
    但真实快照的 final_score/tech_score 非零，不会被误删。
    """
    return (
        (snap.get("final_score") or 0) == 0
        and (snap.get("tech_score") or 0) == 0
        and (snap.get("sentiment_score") or 0) == 0
        and (snap.get("adx_value") or 0) == 0
    )


@app.get("/api/summary")
def summary():
    """三品种当前状态一览。"""
    conn = get_conn()
    snaps = {}
    for sym in ["AU", "AG", "SC", "IC", "IM"]:
        # 取最近若干快照，跳过休市/收盘残留的零快照，展示最后一条有效信号
        rows = conn.execute(
            "SELECT * FROM signal_snapshots WHERE symbol=? ORDER BY cycle_time DESC LIMIT 20",
            (sym,),
        ).fetchall()
        chosen = None
        for r in rows:
            d = dict(r)
            if not _is_residual_zero(d):
                chosen = d
                break
        if chosen is None and rows:
            chosen = dict(rows[0])  # 全为零时退化为最近一条，避免空白
        if chosen:
            snaps[sym] = chosen

    ewmas = {}
    for sym in ["AU", "AG", "SC", "IC", "IM"]:
        row = conn.execute("SELECT * FROM ewma_state WHERE symbol=?", (sym,)).fetchone()
        if row:
            ewmas[sym] = dict(row)

    warmup = {}
    latest_prices = {}
    for sym in ["AU", "AG", "SC", "IC", "IM"]:
        e = ewmas.get(sym, {})
        warmup[sym] = {
            "count": e.get("warmup_count", 0),
            "done": (e.get("warmup_count", 0) >= 20),
        }
        k = conn.execute(
            "SELECT close, timestamp FROM kline_cache WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
            (sym,),
        ).fetchone()
        if k:
            latest_prices[sym] = {"price": k["close"], "time": k["timestamp"]}

    positions = {}
    for sym in ["AU", "AG", "SC", "IC", "IM"]:
        row = conn.execute("SELECT pos_direction FROM ewma_state WHERE symbol=?", (sym,)).fetchone()
        if row and row["pos_direction"]:
            positions[sym] = row["pos_direction"]
        else:
            positions[sym] = "FLAT"

    stats = {}
    for sym in ["AU", "AG", "SC", "IC", "IM"]:
        t = conn.execute(
            "SELECT COUNT(*) as cnt, SUM(pnl) as total_pnl, AVG(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as win_rate FROM trades WHERE symbol=?",
            (sym,),
        ).fetchone()
        stats[sym] = {
            "trades": t["cnt"] or 0,
            "total_pnl": round(t["total_pnl"] or 0, 4),
            "win_rate": round((t["win_rate"] or 0) * 100, 1),
        }

        # 合约代码：优先从缓存文件读取（由交易周期维护），回退到最近交易记录
    contract_codes = {}
    cache_data = {}
    try:
        import json
        cache_path = Path(__file__).resolve().parent.parent / "contract_cache.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
    except Exception:
        pass
    for sym in ["AU", "AG", "SC", "IC", "IM"]:
        cached = cache_data.get(sym)
        if cached and cached.get("code"):
            contract_codes[sym] = cached["code"].upper()
        else:
            row = conn.execute(
                "SELECT contract_code FROM trades WHERE symbol=? ORDER BY id DESC LIMIT 1",
                (sym,),
            ).fetchone()
            contract_codes[sym] = row["contract_code"].upper() if row else sym

    return {
        "snapshots": snaps,
        "ewma": ewmas,
        "positions": positions,
        "stats": stats,
        "warmup": warmup,
        "latest_prices": latest_prices,
        "contract_codes": contract_codes,
    }


@app.get("/api/portfolio")
def portfolio():
    """全部品种组合汇总：累计盈亏、今日盈亏、胜率、持仓。"""
    conn = get_conn()
    names = {"AU": "黄金", "AG": "白银", "SC": "原油", "IC": "中证500", "IM": "中证1000"}
    cache_data = {}
    try:
        import json
        cache_path = Path(__file__).resolve().parent.parent / "contract_cache.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
    except Exception:
        pass
    per = []
    total_pnl = 0.0
    today_pnl = 0.0
    total_trades = 0
    today_trades = 0
    win = 0
    open_count = 0
    for sym in ["AU", "AG", "SC", "IC", "IM"]:
        e = conn.execute("SELECT pos_direction, pos_entry, pos_size FROM ewma_state WHERE symbol=?", (sym,)).fetchone()
        e = dict(e) if e else {}
        pos = e.get("pos_direction") or "FLAT"
        if pos != "FLAT":
            open_count += 1
        st = conn.execute(
            "SELECT COUNT(*) AS cnt, SUM(pnl) AS total_pnl, SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) AS w FROM trades WHERE symbol=?",
            (sym,),
        ).fetchone()
        cnt = st["cnt"] or 0
        sym_total = round(st["total_pnl"] or 0, 4)
        w = st["w"] or 0
        td = conn.execute(
            "SELECT SUM(pnl) AS tp, COUNT(*) AS tc FROM trades WHERE symbol=? AND DATE(close_time)=DATE('now','localtime')",
            (sym,),
        ).fetchone()
        sym_today = round(td["tp"] or 0, 4)
        today_trades += td["tc"] or 0
        total_pnl += sym_total
        today_pnl += sym_today
        total_trades += cnt
        win += w
        k = conn.execute("SELECT close FROM kline_cache WHERE symbol=? ORDER BY timestamp DESC LIMIT 1", (sym,)).fetchone()
        price = k["close"] if k else None
        cached = cache_data.get(sym)
        if cached and cached.get("code"):
            code = cached["code"].upper()
        else:
            row = conn.execute("SELECT contract_code FROM trades WHERE symbol=? ORDER BY id DESC LIMIT 1", (sym,)).fetchone()
            code = row["contract_code"].upper() if row else sym
        per.append({
            "symbol": sym,
            "name": names.get(sym, sym),
            "position": pos,
            "total_pnl": sym_total,
            "today_pnl": sym_today,
            "win_rate": round(w / cnt * 100, 1) if cnt else 0,
            "trades": cnt,
            "price": price,
            "contract_code": code,
        })
    return {
        "per_symbol": per,
        "totals": {
            "total_pnl": round(total_pnl, 4),
            "today_pnl": round(today_pnl, 4),
            "total_trades": total_trades,
            "today_trades": today_trades,
            "win_rate": round(win / total_trades * 100, 1) if total_trades else 0,
            "open_positions": open_count,
        },
    }


@app.get("/api/snapshots")
def snapshots(symbol: str = "AU", limit: int = 50):
    rows = get_conn().execute(
        "SELECT * FROM signal_snapshots WHERE symbol=? ORDER BY cycle_time DESC LIMIT ?",
        (symbol, limit),
    ).fetchall()
    data = [dict(r) for r in rows][::-1]
    # 过滤休市/收盘残留的零快照，避免误导
    data = [d for d in data if not _is_residual_zero(d)]
    return data


@app.get("/api/trades")
def trades(symbol: str = "AU", offset: int = 0, limit: int = 10):
    rows = get_conn().execute(
        "SELECT * FROM trades WHERE symbol=? ORDER BY id DESC LIMIT ? OFFSET ?",
        (symbol, limit, offset),
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/pnl_chart")
def pnl_chart(symbol: str = "AU", hours: int = 0):
    if hours > 0:
        rows = get_conn().execute(
            "SELECT cycle_time, pnl, exit_price, entry_price FROM trades WHERE symbol=? AND cycle_time >= datetime('now', '-' || ? || ' hours', 'localtime') ORDER BY id ASC",
            (symbol, hours),
        ).fetchall()
    else:
        rows = get_conn().execute(
            "SELECT cycle_time, pnl, exit_price, entry_price FROM trades WHERE symbol=? ORDER BY id ASC",
            (symbol,),
        ).fetchall()
    cum = 0
    data = []
    for r in rows:
        cum += r["pnl"]
        price = r["exit_price"] if r["exit_price"] and r["exit_price"] != 0 else r["entry_price"]
        data.append({"time": r["cycle_time"], "pnl": round(r["pnl"], 4), "cum_pnl": round(cum, 4), "price": round(price, 2)})
    return data


@app.get("/api/score_chart")
def score_chart(symbol: str = "AU", limit: int = 100, since: str = "", hours: int = 0):
    if hours > 0:
        rows = get_conn().execute(
            "SELECT cycle_time, final_score, tech_score, sentiment_score, alpha, direction, vol_score, vol_zscore, vol_yz, vol_state, vol_multiplier FROM signal_snapshots WHERE symbol=? AND cycle_time >= datetime('now', '-' || ? || ' hours', 'localtime') ORDER BY cycle_time ASC",
            (symbol, hours),
        ).fetchall()
    elif since:
        rows = get_conn().execute(
            "SELECT cycle_time, final_score, tech_score, sentiment_score, alpha, direction, vol_score, vol_zscore, vol_yz, vol_state, vol_multiplier FROM signal_snapshots WHERE symbol=? AND cycle_time >= ? ORDER BY cycle_time ASC",
            (symbol, since),
        ).fetchall()
    else:
        # 多取一些，过滤零快照后仍能保留足够真实点
        rows = get_conn().execute(
            "SELECT cycle_time, final_score, tech_score, sentiment_score, alpha, direction, vol_score, vol_zscore, vol_yz, vol_state, vol_multiplier FROM signal_snapshots WHERE symbol=? ORDER BY cycle_time DESC LIMIT ?",
            (symbol, min(limit * 3, 500)),
        ).fetchall()
        rows = rows[::-1]
    data = [dict(r) for r in rows]
    # 过滤休市/收盘残留的零快照，避免得分曲线被平零误导
    data = [d for d in data if not _is_residual_zero(d)]
    return data




@app.get("/api/summary_chart")
def summary_chart(window: str = "24h"):
    """汇总页常用图表：组合累计盈亏、各品种盈亏贡献、得分趋势。
    时间维度：1h（近1小时）/ 24h（近24小时）/ 7d（近7天）。"""
    # since 表达式（白名单，内联进 SQL 安全）
    WIN = {
        "1h":  "datetime('now','-1 hours','localtime')",
        "24h": "datetime('now','-24 hours','localtime')",
        "7d":  "datetime('now','-7 days','localtime')",
    }
    if window not in WIN:
        window = "24h"
    since = WIN[window]
    SYMS = ["AU", "AG", "SC", "IC", "IM"]
    conn = get_conn()

    # 1) 组合累计盈亏（来自 trades，按时间累计）
    trade_rows = conn.execute(
        f"SELECT close_time, pnl FROM trades WHERE close_time >= {since} ORDER BY close_time ASC"
    ).fetchall()
    equity_labels, equity_values, cum = [], [], 0.0
    start_t = conn.execute(f"SELECT {since} AS t").fetchone()["t"]
    equity_labels.append(str(start_t)[:16])
    equity_values.append(0.0)
    for r in trade_rows:
        cum += (r["pnl"] or 0)
        equity_labels.append(str(r["close_time"])[:16])
        equity_values.append(round(cum, 4))

    # 2) 各品种盈亏贡献（柱状图）
    per_rows = conn.execute(
        f"SELECT symbol, SUM(pnl) AS pnl FROM trades WHERE close_time >= {since} GROUP BY symbol"
    ).fetchall()
    per_symbol = {s: 0.0 for s in SYMS}
    for r in per_rows:
        per_symbol[r["symbol"]] = round(r["pnl"] or 0, 4)
    window_pnl = round(sum(per_symbol.values()), 4)
    window_trades = len(trade_rows)

    # 3) 得分趋势（signal_snapshots 按时间桶聚合、跨品种平均）
    if window == "1h":
        bucket_expr = "strftime('%Y-%m-%d %H:', cycle_time) || printf('%02d', (cast(strftime('%M', cycle_time) as int)/5)*5)"
    elif window == "24h":
        bucket_expr = "strftime('%Y-%m-%d %H:00', cycle_time)"
    else:
        bucket_expr = "strftime('%Y-%m-%d', cycle_time)"
    score_rows = conn.execute(
        f"""SELECT {bucket_expr} AS b,
                   AVG(final_score)   AS final_score,
                   AVG(tech_score)    AS tech_score,
                   AVG(sentiment_score) AS sentiment_score,
                   AVG(vol_score)     AS vol_score
            FROM signal_snapshots
            WHERE cycle_time >= {since}
              AND NOT (final_score=0 AND tech_score=0 AND sentiment_score=0 AND adx_value=0)
            GROUP BY b ORDER BY b ASC"""
    ).fetchall()
    score_labels = [r["b"][5:] if r["b"] and len(r["b"]) > 5 else (r["b"] or "") for r in score_rows]
    score_series = {
        "final":     [round(r["final_score"], 4) if r["final_score"] is not None else None for r in score_rows],
        "tech":      [round(r["tech_score"], 4) if r["tech_score"] is not None else None for r in score_rows],
        "sentiment": [round(r["sentiment_score"], 4) if r["sentiment_score"] is not None else None for r in score_rows],
        "vol":       [round(r["vol_score"], 4) if r["vol_score"] is not None else None for r in score_rows],
    }

    return {
        "window": window,
        "equity_labels": equity_labels,
        "equity_values": equity_values,
        "per_symbol": per_symbol,
        "window_pnl": window_pnl,
        "window_trades": window_trades,
        "score_labels": score_labels,
        "score_series": score_series,
    }


# ---- 配置读写 ----

@app.get("/api/config")
def get_config():
    """读取完整 config.yaml，屏蔽 akshare_symbol 等纯运行时字段。"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 返回干净的可视配置（含 global 块与根级全局键）
    result = {
        "mode": cfg.get("mode", ""),
        "symbols": {},
        "global": cfg.get("global", {}),
        "sentiment": cfg.get("sentiment", {}),
        "data_freshness_threshold": cfg.get("data_freshness_threshold", 1800),
        "kline_cache_count": cfg.get("kline_cache_count", 100),
        "contract_rollover_gap_pct": cfg.get("contract_rollover_gap_pct", 0.03),
    }
    for sym, s in cfg.get("symbols", {}).items():
        result["symbols"][sym] = {k: v for k, v in s.items()}
    return result


class ConfigSaveRequest(BaseModel):
    symbols: dict = {}      # {AU: {...}, AG: {...}, SC: {...}}
    global_cfg: dict = {}   # 全局指标/波动率常数
    root: dict = {}         # 根级全局标量（mode/阈值/rollover）
    sentiment: dict = {}    # sentiment 子块（max_age_hours 等）


@app.post("/api/config")
def save_config(req: ConfigSaveRequest):
    """保存品种参数到 config.yaml。仅更新 symbols 下可配置字段，保留根级其他 key。"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for sym, params in req.symbols.items():
            if sym not in cfg["symbols"]:
                raise HTTPException(400, f"未知品种: {sym}")

            current = cfg["symbols"][sym]
            # 类型转换：按 config 现有字段类型自适应，避免硬编码白名单漏掉字段（如 entry_threshold）
            for f, v in params.items():
                if f == "volatility" or f not in current:
                    continue
                if v in (None, ""):
                    continue
                cur_t = type(current[f])
                if cur_t in (list, dict):
                    continue
                try:
                    if cur_t is bool:
                        current[f] = str(v).lower() in ("true", "1", "yes")
                    elif cur_t is int:
                        current[f] = int(float(v))
                    elif cur_t is float:
                        current[f] = float(v)
                    else:
                        current[f] = str(v)
                except (ValueError, TypeError):
                    continue

            # 波动率参数（嵌套 volatility 块）
            if "volatility" in params and isinstance(params["volatility"], dict):
                vol_cur = current.get("volatility", {}) or {}
                vol_int = ["period", "state_window", "trend_fast", "trend_slow",
                           "direction_ma_period", "volume_ma_period"]
                vol_float = ["calm_z", "high_z", "extreme_z", "direction_scale_atr",
                             "expansion_strength", "expansion_vol_boost",
                             "expansion_center", "expansion_steepness",
                             "volume_confirm_ratio", "reversal_strength",
                             "reversal_z", "reversal_steepness", "dead_zone"]
                vol_bool = ["enabled", "dynamic_stop", "adjust_reward"]
                vp = params["volatility"]
                for f in vol_int:
                    if f in vp and vp[f] not in (None, ""):
                        vol_cur[f] = int(vp[f])
                for f in vol_float:
                    if f in vp and vp[f] not in (None, ""):
                        vol_cur[f] = float(vp[f])
                for f in vol_bool:
                    if f in vp:
                        v = vp[f]
                        vol_cur[f] = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
                current["volatility"] = vol_cur

            # technical_weights 字典合并（前端传 4 个等权输入）
            if "technical_weights" in params and isinstance(params["technical_weights"], dict):
                tw_cur = current.get("technical_weights", {}) or {}
                for wk, wv in params["technical_weights"].items():
                    if wv in (None, ""):
                        continue
                    try:
                        tw_cur[wk] = float(wv)
                    except (ValueError, TypeError):
                        pass
                current["technical_weights"] = tw_cur

            cfg["symbols"][sym] = current

        # ---- 全局块（global_cfg）+ 根级全局（root）+ sentiment ----
        def _merge_scalars(section, incoming):
            for k, v in incoming.items():
                if v in (None, ""):
                    continue
                if k not in section:
                    section[k] = v
                    continue
                cur_t = type(section[k])
                if cur_t in (list, dict):
                    continue
                try:
                    if cur_t is bool:
                        section[k] = str(v).lower() in ("true", "1", "yes")
                    elif cur_t is int:
                        section[k] = int(float(v))
                    elif cur_t is float:
                        section[k] = float(v)
                    else:
                        section[k] = str(v)
                except (ValueError, TypeError):
                    continue

        if req.global_cfg:
            g_cur = cfg.get("global", {}) or {}
            g_in = dict(req.global_cfg)
            g_vol_in = g_in.pop("volatility", None)
            _merge_scalars(g_cur, g_in)
            if g_vol_in and isinstance(g_vol_in, dict):
                gv_cur = g_cur.get("volatility", {}) or {}
                _merge_scalars(gv_cur, g_vol_in)
                g_cur["volatility"] = gv_cur
            cfg["global"] = g_cur

        if req.root:
            _merge_scalars(cfg, req.root)

        if req.sentiment:
            sc = cfg.get("sentiment", {}) or {}
            _merge_scalars(sc, req.sentiment)
            cfg["sentiment"] = sc

        # 写回
        backup_path = CONFIG_PATH.with_suffix(".yaml.bak." + datetime.now().strftime("%Y%m%d%H%M%S"))
        import shutil
        shutil.copy2(CONFIG_PATH, backup_path)

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        return {"status": "ok", "backup": str(backup_path)}
    except Exception as e:
        raise HTTPException(500, f"保存失败: {str(e)}")


@app.get("/api/trades/count")
def trades_count(symbol: str = "AU"):
    row = get_conn().execute(
        "SELECT COUNT(*) as cnt FROM trades WHERE symbol=?",
        (symbol,),
    ).fetchone()
    return {"total": row["cnt"] if row else 0}


@app.get("/api/trade-detail")
def trade_detail(id: int):
    """返回单笔交易详细信息，含手续费、滑点、保证金计算。"""
    cfg = _load_config()
    row = get_conn().execute(
        "SELECT * FROM trades WHERE id=?", (id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Trade not found")

    r = dict(row)
    sym = r["symbol"]
    sym_cfg = cfg["symbols"].get(sym, {})
    cm = sym_cfg.get("contract_multiplier", 1)
    direction = r["direction"]
    entry = r["entry_price"]
    exit_p = r.get("exit_price") or 0
    size = r["position_size"]
    is_open = exit_p == 0

    # 手续费
    entry_comm = _calc_commission(sym_cfg, entry, size)
    exit_comm = _calc_commission(sym_cfg, exit_p, size) if not is_open else 0
    total_comm = entry_comm + exit_comm

    # 滑点理论成本
    slip_rate = sym_cfg.get("slippage_rate", 0)
    entry_slip = entry * slip_rate * size * cm
    exit_slip = exit_p * slip_rate * size * cm if not is_open else 0

    # 毛利 vs 净利
    if not is_open:
        if direction == "LONG":
            gross_pnl = (exit_p - entry) * size * cm
        else:
            gross_pnl = (entry - exit_p) * size * cm
        net_pnl = gross_pnl - total_comm
    else:
        gross_pnl = 0
        net_pnl = 0

    # 保证金
    margin_rate = sym_cfg.get("margin_rate", 0.1)
    margin = entry * cm * size * margin_rate

    # 止损参数
    hard_stop = sym_cfg.get("hard_stop_loss", 0.04)
    breakeven = sym_cfg.get("breakeven_trigger", 0.04)
    trailing = sym_cfg.get("trailing_stop_pct", 0.02)

    if direction == "LONG":
        hard_stop_price = round(entry * (1 - hard_stop), 2)
        breakeven_price = round(entry * (1 + breakeven), 2)
    else:
        hard_stop_price = round(entry * (1 + hard_stop), 2)
        breakeven_price = round(entry * (1 - breakeven), 2)

    return {
        "id": r["id"],
        "symbol": sym,
        "contract_code": r.get("contract_code", ""),
        "direction": direction,
        "entry_price": entry,
        "exit_price": exit_p,
        "position_size": size,
        "contract_multiplier": cm,
        "open_time": str(r.get("open_time", "")),
        "close_time": str(r.get("close_time", "")),
        "is_open": is_open,
        # 费用明细
        "entry_commission": round(entry_comm, 2),
        "exit_commission": round(exit_comm, 2),
        "total_commission": round(total_comm, 2),
        "entry_slippage_cost": round(entry_slip, 2),
        "exit_slippage_cost": round(exit_slip, 2),
        "slippage_rate": slip_rate,
        # 盈亏
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(r.get("pnl", 0), 2),
        "pnl_pct": round(r.get("pnl_pct", 0) * 100, 4),
        # 风控
        "margin_required": round(margin, 2),
        "margin_rate": margin_rate,
        "hard_stop_loss": hard_stop,
        "hard_stop_price": hard_stop_price,
        "breakeven_trigger": breakeven,
        "trailing_stop_pct": trailing,
        "commission_mode": sym_cfg.get("commission_mode", ""),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
