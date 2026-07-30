"""SQLite 表结构定义与 CRUD 操作。WAL 模式，线程安全。"""

import sqlite3
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "quant_trader.db"

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=5000")  # 防并发写被静默丢弃
        _local.conn.execute("PRAGMA foreign_keys=ON")
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_time      DATETIME NOT NULL,
            symbol          TEXT NOT NULL,
            direction       TEXT NOT NULL,
            entry_price     REAL,
            contract_code   TEXT DEFAULT "",
            exit_price      REAL,
            position_size   REAL,
            pnl             REAL,
            pnl_pct         REAL,
            open_time       DATETIME,
            close_time      DATETIME
        );

        CREATE TABLE IF NOT EXISTS signal_snapshots (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_time            DATETIME NOT NULL,
            symbol                TEXT NOT NULL,
            price                 REAL,
            rsi_value             REAL,
            rsi_signal            REAL,
            macd_line             REAL,
            macd_signal_line      REAL,
            macd_hist             REAL,
            macd_zero_factor      REAL,
            macd_signal           REAL,
            bb_upper              REAL,
            bb_lower              REAL,
            bb_middle             REAL,
            bb_bandwidth_ratio    REAL,
            bb_signal             REAL,
            momentum_value        REAL,
            momentum_accel        REAL,
            momentum_signal       REAL,
            tech_score            REAL,
            sentiment_score       REAL,
            alpha                 REAL,
            final_score           REAL,
            direction             TEXT,
            position_size         REAL,
            position_multiplier   REAL,
            is_stale              INTEGER DEFAULT 0,
            is_rollover           INTEGER DEFAULT 0,
            adx_value             REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS ewma_state (
            symbol                    TEXT PRIMARY KEY,
            ewma_rsi                  REAL DEFAULT 0.0,
            ewma_macd                 REAL DEFAULT 0.0,
            ewma_bb                   REAL DEFAULT 0.0,
            ewma_mom                  REAL DEFAULT 0.0,
            ewma_tech                 REAL DEFAULT 0.0,
            ewma_sent                 REAL DEFAULT 0.0,
            ewma_sent_frozen          INTEGER DEFAULT 0,
            alpha                     REAL DEFAULT 0.5,
            error_count               INTEGER DEFAULT 0,
            cool_remaining            INTEGER DEFAULT 0,
            recovery_correct_streak   INTEGER DEFAULT 0,
            warmup_count              INTEGER DEFAULT 0,
            pos_direction             TEXT DEFAULT '',
            pos_entry                 REAL DEFAULT 0.0,
            pos_size                  REAL DEFAULT 0.0,
            pos_open_time             DATETIME,
            prev_rsi_signal           REAL DEFAULT 0.0,
            prev_macd_signal          REAL DEFAULT 0.0,
            prev_bb_signal            REAL DEFAULT 0.0,
            prev_mom_signal           REAL DEFAULT 0.0,
            prev_price                REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS kline_cache (
            symbol      TEXT NOT NULL,
            timestamp   DATETIME NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL,
            PRIMARY KEY (symbol, timestamp)
        );
    """)
    conn.commit()

    # Migration: 添加 adx_value 列（已有表兼容）
    try:
        conn.execute("ALTER TABLE signal_snapshots ADD COLUMN adx_value REAL DEFAULT 0")
        conn.commit()
    except Exception:
        pass  # 列已存在

    # Migration: 添加波动率模块字段（方案B'，已有表兼容，幂等）
    for col, ctype in (
        ("vol_atr", "REAL"),
        ("vol_yz", "REAL"),
        ("vol_zscore", "REAL"),
        ("vol_multiplier", "REAL"),
        ("vol_score", "REAL"),
    ):
        try:
            conn.execute(
                f"ALTER TABLE signal_snapshots ADD COLUMN {col} {ctype} DEFAULT 0"
            )
            conn.commit()
        except Exception:
            pass  # 列已存在
    try:
        conn.execute(
            "ALTER TABLE signal_snapshots ADD COLUMN vol_state TEXT DEFAULT ''"
        )
        conn.commit()
    except Exception:
        pass  # 列已存在
    try:
        conn.execute(
            "ALTER TABLE ewma_state ADD COLUMN ewma_vol REAL DEFAULT 0.5"
        )
        conn.commit()
    except Exception:
        pass  # 列已存在

def insert_trade(symbol: str, direction: str, entry_price: float,
                 exit_price: float, position_size: float,
                 pnl: float, pnl_pct: float, cycle_time: Optional[datetime] = None,
                 open_time: Optional[datetime] = None, close_time: Optional[datetime] = None,
                 contract_code: str = ""):
    if cycle_time is None:
        cycle_time = datetime.now()
    get_conn().execute(
        "INSERT INTO trades (cycle_time, symbol, direction, entry_price, exit_price, position_size, pnl, pnl_pct, open_time, close_time, contract_code) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (cycle_time, symbol, direction, entry_price, exit_price, position_size, pnl, pnl_pct, open_time, close_time, contract_code),
    )
    get_conn().commit()


def get_open_trades(symbol: str) -> list:
    """返回该品种所有未平仓记录（close_time IS NULL），按 id 升序。"""
    rows = get_conn().execute(
        "SELECT * FROM trades WHERE symbol=? AND close_time IS NULL ORDER BY id ASC",
        (symbol,),
    ).fetchall()
    return [dict(r) for r in rows]


def close_trade_by_id(trade_id: int, exit_price: float, pnl: float,
                      pnl_pct: float, close_time=None, contract_code: str = ""):
    """按 id 关闭指定交易（启动自愈 / 强平通用）。"""
    if close_time is None:
        close_time = datetime.now()
    get_conn().execute(
        "UPDATE trades SET exit_price=?, close_time=?, pnl=?, pnl_pct=?, contract_code=? WHERE id=?",
        (exit_price, close_time, pnl, pnl_pct, contract_code, trade_id),
    )
    get_conn().commit()


def update_trade_close(symbol, direction, exit_price, position_size,
                       pnl, pnl_pct, close_time=None, contract_code=""):
    """
    关闭该品种该方向最早一笔未平仓。

    修复要点：
    - 去掉脆弱的 (symbol, direction, entry_price) 精确匹配：入场价因滑点/取整
      与开仓记录不一致时旧逻辑会匹配失败并插入 stub，导致原始开仓记录永远悬空。
    - 去掉 else 分支的 stub 插入：匹配不上时静默跳过，不再制造孤儿记录。
    """
    conn = get_conn()
    if close_time is None:
        close_time = datetime.now()
    cur = conn.execute(
        "SELECT id FROM trades WHERE symbol=? AND direction=? AND close_time IS NULL ORDER BY id ASC LIMIT 1",
        (symbol, direction),
    )
    row = cur.fetchone()
    if row:
        conn.execute(
            "UPDATE trades SET exit_price=?, close_time=?, pnl=?, pnl_pct=? WHERE id=?",
            (exit_price, close_time, pnl, pnl_pct, row["id"]),
        )
        conn.commit()
    else:
        logger.warning(
            f"update_trade_close: 未找到 {symbol}/{direction} 的未平仓记录，跳过（不插入 stub）"
        )


# ---- signal_snapshots ----

def insert_snapshot(symbol: str, cycle_time: datetime, price: float,
                    rsi_value: float, rsi_signal: float,
                    macd_line: float, macd_signal_line: float, macd_hist: float,
                    macd_zero_factor: float, macd_signal: float,
                    bb_upper: float, bb_lower: float, bb_middle: float,
                    bb_bandwidth_ratio: float, bb_signal: float,
                    momentum_value: float, momentum_accel: float, momentum_signal: float,
                    tech_score: float, sentiment_score: float, alpha: float,
                    final_score: float, direction: str, position_size: float,
                    position_multiplier: float,
                    is_stale: int = 0, is_rollover: int = 0,
                    adx_value: float = 0.0,
                    vol_atr: float = 0.0, vol_yz: float = 0.0,
                    vol_zscore: float = 0.0, vol_state: str = "",
                    vol_multiplier: float = 1.0, vol_score: float = 0.0):
    get_conn().execute(
        """INSERT INTO signal_snapshots
        (cycle_time, symbol, price,
         rsi_value, rsi_signal,
         macd_line, macd_signal_line, macd_hist, macd_zero_factor, macd_signal,
         bb_upper, bb_lower, bb_middle, bb_bandwidth_ratio, bb_signal,
         momentum_value, momentum_accel, momentum_signal,
         tech_score, sentiment_score, alpha, final_score,
         direction, position_size, position_multiplier,
         is_stale, is_rollover, adx_value,
         vol_atr, vol_yz, vol_zscore, vol_state, vol_multiplier, vol_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cycle_time, symbol, price,
         rsi_value, rsi_signal,
         macd_line, macd_signal_line, macd_hist, macd_zero_factor, macd_signal,
         bb_upper, bb_lower, bb_middle, bb_bandwidth_ratio, bb_signal,
         momentum_value, momentum_accel, momentum_signal,
         tech_score, sentiment_score, alpha, final_score,
         direction, position_size, position_multiplier,
         is_stale, is_rollover, adx_value,
         vol_atr, vol_yz, vol_zscore, vol_state, vol_multiplier, vol_score),
    )
    get_conn().commit()


# ---- ewma_state ----

def load_ewma_state(symbol: str) -> dict:
    row = get_conn().execute(
        "SELECT * FROM ewma_state WHERE symbol=?", (symbol,)
    ).fetchone()
    if row is None:
        return {
            "symbol": symbol,
            "ewma_rsi": 0.0, "ewma_macd": 0.0, "ewma_bb": 0.0, "ewma_mom": 0.0,
            "ewma_tech": 0.0, "ewma_sent": 0.0, "ewma_sent_frozen": 0,
            "ewma_vol": 0.5,
            "alpha": 0.5, "error_count": 0, "cool_remaining": 0,
            "recovery_correct_streak": 0, "warmup_count": 0,
            "prev_rsi_signal": 0.0, "prev_macd_signal": 0.0,
            "prev_bb_signal": 0.0, "prev_mom_signal": 0.0, "prev_price": 0.0,
            "pos_direction": "", "pos_entry": 0.0, "pos_size": 0.0, "pos_open_time": None,
        }
    return dict(row)


def save_ewma_state(symbol: str, ewma_rsi: float, ewma_macd: float,
                    ewma_bb: float, ewma_mom: float, ewma_tech: float,
                    ewma_sent: float, ewma_sent_frozen: int, alpha: float,
                    error_count: int, cool_remaining: int, recovery_correct_streak: int,
                    warmup_count: int,
                    prev_rsi_signal: float = 0.0, prev_macd_signal: float = 0.0,
                    prev_bb_signal: float = 0.0, prev_mom_signal: float = 0.0,
                    prev_price: float = 0.0,
                    pos_direction: str = "", pos_entry: float = 0.0, pos_size: float = 0.0, pos_open_time=None,
                    ewma_vol: float = 0.5):
    get_conn().execute(
        """INSERT OR REPLACE INTO ewma_state
        (symbol, ewma_rsi, ewma_macd, ewma_bb, ewma_mom,
         ewma_tech, ewma_sent, ewma_sent_frozen, alpha,
         error_count, cool_remaining, recovery_correct_streak,
         warmup_count,
         prev_rsi_signal, prev_macd_signal, prev_bb_signal, prev_mom_signal, prev_price,
         pos_direction, pos_entry, pos_size, pos_open_time, ewma_vol)
        VALUES (?,?,?,?,?,  ?,?,?,?,  ?,?,?,?,  ?,?,?,?,?,  ?,?,?,?,?)""",
        (symbol, ewma_rsi, ewma_macd, ewma_bb, ewma_mom,
         ewma_tech, ewma_sent, ewma_sent_frozen, alpha,
         error_count, cool_remaining, recovery_correct_streak,
         warmup_count,
         prev_rsi_signal, prev_macd_signal, prev_bb_signal, prev_mom_signal, prev_price,
         pos_direction, pos_entry, pos_size, pos_open_time, ewma_vol),
    )
    get_conn().commit()


def reset_ewma_state(symbol: str):
    get_conn().execute("DELETE FROM ewma_state WHERE symbol=?", (symbol,))
    get_conn().commit()


# ---- kline_cache ----

def get_cached_klines(symbol: str, limit: int = 200) -> list:
    rows = get_conn().execute(
        "SELECT * FROM kline_cache WHERE symbol=? ORDER BY timestamp ASC",
        (symbol,),
    ).fetchall()
    return [dict(r) for r in rows[-limit:]]


def upsert_klines(symbol: str, records: list[dict]):
    conn = get_conn()
    for r in records:
        conn.execute(
            """INSERT OR REPLACE INTO kline_cache
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?,?)""",
            (symbol, r["timestamp"], r["open"], r["high"], r["low"], r["close"], r["volume"]),
        )
    conn.commit()


def clear_klines(symbol: str, keep_last: int = 0):
    """
    清空缓存。keep_last=N 时保留最近 N 根旧数据（换月时供指标计算用）。
    """
    if keep_last > 0:
        rows = get_conn().execute(
            "SELECT rowid FROM kline_cache WHERE symbol=? ORDER BY timestamp ASC",
            (symbol,),
        ).fetchall()
        if len(rows) > keep_last:
            cutoff = rows[-keep_last - 1][0]
            get_conn().execute(
                "DELETE FROM kline_cache WHERE symbol=? AND rowid <= ?",
                (symbol, cutoff),
            )
        get_conn().commit()
        return

    get_conn().execute("DELETE FROM kline_cache WHERE symbol=?", (symbol,))
    get_conn().commit()


def get_latest_close(symbol: str) -> Optional[float]:
    row = get_conn().execute(
        "SELECT close FROM kline_cache WHERE symbol=? ORDER BY timestamp DESC LIMIT 1",
        (symbol,),
    ).fetchone()
    return row["close"] if row else None
