"""
NSE Stockbee Scanner — Database Layer
======================================
Based on Pradeep Bonde (Stockbee) methodology.
SQLite-backed persistence for:
  • signals          – every signal ever generated (MB + EP)
  • watchlist        – active / pending signals being tracked
  • performance      – strategy effectiveness statistics
  • backtest_runs    – historical backtest run summaries
  • backtest_trade_log – individual trade records per backtest run

Status lifecycle (unified across all tables):
  Waiting  → signal generated, entry not yet triggered
  Active   → entry triggered, trade is open
  Target 1 Achieved / Target 2 Achieved / Target 3 Achieved → closed, winner
  Stopped Out → closed, loser (stop loss hit)
  Expired  → watchlist_expiry_days elapsed with no entry trigger

FIX (v2):
  • Removed Darvas Box header
  • Fixed get_open_signals(): SQL AND/OR operator precedence bug
    Old: WHERE status NOT IN (...) AND entry_triggered = 0 OR status = 'Active'
    Fix: WHERE status NOT IN (...) AND (entry_triggered = 0 OR status = 'Active')
  • Removed duplicate upsert_signal / upsert_watchlist / signal_exists functions
    (the shim at the bottom referenced _db() which doesn't exist — caused NameError)
  • Unified status strings: 'Waiting' not 'PENDING', 'Active' not 'ACTIVE', etc.
    signal_tracker.py uses these exact strings — must match.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import SIGNALS_DB, WATCHLIST_DB, WATCHLIST_EXPIRY_DAYS
from logger_utils import get_logger

log = get_logger("scanner")

DB_PATH = SIGNALS_DB   # single consolidated DB

# ─── Schema ───────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id          TEXT PRIMARY KEY,
    symbol             TEXT NOT NULL,
    signal_type        TEXT,                    -- 'MB_BREAKOUT' | 'MB_ANTICIPATION' | 'EP_9M' | etc.
    sector             TEXT,
    scan_date          TEXT NOT NULL,
    current_price      REAL,
    entry_zone_low     REAL,
    entry_zone_high    REAL,
    stop_loss          REAL,
    target1            REAL,
    target2            REAL,
    target3            REAL,
    atr                REAL,
    risk_per_share     REAL,
    position_size      INTEGER,
    capital_required   REAL,
    risk_amount        REAL,
    rr_ratio           REAL,
    volume_ratio       REAL,
    rs_rank            REAL,
    ti65               REAL,
    twolynch_score     REAL,
    composite_score    REAL,
    classification     TEXT,                    -- 'Elite' | 'Strong' | 'Watch'
    market_regime      TEXT,                    -- market regime at time of signal
    -- Forward test tracking
    status             TEXT DEFAULT 'Waiting',  -- Waiting | Active | Target N Achieved | Stopped Out | Expired
    entry_triggered    INTEGER DEFAULT 0,
    t1_achieved        INTEGER DEFAULT 0,
    t2_achieved        INTEGER DEFAULT 0,
    t3_achieved        INTEGER DEFAULT 0,
    stopped_out        INTEGER DEFAULT 0,
    max_fav_excursion  REAL,
    max_adv_excursion  REAL,
    days_to_target     INTEGER,
    realised_rr        REAL,
    last_checked       TEXT,
    created_at         TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS watchlist (
    signal_id        TEXT PRIMARY KEY,
    symbol           TEXT NOT NULL,
    signal_type      TEXT,
    entry_price      REAL,
    stop_loss        REAL,
    target1          REAL,
    target2          REAL,
    target3          REAL,
    composite_score  REAL,
    rs_rank          REAL,
    status           TEXT DEFAULT 'Waiting',
    detected_date    TEXT,
    expiry_date      TEXT,
    last_updated     TEXT
);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    snapshot_date   TEXT NOT NULL,
    score_band      TEXT NOT NULL,
    total_signals   INTEGER,
    triggered       INTEGER,
    t1_hit          INTEGER,
    t2_hit          INTEGER,
    t3_hit          INTEGER,
    stopped         INTEGER,
    win_rate        REAL,
    avg_rr          REAL,
    profit_factor   REAL,
    expectancy      REAL,
    PRIMARY KEY (snapshot_date, score_band)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id               TEXT PRIMARY KEY,
    run_date             TEXT NOT NULL,
    signal_type          TEXT,
    symbols_tested       INTEGER,
    symbols_with_trades  INTEGER,
    total_trades         INTEGER,
    win_rate             REAL,
    profit_factor        REAL,
    expectancy           REAL,
    avg_cagr             REAL,
    avg_drawdown         REAL,
    avg_sharpe           REAL,
    avg_hold_days        REAL,
    notes                TEXT,
    created_at           TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS backtest_symbol_summary (
    run_id          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    trades          INTEGER,
    win_rate        REAL,
    profit_factor   REAL,
    cagr_pct        REAL,
    max_drawdown    REAL,
    sharpe          REAL,
    avg_hold        REAL,
    PRIMARY KEY (run_id, symbol)
);

CREATE TABLE IF NOT EXISTS backtest_trade_log (
    run_id          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    signal_type     TEXT,
    entry_date      TEXT,
    exit_date       TEXT,
    entry_price     REAL,
    exit_price      REAL,
    stop_loss       REAL,
    target1         REAL,
    target2         REAL,
    outcome         TEXT,        -- 'target1_hit' | 'target2_hit' | 'stopped_out' | 'open_at_end' | 'day5_exit'
    rr_realised     REAL,
    hold_days       INTEGER,
    composite_score REAL,
    rs_rank         REAL,
    ti65            REAL,
    twolynch_score  REAL,
    market_regime   TEXT,
    score_band      TEXT         -- 'Elite' | 'Strong' | 'Watch'
);
"""


# ─── Connection context ───────────────────────────────────────────────────────

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _migrate_db(con: sqlite3.Connection) -> None:
    """Add columns that exist in current schema but are missing from older DB on disk."""
    required: dict[str, list[tuple[str, str]]] = {
        "signals": [
            ("signal_type",    "TEXT"),
            ("ti65",           "REAL"),
            ("twolynch_score", "REAL"),
            ("rs_rank",        "REAL"),
            ("market_regime",  "TEXT"),
        ],
        "watchlist": [
            ("signal_type", "TEXT"),
            ("rs_rank",     "REAL"),
        ],
        "performance_snapshots": [
            ("expectancy", "REAL"),
        ],
        "backtest_runs": [
            ("signal_type",   "TEXT"),
            ("expectancy",    "REAL"),
            ("avg_cagr",      "REAL"),
            ("avg_drawdown",  "REAL"),
            ("avg_sharpe",    "REAL"),
            ("avg_hold_days", "REAL"),
            ("notes",         "TEXT"),
        ],
        "backtest_symbol_summary": [
            ("cagr_pct",     "REAL"),
            ("max_drawdown", "REAL"),
            ("sharpe",       "REAL"),
            ("avg_hold",     "REAL"),
        ],
        "backtest_trade_log": [
            ("signal_type",    "TEXT"),
            ("ti65",           "REAL"),
            ("twolynch_score", "REAL"),
            ("market_regime",  "TEXT"),
        ],
    }

    for table, columns in required.items():
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            continue

        existing = {
            row[1] for row in con.execute(f"PRAGMA table_info({table})")
        }
        for col_name, col_type in columns:
            if col_name not in existing:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                log.info("Migration: added column %s.%s (%s)", table, col_name, col_type)


def init_db() -> None:
    with _conn() as con:
        _migrate_db(con)
        con.executescript(DDL)
    log.info("Database initialised: %s", DB_PATH)


# ─── Signal CRUD ──────────────────────────────────────────────────────────────

def upsert_signal(sig) -> None:
    """
    Persist a StockbeeSignal to the signals table.
    Accepts any object with a .to_dict() method.
    INSERT OR IGNORE — never overwrites existing signal tracking data.
    """
    if hasattr(sig, "to_dict"):
        d = sig.to_dict()
    elif hasattr(sig, "_asdict"):
        d = sig._asdict()
    else:
        d = vars(sig)

    row = {
        "signal_id":       d.get("signal_id", ""),
        "symbol":          d.get("symbol", ""),
        "signal_type":     d.get("signal_type", ""),
        "sector":          d.get("sector", ""),
        "scan_date":       str(d.get("scan_date", "")),
        "current_price":   d.get("current_price", d.get("entry_price", 0)),
        "entry_zone_low":  d.get("entry_zone_low", 0),
        "entry_zone_high": d.get("entry_zone_high", 0),
        "stop_loss":       d.get("stop_loss", 0),
        "target1":         d.get("target1", d.get("target_1", 0)),
        "target2":         d.get("target2", d.get("target_2", 0)),
        "target3":         d.get("target3", d.get("target_3", 0)),
        "atr":             d.get("atr", 0),
        "risk_per_share":  d.get("risk_per_share", 0),
        "position_size":   d.get("position_size", 0),
        "capital_required":d.get("capital_required", 0),
        "risk_amount":     d.get("risk_amount", 0),
        "rr_ratio":        d.get("rr_ratio", 0),
        "volume_ratio":    d.get("volume_ratio", 0),
        "rs_rank":         d.get("rs_rank", d.get("rs_rating", 0)),
        "ti65":            d.get("ti65", 0),
        "twolynch_score":  d.get("twolynch_score", 0),
        "composite_score": d.get("composite_score", 0),
        "classification":  d.get("classification", "Watch"),
        "market_regime":   d.get("market_regime", ""),
        "status":          "Waiting",
    }
    cols = ", ".join(row.keys())
    ph   = ", ".join(f":{k}" for k in row)
    with _conn() as con:
        con.execute(f"INSERT OR IGNORE INTO signals ({cols}) VALUES ({ph})", row)


def update_signal_status(signal_id: str, **kwargs) -> None:
    """Update one or more fields on an existing signal row."""
    if not kwargs:
        return
    sets = ", ".join(f"{k} = :{k}" for k in kwargs)
    kwargs["signal_id"]   = signal_id
    kwargs["last_checked"] = datetime.now().isoformat()
    with _conn() as con:
        con.execute(
            f"UPDATE signals SET {sets}, last_checked = :last_checked WHERE signal_id = :signal_id",
            kwargs,
        )


def signal_exists(signal_id: str) -> bool:
    with _conn() as con:
        row = con.execute(
            "SELECT 1 FROM signals WHERE signal_id = ?", (signal_id,)
        ).fetchone()
    return row is not None


def get_open_signals() -> list[dict]:
    """
    Return all signals that are not yet in a terminal state and need verification.

    FIX: Added parentheses around the OR clause to prevent SQL operator
    precedence bug. Without parens the query was:
      (status NOT IN (...) AND entry_triggered = 0) OR (status = 'Active')
    which pulled back every Active signal regardless of the first filter.
    Correct logic is:
      status NOT IN (...) AND (entry_triggered = 0 OR status = 'Active')
    """
    terminal = ("Target 1 Achieved", "Target 2 Achieved", "Target 3 Achieved",
                 "Stopped Out", "Expired")
    placeholders = ", ".join(f"'{s}'" for s in terminal)
    with _conn() as con:
        rows = con.execute(
            f"""SELECT * FROM signals
                WHERE status NOT IN ({placeholders})
                AND (entry_triggered = 0 OR status = 'Active')
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_signals_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql("SELECT * FROM signals ORDER BY scan_date DESC", con)


# ─── Watchlist CRUD ───────────────────────────────────────────────────────────

def upsert_watchlist(sig, expiry_date: Optional[date] = None) -> None:
    """Add a signal to the watchlist. INSERT OR IGNORE — never overwrites tracking."""
    if hasattr(sig, "to_dict"):
        d = sig.to_dict()
    elif hasattr(sig, "_asdict"):
        d = sig._asdict()
    else:
        d = vars(sig)

    from datetime import timedelta
    exp = expiry_date or (date.today() + timedelta(days=WATCHLIST_EXPIRY_DAYS))

    row = {
        "signal_id":     d.get("signal_id", ""),
        "symbol":        d.get("symbol", ""),
        "signal_type":   d.get("signal_type", ""),
        "entry_price":   d.get("current_price", d.get("entry_price", 0)),
        "stop_loss":     d.get("stop_loss", 0),
        "target1":       d.get("target1", d.get("target_1", 0)),
        "target2":       d.get("target2", d.get("target_2", 0)),
        "target3":       d.get("target3", d.get("target_3", 0)),
        "composite_score": d.get("composite_score", 0),
        "rs_rank":       d.get("rs_rank", d.get("rs_rating", 0)),
        "status":        "Waiting",
        "detected_date": str(d.get("scan_date", "")),
        "expiry_date":   str(exp),
        "last_updated":  datetime.now().isoformat(),
    }
    cols = ", ".join(row.keys())
    ph   = ", ".join(f":{k}" for k in row)
    with _conn() as con:
        con.execute(f"INSERT OR IGNORE INTO watchlist ({cols}) VALUES ({ph})", row)


def get_watchlist_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM watchlist ORDER BY composite_score DESC", con
        )


def update_watchlist_status(signal_id: str, status: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE watchlist SET status = ?, last_updated = ? WHERE signal_id = ?",
            (status, datetime.now().isoformat(), signal_id),
        )


# ─── Performance snapshots ────────────────────────────────────────────────────

def save_performance_snapshot(snap: dict) -> None:
    cols = ", ".join(snap.keys())
    ph   = ", ".join(f":{k}" for k in snap)
    with _conn() as con:
        con.execute(
            f"INSERT OR REPLACE INTO performance_snapshots ({cols}) VALUES ({ph})", snap
        )


def get_performance_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM performance_snapshots ORDER BY snapshot_date DESC", con
        )


# ─── Backtest run persistence ─────────────────────────────────────────────────

def save_backtest_run(run: dict) -> None:
    """Save a universe-wide backtest run summary."""
    cols = ", ".join(run.keys())
    ph   = ", ".join(f":{k}" for k in run)
    with _conn() as con:
        con.execute(
            f"INSERT OR REPLACE INTO backtest_runs ({cols}) VALUES ({ph})", run
        )


def save_backtest_symbol_summary(run_id: str, summaries: list[dict]) -> None:
    """Save per-symbol aggregate backtest results for a given run."""
    if not summaries:
        return
    with _conn() as con:
        for t in summaries:
            t["run_id"] = run_id
            cols = ", ".join(t.keys())
            ph   = ", ".join(f":{k}" for k in t)
            con.execute(
                f"INSERT OR REPLACE INTO backtest_symbol_summary ({cols}) VALUES ({ph})", t
            )


def save_backtest_trade_log(run_id: str, trades: list[dict]) -> None:
    """
    Save individual trade records — ground truth for all aggregate stats.
    Includes composite score / RS rank at entry time so we can validate
    whether higher-score signals actually outperform.
    """
    if not trades:
        return
    with _conn() as con:
        for t in trades:
            row = dict(t)
            row["run_id"] = run_id
            cols = ", ".join(row.keys())
            ph   = ", ".join(f":{k}" for k in row)
            con.execute(
                f"INSERT INTO backtest_trade_log ({cols}) VALUES ({ph})", row
            )


def get_backtest_runs_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM backtest_runs ORDER BY run_date DESC", con
        )


def get_backtest_symbol_summary_df(run_id: str) -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM backtest_symbol_summary WHERE run_id = ? ORDER BY cagr_pct DESC",
            con, params=(run_id,),
        )


def get_backtest_trade_log_df(run_id: str) -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql(
            "SELECT * FROM backtest_trade_log WHERE run_id = ? ORDER BY entry_date",
            con, params=(run_id,),
        )
