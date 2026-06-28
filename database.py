"""
NSE Darvas Box Scanner - Database Layer
========================================
SQLite-backed persistence for:
  • signals_history  – every signal ever generated
  • watchlist        – active / pending signals being tracked
  • performance      – strategy effectiveness statistics

Uses parameterised queries throughout; never uses f-strings in SQL.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import SIGNALS_DB, WATCHLIST_DB
from logger_utils import get_logger

log = get_logger("scanner")

DB_PATH = SIGNALS_DB   # single DB for simplicity

# ─── Schema ───────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id          TEXT PRIMARY KEY,
    symbol             TEXT NOT NULL,
    sector             TEXT,
    scan_date          TEXT NOT NULL,
    current_price      REAL,
    box_high           REAL,
    box_low            REAL,
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
    rsi_val            REAL,
    adx_val            REAL,
    volume_ratio       REAL,
    weekly_trend       TEXT,
    monthly_trend      TEXT,
    rs_rating          REAL,
    sepa_score         REAL,
    composite_score    REAL,
    classification     TEXT,
    box_age_bars       INTEGER,
    box_width_pct      REAL,
    box_quality        REAL,
    status             TEXT DEFAULT 'Waiting',
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
    box_high         REAL,
    box_low          REAL,
    entry_price      REAL,
    stop_loss        REAL,
    target1          REAL,
    target2          REAL,
    target3          REAL,
    score            REAL,
    rs_rating        REAL,
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
    run_id          TEXT PRIMARY KEY,
    run_date        TEXT NOT NULL,
    symbols_tested  INTEGER,
    symbols_with_trades INTEGER,
    total_trades    INTEGER,
    win_rate        REAL,
    profit_factor   REAL,
    expectancy      REAL,
    avg_cagr        REAL,
    avg_drawdown    REAL,
    avg_sharpe      REAL,
    avg_hold_days   REAL,
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- One row per SYMBOL within a backtest run (aggregated across all of
-- that symbol's historical trades). Useful for "which stocks were the
-- best/worst performers" analysis.
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

-- One row per INDIVIDUAL TRADE within a backtest run. This is the
-- ground truth — every other aggregate number in the report is derived
-- from this table. Includes the composite score, RS rating, and other
-- signal-quality metrics AT THE TIME the trade was taken, which is
-- what lets us answer "do high-score signals actually perform better."
CREATE TABLE IF NOT EXISTS backtest_trade_log (
    run_id          TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    entry_date      TEXT,
    exit_date       TEXT,
    entry_price     REAL,
    exit_price      REAL,
    stop_loss       REAL,
    target1         REAL,
    target2         REAL,
    outcome         TEXT,       -- 'target1_hit' / 'target2_hit' / 'stopped_out' / 'open_at_end'
    rr_realised     REAL,
    hold_days       INTEGER,
    composite_score REAL,
    rs_rating       REAL,
    sepa_score      REAL,
    rsi_at_entry    REAL,
    adx_at_entry    REAL,
    box_width_pct   REAL,
    box_age_bars    INTEGER,
    score_band      TEXT        -- 'elite' / 'very_strong' / 'strong' / 'watch'
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
    """Add any columns that exist in the current schema but are missing from
    an older on-disk database.  SQLite's CREATE TABLE IF NOT EXISTS never
    alters an existing table, so new columns must be added explicitly."""

    # Map table → list of (column_name, column_type) that should exist.
    # Keep in sync with the DDL above whenever new columns are added.
    required: dict[str, list[tuple[str, str]]] = {
        "performance_snapshots": [
            ("expectancy", "REAL"),
        ],
        "backtest_runs": [
            ("expectancy",  "REAL"),
            ("avg_cagr",    "REAL"),
            ("avg_drawdown","REAL"),
            ("avg_sharpe",  "REAL"),
            ("avg_hold_days","REAL"),
            ("notes",       "TEXT"),
        ],
        "backtest_symbol_summary": [
            ("cagr_pct",     "REAL"),
            ("max_drawdown", "REAL"),
            ("sharpe",       "REAL"),
            ("avg_hold",     "REAL"),
        ],
    }

    for table, columns in required.items():
        # Only bother if the table already exists
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            continue

        existing = {
            row[1]  # column name is index 1 in PRAGMA table_info rows
            for row in con.execute(f"PRAGMA table_info({table})")
        }
        for col_name, col_type in columns:
            if col_name not in existing:
                con.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"
                )
                log.info("Migration: added column %s.%s (%s)", table, col_name, col_type)


def init_db() -> None:
    with _conn() as con:
        _migrate_db(con)   # upgrade old schema BEFORE applying DDL
        con.executescript(DDL)
    log.info("Database initialised: %s", DB_PATH)


# ─── Signal CRUD ─────────────────────────────────────────────────────────────

def upsert_signal(sig) -> None:
    """Insert or replace a Signal object in signals table."""
    row = {
        "signal_id":       sig.signal_id,
        "symbol":          sig.symbol,
        "sector":          sig.sector,
        "scan_date":       sig.scan_date.isoformat(),
        "current_price":   sig.current_price,
        "box_high":        sig.box_high,
        "box_low":         sig.box_low,
        "entry_zone_low":  sig.entry_zone_low,
        "entry_zone_high": sig.entry_zone_high,
        "stop_loss":       sig.stop_loss,
        "target1":         sig.target1,
        "target2":         sig.target2,
        "target3":         sig.target3,
        "atr":             sig.atr,
        "risk_per_share":  sig.risk_per_share,
        "position_size":   sig.position_size,
        "capital_required":sig.capital_required,
        "risk_amount":     sig.risk_amount,
        "rr_ratio":        sig.rr_ratio,
        "rsi_val":         sig.rsi_val,
        "adx_val":         sig.adx_val,
        "volume_ratio":    sig.volume_ratio,
        "weekly_trend":    sig.weekly_trend,
        "monthly_trend":   sig.monthly_trend,
        "rs_rating":       sig.rs_rating,
        "sepa_score":      sig.sepa_score,
        "composite_score": sig.composite_score,
        "classification":  sig.classification,
        "box_age_bars":    sig.box_age_bars,
        "box_width_pct":   sig.box_width_pct,
        "box_quality":     sig.box_quality,
        "status":          sig.status,
    }
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row)
    sql = f"INSERT OR IGNORE INTO signals ({cols}) VALUES ({placeholders})"
    with _conn() as con:
        con.execute(sql, row)


def update_signal_status(signal_id: str, **kwargs) -> None:
    if not kwargs:
        return
    sets = ", ".join(f"{k} = :{k}" for k in kwargs)
    kwargs["signal_id"] = signal_id
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
    with _conn() as con:
        rows = con.execute(
            """SELECT * FROM signals
               WHERE status NOT IN ('Target 2 Achieved','Target 3 Achieved','Stopped Out','Expired')
               AND entry_triggered = 0 OR status = 'Active'
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_signals_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql("SELECT * FROM signals ORDER BY scan_date DESC", con)


# ─── Watchlist CRUD ───────────────────────────────────────────────────────────

def upsert_watchlist(sig, expiry_date: date) -> None:
    row = {
        "signal_id":     sig.signal_id,
        "symbol":        sig.symbol,
        "box_high":      sig.box_high,
        "box_low":       sig.box_low,
        "entry_price":   sig.current_price,
        "stop_loss":     sig.stop_loss,
        "target1":       sig.target1,
        "target2":       sig.target2,
        "target3":       sig.target3,
        "score":         sig.composite_score,
        "rs_rating":     sig.rs_rating,
        "status":        "Waiting",
        "detected_date": sig.scan_date.isoformat(),
        "expiry_date":   expiry_date.isoformat(),
        "last_updated":  datetime.now().isoformat(),
    }
    cols = ", ".join(row.keys())
    ph   = ", ".join(f":{k}" for k in row)
    with _conn() as con:
        con.execute(
            f"INSERT OR IGNORE INTO watchlist ({cols}) VALUES ({ph})", row
        )


def get_watchlist_df() -> pd.DataFrame:
    with _conn() as con:
        return pd.read_sql("SELECT * FROM watchlist ORDER BY score DESC", con)


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
    """Save per-symbol AGGREGATE backtest results for a given run."""
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
    Save individual trade records for a given run — the ground-truth
    trade-by-trade log that every aggregate stat in the report derives
    from. Each row includes the composite score / RS rating the trade
    had AT ENTRY, so we can later answer "do higher-score trades win
    more often" with real evidence instead of assumption.
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




# ─── Stockbee Signal compatibility shim ──────────────────────────────────────

def upsert_signal(sig) -> None:
    """
    Persist a StockbeeSignal (or legacy Signal) to the signals table.
    Accepts any object that has a .to_dict() method, or a legacy Signal namedtuple.
    """
    if hasattr(sig, "to_dict"):
        d = sig.to_dict()
    else:
        # Legacy darvas Signal namedtuple fallback
        d = sig._asdict() if hasattr(sig, "_asdict") else vars(sig)

    signal_id   = d.get("signal_id", "unknown")
    symbol      = d.get("symbol", "")
    scan_date   = d.get("signal_date", d.get("scan_date", ""))

    with _db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO signals (
                signal_id, symbol, scan_date,
                current_price, stop_loss, target1, target2,
                composite_score, classification, status,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Waiting', datetime('now'))
        """, (
            signal_id,
            symbol,
            str(scan_date),
            d.get("entry_price", d.get("current_price", 0)),
            d.get("stop_loss", 0),
            d.get("target_1", d.get("target1", 0)),
            d.get("target_2", d.get("target2", 0)),
            d.get("composite_score", 0),
            d.get("classification", "Watch"),
        ))


def upsert_watchlist(sig, expiry_date=None) -> None:
    """Persist signal to watchlist table."""
    if hasattr(sig, "to_dict"):
        d = sig.to_dict()
    else:
        d = sig._asdict() if hasattr(sig, "_asdict") else vars(sig)

    signal_id = d.get("signal_id", "unknown")
    symbol    = d.get("symbol", "")
    scan_date = d.get("signal_date", d.get("scan_date", ""))

    with _db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO watchlist (
                signal_id, symbol, scan_date,
                entry_price, stop_loss, target1, target2,
                composite_score, classification,
                expiry_date, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Watching')
        """, (
            signal_id, symbol, str(scan_date),
            d.get("entry_price", 0),
            d.get("stop_loss", 0),
            d.get("target_1", 0),
            d.get("target_2", 0),
            d.get("composite_score", 0),
            d.get("classification", "Watch"),
            str(expiry_date) if expiry_date else None,
        ))


def signal_exists(signal_id: str) -> bool:
    """Return True if signal_id already in DB."""
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT 1 FROM signals WHERE signal_id = ?", (signal_id,)
            ).fetchone()
            return row is not None
    except Exception:
        return False

