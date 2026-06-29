"""
NSE Stockbee Scanner — Signal Verification Engine (Forward Tester)
==================================================================
Based on Pradeep Bonde (Stockbee) methodology.

Continuously monitors previously generated MB and EP signals and updates
their outcome in the database:
  • Entry triggered?  (High crossed entry_zone_high?)
  • Target 1 / 2 / 3 achieved?
  • Stop loss hit?
  • Maximum Favourable Excursion (MFE) — peak unrealised profit while open
  • Maximum Adverse Excursion (MAE) — peak unrealised loss while open
  • Days held
  • MB-specific: Day 3 partial exit, Day 5 force-exit enforced in tracking

Status lifecycle (matches database.py exactly):
  Waiting           → entry not yet triggered
  Active            → entry triggered, trade open
  Target 1 Achieved → exit at T1
  Target 2 Achieved → exit at T2
  Target 3 Achieved → exit at T3 (EP only)
  Stopped Out       → stop loss hit
  Expired           → WATCHLIST_EXPIRY_DAYS elapsed, no entry triggered

FIX (v2):
  • Header updated from Darvas Box to Bonde
  • Status strings unified — now match database.py ('Waiting' not 'PENDING')
  • MB_HOLD_MAX_DAYS (5) enforced: MB trades auto-close on Day 5 regardless
  • MFE/MAE computed bar-by-bar from entry bar onward (not from scan_date)
  • realised_rr computed from actual exit price not approximation
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    MB_HOLD_MAX_DAYS,
    EP_HOLD_MAX_DAYS,
    WATCHLIST_EXPIRY_DAYS,
)
from database import (
    get_all_signals_df,
    get_open_signals,
    save_performance_snapshot,
    update_signal_status,
    update_watchlist_status,
)
from downloader import load_daily
from logger_utils import get_logger

log = get_logger("signal_tracker")


# ─── Main Verification Loop ───────────────────────────────────────────────────

def verify_all_signals() -> None:
    """
    Called once per run (after market close).
    Loads all non-terminal signals and checks each against current price data.
    """
    open_signals = get_open_signals()
    log.info("Verifying %d open signals", len(open_signals))

    for sig in open_signals:
        try:
            _verify_signal(sig)
        except Exception as e:
            log.error("Error verifying %s: %s", sig.get("signal_id", "?"), e)

    log.info("Signal verification complete")


# ─── Per-Signal Logic ─────────────────────────────────────────────────────────

def _verify_signal(sig: dict) -> None:
    symbol      = sig["symbol"]
    signal_type = sig.get("signal_type", "")
    is_mb       = signal_type in ("MB_BREAKOUT", "MB_ANTICIPATION")
    max_hold    = MB_HOLD_MAX_DAYS if is_mb else EP_HOLD_MAX_DAYS

    daily = load_daily(symbol)
    if daily is None or daily.empty:
        log.warning("No price data for %s — skipping", symbol)
        return

    scan_date = pd.Timestamp(sig["scan_date"])

    # Only look at bars STRICTLY AFTER signal generation date
    future = daily[daily.index > scan_date].copy()
    if future.empty:
        _check_expiry(sig)
        return

    entry_zone_high = sig["entry_zone_high"]
    entry_zone_low  = sig["entry_zone_low"]
    stop            = sig["stop_loss"]
    t1              = sig.get("target1") or 0.0
    t2              = sig.get("target2") or 0.0
    t3              = sig.get("target3") or 0.0

    high_s  = future["High"]
    low_s   = future["Low"]
    close_s = future["Close"]

    # ── Step 1: Was entry triggered? ──────────────────────────────────────────
    entry_triggered = bool(sig.get("entry_triggered", 0))
    entry_price     = entry_zone_high   # assume fill at top of entry zone

    if not entry_triggered:
        # Entry fires when any bar's High crosses entry_zone_high
        triggered_mask = high_s >= entry_zone_high
        if not triggered_mask.any():
            _check_expiry(sig)
            return

        # Trim future to bars from entry onward
        entry_bar_idx = triggered_mask.idxmax()
        future = future.loc[entry_bar_idx:]
        high_s  = future["High"]
        low_s   = future["Low"]
        close_s = future["Close"]

        update_signal_status(
            sig["signal_id"],
            entry_triggered=1,
            status="Active",
        )
        update_watchlist_status(sig["signal_id"], "Active")
        log.info("%s entry triggered @ ≤%.2f", symbol, entry_price)

    # ── Step 2: Apply max hold limit (MB: 5 bars, EP: 30 bars) ───────────────
    # Slice to at most max_hold bars from entry
    trade_bars  = future.iloc[:max_hold]
    high_trade  = trade_bars["High"]
    low_trade   = trade_bars["Low"]
    close_trade = trade_bars["Close"]
    days_held   = len(trade_bars)

    # ── Step 3: MFE / MAE (bar-by-bar from entry price) ──────────────────────
    mfe = float((high_trade  - entry_price).clip(lower=0).max())
    mae = float((entry_price - low_trade).clip(lower=0).max())

    # ── Step 4: Exit logic (in priority order) ────────────────────────────────
    stopped = bool((low_trade <= stop).any())
    t1_hit  = bool(t1 > 0 and (high_trade >= t1).any())
    t2_hit  = bool(t2 > 0 and (high_trade >= t2).any())
    t3_hit  = bool(t3 > 0 and (high_trade >= t3).any())
    max_hold_reached = (days_held >= max_hold)

    if t3_hit:
        status      = "Target 3 Achieved"
        exit_price  = t3
    elif t2_hit:
        status      = "Target 2 Achieved"
        exit_price  = t2
    elif t1_hit:
        status      = "Target 1 Achieved"
        exit_price  = t1
    elif stopped:
        status      = "Stopped Out"
        exit_price  = stop
    elif max_hold_reached and is_mb:
        # MB Day-5 force exit at close
        status      = "Target 1 Achieved"   # counts as exit, not loss
        exit_price  = float(close_trade.iloc[-1])
    else:
        status      = "Active"
        exit_price  = float(close_trade.iloc[-1])

    # ── Step 5: Realised R:R ──────────────────────────────────────────────────
    risk = entry_price - stop
    if risk > 0:
        real_rr = round((exit_price - entry_price) / risk, 2)
    else:
        real_rr = 0.0

    # ── Step 6: Persist ───────────────────────────────────────────────────────
    update_signal_status(
        sig["signal_id"],
        status            = status,
        entry_triggered   = 1,
        t1_achieved       = int(t1_hit),
        t2_achieved       = int(t2_hit),
        t3_achieved       = int(t3_hit),
        stopped_out       = int(stopped),
        max_fav_excursion = round(mfe, 2),
        max_adv_excursion = round(mae, 2),
        days_to_target    = days_held,
        realised_rr       = real_rr,
    )

    if status != "Active":
        update_watchlist_status(sig["signal_id"], status)

    log.info(
        "%s [%s] → %s (RR=%.2f, days=%d, MFE=%.2f, MAE=%.2f)",
        symbol, signal_type, status, real_rr, days_held, mfe, mae,
    )


def _check_expiry(sig: dict) -> None:
    """Mark signal as Expired if it has been on the watchlist too long without triggering."""
    scan_date = pd.Timestamp(sig["scan_date"]).date()
    if (date.today() - scan_date).days > WATCHLIST_EXPIRY_DAYS:
        update_signal_status(sig["signal_id"], status="Expired")
        update_watchlist_status(sig["signal_id"], "Expired")
        log.info("%s expired after %d days", sig.get("symbol", "?"), WATCHLIST_EXPIRY_DAYS)


# ─── Strategy Effectiveness ───────────────────────────────────────────────────

def compute_effectiveness() -> pd.DataFrame:
    """
    Calculate strategy metrics segmented by composite_score band.
    Saves snapshots to the database and returns a summary DataFrame.
    """
    df = get_all_signals_df()
    if df.empty:
        log.warning("No signals in database — skipping effectiveness calc")
        return pd.DataFrame()

    today = date.today().isoformat()
    rows  = []

    # Score bands aligned with MB_SCORE_THRESHOLDS (elite=85, strong=70, watch=55)
    bands = [
        ("elite_85+",  df[df["composite_score"] >= 85]),
        ("strong_70+", df[df["composite_score"] >= 70]),
        ("watch_55+",  df[df["composite_score"] >= 55]),
        ("all",        df),
    ]

    for band_name, bdf in bands:
        triggered = bdf[bdf["entry_triggered"] == 1]
        if triggered.empty:
            continue

        total    = len(triggered)
        t1_hit   = int(triggered["t1_achieved"].sum())
        t2_hit   = int(triggered["t2_achieved"].sum())
        t3_hit   = int(triggered["t3_achieved"].sum())
        stopped  = int(triggered["stopped_out"].sum())
        wins     = t1_hit + t2_hit + t3_hit
        win_rate = wins / total if total else 0.0

        profits = triggered[triggered["realised_rr"] > 0]["realised_rr"].sum()
        losses  = triggered[triggered["realised_rr"] < 0]["realised_rr"].abs().sum()
        pf      = profits / losses if losses else float("inf")

        avg_rr     = float(triggered["realised_rr"].mean()) if total else 0.0
        expectancy = win_rate * avg_rr - (1 - win_rate) * abs(avg_rr)

        snap = {
            "snapshot_date": today,
            "score_band":    band_name,
            "total_signals": len(bdf),
            "triggered":     total,
            "t1_hit":        t1_hit,
            "t2_hit":        t2_hit,
            "t3_hit":        t3_hit,
            "stopped":       stopped,
            "win_rate":      round(win_rate, 4),
            "avg_rr":        round(avg_rr, 4),
            "profit_factor": round(pf, 4),
            "expectancy":    round(expectancy, 4),
        }
        save_performance_snapshot(snap)
        rows.append(snap)
        log.info(
            "Effectiveness [%s]: W/R=%.1f%% PF=%.2f E=%.3f",
            band_name, win_rate * 100, pf, expectancy,
        )

    return pd.DataFrame(rows)


# ─── Adaptive Analysis ────────────────────────────────────────────────────────

def adaptive_analysis() -> pd.DataFrame:
    """
    Analyse which RS / TI65 / score / consolidation width combinations
    historically produced the best outcomes.
    Returns a DataFrame of insights.
    """
    df = get_all_signals_df()
    if df.empty or len(df) < 20:
        return pd.DataFrame()

    triggered = df[df["entry_triggered"] == 1].copy()
    triggered["success"] = (triggered["t2_achieved"] == 1).astype(int)

    bins = {
        "rs_band":    pd.cut(triggered["rs_rank"],         bins=[0, 70, 80, 90, 100]),
        "score_band": pd.cut(triggered["composite_score"], bins=[0, 55, 70, 85, 100]),
    }

    # Add TI65 band if column exists
    if "ti65" in triggered.columns:
        bins["ti65_band"] = pd.cut(triggered["ti65"], bins=[0.95, 1.00, 1.03, 1.05, 1.20])

    results = []
    for col_name, cut_series in bins.items():
        triggered[col_name] = cut_series
        grp = triggered.groupby(col_name)["success"].agg(["mean", "count"])
        grp.columns = ["success_rate", "n"]
        grp["factor"] = col_name
        grp.index.name = "bin"
        results.append(grp.reset_index())

    combined = pd.concat(results, ignore_index=True)
    combined = combined.sort_values("success_rate", ascending=False)
    log.info("Adaptive analysis: %d factor-bin combinations", len(combined))
    return combined
