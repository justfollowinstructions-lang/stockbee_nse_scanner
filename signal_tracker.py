"""
NSE Darvas Box Scanner - Signal Verification Engine
=====================================================
Continuously monitors previously generated signals and updates
their outcome in the database:
  • Entry triggered?
  • Target 1 / 2 / 3 achieved?
  • Stop loss hit?
  • Maximum Favourable / Adverse Excursion
  • Days held

Also computes strategy-effectiveness metrics broken down by score band.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config import WATCHLIST_EXPIRY_DAYS
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
    Loads all non-terminal signals and checks each one against
    current price data.
    """
    open_signals = get_open_signals()
    log.info("Verifying %d open signals", len(open_signals))

    for sig in open_signals:
        try:
            _verify_signal(sig)
        except Exception as e:
            log.error("Error verifying %s: %s", sig["signal_id"], e)

    log.info("Signal verification complete")


# ─── Per-Signal Logic ────────────────────────────────────────────────────────

def _verify_signal(sig: dict) -> None:
    symbol = sig["symbol"]
    daily  = load_daily(symbol)
    if daily is None or daily.empty:
        return

    scan_date = pd.Timestamp(sig["scan_date"])
    # Only look at bars AFTER the signal was generated
    future = daily[daily.index > scan_date]
    if future.empty:
        _check_expiry(sig)
        return

    entry  = sig["entry_zone_high"]   # we assume fill at top of entry zone
    stop   = sig["stop_loss"]
    t1, t2, t3 = sig["target1"], sig["target2"], sig["target3"]

    high_series  = future["High"]
    low_series   = future["Low"]
    close_series = future["Close"]

    # Was entry triggered? (close entered entry zone)
    entry_triggered = sig["entry_triggered"]
    if not entry_triggered:
        if (close_series <= sig["entry_zone_high"]).any() and \
           (close_series >= sig["entry_zone_low"]).any():
            entry_triggered = 1
            update_signal_status(sig["signal_id"], entry_triggered=1, status="Active")
            update_watchlist_status(sig["signal_id"], "Active")
            log.info("%s entry triggered", symbol)
        else:
            _check_expiry(sig)
            return

    # From here, entry is active – track outcomes
    mfe = float((high_series  - entry).clip(lower=0).max())   # max fav. excursion
    mae = float((entry - low_series).clip(lower=0).max())     # max adv. excursion

    stopped = int((low_series <= stop).any())
    t1_hit  = int((high_series >= t1).any())
    t2_hit  = int((high_series >= t2).any())
    t3_hit  = int((high_series >= t3).any())

    # Determine final status (in priority order)
    if t3_hit:
        status = "Target 3 Achieved"
    elif t2_hit:
        status = "Target 2 Achieved"
    elif t1_hit:
        status = "Target 1 Achieved"
    elif stopped:
        status = "Stopped Out"
    else:
        status = "Active"

    # Days held
    days = len(future)

    # Realised R:R
    realised_price = (
        t2 if t2_hit else t1 if t1_hit else
        stop if stopped else float(close_series.iloc[-1])
    )
    risk  = entry - stop
    real_rr = round((realised_price - entry) / risk, 2) if risk > 0 else 0.0

    update_signal_status(
        sig["signal_id"],
        status             = status,
        entry_triggered    = entry_triggered,
        t1_achieved        = t1_hit,
        t2_achieved        = t2_hit,
        t3_achieved        = t3_hit,
        stopped_out        = stopped,
        max_fav_excursion  = round(mfe, 2),
        max_adv_excursion  = round(mae, 2),
        days_to_target     = days,
        realised_rr        = real_rr,
    )

    if status not in ("Active",):
        update_watchlist_status(sig["signal_id"], status)

    log.info("%s → %s (RR=%.2f, days=%d)", symbol, status, real_rr, days)


def _check_expiry(sig: dict) -> None:
    scan_date = pd.Timestamp(sig["scan_date"]).date()
    if (date.today() - scan_date).days > WATCHLIST_EXPIRY_DAYS:
        update_signal_status(sig["signal_id"], status="Expired")
        update_watchlist_status(sig["signal_id"], "Expired")


# ─── Strategy Effectiveness ───────────────────────────────────────────────────

def compute_effectiveness() -> pd.DataFrame:
    """
    Calculate strategy metrics segmented by composite_score band.
    Saves snapshots to the database and returns a summary DataFrame.
    """
    df = get_all_signals_df()
    if df.empty:
        log.warning("No signals in database – skipping effectiveness calc")
        return pd.DataFrame()

    today = date.today().isoformat()
    rows  = []

    bands = [
        ("score_90+",  df[df["composite_score"] >= 90]),
        ("score_80+",  df[df["composite_score"] >= 80]),
        ("score_70+",  df[df["composite_score"] >= 70]),
        ("all",        df),
    ]

    for band_name, bdf in bands:
        triggered = bdf[bdf["entry_triggered"] == 1]
        if triggered.empty:
            continue

        total       = len(triggered)
        t1_hit      = int(triggered["t1_achieved"].sum())
        t2_hit      = int(triggered["t2_achieved"].sum())
        t3_hit      = int(triggered["t3_achieved"].sum())
        stopped     = int(triggered["stopped_out"].sum())
        wins        = t2_hit + t3_hit
        win_rate    = wins / total if total else 0.0

        # Profit factor = gross profit / gross loss
        profits = triggered[triggered["realised_rr"] > 0]["realised_rr"].sum()
        losses  = triggered[triggered["realised_rr"] < 0]["realised_rr"].abs().sum()
        pf      = profits / losses if losses else float("inf")

        avg_rr    = float(triggered["realised_rr"].mean()) if total else 0.0
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
        log.info("Effectiveness [%s]: W/R=%.1f%% PF=%.2f E=%.3f",
                 band_name, win_rate * 100, pf, expectancy)

    return pd.DataFrame(rows)


# ─── Adaptive Analysis ───────────────────────────────────────────────────────

def adaptive_analysis() -> pd.DataFrame:
    """
    Analyse which RS/ADX/RSI/box-width combinations historically
    produced the best outcomes.  Returns a DataFrame of insights.
    """
    df = get_all_signals_df()
    if df.empty or len(df) < 20:
        return pd.DataFrame()

    triggered = df[df["entry_triggered"] == 1].copy()
    triggered["success"] = (triggered["t2_achieved"] == 1).astype(int)

    bins = {
        "rs_band":    pd.cut(triggered["rs_rating"],   bins=[0, 70, 80, 90, 100]),
        "adx_band":   pd.cut(triggered["adx_val"],     bins=[0, 20, 30, 40, 60]),
        "rsi_band":   pd.cut(triggered["rsi_val"],     bins=[30, 38, 44, 50, 60]),
        "width_band": pd.cut(triggered["box_width_pct"], bins=[0, 10, 20, 30, 50]),
    }

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
    log.info("Adaptive analysis complete – %d factor-bin combinations", len(combined))
    return combined
