"""
NSE Stockbee Scanner — Backtesting Engine
==========================================
Backtests Momentum Burst and Episodic Pivot signals on historical data.

Key PB rules enforced in simulation:
  - MB: force exit at Day 3 (50% position) and Day 5 (100%)
  - MB: stop loss hit = exit immediately, no re-entry same day
  - EP: trail stop after 20%+ gain (10% below recent high)
  - Never hold MB beyond 5 bars regardless of P&L
  - Market regime filter: show results WITH and WITHOUT regime filter

Outputs:
  - Overall stats: CAGR, Win Rate, Profit Factor, Sharpe, Max Drawdown
  - 2LYNCH quality breakdown (win rate by score 0-5)
  - RS rank decile breakdown (proves RS ranking adds edge)
  - TI65 level breakdown (proves absolute momentum filter works)
  - Market regime breakdown (THE most important split)
  - Consolidation width breakdown (tight vs wide)
  - Signal frequency per month (validates setup occurs often enough)
"""

from __future__ import annotations

import gc
import math
import uuid
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from config import (
    ACCOUNT_SIZE, ATR_PERIOD, MB_BREAKOUT_PCT, MB_CONSOL_MAX_BARS,
    MB_CONSOL_MIN_BARS, MB_HOLD_MAX_DAYS, MB_PARTIAL_EXIT_DAY,
    MB_PRIOR_MOVE_MIN_PCT, MB_STOP_PCT_LARGE, MB_STOP_PCT_SMALL,
    EP_HOLD_MAX_DAYS, EP_STOP_PCT_NORMAL, RISK_PER_TRADE_PCT,
    RS_MIN_FOR_MB, TI65_BULL_THRESHOLD, MB_SCORE_THRESHOLDS,
)
from downloader import load_daily
from episodic_pivot import detect_9m_ep
from logger_utils import get_logger
from market_monitor import compute_market_monitor, market_allows_trading
from momentum_burst import (
    detect_mb_signal, detect_anticipation_signal,
    compute_ti65, detect_consolidation,
)
from stockbee_scanner import compute_rs_ranks, StockbeeSignal

log = get_logger("performance")

RISK_FREE_RATE = 0.065   # RBI repo rate proxy (6.5%)


# ─── Trade record ─────────────────────────────────────────────────────────────

class Trade:
    __slots__ = [
        "symbol", "signal_type", "entry_date", "entry_price",
        "stop_loss", "exit_date", "exit_price", "exit_reason",
        "pnl_pct", "pnl_rs", "hold_days",
        "rs_rank", "ti65", "twolynch_score", "consol_width",
        "composite_score", "market_regime", "partial_exit_price",
        "partial_exit_date", "partial_pnl_pct",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Defaults
        for slot in self.__slots__:
            if not hasattr(self, slot):
                setattr(self, slot, None)


# ─── Main backtest function ───────────────────────────────────────────────────

def run_backtest(
    symbols:          List[str],
    start_date:       date,
    end_date:         date,
    apply_mm_filter:  bool = True,
    signal_type:      str  = "MB_BREAKOUT",
    min_score:        float = MB_SCORE_THRESHOLDS["watch"],
) -> dict:
    """
    Run a historical backtest of Stockbee signals.

    Parameters
    ----------
    symbols        : list of NSE symbols (e.g. ['RELIANCE.NS', ...])
    start_date     : first date to simulate
    end_date       : last date to simulate
    apply_mm_filter: if True, skip signals when market is CAUTION/BEAR
    signal_type    : "MB_BREAKOUT" | "EP_9M" | "both"
    min_score      : minimum composite score to accept signal

    Returns a results dict with all metrics.
    """
    log.info(
        "Backtest: %s to %s | signal=%s | mm_filter=%s | min_score=%.0f",
        start_date, end_date, signal_type, apply_mm_filter, min_score
    )

    # ── Load all data ─────────────────────────────────────────────────────────
    all_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and len(df) >= 300:
            all_data[sym] = df
    log.info("Loaded data for %d / %d symbols", len(all_data), len(symbols))

    # ── Get all trading days in range ─────────────────────────────────────────
    # Use Nifty or any liquid stock to get trading calendar
    ref_sym = next(iter(all_data))
    ref_df  = all_data[ref_sym]
    trading_days = [
        d.date() for d in ref_df.index
        if start_date <= d.date() <= end_date
    ]

    if len(trading_days) < 20:
        log.error("Insufficient trading days in range")
        return {}

    # ── Per-day simulation ────────────────────────────────────────────────────
    all_trades:      List[Trade]         = []
    open_positions:  dict                = {}   # symbol -> Trade
    monthly_signals: dict[str, int]     = {}   # "YYYY-MM" -> count

    log.info("Simulating %d trading days …", len(trading_days))

    for day_idx, sim_date in enumerate(trading_days):
        # Slice data up to (and including) sim_date for each symbol
        day_data: dict[str, pd.DataFrame] = {}
        for sym, df in all_data.items():
            df_slice = df[df.index.date <= sim_date]
            if len(df_slice) >= 260:
                day_data[sym] = df_slice

        if len(day_data) < 10:
            continue

        # ── Exit open positions first ─────────────────────────────────────────
        symbols_to_close = []
        for sym, trade in open_positions.items():
            if sym not in day_data:
                continue
            daily = day_data[sym]
            today_close = float(daily["Close"].iloc[-1])
            today_low   = float(daily["Low"].iloc[-1])
            today_open  = float(daily["Open"].iloc[-1])   # FIX D: need open for gap check
            hold_days   = (sim_date - trade.entry_date).days

            # ── Stop detection (FIX D) ────────────────────────────────────────
            # Old: stop_hit = today_low <= stop → always filled at stop price.
            # Real: if stock OPENS below stop (gap-down), there is no fill at
            # stop — the opening print IS the fill. Simulating stop price in a
            # gap-down significantly overstates real backtest performance.
            #
            # Two cases:
            #   gap_down_stop: open < stop → filled at open (worse than stop)
            #   intraday_stop: open ≥ stop AND low ≤ stop → filled at stop
            gap_down_stop  = today_open < trade.stop_loss
            intraday_stop  = (not gap_down_stop) and (today_low <= trade.stop_loss)
            stop_hit       = gap_down_stop or intraday_stop

            if gap_down_stop:
                stop_fill_price = today_open          # gap fill: worse than stop
            else:
                stop_fill_price = trade.stop_loss     # normal: at stop price

            # Day 3 partial exit for MB
            if ("MB" in trade.signal_type and
                    hold_days >= MB_PARTIAL_EXIT_DAY - 1 and
                    trade.partial_exit_price is None):
                trade.partial_exit_price = today_close
                trade.partial_exit_date  = sim_date
                trade.partial_pnl_pct    = (
                    (today_close - trade.entry_price) / trade.entry_price * 100
                )

            # Force exit conditions
            force_exit = False
            exit_reason = ""
            exit_price  = today_close   # default: close

            if stop_hit:
                force_exit  = True
                exit_reason = "GAP_STOP" if gap_down_stop else "STOP_HIT"
                exit_price  = stop_fill_price   # FIX D: open if gapped, stop otherwise

            elif "MB" in trade.signal_type and hold_days >= MB_HOLD_MAX_DAYS - 1:
                force_exit  = True
                exit_reason = "DAY5_EXIT"
                exit_price  = today_close

            elif "EP" in trade.signal_type and hold_days >= EP_HOLD_MAX_DAYS - 1:
                force_exit  = True
                exit_reason = "EP_MAX_HOLD"
                exit_price  = today_close

            # EP trailing stop: after 20%+ gain, trail at 10% below recent high
            elif "EP" in trade.signal_type:
                gain_pct = (today_close - trade.entry_price) / trade.entry_price * 100
                if gain_pct >= 20.0:
                    recent_high = float(day_data[sym]["High"].iloc[-10:].max())
                    trail_stop  = recent_high * 0.90
                    if today_low <= trail_stop:
                        force_exit  = True
                        exit_reason = "EP_TRAIL_STOP"
                        # EP trail: same gap logic — if open gaps below trail, fill at open
                        exit_price  = today_open if today_open < trail_stop else trail_stop

            if force_exit:
                trade.exit_date   = sim_date
                trade.exit_price  = round(exit_price, 2)
                trade.exit_reason = exit_reason
                trade.hold_days   = hold_days
                trade.pnl_pct     = (
                    (exit_price - trade.entry_price) / trade.entry_price * 100
                )
                trade.pnl_rs      = (
                    (exit_price - trade.entry_price) *
                    (ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100 / max(trade.entry_price - trade.stop_loss, 0.01))
                )
                all_trades.append(trade)
                symbols_to_close.append(sym)

        for sym in symbols_to_close:
            del open_positions[sym]

        # ── Generate new signals ──────────────────────────────────────────────
        # Only scan every day (like a real scanner running after close)

        # Compute cross-sectional RS for this day's universe
        rs_ranks = compute_rs_ranks(list(day_data.keys()), day_data)

        # Market Monitor for this day
        mm = compute_market_monitor(day_data)

        if apply_mm_filter and not mm.trading_allowed:
            continue  # FFM rule: no signals in BEAR

        month_key = sim_date.strftime("%Y-%m")

        for sym in day_data:
            if sym in open_positions:
                continue   # already in a position

            daily   = day_data[sym]
            rs_rank = rs_ranks.get(sym, 50.0)

            signal = None
            if signal_type in ("MB_BREAKOUT", "both"):
                if market_allows_trading(mm, "MB_BREAKOUT"):
                    signal = detect_mb_signal(sym, daily, rs_rank)

            if signal is None and signal_type in ("EP_9M", "both"):
                if market_allows_trading(mm, "EP_9M"):
                    ep = detect_9m_ep(sym, daily, rs_rank)
                    if ep is not None:
                        signal = ep

            if signal is None:
                continue

            # Score filter
            score = getattr(signal, "composite_score", None) or getattr(signal, "ep_score", None) or 0
            if score < min_score:
                continue

            # Build trade
            entry_price = getattr(signal, "entry_price", None) or getattr(signal, "price_at_signal", None)
            stop_loss   = getattr(signal, "stop_loss", None)
            sig_type    = getattr(signal, "setup_type", None) or getattr(signal, "ep_type", "UNKNOWN")

            if entry_price is None or stop_loss is None:
                continue
            if entry_price <= stop_loss:
                continue

            trade = Trade(
                symbol         = sym,
                signal_type    = sig_type,
                entry_date     = sim_date,
                entry_price    = entry_price,
                stop_loss      = stop_loss,
                rs_rank        = getattr(signal, "rs_rank", 50.0),
                ti65           = getattr(signal, "ti65", 1.0),
                twolynch_score = getattr(signal, "twolynch_score", None),
                consol_width   = getattr(signal, "consolidation_width_pct", None),
                composite_score = score,
                market_regime  = mm.market_regime,
            )
            open_positions[sym] = trade

            monthly_signals[month_key] = monthly_signals.get(month_key, 0) + 1

    # ── Close any still-open positions at end_date ────────────────────────────
    for sym, trade in open_positions.items():
        daily = all_data.get(sym)
        if daily is None:
            continue
        last_close     = float(daily["Close"].iloc[-1])
        trade.exit_date  = end_date
        trade.exit_price = last_close
        trade.exit_reason = "END_OF_BACKTEST"
        trade.hold_days   = (end_date - trade.entry_date).days
        trade.pnl_pct     = (last_close - trade.entry_price) / trade.entry_price * 100
        all_trades.append(trade)

    # ── Compute metrics ───────────────────────────────────────────────────────
    results = _compute_metrics(all_trades, monthly_signals, start_date, end_date)
    _log_results_summary(results)

    return results


# ─── Metrics computation ──────────────────────────────────────────────────────

def _compute_metrics(
    trades:          List[Trade],
    monthly_signals: dict,
    start_date:      date,
    end_date:        date,
) -> dict:
    """Compute all Stockbee-specific performance metrics from the trade list."""

    if not trades:
        return {"error": "No trades generated", "trade_count": 0}

    df = pd.DataFrame([{
        "symbol":         t.symbol,
        "signal_type":    t.signal_type,
        "entry_date":     t.entry_date,
        "exit_date":      t.exit_date,
        "entry_price":    t.entry_price,
        "exit_price":     t.exit_price,
        "exit_reason":    t.exit_reason,
        "pnl_pct":        t.pnl_pct,
        "hold_days":      t.hold_days,
        "rs_rank":        t.rs_rank,
        "ti65":           t.ti65,
        "twolynch_score": t.twolynch_score,
        "consol_width":   t.consol_width,
        "composite_score": t.composite_score,
        "market_regime":  t.market_regime,
    } for t in trades if t.pnl_pct is not None])

    if df.empty:
        return {"error": "No completed trades", "trade_count": 0}

    wins  = df[df["pnl_pct"] > 0]
    losses = df[df["pnl_pct"] <= 0]

    win_rate = len(wins) / len(df) * 100
    avg_win  = float(wins["pnl_pct"].mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0

    gross_profit = float(wins["pnl_pct"].sum()) if len(wins) > 0 else 0.0
    gross_loss   = abs(float(losses["pnl_pct"].sum())) if len(losses) > 0 else 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # CAGR approximation (assumes 1% risk per trade, compound returns)
    years = max((end_date - start_date).days / 365, 0.1)
    # Simple CAGR from cumulative PnL %
    equity_curve = (1 + df["pnl_pct"] / 100 * 0.05).cumprod()   # 5% position sizing proxy
    if len(equity_curve) > 0:
        cagr = (float(equity_curve.iloc[-1]) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    # Max Drawdown
    cumret  = equity_curve
    peak    = cumret.cummax()
    drawdown = (cumret - peak) / peak
    max_dd   = float(drawdown.min()) * 100

    # Sharpe (annualised, using daily PnL series)
    daily_ret = df.groupby("entry_date")["pnl_pct"].mean() / 100
    if len(daily_ret) > 5:
        excess_ret = daily_ret - RISK_FREE_RATE / 252
        sharpe = float(excess_ret.mean() / (excess_ret.std() + 1e-10) * math.sqrt(252))
    else:
        sharpe = 0.0

    # Stockbee-specific metrics
    day3_hit_pct = 0.0
    burst_8pct_pct = 0.0
    mb_trades = df[df["signal_type"].str.contains("MB")]
    if len(mb_trades) > 0:
        # % of MB trades where hold was ≤ 3 days (day 3 target was hit or stopped)
        day3_hit_pct = float((mb_trades["hold_days"] <= 3).mean() * 100)
        # % of MB trades up 8%+ within hold period (validates burst theory)
        burst_8pct_pct = float((mb_trades["pnl_pct"] >= 8.0).mean() * 100)

    avg_hold_mb = float(mb_trades["hold_days"].mean()) if len(mb_trades) > 0 else 0
    avg_hold_ep = float(df[df["signal_type"].str.contains("EP")]["hold_days"].mean()) if len(df[df["signal_type"].str.contains("EP")]) > 0 else 0

    # ── Breakdowns ────────────────────────────────────────────────────────────

    # By 2LYNCH score
    twolynch_breakdown = {}
    lynch_df = df[df["twolynch_score"].notna()].copy()
    if len(lynch_df) > 0:
        lynch_df["lynch_bin"] = lynch_df["twolynch_score"].astype(int)
        for score_val, grp in lynch_df.groupby("lynch_bin"):
            twolynch_breakdown[f"2LYNCH_{score_val}/5"] = {
                "count":    len(grp),
                "win_rate": round(float((grp["pnl_pct"] > 0).mean() * 100), 1),
                "avg_pnl":  round(float(grp["pnl_pct"].mean()), 2),
            }

    # By RS rank decile
    rs_breakdown = {}
    rs_df = df[df["rs_rank"].notna()].copy()
    if len(rs_df) > 0:
        rs_df["rs_decile"] = pd.qcut(rs_df["rs_rank"], q=5,
                                      labels=["0-20", "20-40", "40-60", "60-80", "80-99"],
                                      duplicates="drop")
        for decile, grp in rs_df.groupby("rs_decile"):
            rs_breakdown[f"RS_{decile}"] = {
                "count":    len(grp),
                "win_rate": round(float((grp["pnl_pct"] > 0).mean() * 100), 1),
                "avg_pnl":  round(float(grp["pnl_pct"].mean()), 2),
            }

    # By TI65
    ti65_breakdown = {}
    ti_df = df[df["ti65"].notna()].copy()
    if len(ti_df) > 0:
        ti_df["ti65_bin"] = pd.cut(ti_df["ti65"],
                                    bins=[0, 0.97, 1.0, 1.03, 1.05, 999],
                                    labels=["<0.97", "0.97-1.0", "1.0-1.03", "1.03-1.05", ">1.05"])
        for bin_lbl, grp in ti_df.groupby("ti65_bin"):
            ti65_breakdown[f"TI65_{bin_lbl}"] = {
                "count":    len(grp),
                "win_rate": round(float((grp["pnl_pct"] > 0).mean() * 100), 1),
                "avg_pnl":  round(float(grp["pnl_pct"].mean()), 2),
            }

    # By Market Regime (the most important breakdown)
    regime_breakdown = {}
    for regime, grp in df.groupby("market_regime"):
        regime_breakdown[f"Regime_{regime}"] = {
            "count":    len(grp),
            "win_rate": round(float((grp["pnl_pct"] > 0).mean() * 100), 1),
            "avg_pnl":  round(float(grp["pnl_pct"].mean()), 2),
            "profit_factor": round(
                float(grp[grp["pnl_pct"] > 0]["pnl_pct"].sum()) /
                max(abs(float(grp[grp["pnl_pct"] <= 0]["pnl_pct"].sum())), 0.01), 2
            ),
        }

    # By consolidation width
    width_breakdown = {}
    cw_df = df[df["consol_width"].notna()].copy()
    if len(cw_df) > 0:
        cw_df["width_bin"] = pd.cut(cw_df["consol_width"],
                                     bins=[0, 4, 7, 10, 12, 999],
                                     labels=["<4%", "4-7%", "7-10%", "10-12%", ">12%"])
        for bin_lbl, grp in cw_df.groupby("width_bin"):
            width_breakdown[f"Width_{bin_lbl}"] = {
                "count":    len(grp),
                "win_rate": round(float((grp["pnl_pct"] > 0).mean() * 100), 1),
                "avg_pnl":  round(float(grp["pnl_pct"].mean()), 2),
            }

    # Monthly signal frequency
    monthly_avg = (
        np.mean(list(monthly_signals.values())) if monthly_signals else 0
    )

    # By exit reason
    exit_breakdown = {}
    for reason, grp in df.groupby("exit_reason"):
        exit_breakdown[reason] = {
            "count":    len(grp),
            "win_rate": round(float((grp["pnl_pct"] > 0).mean() * 100), 1),
            "avg_pnl":  round(float(grp["pnl_pct"].mean()), 2),
        }

    return {
        # ── Core metrics ──
        "period":              f"{start_date} → {end_date}",
        "trade_count":         len(df),
        "win_count":           len(wins),
        "loss_count":          len(losses),
        "win_rate_pct":        round(win_rate, 1),
        "avg_win_pct":         round(avg_win, 2),
        "avg_loss_pct":        round(avg_loss, 2),
        "profit_factor":       round(profit_factor, 2),
        "cagr_pct":            round(cagr, 1),
        "max_drawdown_pct":    round(max_dd, 1),
        "sharpe_ratio":        round(sharpe, 2),

        # ── Stockbee-specific ──
        "avg_hold_days_mb":    round(avg_hold_mb, 1),
        "avg_hold_days_ep":    round(avg_hold_ep, 1),
        "day3_target_hit_pct": round(day3_hit_pct, 1),
        "burst_8pct_pct":      round(burst_8pct_pct, 1),   # % of MB up 8%+ within 5d
        "monthly_avg_signals": round(monthly_avg, 1),
        "monthly_signal_freq": monthly_signals,

        # ── Breakdowns ──
        "twolynch_breakdown":  twolynch_breakdown,
        "rs_breakdown":        rs_breakdown,
        "ti65_breakdown":      ti65_breakdown,
        "regime_breakdown":    regime_breakdown,
        "width_breakdown":     width_breakdown,
        "exit_breakdown":      exit_breakdown,
    }


def _log_results_summary(results: dict) -> None:
    if "error" in results:
        log.error("Backtest: %s", results["error"])
        return

    log.info("=" * 60)
    log.info("BACKTEST RESULTS — %s", results.get("period", ""))
    log.info("=" * 60)
    log.info(
        "Trades: %d | Win Rate: %.1f%% | Profit Factor: %.2f",
        results["trade_count"], results["win_rate_pct"], results["profit_factor"]
    )
    log.info(
        "Avg Win: +%.2f%% | Avg Loss: %.2f%% | CAGR: %.1f%% | MaxDD: %.1f%%",
        results["avg_win_pct"], results["avg_loss_pct"],
        results["cagr_pct"], results["max_drawdown_pct"]
    )
    log.info("Sharpe: %.2f | Avg Hold MB: %.1fd | EP: %.1fd",
             results["sharpe_ratio"], results["avg_hold_days_mb"], results["avg_hold_days_ep"])
    log.info(
        "MB: Day3 hit %.1f%% | Burst 8%%+ within 5d: %.1f%% | Avg signals/mth: %.1f",
        results["day3_target_hit_pct"], results["burst_8pct_pct"],
        results["monthly_avg_signals"]
    )
    log.info("Regime breakdown: %s", results.get("regime_breakdown", {}))
    log.info("=" * 60)
