"""
NSE Stockbee Scanner — Main Orchestrator
=========================================
Replaces the Darvas Box main.py.
Runs the full Pradeep Bonde (Stockbee) scan pipeline:

  [1] Universe fetch
  [2] Price data download / update
  [3] Load all data into memory cache
  [4] Market Monitor (RUNS FIRST — FFM rule)
  [5] Cross-sectional RS ranking (all symbols)
  [6] Momentum Burst scan (MB_BREAKOUT + MB_ANTICIPATION)
  [7] Episodic Pivot scan (EP_9M; EP_REAL if fundamentals)
  [8] Persist signals to database
  [9] Generate Excel report
  [10] Send Telegram notifications
  [11] (Optional) Run backtest

Usage:
  python main.py                   # standard daily scan
  python main.py --weekly          # weekend breadth scan
  python main.py --backtest        # run 3-year backtest
  python main.py --debug-symbol RELIANCE.NS
  python main.py --full-refresh    # re-download all price history
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import date, timedelta

import pandas as pd

from config import (
    NIFTY50_SYMBOL, NIFTY500_SYMBOL, WATCHLIST_EXPIRY_DAYS,
    MB_SCORE_THRESHOLDS,
)
from database import init_db, signal_exists, upsert_signal, upsert_watchlist
from downloader import load_daily, run_download
from logger_utils import get_logger
from market_monitor import (
    compute_market_monitor, generate_market_monitor_report,
    weekly_breadth_scan,
)
from report import generate_report, format_telegram_signals, format_weekly_report
from stockbee_scanner import compute_rs_ranks, scan_all_symbols
from telegram_notify import send_report
from universe import fetch_nse_symbols

log = get_logger("scanner")


def main(args: argparse.Namespace) -> int:
    t0 = time.time()
    log.info("=" * 60)
    log.info("NSE STOCKBEE SCANNER  |  %s", date.today().isoformat())
    log.info("Based on Pradeep Bonde (Stockbee) methodology")
    log.info("=" * 60)

    # ── 1. Init DB ─────────────────────────────────────────────────────────────
    init_db()

    # ── 2. Universe ────────────────────────────────────────────────────────────
    log.info("[1/10] Fetching NSE symbol universe …")
    try:
        symbols = fetch_nse_symbols(force_refresh=args.refresh_universe)
        log.info("Universe: %d symbols", len(symbols))
    except Exception as e:
        log.error("Universe fetch failed: %s", e)
        return 1

    # ── 3. Download / update data ──────────────────────────────────────────────
    log.info("[2/10] Updating price data (full_refresh=%s) …", args.full_refresh)
    try:
        run_download(symbols, full_refresh=args.full_refresh)
    except Exception as e:
        log.error("Download error: %s\n%s", e, traceback.format_exc())

    # ── 4. Load all data into memory cache ────────────────────────────────────
    log.info("[3/10] Loading all price data into memory cache …")
    cached_daily: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and len(df) >= 200:
            cached_daily[sym] = df
    log.info("Loaded %d / %d symbols into cache", len(cached_daily), len(symbols))

    # ── DEBUG mode: single symbol deep diagnostic ─────────────────────────────
    if args.debug_symbol:
        _debug_symbol(args.debug_symbol, cached_daily, symbols)
        return 0

    # ── WEEKLY mode ───────────────────────────────────────────────────────────
    if args.weekly:
        _run_weekly_scan(cached_daily)
        return 0

    # ── BACKTEST mode ─────────────────────────────────────────────────────────
    if args.backtest:
        _run_backtest(symbols)
        return 0

    # ── 5. Market Monitor — RUNS FIRST (FFM rule) ────────────────────────────
    log.info("[4/10] Computing Market Monitor (situational awareness) …")
    try:
        market_snapshot = compute_market_monitor(cached_daily)
        log.info(
            "Market regime: %s (score=%.0f) | 200EMA=%.1f%% | up4=%d | A/D=%.2f",
            market_snapshot.market_regime,
            market_snapshot.regime_score,
            market_snapshot.pct_above_200ema,
            market_snapshot.up_4pct_count,
            market_snapshot.advance_decline_ratio,
        )

        if not market_snapshot.trading_allowed:
            log.warning(
                "FFM RULE: Market is %s — no long signals generated today. "
                "Study charts, do not trade.",
                market_snapshot.market_regime
            )
            # Still send Market Monitor telegram notification
            _send_market_monitor(market_snapshot)
            return 0

    except Exception as e:
        log.error("Market Monitor failed: %s\n%s", e, traceback.format_exc())
        return 1

    # ── 6. Cross-sectional RS ranking ─────────────────────────────────────────
    log.info("[5/10] Computing cross-sectional RS ranks (IBD-style) …")
    try:
        rs_ranks = compute_rs_ranks(list(cached_daily.keys()), cached_daily)
        log.info("RS ranking complete: %d symbols ranked", len(rs_ranks))
    except Exception as e:
        log.error("RS ranking failed: %s", e)
        rs_ranks = {}

    # ── 7. Scan all symbols ───────────────────────────────────────────────────
    log.info("[6/10] Running Stockbee scan (MB + EP) …")
    try:
        signals_today = scan_all_symbols(
            symbols           = list(cached_daily.keys()),
            daily_cache       = cached_daily,
            market_snapshot   = market_snapshot,
            rs_ranks          = rs_ranks,
            fundamentals_cache = {},     # extend here: plug in fundamental data source
            ep_registry       = {},      # extend here: load from DB (prior EP dates)
        )
        log.info("Scan complete: %d signals found", len(signals_today))
    except Exception as e:
        log.error("Scan failed: %s\n%s", e, traceback.format_exc())
        signals_today = []

    # ── 8. Persist signals ─────────────────────────────────────────────────────
    log.info("[7/10] Persisting %d signals to database …", len(signals_today))
    expiry = date.today() + timedelta(days=WATCHLIST_EXPIRY_DAYS)
    new_count = 0
    for sig in signals_today:
        if not signal_exists(sig.signal_id):
            upsert_signal(sig)
            upsert_watchlist(sig, expiry_date=expiry)
            new_count += 1
    log.info("Persisted %d new signals (%d already existed)", new_count, len(signals_today) - new_count)

    # ── 9. Generate Excel report ───────────────────────────────────────────────
    log.info("[8/10] Generating Excel report …")
    try:
        report_path = generate_report(signals_today, market_snapshot)
        log.info("Report: %s", report_path)
    except Exception as e:
        log.error("Report generation failed: %s\n%s", e, traceback.format_exc())
        report_path = None

    # ── 10. Send Telegram ──────────────────────────────────────────────────────
    log.info("[9/10] Sending Telegram notifications …")
    try:
        cards = format_telegram_signals(signals_today, market_snapshot, max_cards=10)
        for card in cards:
            send_report(card)
            time.sleep(0.5)
    except Exception as e:
        log.error("Telegram send failed: %s", e)

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("[10/10] Done in %.1fs", elapsed)
    log.info(
        "SUMMARY: %d total signals | %d MB_BREAKOUT | %d MB_ANTICIPATION | %d EP",
        len(signals_today),
        sum(1 for s in signals_today if s.signal_type == "MB_BREAKOUT"),
        sum(1 for s in signals_today if s.signal_type == "MB_ANTICIPATION"),
        sum(1 for s in signals_today if s.signal_type.startswith("EP")),
    )
    log.info("Market: %s | Report: %s", market_snapshot.market_regime, report_path)

    return 0


# ─── Debug single symbol ──────────────────────────────────────────────────────

def _debug_symbol(
    symbol: str,
    cached_daily: dict,
    all_symbols: list,
) -> None:
    """Deep diagnostic for a single symbol. Shows EXACTLY why it passed/failed."""
    from momentum_burst import (
        compute_ti65, detect_consolidation, detect_prior_uptrend, check_twolynch
    )
    from episodic_pivot import compute_volume_spike, score_neglect, detect_9m_ep

    sym = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    log.info("=" * 60)
    log.info("DEBUG MODE: %s", sym)
    log.info("=" * 60)

    daily = cached_daily.get(sym)
    if daily is None:
        log.error("Symbol not in cache. Check spelling or run with --full-refresh.")
        return

    close  = daily["Close"]
    volume = daily["Volume"]
    n      = len(daily)
    log.info("Data: %d bars, latest close = %.2f on %s", n, float(close.iloc[-1]),
             daily.index[-1].date())

    # RS rank
    rs_ranks = compute_rs_ranks([sym], {sym: daily})
    rs = rs_ranks.get(sym, 50.0)
    log.info("RS Rank: %.1f %s", rs, "✅" if rs >= 70 else "❌ (need ≥ 70)")

    # TI65
    ti = compute_ti65(close)
    log.info("TI65: %.4f %s", ti, "✅" if ti >= 1.03 else "❌ (need ≥ 1.03)")

    # 4% check
    if n >= 2:
        pct_chg = (float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2]) * 100
        log.info("Today's move: %.2f%% %s", pct_chg, "✅" if pct_chg >= 4.0 else "❌ (need ≥ 4.0%)")

    # Volume
    if n >= 2:
        vol_r = float(volume.iloc[-1]) / float(volume.iloc[-2]) if float(volume.iloc[-2]) > 0 else 0
        log.info("Volume ratio (today/yesterday): %.2fx %s", vol_r, "✅" if vol_r >= 1.0 else "❌")

    # Consolidation
    consol = detect_consolidation(daily, n - 1)
    if consol:
        cs, ce, cm = consol
        log.info("Consolidation: %d bars | width=%.1f%% | vol_ratio=%.2f | quality=%.2f ✅",
                 cm["length"], cm["width_pct"], cm["vol_ratio"], cm["quality"])
        # Prior uptrend
        up = detect_prior_uptrend(daily, cs)
        if up:
            _, move_pct, lin = up
            log.info("Prior uptrend: %.1f%% | linearity=%.2f ✅", move_pct, lin)
        else:
            log.info("Prior uptrend: NOT FOUND ❌ (need ≥ %.1f%%)", 8.0)
    else:
        log.info("Consolidation: NOT FOUND ❌")

    # 9M EP check
    ep = detect_9m_ep(sym, daily, rs)
    if ep:
        log.info("9M EP: TRIGGERED ✅ | spike=%.1fx | quiet=%d days",
                 ep.volume_spike_ratio, ep.prior_quiet_days)
    else:
        vol_50d = float(volume.rolling(50).mean().iloc[-1]) if n >= 50 else float(volume.mean())
        spike   = float(volume.iloc[-1]) / vol_50d if vol_50d > 0 else 0
        log.info("9M EP: not triggered | spike=%.1fx (need ≥ 5x)", spike)

    log.info("=" * 60)


# ─── Weekly scan ──────────────────────────────────────────────────────────────

def _run_weekly_scan(cached_daily: dict) -> None:
    log.info("Running WEEKLY BREADTH SCAN …")
    weekly = weekly_breadth_scan(cached_daily)
    report = format_weekly_report(weekly)
    log.info("\n%s", report)
    try:
        send_report(report)
    except Exception:
        pass


# ─── Backtest ─────────────────────────────────────────────────────────────────

def _run_backtest(symbols: list) -> None:
    from backtest import run_backtest, export_backtest_excel
    from config import REPORTS_DIR

    today = date.today()
    start = today - timedelta(days=365 * 3)

    log.info("Running 3-year backtest: %s to %s …", start, today)

    # With market filter (live behaviour: BEAR blocked, CAUTION = EP only)
    results_with = run_backtest(
        symbols         = symbols,
        start_date      = start,
        end_date        = today,
        apply_mm_filter = True,
        signal_type     = "MB_BREAKOUT",
    )

    # Without market filter (shows value of FFM rule — FIX: now actually differs)
    results_without = run_backtest(
        symbols         = symbols,
        start_date      = start,
        end_date        = today,
        apply_mm_filter = False,
        signal_type     = "MB_BREAKOUT",
    )

    log.info("\n=== WITH Market Monitor Filter ===")
    log.info(
        "Trades: %d | Win%%: %.1f | PF: %.2f | CAGR: %.1f%% | E[R]: %.3f",
        results_with.get("trade_count", 0),
        results_with.get("win_rate_pct", 0),
        results_with.get("profit_factor", 0),
        results_with.get("cagr_pct", 0),
        results_with.get("expectancy_r", 0),
    )

    log.info("\n=== WITHOUT Market Monitor Filter ===")
    log.info(
        "Trades: %d | Win%%: %.1f | PF: %.2f | CAGR: %.1f%% | E[R]: %.3f",
        results_without.get("trade_count", 0),
        results_without.get("win_rate_pct", 0),
        results_without.get("profit_factor", 0),
        results_without.get("cagr_pct", 0),
        results_without.get("expectancy_r", 0),
    )

    log.info("\n2LYNCH breakdown (with MM filter):")
    for k, v in results_with.get("twolynch_breakdown", {}).items():
        log.info("  %s: %d trades | Win%%: %.1f | Avg PnL: %.2f%%",
                 k, v["count"], v["win_rate"], v["avg_pnl"])

    log.info("\nRegime breakdown (without MM filter):")
    for k, v in results_without.get("regime_breakdown", {}).items():
        log.info("  %s: %d trades | Win%%: %.1f | PF: %.2f | Avg PnL: %.2f%%",
                 k, v["count"], v["win_rate"], v.get("profit_factor", 0), v["avg_pnl"])

    # ── Excel export (NEW: full reference-quality multi-sheet report) ─────────
    if "error" not in results_with:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        excel_path = REPORTS_DIR / f"bonde_backtest_{today.isoformat()}.xlsx"
        try:
            export_backtest_excel(
                results_with    = results_with,
                results_without = results_without,
                output_path     = str(excel_path),
                signal_type     = "MB_BREAKOUT",
                years           = 3,
                notes           = "Auto-generated by main.py --backtest",
            )
            log.info("Backtest Excel report → %s", excel_path)
        except Exception as e:
            log.error("Excel export failed: %s", e)
    else:
        log.warning("Skipping Excel export — backtest produced no trades")


# ─── Telegram Market Monitor send ─────────────────────────────────────────────

def _send_market_monitor(snapshot) -> None:
    report = generate_market_monitor_report(snapshot)
    try:
        send_report(report)
    except Exception as e:
        log.error("Telegram send failed: %s", e)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE Stockbee Scanner (Pradeep Bonde method)")
    p.add_argument("--full-refresh",       action="store_true",
                   help="Re-download all price history from scratch")
    p.add_argument("--refresh-universe",   action="store_true",
                   help="Force refresh the NSE symbol list")
    p.add_argument("--weekly",             action="store_true",
                   help="Run weekend breadth scan (study mode)")
    p.add_argument("--backtest",           action="store_true",
                   help="Run 3-year historical backtest")
    p.add_argument("--debug-symbol",       type=str, default="",
                   metavar="SYMBOL",
                   help="Deep diagnostic for one symbol (e.g. RELIANCE.NS)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args))
