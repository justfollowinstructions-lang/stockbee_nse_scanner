"""
NSE Stockbee Scanner — Unified Scanner Orchestrator
=====================================================
Replaces scanner.py. Orchestrates:
  1. Market Monitor (situational awareness — runs FIRST)
  2. Cross-sectional RS ranking (IBD-style, all symbols)
  3. Momentum Burst scan (MB_BREAKOUT + MB_ANTICIPATION)
  4. Episodic Pivot scan (EP_9M + EP_REAL if fundamentals available)
  5. Signal deduplication and sorting
  6. Position sizing

PB rule: "Market Monitor first. If market is BEAR, close the book."
"""

from __future__ import annotations

import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from config import (
    ACCOUNT_SIZE, MB_SCORE_THRESHOLDS, EP_SCORE_THRESHOLDS,
    RS_MIN_FOR_MB, RS_MIN_FOR_EP, RISK_PER_TRADE_PCT,
    MB_HOLD_MAX_DAYS, EP_HOLD_MAX_DAYS,
    RS_WEIGHTS, MIN_HISTORY_DAYS, MIN_AVG_VOLUME,
)
from episodic_pivot import (
    EpisodicPivot, detect_9m_ep, detect_real_ep,
    detect_delayed_ep, detect_sugar_baby,
)
from logger_utils import get_logger
from market_monitor import MarketMonitorSnapshot, market_allows_trading
from momentum_burst import MomentumBurstSetup, detect_mb_signal, detect_anticipation_signal

log = get_logger("scanner")


# ─── Unified Signal Dataclass ─────────────────────────────────────────────────

@dataclass
class StockbeeSignal:
    """
    Unified signal wrapping either a MomentumBurstSetup or EpisodicPivot.
    This is what gets persisted, reported, and tracked.
    """
    signal_id:       str
    symbol:          str
    signal_type:     str    # "MB_BREAKOUT" | "MB_ANTICIPATION" | "EP_9M" | "EP_REAL" | "EP_SUGAR_BABY" | "EP_DELAYED"
    signal_date:     date
    setup:           Union[MomentumBurstSetup, EpisodicPivot]
    market_snapshot: MarketMonitorSnapshot
    tradeable:       bool   # market_allows_trading gate
    entry_price:     float
    stop_loss:       float
    target_1:        float  # Day 3 or +20% depending on type
    target_2:        float  # Day 5 or +40% depending on type
    target_3:        float  # +60% for EPs
    max_hold_days:   int
    composite_score: float
    classification:  str    # "Elite" | "Strong" | "Watch"
    position_size:   int
    capital_required: float
    risk_amount:     float

    def to_dict(self) -> dict:
        """Flat dict for database / report rendering."""
        base = {
            "signal_id":        self.signal_id,
            "symbol":           self.symbol,
            "signal_type":      self.signal_type,
            "signal_date":      self.signal_date.isoformat(),
            "tradeable":        self.tradeable,
            "entry_price":      self.entry_price,
            "stop_loss":        self.stop_loss,
            "target_1":         self.target_1,
            "target_2":         self.target_2,
            "target_3":         self.target_3,
            "max_hold_days":    self.max_hold_days,
            "composite_score":  self.composite_score,
            "classification":   self.classification,
            "position_size":    self.position_size,
            "capital_required": self.capital_required,
            "risk_amount":      self.risk_amount,
            "market_regime":    self.market_snapshot.market_regime,
        }

        if isinstance(self.setup, MomentumBurstSetup):
            s = self.setup
            base.update({
                "breakout_pct":           s.breakout_pct,
                "volume_ratio":           s.volume_ratio,
                "close_strength":         s.close_strength,
                "consolidation_bars":     s.consolidation_bars,
                "consolidation_width_pct": s.consolidation_width_pct,
                "prior_move_pct":         s.prior_move_pct,
                "ti65":                   s.ti65,
                "rs_rank":                s.rs_rank,
                "twolynch_score":         s.twolynch_score,
                "twolynch_flags":         str(s.twolynch_flags),
                "is_young_trend":         s.is_young_trend,
                "consolidation_quality":  s.consolidation_quality,
                "linearity_score":        s.linearity_score,
            })
        elif isinstance(self.setup, EpisodicPivot):
            ep = self.setup
            base.update({
                "ep_type":            ep.ep_type,
                "gap_pct":            ep.gap_pct,
                "day_change_pct":     ep.day_change_pct,
                "volume_spike_ratio": ep.volume_spike_ratio,
                "prior_quiet_days":   ep.prior_quiet_days,
                "catalyst_score":     ep.catalyst_score,
                "neglect_score":      ep.neglect_score,
                "rs_rank":            ep.rs_rank,
                "is_high_conviction": ep.is_high_conviction,
            })

        return base


# ─── RS ranking (cross-sectional, IBD-style) ──────────────────────────────────

def compute_rs_ranks(
    symbols: List[str],
    daily_cache: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    """
    True IBD-style RS Rating: percentile rank of each stock's weighted
    return vs every other stock in the universe (NOT vs a benchmark).

    Formula:  RS = 0.40*(C/C65) + 0.20*(C/C130) + 0.20*(C/C195) + 0.20*(C/C260)
    Then rank all stocks 1-99 percentile.

    Returns {symbol: percentile_rank_1_to_99}.
    """
    raw_returns: Dict[str, float] = {}

    periods = {"3m": 65, "6m": 130, "9m": 195, "12m": 260}
    weights  = RS_WEIGHTS  # {"3m": 0.40, "6m": 0.20, "9m": 0.20, "12m": 0.20}

    for sym in symbols:
        df = daily_cache.get(sym)
        if df is None or len(df) < 261:
            continue
        close = df["Close"]
        total_score = 0.0
        total_w     = 0.0
        for key, n in periods.items():
            w = weights.get(key, 0)
            if len(close) > n:
                ret = (float(close.iloc[-1]) / float(close.iloc[-n]) - 1)
                total_score += ret * w
                total_w     += w
        if total_w > 0:
            raw_returns[sym] = total_score / total_w

    if not raw_returns:
        log.warning("RS ranking: no symbols with sufficient history")
        return {}

    # Percentile rank 1-99
    series = pd.Series(raw_returns)
    pct    = series.rank(pct=True, method="average")
    rs_map = (pct * 98 + 1).round(1).to_dict()

    log.info("RS ranking complete: %d symbols ranked", len(rs_map))
    return {str(k): float(v) for k, v in rs_map.items()}


# ─── Main scan function ───────────────────────────────────────────────────────

def scan_all_symbols(
    symbols:            List[str],
    daily_cache:        Dict[str, pd.DataFrame],
    market_snapshot:    MarketMonitorSnapshot,
    rs_ranks:           Dict[str, float],
    fundamentals_cache: Optional[Dict[str, dict]] = None,
    ep_registry:        Optional[Dict[str, date]]  = None,
) -> List[StockbeeSignal]:
    """
    Full Stockbee scan pipeline. Call after compute_rs_ranks() and
    compute_market_monitor().

    ep_registry: {symbol: original_ep_date} for detecting Sugar Baby + Delayed EP.
    fundamentals_cache: {symbol: {sales_growth_current, ...}} for Real EP.

    Returns signals sorted by composite_score descending.
    """
    # ── Market gate ───────────────────────────────────────────────────────────
    if not market_snapshot.trading_allowed:
        log.warning(
            "Market Monitor: %s — ZERO signals generated (FFM rule)",
            market_snapshot.market_regime
        )
        return []

    signals:      List[StockbeeSignal] = []
    mb_symbols:   set = set()   # prevent duplicate MB + Anticipation for same symbol
    skip_reasons: dict = defaultdict(int)
    errors:       int  = 0

    fundamentals_cache = fundamentals_cache or {}
    ep_registry        = ep_registry or {}

    total = len(symbols)

    for i, sym in enumerate(symbols, 1):
        if i % 50 == 0:
            log.info(
                "  [%d/%d] MB signals=%d  EP signals=%d  errors=%d",
                i, total,
                sum(1 for s in signals if "MB" in s.signal_type),
                sum(1 for s in signals if "EP" in s.signal_type),
                errors,
            )

        try:
            daily = daily_cache.get(sym)
            if daily is None:
                skip_reasons["no_data"] += 1
                continue

            if len(daily) < MIN_HISTORY_DAYS:
                skip_reasons["short_history"] += 1
                continue

            avg_vol = float(daily["Volume"].iloc[-20:].mean())
            if avg_vol < MIN_AVG_VOLUME:
                skip_reasons["illiquid"] += 1
                continue

            rs_rank = rs_ranks.get(sym, 50.0)

            # ── Momentum Burst ───────────────────────────────────────────────
            if market_allows_trading(market_snapshot, "MB_BREAKOUT"):
                mb = detect_mb_signal(sym, daily, rs_rank)
                if mb is not None and sym not in mb_symbols:
                    sig = _wrap_mb(mb, market_snapshot, rs_rank)
                    signals.append(sig)
                    mb_symbols.add(sym)

                # Only scan anticipation if NO MB breakout found today
                elif mb is None and sym not in mb_symbols:
                    ant = detect_anticipation_signal(sym, daily, rs_rank)
                    if ant is not None:
                        sig = _wrap_mb(ant, market_snapshot, rs_rank)
                        signals.append(sig)
                        mb_symbols.add(sym)

            # ── Episodic Pivot — 9M ───────────────────────────────────────────
            if market_allows_trading(market_snapshot, "EP_9M"):
                ep9m = detect_9m_ep(sym, daily, rs_rank)
                if ep9m is not None:
                    sig = _wrap_ep(ep9m, market_snapshot)
                    signals.append(sig)

            # ── Episodic Pivot — Real (if fundamentals available) ─────────────
            if market_allows_trading(market_snapshot, "EP_REAL"):
                funda = fundamentals_cache.get(sym)
                if funda:
                    ep_real = detect_real_ep(sym, daily, rs_rank, funda)
                    if ep_real is not None:
                        sig = _wrap_ep(ep_real, market_snapshot)
                        signals.append(sig)

            # ── Sugar Baby / Delayed EP (if prior EP registered) ─────────────
            if sym in ep_registry:
                prior_ep_date = ep_registry[sym]

                sb = detect_sugar_baby(sym, daily, rs_rank, prior_ep_date)
                if sb is not None:
                    signals.append(_wrap_ep(sb, market_snapshot))

                delayed = detect_delayed_ep(sym, daily, rs_rank, prior_ep_date)
                if delayed is not None:
                    signals.append(_wrap_ep(delayed, market_snapshot))

        except KeyboardInterrupt:
            log.info("Scan interrupted at symbol %d/%d", i, total)
            break
        except Exception as exc:
            errors += 1
            log.debug("Error scanning %s: %s", sym, exc)
            if errors <= 5:
                log.debug(traceback.format_exc())
            if errors > 300:
                log.warning("High error count (%d) — check data quality", errors)

    # ── Deduplication (same symbol, same type, same day) ─────────────────────
    seen:         set         = set()
    deduped:      List[StockbeeSignal] = []
    for sig in signals:
        key = (sig.symbol, sig.signal_type)
        if key not in seen:
            seen.add(key)
            deduped.append(sig)

    # ── Sort: tradeable first, then by composite score ────────────────────────
    deduped.sort(key=lambda s: (not s.tradeable, -s.composite_score))

    log.info(
        "Scan complete: %d signals (%d MB, %d EP) | %d errors | skips: %s",
        len(deduped),
        sum(1 for s in deduped if "MB" in s.signal_type),
        sum(1 for s in deduped if "EP" in s.signal_type),
        errors,
        dict(skip_reasons),
    )

    return deduped


# ─── Signal wrappers ──────────────────────────────────────────────────────────

def _wrap_mb(
    setup: MomentumBurstSetup,
    snapshot: MarketMonitorSnapshot,
    rs_rank: float,
) -> StockbeeSignal:
    tradeable = market_allows_trading(snapshot, setup.setup_type)
    sig_id    = f"{setup.symbol}_{setup.signal_date.isoformat()}_{setup.setup_type}"

    return StockbeeSignal(
        signal_id        = sig_id,
        symbol           = setup.symbol,
        signal_type      = setup.setup_type,
        signal_date      = setup.signal_date,
        setup            = setup,
        market_snapshot  = snapshot,
        tradeable        = tradeable,
        entry_price      = setup.entry_price,
        stop_loss        = setup.stop_loss,
        target_1         = setup.target_day3,
        target_2         = setup.target_day5,
        target_3         = round(setup.entry_price * 1.20, 2),
        max_hold_days    = MB_HOLD_MAX_DAYS,
        composite_score  = setup.composite_score,
        classification   = setup.classification,
        position_size    = setup.position_size,
        capital_required = setup.capital_required,
        risk_amount      = setup.risk_amount,
    )


def _wrap_ep(
    ep: EpisodicPivot,
    snapshot: MarketMonitorSnapshot,
) -> StockbeeSignal:
    tradeable = market_allows_trading(snapshot, ep.ep_type)
    sig_id    = f"{ep.symbol}_{ep.signal_date.isoformat()}_{ep.ep_type}"

    return StockbeeSignal(
        signal_id        = sig_id,
        symbol           = ep.symbol,
        signal_type      = ep.ep_type,
        signal_date      = ep.signal_date,
        setup            = ep,
        market_snapshot  = snapshot,
        tradeable        = tradeable,
        entry_price      = ep.price_at_signal,
        stop_loss        = ep.stop_loss,
        target_1         = ep.target_20pct,
        target_2         = ep.target_40pct,
        target_3         = ep.target_60pct,
        max_hold_days    = EP_HOLD_MAX_DAYS,
        composite_score  = ep.ep_score,
        classification   = ep.classification,
        position_size    = ep.position_size,
        capital_required = ep.capital_required,
        risk_amount      = ep.risk_amount,
    )
