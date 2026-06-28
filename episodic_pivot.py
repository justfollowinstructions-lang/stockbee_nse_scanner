"""
NSE Stockbee Scanner — Episodic Pivot Detection Engine
=======================================================
Implements Pradeep Bonde's (Stockbee) Episodic Pivot (EP) patterns:

  EP_9M       — 9 Million EP: massive volume spike after months of neglect
  EP_REAL     — Real catalyst: MAGNA53 criteria (earnings, sales acceleration)
  EP_STORY    — Theme/narrative driven gap-up
  EP_DELAYED  — Delayed reaction to a prior catalyst
  EP_SUGAR_BABY — Serial EP: repeated 9M EP moves from same base

Core insight from PB:
  "The 9M volume day = someone knows something."
  "Real EPs: Sales growth ≥ 39% on CONSECUTIVE quarters. Sales can't be faked."
  "Turnaround EPs make BIGGER moves because the stock was so neglected."

NSE adaptations:
  - Circuits limit gap to 5-10% on first day for many stocks
  - Minimum volume spike = 5x 50-day avg (replaces "9 million")
  - Market cap threshold adjusted for INR (₹10,000 Cr ≈ $1.2B)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ACCOUNT_SIZE, EP_9M_PRIOR_QUIET_DAYS, EP_9M_VOLUME_SPIKE,
    EP_CLOSE_STRENGTH_MIN, EP_DELAYED_REACTION_DAYS,
    EP_EPS_GROWTH_MIN, EP_GAP_MIN_PCT, EP_GAP_STRONG_PCT,
    EP_NEGLECT_MONTHS, EP_SALES_GROWTH_MIN,
    EP_SCORE_THRESHOLDS, EP_SCORE_WEIGHTS,
    EP_STOP_PCT_NORMAL, EP_STOP_PCT_HIGH_CONV,
    EP_VOLUME_SPIKE_RATIO, MIN_AVG_VOLUME, MIN_HISTORY_DAYS,
    RISK_PER_TRADE_PCT, RS_MIN_FOR_EP,
)
from logger_utils import get_logger

log = get_logger("scanner")


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class EpisodicPivot:
    symbol:              str
    ep_type:             str      # EP_9M | EP_REAL | EP_STORY | EP_DELAYED | EP_SUGAR_BABY
    signal_date:         date
    gap_pct:             float    # % gap from prior close to open
    day_change_pct:      float    # % change close-to-close on signal day
    volume_spike_ratio:  float    # today vol / 50d avg vol
    prior_quiet_days:    int      # consecutive low-vol days before spike
    price_at_signal:     float
    stop_loss:           float
    stop_pct:            float    # % stop loss applied
    target_20pct:        float    # +20% target
    target_40pct:        float    # +40% target  
    target_60pct:        float    # +60% target (for high conviction)
    catalyst_score:      float    # 0-15 MAGNA score
    magna_flags:         dict     # which MAGNA criteria passed
    neglect_score:       float    # 0-1 how neglected was the stock
    close_strength:      float    # (close-low)/(high-low) on signal day
    rs_rank:             float
    ep_score:            float    # composite 0-100
    classification:      str      # "Elite" | "Strong" | "Watch"
    position_size:       int      = 0
    capital_required:    float    = 0.0
    risk_amount:         float    = 0.0
    is_high_conviction:  bool     = False
    hold_max_days:       int      = 30

    def __post_init__(self):
        self.classification = self._classify()

    def _classify(self) -> str:
        s = self.ep_score
        if s >= EP_SCORE_THRESHOLDS["elite"]:  return "Elite"
        if s >= EP_SCORE_THRESHOLDS["strong"]: return "Strong"
        if s >= EP_SCORE_THRESHOLDS["watch"]:  return "Watch"
        return "Weak"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compute_gap_pct(daily_df: pd.DataFrame, idx: int) -> Tuple[float, float]:
    """
    Returns (gap_pct, day_change_pct).
    gap_pct      = (open[idx] - close[idx-1]) / close[idx-1] * 100
    day_change_pct = (close[idx] - close[idx-1]) / close[idx-1] * 100
    """
    if idx < 1 or "Open" not in daily_df.columns:
        # Fallback: treat close-to-close as the gap
        if idx >= 1:
            c0 = float(daily_df["Close"].iloc[idx - 1])
            c1 = float(daily_df["Close"].iloc[idx])
            pct = (c1 - c0) / c0 * 100 if c0 > 0 else 0.0
            return pct, pct
        return 0.0, 0.0

    open_today  = float(daily_df["Open"].iloc[idx])
    close_prev  = float(daily_df["Close"].iloc[idx - 1])
    close_today = float(daily_df["Close"].iloc[idx])

    gap_pct     = (open_today - close_prev) / close_prev * 100 if close_prev > 0 else 0.0
    day_chg_pct = (close_today - close_prev) / close_prev * 100 if close_prev > 0 else 0.0
    return round(gap_pct, 2), round(day_chg_pct, 2)


def compute_volume_spike(
    daily_df: pd.DataFrame,
    idx: int,
) -> Tuple[float, int]:
    """
    Returns (spike_ratio, prior_quiet_days).
    spike_ratio      = volume[idx] / 50d_avg_volume
    prior_quiet_days = consecutive days before idx where vol < 2x avg
    """
    volume = daily_df["Volume"]
    if idx < 10:
        return 0.0, 0

    vol_50d = float(volume.iloc[max(0, idx - 50): idx].mean())
    if vol_50d == 0:
        return 0.0, 0

    today_vol    = float(volume.iloc[idx])
    spike_ratio  = today_vol / vol_50d

    # Count consecutive quiet days before the spike
    quiet_days = 0
    for j in range(idx - 1, max(0, idx - 90), -1):
        if float(volume.iloc[j]) < vol_50d * 2.0:
            quiet_days += 1
        else:
            break

    return round(spike_ratio, 2), quiet_days


def score_neglect(daily_df: pd.DataFrame, idx: int) -> float:
    """
    0.0-1.0 neglect score.
    Higher = more neglected (longer neglect period = bigger potential move).

    Factors:
      - Days since last significant price move (> 5% day)
      - Average volume in prior 30 days vs 12-month average
      - 3-month price trajectory (flat or downtrending = neglected)
    """
    if idx < 60:
        return 0.3

    close  = daily_df["Close"]
    volume = daily_df["Volume"]

    # Days since last big day (>5%)
    pct_changes = close.pct_change() * 100
    big_days = pct_changes.iloc[max(0, idx - 180): idx]
    big_day_mask = big_days.abs() >= 5.0
    if big_day_mask.any():
        last_big_day = len(big_days) - big_day_mask.values[::-1].argmax() - 1
        days_since_big = len(big_days) - last_big_day
    else:
        days_since_big = 180  # no big day in 6 months = very neglected

    neglect_by_time = min(days_since_big / 90, 1.0)  # 90 days = full neglect

    # Volume neglect
    vol_30d   = float(volume.iloc[max(0, idx - 30): idx].mean())
    vol_12m   = float(volume.iloc[max(0, idx - 252): idx].mean())
    vol_ratio = vol_30d / vol_12m if vol_12m > 0 else 1.0
    neglect_by_vol = max(0.0, 1.0 - vol_ratio)  # low recent vol = neglected

    # Price trajectory (3 months)
    if idx >= 60:
        price_3m_ago = float(close.iloc[idx - 60])
        price_now    = float(close.iloc[idx - 1])
        price_chg    = (price_now - price_3m_ago) / price_3m_ago if price_3m_ago > 0 else 0
        # Flat or down = more neglected
        neglect_by_price = max(0.0, min(1.0, 0.5 - price_chg))
    else:
        neglect_by_price = 0.3

    score = (neglect_by_time * 0.50 + neglect_by_vol * 0.30 + neglect_by_price * 0.20)
    return round(float(score), 3)


def compute_magna_score(fundamentals: dict) -> Tuple[float, dict]:
    """
    Score the MAGNA53 + Cap10 * 10IPO criteria.

    Inputs (all optional — scanner may not have all fundamental data):
      fundamentals = {
        'sales_growth_current': float,  # % current quarter YoY
        'sales_growth_prior':   float,  # % prior quarter YoY
        'eps_growth':           float,  # % EPS growth YoY
        'analyst_upgrades':     int,    # count of upgrades post-event
        'short_interest_days':  float,  # days to cover
        'market_cap_cr':        float,  # ₹ Crore
        'years_listed':         int,    # years since IPO
      }

    Returns (total_score, flags_dict). Max score = 15.
    """
    flags: dict[str, bool] = {}
    score = 0.0

    # MA — Massive acceleration in EPS
    eps_g = fundamentals.get("eps_growth", None)
    if eps_g is not None:
        flags["MA_eps_acceleration"] = eps_g >= EP_EPS_GROWTH_MIN
        if flags["MA_eps_acceleration"]:
            score += 2.0
    else:
        flags["MA_eps_acceleration"] = False

    # A — Acceleration in Sales Growth (MOST IMPORTANT for PB)
    sg_curr  = fundamentals.get("sales_growth_current", None)
    sg_prior = fundamentals.get("sales_growth_prior", None)
    if sg_curr is not None:
        both_qtrs = (sg_curr >= EP_SALES_GROWTH_MIN and
                     sg_prior is not None and sg_prior >= EP_SALES_GROWTH_MIN)
        one_qtr   = sg_curr >= EP_SALES_GROWTH_MIN
        flags["A_sales_acceleration"] = both_qtrs
        if both_qtrs:
            score += 3.0
        elif one_qtr:
            score += 1.0
    else:
        flags["A_sales_acceleration"] = False

    # 5 — Short interest ≥ 5 days
    si_days = fundamentals.get("short_interest_days", None)
    if si_days is not None:
        flags["5_short_interest"] = si_days >= 5.0
        if flags["5_short_interest"]:
            score += 1.0
    else:
        flags["5_short_interest"] = False

    # 3 — Analyst upgrades ≥ 3
    upgrades = fundamentals.get("analyst_upgrades", None)
    if upgrades is not None:
        flags["3_analyst_upgrades"] = upgrades >= 3
        if flags["3_analyst_upgrades"]:
            score += 1.0
    else:
        flags["3_analyst_upgrades"] = False

    # Cap10 — Market cap ≤ ₹10,000 Cr
    mc = fundamentals.get("market_cap_cr", None)
    if mc is not None:
        flags["Cap10_small_cap"] = mc <= 10_000
        if flags["Cap10_small_cap"]:
            score += 1.0
    else:
        flags["Cap10_small_cap"] = None

    # 10IPO — Listed ≤ 10 years
    yrs = fundamentals.get("years_listed", None)
    if yrs is not None:
        flags["10IPO_young_company"] = yrs <= 10
        if flags["10IPO_young_company"]:
            score += 1.0
    else:
        flags["10IPO_young_company"] = None

    return round(score, 1), flags


def _ep_composite_score(
    gap_pct: float,
    volume_spike: float,
    magna_score: float,
    neglect: float,
    rs_rank: float,
) -> float:
    """Composite EP score 0-100."""
    w = EP_SCORE_WEIGHTS
    score = 0.0

    # Gap size (20 pts): 5% = base, 20%+ = full
    score += w["gap_size"] * min(gap_pct / 20.0, 1.0)

    # Volume spike (25 pts): 3x = base, 10x+ = full
    score += w["volume_spike"] * min(volume_spike / 10.0, 1.0)

    # Catalyst quality (30 pts): MAGNA max = 15 pts, normalize to 0-1
    score += w["catalyst_quality"] * min(magna_score / 15.0, 1.0)

    # Neglect (15 pts)
    score += w["neglect_score"] * neglect

    # RS rank (10 pts)
    score += w["rs_rank"] * min(rs_rank / 99.0, 1.0)

    return round(score, 1)


def _size_ep_position(entry: float, stop: float) -> Tuple[int, float, float]:
    risk_amt = ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100
    risk_ps  = entry - stop
    if risk_ps <= 0:
        return 0, 0.0, 0.0
    size    = max(1, int(risk_amt / risk_ps))
    capital = round(size * entry, 2)
    return size, capital, round(risk_amt, 2)


# ─── 9M EP Detection (main daily EP scan) ─────────────────────────────────────

def detect_9m_ep(
    symbol: str,
    daily_df: pd.DataFrame,
    rs_rank: float,
) -> Optional[EpisodicPivot]:
    """
    9 Million EP — massive volume spike after months of neglect.
    NSE adaptation: volume spike ≥ 5x 50-day average.

    Criteria:
      - Volume spike ≥ EP_9M_VOLUME_SPIKE (5x 50d avg)
      - Prior EP_9M_PRIOR_QUIET_DAYS days had no day with vol > 2x avg
      - Price up ≥ EP_GAP_MIN_PCT (5%) on spike day
      - Close strength ≥ EP_CLOSE_STRENGTH_MIN (60%)
      - RS ≥ RS_MIN_FOR_EP
    """
    if daily_df is None or len(daily_df) < 60:
        return None

    close  = daily_df["Close"]
    high   = daily_df["High"]
    low    = daily_df["Low"]
    volume = daily_df["Volume"]
    n      = len(daily_df)
    idx    = n - 1

    avg_vol_20 = float(volume.iloc[-20:].mean())
    if avg_vol_20 < MIN_AVG_VOLUME:
        return None

    if rs_rank < RS_MIN_FOR_EP:
        return None

    # Volume spike check
    spike_ratio, quiet_days = compute_volume_spike(daily_df, idx)
    if spike_ratio < EP_9M_VOLUME_SPIKE:
        log.debug("%s: skip 9M EP — spike=%.1fx < %.1fx", symbol, spike_ratio, EP_9M_VOLUME_SPIKE)
        return None

    # Prior must have been quiet (no day > 2x avg in last 20 days before today)
    if quiet_days < EP_9M_PRIOR_QUIET_DAYS:
        log.debug("%s: skip 9M EP — only %d quiet days before spike (need %d)",
                  symbol, quiet_days, EP_9M_PRIOR_QUIET_DAYS)
        return None

    # Price up ≥ 5% on spike day
    if idx < 1:
        return None
    gap_pct, day_chg = _compute_gap_pct(daily_df, idx)
    if day_chg < EP_GAP_MIN_PCT:
        log.debug("%s: skip 9M EP — day change %.2f%% < %.1f%%", symbol, day_chg, EP_GAP_MIN_PCT)
        return None

    # Close strength
    today_high  = float(high.iloc[idx])
    today_low   = float(low.iloc[idx])
    today_close = float(close.iloc[idx])
    day_range   = today_high - today_low
    close_str   = (today_close - today_low) / day_range if day_range > 0 else 0.5
    if close_str < EP_CLOSE_STRENGTH_MIN:
        log.debug("%s: skip 9M EP — close strength %.2f < %.2f", symbol, close_str, EP_CLOSE_STRENGTH_MIN)
        return None

    # Neglect score
    neglect = score_neglect(daily_df, idx)

    # EP score
    ep_score = _ep_composite_score(
        gap_pct      = day_chg,
        volume_spike = spike_ratio,
        magna_score  = 0.0,     # no fundamental data for 9M EP
        neglect      = neglect,
        rs_rank      = rs_rank,
    )

    if ep_score < EP_SCORE_THRESHOLDS["watch"]:
        return None

    entry     = today_close
    stop_loss = round(entry * (1 - EP_STOP_PCT_NORMAL / 100), 2)

    pos_size, capital, risk_amt = _size_ep_position(entry, stop_loss)

    log.info(
        "9M EP SIGNAL %-20s score=%5.1f  spike=%.1fx  neglect=%.2f  rs=%.1f",
        symbol, ep_score, spike_ratio, neglect, rs_rank
    )

    return EpisodicPivot(
        symbol             = symbol,
        ep_type            = "EP_9M",
        signal_date        = date.today(),
        gap_pct            = gap_pct,
        day_change_pct     = day_chg,
        volume_spike_ratio = spike_ratio,
        prior_quiet_days   = quiet_days,
        price_at_signal    = round(entry, 2),
        stop_loss          = stop_loss,
        stop_pct           = EP_STOP_PCT_NORMAL,
        target_20pct       = round(entry * 1.20, 2),
        target_40pct       = round(entry * 1.40, 2),
        target_60pct       = round(entry * 1.60, 2),
        catalyst_score     = 0.0,
        magna_flags        = {},
        neglect_score      = neglect,
        close_strength     = round(close_str, 3),
        rs_rank            = round(rs_rank, 1),
        ep_score           = ep_score,
        classification     = "",
        position_size      = pos_size,
        capital_required   = capital,
        risk_amount        = risk_amt,
        is_high_conviction = (spike_ratio >= 8.0 and neglect >= 0.7),
    )


# ─── Real EP Detection (earnings/fundamental catalyst) ────────────────────────

def detect_real_ep(
    symbol: str,
    daily_df: pd.DataFrame,
    rs_rank: float,
    fundamentals: dict,
) -> Optional[EpisodicPivot]:
    """
    Real Catalyst EP using MAGNA53 + Cap10 * 10IPO framework.

    fundamentals dict keys (all optional):
      sales_growth_current: float  — % QoQ sales growth current quarter
      sales_growth_prior:   float  — % QoQ sales growth prior quarter
      eps_growth:           float  — % EPS growth YoY
      analyst_upgrades:     int    — post-event analyst upgrade count
      short_interest_days:  float  — days to cover short interest
      market_cap_cr:        float  — market cap in ₹ Crore
      years_listed:         int    — years since IPO listing

    MAGNA (MA + G + N + A) must all be present for it to qualify as a Real EP.
    """
    if daily_df is None or len(daily_df) < 60:
        return None

    close  = daily_df["Close"]
    high   = daily_df["High"]
    low    = daily_df["Low"]
    volume = daily_df["Volume"]
    n      = len(daily_df)
    idx    = n - 1

    avg_vol_20 = float(volume.iloc[-20:].mean())
    if avg_vol_20 < MIN_AVG_VOLUME:
        return None

    # G — Gap up check
    gap_pct, day_chg = _compute_gap_pct(daily_df, idx)
    if day_chg < EP_GAP_MIN_PCT:
        log.debug("%s: skip Real EP — gap %.2f%% < %.1f%%", symbol, day_chg, EP_GAP_MIN_PCT)
        return None

    # Volume spike check
    spike_ratio, quiet_days = compute_volume_spike(daily_df, idx)
    if spike_ratio < EP_VOLUME_SPIKE_RATIO:
        log.debug("%s: skip Real EP — vol spike %.1fx < %.1fx", symbol, spike_ratio, EP_VOLUME_SPIKE_RATIO)
        return None

    # N — Neglect score
    neglect = score_neglect(daily_df, idx)

    # Add gap to fundamentals for G check
    funda_with_gap = {**fundamentals, "_gap_pct": day_chg}

    # MAGNA score
    magna_score, magna_flags = compute_magna_score(fundamentals)

    # A criterion is the most critical — check sales growth
    sg_curr = fundamentals.get("sales_growth_current", None)
    if sg_curr is not None and sg_curr < EP_SALES_GROWTH_MIN:
        log.debug("%s: skip Real EP — sales growth %.1f%% < %.1f%%",
                  symbol, sg_curr, EP_SALES_GROWTH_MIN)
        return None

    # Close strength
    today_high  = float(high.iloc[idx])
    today_low   = float(low.iloc[idx])
    today_close = float(close.iloc[idx])
    day_range   = today_high - today_low
    close_str   = (today_close - today_low) / day_range if day_range > 0 else 0.5

    # High conviction if A (both quarters) + gap > 10%
    both_sales = magna_flags.get("A_sales_acceleration", False)
    high_conv  = both_sales and day_chg >= EP_GAP_STRONG_PCT

    # Stop loss
    stop_pct  = EP_STOP_PCT_HIGH_CONV if high_conv else EP_STOP_PCT_NORMAL
    stop_loss = round(today_close * (1 - stop_pct / 100), 2)

    ep_score = _ep_composite_score(
        gap_pct      = day_chg,
        volume_spike = spike_ratio,
        magna_score  = magna_score,
        neglect      = neglect,
        rs_rank      = rs_rank,
    )

    if ep_score < EP_SCORE_THRESHOLDS["watch"]:
        return None

    pos_size, capital, risk_amt = _size_ep_position(today_close, stop_loss)

    log.info(
        "REAL EP SIGNAL %-20s score=%5.1f  gap=%.1f%%  spike=%.1fx  magna=%.0f/15",
        symbol, ep_score, day_chg, spike_ratio, magna_score
    )

    return EpisodicPivot(
        symbol             = symbol,
        ep_type            = "EP_REAL",
        signal_date        = date.today(),
        gap_pct            = gap_pct,
        day_change_pct     = day_chg,
        volume_spike_ratio = spike_ratio,
        prior_quiet_days   = quiet_days,
        price_at_signal    = round(today_close, 2),
        stop_loss          = stop_loss,
        stop_pct           = stop_pct,
        target_20pct       = round(today_close * 1.20, 2),
        target_40pct       = round(today_close * 1.40, 2),
        target_60pct       = round(today_close * 1.60, 2),
        catalyst_score     = magna_score,
        magna_flags        = magna_flags,
        neglect_score      = neglect,
        close_strength     = round(close_str, 3),
        rs_rank            = round(rs_rank, 1),
        ep_score           = ep_score,
        classification     = "",
        position_size      = pos_size,
        capital_required   = capital,
        risk_amount        = risk_amt,
        is_high_conviction = high_conv,
    )


# ─── Delayed Reaction EP ──────────────────────────────────────────────────────

def detect_delayed_ep(
    symbol: str,
    daily_df: pd.DataFrame,
    rs_rank: float,
    original_ep_date: date,
) -> Optional[EpisodicPivot]:
    """
    Delayed Reaction EP: enter 1-3 days AFTER the initial catalyst.
    
    Used when:
    - The initial gap-up was not clean (faded badly on day 1)
    - You want to see confirmation before entering
    - Works especially well on short side (initial gap-down then drift lower)

    Looks for: stock that had an original EP event, pulled back slightly,
    and is now resuming with strength.
    """
    if daily_df is None or len(daily_df) < 10:
        return None

    close  = daily_df["Close"]
    volume = daily_df["Volume"]
    n      = len(daily_df)
    idx    = n - 1

    # Find EP day in data
    df_dates = daily_df.index
    if hasattr(df_dates, 'date'):
        date_series = pd.Series([d.date() if hasattr(d, 'date') else d for d in df_dates])
    else:
        date_series = pd.Series(df_dates)

    ep_mask = date_series == original_ep_date
    if not ep_mask.any():
        return None

    ep_loc = ep_mask.values.argmax()
    days_since_ep = idx - ep_loc

    # Must be within 3 days of original EP
    if not (1 <= days_since_ep <= EP_DELAYED_REACTION_DAYS):
        return None

    # Check that stock is still holding above EP day's close
    ep_close = float(close.iloc[ep_loc])
    today_close = float(close.iloc[idx])
    if today_close < ep_close * 0.95:   # dropped > 5% from EP = not valid
        return None

    # Volume check: confirm continued interest
    vol_50d = float(volume.iloc[max(0, idx - 50): idx].mean())
    today_vol = float(volume.iloc[idx])
    spike = today_vol / vol_50d if vol_50d > 0 else 1.0

    # Build gap from EP day
    ep_open = float(close.iloc[ep_loc - 1]) if ep_loc > 0 else ep_close
    day_chg = (today_close - ep_open) / ep_open * 100 if ep_open > 0 else 0.0

    ep_score = _ep_composite_score(
        gap_pct      = day_chg,
        volume_spike = spike,
        magna_score  = 0.0,
        neglect      = 0.5,
        rs_rank      = rs_rank,
    )

    if ep_score < EP_SCORE_THRESHOLDS["watch"]:
        return None

    stop_loss = round(ep_close * (1 - EP_STOP_PCT_NORMAL / 100), 2)
    pos_size, capital, risk_amt = _size_ep_position(today_close, stop_loss)

    log.info(
        "DELAYED EP SIGNAL %-20s  days_since_ep=%d  score=%.1f",
        symbol, days_since_ep, ep_score
    )

    return EpisodicPivot(
        symbol             = symbol,
        ep_type            = "EP_DELAYED",
        signal_date        = date.today(),
        gap_pct            = day_chg,
        day_change_pct     = day_chg,
        volume_spike_ratio = round(spike, 2),
        prior_quiet_days   = days_since_ep,
        price_at_signal    = round(today_close, 2),
        stop_loss          = stop_loss,
        stop_pct           = EP_STOP_PCT_NORMAL,
        target_20pct       = round(today_close * 1.20, 2),
        target_40pct       = round(today_close * 1.40, 2),
        target_60pct       = round(today_close * 1.60, 2),
        catalyst_score     = 0.0,
        magna_flags        = {},
        neglect_score      = 0.5,
        close_strength     = 0.6,
        rs_rank            = round(rs_rank, 1),
        ep_score           = ep_score,
        classification     = "",
        position_size      = pos_size,
        capital_required   = capital,
        risk_amount        = risk_amt,
    )


# ─── Sugar Baby Detection ─────────────────────────────────────────────────────

def detect_sugar_baby(
    symbol: str,
    daily_df: pd.DataFrame,
    rs_rank: float,
    prior_ep_date: date,
) -> Optional[EpisodicPivot]:
    """
    Sugar Baby EP: a serial EP.
    After an initial 9M EP or Real EP, the stock consolidates 1-3 weeks
    then makes ANOTHER 4%+ breakout on high volume.
    These can compound 40-100%+ over weeks.

    PB: "Once a Sugar Baby theme is identified, every breakout from
    consolidation is re-enterable."
    """
    if daily_df is None or len(daily_df) < 20:
        return None

    close  = daily_df["Close"]
    volume = daily_df["Volume"]
    high   = daily_df["High"]
    low    = daily_df["Low"]
    n      = len(daily_df)
    idx    = n - 1

    # Find prior EP date in data
    df_dates = daily_df.index
    date_list = [d.date() if hasattr(d, 'date') else d for d in df_dates]

    if prior_ep_date not in date_list:
        return None

    ep_loc = date_list.index(prior_ep_date)
    bars_since_ep = idx - ep_loc

    # Sugar Baby consolidation: 5-21 bars after original EP
    if not (5 <= bars_since_ep <= 21):
        return None

    # Today: 4%+ breakout (another burst from the base established post-EP)
    if idx < 1:
        return None
    today_close = float(close.iloc[idx])
    prev_close  = float(close.iloc[idx - 1])
    day_chg     = (today_close - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

    if day_chg < 4.0:
        return None

    # Volume spike check
    vol_50d   = float(volume.iloc[max(0, idx - 50): idx].mean())
    today_vol = float(volume.iloc[idx])
    spike     = today_vol / vol_50d if vol_50d > 0 else 1.0

    if spike < 1.5:   # needs at least some volume confirmation
        return None

    # Close strength
    today_high = float(high.iloc[idx])
    today_low  = float(low.iloc[idx])
    day_range  = today_high - today_low
    close_str  = (today_close - today_low) / day_range if day_range > 0 else 0.5
    if close_str < 0.60:
        return None

    ep_score = _ep_composite_score(
        gap_pct      = day_chg,
        volume_spike = spike,
        magna_score  = 5.0,   # inherits some catalyst quality from original EP
        neglect      = 0.4,
        rs_rank      = rs_rank,
    )

    if ep_score < EP_SCORE_THRESHOLDS["watch"]:
        return None

    # Stop below consolidation low (post-EP consolidation)
    post_ep_low = float(low.iloc[ep_loc: idx].min())
    stop_loss   = round(post_ep_low * 0.99, 2)
    pos_size, capital, risk_amt = _size_ep_position(today_close, stop_loss)

    log.info(
        "SUGAR BABY SIGNAL %-20s  bars_since_ep=%d  spike=%.1fx  score=%.1f",
        symbol, bars_since_ep, spike, ep_score
    )

    return EpisodicPivot(
        symbol             = symbol,
        ep_type            = "EP_SUGAR_BABY",
        signal_date        = date.today(),
        gap_pct            = day_chg,
        day_change_pct     = day_chg,
        volume_spike_ratio = round(spike, 2),
        prior_quiet_days   = bars_since_ep,
        price_at_signal    = round(today_close, 2),
        stop_loss          = stop_loss,
        stop_pct           = EP_STOP_PCT_NORMAL,
        target_20pct       = round(today_close * 1.20, 2),
        target_40pct       = round(today_close * 1.40, 2),
        target_60pct       = round(today_close * 1.60, 2),
        catalyst_score     = 5.0,
        magna_flags        = {"serial_ep": True},
        neglect_score      = 0.4,
        close_strength     = round(close_str, 3),
        rs_rank            = round(rs_rank, 1),
        ep_score           = ep_score,
        classification     = "",
        position_size      = pos_size,
        capital_required   = capital,
        risk_amount        = risk_amt,
    )
