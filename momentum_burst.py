"""
NSE Stockbee Scanner — Momentum Burst Detection Engine
=======================================================
Implements Pradeep Bonde's (Stockbee) Momentum Burst patterns:

  • MB_BREAKOUT   — 4%+ range expansion after clean consolidation
  • MB_ANTICIPATION — pre-breakout coiling setup (build watchlist)

Core Logic (from stockbee.blogspot.com):
  1. Stock must have a prior linear uptrend of ≥ 8%
  2. Followed by orderly consolidation: no -4% days, vol dry-up, tight range
  3. Breakout day: ≥ 4% move, volume > yesterday, close in top 30% of range
  4. 2LYNCH checklist qualifies the breakout further
  5. TI65 and IBD RS Rank as mandatory momentum filters

Holding rule: NEVER hold MB beyond 5 days.
Exit 50% on Day 3, 100% on Day 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ACCOUNT_SIZE, ATR_PERIOD, EMA_MID, EMA_SHORT, EMA_TREND,
    MB_2_MAX_UP_DAYS_BEFORE, MB_BREAKOUT_PCT, MB_BREAKOUT_PCT_LARGE,
    MB_BREAKOUT_VOL_MIN_RATIO,
    MB_CLOSE_STRENGTH_MIN, MB_CONSOL_MAX_BARS, MB_CONSOL_MAX_WIDTH_PCT,
    MB_CONSOL_MIN_BARS, MB_CONSOL_NEG_DAY_PCT, MB_CONSOL_VOL_RATIO_MAX,
    MB_HOLD_MAX_DAYS, MB_L_LINEARITY_MIN, MB_N_NARROW_ATR_RATIO,
    MB_N_NEGATIVE_MAX_DOWN, MB_PARTIAL_EXIT_DAY, MB_PRIOR_MOVE_BARS,
    MB_PRIOR_MOVE_MIN_PCT, MB_SCORE_THRESHOLDS, MB_SCORE_WEIGHTS,
    MB_STOP_PCT_LARGE, MB_STOP_PCT_SMALL, MB_VOLUME_RATIO_MIN,
    ANT_CONSOL_MAX, ANT_CONSOL_MIN, ANT_PRIOR_MOVE_MIN, ANT_VOL_DRY_RATIO,
    RISK_PER_TRADE_PCT, RS_MIN_FOR_MB,
    TI65_BEAR_THRESHOLD, TI65_BULL_THRESHOLD,
    MIN_AVG_VOLUME, MIN_HISTORY_DAYS,
)
from logger_utils import get_logger

log = get_logger("scanner")


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class MomentumBurstSetup:
    symbol:                 str
    setup_type:             str        # "MB_BREAKOUT" | "MB_ANTICIPATION"
    signal_date:            date
    entry_price:            float
    stop_loss:              float
    target_day3:            float      # Day 3 partial exit (estimated +8%)
    target_day5:            float      # Day 5 full exit (estimated +15%)
    breakout_pct:           float      # % move on signal day (0 for anticipation)
    volume_ratio:           float      # today_vol / yesterday_vol
    close_strength:         float      # (close - low) / (high - low)
    consolidation_bars:     int        # length of consolidation window
    consolidation_width_pct: float     # (consol_high - consol_low) / consol_low * 100
    prior_move_pct:         float      # prior uptrend magnitude before consolidation
    ti65:                   float      # Trend Intensity = SMA7 / SMA65
    rs_rank:                float      # IBD-style RS rank 1-99
    twolynch_score:         int        # 0-5: how many 2LYNCH criteria met
    twolynch_flags:         dict       # which criteria passed / failed
    is_young_trend:         bool       # Y criterion: 1st or 2nd breakout from base
    consolidation_quality:  float      # 0.0-1.0 quality score
    linearity_score:        float      # % of up-bars in prior trend
    composite_score:        float      # 0-100 overall score
    classification:         str        # "Elite" | "Strong" | "Watch" | "Weak"
    position_size:          int        # shares to buy for 1% risk
    capital_required:       float      # position_size * entry_price
    risk_amount:            float      # Rs amount at risk
    max_hold_days:          int = field(default=MB_HOLD_MAX_DAYS)
    partial_exit_day:       int = field(default=MB_PARTIAL_EXIT_DAY)

    def __post_init__(self):
        self.classification = self._classify()

    def _classify(self) -> str:
        s = self.composite_score
        if s >= MB_SCORE_THRESHOLDS["elite"]:  return "Elite"
        if s >= MB_SCORE_THRESHOLDS["strong"]: return "Strong"
        if s >= MB_SCORE_THRESHOLDS["watch"]:  return "Watch"
        return "Weak"


# ─── Trend Intensity ──────────────────────────────────────────────────────────

def compute_ti65(close: pd.Series) -> float:
    """
    TI65 = 7-day SMA / 65-day SMA.
    Pradeep Bonde's proprietary momentum indicator.
    > 1.05 = strong. > 1.03 = entry threshold. < 0.97 = bearish.
    """
    if len(close) < 65:
        return 1.0
    sma7  = close.iloc[-7:].mean()
    sma65 = close.iloc[-65:].mean()
    if sma65 == 0:
        return 1.0
    return round(float(sma7 / sma65), 4)


# ─── Consolidation Detection ──────────────────────────────────────────────────

def detect_consolidation(
    daily_df: pd.DataFrame,
    signal_bar_idx: int,
) -> Optional[Tuple[int, int, dict]]:
    """
    Scan backward from signal_bar_idx to find a valid consolidation window.

    A valid consolidation:
      - 5 to 20 bars long
      - No single day falls more than -4% (MB_CONSOL_NEG_DAY_PCT)
      - Average volume in window < 90% of 50-day average (MB_CONSOL_VOL_RATIO_MAX)
      - Box width (high-low of entire range) < 12% (MB_CONSOL_MAX_WIDTH_PCT)
      - No massive single-day volume spikes (> 1.5x 50d avg) inside

    Returns (consol_start_idx, signal_bar_idx-1, metrics_dict) or None.
    The signal bar itself is the breakout, so consolidation ends at signal_bar_idx-1.
    """
    n = len(daily_df)
    if signal_bar_idx < MB_CONSOL_MIN_BARS + 10:
        return None

    close  = daily_df["Close"]
    high   = daily_df["High"]
    low    = daily_df["Low"]
    volume = daily_df["Volume"]

    # 50-day average volume (computed before signal bar)
    vol_50d = volume.iloc[max(0, signal_bar_idx - 50): signal_bar_idx].mean()
    if vol_50d == 0:
        return None

    consol_end = signal_bar_idx - 1   # last bar of consolidation

    # Try progressively longer windows
    best_result = None
    for length in range(MB_CONSOL_MIN_BARS, MB_CONSOL_MAX_BARS + 1):
        consol_start = consol_end - length + 1
        if consol_start < 1:
            break

        window_close  = close.iloc[consol_start: consol_end + 1]
        window_high   = high.iloc[consol_start: consol_end + 1]
        window_low    = low.iloc[consol_start: consol_end + 1]
        window_vol    = volume.iloc[consol_start: consol_end + 1]

        # Rule 1: No day down more than -4%
        # FIX C: was `break` — wrong because it stops trying ALL longer lengths
        # even though a shorter sub-window might not contain the -4% bar.
        # Example: bars [1..8] has a -4% day at bar 3.
        #   length=5 tries bars [4..8] → clean → valid 5-bar consolidation found ✅
        #   length=6 tries bars [3..8] → -4% day at bar 3 → should skip THIS length only
        #   length=7 tries bars [2..8] → also fails
        # Old `break` would reject lengths 6,7,...20 after seeing the first failure,
        # but length 5 would already be stored as best_result by then only if we
        # scan short→long. Since we scan 5→20, the -4% at length 6 would break
        # before we ever find that length 5 is valid.
        # Fix: `continue` — skip this length, keep trying longer ones in case
        # the offending bar falls outside a different window boundary.
        pct_chg = window_close.pct_change() * 100
        if (pct_chg < MB_CONSOL_NEG_DAY_PCT).any():
            continue   # FIX C: was break

        # Rule 2: Volume dry-up
        avg_consol_vol = window_vol.mean()
        vol_ratio = avg_consol_vol / vol_50d if vol_50d > 0 else 1.0
        if vol_ratio > MB_CONSOL_VOL_RATIO_MAX:
            continue   # might still qualify at a shorter length

        # Rule 3: Box width check
        consol_high_val = window_high.max()
        consol_low_val  = window_low.min()
        width_pct = (consol_high_val - consol_low_val) / consol_low_val * 100
        if width_pct > MB_CONSOL_MAX_WIDTH_PCT:
            continue

        # Rule 4: No single-day massive volume spike inside (1.5x 50d avg)
        if (window_vol > vol_50d * 1.5).any():
            continue

        # Valid consolidation found — prefer longer (more compressed energy)
        metrics = {
            "consol_start":      consol_start,
            "consol_end":        consol_end,
            "length":            length,
            "width_pct":         round(width_pct, 2),
            "vol_ratio":         round(vol_ratio, 3),
            "consol_high":       round(float(consol_high_val), 2),
            "consol_low":        round(float(consol_low_val), 2),
            "neg4_days":         int((pct_chg < -4.0).sum()),
            "quality":           _consol_quality_score(width_pct, vol_ratio, length),
        }
        best_result = (consol_start, consol_end, metrics)

    return best_result


def _consol_quality_score(width_pct: float, vol_ratio: float, length: int) -> float:
    """
    0.0-1.0 score: tighter box + lower volume + moderate length = higher quality.
    """
    # Width: < 5% = 1.0, 12% = 0.0
    width_score = max(0.0, 1.0 - width_pct / MB_CONSOL_MAX_WIDTH_PCT)
    # Volume: < 50% = 1.0, 90% = 0.0
    vol_score = max(0.0, 1.0 - vol_ratio / MB_CONSOL_VOL_RATIO_MAX)
    # Length: 8-15 bars is ideal
    if 8 <= length <= 15:
        len_score = 1.0
    elif length < 8:
        len_score = length / 8
    else:
        len_score = max(0.0, 1.0 - (length - 15) / 5)
    return round((width_score * 0.5 + vol_score * 0.35 + len_score * 0.15), 3)


# ─── Prior Uptrend Detection ─────────────────────────────────────────────────

def detect_prior_uptrend(
    daily_df: pd.DataFrame,
    consol_start: int,
) -> Optional[Tuple[int, float, float]]:
    """
    Find the prior uptrend before the consolidation started.
    The stock must have moved up ≥ MB_PRIOR_MOVE_MIN_PCT (8%) over
    up to MB_PRIOR_MOVE_BARS (60) bars before consolidation start.

    Returns (uptrend_start_idx, uptrend_pct, linearity_score) or None.
    linearity_score = % of bars that closed higher than prior close.
    """
    if consol_start < 10:
        return None

    close = daily_df["Close"]
    scan_start = max(0, consol_start - MB_PRIOR_MOVE_BARS)
    price_at_consol = float(close.iloc[consol_start])

    # Walk backward to find the lowest point before consolidation
    window = close.iloc[scan_start: consol_start]
    if len(window) < 5:
        return None

    low_idx_rel = window.values.argmin()
    low_price   = float(window.iloc[low_idx_rel])

    if low_price == 0:
        return None

    uptrend_pct = (price_at_consol - low_price) / low_price * 100
    if uptrend_pct < MB_PRIOR_MOVE_MIN_PCT:
        return None

    # Linearity: % of bars closing above prior close in the uptrend window
    uptrend_start = scan_start + low_idx_rel
    trend_close = close.iloc[uptrend_start: consol_start]
    if len(trend_close) < 3:
        return None

    up_bars = (trend_close.diff().iloc[1:] > 0).sum()
    linearity = float(up_bars) / (len(trend_close) - 1) if len(trend_close) > 1 else 0.5

    return (uptrend_start, round(uptrend_pct, 2), round(linearity, 3))


# ─── 2LYNCH Checklist ─────────────────────────────────────────────────────────

def check_twolynch(
    daily_df: pd.DataFrame,
    signal_bar_idx: int,
    consol_start: int,
    consol_end: int,
    linearity_score: float,
    rs_rank: float,
) -> Tuple[int, dict]:
    """
    Evaluate all 5+1 criteria of Pradeep Bonde's 2LYNCH checklist.

    2 = Not up 2 days in a row before breakout
    L = Linearity of prior uptrend
    Y = Young trend (1st or 2nd breakout from this base)
    N = Narrow or negative day immediately before breakout
    C = Consolidation quality (no -4% days, vol dry-up, tight range)
    H = Close near the high on breakout day (≥ 70% close strength)

    Returns (score_0_to_5, flags_dict).
    Note: H is evaluated from the signal bar directly in detect_mb_signal.
    """
    flags: dict[str, bool] = {}
    close  = daily_df["Close"]
    high   = daily_df["High"]
    low    = daily_df["Low"]
    volume = daily_df["Volume"]

    n = len(daily_df)
    if signal_bar_idx < 5:
        return 0, {}

    # ── 2: Not up two consecutive days before breakout ────────────────────────
    # Check bars signal_bar_idx-2 and signal_bar_idx-1 (the two days before)
    pre1 = signal_bar_idx - 1
    pre2 = signal_bar_idx - 2
    if pre2 >= 0:
        day1_up = close.iloc[pre1] > close.iloc[pre2]
        day2_up = close.iloc[pre2] > close.iloc[pre2 - 1] if pre2 > 0 else False
        consecutive_up = day1_up and day2_up
        # Small up (<1%) on day before is acceptable per PB
        pre1_chg = (close.iloc[pre1] - close.iloc[pre2]) / close.iloc[pre2]
        flags["2_not_up_two_days"] = not consecutive_up or pre1_chg < 0.01
    else:
        flags["2_not_up_two_days"] = True

    # ── L: Linearity of prior uptrend ─────────────────────────────────────────
    flags["L_linear_prior_move"] = linearity_score >= MB_L_LINEARITY_MIN

    # ── Y: Young trend (virgin or first pullback) ─────────────────────────────
    # Simple proxy: count 4%+ up-days in prior 60 bars before consolidation
    # A "young" trend has few prior bursts from this base
    pre_consol = close.iloc[max(0, consol_start - 60): consol_start]
    if len(pre_consol) > 5:
        pct_moves = pre_consol.pct_change() * 100
        prior_bursts = int((pct_moves >= 4.0).sum())
    else:
        prior_bursts = 0
    flags["Y_young_trend"] = prior_bursts <= 2  # 0 = virgin, 1-2 = young

    # ── N: Narrow or negative day before breakout ─────────────────────────────
    pre_bar = signal_bar_idx - 1
    if pre_bar >= 1:
        # ATR10 before the signal
        atr_window = min(10, pre_bar)
        atr10 = (high.iloc[pre_bar - atr_window: pre_bar] -
                  low.iloc[pre_bar - atr_window: pre_bar]).mean()

        pre_range = float(high.iloc[pre_bar] - low.iloc[pre_bar])
        pre_chg   = (float(close.iloc[pre_bar]) - float(close.iloc[pre_bar - 1])) / \
                     float(close.iloc[pre_bar - 1])

        narrow = pre_range < atr10 * MB_N_NARROW_ATR_RATIO if atr10 > 0 else False
        slightly_neg = MB_N_NEGATIVE_MAX_DOWN <= pre_chg < 0.005

        flags["N_narrow_or_negative"] = narrow or slightly_neg
    else:
        flags["N_narrow_or_negative"] = False

    # ── C: Consolidation quality check ────────────────────────────────────────
    # Already validated in detect_consolidation; recalc neg-day count here
    window_close = close.iloc[consol_start: consol_end + 1]
    pct_chg = window_close.pct_change() * 100
    neg4_days = int((pct_chg < MB_CONSOL_NEG_DAY_PCT).sum())
    flags["C_clean_consolidation"] = (neg4_days == 0)

    # ── H: Close near high on breakout day ────────────────────────────────────
    sig_high  = float(high.iloc[signal_bar_idx])
    sig_low   = float(low.iloc[signal_bar_idx])
    sig_close = float(close.iloc[signal_bar_idx])
    day_range = sig_high - sig_low
    if day_range > 0:
        close_strength = (sig_close - sig_low) / day_range
        flags["H_close_near_high"] = close_strength >= MB_CLOSE_STRENGTH_MIN
    else:
        flags["H_close_near_high"] = True

    score = sum(flags.values())
    return int(score), flags


# ─── Composite Score ──────────────────────────────────────────────────────────

def compute_mb_score(
    rs_rank: float,
    ti65: float,
    twolynch_score: int,
    consol_quality: float,
    vol_ratio: float,
    linearity: float,
) -> float:
    """
    Composite 0-100 score for a Momentum Burst setup.
    Weights from MB_SCORE_WEIGHTS in config.py.
    """
    w = MB_SCORE_WEIGHTS
    score = 0.0

    # RS rank (25 pts)
    score += w["rs_rank"] * min(rs_rank / 99.0, 1.0)

    # TI65 (20 pts): TI65 ≥ 1.05 = full, 1.03-1.05 = 70%, < 1.03 = 0
    if ti65 >= 1.05:
        ti_score = 1.0
    elif ti65 >= TI65_BULL_THRESHOLD:
        ti_score = 0.7
    else:
        ti_score = 0.0
    score += w["ti65"] * ti_score

    # 2LYNCH (25 pts, each criterion = 5 pts)
    score += w["twolynch_score"] * (twolynch_score / 5.0)

    # Consolidation quality (15 pts)
    score += w["consolidation"] * consol_quality

    # Volume ratio on breakout (10 pts): ratio ≥ 2.0 = full
    score += w["volume_ratio"] * min(vol_ratio / 2.0, 1.0)

    # Linearity (5 pts)
    score += w["linearity"] * min(linearity / 0.80, 1.0)

    return round(score, 1)


# ─── Position Sizing ──────────────────────────────────────────────────────────

def _size_position(entry: float, stop: float) -> Tuple[int, float, float]:
    """
    Risk 1% of ACCOUNT_SIZE per trade.
    Returns (position_size, capital_required, risk_amount).
    """
    risk_amt = ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100
    risk_ps  = entry - stop
    if risk_ps <= 0:
        return 0, 0.0, 0.0
    size     = max(1, int(risk_amt / risk_ps))
    capital  = round(size * entry, 2)
    return size, capital, round(risk_amt, 2)


# ─── Main Detection Function ──────────────────────────────────────────────────

def detect_mb_signal(
    symbol: str,
    daily_df: pd.DataFrame,
    rs_rank: float,
) -> Optional[MomentumBurstSetup]:
    """
    Full Momentum Burst breakout detection pipeline.

    Pipeline:
      1. History & liquidity guard
      2. Compute TI65 — must be ≥ TI65_BULL_THRESHOLD
      3. RS Rank check — must be ≥ RS_MIN_FOR_MB
      4. 4% breakout check on today's bar
      5. Volume check (today > yesterday)
      6. Close strength (≥ 70% of range)
      7. Consolidation detection (5-20 bars prior)
      8. Prior uptrend detection (≥ 8% before consolidation)
      9. 2LYNCH checklist
      10. Price above 200 EMA
      11. Composite score — must be ≥ watch threshold
      12. Build and return MomentumBurstSetup

    Returns MomentumBurstSetup or None.
    """
    if daily_df is None or len(daily_df) < MIN_HISTORY_DAYS:
        log.debug("%s: skip MB — insufficient history (%d bars)",
                  symbol, len(daily_df) if daily_df is not None else 0)
        return None

    close  = daily_df["Close"]
    high   = daily_df["High"]
    low    = daily_df["Low"]
    volume = daily_df["Volume"]
    n      = len(daily_df)
    sig_idx = n - 1  # today's bar

    # ── 1. Liquidity ─────────────────────────────────────────────────────────
    avg_vol_20 = float(volume.iloc[-20:].mean())
    if avg_vol_20 < MIN_AVG_VOLUME:
        log.debug("%s: skip MB — illiquid (avg_vol=%.0f)", symbol, avg_vol_20)
        return None

    # ── 2. TI65 ──────────────────────────────────────────────────────────────
    ti65 = compute_ti65(close)
    if ti65 < TI65_BULL_THRESHOLD:
        log.debug("%s: skip MB — TI65=%.3f below threshold %.3f",
                  symbol, ti65, TI65_BULL_THRESHOLD)
        return None

    if ti65 < TI65_BEAR_THRESHOLD:
        log.debug("%s: skip MB — TI65=%.3f in downtrend", symbol, ti65)
        return None

    # ── 3. RS Rank ────────────────────────────────────────────────────────────
    if rs_rank < RS_MIN_FOR_MB:
        log.debug("%s: skip MB — RS rank %.1f < min %d", symbol, rs_rank, RS_MIN_FOR_MB)
        return None

    # ── 4. 4% Breakout on signal bar ─────────────────────────────────────────
    # FIX A: use max(close-to-close, high-to-prevclose).
    # PB's definition is range-based: did the stock MOVE 4%+ intraday?
    # A stock that opens +6% and closes +3.8% is still a valid burst.
    # close-to-close alone would reject it.
    if sig_idx < 1:
        return None

    today_close = float(close.iloc[sig_idx])
    today_high  = float(high.iloc[sig_idx])
    today_low   = float(low.iloc[sig_idx])
    prev_close  = float(close.iloc[sig_idx - 1])
    if prev_close == 0:
        return None

    close_to_close  = (today_close - prev_close) / prev_close * 100
    high_to_prev    = (today_high  - prev_close) / prev_close * 100
    breakout_pct    = max(close_to_close, high_to_prev)   # FIX A

    if breakout_pct < MB_BREAKOUT_PCT:
        log.debug("%s: skip MB — breakout %.2f%% (c2c=%.2f%% h2p=%.2f%%) < threshold %.1f%%",
                  symbol, breakout_pct, close_to_close, high_to_prev, MB_BREAKOUT_PCT)
        return None

    # ── 5. Volume: must be ≥ 1.5× 50-day average (FIX B) ────────────────────
    # Old check (today > yesterday) was too weak — a stock breaking out on
    # 2× yesterday but both below the 50-day average is NOT a real burst.
    # PB explicitly says volume must confirm vs the baseline average,
    # not just beat the prior day's depressed volume.
    today_vol = float(volume.iloc[sig_idx])
    vol_50d   = float(volume.iloc[max(0, sig_idx - 50): sig_idx].mean()) if sig_idx >= 10 else float(volume.mean())

    if vol_50d == 0:
        return None

    vol_ratio_vs_avg  = today_vol / vol_50d          # primary: vs 50d avg
    vol_ratio_vs_prev = today_vol / float(volume.iloc[sig_idx - 1]) if float(volume.iloc[sig_idx - 1]) > 0 else 0.0

    if vol_ratio_vs_avg < MB_BREAKOUT_VOL_MIN_RATIO:
        log.debug("%s: skip MB — volume %.1f× 50d avg (need ≥ %.1f×)",
                  symbol, vol_ratio_vs_avg, MB_BREAKOUT_VOL_MIN_RATIO)
        return None

    # vol_ratio stored in signal = vs 50d avg (more informative than vs yesterday)
    vol_ratio = vol_ratio_vs_avg

    # ── 6. Close strength: closing in top 30% of range ───────────────────────
    # today_high / today_low already computed in step 4
    day_range = today_high - today_low
    close_str = (today_close - today_low) / day_range if day_range > 0 else 0.5
    if close_str < MB_CLOSE_STRENGTH_MIN:
        log.debug("%s: skip MB — close strength %.2f < %.2f",
                  symbol, close_str, MB_CLOSE_STRENGTH_MIN)
        return None

    # ── 7. Consolidation detection ────────────────────────────────────────────
    consol_result = detect_consolidation(daily_df, sig_idx)
    if consol_result is None:
        log.debug("%s: skip MB — no valid consolidation found", symbol)
        return None

    consol_start, consol_end, consol_metrics = consol_result

    # ── 8. Prior uptrend ──────────────────────────────────────────────────────
    uptrend_result = detect_prior_uptrend(daily_df, consol_start)
    if uptrend_result is None:
        log.debug("%s: skip MB — no prior uptrend ≥ %.1f%%", symbol, MB_PRIOR_MOVE_MIN_PCT)
        return None

    _, prior_move_pct, linearity_score = uptrend_result

    # ── 9. 2LYNCH checklist ───────────────────────────────────────────────────
    twolynch_score, twolynch_flags = check_twolynch(
        daily_df, sig_idx, consol_start, consol_end, linearity_score, rs_rank
    )
    # Minimum 3/5 criteria to even consider
    if twolynch_score < 3:
        log.debug("%s: skip MB — 2LYNCH score %d/5 < 3", symbol, twolynch_score)
        return None

    # ── 10. Price above 200 EMA ───────────────────────────────────────────────
    if len(close) >= 200:
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        if today_close < ema200 * 0.97:   # allow 3% buffer for near cases
            log.debug("%s: skip MB — price %.2f below 200 EMA %.2f", symbol, today_close, ema200)
            return None

    # ── 11. Composite score ───────────────────────────────────────────────────
    score = compute_mb_score(
        rs_rank       = rs_rank,
        ti65          = ti65,
        twolynch_score = twolynch_score,
        consol_quality = consol_metrics["quality"],
        vol_ratio      = vol_ratio,
        linearity      = linearity_score,
    )

    if score < MB_SCORE_THRESHOLDS["watch"]:
        log.debug("%s: skip MB — composite score %.1f < watch threshold %d",
                  symbol, score, MB_SCORE_THRESHOLDS["watch"])
        return None

    # ── 12. Risk management ───────────────────────────────────────────────────
    # Stop below consolidation low with a small buffer
    stop_loss = round(consol_metrics["consol_low"] * 0.99, 2)
    entry     = today_close

    # Targets: Day 3 = +8%, Day 5 = +15% (Pradeep Bonde's MB expected range)
    target_d3 = round(entry * 1.08, 2)
    target_d5 = round(entry * 1.15, 2)

    pos_size, capital, risk_amt = _size_position(entry, stop_loss)

    is_young = twolynch_flags.get("Y_young_trend", False)

    log.info(
        "MB SIGNAL %-20s score=%5.1f  rs=%.1f  ti65=%.3f  2Lynch=%d/5  class=%s",
        symbol, score, rs_rank, ti65, twolynch_score,
        "Elite" if score >= MB_SCORE_THRESHOLDS["elite"] else
        "Strong" if score >= MB_SCORE_THRESHOLDS["strong"] else "Watch"
    )

    return MomentumBurstSetup(
        symbol                = symbol,
        setup_type            = "MB_BREAKOUT",
        signal_date           = date.today(),
        entry_price           = round(entry, 2),
        stop_loss             = stop_loss,
        target_day3           = target_d3,
        target_day5           = target_d5,
        breakout_pct          = round(breakout_pct, 2),
        volume_ratio          = round(vol_ratio, 2),
        close_strength        = round(close_str, 3),
        consolidation_bars    = consol_metrics["length"],
        consolidation_width_pct = consol_metrics["width_pct"],
        prior_move_pct        = prior_move_pct,
        ti65                  = ti65,
        rs_rank               = round(rs_rank, 1),
        twolynch_score        = twolynch_score,
        twolynch_flags        = twolynch_flags,
        is_young_trend        = is_young,
        consolidation_quality = consol_metrics["quality"],
        linearity_score       = linearity_score,
        composite_score       = score,
        classification        = "",   # set in __post_init__
        position_size         = pos_size,
        capital_required      = capital,
        risk_amount           = risk_amt,
    )


# ─── Anticipation Setup ───────────────────────────────────────────────────────

def detect_anticipation_signal(
    symbol: str,
    daily_df: pd.DataFrame,
    rs_rank: float,
) -> Optional[MomentumBurstSetup]:
    """
    Detect a pre-breakout ANTICIPATION setup.
    Stock is still in consolidation — not yet at 4% breakout.
    Used to build a watchlist of stocks likely to break out within 1-5 days.

    Criteria:
      - Prior move up ≥ ANT_PRIOR_MOVE_MIN (10%)
      - Now in consolidation: 3 to 15 bars
      - Volume drying up: consol avg vol < 70% of 50-day avg
      - No -4% days in consolidation
      - Stock making higher lows (not breaking down)
      - Price above 20 EMA and 50 EMA
      - TI65 ≥ TI65_BULL_THRESHOLD
      - RS Rank ≥ RS_MIN_FOR_MB
    """
    if daily_df is None or len(daily_df) < MIN_HISTORY_DAYS:
        return None

    close  = daily_df["Close"]
    high   = daily_df["High"]
    low    = daily_df["Low"]
    volume = daily_df["Volume"]
    n      = len(daily_df)
    sig_idx = n - 1

    # Liquidity
    avg_vol_20 = float(volume.iloc[-20:].mean())
    if avg_vol_20 < MIN_AVG_VOLUME:
        return None

    # TI65 check
    ti65 = compute_ti65(close)
    if ti65 < TI65_BULL_THRESHOLD:
        return None

    # RS check
    if rs_rank < RS_MIN_FOR_MB:
        return None

    # Today should NOT be a 4%+ day (that's a breakout, not anticipation)
    if sig_idx >= 1:
        today_chg = (float(close.iloc[sig_idx]) - float(close.iloc[sig_idx - 1])) / \
                     float(close.iloc[sig_idx - 1]) * 100
        if today_chg >= MB_BREAKOUT_PCT:
            return None   # This is an MB_BREAKOUT signal, not anticipation

    vol_50d = float(volume.rolling(50).mean().iloc[sig_idx]) if len(volume) >= 50 else float(volume.mean())

    # Check recent consolidation window
    for length in range(ANT_CONSOL_MIN, ANT_CONSOL_MAX + 1):
        consol_start = sig_idx - length + 1
        if consol_start < 1:
            break

        window_close = close.iloc[consol_start: sig_idx + 1]
        window_vol   = volume.iloc[consol_start: sig_idx + 1]
        window_low   = low.iloc[consol_start: sig_idx + 1]

        # No -4% days in consolidation
        pct_chg = window_close.pct_change() * 100
        if (pct_chg < MB_CONSOL_NEG_DAY_PCT).any():
            break

        # Volume dry-up
        avg_consol_vol = window_vol.mean()
        if vol_50d > 0 and avg_consol_vol / vol_50d > ANT_VOL_DRY_RATIO:
            continue

        # Width check
        consol_high_val = float(high.iloc[consol_start: sig_idx + 1].max())
        consol_low_val  = float(window_low.min())
        width_pct = (consol_high_val - consol_low_val) / consol_low_val * 100
        if width_pct > MB_CONSOL_MAX_WIDTH_PCT:
            continue

        # Prior uptrend check
        uptrend_result = detect_prior_uptrend(daily_df, consol_start)
        if uptrend_result is None:
            continue
        _, prior_move_pct, linearity_score = uptrend_result
        if prior_move_pct < ANT_PRIOR_MOVE_MIN:
            continue

        # Higher lows check: last few lows should not be making new lows
        if length >= 4:
            lows_in_consol = window_low.values
            if lows_in_consol[-1] < lows_in_consol[0] * 0.97:
                continue   # Breaking down

        # Price above 20 EMA and 50 EMA
        today_close = float(close.iloc[sig_idx])
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[sig_idx])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[sig_idx])
        if today_close < ema20 * 0.97 or today_close < ema50 * 0.97:
            continue

        # Build anticipation signal
        consol_quality = _consol_quality_score(width_pct, avg_consol_vol / vol_50d if vol_50d > 0 else 0.5, length)
        twolynch_score, twolynch_flags = check_twolynch(
            daily_df, sig_idx, consol_start, sig_idx, linearity_score, rs_rank
        )
        # Ignore H for anticipation (not yet broken out)
        twolynch_flags["H_close_near_high"] = None  # type: ignore

        score = compute_mb_score(
            rs_rank        = rs_rank,
            ti65           = ti65,
            twolynch_score = min(twolynch_score, 4),  # H not applicable
            consol_quality = consol_quality,
            vol_ratio      = avg_consol_vol / vol_50d if vol_50d > 0 else 0.5,
            linearity      = linearity_score,
        )

        if score < MB_SCORE_THRESHOLDS["watch"] - 5:
            continue

        # Entry: near top of consolidation range (anticipate the breakout)
        entry     = round(consol_high_val * 1.005, 2)   # just above consol high
        stop_loss = round(consol_low_val * 0.99, 2)
        target_d3 = round(entry * 1.08, 2)
        target_d5 = round(entry * 1.15, 2)

        vol_ratio_now = float(volume.iloc[sig_idx]) / vol_50d if vol_50d > 0 else 1.0
        close_str = (today_close - float(low.iloc[sig_idx])) / \
                    max(float(high.iloc[sig_idx]) - float(low.iloc[sig_idx]), 0.01)

        pos_size, capital, risk_amt = _size_position(entry, stop_loss)

        log.info(
            "ANT SIGNAL %-20s score=%5.1f  rs=%.1f  consol=%d bars  width=%.1f%%",
            symbol, score, rs_rank, length, width_pct
        )

        return MomentumBurstSetup(
            symbol                = symbol,
            setup_type            = "MB_ANTICIPATION",
            signal_date           = date.today(),
            entry_price           = entry,
            stop_loss             = stop_loss,
            target_day3           = target_d3,
            target_day5           = target_d5,
            breakout_pct          = 0.0,   # not yet broken out
            volume_ratio          = round(vol_ratio_now, 2),
            close_strength        = round(close_str, 3),
            consolidation_bars    = length,
            consolidation_width_pct = round(width_pct, 2),
            prior_move_pct        = prior_move_pct,
            ti65                  = ti65,
            rs_rank               = round(rs_rank, 1),
            twolynch_score        = twolynch_score,
            twolynch_flags        = twolynch_flags,
            is_young_trend        = twolynch_flags.get("Y_young_trend", False),
            consolidation_quality = consol_quality,
            linearity_score       = linearity_score,
            composite_score       = score,
            classification        = "",
            position_size         = pos_size,
            capital_required      = capital,
            risk_amount           = risk_amt,
        )

    return None
