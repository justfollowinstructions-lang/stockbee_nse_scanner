"""
NSE Stockbee Scanner — Market Monitor (Situational Awareness Engine)
=====================================================================
Implements Pradeep Bonde's (Stockbee) Market Monitor.

PB: "Without market awareness, your scanner is a money-losing machine.
     Even my best setups fail at 70% rates in bad market conditions."

The Market Monitor answers ONE question every day:
    "Are breakouts likely to work TODAY?"

Metrics tracked (Nifty 500 universe):
  - % stocks above 200 EMA        → primary regime indicator
  - % stocks above 50 EMA         → intermediate trend health
  - % stocks above 20 EMA         → short-term momentum
  - Count stocks up 4%+ today     → momentum burst activity
  - Count stocks down 4%+ today   → distribution pressure
  - Advance / Decline ratio        → broad market participation
  - % at 52-week highs            → leadership breadth
  - % at 52-week lows             → distribution breadth
  - TI65 green count              → absolute momentum health
  - Weekly: stocks up 20%+ in 5d  → momentum market flag
  - Weekly: stocks dn 20%+ in 5d  → correction severity

Market Regime classification:
  BULL    → buy breakouts aggressively
  NEUTRAL → be selective, reduce size
  CAUTION → EPs only (catalyst-driven), no pure MB
  BEAR    → no long setups at all

FFM Rule: "Find Free Money. Trade only when you can find free money."

FIX (v2): trading_allowed was incorrectly set to (regime in BULL, NEUTRAL).
  This caused the summary_line to say "AVOID LONGS" in CAUTION even though
  EP setups are explicitly allowed in CAUTION per PB's rules.
  Corrected to: trading_allowed = (regime != "BEAR")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import (
    EMA_TREND, EMA_MID, EMA_SHORT,
    MB_BREAKOUT_PCT, TI65_STRONG_THRESHOLD,
    MM_52W_HIGH_STRONG, MM_52W_LOW_DANGER,
    MM_BEAR_AD_RATIO, MM_BEAR_DOWN4_COUNT, MM_BEAR_PCT_ABOVE_200,
    MM_BULL_AD_RATIO, MM_BULL_PCT_ABOVE_200, MM_BULL_UP4_COUNT,
    MM_WEEKLY_WINNERS_PCT,
    MIN_AVG_VOLUME,
)
from logger_utils import get_logger

log = get_logger("scanner")


# ─── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class MarketMonitorSnapshot:
    snapshot_date:           date

    # EMA breadth (% of Nifty 500 above each EMA)
    pct_above_200ema:        float   # primary regime indicator
    pct_above_50ema:         float
    pct_above_20ema:         float

    # Daily activity counts
    up_4pct_count:           int     # stocks up 4%+ today
    down_4pct_count:         int     # stocks down 4%+ today
    advance_count:           int     # stocks closing up from yesterday
    decline_count:           int     # stocks closing down from yesterday
    advance_decline_ratio:   float   # advancers / decliners

    # New highs / lows
    pct_52w_highs:           float   # % at 52-week high
    pct_52w_lows:            float   # % at 52-week low

    # Trend Intensity breadth
    ti65_green_count:        int     # stocks with TI65 > 1.05
    ti65_green_pct:          float   # as % of universe

    # Weekly momentum
    weekly_up20_count:       int     # stocks up 20%+ in last 5 bars
    weekly_down20_count:     int     # stocks down 20%+ in last 5 bars

    # Derived
    universe_size:           int     # total stocks counted
    market_regime:           str     # "BULL" | "NEUTRAL" | "CAUTION" | "BEAR"
    regime_score:            float   # 0-100 composite health score
    buy_breakouts_aggressively: bool # True only in BULL
    trading_allowed:         bool    # FIX: False ONLY in BEAR (CAUTION allows EPs)

    # Human-readable summary
    summary_line:            str = field(default="")

    def __post_init__(self):
        self.summary_line = self._build_summary()

    def _build_summary(self) -> str:
        regime_emoji = {
            "BULL": "🟢", "NEUTRAL": "🟡", "CAUTION": "🟠", "BEAR": "🔴"
        }.get(self.market_regime, "⚪")

        if self.buy_breakouts_aggressively:
            action = "✅ BUY AGGRESSIVELY"
        elif self.market_regime == "CAUTION":
            # FIX: CAUTION allows EP setups — do not say "AVOID LONGS"
            action = "🟠 EP ONLY — no MB setups"
        elif self.trading_allowed:
            action = "⚠️ BE SELECTIVE"
        else:
            action = "🚫 AVOID ALL LONGS"

        return (
            f"{regime_emoji} Market Monitor [{self.snapshot_date}] — {self.market_regime}\n"
            f"Above 200 EMA: {self.pct_above_200ema:.1f}% | "
            f"Above 50 EMA: {self.pct_above_50ema:.1f}% | "
            f"Above 20 EMA: {self.pct_above_20ema:.1f}%\n"
            f"Up 4%+: {self.up_4pct_count} | Down 4%+: {self.down_4pct_count} | "
            f"A/D: {self.advance_decline_ratio:.2f}\n"
            f"52W Highs: {self.pct_52w_highs:.1f}% | 52W Lows: {self.pct_52w_lows:.1f}%\n"
            f"TI65 Green: {self.ti65_green_count} ({self.ti65_green_pct:.1f}%) | "
            f"Wk Up20: {self.weekly_up20_count} | Wk Dn20: {self.weekly_down20_count}\n"
            f"Regime Score: {self.regime_score:.0f}/100 | {action}"
        )


# ─── TI65 helper (inline, avoids circular import) ─────────────────────────────

def _ti65(close: pd.Series) -> float:
    if len(close) < 65:
        return 1.0
    sma7  = float(close.iloc[-7:].mean())
    sma65 = float(close.iloc[-65:].mean())
    return sma7 / sma65 if sma65 > 0 else 1.0


# ─── Main computation ─────────────────────────────────────────────────────────

def compute_market_monitor(
    all_daily_data: Dict[str, pd.DataFrame],
) -> MarketMonitorSnapshot:
    """
    Compute the full Market Monitor snapshot from {symbol: daily_df} for Nifty 500.

    Call this FIRST before any individual stock scanning.
    The market regime determines whether to run MB/EP scanning at all.
    """
    universe_size = 0
    above_200     = 0
    above_50      = 0
    above_20      = 0
    up_4_count    = 0
    dn_4_count    = 0
    advance       = 0
    decline       = 0
    highs_52w     = 0
    lows_52w      = 0
    ti65_green    = 0
    wk_up20       = 0
    wk_dn20       = 0

    for symbol, df in all_daily_data.items():
        if df is None or len(df) < 60:
            continue

        close  = df["Close"]
        volume = df["Volume"]
        n      = len(df)
        idx    = n - 1

        # Minimum liquidity (relaxed to 50% for breadth counting)
        avg_vol = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else 0
        if avg_vol < MIN_AVG_VOLUME * 0.5:
            continue

        universe_size += 1
        today_close = float(close.iloc[idx])

        # ── EMA breadth ──────────────────────────────────────────────────────
        if len(close) >= EMA_TREND:
            ema200 = float(close.ewm(span=EMA_TREND, adjust=False).mean().iloc[idx])
            if today_close > ema200:
                above_200 += 1

        if len(close) >= EMA_MID:
            ema50 = float(close.ewm(span=EMA_MID, adjust=False).mean().iloc[idx])
            if today_close > ema50:
                above_50 += 1

        if len(close) >= EMA_SHORT:
            ema20 = float(close.ewm(span=EMA_SHORT, adjust=False).mean().iloc[idx])
            if today_close > ema20:
                above_20 += 1

        # ── Daily % change ────────────────────────────────────────────────────
        if idx >= 1:
            prev_close = float(close.iloc[idx - 1])
            if prev_close > 0:
                day_chg = (today_close - prev_close) / prev_close * 100
                if day_chg >= MB_BREAKOUT_PCT:
                    up_4_count += 1
                elif day_chg <= -MB_BREAKOUT_PCT:
                    dn_4_count += 1
                if day_chg > 0:
                    advance += 1
                elif day_chg < 0:
                    decline += 1

        # ── 52-week highs / lows ─────────────────────────────────────────────
        if len(close) >= 252:
            high_52w = float(close.iloc[-252:].max())
            low_52w  = float(close.iloc[-252:].min())
            if today_close >= high_52w * 0.995:   # within 0.5% of 52w high
                highs_52w += 1
            if today_close <= low_52w * 1.005:    # within 0.5% of 52w low
                lows_52w += 1

        # ── TI65 ─────────────────────────────────────────────────────────────
        ti = _ti65(close)
        if ti >= TI65_STRONG_THRESHOLD:
            ti65_green += 1

        # ── Weekly momentum (last 5 bars) ─────────────────────────────────────
        if idx >= 5:
            price_5d_ago = float(close.iloc[idx - 5])
            if price_5d_ago > 0:
                weekly_chg = (today_close - price_5d_ago) / price_5d_ago * 100
                if weekly_chg >= 20.0:
                    wk_up20 += 1
                elif weekly_chg <= -20.0:
                    wk_dn20 += 1

    if universe_size == 0:
        log.error("Market Monitor: no valid stocks in universe — check data")
        universe_size = 1  # avoid division by zero

    # ── Compute ratios ────────────────────────────────────────────────────────
    pct_200  = above_200 / universe_size * 100
    pct_50   = above_50  / universe_size * 100
    pct_20   = above_20  / universe_size * 100
    pct_52h  = highs_52w / universe_size * 100
    pct_52l  = lows_52w  / universe_size * 100
    ti65_pct = ti65_green / universe_size * 100

    ad_ratio = advance / decline if decline > 0 else float(advance)

    # ── Regime classification ─────────────────────────────────────────────────
    regime = _classify_regime(pct_200, up_4_count, dn_4_count, ad_ratio, pct_52l)

    buy_aggressively = (regime == "BULL")

    # FIX: CAUTION allows EP setups — trading_allowed must be True in CAUTION.
    # Previously was: trading_allowed = (regime in ("BULL", "NEUTRAL"))
    # which caused the summary to say "AVOID LONGS" in CAUTION, misleading the user.
    # market_allows_trading() always correctly gates MB vs EP per setup type.
    trading_allowed = (regime != "BEAR")

    # ── Composite health score (0-100) ────────────────────────────────────────
    regime_score = _compute_regime_score(
        pct_200, pct_50, pct_20, up_4_count, dn_4_count,
        ad_ratio, pct_52h, pct_52l, ti65_pct
    )

    snap = MarketMonitorSnapshot(
        snapshot_date            = date.today(),
        pct_above_200ema         = round(pct_200, 1),
        pct_above_50ema          = round(pct_50, 1),
        pct_above_20ema          = round(pct_20, 1),
        up_4pct_count            = up_4_count,
        down_4pct_count          = dn_4_count,
        advance_count            = advance,
        decline_count            = decline,
        advance_decline_ratio    = round(ad_ratio, 3),
        pct_52w_highs            = round(pct_52h, 2),
        pct_52w_lows             = round(pct_52l, 2),
        ti65_green_count         = ti65_green,
        ti65_green_pct           = round(ti65_pct, 1),
        weekly_up20_count        = wk_up20,
        weekly_down20_count      = wk_dn20,
        universe_size            = universe_size,
        market_regime            = regime,
        regime_score             = round(regime_score, 1),
        buy_breakouts_aggressively = buy_aggressively,
        trading_allowed          = trading_allowed,
    )

    log.info(
        "Market Monitor: %s | score=%.0f | 200EMA=%.1f%% | up4=%d | dn4=%d | A/D=%.2f",
        regime, regime_score, pct_200, up_4_count, dn_4_count, ad_ratio
    )

    return snap


# ─── Regime classification logic ──────────────────────────────────────────────

def _classify_regime(
    pct_above_200: float,
    up_4_count:    int,
    dn_4_count:    int,
    ad_ratio:      float,
    pct_52w_lows:  float,
) -> str:
    """
    PB Market Regime rules:
      BULL    = ≥ 60% above 200 EMA AND up_4 ≥ 15 AND A/D ≥ 1.5
      BEAR    = < 40% above 200 EMA OR A/D ≤ 0.5 OR 52w lows > 5%
      CAUTION = down4 > up4 OR 52w lows > 2%  (and not already BEAR)
      NEUTRAL = everything else
    """
    # BEAR: hard stop on all longs
    if (pct_above_200 < MM_BEAR_PCT_ABOVE_200 or
            ad_ratio <= MM_BEAR_AD_RATIO or
            pct_52w_lows > 5.0):
        return "BEAR"

    # BULL: buy aggressively
    if (pct_above_200 >= MM_BULL_PCT_ABOVE_200 and
            up_4_count >= MM_BULL_UP4_COUNT and
            ad_ratio >= MM_BULL_AD_RATIO):
        return "BULL"

    # CAUTION: EP only, reduce MB exposure
    if dn_4_count > up_4_count or pct_52w_lows > MM_52W_LOW_DANGER * 100:
        return "CAUTION"

    return "NEUTRAL"


def _compute_regime_score(
    pct_200:    float,
    pct_50:     float,
    pct_20:     float,
    up_4:       int,
    dn_4:       int,
    ad_ratio:   float,
    pct_52h:    float,
    pct_52l:    float,
    ti65_pct:   float,
) -> float:
    """
    0-100 composite market health score.
    Higher = healthier market = be more aggressive with breakouts.
    """
    score = 0.0

    # EMA breadth (40 pts total)
    score += 20 * min(pct_200 / 100, 1.0)      # 0-20
    score += 12 * min(pct_50  / 100, 1.0)      # 0-12
    score +=  8 * min(pct_20  / 100, 1.0)      # 0-8

    # Momentum (25 pts): up_4 count vs dn_4 count
    net_momentum = min(max(up_4 - dn_4, -50), 50) / 50  # -1 to +1
    score += 25 * ((net_momentum + 1) / 2)     # 0-25

    # A/D ratio (15 pts): ratio of 2 = full marks
    score += 15 * min(ad_ratio / 2.0, 1.0)

    # 52W highs / lows (10 pts)
    high_score  = min(pct_52h / (MM_52W_HIGH_STRONG * 100), 1.0)
    low_penalty = min(pct_52l / 5.0, 1.0)
    score += 10 * (high_score * 0.5 + (1 - low_penalty) * 0.5)

    # TI65 breadth (10 pts)
    score += 10 * min(ti65_pct / 50.0, 1.0)   # 50% green = full marks

    return round(float(score), 1)


# ─── Trading permission gate ───────────────────────────────────────────────────

def market_allows_trading(
    snapshot: MarketMonitorSnapshot,
    setup_type: str,
) -> bool:
    """
    PB's trading permission gate — the AUTHORITATIVE check per setup type.

    MB setups:  require BULL or NEUTRAL only
    EP setups:  can trade even in CAUTION (catalyst overrides market weakness)
    BEAR:       no long setups regardless of type
    """
    if snapshot.market_regime == "BEAR":
        return False

    if setup_type in ("MB_BREAKOUT", "MB_ANTICIPATION"):
        return snapshot.market_regime in ("BULL", "NEUTRAL")

    if setup_type in ("EP_9M", "EP_REAL", "EP_STORY", "EP_DELAYED", "EP_SUGAR_BABY"):
        return snapshot.market_regime in ("BULL", "NEUTRAL", "CAUTION")

    # Default: honour trading_allowed (which is False only in BEAR)
    return snapshot.trading_allowed


# ─── Telegram-ready report ─────────────────────────────────────────────────────

def generate_market_monitor_report(snapshot: MarketMonitorSnapshot) -> str:
    """
    Compact Market Monitor report formatted for Telegram.
    Uses emoji for quick visual scanning on mobile.
    """
    regime_emoji = {
        "BULL": "🟢 BULL", "NEUTRAL": "🟡 NEUTRAL",
        "CAUTION": "🟠 CAUTION", "BEAR": "🔴 BEAR"
    }.get(snapshot.market_regime, snapshot.market_regime)

    def _bar(pct: float, width: int = 10) -> str:
        filled = round(pct / 100 * width)
        return "█" * filled + "░" * (width - filled) + f" {pct:.0f}%"

    lines = [
        f"📊 *NSE Market Monitor — {snapshot.snapshot_date}*",
        f"Regime: *{regime_emoji}* (Score: {snapshot.regime_score:.0f}/100)",
        "",
        "📈 *EMA Breadth (Nifty 500)*",
        f"Above 200 EMA: {_bar(snapshot.pct_above_200ema)}",
        f"Above  50 EMA: {_bar(snapshot.pct_above_50ema)}",
        f"Above  20 EMA: {_bar(snapshot.pct_above_20ema)}",
        "",
        "⚡ *Daily Momentum*",
        f"Up 4%+ : {snapshot.up_4pct_count:3d} stocks",
        f"Dn 4%+ : {snapshot.down_4pct_count:3d} stocks",
        f"A/D    : {snapshot.advance_count}↑ / {snapshot.decline_count}↓ = {snapshot.advance_decline_ratio:.2f}",
        "",
        "🏔️ *52-Week Extremes*",
        f"At 52W Highs: {snapshot.pct_52w_highs:.1f}%  "
        f"{'✅' if snapshot.pct_52w_highs >= MM_52W_HIGH_STRONG * 100 else '⚠️'}",
        f"At 52W Lows : {snapshot.pct_52w_lows:.1f}%  "
        f"{'🔴' if snapshot.pct_52w_lows >= MM_52W_LOW_DANGER * 100 else '✅'}",
        "",
        "🌡️ *TI65 Momentum*",
        f"TI65 Green  : {snapshot.ti65_green_count} ({snapshot.ti65_green_pct:.1f}%)",
        "",
        "📅 *Weekly Leaders*",
        f"Up 20%+ (5d): {snapshot.weekly_up20_count}",
        f"Dn 20%+ (5d): {snapshot.weekly_down20_count}",
        "",
        "─" * 35,
    ]

    if snapshot.buy_breakouts_aggressively:
        lines.append("✅ *BUY BREAKOUTS AGGRESSIVELY* — FFM is on")
    elif snapshot.market_regime == "CAUTION":
        lines.append("🟠 *EP SETUPS ONLY* — catalyst-driven moves allowed")
        lines.append("   MB setups: avoid | EP (9M / Real / Delayed): OK")
    elif snapshot.trading_allowed:
        lines.append("⚠️ *BE SELECTIVE* — reduce size, higher quality only")
    else:
        lines.append("🚫 *AVOID ALL LONGS* — Find Free Money elsewhere")
        lines.append("   Study charts, do not trade")

    return "\n".join(lines)


# ─── Weekly breadth scan ──────────────────────────────────────────────────────

def weekly_breadth_scan(
    all_daily_data: Dict[str, pd.DataFrame],
    top_n: int = 20,
) -> dict:
    """
    Weekend deep-dive scan (PB's study routine, not for trading signals).

    Returns:
      weekly_winners  — top N stocks up 20%+ in 5 days
      weekly_losers   — top N stocks down 20%+ in 5 days
      monthly_leaders — stocks up 50%+ in 40 days (study for big moves)

    Run every Saturday morning to identify what's working and what's not.
    """
    records = []

    for symbol, df in all_daily_data.items():
        if df is None or len(df) < 45:
            continue

        close = df["Close"]
        n     = len(df)
        idx   = n - 1
        today = float(close.iloc[idx])

        vol_20 = float(df["Volume"].iloc[-20:].mean()) if len(df["Volume"]) >= 20 else 0
        if vol_20 < MIN_AVG_VOLUME * 0.3:
            continue

        wk_chg = None
        if idx >= 5:
            p5 = float(close.iloc[idx - 5])
            wk_chg = (today - p5) / p5 * 100 if p5 > 0 else None

        mo_chg = None
        if idx >= 40:
            p40 = float(close.iloc[idx - 40])
            mo_chg = (today - p40) / p40 * 100 if p40 > 0 else None

        records.append({
            "symbol": symbol,
            "close":  round(today, 2),
            "wk_chg": round(wk_chg, 2) if wk_chg is not None else None,
            "mo_chg": round(mo_chg, 2) if mo_chg is not None else None,
        })

    df_scan = pd.DataFrame(records)
    result  = {}

    wk = df_scan[df_scan["wk_chg"].notna()].copy()
    result["weekly_winners"] = (
        wk[wk["wk_chg"] >= 20].nlargest(top_n, "wk_chg").to_dict("records")
    )
    result["weekly_losers"] = (
        wk[wk["wk_chg"] <= -20].nsmallest(top_n, "wk_chg").to_dict("records")
    )

    mo = df_scan[df_scan["mo_chg"].notna()].copy()
    result["monthly_leaders"] = (
        mo[mo["mo_chg"] >= 50].nlargest(50, "mo_chg").to_dict("records")
    )

    log.info(
        "Weekly breadth: %d winners (+20%%) | %d losers (-20%%) | %d monthly leaders (+50%%)",
        len(result["weekly_winners"]),
        len(result["weekly_losers"]),
        len(result["monthly_leaders"]),
    )

    return result
