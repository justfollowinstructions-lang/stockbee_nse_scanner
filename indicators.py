"""
NSE Darvas Box Scanner - Technical Indicators
Stateless functions operating on pandas Series/DataFrames.
All calculations are vectorised with pandas/numpy – no external TA libraries required.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
import pandas as pd


# ─── Moving Averages ──────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ─── RSI ──────────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── ATR ──────────────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ─── ADX ──────────────────────────────────────────────────────────────────────

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Returns DataFrame with columns: ADX, +DI, -DI."""
    high_diff  = high.diff()
    low_diff   = (-low).diff()

    plus_dm  = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=high.index)
    minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=high.index)

    _atr = atr(high, low, close, period)

    plus_di  = 100 * (plus_dm.ewm(com=period - 1, adjust=False).mean()  / _atr)
    minus_di = 100 * (minus_dm.ewm(com=period - 1, adjust=False).mean() / _atr)

    dx  = (((plus_di - minus_di).abs()) / (plus_di + minus_di).replace(0, np.nan)) * 100
    _adx = dx.ewm(com=period - 1, adjust=False).mean()

    return pd.DataFrame({"ADX": _adx, "+DI": plus_di, "-DI": minus_di})


# ─── Volume helpers ───────────────────────────────────────────────────────────

def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume / n-period average."""
    return volume / volume.rolling(period).mean()


def is_accumulation_day(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Up day on above-average volume."""
    up   = close.diff() > 0
    hvol = volume > volume.rolling(20).mean()
    return up & hvol


# ─── Trend Structure ──────────────────────────────────────────────────────────

def higher_highs_higher_lows(
    high: pd.Series, low: pd.Series, lookback: int = 60
) -> bool:
    """
    True if the broader trend shows HH-HL structure.
    Uses *lookback* bars which should extend BEFORE the consolidation box
    so this check captures the trend, not the sideways action inside the box.
    Requires EITHER highs OR lows to have a positive slope (not both),
    which is more realistic for stocks sitting at box bottoms.
    """
    lb = min(lookback, len(high))
    if lb < 20:
        return False
    h = high.iloc[-lb:]
    l = low.iloc[-lb:]
    x = np.arange(lb)
    h_slope = np.polyfit(x, h.values, 1)[0]
    l_slope = np.polyfit(x, l.values, 1)[0]
    # Normalise by price level so we compare percentages
    h_pct = h_slope / h.mean() if h.mean() > 0 else 0
    l_pct = l_slope / l.mean() if l.mean() > 0 else 0
    # Pass if BOTH slopes are >= -0.001% per bar (not strongly downtrending)
    # and at least one is positive
    return (h_pct >= -0.001 and l_pct >= -0.001) and (h_pct > 0 or l_pct > 0)


def trend_label(close: pd.Series, fast: int = 50, slow: int = 200) -> str:
    """Return 'bullish', 'bearish', or 'neutral'."""
    if len(close) < slow:
        return "neutral"
    f = ema(close, fast).iloc[-1]
    s = ema(close, slow).iloc[-1]
    last = close.iloc[-1]
    if last > f > s:
        return "bullish"
    if last < f < s:
        return "bearish"
    return "neutral"


# ─── Relative Strength Rating (O'Neil style) ─────────────────────────────────

def rs_rating(
    close: pd.Series,
    benchmark: pd.Series,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Single-stock-vs-benchmark RS estimate (1-99) using weighted EXCESS
    return (stock_return - benchmark_return) across 3/6/9/12-month
    periods, mapped through a sigmoid. This avoids the instability of
    a stock/benchmark RATIO, which explodes or inverts whenever the
    benchmark return is small or negative (the previous implementation's
    bug — it floored almost every real-world case to RS=1 or RS=99 with
    nothing in between).

    NOTE: This is a per-symbol approximation, useful when scanning one
    stock in isolation. For the William O'Neil-style RS Rating (a true
    1-99 PERCENTILE RANK across the entire universe), compute raw excess
    returns for all symbols first and pass them to
    `cross_sectional_rs_rating()` below — that is what real IBD/O'Neil
    RS Ratings are based on, and what the scanner pipeline should use
    when ranking 2000+ NSE stocks against each other.
    """
    if weights is None:
        weights = {"3m": 0.40, "6m": 0.20, "9m": 0.20, "12m": 0.20}

    periods = {"3m": 63, "6m": 126, "9m": 189, "12m": 252}
    excess_total = 0.0
    total_w      = 0.0

    for key, w in weights.items():
        n = periods[key]
        if len(close) > n and len(benchmark) > n:
            s_ret = (close.iloc[-1] / close.iloc[-n] - 1)
            b_ret = (benchmark.iloc[-1] / benchmark.iloc[-n] - 1)
            excess_total += (s_ret - b_ret) * w
            total_w += w

    if total_w == 0:
        return 50.0

    excess = excess_total / total_w

    # Sigmoid transform: excess return of 0 -> RS 50.
    # +/-20% excess return -> RS ~80/20; +/-50% -> RS ~95/5. No hard clip cliff.
    scale = 0.15
    raw = 100 / (1 + np.exp(-excess / scale))
    return float(np.clip(raw, 1, 99))


def raw_weighted_return(
    close: pd.Series,
    weights: dict[str, float] | None = None,
) -> Optional[float]:
    """
    Weighted return for ONE stock across 3/6/9/12-month periods,
    with NO benchmark comparison. Used as the input to
    `cross_sectional_rs_rating()` so every symbol in the universe can be
    ranked against every other symbol (the true O'Neil/IBD method).
    Returns None if there isn't enough history.
    """
    if weights is None:
        weights = {"3m": 0.40, "6m": 0.20, "9m": 0.20, "12m": 0.20}

    periods = {"3m": 63, "6m": 126, "9m": 189, "12m": 252}
    total_score = 0.0
    total_w     = 0.0

    for key, w in weights.items():
        n = periods[key]
        if len(close) > n:
            ret = (close.iloc[-1] / close.iloc[-n] - 1)
            total_score += ret * w
            total_w += w

    return total_score / total_w if total_w > 0 else None


def cross_sectional_rs_rating(weighted_returns: pd.Series) -> pd.Series:
    """
    True William O'Neil / IBD-style RS Rating: a 1-99 PERCENTILE RANK
    of each stock's weighted return relative to every other stock in
    the universe (NOT vs. a single benchmark index).

    *weighted_returns* — a Series indexed by symbol, values from
    `raw_weighted_return()` for every symbol in the scan universe.

    Returns a Series of the same index with values in [1, 99].
    """
    pct = weighted_returns.rank(pct=True, method="average")
    rs  = (pct * 98 + 1).round(0)   # map (0,1] -> [1,99]
    return rs.clip(1, 99)


# ─── SEPA (Minervini) Checks ──────────────────────────────────────────────────

def sepa_score(close: pd.Series) -> tuple[float, dict]:
    """
    Returns (score_0_to_10, details_dict).
    Checks the 8 core SEPA template criteria.
    """
    checks: dict[str, bool] = {}
    if len(close) < 252:
        return 0.0, checks

    last    = close.iloc[-1]
    ema_50  = ema(close, 50).iloc[-1]
    ema_150 = ema(close, 150).iloc[-1]
    ema_200 = ema(close, 200).iloc[-1]
    high_52 = close.iloc[-252:].max()
    low_52  = close.iloc[-252:].min()

    checks["price_above_50"]    = last > ema_50
    checks["price_above_150"]   = last > ema_150
    checks["price_above_200"]   = last > ema_200
    checks["ema150_above_200"]  = ema_150 > ema_200
    checks["ema50_above_150"]   = ema_50 > ema_150
    checks["ema50_above_200"]   = ema_50 > ema_200
    checks["within_25pct_high"] = last >= high_52 * 0.75
    checks["low_30pct_below"]   = low_52 <= last * 0.70

    score = sum(checks.values()) / len(checks) * 10
    return round(score, 2), checks
