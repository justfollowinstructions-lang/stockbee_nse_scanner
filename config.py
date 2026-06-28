"""
NSE Stockbee Scanner — Configuration
=====================================
Based on Pradeep Bonde (Stockbee) methodology:
  • Momentum Burst (MB) — 3-5 day swing breakouts
  • Episodic Pivot (EP) — catalyst-driven, larger moves
  • Market Monitor — situational awareness breadth engine

All Darvas Box parameters have been replaced.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
DAILY_DIR      = DATA_DIR / "daily"
SIGNALS_DIR    = DATA_DIR / "signals"
WATCHLIST_DIR  = DATA_DIR / "watchlist"
PERF_DIR       = DATA_DIR / "performance"
LOGS_DIR       = BASE_DIR / "logs"
REPORTS_DIR    = BASE_DIR / "reports"

for d in [DAILY_DIR, SIGNALS_DIR, WATCHLIST_DIR, PERF_DIR, LOGS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database files
SIGNALS_DB     = DATA_DIR / "signals.db"
WATCHLIST_DB   = DATA_DIR / "watchlist.db"
PERFORMANCE_DB = DATA_DIR / "performance.db"

# ─── Data Sources ─────────────────────────────────────────────────────────────
DATA_SOURCES    = ["yfinance", "stooq"]
PRIMARY_SOURCE  = "yfinance"
FALLBACK_SOURCE = "stooq"

# ─── Universe ─────────────────────────────────────────────────────────────────
NIFTY50_SYMBOL  = "^NSEI"
NIFTY500_SYMBOL = "^CRSLDX"

# ─── Account Settings ─────────────────────────────────────────────────────────
ACCOUNT_SIZE       = float(os.getenv("ACCOUNT_SIZE", "1_000_000"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PCT", "1.0"))

# ─── Liquidity & Universe Filters ─────────────────────────────────────────────
MIN_PRICE        = 50.0       # Minimum stock price in INR
MAX_PRICE        = 50_000.0   # Maximum stock price in INR
MIN_AVG_VOLUME   = 200_000    # 20-day avg volume ≥ 200,000 shares
MIN_HISTORY_DAYS = 260        # Need 1 year+ of history for RS ranking

# ─── Stockbee Momentum Burst Parameters ──────────────────────────────────────
MB_BREAKOUT_PCT          = 4.0    # Min % move on breakout day
MB_BREAKOUT_PCT_LARGE    = 3.5    # For Nifty 50 large caps
MB_VOLUME_RATIO_MIN      = 1.0    # Volume today must be > yesterday
MB_CLOSE_STRENGTH_MIN    = 0.70   # (close-low)/(high-low) ≥ 0.70 (H rule)
MB_CONSOL_MIN_BARS       = 5      # Min consolidation bars
MB_CONSOL_MAX_BARS       = 20     # Max consolidation bars
MB_CONSOL_MAX_WIDTH_PCT  = 12.0   # Max % width of consolidation box
MB_CONSOL_NEG_DAY_PCT    = -4.0   # Any day < -4% inside consolidation = fail
MB_CONSOL_VOL_RATIO_MAX  = 0.90   # Consol avg vol must be < 90% of 50d avg
MB_PRIOR_MOVE_MIN_PCT    = 8.0    # Prior uptrend must be ≥ 8% before consolidation
MB_PRIOR_MOVE_BARS       = 60     # Bars to look back for prior uptrend

# 2LYNCH checklist thresholds
MB_2_MAX_UP_DAYS_BEFORE  = 1      # Max 1 up day in last 2 bars before signal
MB_L_LINEARITY_MIN       = 0.60   # % of bars that must close up in prior trend
MB_N_NARROW_ATR_RATIO    = 0.75   # Pre-breakout bar range < 75% of 10-day ATR
MB_N_NEGATIVE_MAX_DOWN   = -0.02  # Pre-breakout bar can be down up to -2%

# ─── Anticipation Setup Parameters ────────────────────────────────────────────
ANT_CONSOL_MIN      = 3    # Min consolidation bars for anticipation watch
ANT_CONSOL_MAX      = 15   # Max consolidation bars for anticipation watch
ANT_PRIOR_MOVE_MIN  = 10.0 # Prior move up must be ≥ 10%
ANT_VOL_DRY_RATIO   = 0.70 # Volume drying: consol avg < 70% of 50d avg

# ─── Trend Intensity (TI65) ───────────────────────────────────────────────────
# TI65 = 7-day SMA / 65-day SMA
TI65_BULL_THRESHOLD   = 1.03  # > 1.03 = bullish momentum (entry filter)
TI65_STRONG_THRESHOLD = 1.05  # > 1.05 = strong momentum (aggressive)
TI65_BEAR_THRESHOLD   = 0.97  # < 0.97 = downtrend, avoid longs

# ─── IBD-style RS Rating ─────────────────────────────────────────────────────
# RS = 0.40*(C/C65) + 0.20*(C/C130) + 0.20*(C/C195) + 0.20*(C/C260)
# Then cross-sectionally ranked 1-99 across entire Nifty 500 universe
RS_WEIGHT_Q1     = 0.40   # Most recent quarter weight (65 bars)
RS_WEIGHT_Q2     = 0.20   # Q2 weight (65-130 bars)
RS_WEIGHT_Q3     = 0.20   # Q3 weight (130-195 bars)
RS_WEIGHT_Q4     = 0.20   # Q4 weight (195-260 bars)
RS_WEIGHTS       = {"3m": RS_WEIGHT_Q1, "6m": RS_WEIGHT_Q2,
                    "9m": RS_WEIGHT_Q3, "12m": RS_WEIGHT_Q4}
RS_MIN_FOR_MB    = 70     # Min RS rank for MB setups
RS_MIN_FOR_EP    = 60     # Min RS rank for EP (catalyst can lift neglected stocks)
RS_STRONG        = 85     # RS ≥ 85 = institutional-grade strength

# ─── Episodic Pivot Parameters ────────────────────────────────────────────────
EP_GAP_MIN_PCT           = 5.0   # Min gap-up % (NSE circuits mean 5% is meaningful)
EP_GAP_STRONG_PCT        = 10.0  # Strong EP gap ≥ 10%
EP_VOLUME_SPIKE_RATIO    = 3.0   # EP day volume ≥ 3x 50-day avg
EP_9M_VOLUME_SPIKE       = 5.0   # 9M EP: spike ≥ 5x 50-day avg
EP_9M_PRIOR_QUIET_DAYS   = 20    # Prior 20 days: no day with vol > 2x avg
EP_CLOSE_STRENGTH_MIN    = 0.60  # EP close strength ≥ 60% of range (not fading)
EP_SALES_GROWTH_MIN      = 39.0  # MAGNA: min quarterly sales growth %
EP_EPS_GROWTH_MIN        = 39.0  # MAGNA: min quarterly EPS growth %
EP_NEGLECT_MONTHS        = 3     # Stock quiet for ≥ 3 months = neglected
EP_STOP_PCT_NORMAL       = 2.5   # Normal EP stop loss %
EP_STOP_PCT_HIGH_CONV    = 10.0  # High conviction EP stop %
EP_DELAYED_REACTION_DAYS = 3     # Delayed Reaction EP: enter within 3 days

# ─── Market Monitor Parameters ────────────────────────────────────────────────
MM_BULL_PCT_ABOVE_200  = 60.0  # % of Nifty 500 above 200 EMA = BULL
MM_BEAR_PCT_ABOVE_200  = 40.0  # Below this % = BEAR (or CAUTION)
MM_BULL_UP4_COUNT      = 15    # ≥ 15 stocks up 4%+ = strong momentum day
MM_BEAR_DOWN4_COUNT    = 30    # ≥ 30 stocks down 4%+ = distribution
MM_BULL_AD_RATIO       = 1.5   # Advance/Decline ≥ 1.5 = bullish day
MM_BEAR_AD_RATIO       = 0.5   # Advance/Decline ≤ 0.5 = bearish day
MM_WEEKLY_WINNERS_PCT  = 0.03  # ≥ 3% of stocks up 20%+ in 5 days = momentum mkt
MM_52W_HIGH_STRONG     = 0.03  # ≥ 3% at 52w high = strong market
MM_52W_LOW_DANGER      = 0.02  # ≥ 2% at 52w low = danger zone

# ─── Risk Management ──────────────────────────────────────────────────────────
MB_HOLD_MAX_DAYS     = 5    # NEVER hold MB beyond 5 days
MB_PARTIAL_EXIT_DAY  = 3    # Sell 50% on Day 3 regardless of price
MB_STOP_PCT_LARGE    = 5.0  # Stop for large cap MB
MB_STOP_PCT_SMALL    = 7.0  # Stop for small cap MB
EP_HOLD_MAX_DAYS     = 30   # EPs can be held for trend moves

ATR_PERIOD           = 14
ATR_STOP_MULTIPLIER  = 1.5  # Used for EP trailing stop

# ─── Moving Average Periods ───────────────────────────────────────────────────
EMA_SHORT  = 20
EMA_MID    = 50
EMA_LONG   = 150
EMA_TREND  = 200

# ─── Composite Scoring Weights ────────────────────────────────────────────────
MB_SCORE_WEIGHTS = {
    "rs_rank":        25,   # Cross-sectional RS rank (most important)
    "ti65":           20,   # Trend intensity (absolute momentum)
    "twolynch_score": 25,   # 2LYNCH quality checklist (5 pts each)
    "consolidation":  15,   # Consolidation tightness & volume dry-up
    "volume_ratio":   10,   # Volume expansion on breakout
    "linearity":       5,   # Prior trend linearity
}

EP_SCORE_WEIGHTS = {
    "gap_size":        20,  # Gap magnitude
    "volume_spike":    25,  # Volume spike ratio
    "catalyst_quality": 30, # MAGNA criteria met
    "neglect_score":   15,  # How neglected was the stock
    "rs_rank":         10,  # RS rank (lower weight: catalyst drives EP)
}

MB_SCORE_THRESHOLDS = {
    "elite":  85,
    "strong": 70,
    "watch":  55,
}

EP_SCORE_THRESHOLDS = {
    "elite":  80,
    "strong": 65,
    "watch":  50,
}

# ─── Rate Limiting & Download ─────────────────────────────────────────────────
BATCH_SIZE                = 50
BATCH_DELAY_SECONDS       = 2.0
TIMEOUT_RETRY_WAIT_SEC    = 30
RATELIMIT_RETRY_WAIT_MIN  = 7
MAX_RETRIES               = 5
EXPONENTIAL_BASE          = 2
RETRY_WAIT_MINUTES        = RATELIMIT_RETRY_WAIT_MIN  # legacy alias

# ─── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_MAX_MSG   = 4000

# ─── Watchlist Expiry ─────────────────────────────────────────────────────────
WATCHLIST_EXPIRY_DAYS = 30

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"

LOG_FILES = {
    "download":       LOGS_DIR / "download.log",
    "update":         LOGS_DIR / "update.log",
    "scanner":        LOGS_DIR / "scanner.log",
    "error":          LOGS_DIR / "error.log",
    "performance":    LOGS_DIR / "performance.log",
    "signal_tracker": LOGS_DIR / "signal_tracker.log",
}
