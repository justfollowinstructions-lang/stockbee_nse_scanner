"""
NSE Darvas Box Scanner - Symbol Universe
Fetches the complete list of NSE-listed equity symbols.
Falls back to a cached list if the live fetch fails.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

from config import DATA_DIR, MIN_AVG_VOLUME, MIN_PRICE, MAX_PRICE
from logger_utils import get_logger

log = get_logger("download")

CACHE_FILE = DATA_DIR / "nse_symbols.json"
NSE_EQUITY_URL = (
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def fetch_nse_symbols(force_refresh: bool = False) -> list[str]:
    """
    Return a list of NSE equity symbols formatted for yfinance (e.g. 'RELIANCE.NS').
    Uses a local cache; refreshes daily or when *force_refresh* is True.
    """
    if not force_refresh and CACHE_FILE.exists():
        age_hours = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        if age_hours < 24:
            log.info("Using cached NSE symbol list (%d symbols)", _count_cache())
            return _load_cache()

    log.info("Fetching NSE equity master list …")
    symbols = _fetch_live()
    if symbols:
        _save_cache(symbols)
        log.info("Fetched %d NSE symbols from live source", len(symbols))
        return symbols

    # Fallback to cache even if stale
    if CACHE_FILE.exists():
        log.warning("Live fetch failed – using stale cached symbol list")
        return _load_cache()

    raise RuntimeError("Could not obtain NSE symbol list from any source.")


# ─── Private helpers ──────────────────────────────────────────────────────────

def _fetch_live() -> list[str]:
    session = requests.Session()
    session.headers.update(HEADERS)

    # Prime NSE cookies
    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1)
        resp = session.get(NSE_EQUITY_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))

        # Column is usually "SYMBOL"
        sym_col = next(c for c in df.columns if "SYMBOL" in c.upper())
        symbols = [f"{s.strip()}.NS" for s in df[sym_col].dropna().unique()]
        return sorted(set(symbols))
    except Exception as exc:
        log.error("Live NSE fetch failed: %s", exc)
        return []


def _load_cache() -> list[str]:
    with open(CACHE_FILE) as f:
        return json.load(f)


def _save_cache(symbols: list[str]) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(symbols, f)


def _count_cache() -> int:
    try:
        return len(_load_cache())
    except Exception:
        return 0
