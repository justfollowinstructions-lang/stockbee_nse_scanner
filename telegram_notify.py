"""
NSE Darvas Box Scanner - Telegram Notifier
==========================================
• Sends the Excel report as a document.
• Posts a formatted summary of top setups.
• Tracks sent signal IDs to prevent duplicate messages.
• Splits long messages automatically.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import requests

from config import (
    DATA_DIR, SCORE_THRESHOLDS, TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID, TELEGRAM_MAX_MSG,
)
from logger_utils import get_logger

log = get_logger("scanner")

_SENT_FILE = DATA_DIR / "telegram_sent.json"
API_BASE   = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


# ─── Public API ───────────────────────────────────────────────────────────────

def send_report(signals: list, report_path: Path) -> None:
    """Send Telegram messages for new signals and attach the Excel report."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not configured – skipping notification")
        return

    sent_ids = _load_sent()
    new_sigs  = [s for s in signals if s.signal_id not in sent_ids]

    if not new_sigs:
        log.info("No new signals to send to Telegram")
        return

    # ── Summary header ──────────────────────────────────────────────────────
    elite = [s for s in new_sigs if s.composite_score >= SCORE_THRESHOLDS["elite"]]
    vs    = [s for s in new_sigs if SCORE_THRESHOLDS["very_strong"] <= s.composite_score < SCORE_THRESHOLDS["elite"]]

    header = (
        f"📊 *NSE Darvas Box Scanner* – Bottom-of-Box Setups\n"
        f"🗓 {_today()}\n\n"
        f"🏆 Elite (≥90): {len(elite)}\n"
        f"⭐ Very Strong (80-89): {len(vs)}\n"
        f"📋 Total New Signals: {len(new_sigs)}\n"
        f"{'─' * 30}"
    )
    _send_message(header)

    # ── Top 10 signals ──────────────────────────────────────────────────────
    top = sorted(new_sigs, key=lambda s: s.composite_score, reverse=True)[:10]
    body_lines = []
    for sig in top:
        emoji = "🏆" if sig.composite_score >= 90 else "⭐" if sig.composite_score >= 80 else "✅"
        body_lines.append(
            f"\n{emoji} *{sig.symbol}* | Score: {sig.composite_score:.1f} | {sig.classification}\n"
            f"   💰 Price: ₹{sig.current_price:,.2f} | RS: {sig.rs_rating:.0f}\n"
            f"   📦 Box: ₹{sig.box_low:,.2f} – ₹{sig.box_high:,.2f}\n"
            f"   🎯 T1: ₹{sig.target1:,.2f} | T2: ₹{sig.target2:,.2f}\n"
            f"   🛑 SL: ₹{sig.stop_loss:,.2f} | R:R {sig.rr_ratio:.2f}x\n"
            f"   📈 RSI: {sig.rsi_val:.1f} | ADX: {sig.adx_val:.1f} | W: {sig.weekly_trend.upper()}"
        )

    full_body = "\n".join(body_lines)
    for chunk in _split_message(full_body):
        _send_message(chunk)
        time.sleep(0.5)

    # ── Excel attachment ─────────────────────────────────────────────────────
    _send_document(report_path)

    # ── Record sent ─────────────────────────────────────────────────────────
    sent_ids.update(s.signal_id for s in new_sigs)
    _save_sent(sent_ids)
    log.info("Telegram: sent %d new signals", len(new_sigs))


# ─── Private helpers ─────────────────────────────────────────────────────────

def _send_message(text: str) -> bool:
    try:
        resp = requests.post(
            f"{API_BASE}/sendMessage",
            json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       text,
                "parse_mode": "Markdown",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log.error("Telegram message failed: %s", e)
        return False


def _send_document(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                f"{API_BASE}/sendDocument",
                data={"chat_id": TELEGRAM_CHAT_ID},
                files={"document": (path.name, f)},
                timeout=60,
            )
        resp.raise_for_status()
        return True
    except Exception as e:
        log.error("Telegram document failed: %s", e)
        return False


def _split_message(text: str) -> list[str]:
    if len(text) <= TELEGRAM_MAX_MSG:
        return [text]
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > TELEGRAM_MAX_MSG:
            chunks.append(current)
            current = line
        else:
            current = (current + "\n" + line) if current else line
    if current:
        chunks.append(current)
    return chunks


def _load_sent() -> set:
    if _SENT_FILE.exists():
        try:
            return set(json.loads(_SENT_FILE.read_text()))
        except Exception:
            pass
    return set()


def _save_sent(ids: set) -> None:
    _SENT_FILE.write_text(json.dumps(list(ids)))


def _today() -> str:
    from datetime import date
    return date.today().strftime("%A, %d %B %Y")
