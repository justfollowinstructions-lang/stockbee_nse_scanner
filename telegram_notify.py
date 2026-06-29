"""
NSE Stockbee Scanner — Telegram Notifier
=========================================
Based on Pradeep Bonde (Stockbee) methodology.

Fixed from Darvas Box version:
  • Removed SCORE_THRESHOLDS import (doesn't exist in Bonde config)
  • send_report(text: str) — accepts pre-formatted string from report.py
  • send_report(text, report_path) — optional Excel attachment
  • Markdown retry with plain text fallback on 400 errors
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests

from config import (
    DATA_DIR,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_MAX_MSG,
)
from logger_utils import get_logger

log = get_logger("scanner")

_SENT_FILE = DATA_DIR / "telegram_sent.json"
_API_BASE  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


# ─── Public API ───────────────────────────────────────────────────────────────

def send_report(text: str, report_path: Optional[Path] = None) -> None:
    """
    Send a pre-formatted text message and optionally attach an Excel file.

    Called from main.py as:
        send_report(card)                    — single string card
        send_report(report)                  — market monitor / weekly report
        send_report(text, report_path=path)  — with Excel attachment

    Parameters
    ----------
    text        : Markdown-formatted string already built by report.py
    report_path : optional Path to an Excel file to attach
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not set — skipping notification")
        return

    if not text or not text.strip():
        log.warning("send_report called with empty text — skipping")
        return

    # Split and send (Telegram 4096-char limit)
    for chunk in _split_message(text):
        ok = _send_message(chunk)
        if not ok:
            log.error("Failed to send Telegram chunk — aborting further sends")
            return
        time.sleep(0.4)   # stay well under Telegram 30 msg/sec limit

    # Optional Excel attachment
    if report_path is not None:
        path = Path(report_path)
        if path.exists():
            _send_document(path)
        else:
            log.warning("report_path does not exist: %s", path)


# ─── Private helpers ──────────────────────────────────────────────────────────

def _send_message(text: str) -> bool:
    """POST a single text message. Returns True on success."""
    try:
        resp = requests.post(
            f"{_API_BASE}/sendMessage",
            json={
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       text,
                "parse_mode": "Markdown",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except requests.exceptions.HTTPError as e:
        # Telegram 400 = bad Markdown — retry as plain text
        log.warning("Telegram Markdown send failed (%s) — retrying as plain text", e)
        try:
            resp = requests.post(
                f"{_API_BASE}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                timeout=15,
            )
            resp.raise_for_status()
            return True
        except Exception as e2:
            log.error("Telegram plain-text send also failed: %s", e2)
            return False
    except Exception as e:
        log.error("Telegram send error: %s", e)
        return False


def _send_document(path: Path) -> bool:
    """Attach an Excel/PDF file to the chat."""
    try:
        with open(path, "rb") as f:
            resp = requests.post(
                f"{_API_BASE}/sendDocument",
                data={"chat_id": TELEGRAM_CHAT_ID},
                files={"document": (path.name, f)},
                timeout=60,
            )
        resp.raise_for_status()
        log.info("Telegram: attached %s", path.name)
        return True
    except Exception as e:
        log.error("Telegram document send failed: %s", e)
        return False


def _split_message(text: str) -> list[str]:
    """Split text into ≤ TELEGRAM_MAX_MSG chunks on newline boundaries."""
    limit = TELEGRAM_MAX_MSG or 4000
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > limit:
            if current:
                chunks.append(current)
            # Hard-chop lines that are themselves too long
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            current = line
        else:
            current = (current + "\n" + line) if current else line
    if current:
        chunks.append(current)
    return chunks
