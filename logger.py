"""
logger.py — Appends every BUY/SELL signal to signals_log.csv.

Columns:
  timestamp, ticker, signal, price, confidence,
  entry_low, entry_high, target, stop, reason

Usage (called automatically by main.py):
  from logger import log_signal
  log_signal(state)          # state dict from the LangGraph result

Read back for analysis:
  from logger import read_log
  rows = read_log(limit=100) # list of dicts
"""

import csv
import os
import threading
from datetime import datetime

LOG_FILE = "signals_log.csv"

COLUMNS = [
    "timestamp",
    "ticker",
    "signal",
    "price",
    "confidence",
    "entry_low",
    "entry_high",
    "target",
    "stop",
    "reason",
]

_lock = threading.Lock()


# ── Internal helpers ───────────────────────────────────────────────────────────

def _ensure_header() -> None:
    """Write the CSV header if the file is new or empty."""
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNS)


def _parse_entry_zone(entry_zone: str, price: float) -> tuple[float, float]:
    """
    Parse strings like '$1.20 - $1.25' or '1.20 - 1.25' into (low, high).
    Falls back to (price, price) on any parse failure.
    """
    try:
        clean = entry_zone.replace("$", "").replace(",", "").strip()
        # Split on ' - ' or '-' between numbers
        parts = [p.strip() for p in clean.split("-") if p.strip()]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
        if len(parts) == 1:
            v = float(parts[0])
            return v, v
    except Exception:
        pass
    return price, price


# ── Public API ─────────────────────────────────────────────────────────────────

def log_signal(state: dict) -> None:
    """
    Append one row to signals_log.csv for a BUY or SELL signal.
    HOLD signals are silently ignored.
    Thread-safe — can be called from any thread or async executor.
    """
    signal = state.get("signal", "HOLD")
    if signal not in ("BUY", "SELL"):
        return

    try:
        ticker     = state.get("ticker", "")
        price      = float(state.get("current_price", 0))
        confidence = int(state.get("confidence", 0))
        entry_zone = state.get("entry_zone", "")
        targets    = state.get("targets", [])
        stop       = float(state.get("stop_loss", 0))
        reasoning  = str(state.get("reasoning", "")).replace("\n", " ").strip()[:500]
        timestamp  = datetime.now().isoformat(timespec="seconds")

        entry_low, entry_high = _parse_entry_zone(entry_zone, price)
        target = float(targets[0]) if targets else 0.0

        row = [
            timestamp,
            ticker,
            signal,
            f"{price:.6f}",
            confidence,
            f"{entry_low:.6f}",
            f"{entry_high:.6f}",
            f"{target:.6f}",
            f"{stop:.6f}",
            reasoning,
        ]

        with _lock:
            _ensure_header()
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

        icon = "🟢" if signal == "BUY" else "🔴"
        print(f"📝 [Logger] {icon} {signal} logged → {LOG_FILE}  (conf={confidence})")

    except Exception as e:
        print(f"⚠️  [Logger] Failed to log signal: {e}")


def read_log(limit: int = 100) -> list[dict]:
    """
    Return the last `limit` rows from signals_log.csv as a list of dicts.
    Returns [] if the file does not exist yet.
    """
    try:
        if not os.path.exists(LOG_FILE):
            return []
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-limit:] if len(rows) > limit else rows
    except Exception as e:
        print(f"⚠️  [Logger] Failed to read log: {e}")
        return []
