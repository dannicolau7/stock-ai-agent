"""
utils/earnings_gate.py — Earnings blackout window gate.

check_earnings_blackout(ticker) → dict:
  blocked:       bool   — True if earnings ≤ BLACKOUT_DAYS away
  days_until:    int    — calendar days to next earnings (999 = unknown)
  earnings_date: str    — ISO date string or ""
  warning:       str    — human-readable message, "" when not blocked

Designed to be called from multiple scanners; results are cached for 24 hours
per ticker to avoid redundant yfinance API calls.

Constants:
  BLACKOUT_DAYS = 10   hard block for new entries
  EXIT_DAYS     = 5    force-close open positions (used by portfolio_agent)
"""

import csv
import threading
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

import yfinance as yf

# ── Constants ──────────────────────────────────────────────────────────────────

BLACKOUT_DAYS  = 10    # block new entries within this many days of earnings
EXIT_DAYS      = 5     # force-close open positions within this many days

_BLOCKED_LOG   = Path(__file__).parent.parent / "data" / "blocked_signals.csv"
_LOG_FIELDS    = [
    "ts", "ticker", "date_blocked", "earnings_date",
    "days_until_earnings", "original_signal", "rvol", "reason", "source",
]

# ── 24-hour cache ──────────────────────────────────────────────────────────────

_cache: dict       = {}
_cache_lock        = threading.Lock()
_CACHE_TTL_S       = 86_400   # 24 hours


# ── yfinance calendar parsing (mirrors data_agent._fetch_earnings_info) ────────

def _fetch_next_earnings(ticker: str) -> Optional[date]:
    """
    Returns the next upcoming earnings date for ticker, or None if unavailable.
    Tries multiple column names to handle yfinance schema variations.
    """
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None or (hasattr(cal, "empty") and cal.empty):
            return None

        today = datetime.now(timezone.utc).date()

        for col in ("Earnings Date", "Earnings Dates", 0):
            try:
                val = cal[col] if isinstance(col, str) else cal.iloc[:, col]
            except Exception:
                continue
            if val is None:
                continue

            items = val if hasattr(val, "__iter__") else [val]
            for raw in items:
                try:
                    d = raw.date() if hasattr(raw, "date") else \
                        datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
                    if (d - today).days >= 0:
                        return d
                except Exception:
                    continue
    except Exception:
        pass
    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def check_earnings_blackout(ticker: str) -> dict:
    """
    Returns earnings blackout status for ticker.
    Result is cached for 24 hours.

    Return dict:
      blocked       bool
      days_until    int   (999 = no data)
      earnings_date str   (ISO date or "")
      warning       str   (non-empty only when blocked or no data)
    """
    with _cache_lock:
        cached = _cache.get(ticker)
        if cached:
            age_s = (datetime.now() - cached["cached_at"]).total_seconds()
            if age_s < _CACHE_TTL_S:
                return cached["result"]

    earnings_dt = _fetch_next_earnings(ticker)
    today       = datetime.now(timezone.utc).date()

    if earnings_dt is None:
        result = {
            "blocked":       False,
            "days_until":    999,
            "earnings_date": "",
            "warning":       f"⚠️ No earnings calendar data for {ticker}",
        }
    else:
        days    = (earnings_dt - today).days
        blocked = days <= BLACKOUT_DAYS
        result  = {
            "blocked":       blocked,
            "days_until":    days,
            "earnings_date": str(earnings_dt),
            "warning":       (
                f"⚠️ {ticker} earnings in {days} days — signal blocked"
                if blocked else ""
            ),
        }

    with _cache_lock:
        _cache[ticker] = {"result": result, "cached_at": datetime.now()}

    return result


def invalidate_cache(ticker: str) -> None:
    """Force a fresh fetch next time check_earnings_blackout is called."""
    with _cache_lock:
        _cache.pop(ticker, None)


# ── Logging ────────────────────────────────────────────────────────────────────

def log_earnings_block(
    ticker:          str,
    original_signal: str,
    days_until:      int,
    earnings_date:   str,
    source:          str = "earnings_gate",
) -> None:
    """Append an earnings-blocked row to data/blocked_signals.csv."""
    try:
        _BLOCKED_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_header = not _BLOCKED_LOG.exists()
        with open(_BLOCKED_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_LOG_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({
                "ts":                   datetime.now().isoformat(),
                "ticker":               ticker,
                "date_blocked":         datetime.now().date().isoformat(),
                "earnings_date":        earnings_date,
                "days_until_earnings":  days_until,
                "original_signal":      original_signal,
                "rvol":                 "",
                "reason":               f"Earnings in {days_until}d",
                "source":               source,
            })
    except Exception as e:
        print(f"⚠️  [EarningsGate] blocked_signals log failed: {e}")
