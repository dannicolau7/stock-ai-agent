"""
watchlist_manager.py — Persists tickers to watchlist.json between sessions.

Typical use from main.py:
  import watchlist_manager as wl
  tickers = wl.load()          # ['BZAI', 'NVDA']
  wl.add('AAPL')
  wl.remove('NVDA')
  wl.list_tickers()

File format (watchlist.json):
  { "tickers": ["BZAI", "NVDA", "AAPL"] }
"""

import json
import os

WATCHLIST_FILE = "watchlist.json"


def load() -> list[str]:
    """Return list of tickers from watchlist.json. Empty list if file absent."""
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, encoding="utf-8") as f:
                data = json.load(f)
            return [t.upper().strip() for t in data.get("tickers", []) if t.strip()]
    except Exception as e:
        print(f"⚠️  [Watchlist] Load error: {e}")
    return []


def save(tickers: list[str]) -> None:
    """Persist the tickers list to watchlist.json."""
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump({"tickers": [t.upper() for t in tickers]}, f, indent=2)
    except Exception as e:
        print(f"⚠️  [Watchlist] Save error: {e}")


def add(ticker: str) -> list[str]:
    """Add ticker to the watchlist. No-op if already present. Returns updated list."""
    ticker  = ticker.upper().strip()
    current = load()
    if ticker in current:
        print(f"⚠️  {ticker} is already in the watchlist")
        return current
    current.append(ticker)
    save(current)
    print(f"✅ Added {ticker}  ({len(current)} total)")
    return current


def remove(ticker: str) -> list[str]:
    """Remove ticker from the watchlist. No-op if not present. Returns updated list."""
    ticker  = ticker.upper().strip()
    current = load()
    if ticker not in current:
        print(f"⚠️  {ticker} not found in watchlist")
        return current
    current.remove(ticker)
    save(current)
    print(f"✅ Removed {ticker}  ({len(current)} remaining)")
    return current


def list_tickers() -> list[str]:
    """Print and return the current watchlist."""
    current = load()
    if not current:
        print("📋 Watchlist is empty.  Use --add TICKER to add one.")
    else:
        print(f"📋 Watchlist  ({len(current)} tickers):")
        for i, t in enumerate(current, 1):
            print(f"   {i:>2}.  {t}")
    return current
