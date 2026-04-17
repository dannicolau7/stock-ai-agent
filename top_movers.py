"""
top_movers.py — Fetch today's top gainers and most active stocks from Yahoo Finance.

No API key required. Called every monitoring cycle to ensure the pipeline always
has real market movers to analyze, not just the static watchlist.
"""

import time
import requests

_HEADERS     = {"User-Agent": "Mozilla/5.0"}
_YF_SCREENER = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
_CACHE: dict = {"ts": 0.0, "tickers": []}
_CACHE_TTL   = 5 * 60   # refresh every 5 minutes


def get_top_movers(
    count: int = 20,
    min_price: float = 1.0,
    max_price: float = 500.0,
    min_volume: int = 500_000,
) -> list:
    """
    Return today's top movers (gainers + most active) as a deduplicated ticker list.
    Results are cached for 5 minutes to avoid hammering Yahoo Finance.

    Filters:
      - price between min_price and max_price
      - volume ≥ min_volume (avoids illiquid micro-caps)
      - ticker is pure alpha, 1–5 chars (no warrants/units)
    """
    if time.time() - _CACHE["ts"] < _CACHE_TTL and _CACHE["tickers"]:
        return _CACHE["tickers"]

    seen    = set()
    tickers = []

    for scr_id in ("day_gainers", "most_actives"):
        try:
            r = requests.get(
                _YF_SCREENER,
                params={"scrIds": scr_id, "count": count, "region": "US", "lang": "en-US"},
                headers=_HEADERS,
                timeout=10,
            )
            if r.status_code != 200:
                continue
            quotes = r.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
            for q in quotes:
                sym    = q.get("symbol", "")
                price  = q.get("regularMarketPrice", 0.0) or 0.0
                volume = q.get("regularMarketVolume", 0) or 0

                if (sym
                        and sym.isalpha()
                        and 1 <= len(sym) <= 5
                        and min_price <= price <= max_price
                        and volume >= min_volume
                        and sym not in seen):
                    seen.add(sym)
                    tickers.append(sym)
        except Exception:
            pass

    _CACHE["ts"]      = time.time()
    _CACHE["tickers"] = tickers
    return tickers
