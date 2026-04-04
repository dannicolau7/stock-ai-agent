"""
market_scanner.py — Broad market sweep for the 7:45 AM morning scan.

Universe: up to ~6,000 active US stocks fetched from Polygon (cached daily).
Data:     yfinance bulk download in batches of 500 — no per-ticker API calls.
Scoring:  pure-math RSI + MACD + Bollinger + volume. No Polygon, no Claude.

Cost: $0 (uses existing Polygon key only for ticker list, not OHLCV).
"""

import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import requests
import yfinance as yf

from agents.tech_agent import _calc_rsi, _calc_macd, _calc_bollinger
from config import POLYGON_API_KEY

CACHE_FILE  = "ticker_universe.json"
CACHE_HOURS = 24   # refresh ticker list once per day

# ── Fallback static universe (used if Polygon fetch fails) ─────────────────────

FALLBACK_UNIVERSE = [
    "BBAI", "SOUN", "BZAI", "AWRE", "LTRX",
    "CXAI", "GFAI", "AITX", "POET", "ARQQ",
    "IONQ", "RGTI", "QUBT", "QBTS", "KULR",
    "CIFR", "BTBT", "PRCT", "WRAP", "MIST",
    "CPIX", "NVAX", "IOVA", "IMVT",
    "NVDA", "AMD", "SMCI", "MSTR", "PLTR",
    "RKLB", "JOBY", "ACHR", "LUNR", "SPCE",
    "FCEL", "PLUG", "BLNK", "CHPT", "BE",
    "AMC", "GME", "CLOV",
]


# ── Ticker universe (Polygon) ──────────────────────────────────────────────────

def _fetch_polygon_tickers(max_tickers: int = 6000) -> list:
    """
    Fetch active US stock tickers from Polygon reference endpoint.
    Paginates automatically. Returns list of ticker strings.
    Rate: uses ~6 Polygon calls for 6,000 tickers (free tier ok).
    """
    tickers = []
    url     = "https://api.polygon.io/v3/reference/tickers"
    params  = {
        "apiKey":  POLYGON_API_KEY,
        "active":  "true",
        "market":  "stocks",
        "locale":  "us",
        "limit":   1000,
        "order":   "asc",
        "sort":    "ticker",
    }

    print(f"📋 [Scanner] Fetching ticker universe from Polygon...")
    while len(tickers) < max_tickers:
        try:
            r    = requests.get(url, params=params, timeout=20)
            data = r.json()
        except Exception as e:
            print(f"⚠️  [Scanner] Polygon ticker fetch error: {e}")
            break

        results = data.get("results", [])
        tickers.extend(t["ticker"] for t in results if "ticker" in t)

        cursor = data.get("next_cursor")
        if not cursor or not results:
            break

        params  = {"apiKey": POLYGON_API_KEY, "cursor": cursor}
        time.sleep(13)   # 5 req/min = 12s between calls; 13s for safety

    print(f"✅ [Scanner] Got {len(tickers)} tickers from Polygon")
    return tickers[:max_tickers]


def _load_universe(max_tickers: int = 6000) -> list:
    """
    Returns ticker universe from cache (refreshed daily) or Polygon fetch.
    Falls back to FALLBACK_UNIVERSE if Polygon is unavailable.
    """
    # Check cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            age_h = (datetime.now().timestamp() - cache.get("ts", 0)) / 3600
            if age_h < CACHE_HOURS:
                tickers = cache.get("tickers", [])
                print(f"✅ [Scanner] Loaded {len(tickers)} tickers from cache ({age_h:.1f}h old)")
                return tickers[:max_tickers]
        except Exception:
            pass

    # Fetch fresh
    tickers = _fetch_polygon_tickers(max_tickers)
    if tickers:
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump({"ts": datetime.now().timestamp(), "tickers": tickers}, f)
        except Exception:
            pass
        return tickers

    # Fallback
    print(f"⚠️  [Scanner] Using fallback universe ({len(FALLBACK_UNIVERSE)} tickers)")
    return FALLBACK_UNIVERSE


# ── Data fetch (batched) ──────────────────────────────────────────────────────

def _bulk_download_batched(tickers: list, batch_size: int = 500,
                            period: str = "3mo") -> dict:
    """
    Batch yfinance downloads. Each batch = 1 HTTP call.
    6,000 tickers / 500 per batch = 12 batches ~ 2-3 minutes.
    """
    result = {}
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    print(f"📥 [Scanner] Downloading {len(tickers)} tickers in {len(batches)} batches...")

    for idx, batch in enumerate(batches):
        try:
            raw = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            single = len(batch) == 1
            for ticker in batch:
                try:
                    df = raw if single else raw.get(ticker)
                    if df is None or df.empty:
                        continue
                    df = df.dropna(how="all")
                    if len(df) < 30:
                        continue
                    result[ticker] = {
                        "closes":  df["Close"].values.astype(float),
                        "volumes": df["Volume"].values.astype(float),
                        "price":   float(df["Close"].iloc[-1]),
                    }
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️  [Scanner] Batch {idx+1} failed: {e}")

        if idx < len(batches) - 1:
            time.sleep(1)  # small pause between batches

    print(f"✅ [Scanner] Got data for {len(result)}/{len(tickers)} tickers")
    return result


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_ticker(closes: np.ndarray, volumes: np.ndarray) -> dict:
    """Pure-math score 0-100. Higher = stronger BUY setup."""
    score   = 0
    reasons = []

    rsi       = _calc_rsi(closes)
    macd      = _calc_macd(closes)
    bollinger = _calc_bollinger(closes)
    price     = float(closes[-1])
    hist      = macd["histogram"]

    avg_vol   = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 1.0
    cur_vol   = float(volumes[-1])
    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

    # RSI (max 30 pts)
    if rsi < 30:
        score += 30; reasons.append(f"RSI {rsi:.0f} oversold")
    elif rsi < 40:
        score += 20; reasons.append(f"RSI {rsi:.0f}")
    elif rsi < 50:
        score += 10
    elif rsi > 70:
        score -= 10

    # MACD histogram (max 20 pts)
    if hist > 0:
        score += 20; reasons.append("MACD bullish")
    elif hist > -0.001:
        score += 5

    # Bollinger position (max 20 pts)
    if price <= bollinger["lower"]:
        score += 20; reasons.append("BB oversold")
    elif price <= bollinger["middle"]:
        score += 10

    # Volume spike (max 20 pts)
    if vol_ratio >= 3.0:
        score += 20; reasons.append(f"vol {vol_ratio:.1f}x")
    elif vol_ratio >= 2.0:
        score += 15; reasons.append(f"vol {vol_ratio:.1f}x")
    elif vol_ratio >= 1.5:
        score += 8

    # Day change momentum (max 10 pts)
    if len(closes) >= 2 and closes[-2]:
        day_chg = (closes[-1] - closes[-2]) / closes[-2] * 100
        if 0 < day_chg < 5:
            score += 10
        elif day_chg >= 5:
            score += 5
        elif day_chg < -5:
            score -= 5

    return {
        "score":     max(0, min(100, score)),
        "price":     round(price, 4),
        "rsi":       round(rsi, 1),
        "vol_ratio": round(vol_ratio, 2),
        "macd_hist": round(hist, 6),
        "reason":    ", ".join(reasons) if reasons else "no strong signal",
    }


# ── Public entry point ────────────────────────────────────────────────────────

def scan_broad_market(extra_tickers: list = None, top_n: int = 5,
                       max_universe: int = 6000) -> list:
    """
    Broad market sweep. Fetches ~6,000 ticker universe from Polygon (cached
    daily), then bulk-downloads OHLCV via yfinance, scores every ticker,
    returns top_n by score.

    Args:
        extra_tickers: always included (e.g. watchlist)
        top_n:         number of top results to return
        max_universe:  cap on ticker universe size

    Returns:
        list of dicts with ticker, score, price, rsi, vol_ratio, reason
    """
    universe = _load_universe(max_universe)

    # Ensure watchlist tickers are always included
    if extra_tickers:
        universe = list(dict.fromkeys(extra_tickers + universe))

    # Filter out obvious garbage (very short tickers are usually fine;
    # skip tickers with special chars that yfinance can't handle)
    universe = [t for t in universe if t.isalpha() and 1 <= len(t) <= 5]

    data = _bulk_download_batched(universe)

    results = []
    for ticker, d in data.items():
        try:
            scored           = _score_ticker(d["closes"], d["volumes"])
            scored["ticker"] = ticker
            results.append(scored)
        except Exception:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:top_n]
    print(f"🏆 [Scanner] Top {top_n}: {[(r['ticker'], r['score']) for r in top]}")
    return top


if __name__ == "__main__":
    import watchlist_manager as wl
    watchlist = wl.load()
    print(f"Starting broad market scan (up to 6,000 stocks)...\n")
    results = scan_broad_market(extra_tickers=watchlist, top_n=10)
    print("\nTop opportunities:")
    for i, r in enumerate(results, 1):
        print(
            f"  {i:2d}. {r['ticker']:6s}  score={r['score']:3d}"
            f"  ${r['price']:.2f}  RSI {r['rsi']:.0f}"
            f"  vol {r['vol_ratio']:.1f}x  |  {r['reason']}"
        )
