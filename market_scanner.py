"""
market_scanner.py — v2: Best-of-Day selection algorithm + legacy broad sweep.

Best-of-Day pipeline  (scan_best_of_day):
  Stage 1  5 hard gates eliminate 99%+ of ~6,000 stocks (minimal API cost).
  Stage 2  Score survivors (max 225 pts) on RVOL, news recency, RSI,
           MACD/EMA cross, gap, Fibonacci support, price support,
           market cap, and price.
  Stage 3  Claude ranks top 3 and picks the single best setup.
  Stage 4  Format and send WhatsApp alert.
  Stage 5  Log pick to best_picks_log.csv for accuracy tracking.

Gate order (cheapest → most expensive):
  Gate 1  RVOL >= 2.0          bulk OHLCV — 0 extra calls
  Gate 3  RSI 28–67            bulk OHLCV — 0 extra calls
  Gate 4  Price $0.50–$50      bulk OHLCV — 0 extra calls
  Gate 5  Not in signals_log   local CSV  — 0 extra calls
  Gate 2  News in last 24h     Polygon API — ≤40 calls (rate-limited)

Legacy broad sweep  (scan_broad_market):
  Quick RSI/MACD/volume scan, no Claude, no news.
  Kept for backward compatibility with scheduler.py.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import anthropic
import numpy as np
import requests
import yfinance as yf

from agents.tech_agent import _calc_rsi, _ema
from config import POLYGON_API_KEY, ANTHROPIC_API_KEY
from features.relative_strength import fetch_benchmarks, compute_rs, SECTOR_ETFS
from features.news_classifier import classify as classify_news, score as news_score, label as news_label
from setups import detect_and_score

# ── Constants ──────────────────────────────────────────────────────────────────

CACHE_FILE     = "ticker_universe.json"
CACHE_HOURS    = 6    # refresh every 6h so hot new listings appear sooner
BEST_PICKS_LOG = "best_picks_log.csv"
SIGNALS_LOG    = "signals_log.csv"
BASE_URL       = "https://api.polygon.io"

RVOL_MIN          = 2.0
RSI_MIN           = 28.0
RSI_MAX           = 67.0
PRICE_MIN         = 0.50
PRICE_MAX         = 50.0
NEWS_HOURS_WINDOW = 24
MIN_SCORE         = 100   # score threshold to qualify for Claude
MAX_NEWS_CHECK    = 40    # max tickers to check news (API rate budget)
DOLLAR_VOL_MIN    = 100_000   # $100k 5-day avg dollar volume — minimum liquidity gate

BEST_PICKS_COLUMNS = [
    "date", "ticker", "setup_type", "total_score",
    "context_score", "setup_score", "execution_score", "risk_penalty",
    "price_at_pick", "news_headline", "news_category",
    "rvol", "rsi", "macd_cross", "ema_cross", "gap_pct",
    "rs_vs_spy", "dollar_vol_m",
    "price_1day_later", "price_3day_later", "actual_gain_loss_pct",
]

FALLBACK_UNIVERSE = [
    "CXAI", "GFAI", "AITX", "POET",
    "IONQ", "RGTI", "QUBT", "KULR", "CIFR", "BTBT", "PRCT", "NVAX", "IOVA",
    "NVDA", "AMD", "SMCI", "MSTR", "PLTR", "RKLB", "JOBY", "ACHR", "LUNR",
    "FCEL", "PLUG", "BLNK", "CHPT", "AMC", "GME",
]


# ── Ticker universe ────────────────────────────────────────────────────────────

def _fetch_polygon_tickers(max_tickers: int = 6000) -> list:
    """
    Fetch active US stock tickers from Polygon (NYSE + NASDAQ only).
    Polygon does not support comma-separated exchange values, so we fetch
    each exchange separately and merge, deduplicating by insertion order.
    On 429 rate-limit, waits 65s and retries once before giving up.
    """
    seen    = set()
    tickers = []

    for exchange in ("XNYS", "XNAS"):
        url    = f"{BASE_URL}/v3/reference/tickers"
        params = {
            "apiKey":   POLYGON_API_KEY,
            "active":   "true",
            "market":   "stocks",
            "locale":   "us",
            "limit":    1000,
            "order":    "asc",
            "sort":     "ticker",
            "exchange": exchange,
        }
        print(f"📋 [Scanner] Fetching {exchange} tickers from Polygon...")
        retried = False
        while len(tickers) < max_tickers:
            try:
                r    = requests.get(url, params=params, timeout=20)
                if r.status_code == 429:
                    if not retried:
                        print(f"⚠️  [Scanner] Polygon rate-limit on {exchange} — waiting 65s...")
                        time.sleep(65)
                        retried = True
                        continue
                    else:
                        print(f"⚠️  [Scanner] Polygon rate-limit persists for {exchange} — skipping")
                        break
                retried = False
                data = r.json()
            except Exception as e:
                print(f"⚠️  [Scanner] Polygon ticker fetch error ({exchange}): {e}")
                break
            results = data.get("results", [])
            for t in results:
                sym = t.get("ticker", "")
                if sym and sym not in seen:
                    seen.add(sym)
                    tickers.append(sym)
            cursor = data.get("next_cursor")
            if not cursor or not results:
                break
            params = {"apiKey": POLYGON_API_KEY, "cursor": cursor}
            time.sleep(13)   # 5 req/min on free tier

    print(f"✅ [Scanner] Got {len(tickers)} tickers from Polygon (NYSE+NASDAQ)")
    return tickers[:max_tickers]


def _fetch_sec_tickers(max_tickers: int = 6000) -> list:
    """
    Fallback universe: download all public company tickers from SEC EDGAR.
    No API key required. Returns ~8,000 tickers covering NYSE + NASDAQ + OTC.
    Filters to symbols that look like real stock tickers (1-5 alpha chars).
    """
    try:
        print("📋 [Scanner] Fetching fallback universe from SEC EDGAR...")
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "argus contact@example.com"},
            timeout=15,
        )
        data = r.json()
        tickers = [
            v["ticker"].upper()
            for v in data.values()
            if (v.get("ticker")
                and v["ticker"].isalpha()          # no warrants (W suffix), units (U), rights (R)
                and 1 <= len(v["ticker"]) <= 5
                and not v["ticker"].endswith(("W", "R", "U", "P", "Q"))  # warrants/rights/units/preferred/bankruptcy
            )
        ]
        print(f"✅ [Scanner] SEC fallback: {len(tickers)} tickers")
        return tickers[:max_tickers]
    except Exception as e:
        print(f"⚠️  [Scanner] SEC fallback failed: {e}")
        return []


def _load_universe(max_tickers: int = 6000) -> list:
    """Cache-aware ticker universe loader. Falls back to FALLBACK_UNIVERSE."""
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
    tickers = _fetch_polygon_tickers(max_tickers)
    if not tickers:
        tickers = _fetch_sec_tickers(max_tickers)
    if tickers:
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump({"ts": datetime.now().timestamp(), "tickers": tickers}, f)
        except Exception:
            pass
        return tickers
    print(f"⚠️  [Scanner] Using hardcoded fallback universe ({len(FALLBACK_UNIVERSE)} tickers)")
    return FALLBACK_UNIVERSE


# ── Bulk OHLCV download ────────────────────────────────────────────────────────

def _bulk_download_batched(tickers: list, batch_size: int = 500,
                            period: str = "3mo") -> dict:
    """
    Batch yfinance downloads — 1 HTTP call per batch of 500.
    Returns dict: ticker → {closes, opens, highs, lows, volumes, price}.
    """
    result  = {}
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    print(f"📥 [Scanner] Downloading {len(tickers)} tickers in {len(batches)} batches...")

    for idx, batch in enumerate(batches):
        try:
            raw    = yf.download(
                batch,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=False,   # avoid SQLite cache contention under concurrent loads
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
                        "opens":   df["Open"].values.astype(float),
                        "highs":   df["High"].values.astype(float),
                        "lows":    df["Low"].values.astype(float),
                        "volumes": df["Volume"].values.astype(float),
                        "price":   float(df["Close"].iloc[-1]),
                    }
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️  [Scanner] Batch {idx+1} failed: {e}")
        if idx < len(batches) - 1:
            time.sleep(1)

    print(f"✅ [Scanner] Got data for {len(result)}/{len(tickers)} tickers")
    return result


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _macd_hist_series(closes: np.ndarray, fast=12, slow=26, sig=9) -> np.ndarray:
    """Full MACD histogram array (not just the last value)."""
    if len(closes) < slow + sig:
        return np.zeros(max(len(closes), 1))
    macd_line   = _ema(closes, fast) - _ema(closes, slow)
    signal_line = _ema(macd_line, sig)
    return macd_line - signal_line


def _has_macd_bullish_cross(closes: np.ndarray) -> bool:
    """True if MACD histogram crossed from ≤0 to >0 within the last 3 bars."""
    hist = _macd_hist_series(closes)
    if len(hist) < 4:
        return False
    for i in range(-3, 0):          # i = -3, -2, -1
        if hist[i] > 0 and hist[i - 1] <= 0:
            return True
    return False


def _has_ema_bullish_cross(closes: np.ndarray) -> bool:
    """True if EMA9 crossed above EMA21 within the last 3 bars."""
    if len(closes) < 25:
        return False
    ema9  = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    for i in range(-3, 0):
        if ema9[i] > ema21[i] and ema9[i - 1] <= ema21[i - 1]:
            return True
    return False


def _at_fib_support(closes: np.ndarray, tolerance: float = 0.02) -> bool:
    """True if current price is within ±tolerance of a 60-bar Fibonacci level."""
    if len(closes) < 30:
        return False
    window = closes[-min(60, len(closes)):]
    high   = float(np.max(window))
    low    = float(np.min(window))
    if high <= low:
        return False
    price  = float(closes[-1])
    span   = high - low
    for fib in (0.236, 0.382, 0.5, 0.618, 0.786):
        level = high - fib * span          # retracement from high
        if abs(price - level) / (price + 1e-9) <= tolerance:
            return True
    return False


def _at_price_support(closes: np.ndarray, lows: np.ndarray,
                       tolerance: float = 0.03) -> bool:
    """True if current price is within ±tolerance of 20-bar support."""
    if len(lows) < 5 or float(closes[-1]) <= 0:
        return False
    support = float(np.min(lows[-min(20, len(lows)):]))
    return abs(float(closes[-1]) - support) / float(closes[-1]) <= tolerance


def _gap_pct(opens: np.ndarray, closes: np.ndarray) -> float:
    """Gap % = (last open − prior close) / prior close × 100."""
    if len(opens) < 1 or len(closes) < 2 or float(closes[-2]) <= 0:
        return 0.0
    return (float(opens[-1]) - float(closes[-2])) / float(closes[-2]) * 100


def _intraday_day_fraction() -> float:
    """
    Fraction of the regular trading session (9:30–16:00 ET) that has elapsed.
    Returns 1.0 outside market hours so complete bars are never rescaled.
    """
    now     = datetime.now(tz=ZoneInfo("America/New_York"))
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    if now < open_t or now >= close_t:
        return 1.0
    elapsed = (now - open_t).total_seconds()
    total   = (close_t - open_t).total_seconds()   # 23 400 s
    return max(0.05, elapsed / total)


def _rvol(volumes: np.ndarray) -> float:
    """
    Pace-adjusted relative volume.

    Baseline: mean of the previous 20 *full* trading days (volumes[-21:-1]),
    so today's partial bar is never mixed into the denominator.

    Today's volume is pace-projected to end-of-day before comparison,
    preventing artificially low RVOL readings at 9:45 AM.
    """
    if len(volumes) < 2:
        return 1.0
    hist = volumes[-21:-1] if len(volumes) >= 21 else volumes[:-1]
    avg  = float(np.mean(hist)) if len(hist) > 0 else 1.0
    if avg <= 0:
        return 1.0
    projected = float(volumes[-1]) / _intraday_day_fraction()
    return round(projected / avg, 2)


def _overextension_pct(closes: np.ndarray, period: int = 20) -> float:
    """
    % deviation of current price above EMA{period}.
    Positive = overextended above MA; negative = below.
    """
    if len(closes) < period:
        return 0.0
    ema_val = float(_ema(closes, period)[-1])
    if ema_val <= 0:
        return 0.0
    return round((float(closes[-1]) - ema_val) / ema_val * 100, 1)


def _spread_proxy(highs: np.ndarray, lows: np.ndarray,
                  closes: np.ndarray, n: int = 5) -> float:
    """
    Estimated bid-ask spread proxy: mean (High−Low)/Close over last n bars.
    Expressed as a percentage. Higher = wider spread / more illiquid.
    """
    n = min(n, len(highs), len(lows), len(closes))
    if n < 1:
        return 0.0
    ranges = []
    for i in range(1, n + 1):
        c = float(closes[-i])
        if c > 0:
            ranges.append((float(highs[-i]) - float(lows[-i])) / c * 100)
    return round(sum(ranges) / len(ranges), 2) if ranges else 0.0


# ── Hot-tickers priority layer ────────────────────────────────────────────────

def _build_hot_tickers() -> list:
    """
    Pull together the highest-conviction tickers from world_context:
      - Earnings catalysts reporting today or tomorrow (BULLISH bias)
      - Unusual bullish options flow (C/P ≥ 5×)
      - StockTwits / social trending tickers
      - Discovery agent candidates

    These are prepended to the universe so they are always scanned first,
    even if the bulk Polygon universe is limited or alphabetically sorted.
    """
    hot = []
    try:
        import world_context as wctx
        ctx = wctx.get()

        # Earnings catalysts today/tomorrow
        for e in ctx["earnings"].get("upcoming", []):
            if e.get("days", 99) <= 1 and e.get("direction") == "BULLISH":
                hot.append(e["ticker"])

        # Unusual bullish options flow
        for o in ctx["social"].get("unusual_opts", []):
            if o.get("bias") == "BULLISH" and o.get("call_put_ratio", 0) >= 5.0:
                hot.append(o["ticker"])

        # StockTwits trending
        for t in ctx["social"].get("trending", [])[:8]:
            hot.append(t["ticker"])

    except Exception:
        pass

    try:
        from discovery_agent import get_discovery_tickers
        hot += get_discovery_tickers()[:15]
    except Exception:
        pass

    # Deduplicate preserving order
    seen = set()
    result = []
    for t in hot:
        if t and t not in seen and t.isalpha() and 1 <= len(t) <= 5:
            seen.add(t)
            result.append(t.upper())

    if result:
        print(f"🔥 [Scanner] Hot-ticker priority layer ({len(result)}): {', '.join(result[:12])}"
              + ("..." if len(result) > 12 else ""))
    return result


# ── Gate helpers ───────────────────────────────────────────────────────────────

def _load_alerted_today() -> set:
    """Tickers already in signals_log.csv today (Gate 5 dedup)."""
    today   = datetime.now().strftime("%Y-%m-%d")
    alerted = set()
    if not os.path.exists(SIGNALS_LOG):
        return alerted
    try:
        with open(SIGNALS_LOG, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("timestamp", "").startswith(today):
                    alerted.add(row.get("ticker", "").upper())
    except Exception:
        pass
    return alerted


def _check_news(ticker: str, verbose: bool = False) -> dict:
    """
    Polygon news check for a single ticker.
    Returns: {has_recent: bool, hours_old: float, headline: str}
    Caller is responsible for rate-limit sleep between calls.
    Retries once on 429 after a 15-second pause.
    """
    try:
        r = requests.get(
            f"{BASE_URL}/v2/reference/news",
            params={"apiKey": POLYGON_API_KEY, "ticker": ticker,
                    "limit": 5, "order": "desc"},
            timeout=15,
        )
        if r.status_code == 429:
            if verbose:
                print(f"   ⏳ {ticker}: rate limited — retrying in 15s")
            time.sleep(15)
            r = requests.get(
                f"{BASE_URL}/v2/reference/news",
                params={"apiKey": POLYGON_API_KEY, "ticker": ticker,
                        "limit": 5, "order": "desc"},
                timeout=15,
            )
            if r.status_code == 429:
                if verbose:
                    print(f"   ⏳ {ticker}: still rate limited — skipping")
                return {"has_recent": False, "hours_old": 999, "headline": ""}
        articles = r.json().get("results", [])
        now      = datetime.now(timezone.utc)
        for article in articles:
            pub_str = article.get("published_utc", "")
            if not pub_str:
                continue
            pub_dt    = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            hours_old = (now - pub_dt).total_seconds() / 3600
            if hours_old <= NEWS_HOURS_WINDOW:
                headline = article.get("title", "")[:80]
                if verbose:
                    print(f"   📰 {ticker}: '{headline[:55]}' ({hours_old:.1f}h ago)")
                return {"has_recent": True, "hours_old": round(hours_old, 1),
                        "headline": headline}
        if verbose:
            print(f"   💤 {ticker}: no news in last {NEWS_HOURS_WINDOW}h")
        return {"has_recent": False, "hours_old": 999, "headline": ""}
    except Exception as e:
        if verbose:
            print(f"   ⚠️  {ticker}: news error ({e})")
        return {"has_recent": False, "hours_old": 999, "headline": ""}


def _get_ticker_info(ticker: str) -> dict:
    """
    Market cap + sector via yfinance. Returns defaults on failure.
    Uses fast_info for market cap (quick) and .info for sector (slower but
    only called for ~20 Gate-2 survivors, so cost is acceptable).
    """
    try:
        t  = yf.Ticker(ticker)
        mc = getattr(t.fast_info, "market_cap", None)
        try:
            sector = t.info.get("sector", "") or ""
        except Exception:
            sector = ""
        return {
            "market_cap": float(mc) if mc and float(mc) > 0 else 0.0,
            "sector":     sector,
        }
    except Exception:
        return {"market_cap": 0.0, "sector": ""}


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_survivor(data: dict, news: dict,
                    spy_closes: np.ndarray = None,
                    qqq_closes: np.ndarray = None,
                    sector_closes: np.ndarray = None) -> dict:
    """
    Layered scoring for a gate-passing stock.

    Layer 1 — Context  (market conditions, RS, liquidity, catalyst type)
    Layer 2 — Setup    (quality of the specific chart pattern)
    Layer 3 — Execution (technical confirmation signals)
    Layer 4 — Risk     (penalties for earnings risk, overbought, bad catalyst)

    total_score = context + setup + execution - risk
    """
    closes  = data["closes"]
    opens   = data["opens"]
    lows    = data["lows"]
    volumes = data["volumes"]
    price   = data["price"]

    rv        = _rvol(volumes)
    rsi       = data.get("_rsi_cached") or _calc_rsi(closes)
    gap       = _gap_pct(opens, closes)
    macd_cross = _has_macd_bullish_cross(closes)
    ema_cross  = _has_ema_bullish_cross(closes)

    # ══════════════════════════════════════════════════════════════════════════
    # Layer 1: Context score  (max ~80)
    # Relative strength, liquidity, news catalyst type, stock profile
    # ══════════════════════════════════════════════════════════════════════════
    ctx        = 0
    ctx_signals = []

    # Relative strength vs SPY + QQQ + sector ETF (multi-horizon, multi-benchmark)
    rs = compute_rs(closes,
                    spy_closes if spy_closes is not None else np.array([]),
                    qqq_closes if qqq_closes is not None else np.array([]),
                    sector_closes)
    if rs["rs_score"] != 0:
        ctx += rs["rs_score"]
        if rs["rs_label"]:
            ctx_signals.append(rs["rs_label"])

    # Dollar volume — proxy for liquidity and institutional interest
    avg_vol_5d   = float(np.mean(volumes[-5:])) if len(volumes) >= 5 else float(volumes[-1])
    dollar_vol   = price * avg_vol_5d
    dollar_vol_m = round(dollar_vol / 1_000_000, 3)
    if dollar_vol >= 5_000_000:
        ctx += 15; ctx_signals.append(f"${dollar_vol_m:.1f}M dvol")
    elif dollar_vol >= 1_000_000:
        ctx += 8;  ctx_signals.append(f"${dollar_vol_m:.1f}M dvol")
    elif dollar_vol >= 500_000:
        ctx += 3

    # News catalyst type — bullish catalysts boost context, bearish go to risk
    headline      = news.get("headline", "")
    news_category = classify_news(headline)
    cat_pts       = news_score(news_category)
    if cat_pts > 0:
        ctx += cat_pts
        ctx_signals.append(news_label(news_category))

    # Stock profile (asset characteristics belong in context, not execution)
    market_cap = data.get("market_cap", 0)
    if 0 < market_cap < 500_000_000:
        ctx += 10; ctx_signals.append("small cap")
    if price < 10:
        ctx += 8;  ctx_signals.append(f"${price:.2f}")

    # Sector tailwind / headwind — did the sector ETF move today?
    if sector_closes is not None and len(sector_closes) >= 2:
        sec_prev = float(sector_closes[-2])
        sec_last = float(sector_closes[-1])
        if sec_prev > 0:
            sec_1d = (sec_last / sec_prev - 1) * 100
            if sec_1d >= 2.0:
                ctx += 12; ctx_signals.append(f"sector +{sec_1d:.1f}% 🚀")
            elif sec_1d >= 1.0:
                ctx += 8;  ctx_signals.append(f"sector +{sec_1d:.1f}%")

    context_score = ctx

    # ══════════════════════════════════════════════════════════════════════════
    # Layer 2: Setup score  — score ALL qualifying setups, keep the best
    # ══════════════════════════════════════════════════════════════════════════
    setup_result  = detect_and_score(data, news, rsi, rv, gap)
    setup_type    = setup_result["setup_type"]
    setup_score   = setup_result["score"]
    setup_signals = setup_result["signals"]

    # ══════════════════════════════════════════════════════════════════════════
    # Layer 3: Execution score  — pure technical confirmation only
    # MACD cross, EMA cross, support — not stock profile signals
    # ══════════════════════════════════════════════════════════════════════════
    exc        = 0
    exc_signals = []

    if macd_cross:
        exc += 20; exc_signals.append("MACD ✅")
    if ema_cross:
        exc += 20; exc_signals.append("EMA9>21 ✅")
    if _at_fib_support(closes):
        exc += 15; exc_signals.append("fib support")
    if _at_price_support(closes, lows):
        exc += 15; exc_signals.append("price support")

    execution_score = exc

    # ══════════════════════════════════════════════════════════════════════════
    # Layer 4: Risk penalty  (subtracted from total)
    # Bad catalyst, overbought RSI, overextension
    # ══════════════════════════════════════════════════════════════════════════
    risk       = 0
    risk_signals = []

    # Bearish catalyst (offering, downgrade, earnings miss, etc.)
    if cat_pts < 0:
        risk += abs(cat_pts)
        risk_signals.append(news_label(news_category))

    # RSI overbought (outside our gate range but just in case)
    if rsi > 70:
        risk += 20; risk_signals.append(f"RSI {rsi:.0f} overbought ⚠️")

    # Sector headwind — sector ETF falling while stock is moving up
    if sector_closes is not None and len(sector_closes) >= 2:
        sec_prev = float(sector_closes[-2])
        sec_last = float(sector_closes[-1])
        if sec_prev > 0:
            sec_1d = (sec_last / sec_prev - 1) * 100
            if sec_1d <= -2.0:
                risk += 15; risk_signals.append(f"sector {sec_1d:.1f}% ⚠️")
            elif sec_1d <= -1.5:
                risk += 8;  risk_signals.append(f"sector {sec_1d:.1f}%")

    # Overextension — price too far above EMA20 (mean-reversion risk)
    overext = _overextension_pct(closes, 20)
    if overext > 30:
        risk += 20; risk_signals.append(f"EMA20 overext +{overext:.0f}% ⚠️")
    elif overext > 20:
        risk += 10; risk_signals.append(f"EMA20 overext +{overext:.0f}%")

    # Spread / execution cost proxy — wide daily range on low dollar volume
    spread = _spread_proxy(data["highs"], lows, closes)
    if spread > 5.0 and dollar_vol < 1_000_000:
        risk += 20; risk_signals.append(f"wide spread ~{spread:.1f}% ⚠️")
    elif spread > 3.5 and dollar_vol < 500_000:
        risk += 10; risk_signals.append(f"spread ~{spread:.1f}%")

    risk_penalty = risk

    # ── Pattern bonus ──────────────────────────────────────────────────────────
    pat_bonus = 0
    try:
        from pattern_detector import detect_patterns
        pat_hits  = detect_patterns(closes, data["highs"], lows, data["volumes"])
        for p in pat_hits:
            if p["pattern"] in ("bull_flag", "cup_handle"):
                pat_bonus += 15
            elif p["pattern"] in ("double_bottom", "ascending_triangle"):
                pat_bonus += 12
            elif p["pattern"] == "breakout" and setup_type != "breakout":
                pat_bonus += 8
    except Exception:
        pat_hits = []
    setup_score += pat_bonus

    # ── Total ─────────────────────────────────────────────────────────────────
    total_score = context_score + setup_score + execution_score - risk_penalty

    all_signals = ctx_signals + setup_signals + exc_signals
    if risk_signals:
        all_signals += [f"⚠️ {s}" for s in risk_signals]

    return {
        "score":          total_score,     # primary sort key
        "context_score":  context_score,
        "setup_score":    setup_score,
        "execution_score": execution_score,
        "risk_penalty":   risk_penalty,
        "setup_type":     setup_type,
        "rvol":           round(rv, 2),
        "rsi":            round(rsi, 1),
        "gap_pct":        round(gap, 2),
        "macd_cross":     macd_cross,
        "ema_cross":      ema_cross,
        "news_category":  news_category,
        "rs_vs_spy":      rs["rs_vs_spy"],    # raw 1-day SPY-only (for logs / display)
        "rs_composite":   rs["rs_composite"], # weighted multi-horizon composite (used for scoring)
        "dollar_vol_m":   dollar_vol_m,
        "signals":        ", ".join(all_signals),
    }


# ── Claude ranking ─────────────────────────────────────────────────────────────

def _claude_rank(candidates: list, verbose: bool = False) -> dict:
    """
    Send top 3 to Claude for final ranking + entry/target/stop for winner.
    Returns parsed dict, or a math-based fallback on API error.
    """
    if not candidates:
        return {}

    stock_blocks = []
    for i, c in enumerate(candidates, 1):
        mc_str   = f"${c.get('market_cap', 0) / 1e6:.0f}M" if c.get("market_cap", 0) > 0 else "unknown"
        macd_lbl = "bullish cross" if c.get("macd_cross") else "neutral"
        ema_lbl  = "EMA9 > EMA21" if c.get("ema_cross") else "neutral"
        cat_str   = news_label(c.get("news_category", "general"))
        rs_str    = f"{c.get('rs_vs_spy', 0.0):+.1f}% vs SPY"
        setup_str = c.get("setup_type", "general")
        stock_blocks.append(
            f"\nStock #{i}: {c['ticker']}\n"
            f"  Setup:      {setup_str}\n"
            f"  Price:      ${c['price']:.4f}\n"
            f"  Score:      {c['score']} pts "
            f"(ctx={c.get('context_score',0)} setup={c.get('setup_score',0)} "
            f"exec={c.get('execution_score',0)} risk=-{c.get('risk_penalty',0)})\n"
            f"  RVOL:       {c['rvol']:.1f}x\n"
            f"  RSI:        {c['rsi']:.1f}\n"
            f"  RS vs SPY:  {rs_str}\n"
            f"  MACD:       {macd_lbl}\n"
            f"  EMA:        {ema_lbl}\n"
            f"  Gap:        {c.get('gap_pct', 0):+.1f}%\n"
            f"  Catalyst:   {cat_str}\n"
            f"  Market cap: {mc_str}\n"
            f"  News:       {c.get('news_headline', '')[:80]}\n"
        )

    prompt = (
        "You are an expert day trader specializing in small/mid-cap momentum stocks.\n\n"
        "Pick the single BEST stock to buy today from these candidates.\n"
        "Consider RVOL, news recency, RSI setup, MACD/EMA signals, and price.\n\n"
        "Stocks to evaluate:\n"
        + "".join(stock_blocks)
        + "\n\nRespond ONLY with this exact JSON (no markdown fences, no extra text):\n"
        "{\n"
        '  "rank": ["TICKER1", "TICKER2", "TICKER3"],\n'
        '  "why": "<2 sentences on why #1 is the best setup today>",\n'
        '  "expected_move": "<e.g. +8-15% in 1-2 days>",\n'
        '  "key_risk": "<1 sentence>",\n'
        '  "entry_low": <float>,\n'
        '  "entry_high": <float>,\n'
        '  "target": <float>,\n'
        '  "stop_loss": <float>\n'
        "}"
    )

    if verbose:
        print("\n🤖 [Claude] Ranking top 3 candidates...")

    try:
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text  = parts[1][4:] if parts[1].startswith("json") else parts[1]
        result = json.loads(text.strip())
        if verbose:
            print(f"✅ [Claude] Ranked: {result.get('rank', [])}")
        return result
    except Exception as e:
        print(f"❌ [Claude] Ranking error: {e} — using math fallback")
        top = max(candidates, key=lambda x: x["score"])
        p   = top["price"]
        others = [c["ticker"] for c in candidates if c["ticker"] != top["ticker"]]
        return {
            "rank":          [top["ticker"]] + others,
            "why":           (f"Highest composite score with RVOL {top['rvol']:.1f}x "
                              f"and RSI {top['rsi']:.0f}. "
                              f"Recent news catalyst within {top.get('news_hours', 24):.0f}h."),
            "expected_move": "+5-10% in 1-2 days",
            "key_risk":      "Low liquidity — use limit orders and size accordingly.",
            "entry_low":     round(p * 0.99, 4),
            "entry_high":    round(p * 1.01, 4),
            "target":        round(p * 1.08, 4),
            "stop_loss":     round(p * 0.95, 4),
        }


# ── WhatsApp formatter ────────────────────────────────────────────────────────

def _format_whatsapp(winner: dict, rank2: dict, rank3: dict,
                     claude: dict, n_scanned: int, n_passed: int) -> str:
    """Build the Best Stock Today WhatsApp message."""
    date_str = datetime.now().strftime("%B %d, %Y")
    time_str = datetime.now().strftime("%I:%M %p")

    p1         = winner["price"]
    entry_low  = claude.get("entry_low",  round(p1 * 0.99, 4))
    entry_high = claude.get("entry_high", round(p1 * 1.01, 4))
    target     = claude.get("target",     round(p1 * 1.08, 4))
    stop       = claude.get("stop_loss",  round(p1 * 0.95, 4))
    tgt_pct    = (target - p1) / p1 * 100 if p1 else 0
    stop_pct   = (stop   - p1) / p1 * 100 if p1 else 0

    headline  = winner.get("news_headline", "")
    if len(headline) > 60:
        headline = headline[:57] + "..."
    macd_str  = "bullish cross" if winner.get("macd_cross") else "neutral"

    setup_str = winner.get("setup_type", "general").replace("_", " ").title()
    ctx_s  = winner.get("context_score", 0)
    stp_s  = winner.get("setup_score", 0)
    exc_s  = winner.get("execution_score", 0)
    rsk_s  = winner.get("risk_penalty", 0)

    also = []
    if rank2:
        r2_setup = rank2.get("setup_type", "general").replace("_", " ").title()
        also.append(f"#2 {rank2['ticker']} ${rank2['price']:.2f} "
                    f"[{r2_setup}] score {rank2['score']}")
    if rank3:
        r3_setup = rank3.get("setup_type", "general").replace("_", " ").title()
        also.append(f"#3 {rank3['ticker']} ${rank3['price']:.2f} "
                    f"[{r3_setup}] score {rank3['score']}")

    msg = (
        f"BEST STOCK TODAY — {date_str}\n"
        f"─────────────────────\n"
        f"#1 {winner['ticker']} ${p1:.2f} [{setup_str}]\n"
        f"Score: {winner['score']} "
        f"(ctx={ctx_s} setup={stp_s} exec={exc_s} risk=-{rsk_s})\n"
        f"Why: {claude.get('why', '')}\n"
        f"Expected: {claude.get('expected_move', '')}\n"
        f"Risk: {claude.get('key_risk', '')}\n"
        f"\n"
        f"Entry:  ${entry_low:.2f} – ${entry_high:.2f}\n"
        f"Target: ${target:.2f} (+{tgt_pct:.1f}%)\n"
        f"Stop:   ${stop:.2f} ({stop_pct:.1f}%)\n"
        f"\n"
        f"RVOL: {winner['rvol']:.1f}x | RSI: {winner['rsi']:.0f} | MACD: {macd_str}\n"
        f"News: {headline}\n"
        f"─────────────────────\n"
        f"Also scored:\n"
    )
    for line in also:
        msg += f"{line}\n"
    msg += (
        f"─────────────────────\n"
        f"Scanned: {n_scanned:,} stocks\n"
        f"Passed filters: {n_passed}\n"
        f"Time: {time_str}"
    )
    return msg


# ── Accuracy log ───────────────────────────────────────────────────────────────

def _ensure_picks_header():
    if not os.path.exists(BEST_PICKS_LOG) or os.path.getsize(BEST_PICKS_LOG) == 0:
        with open(BEST_PICKS_LOG, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(BEST_PICKS_COLUMNS)


def log_best_pick(winner: dict):
    """Append today's best pick to best_picks_log.csv."""
    try:
        _ensure_picks_header()
        row = [
            datetime.now().strftime("%Y-%m-%d"),
            winner["ticker"],
            winner.get("setup_type", "general"),
            winner["score"],
            winner.get("context_score", 0),
            winner.get("setup_score", 0),
            winner.get("execution_score", 0),
            winner.get("risk_penalty", 0),
            f"{winner['price']:.6f}",
            winner.get("news_headline", "")[:120],
            winner.get("news_category", "general"),
            winner["rvol"],
            winner["rsi"],
            1 if winner.get("macd_cross") else 0,
            1 if winner.get("ema_cross") else 0,
            winner.get("gap_pct", 0.0),
            winner.get("rs_vs_spy", 0.0),
            winner.get("dollar_vol_m", 0.0),
            "",    # price_1day_later  — filled next day
            "",    # price_3day_later  — filled in 3 days
            "",    # actual_gain_loss_pct
        ]
        with open(BEST_PICKS_LOG, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        print(f"📝 [Scanner] Best pick logged → {BEST_PICKS_LOG}")
    except Exception as e:
        print(f"⚠️  [Scanner] Failed to log pick: {e}")


def _historical_close_on(ticker: str, target_date) -> float:
    """
    Return the actual closing price on the first trading day on or after
    target_date, using historical bars — not fast_info (which returns today).
    target_date: datetime.date
    """
    from datetime import timedelta
    try:
        start = (target_date - timedelta(days=1)).isoformat()
        end   = (target_date + timedelta(days=7)).isoformat()  # covers weekends/holidays
        df    = yf.download(ticker, start=start, end=end,
                            interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return 0.0
        for idx in df.index:
            idx_date = idx.date() if hasattr(idx, "date") else idx
            if idx_date >= target_date:
                val = df.loc[idx, "Close"]
                return float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
        return 0.0
    except Exception:
        return 0.0


def update_pick_accuracy():
    """
    Back-fill price_1day_later / price_3day_later for past picks using
    the actual historical close on the correct forward date, not today's price.

    price_1day_later  → close on pick_date + 1 calendar day (first trading day on/after)
    price_3day_later  → close on pick_date + 3 calendar days (first trading day on/after)
    actual_gain_loss_pct → pct change from price_at_pick to price_3day_later
    """
    from datetime import timedelta
    if not os.path.exists(BEST_PICKS_LOG):
        return
    try:
        with open(BEST_PICKS_LOG, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return

        today   = datetime.now().date()
        updated = False

        for row in rows:
            try:
                pick_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except ValueError:
                continue

            days_ago = (today - pick_date).days
            price_at = float(row.get("price_at_pick") or 0)
            if price_at <= 0:
                continue

            # 1-day forward close: need at least 2 calendar days so the target
            # date has actually closed (pick_date+1 may still be today)
            if days_ago >= 2 and not row.get("price_1day_later"):
                target_1d = pick_date + timedelta(days=1)
                p = _historical_close_on(row["ticker"], target_1d)
                if p > 0:
                    row["price_1day_later"] = f"{p:.6f}"
                    updated = True

            # 3-day forward close: need at least 4 calendar days
            if days_ago >= 4 and not row.get("price_3day_later"):
                target_3d = pick_date + timedelta(days=3)
                p = _historical_close_on(row["ticker"], target_3d)
                if p > 0:
                    row["price_3day_later"]     = f"{p:.6f}"
                    pct = (p - price_at) / price_at * 100
                    row["actual_gain_loss_pct"] = f"{pct:+.2f}%"
                    updated = True

        if updated:
            with open(BEST_PICKS_LOG, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=BEST_PICKS_COLUMNS, restval="")
                w.writeheader()
                w.writerows(rows)
            print("📊 [Scanner] Accuracy log updated with real forward closes")

    except Exception as e:
        print(f"⚠️  [Scanner] Accuracy update failed: {e}")


# ── Main Best-of-Day pipeline ──────────────────────────────────────────────────

def scan_best_of_day(verbose: bool = False,
                      max_universe: int = 6000,
                      extra_tickers: list = None,
                      min_score: int = None,
                      rvol_bypass: list = None) -> dict:
    """
    Full Best-of-Day selection pipeline.

    Returns the winner dict (empty dict if no qualifying stock found).
    WhatsApp message is in winner['whatsapp_msg'].

    min_score: override MIN_SCORE constant (useful for pre-market sweep
               where daily RVOL is not yet populated).
    """
    print("\n" + "═" * 55)
    print("🏆 [Best-of-Day] Starting selection pipeline...")
    print("═" * 55)

    # Back-fill previous picks' prices
    update_pick_accuracy()

    # ── Load universe ─────────────────────────────────────────────────────────
    # Priority order: extra_tickers → hot tickers from world_context → bulk universe
    hot_tickers = _build_hot_tickers()
    universe    = _load_universe(max_universe)
    priority    = list(dict.fromkeys((extra_tickers or []) + hot_tickers))
    universe    = list(dict.fromkeys(priority + universe))
    universe    = [t for t in universe if t.isalpha() and 1 <= len(t) <= 5]
    n_universe  = len(universe)

    # ── Pre-fetch SPY + QQQ + all sector ETFs for multi-benchmark RS ─────────
    print("📈 [Best-of-Day] Fetching SPY + QQQ + sector ETFs...")
    benchmarks = fetch_benchmarks(include_sectors=True)
    spy_closes = benchmarks["spy"]
    qqq_closes = benchmarks["qqq"]
    loaded = [k.upper() for k, v in benchmarks.items() if len(v) > 0]
    print(f"✅ [Best-of-Day] Benchmarks loaded: {', '.join(loaded) or 'none — RS will be 0'}")

    # ── Circuit breaker — skip BUY picks on crash days ────────────────────────
    try:
        from circuit_breaker import check_market
        spy_chg = (
            float((spy_closes[-1] / spy_closes[-2] - 1) * 100)
            if len(spy_closes) >= 2 and float(spy_closes[-2]) > 0 else None
        )
        cb = check_market(spy_day_chg=spy_chg)
        if not cb["safe"]:
            print(f"\n🚫 [Best-of-Day] Circuit breaker: {cb['reason']}")
            print("   No pick today — wait for safer market conditions.")
            return {}
        print(f"✅ [Best-of-Day] Circuit breaker OK  "
              f"(VIX {cb['vix']:.1f}, SPY {cb['spy_chg']:+.1f}%)")
    except Exception as _cb_err:
        print(f"⚠️  [Best-of-Day] Circuit breaker check failed (fail-open): {_cb_err}")

    # ── Bulk OHLCV download ────────────────────────────────────────────────────
    raw_data     = _bulk_download_batched(universe)
    n_downloaded = len(raw_data)

    # ── Gate 0: Dollar volume >= $100k (5-day avg — liquidity floor) ──────────
    g0 = {}
    for t, d in raw_data.items():
        avg_vol = float(np.mean(d["volumes"][-5:])) if len(d["volumes"]) >= 5 else float(d["volumes"][-1])
        if d["price"] * avg_vol >= DOLLAR_VOL_MIN:
            g0[t] = d
    print(f"🚦 Gate 0 (dvol ≥${DOLLAR_VOL_MIN//1000:.0f}k): {n_downloaded:,} → {len(g0):,} stocks")

    # ── Gate 1: RVOL >= 2.0 (bypassed for pre-market priority tickers) ─────────
    _bypass = {t.upper() for t in (rvol_bypass or [])}
    g1 = {t: d for t, d in g0.items()
          if _rvol(d["volumes"]) >= RVOL_MIN or t.upper() in _bypass}
    bypassed = sum(1 for t in g1 if t.upper() in _bypass and _rvol(g0[t]["volumes"]) < RVOL_MIN)
    print(f"🚦 Gate 1 (RVOL ≥{RVOL_MIN:.1f}):  {len(g0):,} → {len(g1):,} stocks"
          + (f"  (+{bypassed} premarket bypass)" if bypassed else ""))

    # ── Gate 3: RSI 28–67 ─────────────────────────────────────────────────────
    g3 = {}
    for t, d in g1.items():
        rsi = _calc_rsi(d["closes"])
        if RSI_MIN <= rsi <= RSI_MAX:
            d["_rsi_cached"] = rsi
            g3[t] = d
    print(f"🚦 Gate 3 (RSI {RSI_MIN:.0f}–{RSI_MAX:.0f}):    {len(g1):,} → {len(g3):,} stocks")

    # ── Gate 4: Price $0.50–$50 ───────────────────────────────────────────────
    g4 = {t: d for t, d in g3.items()
          if PRICE_MIN <= d["price"] <= PRICE_MAX}
    print(f"🚦 Gate 4 (${PRICE_MIN}–${PRICE_MAX:.0f}):      {len(g3):,} → {len(g4):,} stocks")

    # ── Gate 5: Not already alerted today ─────────────────────────────────────
    alerted = _load_alerted_today()
    g5      = {t: d for t, d in g4.items() if t.upper() not in alerted}
    deduped = len(g4) - len(g5)
    print(f"🚦 Gate 5 (dedup):        {len(g4):,} → {len(g5):,} stocks"
          + (f"  (-{deduped} already alerted)" if deduped else ""))

    if not g5:
        print("\n⚠️  [Best-of-Day] No survivors after gate filtering — no pick today.")
        return {}

    # ── Gate 2: News in last 24h ──────────────────────────────────────────────
    # Sort by RVOL desc, cap at MAX_NEWS_CHECK to stay within API budget
    sorted_g5     = sorted(g5.items(), key=lambda x: _rvol(x[1]["volumes"]), reverse=True)
    news_pool     = sorted_g5[:MAX_NEWS_CHECK]
    est_minutes   = len(news_pool) * 13 // 60
    est_seconds   = len(news_pool) * 13 % 60

    print(f"\n📰 Gate 2 (news ≤24h): checking top {len(news_pool)} survivors "
          f"(~{est_minutes}m {est_seconds}s at 5 req/min)...")

    g2: list = []    # [(ticker, data, news)]
    for i, (ticker, data) in enumerate(news_pool):
        if verbose:
            print(f"   [{i+1:2d}/{len(news_pool)}] {ticker:6s}", end="  ", flush=True)
        if i > 0:
            time.sleep(13)   # Polygon free tier: 5 req/min
        news = _check_news(ticker, verbose=verbose)
        if news["has_recent"]:
            g2.append((ticker, data, news))

    print(f"🚦 Gate 2 (news ≤24h):  {len(news_pool):,} → {len(g2):,} stocks")

    if not g2:
        print("\n⚠️  [Best-of-Day] No stocks with recent news — no pick today.")
        return {}

    # ── Fetch market caps + sector for survivors ───────────────────────────────
    print(f"\n📊 Scoring {len(g2)} survivors...")
    for ticker, data, _ in g2:
        info = _get_ticker_info(ticker)
        data["market_cap"] = info["market_cap"]
        data["sector"]     = info["sector"]

    # ── Score survivors ────────────────────────────────────────────────────────
    scored = []
    for ticker, data, news in g2:
        sector_etf    = SECTOR_ETFS.get(data.get("sector", ""), "")
        sector_arr    = benchmarks.get(sector_etf.lower(), np.array([]))
        sector_closes = sector_arr if len(sector_arr) > 0 else None
        sc = _score_survivor(data, news, spy_closes=spy_closes, qqq_closes=qqq_closes,
                             sector_closes=sector_closes)
        scored.append({
            "ticker":         ticker,
            "price":          data["price"],
            "score":          sc["score"],
            "context_score":  sc["context_score"],
            "setup_score":    sc["setup_score"],
            "execution_score": sc["execution_score"],
            "risk_penalty":   sc["risk_penalty"],
            "setup_type":     sc["setup_type"],
            "rvol":           sc["rvol"],
            "rsi":            sc["rsi"],
            "gap_pct":        sc["gap_pct"],
            "market_cap":     data.get("market_cap", 0),
            "macd_cross":     sc["macd_cross"],
            "ema_cross":      sc["ema_cross"],
            "news_headline":  news.get("headline", ""),
            "news_hours":     news.get("hours_old", 999),
            "news_category":  sc["news_category"],
            "rs_vs_spy":      sc["rs_vs_spy"],
            "dollar_vol_m":   sc["dollar_vol_m"],
            "signals":        sc["signals"],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    if verbose:
        print("\n📋 All scored survivors:")
        for s in scored[:15]:
            print(f"   {s['ticker']:6s}  {s['score']:3d}pts  "
                  f"[{s.get('setup_type','general'):15s}]  "
                  f"ctx={s.get('context_score',0):3d} "
                  f"setup={s.get('setup_score',0):3d} "
                  f"exec={s.get('execution_score',0):3d} "
                  f"risk=-{s.get('risk_penalty',0):2d}  "
                  f"RS {s.get('rs_vs_spy', 0):+.1f}%  "
                  f"| {s['signals'][:45]}")

    # ── Filter by minimum score ────────────────────────────────────────────────
    score_threshold = min_score if min_score is not None else MIN_SCORE
    qualifiers = [s for s in scored if s["score"] >= score_threshold]
    print(f"\n🎯 Qualifying (score ≥{score_threshold}): {len(qualifiers)} stocks")
    if qualifiers:
        print(f"   Top catalyst: {qualifiers[0].get('news_category','general')}  "
              f"RS {qualifiers[0].get('rs_vs_spy', 0):+.1f}% vs SPY")

    if not qualifiers:
        print(f"\n⚠️  [Best-of-Day] No stocks scored ≥{score_threshold} — no pick today.")
        from alerts import send_whatsapp as _sw
        _sw(
            f"📊 Best-of-Day scan complete (7:45 AM)\n"
            f"No qualifying stock today — nothing scored ≥{score_threshold} pts.\n"
            f"Scanned {n_universe} tickers. Stay patient. 💤"
        )
        return {}

    top3 = qualifiers[:3]
    print("\nTop 3 finalists:")
    for s in top3:
        print(f"   {'★' if s is top3[0] else ' '} {s['ticker']:6s}  "
              f"{s['score']:3d}pts  [{s.get('setup_type','general')}]  "
              f"RVOL {s['rvol']:.1f}x  RSI {s['rsi']:.0f}  "
              f"RS {s.get('rs_vs_spy',0):+.1f}%  cat={s.get('news_category','general')}")

    # ── Full agentic pipeline on top 3 (aggregator + validator + Claude) ───────
    from concurrent.futures import ThreadPoolExecutor
    from graph import GRAPH, make_initial_state
    from alerts import send_alert as _send_alert

    print(f"\n🤖 [Best-of-Day] Running top {len(top3)} through full agentic pipeline...")

    def _pipe(candidate: dict) -> dict:
        ticker = candidate["ticker"]
        state  = make_initial_state(ticker)
        state["news_triggered"] = True   # morning picks qualify for lower threshold
        try:
            result = GRAPH.invoke(state)
        except Exception as _e:
            print(f"   ❌ {ticker} pipeline error: {_e}")
            result = {"signal": "HOLD", "confidence": 0, "ticker": ticker}
        result["_candidate"] = candidate
        return result

    with ThreadPoolExecutor(max_workers=3) as _exe:
        pipe_results = list(_exe.map(_pipe, top3))

    # Winner = BUY that passed all validator rules, highest confidence
    buy_results = [
        r for r in pipe_results
        if r.get("final_signal", r.get("signal")) == "BUY"
        and r.get("validator_passed", False)
    ]
    buy_results.sort(key=lambda r: r.get("confidence", 0), reverse=True)

    if buy_results:
        best   = buy_results[0]
        cand   = best["_candidate"]
        ticker = cand["ticker"]
        price  = cand["price"]
        conf   = best.get("confidence", 0)
        agr    = best.get("agreement_score", 0.0)

        print(f"\n🥇 WINNER: {ticker}  conf={conf}/100  agreement={agr:.0f}%")
        print(f"   Signals: {best.get('signal_count_bull', 0)} bull / {best.get('signal_count_bear', 0)} bear")
        print(f"   Entry:   ${best.get('entry_low', 0):.2f} – ${best.get('entry_high', 0):.2f}")
        print(f"   Stop:    ${best.get('stop_loss', 0):.2f} ({best.get('stop_pct', 0):.1f}%)")
        if best.get("reasoning"):
            print(f"   {best['reasoning'][:110]}")

        winner = dict(cand)
        winner.update({
            "entry_low":         best.get("entry_low",  0.0),
            "entry_high":        best.get("entry_high", 0.0),
            "stop_loss":         best.get("stop_loss",  0.0),
            "stop_pct":          best.get("stop_pct",   0.0),
            "targets":           best.get("targets",    []),
            "reasoning":         best.get("reasoning",  ""),
            "trade_horizon":     best.get("trade_horizon",     "swing"),
            "horizon_reasoning": best.get("horizon_reasoning", ""),
            "confidence":        conf,
            "agreement_score":   agr,
            "signal_count_bull": best.get("signal_count_bull", 0),
            "signal_count_bear": best.get("signal_count_bear", 0),
            "top_3_signals":     best.get("top_3_signals",   []),
            "bullish_signals":   best.get("bullish_signals",  []),
            "bearish_signals":   best.get("bearish_signals",  []),
            "consensus":         best.get("consensus",        ""),
            "main_risk":         best.get("main_risk",        ""),
        })

        _regime_str  = str((best.get("market_regime") or {}).get("regime", ""))
        _sector_mom  = best.get("sector_momentum") or {}
        _sector_str  = (
            f"{_sector_mom.get('change_pct', 0):+.1f}% {_sector_mom.get('signal', '')}"
            if _sector_mom else ""
        )
        _catalyst    = (best.get("news_summary") or "")[:100] or cand.get("news_headline", "")

        winner["whatsapp_msg"] = (
            f"BUY {ticker} @ ${price:.2f}  "
            f"conf={conf}  agreement={agr:.0f}%  "
            f"entry ${winner['entry_low']:.2f}–${winner['entry_high']:.2f}"
        )

        _send_alert(
                ticker=ticker,
                signal="BUY",
                price=price,
                entry_low=winner["entry_low"],
                entry_high=winner["entry_high"],
                targets=winner["targets"],
                stop=winner["stop_loss"],
                reason=winner["reasoning"][:240],
                confidence=conf,
                horizon=winner["trade_horizon"],
                horizon_reason=winner["horizon_reasoning"],
                agreement_score=agr,
                signal_count_bull=winner["signal_count_bull"],
                signal_count_bear=winner["signal_count_bear"],
                top_3_signals=winner["top_3_signals"],
                bullish_signals=winner["bullish_signals"],
                bearish_signals=winner["bearish_signals"],
                consensus=winner["consensus"],
                market_regime_str=_regime_str,
                sector_str=_sector_str,
                catalyst_str=_catalyst,
                main_risk=winner["main_risk"],
                det_score=winner.get("score", 0),
            )
        print("✅ [Best-of-Day] Alert sent via full agentic pipeline.")

    else:
        # No BUY survived the full pipeline — fall back to classic Claude rank
        print("\n⚠️  [Best-of-Day] No BUY signals survived validator — using Claude rank fallback")
        claude = _claude_rank(top3, verbose=verbose)
        rank   = claude.get("rank", [s["ticker"] for s in top3])

        rank_map    = {t: i for i, t in enumerate(rank)}
        top3_sorted = sorted(top3, key=lambda x: rank_map.get(x["ticker"], 99))
        winner      = top3_sorted[0]
        rank2       = top3_sorted[1] if len(top3_sorted) > 1 else {}
        rank3       = top3_sorted[2] if len(top3_sorted) > 2 else {}

        print(f"\n🥇 WINNER (fallback): {winner['ticker']}  "
              f"score={winner['score']}/225  ${winner['price']:.4f}")
        print(f"   Why: {claude.get('why', '')[:110]}")
        print(f"   Expected: {claude.get('expected_move', '')}")
        print(f"   Risk: {claude.get('key_risk', '')}")

        msg = _format_whatsapp(winner, rank2, rank3, claude, n_universe, len(g2))
        winner["whatsapp_msg"] = msg

        print("\n" + "─" * 55)
        print("📱 WhatsApp message:")
        print("─" * 55)
        print(msg)
        print("─" * 55)

        from alerts import send_whatsapp
        send_whatsapp(msg)
        print("✅ [Best-of-Day] WhatsApp sent (fallback format).")

    # ── Log to best_picks_log.csv ──────────────────────────────────────────────
    log_best_pick(winner)

    return winner


# ── Legacy broad sweep (kept for backward compatibility) ──────────────────────

def _score_ticker_legacy(closes: np.ndarray, volumes: np.ndarray) -> dict:
    """Original broad-sweep scorer. Pure-math, no news, no Claude."""
    from agents.tech_agent import _calc_macd, _calc_bollinger
    score    = 0
    reasons  = []
    rsi      = _calc_rsi(closes)
    macd_d   = _calc_macd(closes)
    bollinger = _calc_bollinger(closes)
    price    = float(closes[-1])
    hist     = macd_d["histogram"]
    avg_vol  = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 1.0
    cur_vol  = float(volumes[-1])
    vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0

    if rsi < 30:
        score += 30; reasons.append(f"RSI {rsi:.0f} oversold")
    elif rsi < 40:
        score += 20; reasons.append(f"RSI {rsi:.0f}")
    elif rsi < 50:
        score += 10
    elif rsi > 70:
        score -= 10
    if hist > 0:
        score += 20; reasons.append("MACD bullish")
    elif hist > -0.001:
        score += 5
    if price <= bollinger["lower"]:
        score += 20; reasons.append("BB oversold")
    elif price <= bollinger["middle"]:
        score += 10
    if vol_ratio >= 3.0:
        score += 20; reasons.append(f"vol {vol_ratio:.1f}x")
    elif vol_ratio >= 2.0:
        score += 15; reasons.append(f"vol {vol_ratio:.1f}x")
    elif vol_ratio >= 1.5:
        score += 8
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


def scan_broad_market(extra_tickers: list = None, top_n: int = 5,
                       max_universe: int = 6000) -> list:
    """
    Legacy broad sweep — no news, no Claude. Used by scheduler morning digest.
    Returns top_n stocks by pure-math score.
    """
    universe = _load_universe(max_universe)
    if extra_tickers:
        universe = list(dict.fromkeys(extra_tickers + universe))
    universe = [t for t in universe if t.isalpha() and 1 <= len(t) <= 5]

    data = _bulk_download_batched(universe)

    results = []
    for ticker, d in data.items():
        try:
            scored           = _score_ticker_legacy(d["closes"], d["volumes"])
            scored["ticker"] = ticker
            results.append(scored)
        except Exception:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:top_n]
    print(f"🏆 [Scanner] Top {top_n}: {[(r['ticker'], r['score']) for r in top]}")
    return top


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Scanner v2")
    parser.add_argument("--best-of-day", action="store_true",
                        help="Run Best-of-Day selection algorithm (gates + Claude)")
    parser.add_argument("--test",        action="store_true",
                        help="Test mode: full run, no WhatsApp sent")
    parser.add_argument("--verbose",     action="store_true",
                        help="Show gate details, news checks, and scoring breakdown")
    parser.add_argument("--top",         type=int, default=10,
                        help="Top N for broad sweep (default 10)")
    parser.add_argument("--universe",    type=int, default=6000,
                        help="Max tickers to scan (default 6000)")
    args = parser.parse_args()

    import watchlist_manager as wl
    watchlist = wl.load()

    if args.best_of_day:
        result = scan_best_of_day(
            verbose      = args.verbose,
            max_universe = args.universe,
            extra_tickers = watchlist,
        )
        if args.test:
            print("\n✅ [TEST MODE] Run complete — no WhatsApp was sent.")
            if not result:
                print("   No qualifying stock found today.")
            else:
                print(f"   Best pick: {result['ticker']}  "
                      f"score={result['score']}/225  ${result['price']:.4f}")
    else:
        print(f"Starting broad market scan (up to {args.universe:,} stocks)...\n")
        results = scan_broad_market(extra_tickers=watchlist, top_n=args.top,
                                     max_universe=args.universe)
        print("\nTop opportunities:")
        for i, r in enumerate(results, 1):
            print(
                f"  {i:2d}. {r['ticker']:6s}  score={r['score']:3d}"
                f"  ${r['price']:.2f}  RSI {r['rsi']:.0f}"
                f"  vol {r['vol_ratio']:.1f}x  |  {r['reason']}"
            )
