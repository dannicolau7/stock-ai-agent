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

import anthropic
import numpy as np
import requests
import yfinance as yf

from agents.tech_agent import _calc_rsi, _ema
from config import POLYGON_API_KEY, ANTHROPIC_API_KEY

# ── Constants ──────────────────────────────────────────────────────────────────

CACHE_FILE     = "ticker_universe.json"
CACHE_HOURS    = 24
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

BEST_PICKS_COLUMNS = [
    "date", "ticker", "score", "price_at_pick", "news_headline",
    "rvol", "rsi", "price_1day_later", "price_3day_later", "actual_gain_loss_pct",
]

FALLBACK_UNIVERSE = [
    "BBAI", "SOUN", "BZAI", "AWRE", "LTRX", "CXAI", "GFAI", "AITX", "POET",
    "IONQ", "RGTI", "QUBT", "KULR", "CIFR", "BTBT", "PRCT", "NVAX", "IOVA",
    "NVDA", "AMD", "SMCI", "MSTR", "PLTR", "RKLB", "JOBY", "ACHR", "LUNR",
    "FCEL", "PLUG", "BLNK", "CHPT", "AMC", "GME",
]


# ── Ticker universe ────────────────────────────────────────────────────────────

def _fetch_polygon_tickers(max_tickers: int = 6000) -> list:
    """Fetch active US stock tickers from Polygon. Paginates automatically."""
    tickers = []
    url     = f"{BASE_URL}/v3/reference/tickers"
    params  = {
        "apiKey": POLYGON_API_KEY,
        "active": "true",
        "market": "stocks",
        "locale": "us",
        "limit":  1000,
        "order":  "asc",
        "sort":   "ticker",
    }
    print("📋 [Scanner] Fetching ticker universe from Polygon...")
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
        params = {"apiKey": POLYGON_API_KEY, "cursor": cursor}
        time.sleep(13)   # 5 req/min on free tier
    print(f"✅ [Scanner] Got {len(tickers)} tickers from Polygon")
    return tickers[:max_tickers]


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
    if tickers:
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump({"ts": datetime.now().timestamp(), "tickers": tickers}, f)
        except Exception:
            pass
        return tickers
    print(f"⚠️  [Scanner] Using fallback universe ({len(FALLBACK_UNIVERSE)} tickers)")
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


def _rvol(volumes: np.ndarray) -> float:
    """Relative volume: last bar / 20-bar average."""
    avg = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    return float(volumes[-1]) / avg if avg > 0 else 1.0


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
                print(f"   ⏳ {ticker}: rate limited")
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


def _get_market_cap(ticker: str) -> float:
    """Market cap via yfinance fast_info. Returns 0 on failure."""
    try:
        mc = getattr(yf.Ticker(ticker).fast_info, "market_cap", None)
        return float(mc) if mc and float(mc) > 0 else 0.0
    except Exception:
        return 0.0


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_survivor(data: dict, news: dict) -> dict:
    """
    Score a gate-passing stock. Maximum 225 pts.

    RVOL scoring      (max  60)
    News recency      (max  35)
    RSI position      (max  25)
    MACD bullish cross       20
    EMA9 > EMA21             20
    Gap up                   15
    Fibonacci support        15
    Price support            15
    Market cap < $500M       10
    Price < $10              10
    ─────────────────────────────
    Max total               225
    """
    closes  = data["closes"]
    opens   = data["opens"]
    lows    = data["lows"]
    volumes = data["volumes"]
    price   = data["price"]

    score   = 0
    signals = []

    # ── RVOL (max 60) ─────────────────────────────────────────────────────────
    rv = _rvol(volumes)
    if rv > 10:
        score += 60; signals.append(f"RVOL {rv:.1f}x 🔥🔥")
    elif rv >= 5:
        score += 50; signals.append(f"RVOL {rv:.1f}x 🔥")
    elif rv >= 3:
        score += 35; signals.append(f"RVOL {rv:.1f}x")
    elif rv >= 2:
        score += 20; signals.append(f"RVOL {rv:.1f}x")

    # ── News recency (max 35) ─────────────────────────────────────────────────
    hours_old = news.get("hours_old", 999)
    if hours_old < 1:
        score += 35; signals.append("news <1h ⚡")
    elif hours_old < 4:
        score += 25; signals.append(f"news {hours_old:.0f}h ago")
    elif hours_old < 12:
        score += 15; signals.append(f"news {hours_old:.0f}h ago")
    elif hours_old <= 24:
        score += 5

    # ── RSI (max 25) — already filtered 28–67 ────────────────────────────────
    rsi = _calc_rsi(closes)
    if 30 <= rsi <= 40:
        score += 25; signals.append(f"RSI {rsi:.0f} bounce")
    elif 40 < rsi <= 50:
        score += 15; signals.append(f"RSI {rsi:.0f}")
    elif 50 < rsi <= 65:
        score += 5

    # ── MACD bullish cross (last 3 bars) → +20 ────────────────────────────────
    macd_cross = _has_macd_bullish_cross(closes)
    if macd_cross:
        score += 20; signals.append("MACD ✅")

    # ── EMA9 crossed above EMA21 → +20 ────────────────────────────────────────
    ema_cross = _has_ema_bullish_cross(closes)
    if ema_cross:
        score += 20; signals.append("EMA9>21 ✅")

    # ── Gap up with volume confirmation → +15 ─────────────────────────────────
    gap = _gap_pct(opens, closes)
    if gap >= 2.0:
        score += 15; signals.append(f"gap +{gap:.1f}%")

    # ── Fibonacci support → +15 ───────────────────────────────────────────────
    fib_sup = _at_fib_support(closes)
    if fib_sup:
        score += 15; signals.append("fib support")

    # ── Price support → +15 ───────────────────────────────────────────────────
    price_sup = _at_price_support(closes, lows)
    if price_sup:
        score += 15; signals.append("price support")

    # ── Market cap < $500M → +10 ──────────────────────────────────────────────
    market_cap = data.get("market_cap", 0)
    if 0 < market_cap < 500_000_000:
        score += 10; signals.append("small cap")

    # ── Price < $10 → +10 ─────────────────────────────────────────────────────
    if price < 10:
        score += 10; signals.append(f"${price:.2f}")

    return {
        "score":      score,
        "rvol":       round(rv, 2),
        "rsi":        round(rsi, 1),
        "gap_pct":    round(gap, 2),
        "macd_cross": macd_cross,
        "ema_cross":  ema_cross,
        "signals":    ", ".join(signals),
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
        stock_blocks.append(
            f"\nStock #{i}: {c['ticker']}\n"
            f"  Price:      ${c['price']:.4f}\n"
            f"  Score:      {c['score']}/225\n"
            f"  RVOL:       {c['rvol']:.1f}x\n"
            f"  RSI:        {c['rsi']:.1f}\n"
            f"  MACD:       {macd_lbl}\n"
            f"  EMA:        {ema_lbl}\n"
            f"  Gap:        {c.get('gap_pct', 0):+.1f}%\n"
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

    also = []
    if rank2:
        also.append(f"#2 {rank2['ticker']} ${rank2['price']:.2f} — score {rank2['score']}/225")
    if rank3:
        also.append(f"#3 {rank3['ticker']} ${rank3['price']:.2f} — score {rank3['score']}/225")

    msg = (
        f"BEST STOCK TODAY — {date_str}\n"
        f"─────────────────────\n"
        f"#1 {winner['ticker']} ${p1:.2f}\n"
        f"Score: {winner['score']}/225\n"
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
            winner["score"],
            f"{winner['price']:.6f}",
            winner.get("news_headline", "")[:120],
            winner["rvol"],
            winner["rsi"],
            "",    # price_1day_later  — filled next day
            "",    # price_3day_later  — filled in 3 days
            "",    # actual_gain_loss_pct
        ]
        with open(BEST_PICKS_LOG, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        print(f"📝 [Scanner] Best pick logged → {BEST_PICKS_LOG}")
    except Exception as e:
        print(f"⚠️  [Scanner] Failed to log pick: {e}")


def update_pick_accuracy():
    """
    Back-fill price_1day_later / price_3day_later for past picks.
    Called automatically at the start of each morning scan.
    """
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

            try:
                current_p = float(yf.Ticker(row["ticker"]).fast_info.last_price or 0)
            except Exception:
                continue
            if current_p <= 0:
                continue

            if days_ago >= 1 and not row.get("price_1day_later"):
                row["price_1day_later"] = f"{current_p:.6f}"
                updated = True

            if days_ago >= 3 and not row.get("price_3day_later"):
                row["price_3day_later"]     = f"{current_p:.6f}"
                pct = (current_p - price_at) / price_at * 100
                row["actual_gain_loss_pct"] = f"{pct:+.2f}%"
                updated = True

        if updated:
            with open(BEST_PICKS_LOG, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=BEST_PICKS_COLUMNS)
                w.writeheader()
                w.writerows(rows)
            print("📊 [Scanner] Accuracy log updated")

    except Exception as e:
        print(f"⚠️  [Scanner] Accuracy update failed: {e}")


# ── Main Best-of-Day pipeline ──────────────────────────────────────────────────

def scan_best_of_day(paper: bool = False, verbose: bool = False,
                      max_universe: int = 6000,
                      extra_tickers: list = None) -> dict:
    """
    Full Best-of-Day selection pipeline.

    Returns the winner dict (empty dict if no qualifying stock found).
    WhatsApp message is in winner['whatsapp_msg'].
    """
    print("\n" + "═" * 55)
    print("🏆 [Best-of-Day] Starting selection pipeline...")
    print("═" * 55)

    # Back-fill previous picks' prices
    update_pick_accuracy()

    # ── Load universe ─────────────────────────────────────────────────────────
    universe = _load_universe(max_universe)
    if extra_tickers:
        universe = list(dict.fromkeys(extra_tickers + universe))
    universe = [t for t in universe if t.isalpha() and 1 <= len(t) <= 5]
    n_universe = len(universe)

    # ── Bulk OHLCV download ────────────────────────────────────────────────────
    raw_data     = _bulk_download_batched(universe)
    n_downloaded = len(raw_data)

    # ── Gate 1: RVOL >= 2.0 ───────────────────────────────────────────────────
    g1 = {t: d for t, d in raw_data.items()
          if _rvol(d["volumes"]) >= RVOL_MIN}
    print(f"🚦 Gate 1 (RVOL ≥{RVOL_MIN:.1f}):  {n_downloaded:,} → {len(g1):,} stocks")

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

    # ── Fetch market caps for survivors ────────────────────────────────────────
    print(f"\n📊 Scoring {len(g2)} survivors...")
    for ticker, data, _ in g2:
        data["market_cap"] = _get_market_cap(ticker)

    # ── Score survivors ────────────────────────────────────────────────────────
    scored = []
    for ticker, data, news in g2:
        sc = _score_survivor(data, news)
        scored.append({
            "ticker":        ticker,
            "price":         data["price"],
            "score":         sc["score"],
            "rvol":          sc["rvol"],
            "rsi":           sc["rsi"],
            "gap_pct":       sc["gap_pct"],
            "market_cap":    data.get("market_cap", 0),
            "macd_cross":    sc["macd_cross"],
            "ema_cross":     sc["ema_cross"],
            "news_headline": news.get("headline", ""),
            "news_hours":    news.get("hours_old", 999),
            "signals":       sc["signals"],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    if verbose:
        print("\n📋 All scored survivors:")
        for s in scored[:15]:
            print(f"   {s['ticker']:6s}  {s['score']:3d}/225  "
                  f"RVOL {s['rvol']:.1f}x  RSI {s['rsi']:.0f}  "
                  f"| {s['signals'][:65]}")

    # ── Filter by minimum score ────────────────────────────────────────────────
    qualifiers = [s for s in scored if s["score"] >= MIN_SCORE]
    print(f"\n🎯 Qualifying (score ≥{MIN_SCORE}): {len(qualifiers)} stocks")

    if not qualifiers:
        print(f"\n⚠️  [Best-of-Day] No stocks scored ≥{MIN_SCORE} — no pick today.")
        return {}

    top3 = qualifiers[:3]
    print("\nTop 3 finalists:")
    for s in top3:
        print(f"   {'★' if s is top3[0] else ' '} {s['ticker']:6s}  "
              f"{s['score']:3d}/225  RVOL {s['rvol']:.1f}x  "
              f"RSI {s['rsi']:.0f}  |  {s['signals'][:55]}")

    # ── Claude ranking ─────────────────────────────────────────────────────────
    claude = _claude_rank(top3, verbose=verbose)
    rank   = claude.get("rank", [s["ticker"] for s in top3])

    rank_map    = {t: i for i, t in enumerate(rank)}
    top3_sorted = sorted(top3, key=lambda x: rank_map.get(x["ticker"], 99))
    winner      = top3_sorted[0]
    rank2       = top3_sorted[1] if len(top3_sorted) > 1 else {}
    rank3       = top3_sorted[2] if len(top3_sorted) > 2 else {}

    print(f"\n🥇 WINNER: {winner['ticker']}  "
          f"score={winner['score']}/225  ${winner['price']:.4f}")
    print(f"   Why: {claude.get('why', '')[:110]}")
    print(f"   Expected: {claude.get('expected_move', '')}")
    print(f"   Risk: {claude.get('key_risk', '')}")

    # ── Format WhatsApp message ────────────────────────────────────────────────
    msg = _format_whatsapp(winner, rank2, rank3, claude, n_universe, len(g2))
    winner["whatsapp_msg"] = msg

    print("\n" + "─" * 55)
    print("📱 WhatsApp message:")
    print("─" * 55)
    print(msg)
    print("─" * 55)

    # ── Send / paper-mode ─────────────────────────────────────────────────────
    if paper:
        print("\n📋 [PAPER MODE] WhatsApp NOT sent.")
    else:
        from alerts import send_whatsapp
        send_whatsapp(msg)
        print("✅ [Best-of-Day] WhatsApp sent.")

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
            paper        = True,           # always paper in test mode; remove for live
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
