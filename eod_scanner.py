"""
eod_scanner.py — End-of-Day scanner: finds stocks with potential to rise
at tomorrow morning's open.

Schedule:
  3:30 PM ET  --pre-close       Accumulation, coiling, bounce, sector-lag patterns
  4:15 PM ET  --after-close     Earnings releases + EDGAR 8-K filings (4-6 PM window)
  6:00 PM ET  --evening-scan    After-hours prices + Polygon news finalization
  8:00 PM ET  --final-overnight Last EDGAR + news sweep before sleeping

Output: data/tomorrow_watchlist.json
  → read by premarket_scanner.py at 4 AM
  → read by main.py at startup
  → signals feeding signal_aggregator.py (+15 pts if in_tomorrow_watchlist)
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import requests
import yfinance as yf

_ET = ZoneInfo("America/New_York")

_DIR        = os.path.dirname(os.path.abspath(__file__))
TOMORROW_WL = os.path.join(_DIR, "data", "tomorrow_watchlist.json")
_SEP        = "─" * 30

MAX_WORKERS     = 12
MIN_PRICE       = 0.50
MAX_PRICE       = 500.0
SCORE_THRESHOLD = 50   # min score to be a candidate
MAX_SETUPS      = 10   # max items in tomorrow_watchlist

_SECTOR_ETFS: Dict[str, str] = {
    "Technology":             "XLK",
    "Health Care":            "XLV",
    "Financials":             "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials":            "XLI",
    "Communication Services": "XLC",
    "Energy":                 "XLE",
    "Consumer Staples":       "XLP",
    "Real Estate":            "XLRE",
    "Materials":              "XLB",
    "Utilities":              "XLU",
}

EDGAR_FEED_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include"
    "&count=40&search_text=&output=atom"
)
EDGAR_HEADERS = {"User-Agent": "argus-agent research@argus.local"}
ATOM_NS       = "http://www.w3.org/2005/Atom"

_cik_map: Dict[str, str] = {}


# ── Technical helpers ──────────────────────────────────────────────────────────

def _compute_rsi(closes, period: int = 14) -> float:
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = gains[-period:].mean()
    avg_l  = losses[-period:].mean()
    if avg_l == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_g / avg_l), 1)


def _compute_atr(highs, lows, closes, period: int = 14) -> float:
    highs  = np.asarray(highs,  dtype=float)
    lows   = np.asarray(lows,   dtype=float)
    closes = np.asarray(closes, dtype=float)
    if len(closes) < 2:
        return float(highs[-1] - lows[-1]) if len(highs) > 0 else 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    return float(np.mean(trs[-period:])) if trs else 0.0


def _obv_trending_up(closes, volumes) -> bool:
    """True if OBV over last N bars is net positive (accumulation)."""
    closes  = list(closes)
    volumes = list(volumes)
    if len(closes) < 2:
        return False
    obv, obv_start = 0, 0
    for i in range(1, len(closes)):
        obv += volumes[i] if closes[i] > closes[i - 1] else (-volumes[i] if closes[i] < closes[i - 1] else 0)
        if i == 1:
            obv_start = obv
    return obv > obv_start


def _is_hammer_or_doji(open_p: float, high: float, low: float, close: float) -> bool:
    """Returns True if today's candle looks like a hammer or doji."""
    body   = abs(close - open_p)
    range_ = high - low
    if range_ == 0:
        return False
    if body / range_ < 0.15:          # doji
        return True
    lower_wick = min(open_p, close) - low
    upper_wick = high - max(open_p, close)
    return lower_wick > body * 2 and upper_wick < body * 0.5  # hammer


# ── Accumulation detection helper ─────────────────────────────────────────────

def detect_accumulation(ticker: str, bars_5m: list) -> int:
    """
    Returns accumulation score 0-4 using the last 6 5-min bars (~30 min).
    4 = strong accumulation, 0 = no signal.

    Checks:
    1. price_trend:    last bar close > first bar close
    2. volume_trend:   last 3 bars avg vol > first 3 bars avg vol
    3. close_strength: last close above bar midpoint (h+l)/2
    4. above_vwap:     last close > 6-bar typical-price average
    """
    if len(bars_5m) < 6:
        return 0
    last6  = bars_5m[-6:]
    first3 = last6[:3]
    last3  = last6[3:]

    price_trend    = last6[-1]["c"] > last6[0]["c"]
    vol_f          = sum(b["v"] for b in first3) / 3
    vol_l          = sum(b["v"] for b in last3)  / 3
    volume_trend   = (vol_l > vol_f) if vol_f > 0 else False
    lb             = last6[-1]
    close_strength = lb["c"] > (lb["h"] + lb["l"]) / 2
    tp_avg         = sum((b["h"] + b["l"] + b["c"]) / 3 for b in last6) / 6
    above_vwap     = last6[-1]["c"] > tp_avg

    return sum([price_trend, volume_trend, close_strength, above_vwap])


# ── Sector ETF returns ─────────────────────────────────────────────────────────

def _get_sector_returns() -> Dict[str, float]:
    """Fetch today's % return for each major sector ETF (parallel yfinance)."""
    results: Dict[str, float] = {}

    def _fetch(sector: str, etf: str) -> Tuple[str, float]:
        try:
            fi    = yf.Ticker(etf).fast_info
            price = float(fi.get("last_price")     or 0)
            prev  = float(fi.get("previous_close") or 0)
            if price > 0 and prev > 0:
                return sector, round((price - prev) / prev * 100, 2)
        except Exception:
            pass
        return sector, 0.0

    with ThreadPoolExecutor(max_workers=8) as exe:
        futs = [exe.submit(_fetch, s, e) for s, e in _SECTOR_ETFS.items()]
        for fut in as_completed(futs):
            s, r = fut.result()
            results[s] = r
    return results


# ── Per-ticker EOD data ────────────────────────────────────────────────────────

def _fetch_eod_data(ticker: str, sector_returns: Dict[str, float]) -> Optional[Dict]:
    """
    Fetch all EOD data for one ticker.
    Returns None if data is invalid or price out of range.
    """
    try:
        t     = yf.Ticker(ticker)
        daily = t.history(period="60d", interval="1d", auto_adjust=True)
        if daily is None or daily.empty or len(daily) < 10:
            return None

        intra = t.history(period="1d", interval="5m", auto_adjust=True)

        fi         = t.fast_info
        price      = float(fi.get("last_price")                 or 0)
        prev_close = float(fi.get("previous_close")             or daily["Close"].iloc[-2])
        day_high   = float(fi.get("day_high")                   or daily["High"].iloc[-1])
        day_low    = float(fi.get("day_low")                    or daily["Low"].iloc[-1])
        avg_vol    = float(fi.get("three_month_average_volume") or daily["Volume"].mean())
        today_vol  = float(fi.get("volume")                     or daily["Volume"].iloc[-1])

        if price <= 0 or prev_close <= 0:
            return None
        if not (MIN_PRICE <= price <= MAX_PRICE):
            return None

        closes  = daily["Close"].values
        highs   = daily["High"].values
        lows    = daily["Low"].values
        volumes = daily["Volume"].values

        rsi        = _compute_rsi(closes)
        atr        = _compute_atr(highs, lows, closes)
        support    = float(lows[-20:].min())   if len(lows)  >= 20 else float(lows.min())
        resistance = float(highs[-20:].max())  if len(highs) >= 20 else float(highs.max())
        today_open = float(daily["Open"].iloc[-1])
        vol_ratio  = today_vol / avg_vol if avg_vol > 0 else 1.0
        gap_pct    = (today_open - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        obv_up     = _obv_trending_up(closes[-5:], volumes[-5:])

        # 5-min bars for accumulation check
        bars_5m: list = []
        if intra is not None and not intra.empty:
            for _, row in intra.iterrows():
                bars_5m.append({
                    "c": float(row["Close"]),
                    "h": float(row["High"]),
                    "l": float(row["Low"]),
                    "v": float(row["Volume"]),
                })
        accum_score = detect_accumulation(ticker, bars_5m)

        # Sector
        sector = "Technology"
        try:
            info   = t.info
            sector = info.get("sector") or "Technology"
        except Exception:
            pass
        sector_ret = sector_returns.get(sector, 0.0)

        # Earnings proximity
        days_to_earnings = 999
        try:
            cal   = t.calendar
            today = datetime.now(_ET).date()
            if cal is not None and not (hasattr(cal, "empty") and cal.empty):
                col = None
                for c in ["Earnings Date", "Earnings Dates", 0]:
                    try:
                        col = cal[c] if isinstance(c, str) else cal.iloc[:, c]
                        break
                    except Exception:
                        continue
                if col is not None:
                    for raw in (col if hasattr(col, "__iter__") else [col]):
                        try:
                            d    = raw.date() if hasattr(raw, "date") else datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
                            days = (d - today).days
                            if 0 <= days < days_to_earnings:
                                days_to_earnings = days
                        except Exception:
                            continue
        except Exception:
            pass

        today_range = day_high - day_low
        stock_ret   = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

        return {
            "ticker":           ticker,
            "price":            round(price, 4),
            "prev_close":       round(prev_close, 4),
            "today_open":       round(today_open, 4),
            "day_high":         round(day_high, 4),
            "day_low":          round(day_low, 4),
            "today_range":      round(today_range, 4),
            "today_vol":        int(today_vol),
            "avg_vol":          int(avg_vol),
            "vol_ratio":        round(vol_ratio, 2),
            "rsi":              rsi,
            "atr":              round(atr, 4),
            "support":          round(support, 4),
            "resistance":       round(resistance, 4),
            "gap_pct":          round(gap_pct, 2),
            "stock_ret":        round(stock_ret, 2),
            "obv_up":           obv_up,
            "accum_score":      accum_score,
            "sector":           sector,
            "sector_ret":       sector_ret,
            "days_to_earnings": days_to_earnings,
            "hammer_doji":      _is_hammer_or_doji(today_open, day_high, day_low, price),
            # Set by scoring / Claude
            "score":            0,
            "patterns":         [],
            "pattern_name":     "",
            "entry_low":        0.0,
            "entry_high":       0.0,
            "target":           0.0,
            "stop":             0.0,
            "reason":           "",
            "main_risk":        "",
            "confidence":       "LOW",
            "strength":         0,
        }
    except Exception:
        return None


# ── Pattern scoring ────────────────────────────────────────────────────────────

def _score_preclose_patterns(data: Dict) -> Tuple[int, List[str]]:
    """
    Score a ticker against 6 pre-close setup patterns.
    Returns (score, list_of_pattern_names).
    """
    score    = 0
    patterns: List[str] = []

    price      = data["price"]
    prev_close = data["prev_close"]
    day_high   = data["day_high"]
    day_low    = data["day_low"]
    rsi        = data["rsi"]
    atr        = data["atr"]
    vol_ratio  = data["vol_ratio"]
    support    = data["support"]
    resistance = data["resistance"]
    gap_pct    = data["gap_pct"]
    accum      = data["accum_score"]
    sector_ret = data["sector_ret"]
    stock_ret  = data["stock_ret"]

    # ── Pattern 1: Accumulation into close ───────────────────────────────────
    close_to_high = day_high > 0 and abs(price - day_high) / price < 0.01
    p1 = sum([
        stock_ret > 0.1,        # up today
        accum >= 2,             # volume + price trending up (30-min bars)
        close_to_high,          # within 1% of today's high
        data["obv_up"],         # OBV up last 5 days
    ])
    if p1 == 4:
        score += 35
        patterns.append("accumulation")
    elif p1 >= 3:
        score += 20
        patterns.append("accumulation")

    # ── Pattern 2: Oversold bounce setup ─────────────────────────────────────
    at_support = support > 0 and abs(price - support) / price < 0.03
    p2 = sum([
        20 <= rsi <= 35,
        vol_ratio >= 2.0,
        at_support,
        data["hammer_doji"],
    ])
    if p2 == 4:
        score += 40
        patterns.append("oversold_bounce")
    elif p2 >= 3:
        score += 25
        patterns.append("oversold_bounce")

    # ── Pattern 3: Catalyst pending tomorrow ─────────────────────────────────
    dte = data["days_to_earnings"]
    if dte in (0, 1):
        score += 30
        patterns.append("earnings_tomorrow")
    elif dte == 2:
        score += 15
        patterns.append("earnings_soon")

    # ── Pattern 4: Technical coiling (spring loading) ─────────────────────────
    tight_range = atr > 0 and data["today_range"] < atr * 0.5
    low_volume  = vol_ratio < 0.5
    near_res    = resistance > 0 and abs(price - resistance) / price < 0.005
    if sum([tight_range, low_volume, near_res]) == 3:
        score += 30
        patterns.append("coiling")

    # ── Pattern 5: Sector catch-up laggard ───────────────────────────────────
    if sector_ret > 1.0 and stock_ret < 0.2:
        score += 25
        patterns.append("sector_catchup")

    # ── Pattern 6: Gap fill setup ─────────────────────────────────────────────
    recovering = price > day_low * 1.01   # bouncing off low
    if gap_pct < -2.0 and recovering and vol_ratio > 1.2:
        score += 20
        patterns.append("gap_fill")

    return score, patterns


# ── Claude analysis ────────────────────────────────────────────────────────────

def _claude_analyze_setups(candidates: List[Dict]) -> List[Dict]:
    """
    Claude Haiku: analyze pre-close setup candidates.
    Returns candidates enriched with entry/target/stop/reasoning.
    """
    if not candidates:
        return []
    try:
        import anthropic
        client   = anthropic.Anthropic()
        now_et   = datetime.now(_ET)
        tomorrow = (now_et.date() + timedelta(days=1)).strftime("%a %b %d")

        lines = [
            f"- {c['ticker']}: price=${c['price']:.2f} rsi={c['rsi']:.0f} "
            f"patterns={c['patterns']} score={c['score']} "
            f"sector={c.get('sector','?')} sectorRet={c.get('sector_ret',0):+.1f}% "
            f"volRatio={c['vol_ratio']:.1f}x atr={c['atr']:.2f} "
            f"support=${c['support']:.2f} resistance=${c['resistance']:.2f} "
            f"daysToEarnings={c.get('days_to_earnings',999)}"
            for c in candidates
        ]

        prompt = f"""You are an end-of-day swing trade analyst (market closes in ~30 min or just closed).
These stocks show technical setup patterns for tomorrow's open ({tomorrow}).

Candidates:
{chr(10).join(lines)}

For each stock analyze:
- Primary pattern driving the setup
- Setup strength 1-10
- Expected price range at tomorrow's open
- Entry zone (slightly above today's close for confirmation)
- Stop loss (where setup is invalidated — below today's low / key support)
- 1-2 sentence reason why it should move up at tomorrow's open
- Main risk that would kill the setup
- Confidence: HIGH / MEDIUM / LOW

Return ONLY a JSON array:
[{{
  "ticker": "XXXX",
  "pattern_name": "accumulation|oversold_bounce|earnings_tomorrow|coiling|sector_catchup|gap_fill",
  "strength": <int 1-10>,
  "tomorrow_open_target": <float>,
  "entry_low": <float>,
  "entry_high": <float>,
  "target": <float>,
  "stop": <float>,
  "reason": "<1-2 sentences>",
  "main_risk": "<1 sentence>",
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}}]

Rules:
- entry_low: today's close + small buffer (0.3-1%)
- entry_high: today's close + 1-2%
- stop: below today's low
- target: realistic (1-ATR above entry for swing, 2-ATR for breakout)
- HIGH confidence only when multiple patterns align with volume confirmation"""

        resp  = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text  = resp.content[0].text.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return _fallback_setups(candidates)

        verdicts    = json.loads(match.group())
        verdict_map = {v["ticker"]: v for v in verdicts if isinstance(v, dict)}

        for c in candidates:
            v     = verdict_map.get(c["ticker"], {})
            price = c["price"]
            c["pattern_name"]         = v.get("pattern_name",         c["patterns"][0] if c["patterns"] else "unknown")
            c["strength"]             = v.get("strength",             5)
            c["tomorrow_open_target"] = v.get("tomorrow_open_target", price * 1.02)
            c["entry_low"]            = v.get("entry_low",            price * 1.003)
            c["entry_high"]           = v.get("entry_high",           price * 1.012)
            c["target"]               = v.get("target",               price * 1.05)
            c["stop"]                 = v.get("stop",                 c.get("day_low", price * 0.95) * 0.99)
            c["reason"]               = v.get("reason",               "")
            c["main_risk"]            = v.get("main_risk",            "")
            c["confidence"]           = v.get("confidence",           "MEDIUM")
        return candidates

    except Exception as e:
        print(f"⚠️  [EOD] Claude setup analysis failed: {e}")
        return _fallback_setups(candidates)


def _fallback_setups(candidates: List[Dict]) -> List[Dict]:
    """Score-based fallback estimates when Claude is unavailable."""
    for c in candidates:
        price = c["price"]
        c["pattern_name"]         = c["patterns"][0] if c["patterns"] else "unknown"
        c["strength"]             = min(10, max(1, c["score"] // 8))
        c["tomorrow_open_target"] = round(price * 1.02, 2)
        c["entry_low"]            = round(price * 1.003, 2)
        c["entry_high"]           = round(price * 1.012, 2)
        c["target"]               = round(price * 1.05, 2)
        c["stop"]                 = round(c.get("day_low", price * 0.95) * 0.99, 2)
        c["reason"]               = "Score-based estimate (Claude unavailable)"
        c["main_risk"]            = "Verify setup at open"
        c["confidence"]           = "LOW"
    return candidates


def _claude_analyze_8k(ticker: str, summary: str, company: str) -> Dict:
    """Quick single-filing sentiment check via Claude Haiku."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp   = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content":
                f"Is this 8-K filing bullish, bearish, or neutral for {ticker} tomorrow?\n"
                f"Company: {company}\nFiling: {summary[:500]}\n\n"
                f"Return JSON only: {{\"sentiment\": \"BULLISH\"|\"BEARISH\"|\"NEUTRAL\", "
                f"\"impact\": <int 1-10>, \"reason\": \"<1 sentence>\"}}"
            }],
        )
        match = re.search(r"\{.*\}", resp.content[0].text.strip(), re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"sentiment": "NEUTRAL", "impact": 0, "reason": ""}


# ── Persistence ────────────────────────────────────────────────────────────────

def _save_tomorrow_watchlist(setups: List[Dict], generated_at: str = "") -> None:
    os.makedirs(os.path.dirname(TOMORROW_WL), exist_ok=True)
    now_et   = datetime.now(_ET)
    tomorrow = (now_et.date() + timedelta(days=1)).strftime("%Y-%m-%d")
    payload  = {
        "date":         tomorrow,
        "generated_at": generated_at or now_et.strftime("%Y-%m-%d %H:%M ET"),
        "setups":       setups,
    }
    with open(TOMORROW_WL, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"💾 [EOD] Saved {len(setups)} setups → tomorrow_watchlist.json")

    # Notify IntelligenceHub so other agents can react
    try:
        from intelligence_hub import hub as _hub
        _hub.set("tomorrow_ready", True)
        _hub.set("tomorrow_count", len(setups))
        best = setups[0] if setups else {}
        _hub.set("tomorrow_best", {
            "ticker":     best.get("ticker", ""),
            "score":      best.get("score", 0),
            "setup_type": best.get("setup_type", ""),
        })
    except Exception as _hub_err:
        print(f"⚠️  [EOD] Hub notify failed (non-fatal): {_hub_err}")


def _load_tomorrow_watchlist() -> Dict:
    try:
        if not os.path.exists(TOMORROW_WL):
            return {}
        with open(TOMORROW_WL) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  [EOD] Could not load tomorrow watchlist: {e}")
        return {}


def load_tomorrow_tickers() -> List[str]:
    """Return ticker list from tomorrow_watchlist.json (for main.py + premarket_scanner.py)."""
    data = _load_tomorrow_watchlist()
    return [s["ticker"] for s in data.get("setups", []) if s.get("ticker")]


def _setup_to_record(c: Dict) -> Dict:
    """Standardize a candidate dict to the watchlist record format."""
    price = c.get("price", 0)
    return {
        "ticker":     c["ticker"],
        "pattern":    c.get("pattern_name", c["patterns"][0] if c.get("patterns") else "unknown"),
        "score":      c.get("score", 0),
        "entry_low":  round(c.get("entry_low",  price * 1.003), 4),
        "entry_high": round(c.get("entry_high", price * 1.012), 4),
        "target":     round(c.get("target",     price * 1.05),  4),
        "stop":       round(c.get("stop",       c.get("day_low", price * 0.95) * 0.99), 4),
        "reason":     c.get("reason",    "")[:200],
        "main_risk":  c.get("main_risk", "")[:120],
        "confidence": c.get("confidence", "MEDIUM"),
        "strength":   c.get("strength",  5),
        "price":      price,
        "rsi":        c.get("rsi", 50),
        "patterns":   c.get("patterns", []),
    }


# ── Universe builder ───────────────────────────────────────────────────────────

def _build_universe(size: int = 200) -> List[str]:
    import watchlist_manager as wl
    import world_context as _wctx
    from discovery_agent import _get_catalyst_tickers
    watchlist        = wl.load()
    catalyst_tickers = _get_catalyst_tickers()
    universe: list   = []
    try:
        from market_scanner import _load_universe
        universe = _load_universe() or []
    except Exception:
        pass
    return list(dict.fromkeys(catalyst_tickers + watchlist + universe[:size]))


# ── EDGAR helpers ──────────────────────────────────────────────────────────────

def _ensure_cik_map() -> None:
    global _cik_map
    if _cik_map:
        return
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS, timeout=15,
        )
        data     = r.json()
        _cik_map = {str(v["cik_str"]): v["ticker"].upper() for v in data.values()}
        print(f"📋 [EOD/EDGAR] Loaded {len(_cik_map)} CIK→ticker mappings")
    except Exception as e:
        print(f"⚠️  [EOD/EDGAR] CIK map failed: {e}")


def _fetch_recent_8ks(since_minutes: int = 120) -> List[Dict]:
    """Return EDGAR 8-K filings from the last `since_minutes` minutes."""
    _ensure_cik_map()
    results: List[Dict] = []
    try:
        r = requests.get(EDGAR_FEED_URL, headers=EDGAR_HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        root    = ET.fromstring(r.text)
        entries = root.findall(f"{{{ATOM_NS}}}entry")
        cutoff  = datetime.now(_ET).timestamp() - since_minutes * 60

        for entry in entries:
            try:
                updated_el  = entry.find(f"{{{ATOM_NS}}}updated")
                updated_str = updated_el.text if updated_el is not None else ""
                filed_ts    = datetime.fromisoformat(updated_str.replace("Z", "+00:00")).timestamp() if updated_str else 0
                if filed_ts < cutoff:
                    continue

                id_el   = entry.find(f"{{{ATOM_NS}}}id")
                id_str  = id_el.text if id_el is not None else ""
                cik_m   = re.search(r"CIK=(\d+)", id_str)
                if not cik_m:
                    continue
                cik    = cik_m.group(1).lstrip("0")
                ticker = _cik_map.get(cik, "")
                if not ticker:
                    continue

                title_el = entry.find(f"{{{ATOM_NS}}}title")
                co_el    = entry.find(f"{{{ATOM_NS}}}company-name")
                results.append({
                    "ticker":   ticker,
                    "company":  co_el.text if co_el is not None else ticker,
                    "title":    title_el.text if title_el is not None else "",
                    "filed_at": updated_str,
                })
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️  [EOD/EDGAR] Feed error: {e}")
    return results


# ── WhatsApp message formatting ────────────────────────────────────────────────

def _format_preclose_msg(candidates: List[Dict]) -> str:
    now_et   = datetime.now(_ET)
    tomorrow = (now_et.date() + timedelta(days=1)).strftime("%a %b %d")
    shown    = sorted(
        [c for c in candidates if c.get("score", 0) >= 65],
        key=lambda x: x.get("score", 0), reverse=True
    )[:5] or candidates[:3]

    lines = [
        f"🌇 PRE-CLOSE ALERT — {now_et.strftime('%H:%M ET')}",
        f"Tomorrow's potential setups ({tomorrow}):",
        _SEP,
    ]
    for i, c in enumerate(shown, 1):
        em     = (c.get("entry_low",  c["price"]) + c.get("entry_high", c["price"])) / 2
        t_pct  = (c["target"] - em) / em * 100 if em > 0 and c["target"] > 0 else 0
        s_pct  = (c["stop"]   - em) / em * 100 if em > 0 and c["stop"]   > 0 else 0
        lines += [
            f"\n{i}. {c['ticker']} ${c['price']:.2f}  [{c['score']}/100]",
            f"Pattern: {c.get('pattern_name','?').replace('_',' ').title()}  "
            f"Strength: {c.get('strength',5)}/10 [{c.get('confidence','?')}]",
            f"Tomorrow entry: ${c.get('entry_low',0):.2f} – ${c.get('entry_high',0):.2f}",
            f"Tomorrow target: ${c['target']:.2f} (+{t_pct:.1f}%)",
            f"Stop: ${c['stop']:.2f} ({s_pct:.1f}%)",
        ]
        if c.get("reason"):    lines.append(f"Why: {c['reason'][:120]}")
        if c.get("main_risk"): lines.append(f"Risk: {c['main_risk'][:100]}")
    lines += [f"\n{_SEP}", "Market closes in 30 minutes", "Full after-hours scan at 4:15 PM"]
    return "\n".join(lines)


def _format_afterclose_msg(new_picks: List[Dict], removed: List[str], total: int) -> str:
    now_et = datetime.now(_ET)
    lines  = [f"📋 AFTER-CLOSE UPDATE — {now_et.strftime('%H:%M ET')}", _SEP]
    if new_picks:
        lines.append("🆕 NEW PICKS (EDGAR / Earnings):")
        for p in new_picks[:5]:
            lines.append(f"  {p['ticker']}  {p.get('reason','')[:80]}")
    if removed:
        lines.append(f"\n⛔ REMOVED (bad earnings): {', '.join(removed)}")
    lines += [f"\n📋 Total setups for tomorrow: {total}", _SEP, "Evening scan at 6:00 PM"]
    return "\n".join(lines)


def _format_evening_msg(setups: List[Dict]) -> str:
    now_et   = datetime.now(_ET)
    tomorrow = (now_et.date() + timedelta(days=1)).strftime("%a %b %d")
    lines    = [f"🌆 EVENING DIGEST — {now_et.strftime('%H:%M ET')}", f"Tomorrow's top 5 ({tomorrow}):", _SEP]
    for i, c in enumerate(setups[:5], 1):
        price    = c.get("price", 0)
        ah_price = c.get("afterhours_price", 0)
        ah_tag   = f" (AH: {(ah_price - price) / price * 100:+.1f}%)" if ah_price and price else ""
        lines   += [
            f"\n{i}. {c['ticker']} ${price:.2f}{ah_tag}",
            f"   Score: {c.get('score',0)}/100  |  Confidence: {c.get('confidence','?')}",
            f"   Pattern: {c.get('pattern_name','?').replace('_',' ').title()}",
            f"   Entry tomorrow: ${c.get('entry_low',0):.2f} – ${c.get('entry_high',0):.2f}",
            f"   Target: ${c.get('target',0):.2f}  |  Stop: ${c.get('stop',0):.2f}",
        ]
        if c.get("reason"): lines.append(f"   {c['reason'][:120]}")
    lines += [f"\n{_SEP}", "These will be monitored from 4 AM", "Morning brief at 8:00 AM"]
    return "\n".join(lines)


# ── FUNCTION 1 — 3:30 PM Pre-close scan ───────────────────────────────────────

def run_preclose_scan(test: bool = False, verbose: bool = False) -> List[Dict]:
    """
    3:30 PM: Detect stocks accumulating, bouncing, coiling, or lagging their sector.
    Saves tomorrow_watchlist.json, sends WhatsApp.
    """
    from alerts import send_whatsapp
    now_et = datetime.now(_ET)
    print(f"\n{_SEP}")
    print(f"🌇 [EOD/Pre-Close] Starting at {now_et.strftime('%H:%M:%S')} ET")

    # Pre-fetch sector returns (once, shared across all tickers)
    print("   Fetching sector ETF returns...")
    sector_rets = _get_sector_returns()
    if verbose:
        for s, r in sorted(sector_rets.items(), key=lambda x: -abs(x[1]))[:5]:
            print(f"     {s}: {r:+.1f}%")

    # Check for active sector catalysts — lower threshold on catalyst days
    import world_context as _wctx
    _geo             = _wctx.get().get("geo", {})
    _active_catalysts = [c for c in _geo.get("sector_catalysts", []) if c.get("score", 0) >= 7]
    _effective_threshold = 45 if _active_catalysts else SCORE_THRESHOLD
    if _active_catalysts:
        print(f"⚡ [EOD] Catalyst day — threshold lowered to {_effective_threshold} "
              f"({len(_active_catalysts)} active catalyst(s))")

    tickers = _build_universe(size=200)
    print(f"   Scanning {len(tickers)} tickers in parallel...")

    raw: List[Dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futs = {exe.submit(_fetch_eod_data, t, sector_rets): t for t in tickers}
        for fut in as_completed(futs):
            d = fut.result()
            if d:
                raw.append(d)
    print(f"   Data fetched for {len(raw)} tickers")

    # Score patterns
    for d in raw:
        score, patterns  = _score_preclose_patterns(d)
        d["score"]    = score
        d["patterns"] = patterns

    candidates = sorted(
        [d for d in raw if d["score"] >= _effective_threshold and d["patterns"]],
        key=lambda x: x["score"], reverse=True,
    )[:MAX_SETUPS]

    print(f"   {len(candidates)} candidates scored >= {_effective_threshold}")
    if verbose:
        for c in candidates:
            print(f"     {c['ticker']:8s}  score={c['score']}  patterns={c['patterns']}  "
                  f"rsi={c['rsi']:.0f}  vol={c['vol_ratio']:.1f}x")

    if not candidates:
        print("💤 [EOD/Pre-Close] No setups found today.")
        return []

    # Claude analysis
    print(f"   Claude analysis on {len(candidates)} candidates...")
    analyzed = _claude_analyze_setups(candidates)

    # Save tomorrow_watchlist.json
    setups = [_setup_to_record(c) for c in analyzed]
    _save_tomorrow_watchlist(setups, generated_at=now_et.strftime("%Y-%m-%d %H:%M ET"))

    # Send WhatsApp
    msg = _format_preclose_msg(analyzed)
    if not test:
        send_whatsapp(msg)
    else:
        print(f"\n📋 [TEST] Pre-close alert:\n{msg}")

    print(f"✅ [EOD/Pre-Close] Done — {len(analyzed)} setups saved")
    print(f"{_SEP}\n")
    return analyzed


# ── FUNCTION 2 — 4:15 PM After-close scan ─────────────────────────────────────

def run_afterclose_scan(test: bool = False, verbose: bool = False) -> List[Dict]:
    """
    4:15 PM: Catch earnings releases + EDGAR 8-K filings, update tomorrow_watchlist.
    """
    from alerts import send_whatsapp
    now_et = datetime.now(_ET)
    print(f"\n{_SEP}")
    print(f"📋 [EOD/After-Close] Starting at {now_et.strftime('%H:%M:%S')} ET")

    existing         = _load_tomorrow_watchlist()
    setups           = list(existing.get("setups", []))
    existing_tickers = {s["ticker"] for s in setups}

    new_picks: List[Dict] = []
    removed:   List[str]  = []

    # Step 1 — Earnings check for existing picks
    print(f"   Checking earnings for {len(setups)} existing picks...")
    updated: List[Dict] = []
    for setup in setups:
        ticker = setup["ticker"]
        try:
            surprise_pct = None
            t     = yf.Ticker(ticker)
            try:
                hist = t.earnings_history
                if hist is not None and not hist.empty:
                    row      = hist.iloc[-1]
                    reported = float(row.get("Reported EPS") or 0)
                    estimate = float(row.get("EPS Estimate") or 0)
                    if estimate and reported:
                        surprise_pct = (reported - estimate) / abs(estimate) * 100
            except Exception:
                pass

            if surprise_pct is not None:
                if surprise_pct > 10:
                    setup["score"]             = min(100, setup.get("score", 50) + 40)
                    setup["earnings_beat_pct"] = round(surprise_pct, 1)
                    setup["reason"]            = f"EPS beat {surprise_pct:.0f}%. " + setup.get("reason", "")
                    print(f"   🟢 {ticker}: earnings beat {surprise_pct:.1f}%")
                elif surprise_pct > 5:
                    setup["score"] = min(100, setup.get("score", 50) + 25)
                elif surprise_pct < -5:
                    print(f"   🔴 {ticker}: earnings miss {surprise_pct:.1f}% — removing")
                    removed.append(ticker)
                    continue
            updated.append(setup)
        except Exception:
            updated.append(setup)
    setups = updated

    # Step 2 — EDGAR 8-K filings (last 3 hours)
    print("   Scanning EDGAR 8-K filings (last 3h)...")
    filings = _fetch_recent_8ks(since_minutes=180)
    print(f"   Found {len(filings)} recent 8-Ks")

    for filing in filings:
        ticker   = filing["ticker"]
        analysis = _claude_analyze_8k(ticker, filing["title"], filing["company"])
        sentiment = analysis.get("sentiment", "NEUTRAL")
        impact    = analysis.get("impact",    0)
        reason    = analysis.get("reason",    "")

        if verbose:
            print(f"     {ticker}: {sentiment} {impact}/10 — {filing['title'][:50]}")

        if sentiment == "BULLISH" and impact >= 7 and ticker not in existing_tickers:
            price = 0.0
            try:
                price = float(yf.Ticker(ticker).fast_info.get("last_price") or 0)
            except Exception:
                pass
            if not (MIN_PRICE <= price <= MAX_PRICE):
                continue
            new_setup: Dict = {
                "ticker":     ticker,
                "pattern":    "edgar_8k",
                "score":      min(95, 65 + impact * 3),
                "entry_low":  round(price * 1.002, 2),
                "entry_high": round(price * 1.015, 2),
                "target":     round(price * 1.08,  2),
                "stop":       round(price * 0.93,  2),
                "reason":     f"8-K filing: {reason}",
                "main_risk":  "Market reaction at open uncertain",
                "confidence": "HIGH" if impact >= 8 else "MEDIUM",
                "strength":   impact,
                "price":      price,
                "patterns":   ["edgar_8k"],
            }
            setups.append(new_setup)
            existing_tickers.add(ticker)
            new_picks.append(new_setup)
            print(f"   🆕 Added {ticker} (8-K bullish {impact}/10)")

        elif sentiment == "BEARISH" and impact >= 7 and ticker in existing_tickers:
            setups           = [s for s in setups if s["ticker"] != ticker]
            existing_tickers.discard(ticker)
            removed.append(ticker)
            print(f"   ⛔ Removed {ticker} (8-K bearish {impact}/10)")

    # Save + send
    setups.sort(key=lambda x: x.get("score", 0), reverse=True)
    _save_tomorrow_watchlist(setups, generated_at=now_et.strftime("%Y-%m-%d %H:%M ET"))

    msg = _format_afterclose_msg(new_picks, removed, len(setups))
    if not test:
        send_whatsapp(msg)
    else:
        print(f"\n📋 [TEST] After-close update:\n{msg}")

    print(f"✅ [EOD/After-Close] {len(setups)} setups, {len(new_picks)} new, {len(removed)} removed")
    print(f"{_SEP}\n")
    return setups


# ── FUNCTION 3 — 6:00 PM Evening scan ─────────────────────────────────────────

def run_evening_scan(test: bool = False, verbose: bool = False) -> List[Dict]:
    """
    6:00 PM: Process after-hours prices + Polygon news, finalize tomorrow_watchlist.
    """
    from alerts import send_whatsapp
    now_et = datetime.now(_ET)
    print(f"\n{_SEP}")
    print(f"🌆 [EOD/Evening] Starting at {now_et.strftime('%H:%M:%S')} ET")

    existing         = _load_tomorrow_watchlist()
    setups           = list(existing.get("setups", []))
    existing_tickers = {s["ticker"] for s in setups}

    # Step 1 — Scan Polygon news (no ticker filter, last 100 articles)
    print("   Scanning Polygon news for new bullish tickers...")
    new_news_tickers: List[Dict] = []
    try:
        from polygon_feed import _get as poly_get
        resp     = poly_get("/v2/reference/news", {"limit": 100, "order": "desc"})
        articles = resp.get("results", [])
        print(f"   {len(articles)} news articles scanned")
        seen_new: set = set()
        for art in articles:
            headline = art.get("title", "")
            hl       = headline.lower()
            bullish  = any(w in hl for w in [
                "beat", "upgrade", "contract", "deal", "fda approv",
                "raised guidance", "record revenue", "breakthrough", "award",
                "partnership", "buyout", "acquisition",
            ])
            if not bullish:
                continue
            for tk in art.get("tickers", []):
                if tk not in existing_tickers and tk not in seen_new:
                    seen_new.add(tk)
                    new_news_tickers.append({"ticker": tk, "headline": headline[:100]})
        new_news_tickers = new_news_tickers[:10]
        print(f"   {len(new_news_tickers)} potential new tickers from news")
    except Exception as e:
        print(f"⚠️  [EOD/Evening] Polygon news failed: {e}")

    # Step 2 — After-hours prices for existing picks
    print(f"   Fetching AH prices for {len(setups)} existing picks...")
    for setup in setups:
        ticker = setup["ticker"]
        try:
            info     = yf.Ticker(ticker).info
            ah_price = float(info.get("postMarketPrice") or 0)
            price    = setup.get("price", 0) or float(info.get("regularMarketPrice") or 0)
            if ah_price and price:
                setup["afterhours_price"] = round(ah_price, 4)
                ah_chg = (ah_price - price) / price * 100
                if ah_chg > 2.0:
                    setup["entry_low"]  = round(ah_price * 1.005, 2)
                    setup["entry_high"] = round(ah_price * 1.015, 2)
                    setup["score"]      = min(100, setup.get("score", 50) + 10)
                    setup["confidence"] = "HIGH"
                elif ah_chg < -3.0:
                    setup["score"]      = max(0, setup.get("score", 50) - 20)
                    setup["confidence"] = "LOW"
        except Exception:
            pass

    # Add valid news tickers
    for nt in new_news_tickers:
        ticker = nt["ticker"]
        try:
            price = float(yf.Ticker(ticker).fast_info.get("last_price") or 0)
            if not (MIN_PRICE <= price <= MAX_PRICE):
                continue
            setups.append({
                "ticker":     ticker,
                "pattern":    "news_catalyst",
                "score":      55,
                "entry_low":  round(price * 1.003, 2),
                "entry_high": round(price * 1.012, 2),
                "target":     round(price * 1.06, 2),
                "stop":       round(price * 0.94, 2),
                "reason":     f"News: {nt['headline'][:80]}",
                "main_risk":  "No follow-through at open",
                "confidence": "MEDIUM",
                "strength":   5,
                "price":      price,
                "patterns":   ["news_catalyst"],
            })
            existing_tickers.add(ticker)
        except Exception:
            continue

    # Step 3 — Final sort + save
    setups.sort(key=lambda x: x.get("score", 0), reverse=True)
    setups = setups[:MAX_SETUPS]
    _save_tomorrow_watchlist(setups, generated_at=now_et.strftime("%Y-%m-%d %H:%M ET"))

    msg = _format_evening_msg(setups)
    if not test:
        send_whatsapp(msg)
    else:
        print(f"\n📋 [TEST] Evening digest:\n{msg}")

    print(f"✅ [EOD/Evening] {len(setups)} setups finalized in tomorrow_watchlist")
    print(f"{_SEP}\n")
    return setups


# ── FUNCTION 4 — 8:00 PM Final overnight ──────────────────────────────────────

def run_final_overnight(test: bool = False, verbose: bool = False) -> List[Dict]:
    """
    8:00 PM: Last EDGAR + high-impact news sweep. Alerts only on score >= 85.
    """
    from alerts import send_whatsapp
    now_et = datetime.now(_ET)
    print(f"\n{_SEP}")
    print(f"🌙 [EOD/Final] Overnight check at {now_et.strftime('%H:%M:%S')} ET")

    existing         = _load_tomorrow_watchlist()
    setups           = list(existing.get("setups", []))
    existing_tickers = {s["ticker"] for s in setups}
    alerts_sent      = 0
    new_picks: List[Dict] = []

    # EDGAR 8-K filings last 2 hours
    print("   Scanning EDGAR 8-Ks (last 2h)...")
    filings = _fetch_recent_8ks(since_minutes=120)
    for filing in filings:
        ticker   = filing["ticker"]
        analysis = _claude_analyze_8k(ticker, filing["title"], filing["company"])
        sentiment = analysis.get("sentiment", "NEUTRAL")
        impact    = analysis.get("impact",    0)
        reason    = analysis.get("reason",    "")

        if verbose:
            print(f"   {ticker}: {sentiment} {impact}/10 — {filing['title'][:60]}")

        if sentiment == "BULLISH" and impact >= 8:
            score = min(95, 65 + impact * 3)
            if ticker not in existing_tickers:
                try:
                    price = float(yf.Ticker(ticker).fast_info.get("last_price") or 0)
                    if not (MIN_PRICE <= price <= MAX_PRICE):
                        continue
                    new_setup: Dict = {
                        "ticker":     ticker,
                        "pattern":    "edgar_8k_overnight",
                        "score":      score,
                        "entry_low":  round(price * 1.002, 2),
                        "entry_high": round(price * 1.015, 2),
                        "target":     round(price * 1.08, 2),
                        "stop":       round(price * 0.93, 2),
                        "reason":     f"Late 8-K: {reason}",
                        "main_risk":  "Verify momentum at open",
                        "confidence": "HIGH",
                        "strength":   impact,
                        "price":      price,
                        "patterns":   ["edgar_8k"],
                    }
                    setups.append(new_setup)
                    existing_tickers.add(ticker)
                    new_picks.append(new_setup)
                except Exception:
                    pass

            if score >= 85:
                msg = (
                    f"🔔 OVERNIGHT ALERT — {filing['company']}\n"
                    f"8-K: {filing['title'][:80]}\n"
                    f"Ticker: {ticker}  Impact: {impact}/10\n"
                    f"{reason}\n"
                    f"Added to tomorrow's watchlist"
                )
                if not test:
                    send_whatsapp(msg)
                else:
                    print(f"\n📋 [TEST] Overnight alert:\n{msg}")
                alerts_sent += 1

    # Polygon news — high-impact keywords only
    print("   Scanning Polygon news (last 2h, high-impact)...")
    try:
        from polygon_feed import _get as poly_get
        resp  = poly_get("/v2/reference/news", {"limit": 50, "order": "desc"})
        arts  = resp.get("results", [])
        for art in arts:
            hl     = art.get("title", "")
            hl_low = hl.lower()
            if not any(w in hl_low for w in [
                "fda approv", "acquisition", "merger", "takeover",
                "buyout", "government contract", "strategic partnership",
            ]):
                continue
            for ticker in art.get("tickers", [])[:2]:
                if ticker in existing_tickers:
                    continue
                try:
                    price = float(yf.Ticker(ticker).fast_info.get("last_price") or 0)
                    if not (MIN_PRICE <= price <= MAX_PRICE):
                        continue
                    new_setup = {
                        "ticker":     ticker,
                        "pattern":    "breaking_news",
                        "score":      85,
                        "entry_low":  round(price * 1.002, 2),
                        "entry_high": round(price * 1.02,  2),
                        "target":     round(price * 1.10,  2),
                        "stop":       round(price * 0.92,  2),
                        "reason":     f"Breaking: {hl[:80]}",
                        "main_risk":  "Gap-up at open may be sold",
                        "confidence": "HIGH",
                        "strength":   8,
                        "price":      price,
                        "patterns":   ["breaking_news"],
                    }
                    setups.append(new_setup)
                    existing_tickers.add(ticker)
                    new_picks.append(new_setup)
                    msg = (
                        f"🔔 LATE NEWS — {ticker}\n"
                        f"{hl[:100]}\nAdded to tomorrow's watchlist"
                    )
                    if not test:
                        send_whatsapp(msg)
                    else:
                        print(f"\n📋 [TEST] Late news alert:\n{msg}")
                    alerts_sent += 1
                    break
                except Exception:
                    continue
    except Exception as e:
        print(f"⚠️  [EOD/Final] Polygon news failed: {e}")

    # Save if new picks added
    if new_picks:
        setups.sort(key=lambda x: x.get("score", 0), reverse=True)
        setups = setups[:MAX_SETUPS]
        _save_tomorrow_watchlist(setups, generated_at=now_et.strftime("%Y-%m-%d %H:%M ET"))

    print(f"{'✅' if alerts_sent > 0 else '💤'} [EOD/Final] "
          f"{alerts_sent} alert(s), {len(new_picks)} new pick(s)")
    print(f"{_SEP}\n")
    return setups


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argus EOD Scanner")
    parser.add_argument("--pre-close",       action="store_true", help="3:30 PM pre-close scan")
    parser.add_argument("--after-close",     action="store_true", help="4:15 PM after-close scan")
    parser.add_argument("--evening-scan",    action="store_true", help="6:00 PM evening scan")
    parser.add_argument("--final-overnight", action="store_true", help="8:00 PM final check")
    parser.add_argument("--test",            action="store_true", help="Dry-run (no WhatsApp)")
    parser.add_argument("--verbose", "-v",   action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.pre_close:
        run_preclose_scan(test=args.test, verbose=args.verbose)
    elif args.after_close:
        run_afterclose_scan(test=args.test, verbose=args.verbose)
    elif args.evening_scan:
        run_evening_scan(test=args.test, verbose=args.verbose)
    elif args.final_overnight:
        run_final_overnight(test=args.test, verbose=args.verbose)
    else:
        print("Usage: python3 eod_scanner.py --pre-close|--after-close|--evening-scan|--final-overnight [--test] [--verbose]")
        sys.exit(1)
