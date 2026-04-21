"""
scanners/premarket_scanner.py — Focused catalyst + recovery pre-market scanner.

Runs at 7:45 AM ET. Finds stocks gapping up with confirmed catalysts and
beaten-down recovery patterns — the highest-quality pre-market setups only.

Pipeline:
  1. Fetch Polygon top-50 gainers (paid tier) OR scan fallback universe (free tier)
  2. Gap / volume / price / earnings filters
  3. Fetch 30-day OHLCV for each survivor
  4. "Fresh breakout" filter (prevents LWLG-at-peak mistakes)
  5. catalyst_agent.classify_catalyst() — must be tradeable
  6. pattern_agent.detect_recovery_pattern() — recovery_score >= 60
  7. Combined score (abs(catalyst_weight) + recovery_score) >= 130
  8. Sort descending, return top 5, fire WhatsApp digest

Public API:
  run_premarket_scan() → list[dict]
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import yfinance as yf
import numpy as np

_ET   = ZoneInfo("America/New_York")
_DATA = Path(__file__).parent.parent / "data"
_SEP  = "─" * 30

# ── Thresholds ─────────────────────────────────────────────────────────────────

MIN_GAP_PCT        = 5.0     # minimum gap from prev close
MAX_GAP_PCT        = 20.0    # over-extended (likely fade)
MIN_PREMARKET_VOL  = 200_000 # share volume before market open
MIN_PRICE          = 3.0
MAX_PRICE          = 200.0
COMBINED_THRESHOLD = 130     # abs(catalyst_weight) + recovery_score
MAX_WORKERS        = 10

# Fresh-breakout guard
FRESH_RSI_CEIL      = 50     # RSI yesterday must be below this (not already hot)
MAX_EXTENSION_PCT   = 90     # price must be < 90% above 30d low (not already extended)


# ── Step 1 — Fetch gainers ─────────────────────────────────────────────────────

def _fetch_polygon_gainers(limit: int = 50) -> list[dict]:
    """
    Fetch top pre-market gainers from Polygon snapshot.
    Requires paid Polygon plan — returns [] on free-tier 403.
    """
    from config import POLYGON_API_KEY
    url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers"
    try:
        for attempt in range(3):
            r = requests.get(
                url,
                params={"include_otc": "false", "apiKey": POLYGON_API_KEY},
                timeout=15,
            )
            if r.status_code == 403:
                return []     # free tier — caller will use fallback
            if r.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"⏳ [Scanner] Polygon rate limit — retrying in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            tickers_raw = r.json().get("tickers", [])[:limit]
            result = []
            for t in tickers_raw:
                prev_close = float(t.get("prevDay", {}).get("c") or 0)
                price = float(
                    t.get("lastTrade", {}).get("p")
                    or t.get("min", {}).get("c")
                    or t.get("day", {}).get("c")
                    or 0
                )
                volume = int(
                    t.get("day", {}).get("v")
                    or t.get("min", {}).get("v")
                    or 0
                )
                gap_pct = float(t.get("todaysChangePerc") or 0)
                if prev_close > 0 and price > 0:
                    result.append({
                        "ticker":     t["ticker"],
                        "gap_pct":    round(gap_pct, 2),
                        "price":      round(price, 4),
                        "prev_close": round(prev_close, 4),
                        "volume":     volume,
                    })
            print(f"✅ [Scanner] Polygon gainers: {len(result)} tickers")
            return result
    except Exception as e:
        print(f"⚠️  [Scanner] Polygon gainers error: {e}")
    return []


def _build_fallback_universe() -> list[str]:
    """
    Build a focused ~200-ticker universe when Polygon gainers is unavailable.
    Combines sector catalyst tickers + high-beta universe tickers.
    """
    # Catalyst theme tickers (already curated in discovery_agent)
    catalyst_tickers: list[str] = []
    try:
        from discovery_agent import CATALYST_THEME_UNIVERSE, _get_catalyst_tickers
        catalyst_tickers = _get_catalyst_tickers()
        # Also add all theme tickers even without active catalyst
        for tickers in CATALYST_THEME_UNIVERSE.values():
            catalyst_tickers += tickers
    except Exception:
        pass

    # High-beta / high-vol names that gap often
    _VOLATILE = [
        "NVDA", "AMD", "SMCI", "PLTR", "ARM", "IONQ", "RGTI", "QBTS",
        "HIMS", "RIVN", "LCID", "NIO", "TSLA", "COIN", "MSTR", "MARA",
        "RIOT", "CRWD", "PANW", "ZS", "MRNA", "BNTX", "NVAX", "RCKT",
        "IOVA", "ARQQ", "QUBT", "XNDU", "ASTS", "LUNR", "RKLB", "JOBY",
        "ACHR", "AFRM", "UPST", "SOFI", "SQ", "PYPL", "NU", "DAVE",
        "FSLR", "ENPH", "PLUG", "BE", "CLSK", "HUT", "BTBT", "SOXS",
        "SOXL", "TQQQ", "LABU", "SPXU", "UVXY",
    ]

    # Top tickers from static universe
    universe_tickers: list[str] = []
    try:
        import json
        data = json.loads(
            (Path(__file__).parent.parent / "ticker_universe.json").read_text()
        )
        universe_tickers = data.get("tickers", [])[:200]
    except Exception:
        pass

    seen: set = set()
    out: list = []
    for t in catalyst_tickers + _VOLATILE + universe_tickers:
        t = t.upper()
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:250]


def _premarket_data_yf(ticker: str) -> dict | None:
    """
    Fetch pre-market price, gap, and volume for one ticker via yfinance.
    Returns None if it doesn't pass the initial gap/price/volume filter.
    """
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="1m", prepost=True)
        if hist is None or hist.empty:
            return None

        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC").tz_convert(_ET)
        else:
            hist.index = hist.index.tz_convert(_ET)

        pre = hist[hist.index.time < dtime(9, 30)]
        if pre.empty:
            return None

        try:
            prev_close = float(t.fast_info.previous_close or 0)
        except Exception:
            return None
        if prev_close <= 0:
            return None

        price  = float(pre["Close"].iloc[-1])
        volume = int(pre["Volume"].sum())
        if price <= 0:
            return None

        gap_pct = (price - prev_close) / prev_close * 100

        return {
            "ticker":     ticker,
            "gap_pct":    round(gap_pct, 2),
            "price":      round(price, 4),
            "prev_close": round(prev_close, 4),
            "volume":     volume,
        }
    except Exception:
        return None


def _fetch_candidates() -> list[dict]:
    """
    Return raw gap candidates from Polygon (paid) or yfinance scan (free fallback).
    """
    # Try Polygon first
    poly = _fetch_polygon_gainers()
    if poly:
        return poly

    # Fallback: parallel yfinance scan
    print("📡 [Scanner] Polygon gainers unavailable — scanning universe via yfinance...")
    universe = _build_fallback_universe()
    print(f"   Scanning {len(universe)} tickers...")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(_premarket_data_yf, t): t for t in universe}
        for fut in as_completed(futures):
            hit = fut.result()
            if hit is not None:
                results.append(hit)
    return results


# ── Step 2 — Gap / volume / price / earnings filters ──────────────────────────

def _apply_basic_filters(candidates: list[dict]) -> list[dict]:
    """Gap 5–20%, volume > 200k, price $3–$200."""
    passed = []
    for c in candidates:
        gap = c.get("gap_pct", 0)
        vol = c.get("volume", 0)
        px  = c.get("price", 0)
        if not (MIN_GAP_PCT <= gap <= MAX_GAP_PCT):
            continue
        if vol < MIN_PREMARKET_VOL:
            continue
        if not (MIN_PRICE <= px <= MAX_PRICE):
            continue
        passed.append(c)
    return passed


def _apply_earnings_filter(candidates: list[dict]) -> list[dict]:
    """Remove stocks with earnings in the next 10 days."""
    try:
        from utils.earnings_gate import check_earnings_blackout
    except ImportError:
        return candidates

    passed = []
    for c in candidates:
        try:
            eb = check_earnings_blackout(c["ticker"])
            if eb["blocked"]:
                print(
                    f"📅 [Scanner] {c['ticker']} blocked — "
                    f"earnings in {eb['days_until']}d"
                )
                continue
        except Exception:
            pass
        passed.append(c)
    return passed


# ── Step 3 — 30-day OHLCV fetch ───────────────────────────────────────────────

def _fetch_30d_bars(ticker: str) -> list[dict]:
    """
    Fetch 35 days of daily OHLCV via yfinance.
    Returns list of {h, l, c, v} dicts (oldest first).
    """
    try:
        df = yf.Ticker(ticker).history(period="35d", interval="1d")
        if df.empty or len(df) < 5:
            return []
        return [
            {"h": float(r.High), "l": float(r.Low),
             "c": float(r.Close), "v": float(r.Volume)}
            for _, r in df.iterrows()
        ]
    except Exception:
        return []


# ── Step 4 — Fresh breakout filter ────────────────────────────────────────────

def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 2)


def _passes_fresh_breakout(
    candidate: dict,
    bars: list[dict],
) -> tuple[bool, str]:
    """
    Prevents buying stocks that are already extended (LWLG-at-peak mistake).

    Rules:
      1. Today's gap is the move (5–20% — already filtered upstream)
      2. RSI as of yesterday's close must be < 50 (not already hot)
      3. Today's volume spike (pre-market vol > 2× 30d avg daily vol)
      4. Price must be < 90% above the 30-day low (fresh breakout zone)

    Returns (passes, rejection_reason).
    """
    if len(bars) < 5:
        return True, ""   # not enough history — let through

    closes  = np.array([b["c"] for b in bars], dtype=float)
    volumes = np.array([b["v"] for b in bars], dtype=float)
    lows    = np.array([b["l"] for b in bars], dtype=float)

    price = candidate["price"]
    today_vol = candidate["volume"]

    # Rule 2: RSI yesterday (exclude today's candle if it's already included)
    # bars[-1] might be today's partial; use bars[:-1] to get "yesterday's RSI"
    rsi_yesterday = _calc_rsi(closes[:-1]) if len(closes) >= 3 else 50.0
    if rsi_yesterday >= FRESH_RSI_CEIL:
        return False, f"RSI yesterday {rsi_yesterday:.0f} ≥ {FRESH_RSI_CEIL} (already trending)"

    # Rule 3: volume spike is TODAY, not an artifact
    avg_daily_vol = float(np.mean(volumes[:-1])) if len(volumes) > 1 else 0
    if avg_daily_vol > 0:
        rvol = today_vol / avg_daily_vol
        if rvol < 1.5:
            return False, f"Today's RVOL {rvol:.1f}x — no meaningful volume spike"

    # Rule 4: price not already 90%+ above 30d low
    valid_lows = lows[lows > 0]
    if len(valid_lows) > 0:
        low_30d    = float(np.min(valid_lows))
        extension  = (price / low_30d - 1) * 100 if low_30d > 0 else 0
        if extension >= MAX_EXTENSION_PCT:
            return False, (
                f"Price ${price:.2f} is {extension:.0f}% above 30d low "
                f"${low_30d:.2f} — already extended, not a fresh breakout"
            )

    return True, ""


# ── Step 5–6 — Catalyst + pattern scoring ─────────────────────────────────────

def _fetch_news(ticker: str) -> list[dict]:
    try:
        from polygon_feed import get_news
        return get_news(ticker, limit=5)
    except Exception:
        return []


def _score_candidate(
    candidate: dict,
    bars: list[dict],
    news: list[dict],
) -> dict | None:
    """
    Run catalyst_agent + pattern_agent. Return enriched candidate or None
    if it doesn't meet the combined score threshold.
    """
    from agents.catalyst_agent  import classify_catalyst
    from agents.pattern_agent   import detect_recovery_pattern

    ticker = candidate["ticker"]
    price  = candidate["price"]

    # avg volume for RVOL
    if bars:
        volumes   = np.array([b["v"] for b in bars], dtype=float)
        avg_vol   = float(np.mean(volumes[:-1])) if len(volumes) > 1 else 0
        rvol      = candidate["volume"] / avg_vol if avg_vol > 0 else 1.0
    else:
        rvol = 1.0

    # ── Catalyst agent ────────────────────────────────────────────────────────
    catalyst = classify_catalyst(ticker, news)
    if not catalyst.get("is_tradeable"):
        print(
            f"   ⛔ {ticker} — catalyst not tradeable "
            f"({catalyst.get('catalyst_type')} w={catalyst.get('catalyst_weight')})"
        )
        return None

    cat_weight = abs(catalyst.get("catalyst_weight", 0))

    # ── Pattern agent ─────────────────────────────────────────────────────────
    pattern = detect_recovery_pattern(
        ticker, price, bars,
        rel_vol=rvol,
        gap_up_pct=candidate["gap_pct"],
    )
    rec_score = pattern.get("recovery_score", 0)

    if rec_score < 60:
        print(
            f"   ⛔ {ticker} — recovery_score {rec_score} < 60 "
            f"(pattern={pattern.get('pattern_type')})"
        )
        return None

    combined = cat_weight + rec_score
    if combined < COMBINED_THRESHOLD:
        print(
            f"   ⛔ {ticker} — combined {combined} < {COMBINED_THRESHOLD} "
            f"(cat={cat_weight} + rec={rec_score})"
        )
        return None

    # ── Build enriched result ─────────────────────────────────────────────────
    result = {
        **candidate,
        # Catalyst fields
        "catalyst_type":    catalyst["catalyst_type"],
        "catalyst_weight":  catalyst["catalyst_weight"],
        "catalyst_summary": catalyst.get("summary", ""),
        "catalyst_direction": catalyst.get("direction", "neutral"),
        # Pattern fields
        "recovery_score":   rec_score,
        "exhaustion_score": pattern.get("exhaustion_score", 0),
        "pattern_type":     pattern.get("pattern_type", "NONE"),
        "pattern_reasoning": pattern.get("reasoning", []),
        # Levels from pattern_agent
        "entry_zone":   pattern.get("entry_zone", [round(price * 0.99, 2), round(price * 1.01, 2)]),
        "target_price": pattern.get("target_price", round(price * 1.10, 2)),
        "stop_loss":    pattern.get("stop_loss",    round(price * 0.93, 2)),
        "hold_days":    pattern.get("hold_days", 2),
        # Combined score
        "combined_score": combined,
    }

    print(
        f"   ✅ {ticker} — combined={combined} "
        f"cat={catalyst['catalyst_type']}({catalyst['catalyst_weight']:+d}) "
        f"rec={rec_score} "
        f"gap={candidate['gap_pct']:+.1f}%"
    )
    return result


# ── Step 7 — WhatsApp digest ───────────────────────────────────────────────────

def _format_digest(candidates: list[dict]) -> str:
    """Format a compact WhatsApp digest for the top 5 setups."""
    now_et  = datetime.now(tz=_ET)
    date_str = now_et.strftime("%a %b %-d")
    mins_to_open = max(0, (9 * 60 + 30) - (now_et.hour * 60 + now_et.minute))

    lines = [
        f"🌅 CATALYST SCAN — 7:45 AM ET ({date_str})",
        _SEP,
        f"{len(candidates)} setup{'s' if len(candidates) != 1 else ''} "
        f"(combined score ≥ {COMBINED_THRESHOLD})",
        "",
    ]

    cat_icons = {
        "REGULATORY":        "💊",
        "PARTNERSHIP":       "🤝",
        "ANALYST_UPGRADE":   "📈",
        "ANALYST_DOWNGRADE": "📉",
        "LAWSUIT_FRAUD":     "⚠️",
        "CEO_DEPARTURE":     "🚪",
        "EARNINGS":          "📊",
        "GENERAL":           "📰",
    }

    for i, c in enumerate(candidates, 1):
        icon     = cat_icons.get(c.get("catalyst_type", "GENERAL"), "📰")
        ez       = c.get("entry_zone", [c["price"], c["price"]])
        target   = c.get("target_price", 0)
        stop     = c.get("stop_loss", 0)
        vol_k    = c["volume"] / 1_000
        vol_str  = f"{vol_k:.0f}k" if vol_k < 1000 else f"{vol_k/1000:.1f}M"

        entry_mid = (ez[0] + ez[1]) / 2 if ez else c["price"]
        t1_pct    = (target - entry_mid) / entry_mid * 100 if target and entry_mid else 0
        stop_pct  = (stop   - entry_mid) / entry_mid * 100 if stop   and entry_mid else 0

        lines += [
            f"{i}. {c['ticker']}  {c['gap_pct']:+.1f}%  vol {vol_str}  score={c['combined_score']}",
            f"   {icon} {c.get('catalyst_type', 'GENERAL')} ({c.get('catalyst_weight', 0):+d})"
            f"  |  Recovery: {c.get('recovery_score', 0)}",
            f"   Entry ${ez[0]:.2f}–${ez[1]:.2f}"
            + (f"  →  T1 ${target:.2f} ({t1_pct:+.0f}%)" if target else "")
            + (f"  Stop ${stop:.2f} ({stop_pct:.0f}%)" if stop else ""),
        ]
        summary = c.get("catalyst_summary", "")
        if summary:
            lines.append(f"   📰 {summary[:80]}")
        lines.append("")

    lines += [
        _SEP,
        f"⏰ {mins_to_open}min to open — confirm price action at the bell",
    ]
    return "\n".join(lines)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_premarket_scan(*, dry_run: bool = False) -> list[dict]:
    """
    Full pre-market catalyst + recovery scanner.

    Parameters
    ----------
    dry_run : if True, print the WhatsApp message but don't send it

    Returns
    -------
    list of up to 5 enriched candidate dicts sorted by combined_score desc
    """
    now_et = datetime.now(tz=_ET)
    print(f"\n{_SEP}")
    print(f"🌅 [CatalystScanner] {now_et.strftime('%H:%M:%S ET')} — starting scan")

    # ── Step 1: Fetch gainers ─────────────────────────────────────────────────
    raw = _fetch_candidates()
    print(f"   Step 1: {len(raw)} raw candidates from source")

    # ── Step 2: Basic filters ─────────────────────────────────────────────────
    filtered = _apply_basic_filters(raw)
    print(f"   Step 2: {len(filtered)} pass gap/vol/price filter "
          f"({MIN_GAP_PCT}–{MAX_GAP_PCT}% gap, >{MIN_PREMARKET_VOL//1000}k vol, "
          f"${MIN_PRICE}–${MAX_PRICE})")

    if not filtered:
        print("💤 [CatalystScanner] No candidates pass basic filters.")
        return []

    # ── Step 2b: Earnings gate ────────────────────────────────────────────────
    filtered = _apply_earnings_filter(filtered)
    print(f"   Step 2b: {len(filtered)} pass earnings gate")

    if not filtered:
        print("💤 [CatalystScanner] All blocked by earnings gate.")
        return []

    # ── Step 3: Fetch 30d OHLCV + news in parallel ───────────────────────────
    print(f"   Step 3: Fetching 30d bars + news for {len(filtered)} candidates...")

    bars_map: dict[str, list] = {}
    news_map: dict[str, list] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        bar_futs  = {exe.submit(_fetch_30d_bars, c["ticker"]): c["ticker"] for c in filtered}
        news_futs = {exe.submit(_fetch_news, c["ticker"]): c["ticker"]     for c in filtered}

        for fut in as_completed(bar_futs):
            t = bar_futs[fut]
            try:
                bars_map[t] = fut.result()
            except Exception:
                bars_map[t] = []

        for fut in as_completed(news_futs):
            t = news_futs[fut]
            try:
                news_map[t] = fut.result()
            except Exception:
                news_map[t] = []

    # ── Step 4: Fresh breakout filter ────────────────────────────────────────
    fresh: list[dict] = []
    for c in filtered:
        bars = bars_map.get(c["ticker"], [])
        passes, reason = _passes_fresh_breakout(c, bars)
        if passes:
            fresh.append(c)
        else:
            print(f"   🚫 {c['ticker']} — fresh-breakout filter: {reason}")

    print(f"   Step 4: {len(fresh)} pass fresh-breakout filter")

    if not fresh:
        print("💤 [CatalystScanner] All filtered by fresh-breakout check.")
        return []

    # ── Steps 5–6: Catalyst + pattern scoring ────────────────────────────────
    print(f"   Steps 5–6: Scoring {len(fresh)} candidates with catalyst+pattern agents...")
    scored: list[dict] = []
    for c in fresh:
        try:
            result = _score_candidate(
                c,
                bars_map.get(c["ticker"], []),
                news_map.get(c["ticker"], []),
            )
            if result is not None:
                scored.append(result)
        except Exception as e:
            print(f"   ⚠️  {c['ticker']} scoring error: {e}")

    if not scored:
        print("💤 [CatalystScanner] No candidates meet combined score threshold.")
        return []

    # ── Step 7: Sort + top 5 ─────────────────────────────────────────────────
    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    top5 = scored[:5]

    print(f"\n   ✅ Top {len(top5)} catalyst setups:")
    for c in top5:
        print(
            f"      {c['ticker']:8s} combined={c['combined_score']:3d}  "
            f"cat={c['catalyst_type']}({c['catalyst_weight']:+d})  "
            f"rec={c['recovery_score']}  gap={c['gap_pct']:+.1f}%"
        )

    # ── Step 8: WhatsApp digest ───────────────────────────────────────────────
    msg = _format_digest(top5)

    if dry_run:
        print(f"\n📋 [DRY RUN] Would send:\n{msg}")
    else:
        try:
            from alerts import send_whatsapp
            sent = send_whatsapp(msg)
            if sent:
                print("✅ [CatalystScanner] WhatsApp digest sent")
            else:
                print("⚠️  [CatalystScanner] WhatsApp not confirmed")
        except Exception as e:
            print(f"⚠️  [CatalystScanner] WhatsApp error: {e}")

    print(f"{_SEP}\n")
    return top5


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv or "--test" in sys.argv
    results = run_premarket_scan(dry_run=dry)
    if results:
        print(f"\nReturned {len(results)} setup(s):")
        for r in results:
            print(f"  {r['ticker']:8s}  combined={r['combined_score']}  "
                  f"entry={r['entry_zone']}  target={r['target_price']}  "
                  f"stop={r['stop_loss']}")
    else:
        print("No setups found.")
