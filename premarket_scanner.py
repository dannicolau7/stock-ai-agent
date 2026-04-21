"""
premarket_scanner.py — Full pre-market opportunity scanner.

Pipeline (4:00 AM – 9:30 AM ET weekdays):
  Step 1 — Build candidate list (watchlist + movers + universe)
  Step 2 — Filter (gap ≥ 2% OR catalyst news OR EDGAR filing)
  Step 3 — Score each survivor (0–100)
  Step 4 — Claude analysis on top 10 (BUY_AT_OPEN / WATCH / AVOID)
  Step 5 — Save prep_alert_list.json
  Step 6 — Send WhatsApp per BUY/WATCH

CLI args:
  --prep-alert       Run full scan → save list → alert BUY/WATCH   (9:10 AM)
  --confirmation     Re-check prep list → STILL BUY/WEAKENED/STAND DOWN  (9:25 AM)
  --morning-brief    8:00 AM digest (movers + macro + watchlist)
  --quick-check      Watchlist-only quick scan
  --broad-scan       Watchlist + top 150 universe tickers
  --full-scan        Entire universe
  --test             Dry-run (no WhatsApp sent)
  --verbose / -v     Show all details

Backward-compatible exports (used by scheduler.py):
  scan_premarket_gaps(tickers)   → list of gap dicts
  format_premarket_msg(gaps)     → str
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dtime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import yfinance as yf

_ET = ZoneInfo("America/New_York")

# ── File paths ─────────────────────────────────────────────────────────────────
_DIR      = os.path.dirname(os.path.abspath(__file__))
PREP_LIST = os.path.join(_DIR, "data", "prep_alert_list.json")
TODAY_WL  = os.path.join(_DIR, "data", "todays_watchlist.json")

# ── Thresholds ─────────────────────────────────────────────────────────────────
MIN_GAP_PCT         = 2.0       # % gap to qualify
MIN_PREMARKET_VOL   = 30_000    # share volume
MIN_PRICE           = 0.50
MAX_PRICE           = 150.0
MAX_WORKERS         = 12
SCORE_BUY           = 60        # min score for BUY_AT_OPEN alert
SCORE_WATCH         = 40        # min score for WATCH alert
MIN_RVOL            = 1.5       # relative volume floor — below this = hard block
HIGH_RVOL_EXCEPTION = 3.0       # if RVOL > this AND news catalyst, allow regardless

_SEP         = "─" * 30
_BLOCKED_LOG = os.path.join(_DIR, "data", "blocked_signals.csv")


# ── Volume filter helpers ──────────────────────────────────────────────────────

def _compute_rvol(premarket_vol: int, avg_daily_vol: float) -> float:
    """
    Relative volume vs expected pre-market baseline.
    Normalises against avg_daily / 6.5 (one hour of typical trading).
    Returns 0.0 when avg volume is unavailable.
    """
    if avg_daily_vol <= 0:
        return 0.0
    return premarket_vol / (avg_daily_vol / 6.5)


def _has_news_catalyst(c: Dict) -> bool:
    return bool(
        c.get("news_headline")
        or c.get("has_edgar")
        or c.get("is_upgrade")
        or c.get("is_earnings_beat")
        or c.get("is_contract")
    )


def _passes_rvol_filter(c: Dict) -> bool:
    """True if candidate should proceed to Claude analysis."""
    rvol = c.get("rvol", 0.0)
    if rvol == 0.0:
        return True   # unknown — can't filter, let it through
    if rvol >= MIN_RVOL:
        return True
    if rvol > HIGH_RVOL_EXCEPTION and _has_news_catalyst(c):
        return True   # unusual volume IS the signal
    return False


_BLOCKED_LOG_FIELDS = [
    "ts", "ticker", "date_blocked", "earnings_date",
    "days_until_earnings", "original_signal", "rvol", "reason", "source",
]


def _log_blocked(ticker: str, rvol: float, reason: str) -> None:
    """Append a volume-blocked row to data/blocked_signals.csv."""
    import csv
    try:
        os.makedirs(os.path.dirname(_BLOCKED_LOG), exist_ok=True)
        write_header = not os.path.exists(_BLOCKED_LOG)
        with open(_BLOCKED_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_BLOCKED_LOG_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({
                "ts":                  datetime.now().isoformat(),
                "ticker":              ticker,
                "date_blocked":        datetime.now().date().isoformat(),
                "earnings_date":       "",
                "days_until_earnings": "",
                "original_signal":     "",
                "rvol":                round(rvol, 2),
                "reason":              reason,
                "source":              "premarket_scanner",
            })
    except Exception as e:
        print(f"⚠️  [PremarketScanner] blocked_signals log failed: {e}")


# ── Step 2 — Fetch pre-market data ─────────────────────────────────────────────

def get_premarket_data(ticker: str) -> Optional[Dict]:
    """
    Fetch pre-market OHLCV for one ticker via yfinance 1-min bars (prepost=True).
    Returns a data dict if it passes the gap + price filter; None otherwise.
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
            prev_close = float(t.fast_info["previous_close"])
        except Exception:
            return None
        if prev_close <= 0:
            return None

        premarket_price = float(pre["Close"].iloc[-1])
        if premarket_price <= 0:
            return None

        premarket_vol  = int(pre["Volume"].sum())
        premarket_high = float(pre["High"].max())
        premarket_low  = float(pre["Low"].min())
        gap_pct        = (premarket_price - prev_close) / prev_close * 100

        if not (MIN_PRICE <= premarket_price <= MAX_PRICE):
            return None

        try:
            avg_vol = float(t.fast_info.get("three_month_average_volume") or 0)
        except Exception:
            avg_vol = 0.0

        return {
            "ticker":          ticker,
            "gap_pct":         round(gap_pct, 2),
            "premarket_price": round(premarket_price, 4),
            "premarket_high":  round(premarket_high, 4),
            "premarket_low":   round(premarket_low, 4),
            "prev_close":      round(prev_close, 4),
            "premarket_vol":   premarket_vol,
            "avg_vol":         int(avg_vol),
            "rvol":            round(_compute_rvol(premarket_vol, avg_vol), 2),
            # Enrichment fields (filled later)
            "news_headline":    "",
            "has_edgar":        False,
            "is_upgrade":       False,
            "is_earnings_beat": False,
            "is_contract":      False,
            "rsi":              50,
            "score":            0,
        }
    except Exception:
        return None


# ── Step 2 — News / EDGAR enrichment ──────────────────────────────────────────

def _enrich_with_polygon(candidate: Dict) -> Dict:
    """Pull latest news from Polygon; set has_edgar, is_upgrade, is_contract."""
    ticker = candidate["ticker"]
    try:
        from polygon_feed import get_news
        news = get_news(ticker, limit=3)
        if news:
            headline = news[0].get("title", "")
            candidate["news_headline"] = headline[:120]
            hl = headline.lower()
            candidate["is_upgrade"]       = any(w in hl for w in ["upgrade", "outperform", "buy rating", "initiated"])
            candidate["is_earnings_beat"] = any(w in hl for w in ["beats", "beat", "exceeded", "raised guidance", "earnings beat"])
            candidate["is_contract"]      = any(w in hl for w in ["contract", "award", "partnership", "deal", "agreement"])
            candidate["has_edgar"]        = any(
                "8-k" in a.get("title", "").lower() or
                "edgar" in str(a.get("publisher", {}).get("name", "")).lower()
                for a in news
            )
    except Exception:
        pass
    return candidate


# ── Step 3 — Scoring ───────────────────────────────────────────────────────────

def _score_candidate(c: Dict) -> int:
    """Score a pre-market candidate 0–100."""
    score = 0
    gap   = abs(c.get("gap_pct", 0))
    vol   = c.get("premarket_vol", 0)
    rsi   = c.get("rsi", 50)

    # Gap + catalyst base score
    if gap > 10 and c.get("news_headline"):
        score += 70
    elif gap > 5 and c.get("news_headline"):
        score += 55
    elif gap >= 2 and c.get("news_headline"):
        score += 40
    elif gap >= 2:
        score += 20

    if c.get("has_edgar"):
        score = max(score, 65)

    # Volume bonus
    if vol > 500_000:
        score += 30
    elif vol > 100_000:
        score += 20
    elif vol > 50_000:
        score += 10

    # RSI context (from previous close)
    if rsi < 35:
        score += 15
    elif rsi > 70:
        score -= 10

    # Catalyst bonuses
    if c.get("is_upgrade"):
        score += 20
    if c.get("is_earnings_beat"):
        score += 25
    if c.get("is_contract"):
        score += 20

    return min(max(score, 0), 100)


# ── Step 4 — Claude analysis ───────────────────────────────────────────────────

def _claude_analyze(candidates: List[Dict]) -> List[Dict]:
    """
    Send top candidates to Claude (Haiku).
    Returns candidates enriched with verdict + entry/target/stop.
    """
    if not candidates:
        return []

    try:
        import anthropic
        client = anthropic.Anthropic()

        now_et = datetime.now(tz=_ET)
        mins_to_open = max(0, (9 * 60 + 30) - (now_et.hour * 60 + now_et.minute))

        lines = []
        for c in candidates:
            vol_str = (f"{c['premarket_vol'] / 1_000:.0f}k"
                       if c["premarket_vol"] < 1_000_000
                       else f"{c['premarket_vol'] / 1_000_000:.1f}M")
            lines.append(
                f"- {c['ticker']}: gap={c['gap_pct']:+.1f}% "
                f"price=${c['premarket_price']:.2f} "
                f"prevClose=${c['prev_close']:.2f} "
                f"preVol={vol_str} "
                f"avgVol={c['avg_vol'] / 1_000:.0f}k "
                f"score={c['score']} "
                f"news=\"{c.get('news_headline', '')[:80]}\""
            )

        prompt = f"""You are a pre-market stock analyst. Market opens in {mins_to_open} minutes.

Analyze these pre-market movers and give a verdict for each:

{chr(10).join(lines)}

For each ticker determine:
1. Is this a SUSTAINABLE move (real catalyst, volume confirms, likely to continue at open)?
2. Or a FADE (gap-and-dump, no real catalyst, likely to reverse)?

Return a JSON array (one object per ticker):
[
  {{
    "ticker": "XXXX",
    "verdict": "BUY_AT_OPEN" | "WATCH" | "AVOID",
    "entry_low": <float>,
    "entry_high": <float>,
    "target_1": <float>,
    "target_2": <float>,
    "stop_loss": <float>,
    "reasoning": "<2-3 sentences>",
    "main_risk": "<1 sentence>",
    "conviction": <integer 0-100>
  }}
]

Rules:
- BUY_AT_OPEN: Strong catalyst, vol > 100k, gap sustainable (not extended > 20% without massive news)
- WATCH: Moderate catalyst or volume — wait for price action confirmation at the open
- AVOID: No clear catalyst, low volume, over-extended, likely to fade
- entry_low/high: slightly above prev_close for upside gaps (want confirmation, not chasing)
- stop_loss: below pre-market low
- Targets: realistic based on gap size + typical morning momentum

Return ONLY the JSON array, no other text."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        text  = response.content[0].text.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            print("⚠️  [PremarketScanner] Claude returned no JSON array")
            return _fallback_verdicts(candidates)

        verdicts    = json.loads(match.group())
        verdict_map = {v["ticker"]: v for v in verdicts if isinstance(v, dict)}

        for c in candidates:
            v = verdict_map.get(c["ticker"], {})
            c["verdict"]    = v.get("verdict", "WATCH")
            c["entry_low"]  = v.get("entry_low",  c["premarket_price"] * 0.99)
            c["entry_high"] = v.get("entry_high", c["premarket_price"] * 1.01)
            c["target_1"]   = v.get("target_1", 0.0)
            c["target_2"]   = v.get("target_2", 0.0)
            c["stop_loss"]  = v.get("stop_loss", c.get("premarket_low", c["premarket_price"]) * 0.97)
            c["reasoning"]  = v.get("reasoning", "")
            c["main_risk"]  = v.get("main_risk", "")
            c["conviction"] = v.get("conviction", 50)

        return candidates

    except Exception as e:
        print(f"⚠️  [PremarketScanner] Claude analysis failed: {e}")
        return _fallback_verdicts(candidates)


def _fallback_verdicts(candidates: List[Dict]) -> List[Dict]:
    """Score-based fallback when Claude is unavailable."""
    for c in candidates:
        c["verdict"]    = "WATCH" if c["score"] >= SCORE_WATCH else "AVOID"
        c["entry_low"]  = c["premarket_price"] * 0.99
        c["entry_high"] = c["premarket_price"] * 1.01
        c["target_1"]   = c["premarket_price"] * 1.05
        c["target_2"]   = c["premarket_price"] * 1.10
        c["stop_loss"]  = c.get("premarket_low", c["premarket_price"] * 0.95)
        c["reasoning"]  = "Score-based estimate (Claude unavailable)"
        c["main_risk"]  = "Verify catalyst before entry"
        c["conviction"] = c["score"]
    return candidates


# ── Alert formatting ───────────────────────────────────────────────────────────

def _format_premarket_alert(c: Dict) -> str:
    """Format a rich pre-market WhatsApp alert for one ticker."""
    now_et       = datetime.now(tz=_ET)
    mins_to_open = max(0, (9 * 60 + 30) - (now_et.hour * 60 + now_et.minute))

    vol_str = (f"{c['premarket_vol'] / 1_000:.0f}k"
               if c["premarket_vol"] < 1_000_000
               else f"{c['premarket_vol'] / 1_000_000:.1f}M")
    avg_str = (f"{c['avg_vol'] / 1_000:.0f}k"
               if c.get("avg_vol", 0) < 1_000_000 and c.get("avg_vol", 0) > 0
               else f"{c.get('avg_vol', 0) / 1_000_000:.1f}M") if c.get("avg_vol") else "—"

    verdict = c.get("verdict", "WATCH")
    emoji   = "🟢" if verdict == "BUY_AT_OPEN" else ("👁️" if verdict == "WATCH" else "🚫")
    label   = verdict.replace("_", " ")

    entry_low  = c.get("entry_low",  c["premarket_price"])
    entry_high = c.get("entry_high", c["premarket_price"])
    t1         = c.get("target_1", 0.0)
    t2         = c.get("target_2", 0.0)
    stop       = c.get("stop_loss", 0.0)
    entry_mid  = (entry_low + entry_high) / 2 if entry_low and entry_high else c["premarket_price"]

    t1_pct   = (t1 - entry_mid) / entry_mid * 100   if entry_mid > 0 and t1   > 0 else 0
    stop_pct = (stop - entry_mid) / entry_mid * 100 if entry_mid > 0 and stop > 0 else 0

    lines = [
        f"🌅 PRE-MARKET {emoji} {label} — {c['ticker']}",
        _SEP,
        f"Gap: {c['gap_pct']:+.1f}%  |  Price: ${c['premarket_price']:.2f}",
        f"Pre-Vol: {vol_str}  |  Avg: {avg_str}  |  Score: {c['score']}/100",
    ]
    if c.get("news_headline"):
        lines.append(f"📰 {c['news_headline'][:90]}")
    lines += [
        _SEP,
        f"Entry:  ${entry_low:.2f} – ${entry_high:.2f}",
    ]
    if t1 > 0:
        t_line = f"T1: ${t1:.2f} (+{t1_pct:.1f}%)"
        if t2 > 0:
            t_line += f"  T2: ${t2:.2f}"
        lines.append(t_line)
    if stop > 0:
        lines.append(f"Stop: ${stop:.2f} ({stop_pct:.1f}%)")
    lines += [
        _SEP,
        c.get("reasoning", "")[:200],
    ]
    if c.get("main_risk"):
        lines.append(f"⚠️  Risk: {c['main_risk'][:100]}")
    lines.append(f"⏰ {mins_to_open}min to open")

    return "\n".join(l for l in lines if l is not None)


# ── Persistence helpers ────────────────────────────────────────────────────────

def _save_prep_list(candidates: List[Dict]):
    """Save prep alert list to data/prep_alert_list.json."""
    os.makedirs(os.path.dirname(PREP_LIST), exist_ok=True)
    payload = {
        "saved_at": datetime.now(tz=_ET).isoformat(),
        "date":     datetime.now(tz=_ET).strftime("%Y-%m-%d"),
        "items":    candidates,
    }
    with open(PREP_LIST, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"💾 [PremarketScanner] Prep list saved → {len(candidates)} items")


def _load_prep_list() -> List[Dict]:
    """Load today's prep list; returns [] if missing or from a different day."""
    try:
        if not os.path.exists(PREP_LIST):
            return []
        with open(PREP_LIST) as f:
            data = json.load(f)
        today = datetime.now(tz=_ET).strftime("%Y-%m-%d")
        if data.get("date") != today:
            print(f"⚠️  [PremarketScanner] Prep list is from {data.get('date')}, not today — ignoring")
            return []
        return data.get("items", [])
    except Exception as e:
        print(f"⚠️  [PremarketScanner] Could not load prep list: {e}")
        return []


def _update_todays_watchlist(ticker: str, verdict: str, remove: bool = False):
    """Add / update / remove a ticker in data/todays_watchlist.json."""
    os.makedirs(os.path.dirname(TODAY_WL), exist_ok=True)
    today = datetime.now(tz=_ET).strftime("%Y-%m-%d")

    data: dict = {"date": today, "items": {}}
    if os.path.exists(TODAY_WL):
        try:
            with open(TODAY_WL) as f:
                data = json.load(f)
            if data.get("date") != today:
                data = {"date": today, "items": {}}
        except Exception:
            pass

    if remove:
        data["items"].pop(ticker, None)
    else:
        data["items"][ticker] = {
            "verdict":  verdict,
            "added_at": datetime.now(tz=_ET).isoformat(),
        }

    with open(TODAY_WL, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_todays_watchlist() -> List[str]:
    """Return today's BUY/WATCH tickers from data/todays_watchlist.json."""
    try:
        if not os.path.exists(TODAY_WL):
            return []
        today = datetime.now(tz=_ET).strftime("%Y-%m-%d")
        with open(TODAY_WL) as f:
            data = json.load(f)
        if data.get("date") != today:
            return []
        return list(data.get("items", {}).keys())
    except Exception:
        return []


# ── Step 1 — Candidate list ────────────────────────────────────────────────────

def _build_candidate_list(mode: str = "broad") -> List[str]:
    """
    Build the ticker list to scan.
    mode: 'quick' (watchlist only) | 'broad' (watchlist + 150 universe) | 'full' (all)

    Always prepends yesterday's EOD scanner picks (tomorrow_watchlist.json) so they
    receive priority scoring even if their RVOL is low before market opens.
    """
    import watchlist_manager as wl
    watchlist = wl.load()

    # Yesterday's EOD pre-identified setups — prioritise these
    eod_picks: list = []
    try:
        from eod_scanner import load_tomorrow_tickers
        eod_picks = load_tomorrow_tickers()
        if eod_picks:
            print(f"📅 [PremarketScanner] Loading {len(eod_picks)} EOD picks from yesterday")
    except Exception:
        pass

    if mode == "quick":
        return list(dict.fromkeys(eod_picks + watchlist))

    universe: list = []
    try:
        from market_scanner import _load_universe
        universe = _load_universe() or []
    except Exception:
        pass

    limit = None if mode == "full" else 150
    pool  = universe[:limit] if limit else universe
    return list(dict.fromkeys(eod_picks + watchlist + pool))


# ── Main scan pipeline ─────────────────────────────────────────────────────────

def run_premarket_scan(
    mode:      str  = "broad",
    test:      bool = False,
    verbose:   bool = False,
    save_prep: bool = True,
) -> List[Dict]:
    """
    Full pre-market pipeline. Returns actionable (BUY/WATCH) candidates.
    """
    from alerts import send_whatsapp

    now_et = datetime.now(tz=_ET)
    print(f"\n{_SEP}")
    print(f"🌅 [PremarketScanner] {mode.upper()} scan at {now_et.strftime('%H:%M:%S')} ET")

    # Step 1 — Candidates
    tickers = _build_candidate_list(mode)
    print(f"   Step 1: {len(tickers)} tickers to check")

    # Step 2 — Parallel pre-market fetch + gap filter
    raw: List[Dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(get_premarket_data, t): t for t in tickers}
        for fut in as_completed(futures):
            hit = fut.result()
            if hit and (
                abs(hit["gap_pct"]) >= MIN_GAP_PCT
                or hit["premarket_vol"] >= MIN_PREMARKET_VOL
            ):
                raw.append(hit)

    print(f"   Step 2: {len(raw)} qualified by gap/volume filter")
    if not raw:
        print(f"💤 [PremarketScanner] No pre-market movers today.")
        return []

    # Enrich with Polygon news
    enriched = [_enrich_with_polygon(c) for c in raw]

    # Step 3 — Score
    for c in enriched:
        c["score"] = _score_candidate(c)

    enriched.sort(key=lambda x: x["score"], reverse=True)

    # Step 3b — RVOL hard filter (before Claude to save API calls)
    vol_passed: List[Dict] = []
    for c in enriched:
        if _passes_rvol_filter(c):
            vol_passed.append(c)
        else:
            rvol   = c.get("rvol", 0.0)
            reason = f"Low volume {rvol:.1f}x"
            _log_blocked(c["ticker"], rvol, reason)
            print(f"⛔ [PremarketScanner] {c['ticker']} blocked — {reason}")

    if not vol_passed:
        print(f"💤 [PremarketScanner] All candidates blocked by RVOL filter.")
        return []

    blocked_count = len(enriched) - len(vol_passed)
    if blocked_count:
        print(f"   Step 3b: {blocked_count} candidate(s) removed by RVOL < {MIN_RVOL}x filter")

    # Step 3c — Earnings blackout filter
    from utils.earnings_gate import check_earnings_blackout, log_earnings_block
    earnings_passed: List[Dict] = []
    for c in vol_passed:
        eb = check_earnings_blackout(c["ticker"])
        if eb["blocked"]:
            log_earnings_block(
                ticker=c["ticker"],
                original_signal="premarket_scan",
                days_until=eb["days_until"],
                earnings_date=eb["earnings_date"],
                source="premarket_scanner",
            )
            print(f"📅 [PremarketScanner] {c['ticker']} blocked — earnings in {eb['days_until']}d ({eb['earnings_date']})")
        else:
            earnings_passed.append(c)
            if eb["warning"]:   # no earnings data — log as info, don't block
                print(f"   ℹ️  {c['ticker']}: {eb['warning']}")

    earn_blocked = len(vol_passed) - len(earnings_passed)
    if earn_blocked:
        print(f"   Step 3c: {earn_blocked} candidate(s) removed by earnings blackout filter")

    if not earnings_passed:
        print(f"💤 [PremarketScanner] All candidates blocked by earnings blackout.")
        return []

    top10 = earnings_passed[:10]

    if verbose:
        print(f"\n   Top candidates before Claude:")
        for c in top10:
            print(f"     {c['ticker']:8s}  gap={c['gap_pct']:+.1f}%  "
                  f"vol={c['premarket_vol'] / 1_000:.0f}k  rvol={c.get('rvol', 0):.1f}x  "
                  f"score={c['score']}  news={bool(c.get('news_headline'))}")

    # Step 4 — Claude analysis
    print(f"   Step 4: Claude analysis on {len(top10)} candidates...")
    analyzed = _claude_analyze(top10)

    # Filter actionable
    actionable = [
        c for c in analyzed
        if c.get("verdict") in ("BUY_AT_OPEN", "WATCH")
    ]
    actionable.sort(
        key=lambda x: (x.get("verdict") == "BUY_AT_OPEN", x.get("conviction", 0)),
        reverse=True,
    )

    # Step 5 — Save prep list
    if save_prep:
        _save_prep_list(actionable)

    # Add BUY_AT_OPEN tickers to today's watchlist
    for c in actionable:
        if c.get("verdict") == "BUY_AT_OPEN":
            _update_todays_watchlist(c["ticker"], "BUY_AT_OPEN")

    # Step 6 — Send alerts
    buy_count = watch_count = 0
    for c in actionable:
        verdict = c.get("verdict", "WATCH")
        score   = c.get("score", 0)
        if verdict == "BUY_AT_OPEN" or (verdict == "WATCH" and score >= SCORE_WATCH):
            msg = _format_premarket_alert(c)
            if not test:
                send_whatsapp(msg)
            else:
                print(f"\n📋 [TEST] Would send:\n{msg}")
            if verdict == "BUY_AT_OPEN":
                buy_count += 1
            else:
                watch_count += 1

    print(f"✅ [PremarketScanner] Done — {buy_count} BUY, {watch_count} WATCH alerts")
    print(f"{_SEP}\n")
    return actionable


# ── 9:25 AM — Confirmation check ──────────────────────────────────────────────

def run_confirmation_check(test: bool = False, verbose: bool = False) -> List[Dict]:
    """
    Re-check the 9:10 AM prep list; classify STILL_BUY / WEAKENED / STAND_DOWN / CHASING.
    Sends a single WhatsApp summary.
    """
    from alerts import send_whatsapp

    prep_list = _load_prep_list()
    if not prep_list:
        msg = "⏰ Market opens in 5 min — No pre-market picks to confirm today."
        if not test:
            send_whatsapp(msg)
        else:
            print(f"📋 [TEST] {msg}")
        return []

    now_et = datetime.now(tz=_ET)
    print(f"\n{_SEP}")
    print(f"⏰ [Confirmation] Re-checking {len(prep_list)} picks at {now_et.strftime('%H:%M:%S')} ET")

    confirmed: List[Dict] = []
    for c in prep_list:
        ticker = c["ticker"]
        try:
            t = yf.Ticker(ticker)

            # Latest pre-market price
            hist = t.history(period="1d", interval="1m", prepost=True)
            if not hist.empty:
                if hist.index.tz is None:
                    hist.index = hist.index.tz_localize("UTC").tz_convert(_ET)
                else:
                    hist.index = hist.index.tz_convert(_ET)
                pre = hist[hist.index.time < dtime(9, 30)]
                if not pre.empty:
                    new_price = float(pre["Close"].iloc[-1])
                    new_vol   = int(pre["Volume"].sum())
                else:
                    new_price = c.get("premarket_price", 0)
                    new_vol   = c.get("premarket_vol", 0)
            else:
                new_price = c.get("premarket_price", 0)
                new_vol   = c.get("premarket_vol", 0)

            prep_price = c.get("premarket_price", new_price)
            price_chg  = (new_price - prep_price) / prep_price * 100 if prep_price > 0 else 0

            entry_high = c.get("entry_high", prep_price * 1.03)
            stop_loss  = c.get("stop_loss",  prep_price * 0.95)

            if new_price > entry_high * 1.03:
                status = "CHASING"
                _update_todays_watchlist(ticker, "CHASING")
            elif new_price < stop_loss:
                status = "STAND_DOWN"
                _update_todays_watchlist(ticker, "STAND_DOWN", remove=True)
            elif price_chg < -1.5:
                status = "WEAKENED"
                _update_todays_watchlist(ticker, "WEAKENED")
            else:
                status = "STILL_BUY"
                _update_todays_watchlist(ticker, "STILL_BUY")

            c["confirmation_status"] = status
            c["confirmation_price"]  = new_price
            c["confirmation_vol"]    = new_vol
            confirmed.append(c)

            if verbose:
                print(f"   {ticker}: ${prep_price:.2f} → ${new_price:.2f} "
                      f"({price_chg:+.1f}%) → {status}")

        except Exception as e:
            print(f"⚠️  [Confirmation] Error on {ticker}: {e}")
            c["confirmation_status"] = "UNKNOWN"
            confirmed.append(c)

    # Build summary WhatsApp
    status_labels = {
        "STILL_BUY":  "✅ STILL BUY",
        "WEAKENED":   "🟡 WEAKENED",
        "STAND_DOWN": "🚫 STAND DOWN",
        "CHASING":    "⚠️  CHASING — wait for pullback",
        "UNKNOWN":    "❓ UNKNOWN",
    }

    lines = [f"⏰ Pre-Market Confirmation (9:25 AM ET)\n{_SEP}"]
    for c in confirmed:
        status  = c.get("confirmation_status", "UNKNOWN")
        price   = c.get("confirmation_price", c.get("premarket_price", 0))
        vol     = c.get("confirmation_vol", 0)
        vol_str = f"{vol / 1_000:.0f}k" if vol < 1_000_000 else f"{vol / 1_000_000:.1f}M"
        label   = status_labels.get(status, status)

        lines.append(f"{label} — {c['ticker']} @ ${price:.2f}")
        if status in ("STILL_BUY", "WEAKENED"):
            e_lo = c.get("entry_low",  price * 0.99)
            e_hi = c.get("entry_high", price * 1.01)
            stop = c.get("stop_loss",  0.0)
            lines.append(f"  Entry: ${e_lo:.2f}–${e_hi:.2f}  |  Vol: {vol_str}")
            if stop > 0:
                lines.append(f"  Stop: ${stop:.2f}")

    lines.append(f"\n{_SEP}\n🔔 Market opens in ~5 minutes — stay sharp!")
    msg = "\n".join(lines)

    if not test:
        send_whatsapp(msg)
    else:
        print(f"\n📋 [TEST] Confirmation:\n{msg}")

    print(f"✅ [Confirmation] Sent at {now_et.strftime('%H:%M:%S')}")
    print(f"{_SEP}\n")
    return confirmed


# ── 8:00 AM — Morning digest ───────────────────────────────────────────────────

def send_morning_digest(test: bool = False):
    """
    Quick morning WhatsApp digest: top gap movers, macro context, today's watchlist.
    """
    from alerts import send_whatsapp
    import watchlist_manager as wl

    now_et    = datetime.now(tz=_ET)
    watchlist = wl.load()

    lines = [f"☀️ Argus Morning Digest — {now_et.strftime('%a %b %d, %H:%M ET')}\n{_SEP}"]

    # Macro context
    try:
        import world_context as wctx
        ctx   = wctx.get()
        macro = ctx.get("macro", {})
        vix   = macro.get("vix", 0)
        bias  = macro.get("bias", "?")
        f_spy = macro.get("futures_spy", 0)
        f_qqq = macro.get("futures_qqq", 0)

        parts = [f"📊 Macro: SPY {bias}"]
        if vix:
            parts.append(f"VIX {vix:.1f}")
        if f_spy:
            parts.append(f"SPY fut {f_spy:+.1f}%")
        if f_qqq:
            parts.append(f"QQQ fut {f_qqq:+.1f}%")
        lines.append("  |  ".join(parts))
    except Exception:
        lines.append("📊 Macro: (unavailable)")

    # Quick gap scan of watchlist
    try:
        gaps = scan_premarket_gaps(watchlist[:30])
        if gaps:
            lines.append(f"\n⚡ Pre-Market Movers ({len(gaps)} found):")
            for g in gaps[:5]:
                vol_str = f"{g['premarket_vol'] / 1_000:.0f}k"
                lines.append(f"  {g['ticker']:6s}  {g['gap_pct']:+.1f}%  "
                              f"${g['premarket_price']:.2f}  vol {vol_str}")
        else:
            lines.append("\n⚡ Pre-Market: No significant movers yet")
    except Exception as e:
        lines.append(f"\n⚡ Pre-Market: (scan error — {e})")

    # Today's picks (from last night or earlier this morning)
    today_picks = load_todays_watchlist()
    if today_picks:
        lines.append(f"\n📋 Today's Picks: {', '.join(today_picks)}")
    else:
        lines.append(f"\n📋 Watchlist: {', '.join(watchlist)}")

    lines.append(f"\n{_SEP}\nFull pre-market analysis at 9:10 AM → Watch for entry zones 🎯")

    msg = "\n".join(lines)
    if not test:
        send_whatsapp(msg)
    else:
        print(f"\n📋 [TEST] Morning digest:\n{msg}")

    print(f"✅ [Morning Digest] Done at {now_et.strftime('%H:%M:%S')}")


# ── Backward-compatible API (used by scheduler.py & market_scanner.py) ─────────

def scan_premarket_gaps(tickers: List[str]) -> List[Dict]:
    """
    Simple gap scan — no Claude, no scoring.
    Returns dicts sorted by gap_pct desc.  Backward compat with scheduler.py.
    """
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_premarket_data, t): t for t in tickers}
        for fut in as_completed(futures):
            hit = fut.result()
            if hit and abs(hit["gap_pct"]) >= MIN_GAP_PCT and hit["premarket_vol"] >= MIN_PREMARKET_VOL:
                results.append(hit)
    results.sort(key=lambda x: x["gap_pct"], reverse=True)
    return results


def format_premarket_msg(gaps: List[Dict]) -> str:
    """Backward-compatible formatter for simple gap list WhatsApp message."""
    if not gaps:
        return "⚡ Pre-Market (8:30 AM): No qualifying gap movers today."
    lines = ["⚡ Pre-Market Gaps (8:30 AM ET)"]
    for g in gaps:
        vol = g["premarket_vol"]
        vol_str = f"{vol / 1_000:.0f}k" if vol < 1_000_000 else f"{vol / 1_000_000:.1f}M"
        lines.append(f"{g['ticker']:6s}  {g['gap_pct']:+.1f}%  "
                     f"${g['premarket_price']:.2f}  vol {vol_str}")
    lines.append("💡 Confirm with real volume at the open")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argus Pre-Market Scanner")
    parser.add_argument("--quick-check",   action="store_true", help="Watchlist-only scan")
    parser.add_argument("--broad-scan",    action="store_true", help="Watchlist + universe[:150]")
    parser.add_argument("--full-scan",     action="store_true", help="Entire universe")
    parser.add_argument("--morning-brief", action="store_true", help="8:00 AM morning digest")
    parser.add_argument("--prep-alert",    action="store_true", help="9:10 AM full scan + alert")
    parser.add_argument("--confirmation",  action="store_true", help="9:25 AM confirmation check")
    parser.add_argument("--test",          action="store_true", help="Dry-run (no WhatsApp)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Operating window check (skip for --test)
    now_et  = datetime.now(tz=_ET)
    weekday = now_et.weekday() < 5
    h, m    = now_et.hour, now_et.minute
    in_window = weekday and ((4, 0) <= (h, m) <= (9, 30))

    if not in_window and not args.test:
        print(f"⚠️  Pre-market scanner runs 4:00–9:30 AM ET weekdays. "
              f"Now: {now_et.strftime('%H:%M ET %A')}. Use --test to override.")
        sys.exit(0)

    if args.morning_brief:
        send_morning_digest(test=args.test)
    elif args.confirmation:
        run_confirmation_check(test=args.test, verbose=args.verbose)
    elif args.prep_alert:
        run_premarket_scan(mode="broad", test=args.test, verbose=args.verbose, save_prep=True)
    elif args.quick_check:
        run_premarket_scan(mode="quick", test=args.test, verbose=args.verbose, save_prep=False)
    elif args.full_scan:
        run_premarket_scan(mode="full",  test=args.test, verbose=args.verbose, save_prep=True)
    elif args.broad_scan:
        run_premarket_scan(mode="broad", test=args.test, verbose=args.verbose, save_prep=True)
    else:
        # Default: broad scan (prints results, no WhatsApp unless --test removed)
        run_premarket_scan(mode="broad", test=args.test, verbose=args.verbose, save_prep=False)
