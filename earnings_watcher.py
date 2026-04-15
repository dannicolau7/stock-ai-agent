"""
earnings_watcher.py — Earnings calendar intelligence agent.

Runs every 4 hours (and once at startup). Fetches upcoming earnings dates
via yfinance for every ticker in the watchlist + a curated list of major
market-movers (AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, etc.).

For each company reporting in the next 7 days:
  - Historical average post-earnings move (up and down)
  - Beat rate (% of last 8 quarters that beat EPS estimate)
  - Expected direction bias (BULLISH / BEARISH / NEUTRAL)
  - Days until report

Uses Claude Haiku to synthesize which upcoming reports matter most as
market catalysts and which sectors/stocks to watch.

Stored in world_context.earnings and injected into every Claude prompt.
"""

import asyncio
import json
from datetime import datetime, timedelta

import yfinance as yf
import anthropic

from config import ANTHROPIC_API_KEY
import world_context as wctx

EARNINGS_INTERVAL = 4 * 60 * 60   # 4 hours between updates

# Major market-movers that can move entire sectors
MARKET_MOVERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
    "JPM", "GS", "BAC", "WFC",          # financials
    "XOM", "CVX",                         # energy
    "JNJ", "PFE", "MRNA",                # pharma
    "WMT", "COST", "TGT",                # retail
    "NFLX", "DIS", "SPOT",               # media/streaming
    "AMD", "INTC", "QCOM", "AVGO",       # semis
    "BA", "LMT", "RTX",                  # defense
]

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ── Data fetching ──────────────────────────────────────────────────────────────

def _get_earnings_for_ticker(ticker: str) -> dict | None:
    """
    Fetch upcoming earnings date and historical reaction data for a single ticker.
    Returns None if no upcoming earnings within 14 days.
    """
    try:
        t = yf.Ticker(ticker)

        # Calendar gives next earnings date — yfinance returns a dict
        cal = t.calendar
        if not cal or "Earnings Date" not in cal:
            return None

        earn_dates = cal["Earnings Date"]
        if not earn_dates:
            return None

        # May be a list or a single value
        earn_date_raw = earn_dates[0] if isinstance(earn_dates, list) else earn_dates
        if earn_date_raw is None:
            return None

        # Normalize to date
        if hasattr(earn_date_raw, "date"):
            earn_date = earn_date_raw.date()
        elif hasattr(earn_date_raw, "year"):
            earn_date = earn_date_raw
        else:
            earn_date = datetime.strptime(str(earn_date_raw)[:10], "%Y-%m-%d").date()

        today = datetime.now().date()
        days_until = (earn_date - today).days

        if days_until < 0 or days_until > 14:
            return None

        # Historical earnings reactions from earnings_history
        avg_move = 0.0
        beat_rate = 0.0
        try:
            hist = t.earnings_history
            if hist is not None and not hist.empty and len(hist) >= 2:
                # Compute surprise magnitude as proxy for move size
                # earnings_history columns: epsActual, epsEstimate, epsDifference, surprisePercent
                if "surprisePercent" in hist.columns:
                    surprises = hist["surprisePercent"].dropna().abs()
                    beat_flags = (hist["epsDifference"].dropna() > 0)
                    avg_move   = float(surprises.mean()) if len(surprises) > 0 else 0.0
                    beat_rate  = float(beat_flags.mean() * 100) if len(beat_flags) > 0 else 50.0
        except Exception:
            avg_move  = 5.0
            beat_rate = 50.0

        return {
            "ticker":         ticker,
            "date":           earn_date.isoformat(),
            "days":           days_until,
            "avg_move_pct":   round(avg_move, 1),
            "beat_rate":      round(beat_rate, 0),
            "direction":      "BULLISH" if beat_rate >= 60 else "BEARISH" if beat_rate <= 40 else "NEUTRAL",
        }

    except Exception:
        return None


def _fetch_earnings_data(tickers: list[str]) -> list[dict]:
    """Fetch earnings info for all tickers, return those reporting in next 14 days."""
    results = []
    for ticker in tickers:
        info = _get_earnings_for_ticker(ticker)
        if info:
            results.append(info)

    # Sort by days until report
    results.sort(key=lambda x: x["days"])
    return results


def _analyze_with_claude(upcoming: list[dict]) -> dict:
    """
    Ask Claude Haiku which upcoming earnings are most significant as market catalysts.
    """
    if not upcoming:
        return {}

    lines = []
    for e in upcoming[:10]:
        lines.append(
            f"  {e['ticker']}: reports in {e['days']}d  |  "
            f"avg surprise ±{e['avg_move_pct']:.1f}%  |  "
            f"beat rate {e['beat_rate']:.0f}%  |  "
            f"recent bias {e['direction']}"
        )

    prompt = f"""You are a senior equity strategist at a hedge fund. Analyze these upcoming earnings reports.

UPCOMING EARNINGS (next 14 days):
{chr(10).join(lines)}

Instructions:
1. Identify the 3-5 most important reports that could move the broader market or their sector
2. For each, give a brief thesis for why it matters
3. Note any sector read-throughs (e.g. TSLA results affect EV supply chain)
4. Provide overall earnings season tone

Respond ONLY with valid JSON:
{{
  "hot_plays": [
    {{
      "ticker": "NVDA",
      "days": 3,
      "thesis": "one sentence on why this report matters",
      "sector_impact": ["XLK", "SOXX"],
      "direction": "BULLISH"
    }}
  ],
  "earnings_tone": "BULLISH|BEARISH|NEUTRAL|MIXED",
  "key_themes": ["AI spending", "consumer weakness"],
  "summary": "2 sentences on what earnings season means for the market right now"
}}"""

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if "```" in text:
            parts = text.split("```")
            text  = parts[1].lstrip("json").strip() if len(parts) >= 3 else text
        return json.loads(text)
    except Exception as e:
        print(f"⚠️  [EarningsAgent] Claude error: {e}")
        return {}


# ── Single sweep ───────────────────────────────────────────────────────────────

def _run_sweep(extra_tickers: list[str] | None = None) -> bool:
    """Fetch earnings calendar, analyze catalysts, update world_context."""
    print("📅 [EarningsAgent] Scanning earnings calendar...")

    tickers = list(dict.fromkeys(MARKET_MOVERS + (extra_tickers or [])))
    upcoming = _fetch_earnings_data(tickers)

    if not upcoming:
        print("📅 [EarningsAgent] No earnings in next 14 days for tracked tickers")
        wctx.update_earnings({"upcoming": [], "hot_plays": []})
        return False

    print(f"📅 [EarningsAgent] {len(upcoming)} report(s) upcoming — analyzing with Claude Haiku...")

    analysis = _analyze_with_claude(upcoming)
    hot_plays    = analysis.get("hot_plays", [])
    earnings_tone = analysis.get("earnings_tone", "NEUTRAL")
    summary      = analysis.get("summary", "")

    wctx.update_earnings({
        "upcoming":       upcoming,
        "hot_plays":      hot_plays,
        "earnings_tone":  earnings_tone,
        "summary":        summary,
    })

    tone_icon = "🟢" if earnings_tone == "BULLISH" else "🔴" if earnings_tone == "BEARISH" else "🟡"
    print(f"📅 [EarningsAgent] {tone_icon} Tone={earnings_tone}  {len(upcoming)} reports  {len(hot_plays)} hot play(s)")
    for e in upcoming[:5]:
        icon = "🟢" if e["direction"] == "BULLISH" else "🔴" if e["direction"] == "BEARISH" else "🟡"
        print(f"   {icon} {e['ticker']} in {e['days']}d  avg±{e['avg_move_pct']:.1f}%  beat {e['beat_rate']:.0f}%")
    if summary:
        print(f"📅 [EarningsAgent] {summary[:120]}")

    return True


# ── Main loop ──────────────────────────────────────────────────────────────────

async def earnings_watcher_loop(extra_tickers: list[str] | None = None):
    """Async loop started in main.py lifespan. Runs immediately then every 4h."""
    print(f"📅 [EarningsAgent] Started — updating every {EARNINGS_INTERVAL//3600}h")
    loop = asyncio.get_running_loop()

    await loop.run_in_executor(None, _run_sweep, extra_tickers)

    while True:
        try:
            await asyncio.sleep(EARNINGS_INTERVAL)
            await loop.run_in_executor(None, _run_sweep, extra_tickers)
        except asyncio.CancelledError:
            print("📅 [EarningsAgent] Stopped.")
            break
        except Exception as e:
            print(f"❌ [EarningsAgent] Loop error: {e}")
            await asyncio.sleep(600)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import watchlist_manager as wl
    print("=== EarningsAgent Standalone Test ===\n")
    _run_sweep(extra_tickers=wl.load())
    ctx      = wctx.get()
    earnings = ctx["earnings"]
    print(f"\nUpcoming ({len(earnings['upcoming'])}): ")
    for e in earnings["upcoming"]:
        icon = "🟢" if e["direction"] == "BULLISH" else "🔴" if e["direction"] == "BEARISH" else "🟡"
        print(f"  {icon} {e['ticker']} in {e['days']}d  ±{e['avg_move_pct']:.1f}%  beat {e['beat_rate']:.0f}%")
    print(f"\nHot plays ({len(earnings['hot_plays'])}):")
    for h in earnings["hot_plays"]:
        print(f"  {h['ticker']}: {h.get('thesis','')}")
    print(f"\nSummary: {earnings.get('summary','')}")
    print(f"\n--- Prompt section preview ---")
    print(wctx.build_prompt_section())
