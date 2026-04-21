"""
discovery_agent.py — Dynamic watchlist discovery agent.

Runs every 4 hours. Instead of relying solely on a static watchlist,
this agent:

  1. Reads current macro regime + breadth rotation from world_context
  2. Asks Claude Sonnet: "Given this environment, which sectors and
     stocks should we be watching right now?"
  3. Screens the universe for technically qualified candidates in those sectors
  4. Merges with the existing static watchlist and provides a ranked
     "dynamic scan list" for the monitoring loop

Output:
  - world_context.discovery.candidates — list of tickers to prioritize
  - world_context.discovery.rationale  — why each was chosen
  - Logs reasoning to data/discovery_log.jsonl

This makes the system proactive: when macro shifts from BULL to BEAR,
the scanner automatically rotates toward defensive sectors without
manual intervention.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import yfinance as yf
import anthropic

from config import ANTHROPIC_API_KEY
import world_context as wctx

DISCOVERY_INTERVAL = 4 * 60 * 60   # 4 hours
LOG_PATH = Path(__file__).parent / "data" / "discovery_log.jsonl"

# Theme → curated liquid tickers for each catalyst sub-sector
CATALYST_THEME_UNIVERSE: dict[str, list[str]] = {
    "quantum_computing":  ["IONQ", "RGTI", "QBTS", "XNDU", "QUBT", "ARQQ", "IBM"],
    "ai_infrastructure":  ["NVDA", "AMD", "SMCI", "PLTR", "ARM", "INTC", "DELL"],
    "biotech_fda":        ["MRNA", "BNTX", "NVAX", "RCKT", "IOVA", "ACAD", "FOLD"],
    "ev_automotive":      ["TSLA", "RIVN", "LCID", "NIO", "FSR", "XPEV", "LI"],
    "crypto_blockchain":  ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BTBT"],
    "energy_oil":         ["XOM", "CVX", "OXY", "SLB", "HAL", "FANG", "MPC"],
    "defense_aerospace":  ["LMT", "RTX", "BA", "NOC", "GD", "AXON", "RKLB"],
    "semiconductors":     ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "MU", "AMAT"],
    "clean_energy":       ["ENPH", "SEDG", "FSLR", "PLUG", "BE", "BLDP", "RUN"],
    "fintech":            ["SQ", "PYPL", "AFRM", "UPST", "SOFI", "NU", "DAVE"],
    "space":              ["RKLB", "JOBY", "ACHR", "ASTS", "LUNR", "RDW"],
    "cybersecurity":      ["CRWD", "PANW", "ZS", "S", "FTNT", "QLYS", "TENB"],
}

# Sector → representative liquid tickers to screen
SECTOR_UNIVERSE = {
    "XLK":  ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "AVGO", "CRM", "ORCL", "META"],
    "XLF":  ["JPM", "GS", "BAC", "WFC", "MS", "BLK", "AXP", "V", "MA", "C"],
    "XLE":  ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "HAL", "OXY"],
    "XLV":  ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN"],
    "XLY":  ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "CMG"],
    "XLP":  ["PG", "KO", "PEP", "WMT", "COST", "CL", "MO", "PM", "MDLZ", "EL"],
    "XLI":  ["BA", "RTX", "LMT", "HON", "UPS", "CAT", "DE", "GE", "MMM", "FDX"],
    "XLB":  ["LIN", "APD", "ECL", "SHW", "FCX", "NEM", "DD", "PPG", "VMC", "MLM"],
    "XLC":  ["GOOGL", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "SPOT", "EA", "TTWO"],
    "XLRE": ["AMT", "PLD", "EQIX", "CCI", "PSA", "EXR", "SPG", "O", "AVB", "EQR"],
    "XLU":  ["NEE", "DUK", "SO", "AEP", "EXC", "SRE", "PCG", "ED", "XEL", "WEC"],
}

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ── Catalyst helpers ──────────────────────────────────────────────────────────

def _get_catalyst_tickers() -> list[str]:
    """Return deduplicated tickers for all active BULLISH sector catalysts (score >= 6)."""
    geo = wctx.get().get("geo", {})
    catalysts = [
        c for c in geo.get("sector_catalysts", [])
        if c.get("score", 0) >= 6 and c.get("direction") == "BULLISH"
    ]
    tickers: list[str] = []
    for cat in catalysts:
        tickers += cat.get("tickers", [])
        tickers += CATALYST_THEME_UNIVERSE.get(cat.get("theme", ""), [])
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ── Step 1: Ask Claude which sectors to focus on ──────────────────────────────

def _ask_claude_for_sectors(ctx: dict) -> dict:
    """Given current world context, ask Claude which sectors + themes to hunt."""
    macro   = ctx["macro"]
    breadth = ctx["breadth"]
    geo     = ctx["geo"]
    earnings = ctx["earnings"]

    world_section = wctx.build_prompt_section()

    catalyst_tickers = _get_catalyst_tickers()
    catalyst_note = ""
    if catalyst_tickers:
        catalyst_note = f"\nACTIVE SECTOR CATALYSTS: {', '.join(catalyst_tickers[:12])} — prioritize these.\n"

    prompt = f"""You are a portfolio manager choosing which sectors and stocks to scan for trading opportunities TODAY.

CURRENT ENVIRONMENT:
{world_section}{catalyst_note}
Available sector ETFs to choose from:
XLK (Tech), XLF (Financials), XLE (Energy), XLV (Healthcare), XLY (Consumer Disc),
XLP (Consumer Staples), XLI (Industrials), XLB (Materials), XLC (Communication),
XLRE (Real Estate), XLU (Utilities)

Instructions:
1. Choose the 2-4 sectors most likely to produce BUY signals in the NEXT 24-48 hours
2. For each sector, name 2-3 specific stocks within it that have near-term catalysts
3. Name 1-2 sectors to AVOID today
4. Provide a one-line theme for today's scan focus

Respond ONLY with valid JSON:
{{
  "focus_sectors": ["XLF", "XLI"],
  "avoid_sectors": ["XLP", "XLRE"],
  "stock_picks": [
    {{"ticker": "JPM", "sector": "XLF", "reason": "earnings catalyst + rising rates tailwind"}},
    {{"ticker": "CAT", "sector": "XLI", "reason": "infrastructure spending + breakout setup"}}
  ],
  "theme": "Financial + Industrial rotation on earnings catalyst + strong dollar",
  "confidence": 75
}}"""

    try:
        response = _get_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if "```" in text:
            parts = text.split("```")
            text  = parts[1].lstrip("json").strip() if len(parts) >= 3 else text
        return json.loads(text)
    except Exception as e:
        print(f"⚠️  [Discovery] Claude error: {e}")
        return {}


# ── Step 2: Screen sector tickers for technical setup ────────────────────────

def _screen_ticker(ticker: str) -> dict | None:
    """
    Quick technical screen: must pass all gates to be added to scan list.
    Returns summary dict or None if fails gates.
    """
    try:
        t  = yf.Ticker(ticker)
        df = t.history(period="30d", interval="1d", auto_adjust=True)
        if df is None or len(df) < 10:
            return None

        closes = df["Close"]
        volumes = df["Volume"]

        last_close  = float(closes.iloc[-1])
        ma20        = float(closes.tail(20).mean())
        ma10        = float(closes.tail(10).mean())
        avg_vol     = float(volumes.tail(20).mean())
        last_vol    = float(volumes.iloc[-1])

        # Gates: price above 20d MA, recent volume not collapsing
        above_ma    = last_close > ma20
        vol_ok      = last_vol > avg_vol * 0.5   # at least 50% of avg

        if not (above_ma and vol_ok):
            return None

        # RSI (simple)
        deltas = closes.diff().dropna()
        gains  = deltas.clip(lower=0).tail(14)
        losses = (-deltas.clip(upper=0)).tail(14)
        rs     = gains.mean() / losses.mean() if losses.mean() > 0 else 100
        rsi    = round(100 - (100 / (1 + rs)), 1)

        # Skip overbought
        if rsi > 78:
            return None

        return {
            "ticker":    ticker,
            "price":     round(last_close, 2),
            "ma20":      round(ma20, 2),
            "vs_ma20":   round((last_close / ma20 - 1) * 100, 2),
            "rsi":       rsi,
            "vol_ratio": round(last_vol / avg_vol, 2),
        }

    except Exception:
        return None


def _screen_sector(sector_etf: str, extra_picks: list[str]) -> list[dict]:
    """Screen all tickers in a sector + any extra picks Claude suggested."""
    universe = SECTOR_UNIVERSE.get(sector_etf, []) + extra_picks
    universe = list(dict.fromkeys(universe))   # deduplicate

    results = []
    for ticker in universe[:15]:   # cap per sector
        result = _screen_ticker(ticker)
        if result:
            results.append(result)

    return sorted(results, key=lambda x: x["rsi"])   # lowest RSI = least overbought


# ── Step 3: Merge + rank candidates ──────────────────────────────────────────

def _build_candidate_list(
    focus_sectors: list[str],
    avoid_sectors: list[str],
    stock_picks: list[dict],
    static_watchlist: list[str],
) -> list[dict]:
    """Screen focus sectors, merge Claude picks + static watchlist, rank."""
    candidates = []

    # Claude's specific picks (always include if they pass screening)
    claude_tickers = {p["ticker"]: p for p in stock_picks}

    for sector in focus_sectors:
        if sector in avoid_sectors:
            continue
        extra = [p["ticker"] for p in stock_picks if p.get("sector") == sector]
        screened = _screen_sector(sector, extra)
        for s in screened:
            s["source"] = "discovery"
            s["sector"] = sector
            if s["ticker"] in claude_tickers:
                s["reason"] = claude_tickers[s["ticker"]].get("reason", "")
                s["claude_pick"] = True
            else:
                s["claude_pick"] = False
            candidates.append(s)

    # Always include static watchlist tickers (even if not in focus sectors)
    for ticker in static_watchlist:
        if not any(c["ticker"] == ticker for c in candidates):
            result = _screen_ticker(ticker)
            if result:
                result["source"] = "watchlist"
                result["sector"] = "static"
                result["reason"] = "user watchlist"
                result["claude_pick"] = False
                candidates.append(result)

    # Rank: Claude picks first, then by RSI (momentum without being overbought)
    candidates.sort(key=lambda x: (not x.get("claude_pick"), x["rsi"]))
    return candidates[:20]   # top 20


# ── Single sweep ───────────────────────────────────────────────────────────────

def _run_sweep(static_watchlist: list[str] | None = None) -> bool:
    """Run full discovery cycle and update world_context."""
    print("🔍 [Discovery] Starting sector + stock discovery...")

    ctx = wctx.get()
    if not ctx["macro"].get("regime") or ctx["macro"]["regime"] == "UNKNOWN":
        print("🔍 [Discovery] Macro not ready yet — skipping")
        return False

    print("🔍 [Discovery] Asking Claude which sectors to focus on...")
    guidance = _ask_claude_for_sectors(ctx)
    if not guidance:
        return False

    focus_sectors = guidance.get("focus_sectors", ["XLK", "XLF"])
    avoid_sectors = guidance.get("avoid_sectors", [])
    stock_picks   = guidance.get("stock_picks", [])
    theme         = guidance.get("theme", "")

    print(f"🔍 [Discovery] Focus: {focus_sectors}  Avoid: {avoid_sectors}")
    print(f"🔍 [Discovery] Theme: {theme}")
    print(f"🔍 [Discovery] Screening {len(focus_sectors)} sectors + {len(stock_picks)} Claude picks...")

    candidates = _build_candidate_list(
        focus_sectors=focus_sectors,
        avoid_sectors=avoid_sectors,
        stock_picks=stock_picks,
        static_watchlist=static_watchlist or [],
    )

    # Update world_context with discovery results
    wctx.update_social({
        "discovery_candidates": [c["ticker"] for c in candidates],
        "discovery_theme":      theme,
        "discovery_focus":      focus_sectors,
        "discovery_avoid":      avoid_sectors,
    })

    # Log to file
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            "ts":         datetime.now().isoformat(),
            "theme":      theme,
            "focus":      focus_sectors,
            "avoid":      avoid_sectors,
            "candidates": [c["ticker"] for c in candidates],
        }) + "\n")

    print(f"🔍 [Discovery] {len(candidates)} candidates ready for scan:")
    for c in candidates[:10]:
        mark = "⭐" if c.get("claude_pick") else "  "
        print(f"   {mark} {c['ticker']}  RSI={c['rsi']:.0f}  vs20MA={c['vs_ma20']:+.1f}%  "
              f"vol={c['vol_ratio']:.1f}×  {c.get('reason','')[:40]}")

    return True


def get_discovery_tickers() -> list[str]:
    """Returns catalyst tickers (prepended) + discovery candidates, deduplicated."""
    ctx = wctx.get()
    catalyst_tickers      = _get_catalyst_tickers()
    discovery_candidates  = ctx.get("social", {}).get("discovery_candidates", [])
    seen: set[str] = set()
    out: list[str] = []
    for t in catalyst_tickers + discovery_candidates:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ── Main loop ─────────────────────────────────────────────────────────────────

async def discovery_agent_loop(static_watchlist_fn=None):
    """Async loop started in main.py lifespan. Runs every 4 hours."""
    print(f"🔍 [Discovery] Started — scanning every {DISCOVERY_INTERVAL//3600}h")
    loop = asyncio.get_running_loop()

    # Wait for macro to be populated (geo/macro agents run at startup)
    await asyncio.sleep(90)

    watchlist = static_watchlist_fn() if static_watchlist_fn else []
    await loop.run_in_executor(None, _run_sweep, watchlist)

    while True:
        try:
            await asyncio.sleep(DISCOVERY_INTERVAL)
            watchlist = static_watchlist_fn() if static_watchlist_fn else []
            await loop.run_in_executor(None, _run_sweep, watchlist)
        except asyncio.CancelledError:
            print("🔍 [Discovery] Stopped.")
            break
        except Exception as e:
            print(f"❌ [Discovery] Loop error: {e}")
            await asyncio.sleep(600)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import watchlist_manager as wl

    # Need macro/breadth to be populated — run a quick macro sweep first
    from macro_watcher import _run_sweep as macro_sweep
    from breadth_watcher import _run_sweep as breadth_sweep
    from geo_watcher import _run_sweep as geo_sweep

    print("=== Discovery Agent Standalone Test ===\n")
    print("Populating macro/breadth context first...")
    macro_sweep()
    breadth_sweep()
    geo_sweep()

    print("\nRunning discovery...")
    _run_sweep(static_watchlist=wl.load())

    candidates = get_discovery_tickers()
    print(f"\nFinal candidate list ({len(candidates)} tickers):")
    print(" ".join(candidates))
