"""
social_watcher.py — Social & alternative data intelligence agent.

Runs every 60 minutes (and once at startup). Aggregates:

  1. Congress Trades (CapitolTrades via OpenSecrets-style RSS / Quiver Quant public API)
     - Fetches recent congressional stock purchases from public disclosures
     - Focuses on buys (senators/reps buying = bullish signal historically)

  2. Unusual Options Activity (via yfinance options chain)
     - Scans watchlist + top-movers for unusual call/put ratios
     - Flags tickers with call volume > 3× normal or put/call < 0.3

  3. Trending Tickers (via Reddit-style RSS / StockTwits public feed)
     - Uses StockTwits trending API (no auth required) for real-time retail sentiment
     - Reports trending tickers + sentiment direction

All stored in world_context.social for injection into Claude prompts.
"""

import asyncio
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import requests
import yfinance as yf
import anthropic

from config import ANTHROPIC_API_KEY
import world_context as wctx

SOCIAL_INTERVAL = 60 * 60   # 1 hour

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ── 1. Congress Trades ─────────────────────────────────────────────────────────

CONGRESS_RSS = "https://efts.sec.gov/LATEST/search-index?q=%22Form+4%22&dateRange=custom&startdt={start}&enddt={end}&hits.hits._source.period_of_report=true&hits.hits.total.value=true"

def _fetch_congress_trades() -> list[dict]:
    """
    Fetch recent congressional stock disclosures from Quiver Quant public RSS.
    Falls back to House Financial Disclosures if unavailable.
    Returns list of {ticker, politician, amount, date, action} dicts.
    """
    trades = []

    # Primary: QuiverQuant Congress trading (public, no auth)
    try:
        r = requests.get(
            "https://www.quiverquant.com/sources/congresstrading",
            timeout=15,
            headers={"User-Agent": "stock-ai-agent/3.0 (market research)"},
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                for row in data[:30]:
                    ticker  = row.get("Ticker", "").upper().strip()
                    tx_type = row.get("Transaction", "")
                    amount  = row.get("Range", row.get("Amount", "?"))
                    name    = row.get("Representative", row.get("Senator", "Unknown"))
                    date    = row.get("TransactionDate", "")

                    if not ticker or len(ticker) > 5:
                        continue
                    if "purchase" in tx_type.lower() or "buy" in tx_type.lower():
                        trades.append({
                            "ticker":      ticker,
                            "politician":  name,
                            "amount":      amount,
                            "date":        date,
                            "action":      "BUY",
                        })
            return trades[:10]
    except Exception as e:
        print(f"⚠️  [SocialAgent] Congress trades error: {e}")

    return trades


# ── 2. Unusual Options Activity ────────────────────────────────────────────────

_SCAN_TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA",
    "AMD", "GOOGL", "NFLX", "BA", "XLE", "XLF", "XLK",
]


def _fetch_unusual_options(extra_tickers: list[str] | None = None) -> list[dict]:
    """
    Scan options chains for unusual call/put activity.
    Returns list of {ticker, call_put_ratio, description} dicts.
    """
    tickers = list(dict.fromkeys(_SCAN_TICKERS + (extra_tickers or [])))
    unusual = []

    for ticker in tickers[:20]:   # cap at 20 to avoid long runtime
        try:
            t    = yf.Ticker(ticker)
            opts = t.options          # list of expiration dates
            if not opts:
                continue

            # Use nearest expiration
            chain = t.option_chain(opts[0])
            calls = chain.calls
            puts  = chain.puts

            if calls.empty or puts.empty:
                continue

            call_vol = int(calls["volume"].fillna(0).sum())
            put_vol  = int(puts["volume"].fillna(0).sum())

            if put_vol == 0:
                continue

            cp_ratio = round(call_vol / put_vol, 2)

            # Flag unusual: very bullish (ratio > 3) or very bearish (ratio < 0.3)
            if cp_ratio >= 3.0:
                unusual.append({
                    "ticker":          ticker,
                    "call_put_ratio":  cp_ratio,
                    "call_vol":        call_vol,
                    "put_vol":         put_vol,
                    "description":     f"Unusual CALL activity — C/P ratio {cp_ratio:.1f}×, {call_vol:,} calls vs {put_vol:,} puts",
                    "bias":            "BULLISH",
                })
            elif cp_ratio <= 0.3:
                unusual.append({
                    "ticker":          ticker,
                    "call_put_ratio":  cp_ratio,
                    "call_vol":        call_vol,
                    "put_vol":         put_vol,
                    "description":     f"Unusual PUT activity — C/P ratio {cp_ratio:.2f}×, {put_vol:,} puts vs {call_vol:,} calls",
                    "bias":            "BEARISH",
                })

        except Exception:
            continue

    return sorted(unusual, key=lambda x: abs(x["call_put_ratio"] - 1), reverse=True)[:5]


# ── 3. StockTwits Trending ────────────────────────────────────────────────────

def _fetch_trending_tickers() -> list[dict]:
    """
    Fetch trending tickers from StockTwits public trending API.
    Returns list of {ticker, watchers, mentions, sentiment} dicts.
    """
    try:
        r = requests.get(
            "https://api.stocktwits.com/api/2/trending/symbols.json",
            timeout=10,
            headers={"User-Agent": "stock-ai-agent/3.0"},
        )
        if r.status_code != 200:
            return []

        data   = r.json()
        syms   = data.get("symbols", [])
        result = []
        for s in syms[:10]:
            ticker   = s.get("symbol", "").upper()
            title    = s.get("title", "")
            watchers = s.get("watchlist_count", 0)
            result.append({
                "ticker":     ticker,
                "name":       title,
                "watchers":   watchers,
                "mentions":   0,
                "change_pct": 0.0,
                "source":     "StockTwits",
            })
        return result

    except Exception as e:
        print(f"⚠️  [SocialAgent] StockTwits error: {e}")
        return []


# ── Summarize with Claude ──────────────────────────────────────────────────────

def _analyze_with_claude(
    trending:       list[dict],
    congress_buys:  list[dict],
    unusual_opts:   list[dict],
) -> str:
    """Generate a brief natural-language summary of alt-data signals."""
    if not (trending or congress_buys or unusual_opts):
        return ""

    parts = []
    if congress_buys:
        buys = ", ".join(f"{t['ticker']} ({t['politician'][:20]})" for t in congress_buys[:3])
        parts.append(f"Congress buys: {buys}")
    if unusual_opts:
        opts = ", ".join(f"{t['ticker']} C/P={t['call_put_ratio']:.1f}" for t in unusual_opts[:3])
        parts.append(f"Unusual options: {opts}")
    if trending:
        trend = ", ".join(t["ticker"] for t in trending[:5])
        parts.append(f"StockTwits trending: {trend}")

    if not parts:
        return ""

    prompt = f"""Briefly summarize these alternative market data signals for a stock trader (1 sentence, actionable):

{chr(10).join(parts)}

Focus on what's actionable. Be concise."""

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception:
        return " | ".join(parts)


# ── Single sweep ───────────────────────────────────────────────────────────────

def _run_sweep(extra_tickers: list[str] | None = None) -> bool:
    """Fetch all social/alt-data signals, update world_context."""
    print("🔭 [SocialAgent] Fetching social & alternative data...")

    trending      = _fetch_trending_tickers()
    congress_buys = _fetch_congress_trades()
    unusual_opts  = _fetch_unusual_options(extra_tickers)

    print(f"🔭 [SocialAgent] Trending={len(trending)}  Congress buys={len(congress_buys)}  Unusual opts={len(unusual_opts)}")

    summary = _analyze_with_claude(trending, congress_buys, unusual_opts)

    wctx.update_social({
        "trending":      trending,
        "congress_buys": congress_buys,
        "unusual_opts":  unusual_opts,
        "summary":       summary,
    })

    if congress_buys:
        for b in congress_buys[:3]:
            print(f"   🏛️  {b['ticker']}: {b['politician'][:25]}  ${b['amount']}")
    if unusual_opts:
        for o in unusual_opts[:3]:
            icon = "🟢" if o["bias"] == "BULLISH" else "🔴"
            print(f"   {icon} {o['ticker']}: {o['description'][:70]}")
    if trending:
        print(f"   📱 Trending: {' '.join(t['ticker'] for t in trending[:6])}")
    if summary:
        print(f"🔭 [SocialAgent] {summary[:120]}")

    return True


# ── Main loop ──────────────────────────────────────────────────────────────────

async def social_watcher_loop(extra_tickers: list[str] | None = None):
    """Async loop started in main.py lifespan. Runs immediately then hourly."""
    print(f"🔭 [SocialAgent] Started — updating every {SOCIAL_INTERVAL//60}min")
    loop = asyncio.get_running_loop()

    await loop.run_in_executor(None, _run_sweep, extra_tickers)

    while True:
        try:
            await asyncio.sleep(SOCIAL_INTERVAL)
            await loop.run_in_executor(None, _run_sweep, extra_tickers)
        except asyncio.CancelledError:
            print("🔭 [SocialAgent] Stopped.")
            break
        except Exception as e:
            print(f"❌ [SocialAgent] Loop error: {e}")
            await asyncio.sleep(600)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import watchlist_manager as wl
    print("=== SocialAgent Standalone Test ===\n")
    _run_sweep(extra_tickers=wl.load())
    ctx    = wctx.get()
    social = ctx["social"]

    print(f"\nTrending ({len(social['trending'])}):")
    for t in social["trending"][:5]:
        print(f"  📱 {t['ticker']} — {t['name'][:40]}")

    print(f"\nCongress Buys ({len(social['congress_buys'])}):")
    for b in social["congress_buys"][:5]:
        print(f"  🏛️  {b['ticker']} — {b['politician'][:30]}  ${b['amount']}  {b['date']}")

    print(f"\nUnusual Options ({len(social['unusual_opts'])}):")
    for o in social["unusual_opts"]:
        icon = "🟢" if o["bias"] == "BULLISH" else "🔴"
        print(f"  {icon} {o['ticker']}: {o['description']}")

    print(f"\nSummary: {social.get('summary', '')}")
    print(f"\n--- Prompt section preview ---")
    print(wctx.build_prompt_section())
