"""
breadth_watcher.py — Market breadth & internal health agent.

Runs every 30 minutes (and once at startup). Uses yfinance to measure:
  - VIX level + term structure (VIX vs VIX3M)
  - SPY vs its 50 and 200-day moving averages
  - Advance/Decline proxy via equal-weight vs cap-weight spread (RSP vs SPY)
  - Sector rotation: all 11 SPDR sector ETFs vs their 20d MA
  - Put/Call ratio proxy via QQQ vs SQQQ flow

Uses Claude Haiku to synthesize breadth signals into:
  - Overall market health label
  - Trending sectors (above/below 20d MA)
  - Summary for prompt injection

Stored in world_context.breadth.
"""

import asyncio
import json

import yfinance as yf
import anthropic
import pandas as pd

from config import ANTHROPIC_API_KEY
import world_context as wctx

BREADTH_INTERVAL = 30 * 60   # 30 minutes

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Disc",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLC": "Communication",
}

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_breadth_data() -> dict:
    """
    Download all breadth indicators in one batch yfinance call.
    Returns computed indicators as a dict.
    """
    tickers = ["^VIX", "^VIX3M", "SPY", "QQQ", "IWM", "RSP"] + list(SECTOR_ETFS.keys())
    tickers_str = " ".join(tickers)

    try:
        df = yf.download(
            tickers_str,
            period="60d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"⚠️  [BreadthAgent] yfinance download failed: {e}")
        return {}

    def get_closes(ticker: str) -> pd.Series:
        try:
            t_df = df[ticker] if len(tickers) > 1 else df
            if t_df is None or t_df.empty:
                return pd.Series(dtype=float)
            return t_df["Close"].dropna()
        except Exception:
            return pd.Series(dtype=float)

    result: dict = {}

    # VIX & term structure
    vix_c   = get_closes("^VIX")
    vix3m_c = get_closes("^VIX3M")
    if len(vix_c) >= 1:
        result["vix"]  = float(vix_c.iloc[-1])
    if len(vix3m_c) >= 1:
        result["vix3m"] = float(vix3m_c.iloc[-1])

    vix  = result.get("vix", 0)
    vix3m = result.get("vix3m", 0)
    if vix > 0 and vix3m > 0:
        result["vix_term_structure"] = (
            "BACKWARDATION (fear)" if vix > vix3m
            else "CONTANGO (calm)"
        )
    else:
        result["vix_term_structure"] = "UNKNOWN"

    # SPY vs 50d / 200d MA
    spy_c = get_closes("SPY")
    if len(spy_c) >= 50:
        spy_last  = float(spy_c.iloc[-1])
        ma50      = float(spy_c.tail(50).mean())
        result["spy_vs_50ma"]  = round((spy_last / ma50 - 1) * 100, 2)
    if len(spy_c) >= 200:
        ma200     = float(spy_c.tail(200).mean())
        result["spy_vs_200ma"] = round((spy_last / ma200 - 1) * 100, 2)

    # IWM vs SPY (small vs large — risk appetite)
    iwm_c = get_closes("IWM")
    if len(spy_c) >= 20 and len(iwm_c) >= 20:
        spy_ret = (float(spy_c.iloc[-1]) - float(spy_c.iloc[-20])) / float(spy_c.iloc[-20]) * 100
        iwm_ret = (float(iwm_c.iloc[-1]) - float(iwm_c.iloc[-20])) / float(iwm_c.iloc[-20]) * 100
        result["small_vs_large_20d"] = round(iwm_ret - spy_ret, 2)  # positive = risk-on

    # RSP vs SPY (equal-weight vs cap-weight = breadth proxy)
    rsp_c = get_closes("RSP")
    if len(rsp_c) >= 20 and len(spy_c) >= 20:
        rsp_ret = (float(rsp_c.iloc[-1]) - float(rsp_c.iloc[-20])) / float(rsp_c.iloc[-20]) * 100
        spy_ret_20 = (float(spy_c.iloc[-1]) - float(spy_c.iloc[-20])) / float(spy_c.iloc[-20]) * 100
        result["equal_vs_cap_20d"] = round(rsp_ret - spy_ret_20, 2)  # positive = broad participation

    # Sector status vs 20d MA
    sectors_above = []
    sectors_below = []
    sector_data   = {}
    for etf in SECTOR_ETFS:
        c = get_closes(etf)
        if len(c) >= 20:
            last  = float(c.iloc[-1])
            ma20  = float(c.tail(20).mean())
            chg   = round((last / ma20 - 1) * 100, 2)
            sector_data[etf] = chg
            if chg >= 0:
                sectors_above.append(etf)
            else:
                sectors_below.append(etf)

    result["sectors_above_20ma"] = sectors_above
    result["sectors_below_20ma"] = sectors_below
    result["sector_data"]        = sector_data

    # Advance/Decline proxy: count of sectors above 20MA out of 11
    if sector_data:
        result["ad_ratio"] = round(len(sectors_above) / len(sector_data) * 10, 1)  # scale 0–10
        result["pct_above_200ma"] = round(len(sectors_above) / len(sector_data) * 100, 0)

    return result


def _analyze_with_claude(data: dict) -> dict:
    """Send breadth data to Claude Haiku for market health classification."""
    if not data:
        return {}

    vix            = data.get("vix", 0)
    vix_ts         = data.get("vix_term_structure", "UNKNOWN")
    spy_50         = data.get("spy_vs_50ma", 0)
    spy_200        = data.get("spy_vs_200ma", 0)
    ad_ratio       = data.get("ad_ratio", 5)
    sectors_above  = data.get("sectors_above_20ma", [])
    sectors_below  = data.get("sectors_below_20ma", [])
    small_vs_large = data.get("small_vs_large_20d", 0)
    equal_vs_cap   = data.get("equal_vs_cap_20d", 0)
    sector_data    = data.get("sector_data", {})

    sector_lines = "\n".join(
        f"  {etf} ({SECTOR_ETFS.get(etf, etf)}): {chg:+.2f}% vs 20d MA"
        for etf, chg in sorted(sector_data.items(), key=lambda x: -x[1])
    )

    prompt = f"""You are a senior technical market analyst. Classify market breadth and internal health.

MARKET BREADTH DATA:
VIX: {vix:.1f}  |  Term Structure: {vix_ts}
SPY vs 50d MA: {spy_50:+.2f}%
SPY vs 200d MA: {spy_200:+.2f}%
A/D Proxy (sector above/below 20d MA): {len(sectors_above)}/{len(sectors_above)+len(sectors_below)} sectors healthy
A/D Ratio (0-10 scale): {ad_ratio:.1f}
Small-cap vs Large-cap (20d): {small_vs_large:+.2f}% (positive = risk-on)
Equal-weight vs Cap-weight (20d): {equal_vs_cap:+.2f}% (positive = broad participation)

SECTOR PERFORMANCE vs 20d MA:
{sector_lines}

Classify overall market health and identify rotation patterns.

Respond ONLY with valid JSON:
{{
  "health": "STRONG|HEALTHY|NEUTRAL|WEAKENING|BEARISH",
  "rotation": "RISK_ON|RISK_OFF|DEFENSIVE|CYCLICAL|MIXED",
  "leading_sectors": ["XLK", "XLF"],
  "lagging_sectors": ["XLU", "XLRE"],
  "breadth_signal": "IMPROVING|STABLE|DETERIORATING",
  "summary": "2 sentences on market internal health and what it means for stock picking"
}}"""

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if "```" in text:
            parts = text.split("```")
            text  = parts[1].lstrip("json").strip() if len(parts) >= 3 else text
        return json.loads(text)
    except Exception as e:
        print(f"⚠️  [BreadthAgent] Claude error: {e}")
        return {}


# ── Single sweep ───────────────────────────────────────────────────────────────

def _run_sweep() -> bool:
    """Fetch breadth data, analyze, update world_context."""
    print("📈 [BreadthAgent] Fetching market breadth indicators...")
    data = _fetch_breadth_data()
    if not data:
        print("📈 [BreadthAgent] No data returned")
        return False

    vix   = data.get("vix", 0)
    ad    = data.get("ad_ratio", 0)
    s_ab  = len(data.get("sectors_above_20ma", []))
    s_tot = s_ab + len(data.get("sectors_below_20ma", []))
    print(f"📈 [BreadthAgent] VIX={vix:.1f}  Sectors above 20d MA: {s_ab}/{s_tot}  A/D ratio={ad:.1f}")

    print("📈 [BreadthAgent] Analyzing with Claude Haiku...")
    analysis = _analyze_with_claude(data)
    if not analysis:
        # Store raw data even without Claude analysis
        wctx.update_breadth({
            "vix":               data.get("vix", 0),
            "vix_term_structure": data.get("vix_term_structure", "UNKNOWN"),
            "ad_ratio":          data.get("ad_ratio", 0),
            "pct_above_200ma":   data.get("pct_above_200ma", 0),
        })
        return False

    health         = analysis.get("health", "NEUTRAL")
    rotation       = analysis.get("rotation", "MIXED")
    leading        = analysis.get("leading_sectors", [])
    lagging        = analysis.get("lagging_sectors", [])
    breadth_signal = analysis.get("breadth_signal", "STABLE")
    summary        = analysis.get("summary", "")

    wctx.update_breadth({
        "health":             health,
        "rotation":           rotation,
        "vix":                data.get("vix", 0),
        "vix_term_structure": data.get("vix_term_structure", "UNKNOWN"),
        "pct_above_200ma":    data.get("pct_above_200ma", 0),
        "ad_ratio":           data.get("ad_ratio", 0),
        "leading_sectors":    leading,
        "lagging_sectors":    lagging,
        "breadth_signal":     breadth_signal,
        "summary":            summary,
    })

    health_icon = "🟢" if health in ("STRONG", "HEALTHY") else "🔴" if health == "BEARISH" else "🟡"
    print(f"📈 [BreadthAgent] {health_icon} Health={health}  Rotation={rotation}  Signal={breadth_signal}")
    print(f"📈 [BreadthAgent] Leading: {','.join(leading) or 'none'}  Lagging: {','.join(lagging) or 'none'}")
    if summary:
        print(f"📈 [BreadthAgent] {summary[:120]}")
    return True


# ── Main loop ──────────────────────────────────────────────────────────────────

async def breadth_watcher_loop():
    """Async loop started in main.py lifespan. Runs immediately then every 30min."""
    print(f"📈 [BreadthAgent] Started — updating every {BREADTH_INTERVAL//60}min")
    loop = asyncio.get_running_loop()

    await loop.run_in_executor(None, _run_sweep)

    while True:
        try:
            await asyncio.sleep(BREADTH_INTERVAL)
            await loop.run_in_executor(None, _run_sweep)
        except asyncio.CancelledError:
            print("📈 [BreadthAgent] Stopped.")
            break
        except Exception as e:
            print(f"❌ [BreadthAgent] Loop error: {e}")
            await asyncio.sleep(300)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== BreadthAgent Standalone Test ===\n")
    _run_sweep()
    ctx     = wctx.get()
    breadth = ctx["breadth"]
    print(f"\nHealth:       {breadth['health']}")
    print(f"Rotation:     {breadth.get('rotation','?')}")
    print(f"VIX:          {breadth['vix']:.1f}  ({breadth['vix_term_structure']})")
    print(f"A/D Ratio:    {breadth['ad_ratio']:.1f}/10")
    print(f"Leading:      {breadth.get('leading_sectors', [])}")
    print(f"Lagging:      {breadth.get('lagging_sectors', [])}")
    print(f"\nSummary: {breadth['summary']}")
    print(f"\n--- Prompt section preview ---")
    print(wctx.build_prompt_section())
