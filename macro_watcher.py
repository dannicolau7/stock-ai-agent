"""
macro_watcher.py — Macro environment intelligence agent.

Runs every 60 minutes (and once at startup). Fetches from yfinance:
  - Treasury yields (2y, 10y, 30y) → yield curve shape
  - VIX → fear level and term structure
  - DXY (dollar index) → currency strength
  - Oil (WTI) and Gold → commodity regime
  - SPY, QQQ, IWM → relative performance across cap sizes

Uses Claude Haiku to synthesize all macro signals into:
  - Macro regime label (BULL / BEAR / STAGFLATION / RECOVERY / etc.)
  - Fed stance (HAWKISH / DOVISH / NEUTRAL / PAUSED)
  - Overall bias for signal decisions
  - Natural language summary

Stored in world_context.macro and injected into every Claude prompt.
"""

import asyncio
import json

import yfinance as yf
import anthropic

from config import ANTHROPIC_API_KEY
import world_context as wctx

MACRO_INTERVAL = 60 * 60   # 1 hour between updates

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_macro_data() -> dict:
    """
    Fetch macro indicators from yfinance in one batch download.
    Returns raw numbers — no interpretation yet.
    """
    symbols = {
        "yield_10y": "^TNX",     # 10-year treasury yield
        "yield_2y":  "^IRX",     # 13-week T-bill (proxy for 2y / Fed funds rate)
        "yield_30y": "^TYX",     # 30-year treasury
        "vix":       "^VIX",     # volatility index
        "vix3m":     "^VIX3M",   # 3-month VIX (term structure)
        "dxy":       "DX-Y.NYB", # US dollar index
        "oil":       "CL=F",     # WTI crude oil
        "gold":      "GC=F",     # gold futures
        "spy":       "SPY",
        "qqq":       "QQQ",
        "iwm":       "IWM",      # small caps
        "ief":       "IEF",      # 7-10y Treasury ETF (proxy for bond demand)
    }

    tickers_str = " ".join(symbols.values())
    try:
        df = yf.download(
            tickers_str,
            period="5d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"⚠️  [MacroAgent] yfinance download failed: {e}")
        return {}

    result = {}
    for key, ticker in symbols.items():
        try:
            t_df = df[ticker] if len(symbols) > 1 else df
            if t_df is None or t_df.empty:
                continue
            closes = t_df["Close"].dropna()
            if len(closes) >= 2:
                result[key]             = float(closes.iloc[-1])
                result[f"{key}_prev"]   = float(closes.iloc[-2])
                result[f"{key}_chg"]    = float(closes.iloc[-1] - closes.iloc[-2])
                result[f"{key}_chg_pct"] = float(
                    (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100
                )
            elif len(closes) == 1:
                result[key] = float(closes.iloc[-1])
        except Exception:
            continue

    return result


def _analyze_with_claude(data: dict) -> dict:
    """
    Send macro data to Claude Haiku for regime classification and assessment.
    """
    if not data:
        return {}

    # Build the data section
    yield_10y   = data.get("yield_10y", 0)
    yield_2y    = data.get("yield_2y", 0)
    yield_30y   = data.get("yield_30y", 0)
    curve_bps   = round((yield_10y - yield_2y) * 100, 1) if yield_10y and yield_2y else 0
    vix         = data.get("vix", 0)
    vix3m       = data.get("vix3m", 0)
    vix_term    = "BACKWARDATION (fear)" if vix > vix3m > 0 else "CONTANGO (calm)" if vix3m > vix > 0 else "FLAT"
    dxy         = data.get("dxy", 0)
    oil         = data.get("oil", 0)
    gold        = data.get("gold", 0)
    spy_chg     = data.get("spy_chg_pct", 0)
    qqq_chg     = data.get("qqq_chg_pct", 0)
    iwm_chg     = data.get("iwm_chg_pct", 0)

    prompt = f"""You are a macro strategist. Analyze these market indicators and classify the current macro environment.

CURRENT MACRO DATA:
Treasury Yields:
  10-year: {yield_10y:.2f}%
  2-year (proxy): {yield_2y:.2f}%
  30-year: {yield_30y:.2f}%
  Yield Curve (10y-2y): {curve_bps:+.0f} bps {"(INVERTED - recession signal)" if curve_bps < 0 else "(normal)"}

Fear & Liquidity:
  VIX (30-day vol): {vix:.1f}
  VIX 3-month: {vix3m:.1f}
  VIX Term Structure: {vix_term}

Currency & Commodities:
  DXY (Dollar Index): {dxy:.1f}
  WTI Oil: ${oil:.1f}
  Gold: ${gold:.0f}

Equity Market Performance (today):
  SPY (Large Cap): {spy_chg:+.2f}%
  QQQ (Tech): {qqq_chg:+.2f}%
  IWM (Small Cap): {iwm_chg:+.2f}%

Classify the current macro regime and provide trading bias.

Respond ONLY with valid JSON:
{{
  "regime": "BULL|BEAR|STAGFLATION|RECOVERY|RISK_OFF|NEUTRAL",
  "fed_stance": "HAWKISH|DOVISH|NEUTRAL|PAUSED",
  "bias": "BULLISH|BEARISH|NEUTRAL",
  "yield_curve_signal": "INVERTED|FLAT|STEEP|NORMAL",
  "risk_environment": "RISK_ON|RISK_OFF|NEUTRAL",
  "sector_implications": {{
    "favor": ["XLE", "XLV"],
    "avoid": ["XLK", "ARKK"]
  }},
  "summary": "2-3 sentence plain English assessment of current macro environment and what it means for stock picking"
}}"""

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if "```" in text:
            parts = text.split("```")
            text  = parts[1].lstrip("json").strip() if len(parts) >= 3 else text
        return json.loads(text)
    except Exception as e:
        print(f"⚠️  [MacroAgent] Claude error: {e}")
        return {}


# ── Single sweep ───────────────────────────────────────────────────────────────

def _run_sweep() -> bool:
    """Fetch macro data, analyze, update world_context."""
    print("📊 [MacroAgent] Fetching macro indicators...")
    data = _fetch_macro_data()
    if not data:
        print("📊 [MacroAgent] No data returned")
        return False

    yield_10y = data.get("yield_10y", 0)
    yield_2y  = data.get("yield_2y",  0)
    vix       = data.get("vix", 0)
    oil       = data.get("oil", 0)
    gold      = data.get("gold", 0)
    dxy       = data.get("dxy",  0)
    curve_bps = round((yield_10y - yield_2y) * 100, 1) if yield_10y and yield_2y else 0

    print(f"📊 [MacroAgent] 10y={yield_10y:.2f}%  2y={yield_2y:.2f}%  "
          f"curve={curve_bps:+.0f}bp  VIX={vix:.1f}  Oil=${oil:.1f}  Gold=${gold:.0f}")

    print("📊 [MacroAgent] Analyzing with Claude Haiku...")
    analysis = _analyze_with_claude(data)
    if not analysis:
        # Store raw data even without Claude analysis
        wctx.update_macro({
            "yield_10y":  yield_10y,
            "yield_2y":   yield_2y,
            "yield_curve": curve_bps,
            "vix":        vix,
            "dxy":        dxy,
            "oil":        oil,
            "gold":       gold,
        })
        return False

    regime      = analysis.get("regime", "NEUTRAL")
    fed_stance  = analysis.get("fed_stance", "UNKNOWN")
    bias        = analysis.get("bias", "NEUTRAL")
    summary     = analysis.get("summary", "")
    sector_impl = analysis.get("sector_implications", {})

    wctx.update_macro({
        "regime":      regime,
        "fed_stance":  fed_stance,
        "bias":        bias,
        "summary":     summary,
        "yield_10y":   yield_10y,
        "yield_2y":    yield_2y,
        "yield_curve": curve_bps,
        "vix":         vix,
        "dxy":         dxy,
        "oil":         oil,
        "gold":        gold,
    })

    bias_icon = "🟢" if bias == "BULLISH" else "🔴" if bias == "BEARISH" else "🟡"
    print(f"📊 [MacroAgent] {bias_icon} Regime={regime}  Fed={fed_stance}  Bias={bias}")
    if summary:
        print(f"📊 [MacroAgent] {summary[:120]}")
    return True


# ── Main loop ──────────────────────────────────────────────────────────────────

async def macro_watcher_loop():
    """Async loop started in main.py lifespan. Runs immediately then hourly."""
    print(f"📊 [MacroAgent] Started — updating every {MACRO_INTERVAL//60}min")
    loop = asyncio.get_running_loop()

    # Run immediately at startup
    await loop.run_in_executor(None, _run_sweep)

    while True:
        try:
            await asyncio.sleep(MACRO_INTERVAL)
            await loop.run_in_executor(None, _run_sweep)
        except asyncio.CancelledError:
            print("📊 [MacroAgent] Stopped.")
            break
        except Exception as e:
            print(f"❌ [MacroAgent] Loop error: {e}")
            await asyncio.sleep(300)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== MacroAgent Standalone Test ===\n")
    _run_sweep()
    ctx   = wctx.get()
    macro = ctx["macro"]
    print(f"\nRegime:      {macro['regime']}")
    print(f"Fed Stance:  {macro['fed_stance']}")
    print(f"Bias:        {macro['bias']}")
    print(f"10y Yield:   {macro['yield_10y']:.2f}%")
    print(f"2y Yield:    {macro['yield_2y']:.2f}%")
    print(f"Curve:       {macro['yield_curve']:+.0f}bp")
    print(f"VIX:         {macro['vix']:.1f}")
    print(f"Oil:         ${macro['oil']:.1f}")
    print(f"Gold:        ${macro['gold']:.0f}")
    print(f"\nSummary: {macro['summary']}")
    print(f"\n--- Prompt section preview ---")
    print(wctx.build_prompt_section())
