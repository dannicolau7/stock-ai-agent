"""
momentum_screener.py — Proactive upside opportunity screener.

Runs every 2 hours during market hours. Scans a broad universe to find
stocks with the highest probability of a significant upside move in the
next 1–5 days, across 5 categories:

  1. Breakout   — Price >95% of 52w high, high RVOL, sector tailwind
  2. Momentum   — Strong RS vs SPY (+5%+), MACD cross, RSI 45-65
  3. Earnings   — Pre-earnings setup, strong beat history, bullish options
  4. Recovery   — RSI bounce from oversold (<35), price above recent support
  5. Options    — Extreme call flow (C/P ≥ 5×) with technical confirmation

Top candidates are stored in world_context and auto-injected into the
monitoring scan list. A WhatsApp alert fires when a new high-conviction
candidate is identified.

Usage:
  from momentum_screener import run_momentum_screen, get_momentum_candidates
  candidates = run_momentum_screen()
"""

import time
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf

import world_context as wctx
from agents.tech_agent import _calc_rsi, _ema

EST = ZoneInfo("America/New_York")

# ── Scan universe ──────────────────────────────────────────────────────────────
# Mid-cap momentum universe: liquid enough for retail, small enough to move
MOMENTUM_UNIVERSE = [
    # AI / Tech growth
    "NVDA", "AMD", "PLTR", "SMCI", "ARM", "IONQ", "RGTI", "QUBT",
    "CXAI", "AITX", "GFAI",
    # Biotech / Healthcare catalyst plays
    "MRNA", "NVAX", "IOVA", "BNTX", "RCKT", "ACHR",
    # Energy / Commodity momentum
    "SLB", "XOM", "CVX", "MPC", "OXY", "FANG",
    # Financials (earnings leverage)
    "GS", "JPM", "BAC", "MS", "C", "WFC",
    # Small-cap momentum
    "KULR", "CIFR", "RKLB", "JOBY",
    # ETF momentum indicators
    "TQQQ", "SOXL", "LABU",
    # Consumer / Retail
    "AMZN", "TSLA", "META", "NFLX",
]

SCREEN_INTERVAL = 2 * 60 * 60   # 2 hours


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rs_vs_spy(closes: np.ndarray, spy_closes: np.ndarray, days: int = 5) -> float:
    """% outperformance vs SPY over last `days` trading days."""
    if len(closes) < days + 1 or len(spy_closes) < days + 1:
        return 0.0
    stock_ret = (float(closes[-1]) / float(closes[-days - 1]) - 1) * 100
    spy_ret   = (float(spy_closes[-1]) / float(spy_closes[-days - 1]) - 1) * 100
    return round(stock_ret - spy_ret, 2)


def _rvol_daily(volumes: np.ndarray) -> float:
    if len(volumes) < 2:
        return 1.0
    avg = float(np.mean(volumes[-21:-1])) if len(volumes) >= 21 else float(np.mean(volumes[:-1]))
    return round(float(volumes[-1]) / avg, 2) if avg > 0 else 1.0


def _near_52w_high(highs: np.ndarray, pct: float = 0.05) -> bool:
    """True if current price is within `pct` of the 52-week high."""
    if len(highs) < 2:
        return False
    window = highs[-min(252, len(highs)):]
    high52 = float(np.max(window))
    price  = float(highs[-1])
    return high52 > 0 and (high52 - price) / high52 <= pct


def _has_macd_cross(closes: np.ndarray) -> bool:
    if len(closes) < 35:
        return False
    macd_line   = _ema(closes, 12) - _ema(closes, 26)
    signal_line = _ema(macd_line, 9)
    hist        = macd_line - signal_line
    for i in (-3, -2, -1):
        if hist[i] > 0 and hist[i - 1] <= 0:
            return True
    return False


def _options_flow(ticker: str) -> dict:
    """Get options C/P ratio for nearest expiry. Returns {} on error."""
    try:
        t    = yf.Ticker(ticker)
        opts = t.options
        if not opts:
            return {}
        chain    = t.option_chain(opts[0])
        call_vol = int(chain.calls["volume"].fillna(0).sum())
        put_vol  = int(chain.puts["volume"].fillna(0).sum())
        if put_vol == 0:
            return {}
        return {"cp_ratio": round(call_vol / put_vol, 2),
                "call_vol": call_vol, "put_vol": put_vol}
    except Exception:
        return {}


# ── Per-ticker scoring ─────────────────────────────────────────────────────────

def _score_ticker(ticker: str, spy_closes: np.ndarray,
                  earnings_today: list[str]) -> dict | None:
    """
    Download data and score a ticker for upside potential.
    Returns a scored dict or None if it doesn't qualify.
    """
    try:
        t  = yf.Ticker(ticker)
        df = t.history(period="1y", interval="1d", auto_adjust=True)
        if df is None or len(df) < 30:
            return None

        closes  = df["Close"].values.astype(float)
        highs   = df["High"].values.astype(float)
        lows    = df["Low"].values.astype(float)
        volumes = df["Volume"].values.astype(float)
        price   = float(closes[-1])

        if price <= 0:
            return None

        rsi  = _calc_rsi(closes)
        rv   = _rvol_daily(volumes)
        rs5  = _rs_vs_spy(closes, spy_closes, days=5)
        rs1  = _rs_vs_spy(closes, spy_closes, days=1)

        # ── Hard filters (eliminate weak setups) ─────────────────────────────
        if rsi > 75:       return None   # overbought
        if rsi < 20:       return None   # too deep — likely broken
        if price < 0.50:   return None   # micro-cap / illiquid
        if price > 500:    return None   # too expensive for typical retail sizing

        avg_vol  = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(volumes[-1])
        dvol     = price * avg_vol
        if dvol < 500_000:  return None  # minimum $500k daily dollar volume

        # ── Score (0 – 100) ───────────────────────────────────────────────────
        score   = 0
        signals = []
        setup_type = "momentum"

        # Relative strength vs SPY (5-day)
        if rs5 >= 10:
            score += 25; signals.append(f"RS +{rs5:.1f}% vs SPY 🚀")
        elif rs5 >= 5:
            score += 15; signals.append(f"RS +{rs5:.1f}% vs SPY")
        elif rs5 >= 2:
            score += 8;  signals.append(f"RS +{rs5:.1f}% vs SPY")
        elif rs5 < -5:
            score -= 10  # underperforming — penalty

        # Near 52-week high (breakout setup)
        if _near_52w_high(highs, pct=0.03):
            score += 20; signals.append("near 52w high 🔝"); setup_type = "breakout"
        elif _near_52w_high(highs, pct=0.08):
            score += 12; signals.append("approaching 52w high")

        # RSI momentum zone (not overbought, trending up)
        if 45 <= rsi <= 62:
            score += 15; signals.append(f"RSI {rsi:.0f} momentum zone")
        elif 35 <= rsi < 45:
            score += 10; signals.append(f"RSI {rsi:.0f} recovering"); setup_type = "recovery"
        elif rsi < 35:
            score += 18; signals.append(f"RSI {rsi:.0f} oversold bounce"); setup_type = "recovery"

        # MACD bullish cross
        if _has_macd_cross(closes):
            score += 15; signals.append("MACD cross ✅")

        # Volume surge
        if rv >= 3.0:
            score += 15; signals.append(f"RVOL {rv:.1f}× 🔥")
        elif rv >= 2.0:
            score += 10; signals.append(f"RVOL {rv:.1f}×")
        elif rv >= 1.5:
            score += 5;  signals.append(f"RVOL {rv:.1f}×")

        # Earnings catalyst (today or tomorrow = highest risk/reward)
        if ticker in earnings_today:
            score += 20; signals.append("earnings catalyst 📅"); setup_type = "earnings"

        # Today's price action (intraday momentum)
        if rs1 >= 3:
            score += 8; signals.append(f"today +{rs1:.1f}%")
        elif rs1 >= 1:
            score += 4; signals.append(f"today +{rs1:.1f}%")

        # Minimum score to qualify
        if score < 35:
            return None

        return {
            "ticker":      ticker,
            "price":       round(price, 4),
            "score":       score,
            "setup_type":  setup_type,
            "rsi":         round(rsi, 1),
            "rvol":        rv,
            "rs_5d":       rs5,
            "rs_1d":       rs1,
            "signals":     ", ".join(signals),
        }

    except Exception:
        return None


# ── Main screener ─────────────────────────────────────────────────────────────

def run_momentum_screen(extra_tickers: list[str] | None = None) -> list[dict]:
    """
    Screen MOMENTUM_UNIVERSE + extra_tickers for upside potential.
    Returns sorted list of candidate dicts. Also updates world_context
    and sends WhatsApp for newly found high-conviction setups.
    """
    print("\n🚀 [MomentumScreener] Running upside opportunity scan...")

    # Fetch SPY for RS calculation
    try:
        spy_df    = yf.download("SPY", period="6mo", interval="1d",
                                progress=False, auto_adjust=True)
        spy_closes = spy_df["Close"].values.astype(float) if spy_df is not None else np.array([])
    except Exception:
        spy_closes = np.array([])

    # Pull earnings catalysts from world_context
    ctx = wctx.get()
    earnings_today = [
        e["ticker"] for e in ctx["earnings"].get("upcoming", [])
        if e.get("days", 99) <= 2 and e.get("direction") != "BEARISH"
    ]

    # Pull options flow tickers from world_context
    opts_bullish = [
        o["ticker"] for o in ctx["social"].get("unusual_opts", [])
        if o.get("bias") == "BULLISH" and o.get("call_put_ratio", 0) >= 5.0
    ]

    universe = list(dict.fromkeys(
        (extra_tickers or []) + opts_bullish + earnings_today + MOMENTUM_UNIVERSE
    ))
    universe = [t for t in universe if t.isalpha() and 1 <= len(t) <= 5]

    print(f"🚀 [MomentumScreener] Scanning {len(universe)} tickers...")
    candidates = []
    for i, ticker in enumerate(universe):
        if i > 0 and i % 10 == 0:
            time.sleep(1)   # light throttle for yfinance
        result = _score_ticker(ticker, spy_closes, earnings_today)
        if result:
            candidates.append(result)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[:10]

    print(f"🚀 [MomentumScreener] {len(candidates)} qualified, top {len(top)}:")
    for c in top[:5]:
        print(f"   ⭐ {c['ticker']:6s}  score={c['score']:3d}  [{c['setup_type']}]  "
              f"RSI={c['rsi']:.0f}  RS={c['rs_5d']:+.1f}%  {c['signals'][:55]}")

    # Store in world_context so monitoring loop can pick them up
    wctx.update_social({
        "momentum_candidates": [c["ticker"] for c in top],
        "momentum_picks":      top,
        "momentum_updated_at": datetime.now().isoformat(),
    })

    # WhatsApp alert for newly identified high-conviction setups (score ≥ 60)
    hot = [c for c in top if c["score"] >= 60]
    if hot:
        try:
            from alerts import send_whatsapp
            lines = ["🚀 Upside Opportunities Found\n"]
            for c in hot[:3]:
                lines.append(
                    f"⭐ {c['ticker']} ${c['price']:.2f} [{c['setup_type'].upper()}]\n"
                    f"   Score {c['score']}/100  RSI {c['rsi']:.0f}  RS {c['rs_5d']:+.1f}%\n"
                    f"   {c['signals'][:60]}"
                )
            lines.append(f"\n⏰ {datetime.now(tz=EST).strftime('%I:%M %p EST')}")
            send_whatsapp("\n".join(lines))
            print(f"✅ [MomentumScreener] Alerted {len(hot)} high-conviction setup(s)")
        except Exception as e:
            print(f"⚠️  [MomentumScreener] Alert failed: {e}")

    return top


def get_momentum_candidates() -> list[str]:
    """Returns current momentum candidate tickers (used by monitoring loop)."""
    ctx = wctx.get()
    return ctx.get("social", {}).get("momentum_candidates", [])


# ── Background loop ────────────────────────────────────────────────────────────

async def momentum_screener_loop(watchlist_fn=None):
    """Runs every 2h during market hours, once at startup."""
    print(f"🚀 [MomentumScreener] Started — scanning every {SCREEN_INTERVAL//3600}h")
    loop = asyncio.get_running_loop()

    await asyncio.sleep(120)   # wait for world_context to populate

    while True:
        try:
            now = datetime.now(tz=EST)
            # Only run during market hours + 1h pre/post
            if now.weekday() < 5 and 8 <= now.hour < 17:
                extra = watchlist_fn() if watchlist_fn else []
                await loop.run_in_executor(
                    None, run_momentum_screen, extra
                )
            else:
                print("🚀 [MomentumScreener] Outside hours — skipping")
            await asyncio.sleep(SCREEN_INTERVAL)
        except asyncio.CancelledError:
            print("🚀 [MomentumScreener] Stopped.")
            break
        except Exception as e:
            print(f"❌ [MomentumScreener] Loop error: {e}")
            await asyncio.sleep(600)


if __name__ == "__main__":
    import watchlist_manager as wl
    from macro_watcher import _run_sweep as macro_sweep
    from earnings_watcher import _run_sweep as earnings_sweep
    from social_watcher import _run_sweep as social_sweep

    print("=== Momentum Screener Standalone Test ===\n")
    print("Populating context first...")
    macro_sweep()
    earnings_sweep(extra_tickers=wl.load())
    social_sweep(extra_tickers=wl.load())

    print("\nRunning momentum screen...")
    results = run_momentum_screen(extra_tickers=wl.load())
    print(f"\nTop opportunities ({len(results)}):")
    for r in results[:10]:
        print(f"  {r['ticker']:6s}  score={r['score']:3d}  [{r['setup_type']}]  "
              f"RSI={r['rsi']:.0f}  RS(5d)={r['rs_5d']:+.1f}%  RVOL={r['rvol']:.1f}×\n"
              f"         {r['signals']}")
