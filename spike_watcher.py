"""
spike_watcher.py — Real-time price + volume spike detection across all stocks.

Runs every 60 seconds during market hours. On each cycle it checks:
  - All watchlist tickers (always)
  - The current 200-ticker chunk of the full ~6,000-stock universe (rotating)
    → full universe covered every ~30 minutes

Uses a single yfinance batch download per cycle (zero Polygon API calls).
When a ticker spikes ≥ 2% on ≥ 2.5× average volume:

  1. Sends preliminary WhatsApp: "⚡ AWRE spiking +4.2% on 3.8× vol"
  2. Triggers full LangGraph pipeline with news_triggered=True
     (lowers confidence threshold from 65 → 55)

Alert latency: 30–50 min → < 2 min.
30-minute cooldown per ticker prevents re-triggering on the same move.
"""

import asyncio
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf

EST              = ZoneInfo("America/New_York")
SPIKE_INTERVAL   = 60       # seconds between cycles
PRICE_SPIKE_PCT  = 0.02     # 2% move in last 1-min bar
VOL_SPIKE_RATIO  = 2.5      # volume ≥ 2.5× bar average
SPIKE_COOLDOWN_S = 30 * 60  # 30-min cooldown per ticker
CHUNK_SIZE       = 50       # tickers per cycle — keeps FD usage manageable

_spike_alerted:  dict = {}  # {ticker: time.time()} of last spike trigger
_universe_cache: list = []  # loaded once at startup
_chunk_idx:      int  = 0   # current position in universe rotation


# ── Universe loader ────────────────────────────────────────────────────────────

def _load_full_universe() -> list:
    """Load the full ticker universe from market_scanner's cache."""
    global _universe_cache
    if _universe_cache:
        return _universe_cache
    try:
        from market_scanner import _load_universe
        _universe_cache = _load_universe()
        print(f"⚡ [SpikeWatcher] Universe loaded: {len(_universe_cache):,} tickers")
    except Exception as e:
        print(f"⚡ [SpikeWatcher] Universe load failed: {e} — watchlist only")
        _universe_cache = []
    return _universe_cache


def _get_chunk(universe: list) -> list:
    """Return the next CHUNK_SIZE slice of the universe, rotating each call."""
    global _chunk_idx
    if not universe:
        return []
    start      = (_chunk_idx * CHUNK_SIZE) % len(universe)
    chunk      = universe[start : start + CHUNK_SIZE]
    _chunk_idx += 1
    return chunk


# ── Market hours ───────────────────────────────────────────────────────────────

def _market_open() -> bool:
    now = datetime.now(tz=EST)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


# ── Cooldown helpers ───────────────────────────────────────────────────────────

def _spike_in_cooldown(ticker: str) -> bool:
    last = _spike_alerted.get(ticker)
    return last is not None and (time.time() - last) < SPIKE_COOLDOWN_S


def _mark_spike(ticker: str):
    _spike_alerted[ticker] = time.time()


# ── Spike detection ────────────────────────────────────────────────────────────

def _fetch_spikes(tickers: list) -> list:
    """
    Single batch yfinance download for all tickers.
    Returns list of {ticker, price_chg_pct, vol_ratio, direction, price}.
    """
    if not tickers:
        return []
    try:
        df = yf.download(
            tickers,
            period="1d",
            interval="1m",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return []

        results = []
        single  = len(tickers) == 1

        for ticker in tickers:
            try:
                t_df = df if single else df.get(ticker)
                if t_df is None or t_df.empty:
                    continue

                t_df = t_df.dropna(subset=["Close", "Volume"])
                if len(t_df) < 3:
                    continue

                avg_vol    = float(t_df["Volume"].mean())
                last_bar   = t_df.iloc[-1]
                prev_bar   = t_df.iloc[-2]
                prev_close = float(prev_bar["Close"])

                if prev_close <= 0 or avg_vol <= 0:
                    continue

                price_chg = (float(last_bar["Close"]) - prev_close) / prev_close
                vol_ratio = float(last_bar["Volume"]) / avg_vol

                if abs(price_chg) >= PRICE_SPIKE_PCT and vol_ratio >= VOL_SPIKE_RATIO:
                    results.append({
                        "ticker":        ticker,
                        "price_chg_pct": price_chg * 100,
                        "vol_ratio":     vol_ratio,
                        "direction":     "UP" if price_chg > 0 else "DOWN",
                        "price":         float(last_bar["Close"]),
                    })

            except Exception:
                continue

        return results

    except Exception as e:
        print(f"⚠️  [SpikeWatcher] Batch fetch error: {e}")
        return []


# ── Main loop ──────────────────────────────────────────────────────────────────

async def spike_watcher_loop(run_once_fn, get_tickers_fn):
    """
    Async loop started in main.py lifespan.

    Args:
        run_once_fn:    main.run_once coroutine — called with (ticker, news_triggered=True)
        get_tickers_fn: callable returning current watchlist (e.g. wl.load)
    """
    print(f"⚡ [SpikeWatcher] Started — checking every {SPIKE_INTERVAL}s")

    loop = asyncio.get_running_loop()

    # Load universe once at startup (uses market_scanner's cache, fast if cache is fresh)
    universe = await loop.run_in_executor(None, _load_full_universe)
    total_chunks = max(1, len(universe) // CHUNK_SIZE)
    if universe:
        print(f"⚡ [SpikeWatcher] Full universe: {len(universe):,} tickers "
              f"in {total_chunks} chunks — full cycle every ~{total_chunks}min")

    while True:
        try:
            await asyncio.sleep(SPIKE_INTERVAL)

            if not _market_open():
                continue

            watchlist = get_tickers_fn()
            chunk     = _get_chunk(universe)

            # Watchlist always checked; universe chunk rotates each cycle
            to_check = list(dict.fromkeys(watchlist + chunk))

            spikes = await loop.run_in_executor(None, _fetch_spikes, to_check)

            for s in spikes:
                ticker    = s["ticker"]
                chg_pct   = s["price_chg_pct"]
                vol_ratio = s["vol_ratio"]
                direction = s["direction"]
                price     = s["price"]

                if _spike_in_cooldown(ticker):
                    continue

                _mark_spike(ticker)

                arrow = "📈" if direction == "UP" else "📉"
                sign  = "+" if chg_pct >= 0 else ""
                print(f"\n⚡ [SpikeWatcher] {ticker} spike: {sign}{chg_pct:.1f}%  vol={vol_ratio:.1f}×  — running pipeline...")

                # Trigger full pipeline immediately with news_triggered=True
                try:
                    await run_once_fn(ticker, news_triggered=True)
                except Exception as e:
                    print(f"❌ [SpikeWatcher] Pipeline error for {ticker}: {e}")

        except asyncio.CancelledError:
            print("⚡ [SpikeWatcher] Stopped.")
            break
        except Exception as e:
            print(f"❌ [SpikeWatcher] Loop error: {e}")
            await asyncio.sleep(30)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from top_movers import get_top_movers

    universe = _load_full_universe()
    watchlist = get_top_movers()
    chunk    = universe[:CHUNK_SIZE] if universe else watchlist
    to_check = list(dict.fromkeys(watchlist + chunk))

    print(f"Checking {len(to_check)} tickers (watchlist={len(watchlist)} + chunk={len(chunk)})...")
    spikes = _fetch_spikes(to_check)
    if spikes:
        for s in spikes:
            print(f"  ⚡ {s['ticker']}: {s['price_chg_pct']:+.1f}%  vol={s['vol_ratio']:.1f}×  ${s['price']:.2f}")
    else:
        print("  No spikes detected right now.")
