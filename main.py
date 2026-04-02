"""
main.py — entry point for the Stock AI Agent.

Usage:
  python main.py                         # monitor TICKER from .env
  python main.py --ticker AAPL           # override ticker
  python main.py --ticker BZAI --paper   # paper trading (no real alerts)
  python main.py --interval 120          # check every 2 minutes
  python main.py --port 8080             # dashboard on custom port

Features:
  - Market hours awareness (9:30 AM - 4:00 PM EST)
  - Signal memory + stop loss / target monitoring
  - Daily summary report at 4:30 PM EST
  - Paper trading mode (logs only, no SMS/push)
  - FastAPI dashboard at http://localhost:{port}
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

import config
from graph import GRAPH, make_initial_state
import watchlist_manager as wl
import logger


# ── CLI args ───────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Stock AI Agent")
    p.add_argument("--ticker",   default=None,  help="Ticker symbol (overrides .env TICKER)")
    p.add_argument("--interval", type=int, default=300, help="Scan interval seconds (default 300 = 5 min)")
    p.add_argument("--paper",    action="store_true",   help="Paper trading — no real SMS/push alerts")
    p.add_argument("--port",     type=int, default=8000, help="Dashboard port (default 8000)")
    # Watchlist management (these exit immediately without starting the server)
    p.add_argument("--add",    metavar="TICKER", default=None, help="Add ticker to watchlist and exit")
    p.add_argument("--remove", metavar="TICKER", default=None, help="Remove ticker from watchlist and exit")
    p.add_argument("--list",   action="store_true",            help="List watchlist tickers and exit")
    args, _ = p.parse_known_args()
    return args

_args    = _parse_args()

# ── Handle watchlist management commands immediately (no server needed) ────────
if _args.add:
    wl.add(_args.add)
    raise SystemExit(0)
if _args.remove:
    wl.remove(_args.remove)
    raise SystemExit(0)
if _args.list:
    wl.list_tickers()
    raise SystemExit(0)

# Resolve active ticker: CLI flag → .env → first watchlist entry
_watchlist = wl.load()
TICKER   = _args.ticker or config.TICKER or (_watchlist[0] if _watchlist else "BZAI")
INTERVAL = _args.interval
PAPER    = _args.paper
PORT     = _args.port
EST      = ZoneInfo("America/New_York")


# ── Market hours helpers ───────────────────────────────────────────────────────

def is_market_open() -> bool:
    now = datetime.now(tz=EST)
    if now.weekday() >= 5:            # Saturday / Sunday
        return False
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_t <= now <= close_t


def is_report_window() -> bool:
    """True for INTERVAL seconds after 4:30 PM EST on weekdays."""
    now = datetime.now(tz=EST)
    if now.weekday() >= 5:
        return False
    report_t = now.replace(hour=16, minute=30, second=0, microsecond=0)
    delta = (now - report_t).total_seconds()
    return 0 <= delta < INTERVAL


# ── Shared in-memory state ─────────────────────────────────────────────────────

latest_state:     dict  = {}
signal_history:   list  = []     # rolling last 200 runs
latest_bars:      list  = []
latest_news:      list  = []

# {ticker: {"signal", "price", "stop_loss", "targets", "timestamp"}}
signal_memory:    dict  = {}

# Signals fired today (reset each day)
daily_log:        list  = []
report_sent_date: str   = ""      # "YYYY-MM-DD" of last report

_executor = ThreadPoolExecutor(max_workers=1)


# ── Graph execution ────────────────────────────────────────────────────────────

def _run_sync(ticker: str, paper: bool) -> dict:
    return GRAPH.invoke(make_initial_state(ticker, paper_trading=paper))


def _store_result(result: dict):
    global latest_state, signal_history, latest_bars, latest_news

    latest_bars  = result.get("bars", [])
    latest_news  = result.get("raw_news", [])

    latest_state = {
        k: v for k, v in result.items()
        if k not in ("bars", "snapshot", "raw_news")
    }
    latest_state["timestamp"] = datetime.now().isoformat()

    signal_history.append({
        "timestamp":  latest_state["timestamp"],
        "price":      latest_state.get("current_price", 0),
        "signal":     latest_state.get("signal", "HOLD"),
        "confidence": latest_state.get("confidence", 0),
        "rsi":        latest_state.get("rsi", 50),
    })
    if len(signal_history) > 200:
        signal_history.pop(0)


def _update_signal_memory(result: dict):
    ticker  = result.get("ticker", TICKER)
    signal  = result.get("signal", "HOLD")
    price   = result.get("current_price", 0.0)

    if signal in ("BUY", "SELL"):
        signal_memory[ticker] = {
            "signal":    signal,
            "price":     price,
            "stop_loss": result.get("stop_loss", 0.0),
            "targets":   result.get("targets", []),
            "timestamp": datetime.now().isoformat(),
        }
        daily_log.append({
            "signal":     signal,
            "ticker":     ticker,
            "price":      price,
            "confidence": result.get("confidence", 0),
            "timestamp":  datetime.now().isoformat(),
        })
        # Persist signal to CSV log
        logger.log_signal(result)


def _check_exits(ticker: str, price: float) -> str | None:
    mem = signal_memory.get(ticker)
    if not mem or mem.get("signal") != "BUY":
        return None
    stop    = mem.get("stop_loss", 0.0)
    targets = [t for t in mem.get("targets", []) if t > 0]
    if stop and price <= stop:
        return "STOP_LOSS"
    if targets and price >= min(targets):
        return "TARGET_HIT"
    return None


async def run_once():
    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run_sync, TICKER, PAPER)

    _store_result(result)
    _update_signal_memory(result)

    # Stop loss / target monitoring
    price       = result.get("current_price", 0.0)
    exit_reason = _check_exits(TICKER, price)
    if exit_reason:
        icon = "🛑" if exit_reason == "STOP_LOSS" else "🎯"
        print(f"{icon} [Monitor] {exit_reason} triggered for {TICKER} @ ${price:.4f}")
        if not PAPER:
            try:
                from alerts import send_push, send_sms
                label = "🛑 STOP LOSS HIT" if exit_reason == "STOP_LOSS" else "🎯 TARGET HIT"
                msg   = f"{label}\n{TICKER} @ ${price:.4f}"
                send_push(f"{exit_reason} — {TICKER}", msg, priority=1)
                send_sms(msg)
            except Exception as e:
                print(f"❌ [Monitor] Exit alert failed: {e}")
        signal_memory.pop(TICKER, None)

    sig  = result.get("signal", "HOLD")
    conf = result.get("confidence", 0)
    icon = "🟢" if sig == "BUY" else "🔴" if sig == "SELL" else "🟡"
    print(
        f"{icon} [Monitor] {sig}  conf={conf}/100  "
        f"${price:.4f}  RSI={result.get('rsi', 0):.1f}"
        + ("  📋 PAPER" if PAPER else "")
    )


# ── Daily report ───────────────────────────────────────────────────────────────

async def _send_daily_report():
    today    = datetime.now(tz=EST).strftime("%Y-%m-%d")
    today_signals = [e for e in daily_log if e["timestamp"][:10] == today]

    lines = [f"📅 Daily Report — {TICKER} — {today}"]
    if not today_signals:
        lines.append("No BUY/SELL signals fired today.")
    else:
        lines.append(f"Signals fired: {len(today_signals)}")
        for s in today_signals:
            icon = "🟢" if s["signal"] == "BUY" else "🔴"
            lines.append(f"  {icon} {s['signal']} @ ${s['price']:.4f}  conf={s['confidence']}")
    if PAPER:
        lines.append("\n📋 PAPER TRADING MODE")

    report = "\n".join(lines)
    print("\n" + "─" * 50)
    print(report)
    print("─" * 50 + "\n")

    if not PAPER:
        try:
            from alerts import send_push, send_sms
            send_push(f"Daily Report — {TICKER}", report)
            send_sms(report)
        except Exception as e:
            print(f"❌ [Monitor] Daily report delivery failed: {e}")


# ── Monitoring loop ────────────────────────────────────────────────────────────

async def monitoring_loop():
    global report_sent_date
    mode = "📋 PAPER" if PAPER else "🔴 LIVE"
    print(f"🚀 Stock AI Agent starting  [{mode}]  ticker={TICKER}  interval={INTERVAL}s")
    print(f"📊 Dashboard → http://localhost:{PORT}")

    # Print watchlist on startup
    saved = wl.load()
    if saved:
        print(f"📋 Watchlist: {', '.join(saved)}")
    print()

    while True:
        try:
            if is_market_open():
                await run_once()
            else:
                now_est  = datetime.now(tz=EST)
                now_date = now_est.strftime("%Y-%m-%d")
                if is_report_window() and report_sent_date != now_date:
                    await _send_daily_report()
                    report_sent_date = now_date
                else:
                    print(
                        f"🕐 [Monitor] Market closed "
                        f"({now_est.strftime('%a %H:%M EST')}) — "
                        f"next check in {INTERVAL}s"
                    )
        except Exception as e:
            print(f"❌ [Monitor] Loop error: {e}")

        await asyncio.sleep(INTERVAL)


# ── FastAPI app ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(monitoring_loop())
    yield
    task.cancel()


app = FastAPI(title=f"Stock AI Agent — {TICKER}", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    try:
        with open("dashboard/index.html") as f:
            return f.read()
    except FileNotFoundError:
        return "<h2>dashboard/index.html not found</h2>"


@app.get("/api/state")
async def api_state():
    return JSONResponse({
        "state":        latest_state,
        "history":      signal_history,
        "paper_trading": PAPER,
        "ticker":       TICKER,
        "market_open":  is_market_open(),
    })


@app.get("/api/bars")
async def api_bars():
    tv_bars = [
        {
            "time":   b.get("t", 0) // 1000,
            "open":   b.get("o"),
            "high":   b.get("h"),
            "low":    b.get("l"),
            "close":  b.get("c"),
            "volume": b.get("v"),
        }
        for b in latest_bars
    ]
    return JSONResponse({"bars": tv_bars, "ticker": TICKER})


@app.get("/api/news")
async def api_news():
    news = [
        {
            "title":     n.get("title", ""),
            "url":       n.get("article_url", ""),
            "published": (n.get("published_utc") or "")[:10],
            "source":    (n.get("publisher") or {}).get("name", ""),
        }
        for n in latest_news[:10]
    ]
    return JSONResponse({"news": news, "ticker": TICKER})


@app.get("/api/watchlist")
async def api_watchlist():
    return JSONResponse({
        "watchlist": [{
            "ticker":     TICKER,
            "signal":     latest_state.get("signal", "HOLD"),
            "confidence": latest_state.get("confidence", 0),
            "price":      latest_state.get("current_price", 0),
            "rsi":        latest_state.get("rsi", 50),
        }],
        "memory": signal_memory,
    })


@app.post("/api/run")
async def api_trigger_run():
    asyncio.create_task(run_once())
    return {"status": "triggered", "ticker": TICKER}


@app.get("/api/log")
async def api_log():
    """Return last 100 rows of signals_log.csv for the dashboard."""
    return JSONResponse({"log": logger.read_log(limit=100)})


@app.get("/api/watchlist/saved")
async def api_watchlist_saved():
    """Return the persisted watchlist.json tickers."""
    return JSONResponse({"tickers": wl.load()})


@app.post("/api/watchlist/add/{ticker}")
async def api_watchlist_add(ticker: str):
    updated = wl.add(ticker.upper())
    return {"tickers": updated}


@app.delete("/api/watchlist/remove/{ticker}")
async def api_watchlist_remove(ticker: str):
    updated = wl.remove(ticker.upper())
    return {"tickers": updated}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
