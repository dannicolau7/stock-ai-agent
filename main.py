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
from dataclasses import dataclass, field
from datetime import datetime
import math
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
    p.add_argument("--ticker",   nargs="+", default=None,  help="One or more ticker symbols (e.g. --ticker BZAI AWRE)")
    p.add_argument("--interval", type=int, default=300, help="Scan interval seconds (default 300 = 5 min)")
    p.add_argument("--paper",    action="store_true",   help="Paper trading — no real SMS/push alerts")
    p.add_argument("--port",     type=int, default=8000, help="Dashboard port (default 8000)")
    # Watchlist management (these exit immediately without starting the server)
    p.add_argument("--add",    metavar="TICKER", default=None, help="Add ticker to watchlist and exit")
    p.add_argument("--remove", metavar="TICKER", default=None, help="Remove ticker from watchlist and exit")
    p.add_argument("--list",   action="store_true",            help="List watchlist tickers and exit")
    args, _ = p.parse_known_args()
    return args

# ── Safe module-level defaults (populated by _setup_from_args at startup) ──────
TICKERS  = ["BZAI"]
TICKER   = "BZAI"
INTERVAL = 300
PAPER    = False
PORT     = 8000
EST      = ZoneInfo("America/New_York")


def _setup_from_args() -> None:
    """Parse CLI args and populate module-level config. Called only from __main__."""
    global TICKERS, TICKER, INTERVAL, PAPER, PORT

    args = _parse_args()

    # Handle watchlist management commands immediately (no server needed)
    if args.add:
        wl.add(args.add)
        raise SystemExit(0)
    if args.remove:
        wl.remove(args.remove)
        raise SystemExit(0)
    if args.list:
        wl.list_tickers()
        raise SystemExit(0)

    # Resolve tickers: CLI flag → .env → watchlist → default
    watchlist   = wl.load()
    cli_tickers = args.ticker
    if cli_tickers:
        TICKERS = [t.upper() for t in cli_tickers]
    elif watchlist:
        TICKERS = watchlist
    elif config.TICKER:
        TICKERS = [t.strip().upper() for t in config.TICKER.split() if t.strip()]
    else:
        TICKERS = ["BZAI"]

    TICKER   = TICKERS[0]
    INTERVAL = args.interval
    PAPER    = args.paper
    PORT     = args.port


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


# ── Runtime state ──────────────────────────────────────────────────────────────

@dataclass
class AppState:
    """
    All mutable runtime state in one place.
    Swap with AppState() in tests for full isolation — no module-level patching needed.
    """
    ticker_states:    dict = field(default_factory=dict)   # {ticker: state_dict}
    histories:        dict = field(default_factory=dict)   # {ticker: [last 200]}
    bars_map:         dict = field(default_factory=dict)   # {ticker: bars_list}
    news_map:         dict = field(default_factory=dict)   # {ticker: news_list}
    signal_memory:    dict = field(default_factory=dict)   # {ticker: {signal, price, ...}}
    daily_log:        list = field(default_factory=list)   # BUY/SELL events today
    report_sent_date: str  = ""                            # "YYYY-MM-DD"


_app_state = AppState()
_executor  = ThreadPoolExecutor(max_workers=4)   # supports parallel ticker scanning


# ── Graph execution ────────────────────────────────────────────────────────────

def _run_sync(ticker: str, paper: bool,
              already_alerted: bool = False,
              news_triggered: bool = False) -> dict:
    state = make_initial_state(ticker, paper_trading=paper)
    state["already_alerted"] = already_alerted
    state["news_triggered"]  = news_triggered
    return GRAPH.invoke(state)


def _store_result(result: dict):
    ticker = result.get("ticker", TICKER)

    bars  = result.get("bars", [])
    news  = result.get("raw_news", [])
    state = {k: v for k, v in result.items() if k not in ("bars", "snapshot", "raw_news")}
    state["timestamp"] = datetime.now().isoformat()

    _app_state.bars_map[ticker]       = bars
    _app_state.news_map[ticker]       = news
    _app_state.ticker_states[ticker]  = state

    hist = _app_state.histories.setdefault(ticker, [])
    hist.append({
        "timestamp":  state["timestamp"],
        "price":      state.get("current_price", 0),
        "signal":     state.get("signal", "HOLD"),
        "confidence": state.get("confidence", 0),
        "rsi":        state.get("rsi", 50),
    })
    if len(hist) > 200:
        hist.pop(0)


def _update_signal_memory(result: dict):
    ticker  = result.get("ticker", TICKER)
    signal  = result.get("signal", "HOLD")
    price   = result.get("current_price", 0.0)

    if signal in ("BUY", "SELL"):
        _app_state.signal_memory[ticker] = {
            "signal":    signal,
            "price":     price,
            "stop_loss": result.get("stop_loss", 0.0),
            "targets":   result.get("targets", []),
            "timestamp": datetime.now().isoformat(),
        }
        _app_state.daily_log.append({
            "signal":     signal,
            "ticker":     ticker,
            "price":      price,
            "confidence": result.get("confidence", 0),
            "timestamp":  datetime.now().isoformat(),
        })
        # Persist signal to CSV log
        logger.log_signal(result)


def _check_exits(ticker: str, price: float) -> str | None:
    mem = _app_state.signal_memory.get(ticker)
    if not mem or mem.get("signal") != "BUY":
        return None
    stop        = mem.get("stop_loss", 0.0)
    entry_price = mem.get("price", 0.0)
    # Only count targets that are genuinely above the entry price
    targets = [t for t in mem.get("targets", []) if t > entry_price]
    if stop and price <= stop:
        return "STOP_LOSS"
    if targets and price >= min(targets):
        return "TARGET_HIT"
    return None


async def run_once(ticker: str = None, news_triggered: bool = False):
    ticker = ticker or TICKER
    loop   = asyncio.get_running_loop()
    # Suppress re-alerting if this ticker already has an active BUY position
    already_alerted = ticker in _app_state.signal_memory
    result = await loop.run_in_executor(
        _executor, _run_sync, ticker, PAPER, already_alerted, news_triggered
    )

    _store_result(result)
    _update_signal_memory(result)

    # Stop loss / target monitoring
    price       = result.get("current_price", 0.0)
    exit_reason = _check_exits(ticker, price)
    if exit_reason:
        icon = "🛑" if exit_reason == "STOP_LOSS" else "🎯"
        print(f"{icon} [Monitor] {exit_reason} triggered for {ticker} @ ${price:.4f}")
        if not PAPER:
            try:
                from alerts import send_push, send_whatsapp
                label = "🛑 STOP LOSS HIT" if exit_reason == "STOP_LOSS" else "🎯 TARGET HIT"
                msg   = f"{label}\n{ticker} @ ${price:.4f}"
                send_push(f"{exit_reason} — {ticker}", msg)
                send_whatsapp(msg)
            except Exception as e:
                print(f"❌ [Monitor] Exit alert failed: {e}")
        _app_state.signal_memory.pop(ticker, None)

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
    today_signals = [e for e in _app_state.daily_log if e["timestamp"][:10] == today]

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
            from alerts import send_push, send_whatsapp
            send_push(f"Daily Report — {TICKER}", report)
            send_whatsapp(report)
        except Exception as e:
            print(f"❌ [Monitor] Daily report delivery failed: {e}")


# ── Monitoring loop ────────────────────────────────────────────────────────────

async def monitoring_loop():
    mode = "📋 PAPER" if PAPER else "🔴 LIVE"
    print(f"🚀 Stock AI Agent starting  [{mode}]  tickers={', '.join(TICKERS)}  interval={INTERVAL}s")
    print(f"📊 Dashboard → http://localhost:{PORT}")

    # Print watchlist on startup
    saved = wl.load()
    if saved:
        print(f"📋 Watchlist: {', '.join(saved)}")
    print()

    # Semaphore limits concurrent Polygon-hitting pipelines to protect free tier (5 req/min)
    _sem = asyncio.Semaphore(2)

    async def _run_gated(t: str):
        async with _sem:
            await run_once(t)

    while True:
        try:
            if is_market_open():
                # Merge static watchlist with discovery candidates (deduped)
                from discovery_agent import get_discovery_tickers
                discovery = get_discovery_tickers()
                scan_list = list(dict.fromkeys(TICKERS + discovery[:10]))
                await asyncio.gather(*[_run_gated(t) for t in scan_list])
            else:
                now_est = datetime.now(tz=EST)
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
    from scheduler import scheduler_loop
    from news_watcher import news_watcher_loop, yf_news_watcher_loop
    from spike_watcher import spike_watcher_loop
    from edgar_watcher import edgar_watcher_loop
    from geo_watcher        import geo_watcher_loop
    from macro_watcher      import macro_watcher_loop
    from earnings_watcher   import earnings_watcher_loop
    from breadth_watcher    import breadth_watcher_loop
    from social_watcher     import social_watcher_loop
    from discovery_agent    import discovery_agent_loop
    from portfolio_agent    import portfolio_agent_loop, init_positions_table
    from reflection_agent   import reflection_agent_loop
    from performance_tracker import init_db, performance_tracker_loop

    init_db()
    init_positions_table()

    task_monitor    = asyncio.create_task(monitoring_loop())
    task_scheduler  = asyncio.create_task(
        scheduler_loop(paper=PAPER, daily_log=_app_state.daily_log,
                        signal_memory=_app_state.signal_memory)
    )
    task_watcher    = asyncio.create_task(news_watcher_loop(paper=PAPER))
    task_yf_news    = asyncio.create_task(yf_news_watcher_loop(paper=PAPER))
    task_spike      = asyncio.create_task(spike_watcher_loop(run_once, PAPER, wl.load))
    task_edgar      = asyncio.create_task(edgar_watcher_loop(paper=PAPER))
    task_geo        = asyncio.create_task(geo_watcher_loop())
    task_macro      = asyncio.create_task(macro_watcher_loop())
    task_earnings   = asyncio.create_task(earnings_watcher_loop(extra_tickers=TICKERS))
    task_breadth    = asyncio.create_task(breadth_watcher_loop())
    task_social     = asyncio.create_task(social_watcher_loop(extra_tickers=TICKERS))
    task_discovery  = asyncio.create_task(discovery_agent_loop(static_watchlist_fn=wl.load))
    task_portfolio  = asyncio.create_task(portfolio_agent_loop(paper=PAPER))
    task_tracker    = asyncio.create_task(performance_tracker_loop())
    task_reflection = asyncio.create_task(reflection_agent_loop(paper=PAPER))
    yield
    task_monitor.cancel()
    task_scheduler.cancel()
    task_watcher.cancel()
    task_yf_news.cancel()
    task_spike.cancel()
    task_edgar.cancel()
    task_geo.cancel()
    task_macro.cancel()
    task_earnings.cancel()
    task_breadth.cancel()
    task_social.cancel()
    task_discovery.cancel()
    task_portfolio.cancel()
    task_tracker.cancel()
    task_reflection.cancel()


app = FastAPI(title=f"Stock AI Agent — {TICKER}", lifespan=lifespan)


def _json_safe(value):
    """Recursively replace NaN/Inf values so FastAPI JSON responses stay valid."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    try:
        with open("dashboard/index.html") as f:
            return f.read()
    except FileNotFoundError:
        return "<h2>dashboard/index.html not found</h2>"


@app.get("/api/state")
async def api_state(ticker: str = None):
    t = (ticker or TICKER).upper()
    state   = _app_state.ticker_states.get(t, {})
    history = _app_state.histories.get(t, [])
    return JSONResponse(_json_safe({
        "state":         state,
        "history":       history,
        "paper_trading": PAPER,
        "ticker":        t,
        "tickers":       TICKERS,
        "market_open":   is_market_open(),
    }))


@app.get("/api/bars")
async def api_bars(ticker: str = None):
    t    = (ticker or TICKER).upper()
    bars = _app_state.bars_map.get(t, [])
    tv_bars = [
        {
            "time":   b.get("t", 0) // 1000,
            "open":   b.get("o"),
            "high":   b.get("h"),
            "low":    b.get("l"),
            "close":  b.get("c"),
            "volume": b.get("v"),
        }
        for b in bars
    ]
    return JSONResponse(_json_safe({"bars": tv_bars, "ticker": t}))


@app.get("/api/news")
async def api_news():
    raw  = _app_state.news_map.get(TICKER, [])
    news = [
        {
            "title":     n.get("title", ""),
            "url":       n.get("article_url", ""),
            "published": (n.get("published_utc") or "")[:10],
            "source":    (n.get("publisher") or {}).get("name", ""),
        }
        for n in raw[:10]
    ]
    return JSONResponse(_json_safe({"news": news, "ticker": TICKER}))


@app.get("/api/watchlist")
async def api_watchlist():
    rows = []
    for t in TICKERS:
        s = _app_state.ticker_states.get(t, {})
        rows.append({
            "ticker":     t,
            "signal":     s.get("signal", "—"),
            "confidence": s.get("confidence", 0),
            "price":      s.get("current_price", 0),
            "rsi":        s.get("rsi", 0),
        })
    return JSONResponse(_json_safe({"watchlist": rows, "memory": _app_state.signal_memory}))


@app.post("/api/run")
async def api_trigger_run():
    async def _run_all():
        for i, t in enumerate(TICKERS):
            if i > 0:
                await asyncio.sleep(20)
            await run_once(t)
    asyncio.create_task(_run_all())
    return {"status": "triggered", "tickers": TICKERS}


@app.get("/api/log")
async def api_log():
    """Return last 100 rows of signals_log.csv for the dashboard."""
    return JSONResponse(_json_safe({"log": logger.read_log(limit=100)}))


@app.get("/api/performance")
async def api_performance():
    """
    Aggregate performance metrics from signals_log.csv + best_picks_log.csv.
    Also includes live VIX / SPY circuit breaker status.
    """
    import csv as _csv
    from datetime import date as _date

    today_str = _date.today().isoformat()

    # ── signals_log.csv ───────────────────────────────────────────────────────
    signals       = logger.read_log(limit=10_000)
    buy_signals   = [s for s in signals if s.get("signal") == "BUY"]
    sell_signals  = [s for s in signals if s.get("signal") == "SELL"]
    today_signals = [s for s in signals if s.get("timestamp", "").startswith(today_str)]

    # ── best_picks_log.csv ────────────────────────────────────────────────────
    picks: list = []
    try:
        import os as _os
        if _os.path.exists("best_picks_log.csv"):
            with open("best_picks_log.csv", newline="", encoding="utf-8") as f:
                picks = list(_csv.DictReader(f))
    except Exception:
        picks = []

    resolved = [
        p for p in picks
        if p.get("actual_gain_loss_pct") not in (None, "", "—")
    ]

    gains, losses = [], []
    for p in resolved:
        try:
            pct = float(str(p["actual_gain_loss_pct"]).replace("%", "").replace("+", ""))
            (gains if pct > 0 else losses).append(pct)
        except ValueError:
            pass

    all_pcts   = gains + losses
    win_rate   = round(len(gains) / len(resolved) * 100, 1) if resolved else None
    avg_gain   = round(sum(gains)  / len(gains),  2) if gains  else None
    avg_loss   = round(sum(losses) / len(losses), 2) if losses else None
    total_pnl  = round(sum(all_pcts), 2)             if all_pcts else None

    def _pick_summary(p: dict) -> dict:
        try:
            pct = float(str(p.get("actual_gain_loss_pct", "0")).replace("%", "").replace("+", ""))
        except ValueError:
            pct = 0.0
        return {
            "date":       p.get("date", ""),
            "ticker":     p.get("ticker", ""),
            "score":      p.get("total_score") or p.get("score", ""),
            "price":      p.get("price_at_pick", ""),
            "setup":      p.get("setup_type", ""),
            "gain_pct":   pct,
            "result":     "WIN" if pct > 0 else ("LOSS" if pct < 0 else "—"),
        }

    best_pick  = max(resolved, key=lambda p: float(str(p.get("actual_gain_loss_pct","0")).replace("%","").replace("+","")), default=None)
    worst_pick = min(resolved, key=lambda p: float(str(p.get("actual_gain_loss_pct","0")).replace("%","").replace("+","")), default=None)

    # ── Circuit breaker status ────────────────────────────────────────────────
    cb_status = {"safe": True, "reason": "OK", "vix": 0.0, "spy_chg": 0.0}
    try:
        from circuit_breaker import check_market
        cb_status = check_market()
    except Exception:
        pass

    return JSONResponse(_json_safe({
        "signals_total":         len(signals),
        "signals_today":         len(today_signals),
        "buy_count":             len(buy_signals),
        "sell_count":            len(sell_signals),
        "best_picks_total":      len(picks),
        "resolved_picks":        len(resolved),
        "win_rate":              win_rate,
        "avg_gain_pct":          avg_gain,
        "avg_loss_pct":          avg_loss,
        "total_pnl_pct":         total_pnl,
        "best_pick":             _pick_summary(best_pick)  if best_pick  else None,
        "worst_pick":            _pick_summary(worst_pick) if worst_pick else None,
        "recent_picks":          [_pick_summary(p) for p in picks[-10:]],
        "vix":                   cb_status["vix"],
        "spy_chg":               cb_status["spy_chg"],
        "circuit_breaker_active": not cb_status["safe"],
        "circuit_breaker_reason": cb_status["reason"],
    }))


@app.get("/api/portfolio")
async def api_portfolio():
    """Open positions, exit alerts, 30d win rate, and discovery candidates."""
    from portfolio_agent  import get_portfolio_summary, get_cached_portfolio
    from discovery_agent  import get_discovery_tickers
    from reflection_agent import load_learnings
    summary    = get_portfolio_summary(paper=PAPER)
    learnings  = load_learnings()
    return JSONResponse(_json_safe({
        **summary,
        "discovery_candidates": get_discovery_tickers()[:10],
        "learnings": {
            "win_rate_30d":      learnings.get("last_30d_win_rate"),
            "confidence_adj":    learnings.get("confidence_adj", 0),
            "latest_insight":    learnings.get("insights", [""])[-1],
            "favor_conditions":  learnings.get("favor_conditions", []),
            "avoid_conditions":  learnings.get("avoid_conditions", []),
        },
    }))


@app.get("/api/watchlist/saved")
async def api_watchlist_saved():
    """Return the persisted watchlist.json tickers."""
    return JSONResponse(_json_safe({"tickers": wl.load()}))


@app.post("/api/watchlist/add/{ticker}")
async def api_watchlist_add(ticker: str):
    updated = wl.add(ticker.upper())
    return {"tickers": updated}


@app.delete("/api/watchlist/remove/{ticker}")
async def api_watchlist_remove(ticker: str):
    updated = wl.remove(ticker.upper())
    return {"tickers": updated}


# ── Sector rotation heatmap ────────────────────────────────────────────────────

_SECTORS = [
    ("XLK", "Technology"),    ("XLF", "Financials"),    ("XLE", "Energy"),
    ("XLV", "Healthcare"),    ("XLY", "Consumer Disc."), ("XLP", "Consumer Staples"),
    ("XLI", "Industrials"),   ("XLRE","Real Estate"),    ("XLB", "Materials"),
    ("XLU", "Utilities"),     ("XLC", "Communication"),
]
_sector_cache: dict = {}

@app.get("/api/sectors")
async def api_sectors():
    """Return today's % change for all 11 SPDR sector ETFs (15-min cache)."""
    import yfinance as yf
    import time as _time
    now = _time.time()
    if _sector_cache.get("ts", 0) + 900 > now:
        return JSONResponse(_json_safe(_sector_cache.get("data", {})))
    try:
        etfs = [e for e, _ in _SECTORS]
        raw  = yf.download(etfs, period="2d", interval="1d",
                           auto_adjust=True, progress=False, group_by="ticker")
        sectors = []
        for etf, name in _SECTORS:
            try:
                df  = raw[etf] if len(etfs) > 1 else raw
                cls = df["Close"].dropna().values.astype(float)
                chg = round((cls[-1] / cls[-2] - 1) * 100, 2) if len(cls) >= 2 else 0.0
                sectors.append({"etf": etf, "name": name, "change_pct": chg,
                                 "price": round(float(cls[-1]), 2)})
            except Exception:
                sectors.append({"etf": etf, "name": name, "change_pct": 0.0, "price": 0.0})
        sectors.sort(key=lambda x: x["change_pct"], reverse=True)
        result = {"sectors": sectors, "cached_at": datetime.now().strftime("%H:%M:%S")}
        _sector_cache.update({"ts": now, "data": result})
        return JSONResponse(_json_safe(result))
    except Exception as e:
        return JSONResponse({"sectors": [], "error": str(e)})


# ── Pre-market gap cache ───────────────────────────────────────────────────────

_premarket_cache: dict = {}

@app.get("/api/premarket")
async def api_premarket():
    """Return cached pre-market gap results (populated by the 8:30 AM scheduler job)."""
    return JSONResponse(_json_safe(_premarket_cache or {"gaps": [], "scanned_at": None, "count": 0}))


# ── Backtest endpoint ──────────────────────────────────────────────────────────

@app.post("/api/backtest")
async def api_backtest(body: dict):
    """
    Run a walk-forward backtest.
    Body: {"tickers": ["AAPL", "TSLA"], "period": "1y"}
    Returns win_rate, avg_gain_pct, avg_loss_pct, total_pnl_pct, max_drawdown_pct,
            total_trades, trades[], per_ticker{}
    """
    tickers = [str(t).upper().strip() for t in body.get("tickers", [TICKER])]
    period  = str(body.get("period", "1y"))
    if period not in ("3mo", "6mo", "1y", "2y"):
        period = "1y"
    if not tickers:
        return JSONResponse({"error": "no tickers provided"}, status_code=400)
    try:
        from backtester import backtest as _backtest
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, lambda: _backtest(tickers, period))
        return JSONResponse(_json_safe(result))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    _setup_from_args()
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)
