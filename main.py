"""
main.py — entry point for the Argus.

Usage:
  python main.py                         # monitor TICKER from .env
  python main.py --ticker AAPL           # override ticker
  python main.py --interval 120          # check every 2 minutes
  python main.py --port 8080             # dashboard on custom port

Features:
  - Market hours awareness (9:30 AM - 4:00 PM EST)
  - Signal memory + stop loss / target monitoring
  - Daily summary report at 4:30 PM EST
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
import logger


# ── CLI args ───────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Argus — Autonomous Market Intelligence")
    p.add_argument("--interval", type=int, default=300, help="Scan interval seconds (default 300 = 5 min)")
    p.add_argument("--port",     type=int, default=8000, help="Dashboard port (default 8000)")
    args, _ = p.parse_known_args()
    return args

# ── Safe module-level defaults ─────────────────────────────────────────────────
TICKERS: list  = []   # populated by build_scan_list() at runtime
TICKER:  str   = ""   # first ticker in TICKERS (used as API fallback)
INTERVAL       = 300
PORT           = 8000
EST            = ZoneInfo("America/New_York")
_scan_built_at: float = 0.0   # epoch time of last build_scan_list() call


def _setup_from_args() -> None:
    """Parse CLI args and populate module-level config. Called only from __main__."""
    global INTERVAL, PORT
    args     = _parse_args()
    INTERVAL = args.interval
    PORT     = args.port


# ── Autonomous scan list builder ───────────────────────────────────────────────

def build_scan_list() -> list:
    """
    Build the scan list from all autonomous sources.
    Called at startup and refreshed every 30 minutes.
    Returns a deduplicated list of tickers ordered by priority.
    """
    global TICKERS, TICKER, _scan_built_at
    tickers = []

    # 1. Top movers right now (biggest source — 5-min cache)
    try:
        from top_movers import get_top_movers
        movers = get_top_movers()
        tickers += movers
        print(f"   🚀 Top movers: {movers[:10]}")
    except Exception as e:
        print(f"   ⚠️  Top movers: {e}")

    # 2. Momentum screener (2-hour scan)
    try:
        from momentum_screener import get_momentum_candidates
        momentum = get_momentum_candidates()
        tickers += momentum
        if momentum:
            print(f"   📈 Momentum: {momentum[:5]}")
    except Exception as e:
        print(f"   ⚠️  Momentum: {e}")

    # 3. Discovery agent (4-hour Claude-driven scan)
    try:
        from discovery_agent import get_discovery_tickers
        discovery = get_discovery_tickers()
        tickers += discovery
        if discovery:
            print(f"   🔍 Discovery: {discovery[:5]}")
    except Exception as e:
        print(f"   ⚠️  Discovery: {e}")

    # 4. Options flow unusual activity
    try:
        import world_context as wctx
        social = wctx.get().get("social", {})
        opts = [o["ticker"] for o in social.get("unusual_opts", [])[:10]
                if isinstance(o, dict) and "ticker" in o]
        tickers += opts
        if opts:
            print(f"   🎯 Options flow: {opts}")
    except Exception as e:
        print(f"   ⚠️  Options flow: {e}")

    # 5. Earnings hot plays
    try:
        import world_context as wctx
        earnings = wctx.get().get("earnings", {})
        hot = [p["ticker"] for p in earnings.get("hot_plays", [])[:10]
               if isinstance(p, dict) and "ticker" in p]
        tickers += hot
        if hot:
            print(f"   📅 Earnings: {hot}")
    except Exception as e:
        print(f"   ⚠️  Earnings: {e}")

    # 6. EOD setups from yesterday
    try:
        import json as _json, os as _os
        path = _os.path.join("data", "tomorrow_watchlist.json")
        if _os.path.exists(path):
            eod = [s["ticker"] for s in _json.load(open(path)).get("setups", [])
                   if s.get("ticker")]
            tickers += eod
            if eod:
                print(f"   📊 EOD setups: {eod}")
    except Exception as e:
        print(f"   ⚠️  EOD setups: {e}")

    # 7. Pre-market gaps
    try:
        import json as _json, os as _os
        path = _os.path.join("data", "prep_alert_list.json")
        if _os.path.exists(path):
            prep = [s.get("ticker") for s in _json.load(open(path)).get("stocks", [])
                    if s.get("ticker")]
            tickers += prep
            if prep:
                print(f"   🌅 Pre-market: {prep}")
    except Exception as e:
        print(f"   ⚠️  Pre-market: {e}")

    # 8. Social trending (stored as dicts: {"ticker": "NVDA", ...})
    try:
        import world_context as wctx
        trending_raw = wctx.get().get("social", {}).get("trending", [])[:5]
        trending = [t["ticker"] if isinstance(t, dict) else t for t in trending_raw if t]
        if trending:
            tickers += trending
            print(f"   📱 Trending: {trending}")
    except Exception as e:
        print(f"   ⚠️  Trending: {e}")

    # Deduplicate preserving order
    seen   = set()
    unique = []
    for t in tickers:
        if t and isinstance(t, str) and t not in seen:
            seen.add(t)
            unique.append(t.upper().strip())

    TICKERS         = unique
    TICKER          = unique[0] if unique else ""
    _scan_built_at  = __import__("time").time()

    print(f"\n✅ Scan list: {len(unique)} stocks")
    if unique:
        print(f"   {unique[:20]}")
    return unique


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


# ── Pipeline trace (updated on every GRAPH.invoke call) ────────────────────────

_last_trace: dict = {
    "ticker": None, "timestamp": None, "nodes": [],
    "total_time_ms": 0, "final_signal": None, "confidence": 0, "alert_fired": False,
}

# Keys each node is primarily responsible for writing (for trace display grouping)
_NODE_KEY_OUTPUTS = {
    "fetch_data":          ["current_price", "rsi", "volume_spike_ratio"],
    "parallel_analyze":    ["news_sentiment", "sentiment_score", "macd_signal", "ema_signal", "volume_spike"],
    "aggregate_signals":   ["score_breakdown", "signal", "confidence"],
    "decide":              ["signal", "confidence", "should_alert", "entry_zone", "targets", "stop_loss"],
    "validate_decision":   ["validator_passed", "validator_overrides", "final_signal"],
    "assess_risk":         ["risk_approved", "risk_veto_reason", "risk_multiplier", "risk_warnings"],
    "size_position":       ["position_size_pct", "position_size_usd", "max_shares", "scale_in"],
    "check_execution":     ["executable", "order_type", "execution_reason"],
    "alert":               ["alert_sent", "alert_reason_code"],
}

_NODE_TIME_WEIGHTS = {
    "fetch_data": 0.38, "parallel_analyze": 0.26, "aggregate_signals": 0.07,
    "decide": 0.12, "validate_decision": 0.05, "assess_risk": 0.04,
    "size_position": 0.03, "check_execution": 0.03, "alert": 0.02,
}


def _build_trace(ticker: str, state: dict, elapsed_ms: float) -> dict:
    """Build a synthetic per-node trace from the final pipeline state."""
    nodes = []
    for node_name, keys in _NODE_KEY_OUTPUTS.items():
        outputs = {k: state.get(k) for k in keys if state.get(k) is not None}
        if node_name == "assess_risk" and state.get("risk_approved") is False:
            status = "vetoed"
        elif node_name == "check_execution" and state.get("executable") is False:
            status = "blocked"
        elif node_name == "alert" and not state.get("alert_sent"):
            status = "skipped"
        else:
            status = "completed"
        nodes.append({
            "name":    node_name,
            "status":  status,
            "time_ms": round(elapsed_ms * _NODE_TIME_WEIGHTS.get(node_name, 0.04)),
            "outputs": _json_safe(outputs),
        })
    return {
        "ticker":        ticker,
        "timestamp":     datetime.now().isoformat(),
        "nodes":         nodes,
        "total_time_ms": round(elapsed_ms),
        "final_signal":  state.get("signal", "HOLD"),
        "confidence":    state.get("confidence", 0),
        "alert_fired":   bool(state.get("alert_sent")),
    }


# ── Graph execution ────────────────────────────────────────────────────────────

def _run_sync(ticker: str,
              already_alerted: bool = False,
              news_triggered: bool = False) -> dict:
    global _last_trace
    import time as _time
    state = make_initial_state(ticker)
    state["already_alerted"] = already_alerted
    state["news_triggered"]  = news_triggered
    t0 = _time.monotonic()
    result = GRAPH.invoke(state)
    elapsed_ms = (_time.monotonic() - t0) * 1000
    _last_trace = _build_trace(ticker, result, elapsed_ms)
    return result


def _store_result(result: dict):
    ticker = result.get("ticker") or (TICKERS[0] if TICKERS else "")

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
    ticker  = result.get("ticker") or (TICKERS[0] if TICKERS else "")
    signal  = result.get("signal", "HOLD")
    price   = result.get("current_price", 0.0)

    # Only record BUY signals that were actually delivered — not suppressed trades,
    # not SELL signals (shorts are unsupported: sizing/execution/exit lifecycle
    # are all BUY-only, so recording a SELL would orphan it in signal_memory).
    alert_sent = result.get("alert_sent", False)
    if signal == "BUY" and alert_sent:
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
    ticker = ticker or (TICKERS[0] if TICKERS else "")
    loop   = asyncio.get_running_loop()
    # Suppress re-alerting if this ticker already has an active BUY position
    already_alerted = ticker in _app_state.signal_memory
    result = await loop.run_in_executor(
        _executor, _run_sync, ticker, already_alerted, news_triggered
    )

    _store_result(result)
    _update_signal_memory(result)

    # Stop loss / target monitoring
    price       = result.get("current_price", 0.0)
    exit_reason = _check_exits(ticker, price)
    if exit_reason:
        icon = "🛑" if exit_reason == "STOP_LOSS" else "🎯"
        print(f"{icon} [Monitor] {exit_reason} triggered for {ticker} @ ${price:.4f}")
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
    )


# ── Daily report ───────────────────────────────────────────────────────────────

async def _send_daily_report():
    today    = datetime.now(tz=EST).strftime("%Y-%m-%d")
    today_signals = [e for e in _app_state.daily_log if e["timestamp"][:10] == today]

    lines = [f"📅 Daily Report — Argus Autonomous — {today}"]
    if not today_signals:
        lines.append("No BUY/SELL signals fired today.")
    else:
        lines.append(f"Signals fired: {len(today_signals)}")
        for s in today_signals:
            icon = "🟢" if s["signal"] == "BUY" else "🔴"
            lines.append(f"  {icon} {s['signal']} @ ${s['price']:.4f}  conf={s['confidence']}")
    report = "\n".join(lines)
    print("\n" + "─" * 50)
    print(report)
    print("─" * 50 + "\n")

    try:
        from alerts import send_push, send_whatsapp
        send_push("Daily Report — Argus", report)
        send_whatsapp(report)
    except Exception as e:
        print(f"❌ [Monitor] Daily report delivery failed: {e}")


# ── Monitoring loop ────────────────────────────────────────────────────────────

async def monitoring_loop():
    print(
        f"\n🤖 ARGUS — Full Autonomous Mode\n"
        f"   No watchlist — finding stocks independently\n"
        f"   Sources: movers + momentum + discovery +\n"
        f"            options + earnings + EOD + premarket\n"
        f"   Refreshing every 30 minutes\n"
        f"📊 Dashboard → http://localhost:{PORT}\n"
    )

    # Semaphore limits concurrent Polygon-hitting pipelines to protect free tier
    _sem = asyncio.Semaphore(2)

    async def _run_gated(t: str):
        async with _sem:
            await run_once(t)

    _SCAN_REFRESH_S = 30 * 60   # rebuild scan list every 30 minutes

    # Build initial scan list (retry until non-empty)
    print("🔄 [Monitor] Building initial scan list...")
    loop = asyncio.get_running_loop()
    scan_list = await loop.run_in_executor(None, build_scan_list)
    while not scan_list:
        print("⏳ [Monitor] No stocks found yet — retrying in 60s")
        await asyncio.sleep(60)
        scan_list = await loop.run_in_executor(None, build_scan_list)

    while True:
        try:
            now_est   = datetime.now(tz=EST)
            h, m      = now_est.hour, now_est.minute
            weekday   = now_est.weekday() < 5
            premarket = weekday and (8, 0) <= (h, m) < (9, 30)
            intraday  = is_market_open()

            # Refresh scan list every 30 minutes
            import time as _time
            if _time.time() - _scan_built_at >= _SCAN_REFRESH_S:
                print("🔄 [Monitor] Refreshing scan list...")
                scan_list = await loop.run_in_executor(None, build_scan_list)
                if not scan_list:
                    print("⏳ [Monitor] Scan list empty after refresh — keeping previous")
                    scan_list = TICKERS or []
                else:
                    print(f"🔄 [Monitor] Scan list refreshed: {len(scan_list)} stocks")

            if intraday or premarket:
                mode_label = "🌅 PRE-MARKET" if premarket else "📈 INTRADAY"
                print(f"\n{mode_label} [Monitor] Scanning {len(scan_list)} stocks "
                      f"({now_est.strftime('%H:%M ET')})")

                await asyncio.gather(*[_run_gated(t) for t in scan_list])
            else:
                print(
                    f"🕐 [Monitor] Market closed "
                    f"({now_est.strftime('%a %H:%M ET')}) — "
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
    from momentum_screener  import momentum_screener_loop
    import task_supervisor as sup

    init_db()
    init_positions_table()

    sup.start("monitor",    monitoring_loop)
    sup.start("scheduler",  lambda: scheduler_loop(
        daily_log=_app_state.daily_log,
        signal_memory=_app_state.signal_memory))
    sup.start("news",       news_watcher_loop)
    sup.start("yf_news",    yf_news_watcher_loop)
    from top_movers import get_top_movers
    sup.start("spike",      lambda: spike_watcher_loop(run_once, get_top_movers))
    sup.start("edgar",      edgar_watcher_loop)
    sup.start("geo",        geo_watcher_loop)
    sup.start("macro",      macro_watcher_loop)
    sup.start("earnings",   earnings_watcher_loop)
    sup.start("breadth",    breadth_watcher_loop)
    sup.start("social",     social_watcher_loop)
    sup.start("discovery",  discovery_agent_loop)
    sup.start("portfolio",  portfolio_agent_loop)
    sup.start("tracker",    performance_tracker_loop)
    sup.start("reflection", reflection_agent_loop)
    sup.start("momentum",   momentum_screener_loop)

    yield
    await sup.cancel_all()


app = FastAPI(
    title="Argus — Agentic Market Intelligence",
    description="Autonomous trading agent: perceives, reasons, acts, and learns.",
    version="4.0",
    lifespan=lifespan,
)


def _json_safe(value):
    """
    Recursively sanitize a value for JSON serialization.
    Handles Python float/int, numpy scalars (np.float64, np.int64, np.bool_),
    pandas NA/NaT/NaN, and nested dicts/lists/tuples.
    """
    # numpy scalar types (float and int families)
    try:
        import numpy as np
        if isinstance(value, np.floating):
            return None if not math.isfinite(float(value)) else float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return [_json_safe(v) for v in value.tolist()]
    except ImportError:
        pass

    # pandas NA / NaT / pd.NA
    try:
        import pandas as pd
        if value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except ImportError:
        pass

    # Python native float — must come after numpy check
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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
    t = (ticker or (TICKERS[0] if TICKERS else "")).upper()
    state   = _app_state.ticker_states.get(t, {})
    history = _app_state.histories.get(t, [])
    return JSONResponse(_json_safe({
        "state":         state,
        "history":       history,
        "ticker":        t,
        "tickers":       TICKERS,
        "scan_count":    len(TICKERS),
        "market_open":   is_market_open(),
    }))


@app.get("/api/bars")
async def api_bars(ticker: str = None):
    t    = (ticker or (TICKERS[0] if TICKERS else "")).upper()
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
async def api_news(ticker: str = None):
    t    = (ticker or (TICKERS[0] if TICKERS else "")).upper()
    raw  = _app_state.news_map.get(t, [])
    news = [
        {
            "title":     n.get("title", ""),
            "url":       n.get("article_url", ""),
            "published": (n.get("published_utc") or "")[:10],
            "source":    (n.get("publisher") or {}).get("name", ""),
        }
        for n in raw[:10]
    ]
    return JSONResponse(_json_safe({"news": news, "ticker": t}))


@app.get("/api/watchlist")
async def api_watchlist():
    """Return current autonomous scan list with latest signal state per ticker."""
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
    return JSONResponse(_json_safe({
        "scan_list": rows,
        "count":     len(rows),
        "memory":    _app_state.signal_memory,
    }))


@app.post("/api/run")
async def api_trigger_run():
    scan = list(TICKERS[:10])   # cap at 10 for manual trigger
    async def _run_all():
        for i, t in enumerate(scan):
            if i > 0:
                await asyncio.sleep(20)
            await run_once(t)
    asyncio.create_task(_run_all())
    return {"status": "triggered", "tickers": scan}


@app.get("/api/health")
async def api_health():
    """Per-task health status: running / error / cancelled + restart count."""
    import task_supervisor as sup
    import world_context as wctx
    ctx = wctx.get()
    return JSONResponse(_json_safe({
        "tasks":   sup.get_health(),
        "context": {
            "geo_updated_at":      ctx["geo"]["updated_at"],
            "macro_updated_at":    ctx["macro"]["updated_at"],
            "breadth_updated_at":  ctx["breadth"]["updated_at"],
            "earnings_updated_at": ctx["earnings"]["updated_at"],
            "social_updated_at":   ctx["social"]["updated_at"],
        },
    }))


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
    summary    = get_portfolio_summary()
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


@app.get("/api/scan-list")
async def api_scan_list():
    """Return current autonomous scan list and metadata."""
    import time as _t
    age_min = round((_t.time() - _scan_built_at) / 60, 1) if _scan_built_at else None
    return JSONResponse(_json_safe({
        "tickers":    TICKERS,
        "count":      len(TICKERS),
        "age_min":    age_min,
        "autonomous": True,
    }))


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


# ── Intel / Momentum / Alerts endpoints ───────────────────────────────────────

@app.get("/api/intel")
async def api_intel():
    """World context snapshot: macro, geo, earnings, breadth, social."""
    import world_context as wctx
    ctx = wctx.get()
    return JSONResponse(_json_safe({
        "macro":    ctx["macro"],
        "geo":      ctx["geo"],
        "earnings": ctx["earnings"],
        "breadth":  ctx["breadth"],
        "social":   ctx["social"],
    }))


@app.get("/api/momentum")
async def api_momentum():
    """Momentum screener results from the last 2-hour scan."""
    import world_context as wctx
    ctx    = wctx.get()
    social = ctx.get("social", {})
    return JSONResponse(_json_safe({
        "candidates": social.get("momentum_picks", []),
        "tickers":    social.get("momentum_candidates", []),
        "updated_at": social.get("momentum_updated_at"),
    }))


@app.get("/api/alerts")
async def api_alerts():
    """Recent BUY/SELL signals (last 30) from signals_log.csv."""
    signals = logger.read_log(limit=200)
    alerts  = [s for s in signals if s.get("signal") in ("BUY", "SELL")]
    return JSONResponse(_json_safe({
        "alerts": list(reversed(alerts[-30:])),
        "total":  len(alerts),
    }))


# ── Weekly eval approval endpoints ────────────────────────────────────────────

@app.get("/api/eval/status")
async def api_eval_status():
    """Return current weekly eval proposal (pending / approved / rejected)."""
    try:
        from agents.eval_agent import get_pending_approval, _load_weekly
        data    = _load_weekly()
        pending = get_pending_approval()
        history = data.get("history", [])
        return {
            "pending":            pending,
            "history_count":      len(history),
            "last_3_weeks":       history[-3:] if history else [],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/eval/approve/{code}")
async def api_eval_approve(code: str):
    """Approve the pending weekly eval proposal and apply weight adjustments."""
    try:
        from agents.eval_agent import approve_learnings
        result = approve_learnings(code.upper())
        if result.get("ok"):
            return {"status": "approved", "weights_applied": len(result.get("applied", {}))}
        return {"status": "error", "reason": result.get("reason", "unknown")}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@app.get("/api/eval/reject/{code}")
async def api_eval_reject(code: str):
    """Reject the pending weekly eval proposal without applying changes."""
    try:
        from agents.eval_agent import reject_learnings
        result = reject_learnings(code.upper())
        if result.get("ok"):
            return {"status": "rejected"}
        return {"status": "error", "reason": result.get("reason", "unknown")}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── Backtest endpoint ──────────────────────────────────────────────────────────

@app.post("/api/backtest")
async def api_backtest(body: dict):
    """
    Run a walk-forward backtest.
    Body: {"tickers": ["AAPL", "TSLA"], "period": "1y"}
    Returns win_rate, avg_gain_pct, avg_loss_pct, total_pnl_pct, max_drawdown_pct,
            total_trades, trades[], per_ticker{}
    """
    tickers = [str(t).upper().strip() for t in body.get("tickers", TICKERS[:3] or ["SPY"])]
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


# ── Agent monitor endpoints ────────────────────────────────────────────────────

@app.get("/api/agent-trace")
async def api_agent_trace():
    """Last pipeline execution trace — per-node status, timing, and key outputs."""
    return JSONResponse(_json_safe(_last_trace))


@app.get("/api/agent-status")
async def api_agent_status():
    """Status of all background agents + world_context freshness."""
    import task_supervisor as sup
    import world_context as wctx
    ctx = wctx.get()
    health = sup.get_health()
    task_labels = {
        "monitor":    "Pipeline Monitor",   "scheduler":  "Scheduler",
        "news":       "News Watcher",       "yf_news":    "YF News",
        "spike":      "Spike Watcher",      "edgar":      "EDGAR Watcher",
        "geo":        "Geo Watcher",        "macro":      "Macro Watcher",
        "earnings":   "Earnings Watcher",   "breadth":    "Breadth Watcher",
        "social":     "Social Watcher",     "discovery":  "Discovery Agent",
        "portfolio":  "Portfolio Agent",    "tracker":    "Perf Tracker",
        "reflection": "Reflection Agent",   "momentum":   "Momentum Screener",
    }
    agents = [
        {
            "id":         tid,
            "label":      label,
            "status":     health.get(tid, {}).get("status", "unknown"),
            "alive":      health.get(tid, {}).get("alive", False),
            "restarts":   health.get(tid, {}).get("restarts", 0),
            "last_error": health.get(tid, {}).get("last_error"),
            "started_at": health.get(tid, {}).get("started_at"),
        }
        for tid, label in task_labels.items()
    ]
    return JSONResponse(_json_safe({
        "agents": agents,
        "context": {
            "macro_regime":     ctx["macro"].get("regime"),
            "macro_updated":    ctx["macro"].get("updated_at"),
            "geo_updated":      ctx["geo"].get("updated_at"),
            "earnings_updated": ctx["earnings"].get("updated_at"),
            "breadth_updated":  ctx["breadth"].get("updated_at"),
            "social_updated":   ctx["social"].get("updated_at"),
        },
    }))


@app.get("/api/agent-flow")
async def api_agent_flow():
    """Static pipeline structure: nodes and edges for flow diagram rendering."""
    return JSONResponse({
        "nodes": [
            {"id": "fetch_data",        "label": "Data Fetch",   "group": "data"},
            {"id": "parallel_analyze",  "label": "News + Tech",  "group": "analysis"},
            {"id": "aggregate_signals", "label": "Signal Agg",   "group": "analysis"},
            {"id": "decide",            "label": "Decision",     "group": "decision"},
            {"id": "validate_decision", "label": "Validator",    "group": "decision"},
            {"id": "assess_risk",       "label": "Risk Check",   "group": "risk"},
            {"id": "size_position",     "label": "Sizing",       "group": "risk"},
            {"id": "check_execution",   "label": "Execution",    "group": "execution"},
            {"id": "alert",             "label": "Alert",        "group": "execution"},
        ],
        "edges": [
            {"from": "fetch_data",        "to": "parallel_analyze"},
            {"from": "parallel_analyze",  "to": "aggregate_signals"},
            {"from": "aggregate_signals", "to": "decide"},
            {"from": "decide",            "to": "validate_decision"},
            {"from": "validate_decision", "to": "assess_risk"},
            {"from": "assess_risk",       "to": "size_position"},
            {"from": "size_position",     "to": "check_execution"},
            {"from": "check_execution",   "to": "alert"},
        ],
    })


@app.get("/api/pipeline-run")
async def api_pipeline_run(ticker: str = None):
    """Run a pipeline audit and return a detailed per-node trace."""
    global _last_trace
    t = (ticker or (TICKERS[0] if TICKERS else "SPY")).upper()
    try:
        from graph import run_pipeline_audit, make_initial_state as _mis
        import time as _time
        loop = asyncio.get_running_loop()
        initial_state = _mis(t)
        t0 = _time.monotonic()
        audit = await loop.run_in_executor(_executor, run_pipeline_audit, initial_state)
        elapsed_ms = (_time.monotonic() - t0) * 1000

        nodes = [
            {
                "name":    step["step"],
                "status":  "completed" if step["changed"] else "pass-through",
                "time_ms": 0,
                "outputs": _json_safe(step["changed"]),
            }
            for step in audit.get("steps", [])
        ]
        final_snap = audit.get("final", {})
        _last_trace = {
            "ticker":        t,
            "timestamp":     datetime.now().isoformat(),
            "nodes":         nodes,
            "total_time_ms": round(elapsed_ms),
            "final_signal":  final_snap.get("signal", "HOLD"),
            "confidence":    final_snap.get("confidence", 0),
            "alert_fired":   False,
            "source":        "audit",
        }
        return JSONResponse(_json_safe(_last_trace))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    _setup_from_args()
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)
