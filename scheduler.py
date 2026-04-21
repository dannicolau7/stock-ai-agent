"""
scheduler.py — APScheduler-based trading schedule (America/New_York).

  7:45 AM  Mon–Fri   SCAN 1 — Pre-market TOP 5 (fresh catalysts only)
  9:45 AM  Mon–Fri   SCAN 2 — Opening range: confirm gap holders → fire BUY alerts
  2:00 PM  Mon–Fri   SCAN 3 — Power hour: intraday momentum → tomorrow's setups
  :00,:30  Mon–Fri 9:30–16:00  Exit agent — monitor all open positions
  4:15 PM  Mon–Fri   Daily P&L summary
  Sun 7:00 PM ET      Weekly eval report

Market status is validated via Polygon API (5-min cache) before each intraday scan,
with a time-based fallback if Polygon is unavailable.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from alerts import send_whatsapp
from graph import GRAPH, make_initial_state
from intelligence_hub import hub

EST = ZoneInfo("America/New_York")
_executor = ThreadPoolExecutor(max_workers=4)

# ── Module-level state ─────────────────────────────────────────────────────────

# Populated by scheduler_loop(); read by job functions
_daily_log:     list = []
_signal_memory: dict = {}

# Scan 1 results cached for Scan 2 to re-check
_scan1_results: list = []

# ── Market status check ────────────────────────────────────────────────────────

_market_status_cache: dict = {"open": False, "fetched_at": 0.0}
_MARKET_STATUS_TTL = 300   # 5-minute cache


def _is_market_open() -> bool:
    """
    Returns True if the US equities market is currently open.
    Checks Polygon's /v1/marketstatus/now (5-min cache); falls back to time-based.
    """
    now_ts = time.monotonic()
    if now_ts - _market_status_cache["fetched_at"] < _MARKET_STATUS_TTL:
        return _market_status_cache["open"]

    api_key = os.getenv("POLYGON_API_KEY", "")
    is_open: Optional[bool] = None

    if api_key:
        try:
            import requests
            r = requests.get(
                "https://api.polygon.io/v1/marketstatus/now",
                params={"apiKey": api_key},
                timeout=5,
            )
            if r.ok:
                status = r.json().get("market", "closed")
                is_open = status == "open"
        except Exception as _e:
            print(f"⚠️  [Scheduler] Polygon market status failed: {_e}")

    if is_open is None:
        # Time-based fallback
        est = datetime.now(tz=EST)
        if est.weekday() >= 5:
            is_open = False
        else:
            open_t  = est.replace(hour=9,  minute=30, second=0, microsecond=0)
            close_t = est.replace(hour=16, minute=0,  second=0, microsecond=0)
            is_open = open_t <= est <= close_t

    _market_status_cache.update({"open": is_open, "fetched_at": now_ts})
    return is_open


# ── Graph runner helpers ───────────────────────────────────────────────────────

def _run_graph_sync(ticker: str) -> dict:
    return GRAPH.invoke(make_initial_state(ticker))


async def _run_ticker(ticker: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run_graph_sync, ticker)


# ── SCAN 1 — Pre-market TOP 5 (7:45 AM) ───────────────────────────────────────

async def _scan1_premarket_top5():
    """
    7:45 AM Mon–Fri.
    Runs the broad pre-market scan, filters for fresh catalysts, and sends a
    '📊 PRE-MARKET TOP 5' WhatsApp digest.
    Results are cached in _scan1_results for Scan 2 to re-check at 9:45 AM.
    """
    global _scan1_results
    print("\n" + "─" * 50)
    print("🌅 [7:45 AM] SCAN 1 — Pre-market TOP 5 starting...")

    loop = asyncio.get_running_loop()
    try:
        from premarket_scanner import run_premarket_scan, _has_news_catalyst

        # Run full broad scan (Claude-enriched): entry/target/stop/catalyst
        candidates: list = await loop.run_in_executor(
            _executor,
            lambda: run_premarket_scan(
                mode="broad", test=False, verbose=False, save_prep=True
            ),
        )
    except Exception as e:
        print(f"❌ [7:45 AM] Pre-market scan error: {e}")
        send_whatsapp("📊 PRE-MARKET TOP 5: Scan failed — check logs.")
        return

    # Prefer catalysts; fall back to all candidates if none have one
    with_catalyst = [c for c in candidates if _has_news_catalyst(c)]
    pool = with_catalyst if with_catalyst else candidates

    # Sort by score descending and cache for Scan 2
    pool.sort(key=lambda x: x.get("score", 0), reverse=True)
    _scan1_results = pool[:10]   # keep up to 10 for Scan 2 re-check

    top5 = pool[:5]
    if not top5:
        send_whatsapp("📊 PRE-MARKET TOP 5: No qualifying setups today.")
        print("💤 [7:45 AM] No pre-market candidates qualify.")
        return

    _SEP = "─" * 33
    lines = [f"📊 PRE-MARKET TOP 5\n{_SEP}"]
    for i, c in enumerate(top5, 1):
        pm_price   = c.get("premarket_price", 0)
        entry_low  = c.get("entry_low",  pm_price * 0.99)
        target_1   = c.get("target_1", 0)
        stop_loss  = c.get("stop_loss", 0)
        gap_pct    = c.get("gap_pct", 0)

        # Catalyst label
        if c.get("is_earnings_beat"):
            catalyst = "Earnings Beat"
        elif c.get("is_upgrade"):
            catalyst = "Analyst Upgrade"
        elif c.get("has_edgar"):
            catalyst = "EDGAR Filing"
        elif c.get("is_contract"):
            catalyst = "Contract/Deal"
        elif c.get("news_headline"):
            catalyst = c["news_headline"][:70]
        else:
            catalyst = "Technical Setup"

        entry_str  = f"${entry_low:.2f}"
        target_str = f"${target_1:.2f}" if target_1 > 0 else "—"
        stop_str   = f"${stop_loss:.2f}" if stop_loss > 0 else "—"

        lines.append(
            f"{i}. {c['ticker']}  ${pm_price:.2f}  {gap_pct:+.1f}%\n"
            f"   Entry {entry_str}  T1 {target_str}  Stop {stop_str}\n"
            f"   📰 {catalyst}"
        )

    lines.append(_SEP)
    lines.append("⏰ Market opens 9:30 AM — confirming at 9:45 AM")
    send_whatsapp("\n".join(lines))
    print(f"✅ [7:45 AM] Scan 1 sent — {len(top5)} picks, {len(with_catalyst)} with catalyst")


# ── SCAN 2 — Opening range confirmed entries (9:45 AM) ────────────────────────

async def _scan2_confirm_entries():
    """
    9:45 AM Mon–Fri.
    Re-checks Scan 1 picks with live prices.
    Stocks that held ≥80% of their pre-market gap pass to the GRAPH pipeline,
    which fires individual BUY alerts via alert_node.
    Sends a '✅ CONFIRMED ENTRIES' summary.
    """
    print("\n" + "─" * 50)
    print("⏰ [9:45 AM] SCAN 2 — Opening range confirmation...")

    if not _is_market_open():
        print("⚠️  [9:45 AM] Market closed — skipping Scan 2")
        return

    # Load candidates: prefer in-memory Scan 1 results, fall back to saved prep list
    candidates = _scan1_results
    if not candidates:
        try:
            from premarket_scanner import _load_prep_list
            candidates = _load_prep_list()
        except Exception:
            pass

    if not candidates:
        send_whatsapp("✅ CONFIRMED ENTRIES: No pre-market picks to re-check.")
        return

    import yfinance as yf

    confirmed = []
    skipped   = []

    for c in candidates[:10]:
        ticker     = c["ticker"]
        pm_price   = c.get("premarket_price", 0)
        prev_close = c.get("prev_close", 0)
        gap_dollars = pm_price - prev_close   # signed gap

        if abs(gap_dollars) < 0.01 or prev_close <= 0:
            skipped.append(f"{ticker} (no gap data)")
            continue

        try:
            live = float(yf.Ticker(ticker).fast_info["last_price"] or 0)
            if live <= 0:
                skipped.append(f"{ticker} (no live price)")
                continue
        except Exception as e:
            skipped.append(f"{ticker} ({e})")
            continue

        # Gap retention: how much of the pre-market gap is still intact
        gap_retained = (live - prev_close) / gap_dollars if gap_dollars != 0 else 0
        held = gap_retained >= 0.80

        print(
            f"   {'✅' if held else '❌'} {ticker}  "
            f"pm=${pm_price:.2f}  live=${live:.2f}  "
            f"retained={gap_retained*100:.0f}%"
        )

        if held:
            confirmed.append({**c, "live_price": live, "gap_retained": gap_retained})

    # Fire BUY alerts for confirmed tickers via full pipeline
    results = []
    for c in confirmed[:5]:
        ticker = c["ticker"]
        try:
            result = await _run_ticker(ticker)
            results.append({
                "ticker":       ticker,
                "live":         c["live_price"],
                "gap_retained": c["gap_retained"],
                "signal":       result.get("signal", "HOLD"),
                "confidence":   result.get("confidence", 0),
                "alert_sent":   result.get("alert_sent", False),
            })
        except Exception as e:
            print(f"❌ [9:45 AM] Pipeline error for {ticker}: {e}")

    # Build summary WhatsApp
    _SEP = "─" * 33
    lines = [f"✅ CONFIRMED ENTRIES\n{_SEP}"]

    if not results:
        lines.append("0 stocks held their gap. Standing down.")
    else:
        for r in results:
            icon  = "🟢" if r["signal"] == "BUY" else "⚡"
            alert = "→ ALERT SENT" if r["alert_sent"] else ""
            lines.append(
                f"{icon} {r['ticker']}  ${r['live']:.2f}  "
                f"Gap held {r['gap_retained']*100:.0f}%  "
                f"conf={r['confidence']:.0f}  {alert}".rstrip()
            )

    if skipped:
        lines.append(f"\nSkipped: {', '.join(skipped[:5])}")

    lines.append(_SEP)
    send_whatsapp("\n".join(lines))
    confirmed_buys = sum(1 for r in results if r["signal"] == "BUY")
    print(f"✅ [9:45 AM] Scan 2 done — {len(confirmed)}/{len(candidates)} held gap, {confirmed_buys} BUY alerts")


# ── SCAN 3 — Power hour momentum (2:00 PM) ────────────────────────────────────

def _fetch_intraday_momentum(ticker: str) -> Optional[dict]:
    """
    Fetch intraday data and return a momentum dict if the stock qualifies:
    day_chg between +3% and +8%, building volume over the last 30 min.
    Returns None if it doesn't qualify or data is unavailable.
    """
    try:
        import yfinance as yf
        import numpy as np

        t    = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="5m")
        if hist is None or hist.empty or len(hist) < 10:
            return None

        prev_close = float(t.fast_info.get("previous_close") or 0)
        if prev_close <= 0:
            return None

        current   = float(hist["Close"].iloc[-1])
        day_high  = float(hist["High"].max())
        day_chg   = (current - prev_close) / prev_close * 100

        if not (3.0 <= day_chg <= 8.0):
            return None

        # Relative volume: project today's partial volume to full day
        # At 2 PM ET we're 4.5h into the 6.5h session
        today_vol = int(hist["Volume"].sum())
        avg_vol   = float(t.fast_info.get("three_month_average_volume") or 0)
        time_adj  = 6.5 / 4.5
        projected_rvol = (today_vol * time_adj / avg_vol) if avg_vol > 0 else 0

        # Volume building: last 6 bars vs prior 6 bars (30-min window each)
        if len(hist) >= 12:
            recent_vol = float(hist["Volume"].iloc[-6:].mean())
            prior_vol  = float(hist["Volume"].iloc[-12:-6].mean())
            building   = prior_vol > 0 and recent_vol > prior_vol * 1.15
        else:
            building = False

        # Near session high (within 3%)
        near_high = current >= day_high * 0.97

        # Composite score: weight day change + rvol + building + near_high
        score = (
            min(day_chg, 8.0) * 5        # 0-40
            + min(projected_rvol, 5.0) * 6  # 0-30
            + (15 if building else 0)
            + (15 if near_high else 0)
        )

        return {
            "ticker":    ticker,
            "price":     round(current, 4),
            "day_chg":   round(day_chg, 2),
            "rvol":      round(projected_rvol, 2),
            "building":  building,
            "near_high": near_high,
            "score":     round(score, 1),
        }
    except Exception:
        return None


async def _scan3_power_hour():
    """
    2:00 PM Mon–Fri.
    Scans the full dynamic ticker universe for intraday momentum (up 3-8%,
    building volume).  Sends a '🔜 TOMORROW'S SETUPS' WhatsApp digest.
    """
    print("\n" + "─" * 50)
    print("🔜 [2:00 PM] SCAN 3 — Power hour momentum scan starting...")

    if not _is_market_open():
        print("⚠️  [2:00 PM] Market closed — skipping Scan 3")
        return

    # Pull from the same dynamic scan list used by main.py
    try:
        from main import build_scan_list
        tickers = build_scan_list()
    except Exception:
        # Fallback: top movers
        try:
            from top_movers import get_top_movers
            tickers = get_top_movers()
        except Exception as e:
            print(f"❌ [2:00 PM] Could not build ticker list: {e}")
            return

    print(f"   Scanning {len(tickers)} tickers for intraday momentum...")

    # Parallel fetch — I/O bound, use thread pool
    from concurrent.futures import ThreadPoolExecutor, as_completed
    hits: list = []
    with ThreadPoolExecutor(max_workers=8) as exe:
        futs = {exe.submit(_fetch_intraday_momentum, t): t for t in tickers}
        for fut in as_completed(futs):
            result = fut.result()
            if result:
                hits.append(result)

    hits.sort(key=lambda x: x["score"], reverse=True)
    top = hits[:5]

    if not top:
        send_whatsapp(
            "🔜 TOMORROW'S SETUPS (2 PM)\nNo intraday momentum setups found today."
        )
        print("💤 [2:00 PM] No momentum setups (0 stocks up 3-8% with building volume)")
        return

    _SEP = "─" * 33
    lines = [f"🔜 TOMORROW'S SETUPS\n{_SEP}"]
    for i, h in enumerate(top, 1):
        vol_tag  = "📈 vol building" if h["building"] else ""
        high_tag = "near HOD" if h["near_high"] else ""
        tags     = "  ".join(t for t in [vol_tag, high_tag] if t)
        lines.append(
            f"{i}. {h['ticker']}  ${h['price']:.2f}  +{h['day_chg']:.1f}%  "
            f"RVOL {h['rvol']:.1f}x\n"
            f"   Score {h['score']:.0f}  {tags}".rstrip()
        )
    lines.append(_SEP)
    lines.append("💡 Swing candidates for tomorrow's open")

    send_whatsapp("\n".join(lines))
    print(f"✅ [2:00 PM] Scan 3 sent — {len(hits)} momentum stocks, top: {top[0]['ticker']} (+{top[0]['day_chg']:.1f}%)")


# ── Exit agent — every 30 min during market hours ─────────────────────────────

async def _exit_agent_check():
    """
    Fires every 30 min. Skips if market is closed.
    Iterates open positions in signal_memory and runs manage_exit() for each.
    """
    if not _is_market_open():
        return

    positions = dict(_signal_memory)   # snapshot to avoid mutation during iteration
    if not positions:
        return

    print(f"\n⏰ [Exit check] {len(positions)} open position(s): {list(positions)}")

    from agents.exit_agent import manage_exit

    for ticker, mem in positions.items():
        entry_price = mem.get("price", 0)
        entry_date  = mem.get("timestamp")
        target      = mem.get("targets", [0])[0] if mem.get("targets") else 0
        stop        = mem.get("stop_loss", 0)

        if not all([entry_price, target, stop]):
            continue

        try:
            # manage_exit is async
            from datetime import datetime as _dt, timezone as _tz
            if isinstance(entry_date, str):
                entry_date = _dt.fromisoformat(entry_date)
            elif entry_date is None:
                entry_date = _dt.now(_tz.utc)

            result = await manage_exit(
                ticker=ticker,
                entry_price=entry_price,
                entry_date=entry_date,
                target=target,
                stop=stop,
            )
            sig = result.get("signal", "")
            if sig in ("SELL", "STOP", "TAKE_PROFIT"):
                print(f"   🔔 Exit signal for {ticker}: {sig} @ ${result.get('exit_price', 0):.2f}")
                _signal_memory.pop(ticker, None)
        except Exception as e:
            print(f"   ⚠️  Exit check error {ticker}: {e}")


# ── 4:15 PM — Daily P&L summary ───────────────────────────────────────────────

async def _send_daily_pnl():
    """4:15 PM Mon–Fri — reads today's signals from daily_log and SQLite P&L."""
    print("\n" + "─" * 50)
    print("📊 [4:15 PM] Daily P&L summary...")

    today = datetime.now(tz=EST).strftime("%Y-%m-%d")
    sigs  = [e for e in _daily_log if str(e.get("timestamp", ""))[:10] == today]

    lines = [f"📊 Daily P&L — {today}\n"]
    if not sigs:
        lines.append("No BUY/SELL signals fired today.")
    else:
        lines.append(f"Signals fired: {len(sigs)}")
        for s in sigs:
            icon = "🟢" if s["signal"] == "BUY" else "🔴"
            t    = str(s.get("timestamp", ""))[-8:-3]   # HH:MM
            lines.append(
                f"  {icon} {s['signal']} {s['ticker']} @ "
                f"${s.get('price', 0):.2f}  conf={s.get('confidence', 0):.0f}  {t}"
            )

    # Quick SQLite win-rate for today
    try:
        import performance_tracker as pt
        stats = pt.get_summary_stats()
        if isinstance(stats, dict):
            wr  = stats.get("win_rate_7d", 0) or stats.get("win_rate", 0)
            cnt = stats.get("trade_count_7d", 0) or stats.get("total", 0)
            if wr:
                lines.append(f"\n7d win-rate: {wr:.0f}%  ({cnt} trades)")
    except Exception:
        pass

    # Open positions
    if _signal_memory:
        lines.append("\nOpen positions:")
        for ticker, m in _signal_memory.items():
            lines.append(
                f"  {m.get('signal','?')} {ticker} @ "
                f"${m.get('price', 0):.2f}  SL ${m.get('stop_loss', 0):.2f}"
            )

    send_whatsapp("\n".join(lines))
    print("✅ [4:15 PM] Daily P&L sent.")


# ── Sunday 7 PM — Weekly eval ─────────────────────────────────────────────────

async def _weekly_eval():
    """Sunday 7:00 PM — run the eval agent."""
    print("\n" + "─" * 50)
    print("📋 [Sun 7 PM] Weekly eval starting...")
    loop = asyncio.get_running_loop()
    try:
        from agents.eval_agent import run_weekly_eval
        await loop.run_in_executor(_executor, run_weekly_eval)
    except Exception as e:
        print(f"❌ [Sun 7 PM] Weekly eval error: {e}")


# ── Hub daily reset (3 AM) ────────────────────────────────────────────────────

async def _daily_reset():
    """3:00 AM Mon–Fri — reset alert dedup so each ticker can fire once per day."""
    print("🔄 [3:00 AM] Daily hub reset")
    hub.reset_daily()


# ── Scheduler factory ─────────────────────────────────────────────────────────

def _build_scheduler() -> AsyncIOScheduler:
    TZ  = "America/New_York"
    WD  = "mon-fri"
    kw  = dict(timezone=TZ, misfire_grace_time=600, coalesce=True)

    sched = AsyncIOScheduler(timezone=TZ)

    # Daily hub reset (3 AM every day)
    sched.add_job(
        _daily_reset,
        CronTrigger(hour=3, minute=0, timezone=TZ),
        id="daily_reset", replace_existing=True, **{k: v for k, v in kw.items() if k != "timezone"},
    )

    # SCAN 1 — 7:45 AM Mon–Fri
    sched.add_job(
        _scan1_premarket_top5,
        CronTrigger(day_of_week=WD, hour=7, minute=45, timezone=TZ),
        id="scan1_premarket", replace_existing=True,
        misfire_grace_time=600, coalesce=True,
    )

    # SCAN 2 — 9:45 AM Mon–Fri
    sched.add_job(
        _scan2_confirm_entries,
        CronTrigger(day_of_week=WD, hour=9, minute=45, timezone=TZ),
        id="scan2_opening_range", replace_existing=True,
        misfire_grace_time=300, coalesce=True,
    )

    # SCAN 3 — 2:00 PM Mon–Fri
    sched.add_job(
        _scan3_power_hour,
        CronTrigger(day_of_week=WD, hour=14, minute=0, timezone=TZ),
        id="scan3_power_hour", replace_existing=True,
        misfire_grace_time=600, coalesce=True,
    )

    # Exit agent — every 30 min on the hour/half-hour, Mon–Fri
    sched.add_job(
        _exit_agent_check,
        CronTrigger(day_of_week=WD, hour="9-15", minute="0,30", timezone=TZ),
        id="exit_check", replace_existing=True,
        misfire_grace_time=120, coalesce=True,
    )

    # 4:15 PM P&L summary — Mon–Fri
    sched.add_job(
        _send_daily_pnl,
        CronTrigger(day_of_week=WD, hour=16, minute=15, timezone=TZ),
        id="daily_pnl", replace_existing=True,
        misfire_grace_time=600, coalesce=True,
    )

    # Sunday 7 PM weekly eval
    sched.add_job(
        _weekly_eval,
        CronTrigger(day_of_week="sun", hour=19, minute=0, timezone=TZ),
        id="weekly_eval", replace_existing=True,
        misfire_grace_time=1800, coalesce=True,
    )

    return sched


# ── Entry point (called by main.py) ───────────────────────────────────────────

async def scheduler_loop(daily_log: list, signal_memory: dict):
    """
    Main entry point — called via asyncio.create_task() from main.py lifespan.
    Stores shared state references, builds + starts the APScheduler, then
    waits indefinitely (until the task is cancelled on shutdown).
    """
    global _daily_log, _signal_memory
    _daily_log     = daily_log
    _signal_memory = signal_memory

    sched = _build_scheduler()
    sched.start()

    now = datetime.now(tz=EST)
    print(
        f"🗓️  [Scheduler] APScheduler started  "
        f"{now.strftime('%Y-%m-%d %H:%M %Z')}\n"
        f"   Jobs: {', '.join(j.id for j in sched.get_jobs())}"
    )

    try:
        # Run until cancelled (main.py shutdown)
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        sched.shutdown(wait=False)
        print("🗓️  [Scheduler] Stopped.")
