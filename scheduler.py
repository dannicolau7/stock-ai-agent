"""
scheduler.py — Full-day trading schedule.

  3:00 AM  Overnight news scan  (WhatsApp if score >= 85)
  7:45 AM  Broad market sweep (~50 tickers, no Claude)
  8:00 AM  Morning digest WhatsApp — "Top 5 opportunities"
  9:25 AM  "Market opens in 5 min" alert
  9:30 AM  Live monitoring (handled by main.py's monitoring_loop)
  4:00 PM  "Market closed" + after-hours summary
  4:30 PM  Daily performance report WhatsApp
  6:00 PM  Earnings scan  (WhatsApp if score >= 75)
  8:00 PM  Evening news scan  (WhatsApp if score >= 85)
 10:00 PM  Final scan  (WhatsApp if score >= 85)
 11:00 PM  Good night + graceful shutdown
"""

import asyncio
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import watchlist_manager as wl
from alerts import send_whatsapp
from graph import GRAPH, make_initial_state
from market_scanner import scan_broad_market

EST      = ZoneInfo("America/New_York")
_executor = ThreadPoolExecutor(max_workers=2)

# ── Fire-once-per-day guard ────────────────────────────────────────────────────

_fired_today: dict = {}
_sweep_results: list = []   # shared between 7:45 sweep and 8:00 digest


def _today() -> str:
    return datetime.now(tz=EST).strftime("%Y-%m-%d")


def _should_fire(event: str) -> bool:
    return _fired_today.get(event) != _today()


def _mark_fired(event: str):
    _fired_today[event] = _today()


def _in_window(hour: int, minute: int, window_min: int = 3) -> bool:
    now    = datetime.now(tz=EST)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    delta  = (now - target).total_seconds()
    return 0 <= delta < (window_min * 60)


# ── Graph runner (off-hours) ───────────────────────────────────────────────────

def _run_graph_sync(ticker: str, paper: bool) -> dict:
    return GRAPH.invoke(make_initial_state(ticker, paper_trading=paper))


async def _run_ticker(ticker: str, paper: bool) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run_graph_sync, ticker, paper)


# ── Scheduled events ──────────────────────────────────────────────────────────

async def _run_off_hours_scan(event_name: str, paper: bool,
                               alert_threshold: int, label: str,
                               daily_log: list):
    tickers = wl.load()
    print(f"\n{'─'*50}")
    print(f"⏰ [{label}] Scanning {len(tickers)} tickers...")

    fired_count = 0
    for i, ticker in enumerate(tickers):
        if i > 0:
            await asyncio.sleep(20)   # Polygon rate limit buffer
        try:
            result = await _run_ticker(ticker, paper)
            conf   = result.get("confidence", 0)
            sig    = result.get("signal", "HOLD")
            price  = result.get("current_price", 0.0)

            if sig in ("BUY", "SELL") and conf >= alert_threshold:
                emoji = "🟢 BUY" if sig == "BUY" else "🔴 SELL"
                msg = (
                    f"{label} ALERT\n"
                    f"{emoji} {ticker} — score {conf}/100\n"
                    f"${price:.2f} | RSI {result.get('rsi', 0):.0f}\n"
                    f"Entry: {result.get('entry_zone', '—')}\n"
                    f"Target: ${result.get('targets', [0])[0]:.2f} | "
                    f"Stop: ${result.get('stop_loss', 0):.2f}\n"
                    f"{result.get('reasoning', '')[:200]}"
                )
                if not paper:
                    send_whatsapp(msg)
                else:
                    print(f"📋 [PAPER] Would send:\n{msg}")
                fired_count += 1

        except Exception as e:
            print(f"❌ [{label}] Error on {ticker}: {e}")

    if fired_count == 0:
        print(f"💤 [{label}] No signals >= {alert_threshold} confidence. Silent.")
    print(f"{'─'*50}\n")


async def _run_morning_sweep(paper: bool):
    global _sweep_results
    print("\n⏰ [7:45 AM] Starting broad market sweep...")
    watchlist = wl.load()
    loop = asyncio.get_running_loop()
    _sweep_results = await loop.run_in_executor(
        _executor, scan_broad_market, watchlist, 5
    )
    print(f"✅ [7:45 AM] Sweep done. Top pick: {_sweep_results[0]['ticker'] if _sweep_results else '—'}")


async def _send_morning_digest(paper: bool):
    if not _sweep_results:
        print("⚠️  [8:00 AM] No sweep results — skipping digest")
        return

    lines = ["☀️ Good morning! Top 5 opportunities today (7:45 AM sweep)\n"]
    for i, r in enumerate(_sweep_results, 1):
        sig = "BUY setup" if r["score"] >= 65 else "WATCH" if r["score"] >= 50 else "HOLD"
        lines.append(
            f"{i}. {r['ticker']} — score {r['score']} | ${r['price']:.2f} | {sig}\n"
            f"   RSI {r['rsi']:.0f}, vol {r['vol_ratio']:.1f}x | {r['reason']}"
        )
    lines.append("\nMarket opens 9:30 AM EST. Good luck! 🚀")
    msg = "\n".join(lines)

    if not paper:
        send_whatsapp(msg)
    else:
        print(f"📋 [PAPER] Morning digest:\n{msg}")


async def _send_market_opens_soon(paper: bool):
    top = _sweep_results[:2] if _sweep_results else []
    watch_lines = "\n".join(
        f"- {r['ticker']} ${r['price']:.2f} (score {r['score']})"
        for r in top
    ) or "- " + ", ".join(wl.load())

    msg = (
        f"⏰ Market opens in 5 min (9:30 AM EST)\n\n"
        f"Top watch today:\n{watch_lines}\n\n"
        f"Stay sharp! 👀"
    )
    if not paper:
        send_whatsapp(msg)
    else:
        print(f"📋 [PAPER] Pre-market alert:\n{msg}")


async def _send_market_closed(paper: bool, daily_log: list, signal_memory: dict):
    today  = _today()
    sigs   = [e for e in daily_log if e.get("timestamp", "")[:10] == today]

    lines  = ["🔔 Market closed (4:00 PM EST)\n"]
    if sigs:
        lines.append(f"Signals fired today: {len(sigs)}")
        for s in sigs:
            icon = "🟢" if s["signal"] == "BUY" else "🔴"
            t    = s.get("timestamp", "")[-8:-3]
            lines.append(f"  {icon} {s['signal']} {s['ticker']} @ ${s['price']:.2f}  conf={s['confidence']}  {t}")
    else:
        lines.append("No BUY/SELL signals fired today.")

    if signal_memory:
        lines.append("\nOpen positions:")
        for ticker, m in signal_memory.items():
            lines.append(f"  {m['signal']} {ticker} @ ${m['price']:.2f} | SL ${m['stop_loss']:.2f}")

    msg = "\n".join(lines)
    if not paper:
        send_whatsapp(msg)
    else:
        print(f"📋 [PAPER] Market closed:\n{msg}")


async def _send_daily_report(paper: bool, daily_log: list):
    today  = _today()
    sigs   = [e for e in daily_log if e.get("timestamp", "")[:10] == today]

    lines  = [f"📊 Daily Report — {today}\n"]
    if not sigs:
        lines.append("No BUY/SELL signals fired today.")
    else:
        lines.append(f"Total signals: {len(sigs)}")
        for s in sigs:
            icon = "🟢" if s["signal"] == "BUY" else "🔴"
            lines.append(f"  {icon} {s['signal']} {s['ticker']} @ ${s['price']:.2f}  conf={s['confidence']}")

    if paper:
        lines.append("\n📋 PAPER TRADING MODE")

    msg = "\n".join(lines)
    if not paper:
        send_whatsapp(msg)
    else:
        print(f"📋 [PAPER] Daily report:\n{msg}")
    print("\n" + "─"*50 + "\n" + msg + "\n" + "─"*50 + "\n")


async def _send_good_night(paper: bool, daily_log: list):
    today = _today()
    sigs  = [e for e in daily_log if e.get("timestamp", "")[:10] == today]
    msg   = (
        f"🌙 Good night! Stock AI Agent shutting down.\n\n"
        f"Today's summary:\n"
        f"- Signals fired: {len(sigs)}\n"
        f"- Scans completed: {sum(1 for v in _fired_today.values() if v == today)}\n\n"
        f"Agent will restart at 3:00 AM.\nSleep well! 😴"
    )
    if not paper:
        send_whatsapp(msg)
    else:
        print(f"📋 [PAPER] Good night:\n{msg}")

    print("🌙 [Scheduler] Good night — shutting down agent...")
    await asyncio.sleep(2)
    os.kill(os.getpid(), signal.SIGTERM)


# ── Main scheduler loop ────────────────────────────────────────────────────────

async def scheduler_loop(paper: bool, daily_log: list, signal_memory: dict):
    print("🗓️  [Scheduler] Started — polling every 30s")
    while True:
        try:
            now     = datetime.now(tz=EST)
            weekday = now.weekday() < 5   # Mon–Fri

            # ── 3:00 AM — Overnight scan (every day)
            if _in_window(3, 0) and _should_fire("overnight_news"):
                _mark_fired("overnight_news")
                await _run_off_hours_scan(
                    "overnight_news", paper,
                    alert_threshold=85,
                    label="3:00 AM OVERNIGHT",
                    daily_log=daily_log,
                )

            # ── 7:45 AM — Broad market sweep
            if weekday and _in_window(7, 45) and _should_fire("morning_sweep"):
                _mark_fired("morning_sweep")
                await _run_morning_sweep(paper)

            # ── 8:00 AM — Morning digest
            if weekday and _in_window(8, 0) and _should_fire("morning_digest"):
                _mark_fired("morning_digest")
                await _send_morning_digest(paper)

            # ── 9:25 AM — Market opens soon
            if weekday and _in_window(9, 25) and _should_fire("market_opens_soon"):
                _mark_fired("market_opens_soon")
                await _send_market_opens_soon(paper)

            # ── 4:00 PM — Market closed
            if weekday and _in_window(16, 0) and _should_fire("market_closed"):
                _mark_fired("market_closed")
                await _send_market_closed(paper, daily_log, signal_memory)

            # ── 4:30 PM — Daily report
            if weekday and _in_window(16, 30) and _should_fire("daily_report"):
                _mark_fired("daily_report")
                await _send_daily_report(paper, daily_log)

            # ── 6:00 PM — Earnings scan
            if weekday and _in_window(18, 0) and _should_fire("earnings_scan"):
                _mark_fired("earnings_scan")
                await _run_off_hours_scan(
                    "earnings_scan", paper,
                    alert_threshold=75,
                    label="6:00 PM EARNINGS",
                    daily_log=daily_log,
                )

            # ── 8:00 PM — Evening scan
            if weekday and _in_window(20, 0) and _should_fire("evening_scan"):
                _mark_fired("evening_scan")
                await _run_off_hours_scan(
                    "evening_scan", paper,
                    alert_threshold=85,
                    label="8:00 PM EVENING",
                    daily_log=daily_log,
                )

            # ── 10:00 PM — Final scan
            if weekday and _in_window(22, 0) and _should_fire("final_scan"):
                _mark_fired("final_scan")
                await _run_off_hours_scan(
                    "final_scan", paper,
                    alert_threshold=85,
                    label="10:00 PM FINAL",
                    daily_log=daily_log,
                )

            # ── 11:00 PM — Good night + shutdown
            if weekday and _in_window(23, 0) and _should_fire("good_night"):
                _mark_fired("good_night")
                await _send_good_night(paper, daily_log)
                return

        except Exception as e:
            print(f"❌ [Scheduler] Loop error: {e}")

        await asyncio.sleep(30)
