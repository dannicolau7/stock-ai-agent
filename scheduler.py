"""
scheduler.py — Full-day trading schedule.

  3:00 AM  Overnight news scan  (WhatsApp if score >= 85)
  4:00 AM  Early gap check (watchlist, gap ≥ 5% → WhatsApp)
  6:00 AM  Broader gap check (watchlist + universe[:50])
  7:30 AM  Quick Claude pre-market scan (watchlist gappers)
  7:45 AM  Best-of-Day selection (gate filter → score → Claude → WhatsApp)
  8:00 AM  Morning digest (movers + macro + today's picks → WhatsApp)
  8:30 AM  Pre-market gap scanner (dashboard cache update + WhatsApp)
  8:45 AM  Broad pre-market Claude scan (save prep list)
  9:10 AM  PREP alert (final scan → save prep_alert_list.json → WhatsApp)
  9:25 AM  Confirmation check (STILL BUY / WEAKENED / STAND DOWN → WhatsApp)
  9:30 AM  Live monitoring (handled by main.py's monitoring_loop)
  3:30 PM  EOD pre-close scan (accumulation/bounce/coiling → tomorrow_watchlist.json)
  4:00 PM  "Market closed" + after-hours summary
  4:15 PM  EOD after-close (earnings + EDGAR 8-Ks → update tomorrow_watchlist.json)
  4:30 PM  Daily performance report WhatsApp
  6:00 PM  EOD evening scan (AH prices + Polygon news → finalize tomorrow_watchlist.json)
  8:00 PM  EOD final overnight (last EDGAR + breaking news sweep)
 10:00 PM  Final scan  (WhatsApp if score >= 85)
 11:00 PM  Good night + graceful shutdown
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import watchlist_manager as wl
from alerts import send_whatsapp
from graph import GRAPH, make_initial_state
from intelligence_hub import hub
from market_scanner import scan_best_of_day, scan_broad_market

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


def _in_window(hour: int, minute: int, window_min: int = 10) -> bool:
    now    = datetime.now(tz=EST)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    delta  = (now - target).total_seconds()
    return 0 <= delta < (window_min * 60)


# ── Graph runner (off-hours) ───────────────────────────────────────────────────

def _run_graph_sync(ticker: str) -> dict:
    return GRAPH.invoke(make_initial_state(ticker))


async def _run_ticker(ticker: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run_graph_sync, ticker)


# ── Scheduled events ──────────────────────────────────────────────────────────

async def _run_off_hours_scan(event_name: str,
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
            result = await _run_ticker(ticker)
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
                send_whatsapp(msg)
                fired_count += 1

        except Exception as e:
            print(f"❌ [{label}] Error on {ticker}: {e}")

    if fired_count == 0:
        print(f"💤 [{label}] No signals >= {alert_threshold} confidence. Silent.")
    print(f"{'─'*50}\n")


async def _run_morning_sweep():
    """
    7:45 AM — Best-of-Day selection pipeline.
    Pre-seeds universe with pre-market gappers (daily RVOL is ~0 before open,
    so we collect candidates from the premarket scanner first and prepend them
    so they survive Gate 1 ahead of the bulk universe).
    Runs gate filtering → scoring → Claude ranking → WhatsApp (handled internally).
    Takes ~10–15 min due to bulk download + Polygon news rate limits.
    """
    global _sweep_results
    print("\n⏰ [7:45 AM] Starting Best-of-Day selection pipeline...")
    watchlist = wl.load()

    # ── Heartbeat: confirm agent is alive before the scan begins ──────────────
    try:
        import world_context as wctx
        ctx      = wctx.get()
        macro    = ctx["macro"]
        earnings = ctx["earnings"]
        vix_str  = f"VIX {macro.get('vix', 0):.1f}" if macro.get("vix") else "VIX —"
        spy_str  = f"SPY {macro.get('bias', '?')}"
        hot_str  = ""
        today_earns = [
            e["ticker"] for e in earnings.get("upcoming", []) if e.get("days", 99) <= 1
        ]
        if today_earns:
            hot_str = f"\nEarnings today: {', '.join(today_earns)}"
        hb_msg = (
            f"⏰ Argus is awake (7:45 AM EST)\n"
            f"{spy_str}  |  {vix_str}\n"
            f"Watchlist: {', '.join(watchlist)}"
            f"{hot_str}\n"
            f"Running Best-of-Day scan now... 🔍"
        )
        from alerts import send_whatsapp
        send_whatsapp(hb_msg)
    except Exception as _hb_err:
        print(f"⚠️  [7:45 AM] Heartbeat failed: {_hb_err}")

    # Collect pre-market gappers to prioritise them (daily volume bars
    # are empty before 9:30 AM, so RVOL would be ~0 for every stock).
    premarket_priority: list = []
    try:
        from premarket_scanner import scan_premarket_gaps
        gaps = scan_premarket_gaps(watchlist)
        premarket_priority = [g["ticker"] for g in gaps if g["gap_pct"] >= 2.0]
        if premarket_priority:
            print(f"⚡ [7:45 AM] Pre-market gappers to prioritise: {premarket_priority}")
    except Exception as _e:
        print(f"⚠️  [7:45 AM] Pre-market scan skipped: {_e}")

    extra = list(dict.fromkeys(premarket_priority + watchlist))
    loop  = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: scan_best_of_day(
            verbose=False,
            extra_tickers=extra,
            min_score=75,          # lower threshold at 7:45 AM
            rvol_bypass=premarket_priority,  # skip Gate 1 for known gappers
        ),
    )
    # Store winner as single-item list so _send_market_opens_soon can reference it
    _sweep_results = [result] if result else []
    top = result.get("ticker", "—") if result else "—"
    print(f"✅ [7:45 AM] Best-of-Day complete. Winner: {top}")


async def _run_early_gap_check(hour: int, minute: int,
                               mode: str = "quick", gap_threshold: float = 3.0):
    """
    Quick gap scan on watchlist — fires only for large movers (gap ≥ gap_threshold).
    Used at 4 AM, 6 AM.
    """
    label = f"{hour}:{minute:02d} AM"
    print(f"\n{'─'*50}")
    print(f"⚡ [{label}] Early gap check (mode={mode}, threshold={gap_threshold}%)...")
    try:
        from premarket_scanner import scan_premarket_gaps, format_premarket_msg
        import watchlist_manager as _wl
        tickers = _wl.load()
        if mode == "broad":
            try:
                from market_scanner import _load_universe
                universe = _load_universe() or []
                tickers  = list(dict.fromkeys(tickers + universe[:50]))
            except Exception:
                pass

        loop = asyncio.get_running_loop()
        gaps = await loop.run_in_executor(_executor, scan_premarket_gaps, tickers)
        big  = [g for g in gaps if g["gap_pct"] >= gap_threshold]
        if big:
            msg = format_premarket_msg(big[:3])
            print(f"🔔 [{label}] {len(big)} big mover(s) found:\n{msg}")
            send_whatsapp(msg)
        else:
            print(f"💤 [{label}] No movers ≥ {gap_threshold}% yet.")
    except Exception as e:
        print(f"❌ [{label}] Gap check error: {e}")


async def _run_premarket_scan():
    """8:30 AM — scan for pre-market gap movers and alert via WhatsApp."""
    print(f"\n{'─'*50}")
    print("⚡ [8:30 AM] Running pre-market gap scanner...")
    try:
        from premarket_scanner import scan_premarket_gaps, format_premarket_msg
        import main as _main
        tickers = list(dict.fromkeys(wl.load()))  # watchlist first
        loop    = asyncio.get_running_loop()
        gaps    = await loop.run_in_executor(_executor, scan_premarket_gaps, tickers)
        # Store in main.py's cache so the dashboard can show it
        _main._premarket_cache.update({
            "gaps":       gaps[:10],
            "scanned_at": datetime.now(tz=EST).strftime("%H:%M:%S"),
            "count":      len(gaps),
        })
        if gaps:
            msg = format_premarket_msg(gaps[:3])
            print(f"✅ [8:30 AM] {len(gaps)} gap(s) found:\n{msg}")
            send_whatsapp(msg)
        else:
            print("✅ [8:30 AM] No qualifying pre-market gaps today.")
    except Exception as e:
        print(f"❌ [8:30 AM] Pre-market scan error: {e}")


async def _run_morning_digest():
    """8:00 AM — WhatsApp morning digest: movers + macro + today's picks."""
    print(f"\n{'─'*50}")
    print("☀️  [8:00 AM] Sending morning digest...")
    try:
        from premarket_scanner import send_morning_digest
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_executor, send_morning_digest)
    except Exception as e:
        print(f"❌ [8:00 AM] Morning digest error: {e}")


async def _run_broad_premarket_scan():
    """8:45 AM — Full broad pre-market scan with Claude analysis."""
    print(f"\n{'─'*50}")
    print("🌅 [8:45 AM] Running broad pre-market scan (Claude analysis)...")
    try:
        from premarket_scanner import run_premarket_scan
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _executor,
            lambda: run_premarket_scan(mode="broad", test=False, verbose=False, save_prep=True),
        )
    except Exception as e:
        print(f"❌ [8:45 AM] Broad pre-market scan error: {e}")


async def _run_prep_alert():
    """9:10 AM — Final prep scan + save prep_alert_list.json + send PREP alerts."""
    print(f"\n{'─'*50}")
    print("🎯 [9:10 AM] Running PREP alert scan...")
    try:
        from premarket_scanner import run_premarket_scan
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _executor,
            lambda: run_premarket_scan(mode="broad", test=False, verbose=False, save_prep=True),
        )
    except Exception as e:
        print(f"❌ [9:10 AM] Prep alert error: {e}")


async def _run_confirmation_alert():
    """9:25 AM — Re-check prep list → STILL BUY / WEAKENED / STAND DOWN."""
    print(f"\n{'─'*50}")
    print("⏰ [9:25 AM] Running pre-market confirmation check...")
    try:
        from premarket_scanner import run_confirmation_check
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _executor,
            lambda: run_confirmation_check(test=False, verbose=False),
        )
    except Exception as e:
        print(f"❌ [9:25 AM] Confirmation check error: {e}")


async def _send_market_opens_soon():
    """Fallback 9:25 AM message used only if confirmation check is skipped."""
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
    send_whatsapp(msg)


async def _send_market_closed(daily_log: list, signal_memory: dict):
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
    send_whatsapp(msg)


async def _send_daily_report(daily_log: list):
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

    msg = "\n".join(lines)
    send_whatsapp(msg)
    print("\n" + "─"*50 + "\n" + msg + "\n" + "─"*50 + "\n")


async def _send_good_night(daily_log: list):
    today = _today()
    sigs  = [e for e in daily_log if e.get("timestamp", "")[:10] == today]
    msg   = (
        f"🌙 Good night! Argus going to sleep.\n\n"
        f"Today's summary:\n"
        f"- Signals fired: {len(sigs)}\n"
        f"- Scans completed: {sum(1 for v in _fired_today.values() if v == today)}\n\n"
        f"Agent continues running overnight.\nSleep well! 😴"
    )
    send_whatsapp(msg)
    print("🌙 [Scheduler] Good night — daily schedule complete.")


# ── Main scheduler loop ────────────────────────────────────────────────────────

async def scheduler_loop(daily_log: list, signal_memory: dict):
    print("🗓️  [Scheduler] Started — polling every 30s")
    while True:
        try:
            now     = datetime.now(tz=EST)
            weekday = now.weekday() < 5   # Mon–Fri

            # ── 3:00 AM — Overnight scan (every day) + daily hub reset
            if _in_window(3, 0) and _should_fire("overnight_news"):
                _mark_fired("overnight_news")
                hub.reset_daily()   # clear alert dedup so each ticker can fire once today
                await _run_off_hours_scan(
                    "overnight_news",
                    alert_threshold=85,
                    label="3:00 AM OVERNIGHT",
                    daily_log=daily_log,
                )

            # ── 4:00 AM — Early gap check (watchlist, big movers ≥ 5%)
            if weekday and _in_window(4, 0) and _should_fire("early_gap_4am"):
                _mark_fired("early_gap_4am")
                await _run_early_gap_check(4, 0, mode="quick", gap_threshold=5.0)

            # ── 6:00 AM — Broader gap check (watchlist + universe[:50])
            if weekday and _in_window(6, 0) and _should_fire("early_gap_6am"):
                _mark_fired("early_gap_6am")
                await _run_early_gap_check(6, 0, mode="broad", gap_threshold=3.0)

            # ── 7:30 AM — Quick Claude scan on watchlist gappers
            if weekday and _in_window(7, 30) and _should_fire("quick_pm_scan_730"):
                _mark_fired("quick_pm_scan_730")
                try:
                    from premarket_scanner import run_premarket_scan
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        _executor,
                        lambda: run_premarket_scan(mode="quick", test=False, verbose=False, save_prep=False),
                    )
                except Exception as _e:
                    print(f"❌ [7:30 AM] Quick pre-market scan error: {_e}")

            # ── 7:45 AM — Best-of-Day selection (includes WhatsApp send)
            if weekday and _in_window(7, 45) and _should_fire("morning_sweep"):
                _mark_fired("morning_sweep")
                await _run_morning_sweep()

            # ── 8:00 AM — Morning digest
            if weekday and _in_window(8, 0) and _should_fire("morning_digest"):
                _mark_fired("morning_digest")
                await _run_morning_digest()

            # ── 8:30 AM — Pre-market gap scan (dashboard cache)
            if weekday and _in_window(8, 30) and _should_fire("premarket_gaps"):
                _mark_fired("premarket_gaps")
                await _run_premarket_scan()

            # ── 8:45 AM — Broad pre-market scan with Claude analysis
            if weekday and _in_window(8, 45) and _should_fire("broad_pm_scan"):
                _mark_fired("broad_pm_scan")
                await _run_broad_premarket_scan()

            # ── 9:10 AM — Final PREP alert (save prep_alert_list.json)
            if weekday and _in_window(9, 10) and _should_fire("prep_alert"):
                _mark_fired("prep_alert")
                await _run_prep_alert()

            # ── 9:25 AM — Confirmation check (STILL BUY / WEAKENED / STAND DOWN)
            if weekday and _in_window(9, 25) and _should_fire("market_opens_soon"):
                _mark_fired("market_opens_soon")
                await _run_confirmation_alert()

            # ── 3:30 PM — EOD pre-close scan (accumulation / coiling / bounce)
            if weekday and _in_window(15, 30) and _should_fire("eod_preclose"):
                _mark_fired("eod_preclose")
                try:
                    from eod_scanner import run_preclose_scan
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(_executor, lambda: run_preclose_scan(test=False))
                except Exception as _e:
                    print(f"❌ [3:30 PM] EOD pre-close error: {_e}")

            # ── 4:00 PM — Market closed
            if weekday and _in_window(16, 0) and _should_fire("market_closed"):
                _mark_fired("market_closed")
                await _send_market_closed(daily_log, signal_memory)

            # ── 4:15 PM — EOD after-close (earnings releases + EDGAR 8-Ks)
            if weekday and _in_window(16, 15) and _should_fire("eod_afterclose"):
                _mark_fired("eod_afterclose")
                try:
                    from eod_scanner import run_afterclose_scan
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(_executor, lambda: run_afterclose_scan(test=False))
                except Exception as _e:
                    print(f"❌ [4:15 PM] EOD after-close error: {_e}")

            # ── 4:30 PM — Daily report
            if weekday and _in_window(16, 30) and _should_fire("daily_report"):
                _mark_fired("daily_report")
                await _send_daily_report(daily_log)

            # ── 6:00 PM — EOD evening scan (AH prices + news finalization)
            if weekday and _in_window(18, 0) and _should_fire("eod_evening"):
                _mark_fired("eod_evening")
                try:
                    from eod_scanner import run_evening_scan
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(_executor, lambda: run_evening_scan(test=False))
                except Exception as _e:
                    print(f"❌ [6:00 PM] EOD evening scan error: {_e}")

            # ── 8:00 PM — EOD final overnight check
            if weekday and _in_window(20, 0) and _should_fire("eod_final"):
                _mark_fired("eod_final")
                try:
                    from eod_scanner import run_final_overnight
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(_executor, lambda: run_final_overnight(test=False))
                except Exception as _e:
                    print(f"❌ [8:00 PM] EOD final overnight error: {_e}")

            # ── 10:00 PM — Final scan
            if weekday and _in_window(22, 0) and _should_fire("final_scan"):
                _mark_fired("final_scan")
                await _run_off_hours_scan(
                    "final_scan",
                    alert_threshold=85,
                    label="10:00 PM FINAL",
                    daily_log=daily_log,
                )

            # ── 11:00 PM — Good night + shutdown
            if weekday and _in_window(23, 0) and _should_fire("good_night"):
                _mark_fired("good_night")
                await _send_good_night(daily_log)
                return

        except Exception as e:
            print(f"❌ [Scheduler] Loop error: {e}")

        await asyncio.sleep(30)
