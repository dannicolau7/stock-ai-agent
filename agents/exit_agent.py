"""
agents/exit_agent.py — Catalyst swing trade exit manager.

manage_exit(ticker, entry_price, entry_date, target, stop)
  Async coroutine — polls price every 30 min during market hours and fires
  SELL or TAKE_PROFIT when any exit condition is met.
  Designed to be spawned with: asyncio.create_task(manage_exit(...))

Exit conditions (checked in priority order):
  1. price >= target * 0.98                   → TAKE_PROFIT (at target)
  2. price <= effective_stop                  → SELL (stop / trailing stop hit)
  3. next_business_day AND price > entry*1.05 → TAKE_PROFIT (overnight winner)
  4. next_business_day AND price < entry      → SELL (overnight loser)
  5. days >= 3 AND price < entry*1.03         → SELL (news faded, catalyst expired)

Trailing stop tiers (raise effective_stop as unrealised profit grows):
  price >= entry + 5%  → stop → entry_price (breakeven, tier 1)
  price >= entry + 10% → stop → session_high * 0.93  (7% trail, tier 2)
  price >= entry + 15% → stop → session_high * 0.90  (10% trail, tier 3)

Also exports:
  start_exit_watch(...)   — schedule manage_exit() as an asyncio.Task
  cancel_exit_watch(...)  — cancel an active watcher
  get_active_watches()    — list of tickers currently being monitored
"""

import asyncio
import csv
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf
from langsmith import traceable

_EST  = ZoneInfo("America/New_York")
_DATA = Path(__file__).parent.parent / "data"

_TRADES_LOG        = _DATA / "trades_log.csv"
_TRADES_LOG_FIELDS = [
    "ts", "ticker", "signal", "price", "entry_price", "pnl_pct", "outcome",
    "reason", "hold_hours", "session_high",
]

POLL_INTERVAL_S = 30 * 60   # 30 minutes between price checks
MARKET_OPEN     = (9,  30)  # HH, MM  Eastern Time
MARKET_CLOSE    = (16,  0)


# ── NYSE holidays (observed dates) ────────────────────────────────────────────

_NYSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025,  1,  1), date(2025,  1, 20), date(2025,  2, 17),
    date(2025,  4, 18), date(2025,  5, 26), date(2025,  6, 19),
    date(2025,  7,  4), date(2025,  9,  1), date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026,  1,  1), date(2026,  1, 19), date(2026,  2, 16),
    date(2026,  4,  3), date(2026,  5, 25), date(2026,  6, 19),
    date(2026,  7,  3), date(2026,  9,  7), date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027,  1,  1), date(2027,  1, 18), date(2027,  2, 15),
    date(2027,  3, 26), date(2027,  5, 31), date(2027,  6, 18),
    date(2027,  7,  5), date(2027,  9,  6), date(2027, 11, 25),
    date(2027, 12, 24),
}


def _is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _NYSE_HOLIDAYS


def next_business_day(from_date: date) -> date:
    """Return the first NYSE trading day strictly after from_date."""
    d = from_date + timedelta(days=1)
    while not _is_business_day(d):
        d += timedelta(days=1)
    return d


def _is_market_hours(dt: datetime) -> bool:
    """True when dt falls inside NYSE trading hours (9:30–16:00 ET, business days)."""
    if not _is_business_day(dt.date()):
        return False
    t      = dt.time()
    open_  = dt.replace(hour=MARKET_OPEN[0],  minute=MARKET_OPEN[1],
                         second=0, microsecond=0)
    close_ = dt.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1],
                         second=0, microsecond=0)
    return open_.time() <= t < close_.time()


def _seconds_until_market_open(dt: datetime) -> float:
    """Seconds from dt until the next NYSE market open."""
    d = dt.date()
    while True:
        if _is_business_day(d):
            candidate = datetime(d.year, d.month, d.day,
                                  MARKET_OPEN[0], MARKET_OPEN[1],
                                  tzinfo=_EST)
            if candidate > dt:
                return (candidate - dt).total_seconds()
        d += timedelta(days=1)


# ── Per-trade state ────────────────────────────────────────────────────────────

@dataclass
class ExitWatch:
    ticker:       str
    entry_price:  float
    entry_date:   datetime   # tz-aware (EST preferred)
    target:       float
    stop:         float      # original hard stop (never lowered)

    # Live state — initialised from entry_price
    peak_price:    float = field(init=False)
    effective_stop: float = field(init=False)
    trail_level:   int   = field(init=False, default=0)  # 0 = none, 1/2/3 = tiers

    def __post_init__(self):
        self.peak_price     = self.entry_price
        self.effective_stop = self.stop

    def update_peak(self, price: float) -> None:
        if price > self.peak_price:
            self.peak_price = price

    def advance_trailing_stop(self, price: float) -> bool:
        """
        Raise effective_stop based on current unrealised gain.
        Returns True when the stop was moved (for logging).
        """
        gain_pct = (price - self.entry_price) / self.entry_price * 100
        raised   = False

        if gain_pct >= 15 and self.trail_level < 3:
            new_stop = round(self.peak_price * 0.90, 4)   # 10% below peak
            if new_stop > self.effective_stop:
                self.effective_stop = new_stop
                self.trail_level    = 3
                raised = True

        elif gain_pct >= 10 and self.trail_level < 2:
            new_stop = round(self.peak_price * 0.93, 4)   # 7% below peak
            if new_stop > self.effective_stop:
                self.effective_stop = new_stop
                self.trail_level    = 2
                raised = True

        elif gain_pct >= 5 and self.trail_level < 1:
            new_stop = round(self.entry_price, 4)          # breakeven
            if new_stop > self.effective_stop:
                self.effective_stop = new_stop
                self.trail_level    = 1
                raised = True

        return raised


# ── Price helper ───────────────────────────────────────────────────────────────

def _get_price(ticker: str) -> float:
    try:
        fi = yf.Ticker(ticker).fast_info
        p  = float(fi.last_price or fi.previous_close or 0)
        return p if math.isfinite(p) and p > 0 else 0.0
    except Exception as e:
        print(f"⚠️  [ExitAgent] Price fetch error for {ticker}: {e}")
        return 0.0


# ── CSV logging ────────────────────────────────────────────────────────────────

def _log_exit(
    ticker:      str,
    signal:      str,
    exit_price:  float,
    entry_price: float,
    pnl_pct:     float,
    reason:      str,
    hold_hours:  float,
    session_high: float,
    outcome:     str,
) -> None:
    try:
        _DATA.mkdir(parents=True, exist_ok=True)
        write_header = not _TRADES_LOG.exists()
        with open(_TRADES_LOG, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=_TRADES_LOG_FIELDS, extrasaction="ignore"
            )
            if write_header:
                writer.writeheader()
            writer.writerow({
                "ts":           datetime.now().isoformat(),
                "ticker":       ticker,
                "signal":       signal,
                "price":        round(exit_price,  4),
                "entry_price":  round(entry_price, 4),
                "pnl_pct":      round(pnl_pct, 2),
                "outcome":      outcome,
                "reason":       reason,
                "hold_hours":   round(hold_hours, 1),
                "session_high": round(session_high, 4),
            })
        print(f"📝 [ExitAgent] {signal} {ticker} {pnl_pct:+.1f}% logged → trades_log.csv")
    except Exception as e:
        print(f"⚠️  [ExitAgent] trades_log write failed: {e}")


# ── WhatsApp / alert ───────────────────────────────────────────────────────────

def _send_exit_alert(
    ticker:      str,
    signal:      str,
    entry_price: float,
    exit_price:  float,
    pnl_pct:     float,
    reason:      str,
    hold_hours:  float,
) -> None:
    try:
        from alerts import send_whatsapp
        now_str = datetime.now(tz=_EST).strftime("%I:%M %p ET")
        emoji   = "💰 TAKE PROFIT" if signal == "TAKE_PROFIT" else "🛑 SELL"
        pnl_str = f"+{pnl_pct:.1f}%" if pnl_pct >= 0 else f"{pnl_pct:.1f}%"

        msg = (
            f"{emoji} {ticker}\n"
            f"Entry: ${entry_price:.2f}  →  Exit: ${exit_price:.2f}\n"
            f"Reason: {reason}\n"
            f"P&L: {pnl_str}\n"
            f"Hold time: {hold_hours:.0f}h\n"
            f"⏰ {now_str}"
        )
        sent = send_whatsapp(msg)
        if not sent:
            print(f"⚠️  [ExitAgent] WhatsApp not confirmed for {ticker} {signal}")
    except Exception as e:
        print(f"⚠️  [ExitAgent] WhatsApp failed: {e}")


# ── Exit condition evaluation ──────────────────────────────────────────────────

_TRAIL_LABELS = {0: "hard stop", 1: "breakeven stop", 2: "7% trailing stop", 3: "10% trailing stop"}


def _check_exit_conditions(
    watch: ExitWatch,
    price: float,
    now: datetime,
) -> tuple[str | None, str]:
    """
    Evaluate all exit conditions in priority order.
    Returns (signal, reason) or (None, "") if no exit triggered.
    """
    entry   = watch.entry_price
    target  = watch.target
    eff_stp = watch.effective_stop

    entry_date = watch.entry_date
    if entry_date.tzinfo is None:
        entry_date = entry_date.replace(tzinfo=_EST)

    nbd       = next_business_day(entry_date.date())
    today     = now.date()
    days_held = (now - entry_date).total_seconds() / 86400

    # 1. At or near target
    if price >= target * 0.98:
        pct = (price - entry) / entry * 100
        return "TAKE_PROFIT", f"Target ${target:.2f} reached (${price:.2f}, +{pct:.1f}%)"

    # 2. Stop or trailing stop hit
    if price <= eff_stp:
        label = _TRAIL_LABELS.get(watch.trail_level, "stop")
        return "SELL", f"{label} hit at ${eff_stp:.2f} (price ${price:.2f})"

    # 3. Next business day: price held and up 5%+ → book profit
    if today >= nbd and price > entry * 1.05:
        pct = (price - entry) / entry * 100
        return "TAKE_PROFIT", f"Overnight winner +{pct:.1f}% — book day-2 gains"

    # 4. Next business day: price below entry → cut loss
    if today >= nbd and price < entry:
        pct = (price - entry) / entry * 100
        return "SELL", f"Below entry {pct:.1f}% by next business day — cut loss"

    # 5. 3+ days held without reaching +5% → news faded
    if days_held >= 3 and price < entry * 1.05:
        pct = (price - entry) / entry * 100
        return "SELL", (
            f"News faded — {days_held:.0f}d held, "
            f"price {pct:+.1f}% vs entry (no momentum)"
        )

    return None, ""


# ── Core coroutine ─────────────────────────────────────────────────────────────

@traceable(name="exit_agent", tags=["pipeline", "exit"])
async def manage_exit(
    ticker:      str,
    entry_price: float,
    entry_date:  datetime,
    target:      float,
    stop:        float,
) -> dict:
    """
    Monitor a catalyst swing trade and fire an exit when conditions are met.

    Parameters
    ----------
    ticker       : stock symbol
    entry_price  : fill price at entry
    entry_date   : datetime when the trade was opened (EST preferred)
    target       : price target (TAKE_PROFIT fires at 98% of this)
    stop         : initial hard stop (raised automatically by trailing logic)

    Returns
    -------
    dict: {ticker, signal, exit_price, entry_price, pnl_pct, reason,
           hold_hours, session_high, trail_level}

    Usage
    -----
    task = asyncio.create_task(manage_exit("HIMS", 31.0, datetime.now(EST), 36.0, 28.5))
    result = await task   # resolves when exit fires
    """
    # Ensure entry_date is tz-aware
    if entry_date.tzinfo is None:
        entry_date = entry_date.replace(tzinfo=_EST)

    watch = ExitWatch(
        ticker=ticker,
        entry_price=entry_price,
        entry_date=entry_date,
        target=target,
        stop=stop,
    )

    print(
        f"👁️  [ExitAgent] Watching {ticker} — "
        f"entry=${entry_price:.2f}  target=${target:.2f}  stop=${stop:.2f}  "
        f"next_bd={next_business_day(entry_date.date())}"
    )

    loop = asyncio.get_running_loop()

    while True:
        try:
            now = datetime.now(_EST)

            # ── Outside market hours: sleep until next open ───────────────────
            if not _is_market_hours(now):
                secs = _seconds_until_market_open(now)
                sleep_secs = min(secs, 3600)   # wake at most every hour to re-check
                print(
                    f"🌙 [ExitAgent] {ticker} — market closed, "
                    f"sleeping {sleep_secs/60:.0f} min "
                    f"(opens in {secs/3600:.1f}h)"
                )
                await asyncio.sleep(sleep_secs)
                continue

            # ── Fetch current price ───────────────────────────────────────────
            price = await loop.run_in_executor(None, _get_price, ticker)
            if price <= 0:
                print(f"⚠️  [ExitAgent] {ticker} price unavailable — retrying in 5 min")
                await asyncio.sleep(300)
                continue

            # ── Update peak + advance trailing stop ───────────────────────────
            watch.update_peak(price)
            raised = watch.advance_trailing_stop(price)
            if raised:
                label = {1: "breakeven", 2: "7% trail", 3: "10% trail"}.get(
                    watch.trail_level, "?"
                )
                print(
                    f"📈 [ExitAgent] {ticker} trailing stop → "
                    f"${watch.effective_stop:.2f}  [{label}]"
                )

            pnl_pct = (price - entry_price) / entry_price * 100
            print(
                f"👁️  [ExitAgent] {ticker} @ ${price:.2f}  "
                f"{pnl_pct:+.1f}%  "
                f"peak=${watch.peak_price:.2f}  "
                f"eff_stop=${watch.effective_stop:.2f}  "
                f"trail=T{watch.trail_level}"
            )

            # ── Check exit conditions ─────────────────────────────────────────
            signal, reason = _check_exit_conditions(watch, price, now)

            if signal:
                hold_hours = (now - entry_date).total_seconds() / 3600
                outcome    = "WIN" if pnl_pct >= 0 else "LOSS"

                _log_exit(
                    ticker=ticker,
                    signal=signal,
                    exit_price=price,
                    entry_price=entry_price,
                    pnl_pct=pnl_pct,
                    reason=reason,
                    hold_hours=hold_hours,
                    session_high=watch.peak_price,
                    outcome=outcome,
                )
                _send_exit_alert(
                    ticker=ticker,
                    signal=signal,
                    entry_price=entry_price,
                    exit_price=price,
                    pnl_pct=pnl_pct,
                    reason=reason,
                    hold_hours=hold_hours,
                )

                icon = "💰" if signal == "TAKE_PROFIT" else "🛑"
                print(
                    f"{icon} [ExitAgent] {signal} {ticker} "
                    f"@ ${price:.2f}  {pnl_pct:+.1f}%  "
                    f"held {hold_hours:.0f}h  — {reason}"
                )

                return {
                    "ticker":       ticker,
                    "signal":       signal,
                    "exit_price":   round(price, 4),
                    "entry_price":  round(entry_price, 4),
                    "pnl_pct":      round(pnl_pct, 2),
                    "reason":       reason,
                    "hold_hours":   round(hold_hours, 1),
                    "session_high": round(watch.peak_price, 4),
                    "trail_level":  watch.trail_level,
                }

            await asyncio.sleep(POLL_INTERVAL_S)

        except asyncio.CancelledError:
            print(f"🛑 [ExitAgent] {ticker} watch cancelled.")
            raise
        except Exception as e:
            print(f"❌ [ExitAgent] {ticker} loop error: {e} — retrying in 5 min")
            await asyncio.sleep(300)


# ── Task registry ──────────────────────────────────────────────────────────────

_active_watches: dict[str, asyncio.Task] = {}


def start_exit_watch(
    ticker:      str,
    entry_price: float,
    entry_date:  datetime,
    target:      float,
    stop:        float,
) -> "asyncio.Task | None":
    """
    Schedule manage_exit() as an asyncio.Task.
    Returns the Task, or None if this ticker is already being watched.

    Must be called from inside a running event loop.
    """
    if ticker in _active_watches and not _active_watches[ticker].done():
        print(f"⚠️  [ExitAgent] {ticker} already being watched — ignoring duplicate.")
        return None
    task = asyncio.create_task(
        manage_exit(ticker, entry_price, entry_date, target, stop),
        name=f"exit_watch_{ticker}",
    )
    _active_watches[ticker] = task
    return task


def cancel_exit_watch(ticker: str) -> None:
    """Cancel an active exit watcher (e.g. position manually closed)."""
    task = _active_watches.pop(ticker, None)
    if task and not task.done():
        task.cancel()
        print(f"🛑 [ExitAgent] {ticker} watch cancelled.")


def get_active_watches() -> list[str]:
    """Return tickers currently being monitored."""
    return [t for t, task in _active_watches.items() if not task.done()]


# ── Standalone test (paper / simulation mode) ──────────────────────────────────

if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "HIMS"

    # Fetch real current price
    try:
        current = float(yf.Ticker(ticker).fast_info.last_price or 0)
    except Exception:
        current = 30.0

    entry  = current * 0.92   # simulate entry 8% below current
    target = current * 1.10
    stop   = current * 0.85

    print(f"\n=== ExitAgent paper test: {ticker} ===")
    print(f"  entry=${entry:.2f}  current=${current:.2f}  target=${target:.2f}  stop=${stop:.2f}")
    print()

    # Unit-test the exit condition checker without running the async loop
    watch = ExitWatch(
        ticker=ticker,
        entry_price=entry,
        entry_date=datetime.now(_EST) - timedelta(hours=2),
        target=target,
        stop=stop,
    )
    watch.update_peak(current)
    watch.advance_trailing_stop(current)

    signal, reason = _check_exit_conditions(watch, current, datetime.now(_EST))
    print(f"  trail_level : {watch.trail_level}")
    print(f"  effective_stop: ${watch.effective_stop:.2f}")
    print(f"  exit signal : {signal or 'none — still holding'}")
    if reason:
        print(f"  reason      : {reason}")

    # Also verify the next_business_day helper
    today = date.today()
    nbd   = next_business_day(today)
    print(f"\n  next_business_day({today}) = {nbd}")

    # Verify business day calculation across a holiday
    xmas = date(2026, 12, 25)
    print(f"  next_business_day(2026-12-25) = {next_business_day(xmas)}")
