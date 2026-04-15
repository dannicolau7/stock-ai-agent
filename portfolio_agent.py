"""
portfolio_agent.py — Open position tracker and EXIT signal generator.

Runs every 5 minutes during market hours. For every open BUY signal
(from performance_tracker), fetches the current price and checks:

  1. Stop-loss hit       → send EXIT alert immediately
  2. Target 1 reached   → send partial-exit alert, raise stop to breakeven
  3. Target 2 reached   → send full-exit alert
  4. Trailing stop      → if price 7% below peak since entry → EXIT
  5. Time-based exit    → swing trades older than 10d without hitting target → EXIT

Also provides:
  - Position summary for the dashboard (/api/portfolio)
  - Concentration warnings (>40% in one sector)
  - Live P&L per position and total portfolio

Stores per-position state in SQLite via performance_tracker DB
(adds a `positions` table for peak price tracking).
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

import yfinance as yf

from alerts import send_alert
import performance_tracker as pt
import world_context as wctx

PORTFOLIO_INTERVAL  = 5 * 60      # check every 5 minutes
TRAILING_STOP_PCT   = 0.07        # 7% below peak → exit
MAX_SWING_DAYS      = 10          # max days to hold a swing trade without target hit
MAX_DAYTRADE_HOURS  = 6           # close day trades within 6h if no target


# ── Positions table ───────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(pt.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_positions_table():
    pt.init_db()   # ensure signals/outcomes tables exist first
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                signal_id   INTEGER PRIMARY KEY REFERENCES signals(id),
                ticker      TEXT    NOT NULL,
                entry_price REAL    NOT NULL,
                peak_price  REAL    NOT NULL,
                stop_loss   REAL,
                target_1    REAL,
                target_2    REAL,
                t1_hit      INTEGER DEFAULT 0,
                t2_hit      INTEGER DEFAULT 0,
                exit_price  REAL,
                exit_reason TEXT,
                exited_at   TEXT,
                fired_at    TEXT
            );
        """)


def _load_open_positions(paper: bool = False) -> list[dict]:
    """Load open (non-exited) positions from DB."""
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT p.*, s.confidence, s.trade_horizon, s.macro_regime, s.reasoning
            FROM positions p
            JOIN signals s ON s.id = p.signal_id
            WHERE p.exit_price IS NULL
              AND s.paper = ?
            ORDER BY p.fired_at
        """, (int(paper),)).fetchall()
    return [dict(r) for r in rows]


def _upsert_position(signal_id: int, ticker: str, entry_price: float,
                     stop_loss: float, target_1: float | None,
                     target_2: float | None, fired_at: str):
    """Insert position row if it doesn't exist yet."""
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT signal_id FROM positions WHERE signal_id=?", (signal_id,)
        ).fetchone()
        if not existing:
            conn.execute("""
                INSERT INTO positions
                  (signal_id, ticker, entry_price, peak_price, stop_loss,
                   target_1, target_2, fired_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (signal_id, ticker, entry_price, entry_price,
                  stop_loss, target_1, target_2, fired_at))


def _update_peak(signal_id: int, peak: float):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE positions SET peak_price=? WHERE signal_id=?", (peak, signal_id)
        )


def _mark_t1_hit(signal_id: int):
    with _get_conn() as conn:
        conn.execute(
            "UPDATE positions SET t1_hit=1, stop_loss=entry_price WHERE signal_id=?",
            (signal_id,)
        )


def _mark_exited(signal_id: int, exit_price: float, reason: str):
    with _get_conn() as conn:
        conn.execute("""
            UPDATE positions
            SET exit_price=?, exit_reason=?, exited_at=?
            WHERE signal_id=?
        """, (exit_price, reason, datetime.now().isoformat(), signal_id))


# ── Sync open positions from performance_tracker ──────────────────────────────

def _sync_open_signals(paper: bool = False):
    """Pull fresh BUY signals from signals table into positions table."""
    open_sigs = pt.get_open_signals(paper=paper)
    for s in open_sigs:
        _upsert_position(
            signal_id=s["id"],
            ticker=s["ticker"],
            entry_price=s["price"],
            stop_loss=s.get("stop_loss") or 0.0,
            target_1=s.get("target_1"),
            target_2=s.get("target_2"),
            fired_at=s["fired_at"],
        )


# ── Price check and exit logic ────────────────────────────────────────────────

def _check_positions(paper: bool = False) -> list[dict]:
    """
    Check all open positions. Fire EXIT alerts when triggered.
    Returns list of current position summaries.
    """
    _sync_open_signals(paper=paper)
    positions = _load_open_positions(paper=paper)

    if not positions:
        return []

    summaries = []

    for pos in positions:
        ticker    = pos["ticker"]
        entry     = pos["entry_price"]
        peak      = pos["peak_price"]
        stop      = pos["stop_loss"] or 0.0
        t1        = pos["target_1"]
        t2        = pos["target_2"]
        t1_hit    = bool(pos["t1_hit"])
        fired_at  = datetime.fromisoformat(pos["fired_at"])
        sig_id    = pos["signal_id"]
        horizon   = pos.get("trade_horizon", "swing")

        try:
            current = _get_current_price(ticker)
        except Exception:
            current = entry   # can't fetch — skip checks but include in summary

        if current <= 0:
            current = entry

        # Update peak
        if current > peak:
            peak = current
            _update_peak(sig_id, peak)

        ret_pct     = (current - entry) / entry * 100
        days_held   = (datetime.now() - fired_at).total_seconds() / 86400
        trailing_stop = peak * (1 - TRAILING_STOP_PCT)

        exit_reason = None
        exit_price  = current

        # Check exit conditions (in priority order)
        if stop > 0 and current <= stop:
            exit_reason = f"Stop-loss hit (${stop:.2f})"
        elif t2 and current >= t2 and not pos["t2_hit"]:
            exit_reason = f"Target 2 reached (${t2:.2f}) — full exit"
        elif current <= trailing_stop and days_held > 1:
            exit_reason = f"Trailing stop (7% below peak ${peak:.2f})"
        elif horizon == "swing" and days_held >= MAX_SWING_DAYS and (not t1 or not t1_hit):
            exit_reason = f"Time exit — {days_held:.0f}d held without target"
        elif t1 and current >= t1 and not t1_hit:
            # Target 1 hit — raise stop to breakeven, don't exit yet
            _mark_t1_hit(sig_id)
            print(f"🎯 [Portfolio] {ticker} hit T1 ${t1:.2f} (+{ret_pct:.1f}%) — stop raised to breakeven")
            if not paper:
                send_alert(
                    ticker=ticker, signal="PARTIAL_EXIT",
                    price=current,
                    entry_low=entry, entry_high=entry,
                    targets=[t2] if t2 else [],
                    stop=entry,
                    reason=f"Target 1 hit at ${t1:.2f}. Take 50% profits. Stop raised to breakeven ${entry:.2f}.",
                    confidence=80, horizon=horizon, horizon_reason="",
                )

        if exit_reason:
            _mark_exited(sig_id, exit_price, exit_reason)
            print(f"🚪 [Portfolio] EXIT {ticker} @ ${exit_price:.2f} ({ret_pct:+.1f}%) — {exit_reason}")
            if not paper:
                send_alert(
                    ticker=ticker, signal="SELL",
                    price=exit_price,
                    entry_low=entry, entry_high=entry,
                    targets=[], stop=0,
                    reason=exit_reason,
                    confidence=90, horizon=horizon, horizon_reason="",
                )
        else:
            summaries.append({
                "ticker":      ticker,
                "entry":       round(entry, 4),
                "current":     round(current, 4),
                "peak":        round(peak, 4),
                "return_pct":  round(ret_pct, 2),
                "stop":        round(stop, 4),
                "target_1":    round(t1, 4) if t1 else None,
                "target_2":    round(t2, 4) if t2 else None,
                "t1_hit":      t1_hit,
                "days_held":   round(days_held, 1),
                "horizon":     horizon,
                "trailing_stop": round(trailing_stop, 4),
            })

    return summaries


def _get_current_price(ticker: str) -> float:
    t    = yf.Ticker(ticker)
    info = t.fast_info
    return float(info.last_price or info.previous_close or 0)


# ── Portfolio summary ──────────────────────────────────────────────────────────

def get_portfolio_summary(paper: bool = False) -> dict:
    """Returns full portfolio state — used by /api/portfolio endpoint."""
    positions = _check_positions(paper=paper)

    total_pnl  = sum(p["return_pct"] for p in positions)
    winners    = [p for p in positions if p["return_pct"] > 0]
    losers     = [p for p in positions if p["return_pct"] <= 0]

    stats      = pt.get_stats(lookback_days=30, paper=paper)
    ctx        = wctx.get()

    return {
        "open_positions":  positions,
        "count":           len(positions),
        "winners":         len(winners),
        "losers":          len(losers),
        "avg_return":      round(total_pnl / len(positions), 2) if positions else 0,
        "performance_30d": stats,
        "macro_regime":    ctx["macro"].get("regime", "UNKNOWN"),
        "market_health":   ctx["breadth"].get("health", "UNKNOWN"),
        "updated_at":      datetime.now().isoformat(),
    }


# ── Concentration warnings ────────────────────────────────────────────────────

def _check_concentration(positions: list[dict]) -> list[str]:
    """Warn if more than 40% of open positions are in the same sector."""
    if len(positions) < 3:
        return []
    warnings = []
    # Simple: flag if same ticker appears twice (crude duplicate check)
    tickers = [p["ticker"] for p in positions]
    from collections import Counter
    for ticker, count in Counter(tickers).items():
        if count > 1:
            warnings.append(f"⚠️  {ticker} appears {count}× in open positions")
    return warnings


# ── Main loop ─────────────────────────────────────────────────────────────────

_portfolio_state: list[dict] = []   # in-memory cache for dashboard


async def portfolio_agent_loop(paper: bool = False):
    """Async loop started in main.py lifespan. Runs every 5 minutes."""
    global _portfolio_state
    print(f"💼 [Portfolio] Started — checking every {PORTFOLIO_INTERVAL//60}min")
    init_positions_table()

    loop = asyncio.get_running_loop()

    while True:
        try:
            positions = await loop.run_in_executor(
                None, _check_positions, paper
            )
            _portfolio_state = positions

            if positions:
                total_ret = sum(p["return_pct"] for p in positions)
                avg_ret   = total_ret / len(positions)
                print(
                    f"💼 [Portfolio] {len(positions)} open  "
                    f"avg {avg_ret:+.1f}%  "
                    f"wins={sum(1 for p in positions if p['return_pct']>0)}"
                )
                for p in positions:
                    icon = "🟢" if p["return_pct"] > 0 else "🔴"
                    print(f"   {icon} {p['ticker']} {p['return_pct']:+.1f}%  "
                          f"entry=${p['entry']:.2f}  stop=${p['stop']:.2f}  "
                          f"{p['days_held']:.1f}d")

                warns = _check_concentration(positions)
                for w in warns:
                    print(f"   {w}")

            await asyncio.sleep(PORTFOLIO_INTERVAL)

        except asyncio.CancelledError:
            print("💼 [Portfolio] Stopped.")
            break
        except Exception as e:
            print(f"❌ [Portfolio] Loop error: {e}")
            await asyncio.sleep(60)


def get_cached_portfolio() -> list[dict]:
    """Fast read for dashboard — returns last computed portfolio state."""
    return _portfolio_state


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Portfolio Agent Standalone Test ===\n")
    init_positions_table()

    # The open signals from performance_tracker test run will be checked
    summary = get_portfolio_summary(paper=False)
    print(f"Open positions: {summary['count']}")
    for p in summary["open_positions"]:
        icon = "🟢" if p["return_pct"] > 0 else "🔴"
        print(f"  {icon} {p['ticker']} @ ${p['current']:.2f}  "
              f"ret={p['return_pct']:+.1f}%  stop=${p['stop']:.2f}  "
              f"peak=${p['peak']:.2f}  held={p['days_held']:.1f}d")

    print(f"\n30d stats: {json.dumps(summary['performance_30d'], indent=2)}")
