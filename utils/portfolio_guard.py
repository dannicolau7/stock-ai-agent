"""
utils/portfolio_guard.py — Hard portfolio limits that override any signal.

Guards (evaluated in order, cheapest first):
  1. weekly_drawdown  — 10% weekly loss → 3-day cooldown (persisted)
  2. daily_loss       — 5% daily loss → halt today (resets next trading day)
  3. open_positions   — ≥5 open BUYs → reject new BUYs
  4. sector_exposure  — adding this ticker would push sector > 30% → reject
  5. correlation      — new ticker > 80% correlated with an open position → reject

Position sizing (not a guard — returns suggested_shares for informational use):
  position_size = (PORTFOLIO_SIZE * MAX_RISK_PCT) / (entry - stop)

Public API
----------
  check(ticker, signal, entry_price, stop_loss, sector) -> GuardResult
  force_resume()  — clear all halts/cooldowns
  get_status()    — dict for dashboard
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MAX_OPEN_POSITIONS   = 5
MAX_RISK_PCT         = 0.02    # 2% of account per trade
MAX_SECTOR_PCT       = 0.30    # 30% max in one sector
DAILY_LOSS_LIMIT     = 0.05    # 5% daily loss → halt today
WEEKLY_DRAWDOWN_HALT = 0.10    # 10% weekly drawdown → 3-day cooldown
CORR_THRESHOLD       = 0.80    # 80% correlation → reject
CORR_LOOKBACK_DAYS   = 30
CORR_CACHE_TTL_S     = 3600    # 1-hour correlation cache

_STATE_PATH = Path(__file__).parent.parent / "data" / "portfolio_guard_state.json"
_DB_PATH    = Path(__file__).parent.parent / "data" / "performance.db"

try:
    from config import PORTFOLIO_SIZE
except Exception:
    PORTFOLIO_SIZE = 25_000.0   # fallback


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GuardResult:
    blocked:            bool
    reason:             str
    guard_name:         str  = ""
    suggested_shares:   int  = 0
    risk_usd:           float = 0.0
    position_size_usd:  float = 0.0


# ── Persistent state ──────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text())
    except Exception:
        pass
    return {"weekly_cooldown_until": None, "daily_halt_date": None}


def _save_state(s: dict) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(s))
    except Exception as e:
        print(f"⚠️  [Guard] State save failed: {e}")


# ── DB helpers ────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def _get_open_positions() -> list[dict]:
    """BUY signals from last 14 days without a completed 7d outcome."""
    try:
        since = (datetime.now() - timedelta(days=14)).isoformat()
        with _conn() as conn:
            rows = conn.execute("""
                SELECT s.ticker, s.price, s.stop_loss, s.sector, s.fired_at
                FROM signals s
                WHERE s.signal = 'BUY'
                  AND s.fired_at >= ?
                  AND s.paper = 0
                  AND s.id NOT IN (
                      SELECT signal_id FROM outcomes
                      WHERE checkpoint='7d' AND win IS NOT NULL
                  )
                ORDER BY s.fired_at DESC
            """, (since,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _get_pnl_since(since: date) -> float:
    """
    Estimate account P&L as a fraction of PORTFOLIO_SIZE since `since`.

    Uses completed outcome rows (checkpoint='1d' or '3d' as proxy for closed).
    Position weight ≈ MAX_RISK_PCT / |risk_pct_stop| capped at 25%.
    """
    try:
        since_iso = datetime.combine(since, datetime.min.time()).isoformat()
        with _conn() as conn:
            rows = conn.execute("""
                SELECT s.price, s.stop_loss, o.return_pct
                FROM outcomes o
                JOIN signals s ON s.id = o.signal_id
                WHERE o.checkpoint = '1d'
                  AND o.win IS NOT NULL
                  AND o.checked_at >= ?
                  AND s.paper = 0
            """, (since_iso,)).fetchall()

        pnl_pct = 0.0
        for r in rows:
            entry     = float(r["price"] or 0)
            stop      = float(r["stop_loss"] or 0)
            ret       = float(r["return_pct"] or 0) / 100.0
            if entry > 0 and stop > 0 and entry > stop:
                risk_pct  = abs((stop - entry) / entry)
                weight    = min(MAX_RISK_PCT / risk_pct, 0.25) if risk_pct > 0 else MAX_RISK_PCT
                pnl_pct  += ret * weight
        return pnl_pct
    except Exception:
        return 0.0


# ── Correlation cache ─────────────────────────────────────────────────────────

_corr_cache: dict[str, tuple[float, float]] = {}   # key → (corr, ts)


def _max_correlation(ticker: str, open_tickers: list[str]) -> float:
    """
    Return the max Pearson correlation of `ticker` vs each open position
    over the last CORR_LOOKBACK_DAYS days of daily closes.
    Returns 0.0 on any error (fail open — don't block due to data issues).
    """
    if not open_tickers:
        return 0.0

    cache_key = f"{ticker}:{'|'.join(sorted(open_tickers))}"
    cached = _corr_cache.get(cache_key)
    if cached and (time.time() - cached[1]) < CORR_CACHE_TTL_S:
        return cached[0]

    try:
        import yfinance as yf
        all_tickers = [ticker] + open_tickers
        period = f"{CORR_LOOKBACK_DAYS + 5}d"
        raw = yf.download(all_tickers, period=period, progress=False, auto_adjust=True)
        closes = raw["Close"] if isinstance(raw.columns, object) and "Close" in raw else raw
        rets   = closes.pct_change().dropna()
        if ticker not in rets.columns:
            return 0.0
        target = rets[ticker]
        max_corr = 0.0
        for ot in open_tickers:
            if ot in rets.columns:
                c = float(target.corr(rets[ot]))
                if c > max_corr:
                    max_corr = c
        _corr_cache[cache_key] = (max_corr, time.time())
        return max_corr
    except Exception as e:
        print(f"⚠️  [Guard] Correlation check failed: {e}")
        return 0.0


# ── Position sizing ───────────────────────────────────────────────────────────

def _size(entry: float, stop: float) -> tuple[int, float, float]:
    """Returns (suggested_shares, risk_usd, position_size_usd)."""
    if entry <= 0 or stop <= 0 or entry <= stop:
        return 0, 0.0, 0.0
    risk_per_share = entry - stop
    risk_usd       = PORTFOLIO_SIZE * MAX_RISK_PCT
    shares         = int(risk_usd / risk_per_share)
    pos_usd        = shares * entry
    return shares, round(risk_usd, 2), round(pos_usd, 2)


# ── Public API ────────────────────────────────────────────────────────────────

def check(
    ticker:      str,
    signal:      str,
    entry_price: float = 0.0,
    stop_loss:   float = 0.0,
    sector:      str   = "",
) -> GuardResult:
    """
    Evaluate all hard portfolio limits.
    Returns GuardResult(blocked=True, ...) if any limit is breached.
    For non-BUY signals (SELL, HOLD) only the halt/cooldown guards apply.
    """
    today   = date.today()
    state   = _load_state()
    shares, risk_usd, pos_usd = _size(entry_price, stop_loss)

    # ── 1. Weekly drawdown cooldown (persistent across restarts) ──────────────
    cooldown_until = state.get("weekly_cooldown_until")
    if cooldown_until:
        try:
            cool_date = date.fromisoformat(cooldown_until)
            if today < cool_date:
                return GuardResult(
                    blocked=True,
                    reason=f"Weekly drawdown cooldown active until {cooldown_until}",
                    guard_name="weekly_drawdown",
                    suggested_shares=shares, risk_usd=risk_usd, position_size_usd=pos_usd,
                )
        except ValueError:
            pass   # corrupt date string — ignore

    # ── 2. Daily loss limit (resets each day) ─────────────────────────────────
    daily_halt_date = state.get("daily_halt_date")
    if daily_halt_date == str(today):
        return GuardResult(
            blocked=True,
            reason=f"Daily loss limit ({DAILY_LOSS_LIMIT:.0%}) hit — trading halted today",
            guard_name="daily_loss",
            suggested_shares=shares, risk_usd=risk_usd, position_size_usd=pos_usd,
        )

    # Check actual daily P&L
    daily_pnl = _get_pnl_since(today)
    if daily_pnl <= -DAILY_LOSS_LIMIT:
        state["daily_halt_date"] = str(today)
        _save_state(state)
        _send_halt_alert(f"🚨 DAILY LOSS LIMIT HIT — P&L {daily_pnl:.1%} today. Trading halted.")
        return GuardResult(
            blocked=True,
            reason=f"Daily loss limit hit ({daily_pnl:.1%}). Trading halted for today.",
            guard_name="daily_loss",
            suggested_shares=shares, risk_usd=risk_usd, position_size_usd=pos_usd,
        )

    # Check weekly P&L (Mon–Sun window)
    week_start = today - timedelta(days=today.weekday())
    weekly_pnl = _get_pnl_since(week_start)
    if weekly_pnl <= -WEEKLY_DRAWDOWN_HALT:
        resume_date = today + timedelta(days=3)
        state["weekly_cooldown_until"] = str(resume_date)
        _save_state(state)
        _send_halt_alert(
            f"🚨 WEEKLY DRAWDOWN HALT — {weekly_pnl:.1%} this week. "
            f"3-day cooldown until {resume_date}. No new signals."
        )
        return GuardResult(
            blocked=True,
            reason=f"Weekly drawdown {weekly_pnl:.1%} ≥ {WEEKLY_DRAWDOWN_HALT:.0%}. 3-day cooldown.",
            guard_name="weekly_drawdown",
            suggested_shares=shares, risk_usd=risk_usd, position_size_usd=pos_usd,
        )

    # ── BUY-only guards ───────────────────────────────────────────────────────
    if signal != "BUY":
        return GuardResult(
            blocked=False, reason="",
            suggested_shares=shares, risk_usd=risk_usd, position_size_usd=pos_usd,
        )

    open_pos = _get_open_positions()

    # ── 3. Open position limit ────────────────────────────────────────────────
    if len(open_pos) >= MAX_OPEN_POSITIONS:
        # Find worst performer to suggest closing
        worst = _find_worst_performer(open_pos)
        reason = f"Max {MAX_OPEN_POSITIONS} open positions held."
        if worst:
            reason += f" Consider closing {worst} first."
        return GuardResult(
            blocked=True, reason=reason, guard_name="open_positions",
            suggested_shares=shares, risk_usd=risk_usd, position_size_usd=pos_usd,
        )

    # ── 4. Sector exposure ────────────────────────────────────────────────────
    if sector:
        sector_count = sum(1 for p in open_pos if (p.get("sector") or "").upper() == sector.upper())
        sector_pct   = (sector_count + 1) / MAX_OPEN_POSITIONS   # +1 for new trade
        if sector_pct > MAX_SECTOR_PCT:
            return GuardResult(
                blocked=True,
                reason=f"Sector {sector!r} exposure would reach {sector_pct:.0%} (limit {MAX_SECTOR_PCT:.0%})",
                guard_name="sector_exposure",
                suggested_shares=shares, risk_usd=risk_usd, position_size_usd=pos_usd,
            )

    # ── 5. Correlation check (slowest — runs last) ────────────────────────────
    open_tickers = [p["ticker"] for p in open_pos if p["ticker"] != ticker]
    if open_tickers:
        max_corr = _max_correlation(ticker, open_tickers)
        if max_corr > CORR_THRESHOLD:
            corr_ticker = _find_most_correlated(ticker, open_tickers)
            return GuardResult(
                blocked=True,
                reason=(
                    f"{ticker} is {max_corr:.0%} correlated with {corr_ticker or 'an open position'} "
                    f"(limit {CORR_THRESHOLD:.0%})"
                ),
                guard_name="correlation",
                suggested_shares=shares, risk_usd=risk_usd, position_size_usd=pos_usd,
            )

    # ── All clear ─────────────────────────────────────────────────────────────
    return GuardResult(
        blocked=False, reason="",
        suggested_shares=shares, risk_usd=risk_usd, position_size_usd=pos_usd,
    )


def force_resume() -> None:
    """Clear all halts and cooldowns (manual override)."""
    _save_state({"weekly_cooldown_until": None, "daily_halt_date": None})
    _corr_cache.clear()
    print("✅ [Guard] All halts cleared — trading resumed")


def get_status() -> dict:
    """Dashboard-friendly status snapshot."""
    today      = date.today()
    state      = _load_state()
    open_pos   = _get_open_positions()
    daily_pnl  = _get_pnl_since(today)
    week_start = today - timedelta(days=today.weekday())
    weekly_pnl = _get_pnl_since(week_start)

    cooldown_until = state.get("weekly_cooldown_until")
    weekly_halt    = bool(cooldown_until and date.fromisoformat(cooldown_until) > today)
    daily_halt     = state.get("daily_halt_date") == str(today)

    return {
        "trading_halted":        weekly_halt or daily_halt,
        "daily_halt":            daily_halt,
        "weekly_halt":           weekly_halt,
        "weekly_cooldown_until": cooldown_until,
        "open_positions":        len(open_pos),
        "open_tickers":          [p["ticker"] for p in open_pos],
        "daily_pnl_pct":         round(daily_pnl * 100, 2),
        "weekly_pnl_pct":        round(weekly_pnl * 100, 2),
        "limits": {
            "max_open":           MAX_OPEN_POSITIONS,
            "max_risk_pct":       MAX_RISK_PCT,
            "max_sector_pct":     MAX_SECTOR_PCT,
            "daily_loss_limit":   DAILY_LOSS_LIMIT,
            "weekly_drawdown":    WEEKLY_DRAWDOWN_HALT,
            "corr_threshold":     CORR_THRESHOLD,
        },
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _send_halt_alert(msg: str) -> None:
    """Send WhatsApp alert when a circuit-breaker fires."""
    try:
        from alerts import send_whatsapp
        send_whatsapp(msg)
        print(f"📱 [Guard] Alert sent: {msg[:80]}")
    except Exception as e:
        print(f"⚠️  [Guard] WhatsApp alert failed: {e}")


def _find_worst_performer(open_pos: list[dict]) -> str:
    """
    Return ticker with worst current return vs entry price.
    Falls back to oldest position if yfinance unavailable.
    """
    try:
        import yfinance as yf
        tickers = [p["ticker"] for p in open_pos]
        if not tickers:
            return ""
        raw = yf.download(tickers, period="1d", progress=False, auto_adjust=True)
        closes = raw["Close"] if "Close" in raw else raw
        if hasattr(closes, "iloc"):
            last = closes.iloc[-1]
        else:
            return tickers[-1]

        worst_ret  = float("inf")
        worst_tick = tickers[-1]
        for p in open_pos:
            t = p["ticker"]
            if t in last.index:
                entry = float(p.get("price") or 0)
                if entry > 0:
                    ret = (float(last[t]) - entry) / entry
                    if ret < worst_ret:
                        worst_ret  = ret
                        worst_tick = t
        return worst_tick
    except Exception:
        # Return oldest position as fallback
        return open_pos[-1]["ticker"] if open_pos else ""


def _find_most_correlated(ticker: str, open_tickers: list[str]) -> str:
    """Return the open ticker most correlated with `ticker`."""
    try:
        import yfinance as yf
        all_t  = [ticker] + open_tickers
        raw    = yf.download(all_t, period=f"{CORR_LOOKBACK_DAYS + 5}d", progress=False, auto_adjust=True)
        closes = raw["Close"] if "Close" in raw else raw
        rets   = closes.pct_change().dropna()
        if ticker not in rets.columns:
            return open_tickers[0] if open_tickers else ""
        target   = rets[ticker]
        best_t   = open_tickers[0]
        best_c   = -1.0
        for ot in open_tickers:
            if ot in rets.columns:
                c = float(target.corr(rets[ot]))
                if c > best_c:
                    best_c = c
                    best_t = ot
        return best_t
    except Exception:
        return open_tickers[0] if open_tickers else ""
