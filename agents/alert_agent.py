"""
alert_agent.py — LangGraph node: fires SMS via Twilio + push via Pushover using alerts.py.
Respects paper_trading flag (logs only, no real alerts).
Only fires when should_alert is True.
"""

from datetime import date
from alerts import send_alert
import performance_tracker as pt

# Idempotency store: (ticker, signal, date_str) → True
# Prevents duplicate alerts from multiple watcher paths firing same signal on same day.
_fired_today: dict[tuple, bool] = {}
_fired_date: date | None = None


def _is_duplicate(ticker: str, signal: str) -> bool:
    """Return True if this (ticker, signal) already fired today."""
    global _fired_today, _fired_date
    today = date.today()
    if _fired_date != today:          # new day — reset
        _fired_today.clear()
        _fired_date = today
    return _fired_today.get((ticker, signal), False)


def _mark_fired(ticker: str, signal: str):
    global _fired_date
    _fired_date = date.today()   # anchor date so _is_duplicate doesn't clear on same call
    _fired_today[(ticker, signal)] = True


def alert_node(state: dict) -> dict:
    signal        = state.get("signal", "HOLD")
    confidence    = state.get("confidence", 0)
    should_alert  = state.get("should_alert", False)
    paper_trading = state.get("paper_trading", False)
    ticker        = state["ticker"]
    price         = state.get("current_price", 0.0)

    # Suppress duplicate BUY — position already open
    if state.get("already_alerted") and signal == "BUY":
        print(f"🔕 [AlertAgent] {ticker} — duplicate BUY suppressed (already in active position)")
        return {**state, "alert_sent": False}

    # Idempotency gate — same signal already fired today from any watcher path
    if signal in ("BUY", "SELL") and _is_duplicate(ticker, signal):
        print(f"🔕 [AlertAgent] {ticker} {signal} — idempotency gate (already fired today)")
        return {**state, "alert_sent": False}

    if not should_alert:
        if signal == "WATCH":
            exec_reason = state.get("execution_reason", "execution gate blocked")
            print(f"👁️  [AlertAgent] {ticker} WATCH — {exec_reason}")
        else:
            emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
            print(f"{emoji} [AlertAgent] {signal} conf={confidence}/100 — no alert needed")
        return {**state, "alert_sent": False}

    if paper_trading:
        print(
            f"📋 [AlertAgent] PAPER MODE — {signal} on {ticker} @ ${price:.4f}  "
            f"conf={confidence}/100  (alerts suppressed)"
        )
        return {**state, "alert_sent": False}

    try:
        emoji = "🟢" if signal == "BUY" else "🔴"
        print(
            f"{emoji} [AlertAgent] Firing {signal} alert  "
            f"{ticker} @ ${price:.4f}  conf={confidence}/100..."
        )

        # Parse entry zone string ("$1.75 - $1.85") into low/high floats
        entry_zone = state.get("entry_zone", "")
        try:
            parts = entry_zone.replace("$", "").split("-")
            entry_low  = float(parts[0].strip())
            entry_high = float(parts[1].strip())
        except Exception:
            entry_low = entry_high = price

        targets          = state.get("targets", [])
        stop_loss        = state.get("stop_loss", 0.0)
        reasoning        = state.get("reasoning", "")[:200]
        trade_horizon    = state.get("trade_horizon", "swing")
        horizon_reasoning = state.get("horizon_reasoning", "")

        sent = send_alert(
            ticker=ticker,
            signal=signal,
            price=price,
            entry_low=entry_low,
            entry_high=entry_high,
            targets=targets,
            stop=stop_loss,
            reason=reasoning,
            confidence=int(confidence),
            horizon=trade_horizon,
            horizon_reason=horizon_reasoning,
        )
        if sent:
            print("✅ [AlertAgent] Delivered via WhatsApp + Push")
            _mark_fired(ticker, signal)   # idempotency
            pt.record_signal(state)       # log to performance tracker
        else:
            print("⚠️  [AlertAgent] Delivery failed (check Twilio/Pushover config)")
        return {**state, "alert_sent": sent}

    except Exception as e:
        print(f"❌ [AlertAgent] Error: {e}")
        return {**state, "alert_sent": False}
