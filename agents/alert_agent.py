"""
alert_agent.py — LangGraph node: fires SMS via Twilio + push via Pushover using alerts.py.
Respects paper_trading flag (logs only, no real alerts).
Only fires when should_alert is True.
"""

from alerts import send_alert


def alert_node(state: dict) -> dict:
    signal        = state.get("signal", "HOLD")
    confidence    = state.get("confidence", 0)
    should_alert  = state.get("should_alert", False)
    paper_trading = state.get("paper_trading", False)
    ticker        = state["ticker"]
    price         = state.get("current_price", 0.0)

    if not should_alert:
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

        targets   = state.get("targets", [])
        target    = targets[0] if targets else price
        stop_loss = state.get("stop_loss", 0.0)
        reasoning = state.get("reasoning", "")[:200]

        sent = send_alert(
            ticker=ticker,
            signal=signal,
            price=price,
            entry_low=entry_low,
            entry_high=entry_high,
            target=target,
            stop=stop_loss,
            reason=reasoning,
            confidence=int(confidence),
        )
        if sent:
            print("✅ [AlertAgent] Delivered via WhatsApp + Push")
        else:
            print("⚠️  [AlertAgent] Delivery failed (check Twilio/Pushover config)")
        return {**state, "alert_sent": sent}

    except Exception as e:
        print(f"❌ [AlertAgent] Error: {e}")
        return {**state, "alert_sent": False}
