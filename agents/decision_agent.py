"""
decision_agent.py — LangGraph node: calls analyzer.py with full context.
Gate: only flags should_alert=True when confidence >= 65.
"""

from langsmith import traceable
from analyzer import analyze_market

CONFIDENCE_THRESHOLD      = 65
NEWS_CONFIDENCE_THRESHOLD = 55   # lower gate when a news/spike/EDGAR event is the trigger


def _get_threshold(news_triggered: bool) -> int:
    """Returns confidence threshold, adjusted by reflection agent learnings."""
    base = NEWS_CONFIDENCE_THRESHOLD if news_triggered else CONFIDENCE_THRESHOLD
    try:
        import world_context as wctx
        adj = wctx.get()["macro"].get("confidence_adj", 0)
        return max(45, min(80, base + int(adj)))
    except Exception:
        return base


@traceable(name="decision_agent", tags=["claude", "signal"])
def decision_node(state: dict) -> dict:
    ticker = state["ticker"]
    price  = state.get("current_price", 0.0)

    print(f"🧠 [DecisionAgent] Analyzing {ticker} @ ${price:.4f}...")

    try:
        result     = analyze_market(state)
        signal     = result["signal"]
        confidence = result["confidence"]

        threshold = _get_threshold(state.get("news_triggered", False))

        if confidence < threshold:
            print(
                f"⚠️  [DecisionAgent] Confidence {confidence}/100 < threshold {threshold} "
                f"— overriding {signal} → HOLD"
            )
            signal         = "HOLD"
            result["signal"] = "HOLD"
            should_alert   = False
        else:
            should_alert = signal in ("BUY", "SELL")
            emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
            print(
                f"{emoji} [DecisionAgent] {signal}  "
                f"confidence={confidence}/100  "
                f"alert={'YES' if should_alert else 'NO'}"
            )

        return {
            **state,
            "signal":       signal,
            "confidence":   confidence,
            "setup_type":   state.get("setup_type", "general"),
            "entry_zone":   result["entry_zone"],
            "targets":      result["targets"],
            "stop_loss":    result["stop_loss"],
            "reasoning":    result["reasoning"],
            "action_plan":  result.get("action_plan", ""),
            "rr_ratio":     result.get("rr_ratio", 0.0),
            "should_alert": should_alert,
        }

    except Exception as e:
        print(f"❌ [DecisionAgent] Error: {e}")
        return {
            **state,
            "signal":       "HOLD",
            "confidence":   0,
            "setup_type":   state.get("setup_type", "general"),
            "entry_zone":   f"${price:.4f}",
            "targets":      [],
            "stop_loss":    round(price * 0.95, 4),
            "reasoning":    f"Decision agent failed: {e}",
            "action_plan":  "",
            "rr_ratio":     0.0,
            "should_alert": False,
        }
