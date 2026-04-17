"""
decision_agent.py — LangGraph node: calls analyzer.py with full context.
Gate: only flags should_alert=True when confidence >= 65.

Threshold overrides (applied in order of priority):
  CONFIDENCE_THRESHOLD      65   — default
  NEWS_CONFIDENCE_THRESHOLD 55   — ticker was news/spike/EDGAR triggered
  OPTIONS_FLOW_THRESHOLD    58   — extreme call dominance (C/P ≥ 10×) + RVOL ≥ 1.5
"""

from langsmith import traceable
from analyzer import analyze_market
from intelligence_hub import hub

CONFIDENCE_THRESHOLD      = 60
NEWS_CONFIDENCE_THRESHOLD = 52
OPTIONS_FLOW_THRESHOLD    = 55   # extreme call dominance lowers the bar


def _options_flow_override(ticker: str, vol_ratio: float) -> bool:
    """
    Returns True if this ticker has extreme bullish options flow (C/P ≥ 10×)
    combined with at least moderate volume (≥ 1.5×), making it worth alerting
    at a lower confidence threshold.
    """
    if vol_ratio < 1.5:
        return False
    try:
        import world_context as wctx
        unusual = wctx.get()["social"].get("unusual_opts", [])
        for uo in unusual:
            if (uo.get("ticker") == ticker
                    and uo.get("bias") == "BULLISH"
                    and uo.get("call_put_ratio", 0) >= 10.0):
                return True
    except Exception:
        pass
    return False


def _get_threshold(state: dict) -> int:
    """Returns confidence threshold, adjusted by options flow and reflection learnings."""
    news_triggered = state.get("news_triggered", False)
    base = NEWS_CONFIDENCE_THRESHOLD if news_triggered else CONFIDENCE_THRESHOLD

    # Options flow override — extreme call dominance signals institutional conviction
    ticker    = state.get("ticker", "")
    vol_ratio = state.get("volume_spike_ratio", 0.0)
    if _options_flow_override(ticker, vol_ratio):
        prior = base
        base  = min(base, OPTIONS_FLOW_THRESHOLD)
        if base < prior:
            print(f"   🎯 [DecisionAgent] Options flow override for {ticker}: "
                  f"threshold {prior} → {base}")

    try:
        import world_context as wctx
        adj = wctx.get()["macro"].get("confidence_adj", 0)
    except Exception:
        adj = 0

    return max(45, min(80, base + int(adj)))


@traceable(name="decision_agent", tags=["claude", "signal"])
def decision_node(state: dict) -> dict:
    ticker = state["ticker"]
    price  = state.get("current_price", 0.0)

    # ── Alert deduplication ──────────────────────────────────────────────────
    news_triggered = state.get("news_triggered", False)
    major_catalyst = state.get("major_catalyst", False)
    if hub.was_alerted_today(ticker) and not (news_triggered or major_catalyst):
        print(f"⏭️  [DecisionAgent] {ticker} already alerted today — HOLD (dedup)")
        threshold = _get_threshold(state)
        return {
            **state,
            "signal":           "HOLD",
            "confidence":       0,
            "model_signal":     "DEDUP",
            "model_confidence": 0,
            "threshold_used":   threshold,
            "decision_delta":   "dedup: already alerted today",
            "setup_type":       state.get("setup_type", "general"),
            "entry_zone":       f"${price:.4f}",
            "entry_low":        price,
            "entry_high":       price,
            "targets":          [],
            "stop_loss":        round(price * 0.95, 4),
            "stop_pct":         -5.0,
            "reasoning":        f"{ticker} already alerted today — cooling off.",
            "action_plan":      "",
            "rr_ratio":         0.0,
            "trade_horizon":    "swing",
            "horizon_reasoning":"",
            "main_risk":        "",
            "top_3_signals":    [],
            "should_alert":     False,
            "price":            price,
        }

    # ── Portfolio context ────────────────────────────────────────────────────
    paper_mode = state.get("paper_trading", False)
    portfolio  = hub.get_portfolio_context(ticker, paper=paper_mode)
    if portfolio.get("already_open"):
        print(f"⚠️  [DecisionAgent] {ticker} already in portfolio — will cap confidence")

    print(f"🧠 [DecisionAgent] Analyzing {ticker} @ ${price:.4f}...")

    # Capture deterministic score before LLM runs
    score_bd     = state.get("score_breakdown", {})
    det_score    = score_bd.get("final_score") or score_bd.get("raw_score") or 0

    try:
        result       = analyze_market(state)
        model_signal = result["signal"]
        confidence   = result["confidence"]
        threshold    = _get_threshold(state)

        # ── Regime confidence cap ─────────────────────────────────────────────
        thresholds = hub.get_regime_thresholds()
        conf_cap   = thresholds.get("confidence_cap", 100)
        # Extra cap if ticker already in portfolio (avoid doubling down)
        if portfolio.get("already_open"):
            conf_cap = min(conf_cap, 65)
        if confidence > conf_cap:
            print(f"⚠️  [DecisionAgent] Confidence capped {confidence} → {conf_cap} "
                  f"(regime={thresholds.get('regime','?')}"
                  f"{', already_open' if portfolio.get('already_open') else ''})")
            confidence = conf_cap

        # Build decision_delta: human-readable explanation of model vs deterministic
        if model_signal in ("BUY", "SELL") and confidence < threshold:
            decision_delta = (
                f"model={model_signal}@{confidence} BLOCKED by threshold {threshold} "
                f"(det_score={det_score})"
            )
            signal       = "HOLD"
            result["signal"] = "HOLD"
            should_alert = False
            print(
                f"⚠️  [DecisionAgent] Confidence {confidence}/100 < threshold {threshold} "
                f"— overriding {model_signal} → HOLD  (det_score={det_score})"
            )
        elif model_signal in ("BUY", "SELL") and det_score < 35:
            decision_delta = (
                f"model={model_signal}@{confidence} OVERRIDES low det_score={det_score} "
                f"(threshold={threshold})"
            )
            signal       = model_signal
            should_alert = True
            emoji = "🟢" if signal == "BUY" else "🔴"
            print(
                f"{emoji} [DecisionAgent] {signal} conf={confidence}/100  "
                f"⚠️  model overrides low det_score={det_score} (threshold={threshold})"
            )
        else:
            decision_delta = (
                f"model={model_signal}@{confidence} det_score={det_score} threshold={threshold}"
            )
            signal       = model_signal
            should_alert = signal in ("BUY", "SELL")
            emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
            print(
                f"{emoji} [DecisionAgent] {signal}  "
                f"confidence={confidence}/100  det_score={det_score}  "
                f"alert={'YES' if should_alert else 'NO'}"
            )

        import performance_tracker as pt
        audit_state = {
            **state,
            "model_signal":     model_signal,
            "model_confidence": confidence,
            "threshold_used":   threshold,
            "decision_delta":   decision_delta,
            "signal":           signal,
        }
        pt.record_decision_audit(audit_state)

        # entry_low/entry_high/stop_pct now come from analyzer directly
        _entry_low  = result.get("entry_low", price)
        _entry_high = result.get("entry_high", price)
        _stop_pct   = result.get("stop_pct", round((result["stop_loss"] - price) / price * 100, 1) if price > 0 else 0.0)

        return {
            **audit_state,
            "confidence":       confidence,
            "setup_type":       state.get("setup_type", "general"),
            "entry_zone":       result["entry_zone"],
            "entry_low":        _entry_low,
            "entry_high":       _entry_high,
            "targets":          result["targets"],
            "stop_loss":        result["stop_loss"],
            "stop_pct":         _stop_pct,
            "reasoning":        result["reasoning"],
            "action_plan":      result.get("action_plan", ""),
            "rr_ratio":         result.get("rr_ratio", 0.0),
            "trade_horizon":    result.get("trade_horizon", state.get("trade_horizon", "swing")),
            "horizon_reasoning": result.get("horizon_reasoning", state.get("horizon_reasoning", "")),
            "main_risk":        result.get("main_risk", ""),
            "top_3_signals":    result.get("top_3_signals", []),
            "should_alert":     should_alert,
            # Convenience aliases
            "price":            price,
        }

    except Exception as e:
        print(f"❌ [DecisionAgent] Error: {e}")
        return {
            **state,
            "signal":           "HOLD",
            "confidence":       0,
            "model_signal":     "ERROR",
            "model_confidence": 0,
            "threshold_used":   _get_threshold(state),
            "decision_delta":   f"pipeline_error: {e}",
            "setup_type":       state.get("setup_type", "general"),
            "entry_zone":       f"${price:.4f}",
            "targets":          [],
            "stop_loss":        round(price * 0.95, 4),
            "reasoning":        f"Decision agent failed: {e}",
            "action_plan":      "",
            "rr_ratio":         0.0,
            "should_alert":     False,
            # Convenience aliases
            "price":            price,
            "entry_low":        price,
            "entry_high":       price,
            "stop_pct":         -5.0,
        }

run_decision_agent = decision_node
