"""
alert_agent.py — LangGraph node: fires SMS via Twilio + push via Pushover using alerts.py.
Respects paper_trading flag (logs only, no real alerts).
Only fires when should_alert is True.

alert_reason_code values:
  duplicate_position  — same ticker already has an active BUY in signal_memory
  duplicate_today     — same (ticker, signal) already fired today (SQLite-persistent)
  threshold_blocked   — should_alert=False (confidence < threshold or risk veto)
  paper_mode          — paper_trading=True, alert suppressed
  delivery_failed     — send_alert returned False
  sent                — alert delivered successfully
"""

from datetime import datetime, timezone
from alerts import send_alert
import performance_tracker as pt

_PRICE_MAX_AGE_S = 90   # re-fetch price if pipeline took longer than this


def _fresh_price(ticker: str, fetched_at_iso: str, current: float) -> float:
    """Return current price, re-fetching via yfinance if the cached value is >90s old."""
    try:
        if fetched_at_iso:
            fetched = datetime.fromisoformat(fetched_at_iso)
            age_s   = (datetime.now(timezone.utc) - fetched).total_seconds()
            if age_s <= _PRICE_MAX_AGE_S:
                return current
            print(f"⏱️  [AlertAgent] Price age {age_s:.0f}s > {_PRICE_MAX_AGE_S}s — re-fetching {ticker}...")
        import yfinance as yf
        price = float(yf.Ticker(ticker).fast_info["last_price"] or current)
        now_s = datetime.now().strftime("%H:%M:%S")
        print(f"   💰 Fresh price: ${price:.4f}  [fetched {now_s}]")
        return price
    except Exception as e:
        print(f"⚠️  [AlertAgent] Price re-fetch failed ({e}) — using cached ${current:.4f}")
        return current


def alert_node(state: dict) -> dict:
    signal        = state.get("signal", "HOLD")
    confidence    = state.get("confidence", 0)
    should_alert  = state.get("should_alert", False)
    paper_trading = state.get("paper_trading", False)
    ticker        = state["ticker"]
    price         = _fresh_price(
        ticker,
        state.get("price_fetched_at", ""),
        state.get("current_price", 0.0),
    )

    # ── Suppress duplicate BUY — position already open ─────────────────────────
    if state.get("already_alerted") and signal == "BUY":
        print(f"🔕 [AlertAgent] {ticker} — duplicate BUY suppressed (already in active position)")
        return {**state, "alert_sent": False, "alert_reason_code": "duplicate_position"}

    # ── Idempotency gate — persistent across restarts (SQLite) ─────────────────
    if signal in ("BUY", "SELL") and pt.is_alert_fired(ticker, signal, paper=paper_trading):
        print(f"🔕 [AlertAgent] {ticker} {signal} — idempotency gate (already fired today)")
        return {**state, "alert_sent": False, "alert_reason_code": "duplicate_today"}

    if not should_alert:
        exec_reason = state.get("execution_reason", "")
        if signal == "WATCH":
            print(f"👁️  [AlertAgent] {ticker} WATCH — {exec_reason}")
        elif signal in ("BUY", "SELL"):
            veto = state.get("risk_veto_reason", "")
            detail = veto or exec_reason or state.get("decision_delta", "threshold blocked")
            emoji = "🟢" if signal == "BUY" else "🔴"
            print(f"{emoji} [AlertAgent] {signal} conf={confidence}/100 blocked — {detail}")
        else:
            print(f"🟡 [AlertAgent] HOLD conf={confidence}/100 — no alert needed")
        return {**state, "alert_sent": False, "alert_reason_code": "threshold_blocked"}

    if paper_trading:
        print(
            f"📋 [AlertAgent] PAPER MODE — {signal} on {ticker} @ ${price:.4f}  "
            f"conf={confidence}/100  (alerts suppressed)"
        )
        return {**state, "alert_sent": False, "alert_reason_code": "paper_mode"}

    try:
        emoji = "🟢" if signal == "BUY" else "🔴"
        print(
            f"{emoji} [AlertAgent] Firing {signal} alert  "
            f"{ticker} @ ${price:.4f}  conf={confidence}/100..."
        )

        # Parse entry zone string ("$1.75 - $1.85") robustly
        entry_zone = state.get("entry_zone", "")
        try:
            import re
            nums = re.findall(r"[\d.]+", entry_zone)
            entry_low  = float(nums[0]) if nums else price
            entry_high = float(nums[1]) if len(nums) > 1 else entry_low
        except Exception:
            entry_low = entry_high = price

        targets           = state.get("targets", [])
        stop_loss         = state.get("stop_loss", 0.0)
        reasoning         = state.get("reasoning", "")[:240]
        trade_horizon     = state.get("trade_horizon", "swing")
        horizon_reasoning = state.get("horizon_reasoning", "")

        # Aggregator / validator fields for rich alert format
        agreement_score   = state.get("agreement_score")      # None → legacy format
        market_regime     = state.get("market_regime", {})
        sector_mom        = state.get("sector_momentum", {})
        regime_str        = market_regime.get("regime", "") if isinstance(market_regime, dict) else ""
        sector_str        = (
            f"{sector_mom.get('change_pct', 0):+.1f}% {sector_mom.get('signal', '')}"
            if isinstance(sector_mom, dict) else ""
        )
        # Catalyst: fresh news summary or EDGAR flag
        raw_news     = state.get("raw_news", [])
        news_summary = state.get("news_summary", "")
        catalyst_str = news_summary[:100] if news_summary else ("EDGAR filing" if state.get("has_edgar_filing") else "")
        # det_score from score_breakdown
        score_bd  = state.get("score_breakdown", {})
        det_score = score_bd.get("final_score", 0) if isinstance(score_bd, dict) else 0

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
            # Aggregator extras
            agreement_score=agreement_score,
            signal_count_bull=state.get("signal_count_bull", 0),
            signal_count_bear=state.get("signal_count_bear", 0),
            top_3_signals=state.get("top_3_signals", []),
            bullish_signals=state.get("bullish_signals", []),
            bearish_signals=state.get("bearish_signals", []),
            consensus=state.get("consensus", ""),
            market_regime_str=regime_str,
            sector_str=sector_str,
            catalyst_str=catalyst_str,
            main_risk=state.get("main_risk", ""),
            det_score=det_score,
        )
        if sent:
            print("✅ [AlertAgent] Delivered via WhatsApp + Push")
            pt.mark_alert_fired(ticker, signal, paper=paper_trading)
            pt.record_signal(state)
            from intelligence_hub import hub
            hub.mark_alerted(ticker, signal)
            print(f"✅ [Hub] {ticker} marked as alerted post-delivery")
            reason_code = "sent"
        else:
            print("⚠️  [AlertAgent] Delivery failed (check Twilio/Pushover config)")
            reason_code = "delivery_failed"

        return {**state, "alert_sent": sent, "alert_reason_code": reason_code}

    except Exception as e:
        print(f"❌ [AlertAgent] Error: {e}")
        return {**state, "alert_sent": False, "alert_reason_code": "delivery_failed"}
