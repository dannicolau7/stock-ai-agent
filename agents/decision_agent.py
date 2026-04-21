"""
decision_agent.py — LangGraph node: calls analyzer.py with full context.
Gate: only flags should_alert=True when confidence >= 65.

Threshold overrides (applied in order of priority):
  CONFIDENCE_THRESHOLD      65   — default
  NEWS_CONFIDENCE_THRESHOLD 55   — ticker was news/spike/EDGAR triggered
  OPTIONS_FLOW_THRESHOLD    58   — extreme call dominance (C/P ≥ 10×) + RVOL ≥ 1.5
"""

import csv
import os
from pathlib import Path

from langsmith import traceable
from analyzer import analyze_market
from intelligence_hub import hub
from utils.tracing import annotate_run, get_current_run_id

CONFIDENCE_THRESHOLD      = 65
NEWS_CONFIDENCE_THRESHOLD = 52
OPTIONS_FLOW_THRESHOLD    = 55   # extreme call dominance lowers the bar
SELL_CONFIDENCE_THRESHOLD = 65   # SELL always requires >= 65, regardless of news override
MIN_ALERT_SCORE           = 40   # abs(det_score) must be >= 40 for any alert
MIN_RVOL                  = 1.5  # relative volume floor — below this → HOLD
HIGH_RVOL_EXCEPTION       = 3.0  # if RVOL > this AND news catalyst, allow regardless

_BLOCKED_LOG = Path(__file__).parent.parent / "data" / "blocked_signals.csv"


_BLOCKED_LOG_FIELDS = [
    "ts", "ticker", "date_blocked", "earnings_date",
    "days_until_earnings", "original_signal", "rvol", "reason", "source",
]


def _log_blocked_signal(ticker: str, rvol: float, reason: str) -> None:
    """Append a volume-blocked row to data/blocked_signals.csv."""
    try:
        from datetime import datetime as _dt
        _BLOCKED_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_header = not _BLOCKED_LOG.exists()
        with open(_BLOCKED_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_BLOCKED_LOG_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({
                "ts":                  _dt.now().isoformat(),
                "ticker":              ticker,
                "date_blocked":        _dt.now().date().isoformat(),
                "earnings_date":       "",
                "days_until_earnings": "",
                "original_signal":     "",
                "rvol":                round(rvol, 2),
                "reason":              reason,
                "source":              "decision_agent",
            })
    except Exception as e:
        print(f"⚠️  [DecisionAgent] blocked_signals log failed: {e}")


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
    annotate_run(state)

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
    portfolio  = hub.get_portfolio_context(ticker)
    if portfolio.get("already_open"):
        print(f"⚠️  [DecisionAgent] {ticker} already in portfolio — will cap confidence")

    # ── TAKE_PROFIT check — before LLM, no confidence gate needed ──────────────
    entry_price = portfolio.get("entry_price", 0.0)
    if portfolio.get("already_open") and entry_price > 0 and price >= entry_price * 1.10:
        pnl_pct = round((price - entry_price) / entry_price * 100, 1)
        print(f"💰 [DecisionAgent] TAKE PROFIT {ticker} +{pnl_pct}% from ${entry_price:.2f}")
        return {
            **state,
            "signal":             "TAKE_PROFIT",
            "confidence":         90,
            "model_signal":       "TAKE_PROFIT",
            "model_confidence":   90,
            "threshold_used":     SELL_CONFIDENCE_THRESHOLD,
            "decision_delta":     f"take_profit: +{pnl_pct}% from ${entry_price:.2f}",
            "setup_type":         "take_profit",
            "entry_zone":         f"${price:.4f}",
            "entry_low":          price,
            "entry_high":         price,
            "targets":            [],
            "stop_loss":          round(entry_price, 4),
            "stop_pct":           0.0,
            "reasoning":          f"Take profit triggered: +{pnl_pct}% gain from entry ${entry_price:.2f}",
            "action_plan":        f"Exit at market ${price:.2f}",
            "rr_ratio":           0.0,
            "trade_horizon":      "exit",
            "horizon_reasoning":  "",
            "main_risk":          "",
            "top_3_signals":      [],
            "should_alert":       True,
            "price":              price,
            "take_profit_pct":    pnl_pct,
            "take_profit_entry":  entry_price,
        }

    # ── Volume guardrail — hard block before LLM call ───────────────────────────
    vol_ratio      = state.get("volume_spike_ratio", 0.0)
    news_catalyst  = (state.get("news_triggered", False)
                      or state.get("has_edgar_filing", False)
                      or state.get("major_catalyst", False))
    _rvol_known    = vol_ratio > 0
    _high_vol_news = vol_ratio > HIGH_RVOL_EXCEPTION and news_catalyst
    if _rvol_known and vol_ratio < MIN_RVOL and not _high_vol_news:
        reason = f"⛔ BLOCKED - Low volume {vol_ratio:.1f}x"
        _log_blocked_signal(ticker, vol_ratio, reason)
        print(f"⛔ [DecisionAgent] {ticker} blocked — Low volume {vol_ratio:.1f}x < {MIN_RVOL}x")
        threshold = _get_threshold(state)
        return {
            **state,
            "signal":           "HOLD",
            "confidence":       0,
            "model_signal":     "BLOCKED",
            "model_confidence": 0,
            "threshold_used":   threshold,
            "decision_delta":   reason,
            "setup_type":       state.get("setup_type", "general"),
            "entry_zone":       f"${price:.4f}",
            "entry_low":        price,
            "entry_high":       price,
            "targets":          [],
            "stop_loss":        round(price * 0.95, 4),
            "stop_pct":         -5.0,
            "reasoning":        f"Blocked: relative volume {vol_ratio:.1f}x below {MIN_RVOL}x minimum.",
            "action_plan":      "",
            "rr_ratio":         0.0,
            "trade_horizon":    "swing",
            "horizon_reasoning": "",
            "main_risk":        "low_volume",
            "top_3_signals":    [],
            "should_alert":     False,
            "price":            price,
        }

    # ── Earnings blackout gate ────────────────────────────────────────────────
    # Prefer earnings data already fetched by data_agent (free); fall back to
    # earnings_gate fetch only when not available in state.
    from utils.earnings_gate import check_earnings_blackout, log_earnings_block, BLACKOUT_DAYS
    _ei           = state.get("earnings_info", {})
    _days_to_earn = _ei.get("days_to_earnings", 999) if isinstance(_ei, dict) else 999
    _earn_date    = _ei.get("earnings_date", "") if isinstance(_ei, dict) else ""

    if _days_to_earn == 999:
        _eb = check_earnings_blackout(ticker)
        _days_to_earn = _eb["days_until"]
        _earn_date    = _eb["earnings_date"]
        if _eb["warning"] and not _eb["blocked"]:
            print(f"   ℹ️  [DecisionAgent] {_eb['warning']}")

    if _days_to_earn <= BLACKOUT_DAYS:
        _earn_reason = f"⚠️ {ticker} earnings in {_days_to_earn} days — signal blocked"
        log_earnings_block(
            ticker=ticker,
            original_signal=state.get("signal", "PENDING"),
            days_until=_days_to_earn,
            earnings_date=_earn_date,
            source="decision_agent",
        )
        print(f"📅 [DecisionAgent] {_earn_reason}")
        try:
            from alerts import send_whatsapp
            send_whatsapp(_earn_reason)
        except Exception:
            pass
        threshold = _get_threshold(state)
        return {
            **state,
            "signal":            "HOLD",
            "confidence":        0,
            "model_signal":      "EARNINGS_BLOCKED",
            "model_confidence":  0,
            "threshold_used":    threshold,
            "decision_delta":    _earn_reason,
            "setup_type":        state.get("setup_type", "general"),
            "entry_zone":        f"${price:.4f}",
            "entry_low":         price,
            "entry_high":        price,
            "targets":           [],
            "stop_loss":         round(price * 0.95, 4),
            "stop_pct":          -5.0,
            "reasoning":         _earn_reason,
            "action_plan":       f"Re-evaluate after earnings on {_earn_date}.",
            "rr_ratio":          0.0,
            "trade_horizon":     "swing",
            "horizon_reasoning": "",
            "main_risk":         "earnings_event",
            "top_3_signals":     [],
            "should_alert":      False,
            "price":             price,
        }

    # ── Regime gate — block BUY in BEAR/PANIC markets ────────────────────────
    try:
        from utils.regime_gate import check_buy_allowed, get_regime
        _regime_state = get_regime()
        _buy_ok, _buy_reason = check_buy_allowed(_regime_state)
        if not _buy_ok:
            print(f"📊 [DecisionAgent] {ticker} BUY blocked by regime gate: {_buy_reason}")
            # Still run model but override result to HOLD below
        # Store regime info in state for alert header
        state = {**state, "_regime_state": _regime_state}
    except Exception as _rg_err:
        print(f"⚠️  [DecisionAgent] Regime gate error (skipping): {_rg_err}")
        _buy_ok, _buy_reason, _regime_state = True, "", None

    # ── Portfolio hard limits ─────────────────────────────────────────────────
    try:
        from utils.portfolio_guard import check as _pg_check
        _pg = _pg_check(
            ticker=ticker,
            signal=state.get("signal", "HOLD"),
            entry_price=state.get("current_price", price),
            stop_loss=state.get("stop_loss", 0.0),
            sector=state.get("sector", ""),
        )
        if _pg.blocked:
            print(f"🛡️  [PortfolioGuard] {ticker} blocked — {_pg.reason}")
            threshold = _get_threshold(state)
            return {
                **state,
                "signal":            "HOLD",
                "confidence":        0,
                "model_signal":      "GUARD_BLOCKED",
                "model_confidence":  0,
                "threshold_used":    threshold,
                "decision_delta":    f"portfolio_guard:{_pg.guard_name} — {_pg.reason}",
                "setup_type":        state.get("setup_type", "general"),
                "entry_zone":        f"${price:.4f}",
                "entry_low":         price,
                "entry_high":        price,
                "targets":           [],
                "stop_loss":         round(price * 0.95, 4),
                "stop_pct":          -5.0,
                "reasoning":         _pg.reason,
                "action_plan":       "",
                "rr_ratio":          0.0,
                "trade_horizon":     "swing",
                "horizon_reasoning": "",
                "main_risk":         _pg.guard_name,
                "top_3_signals":     [],
                "should_alert":      False,
                "price":             price,
            }
        if _pg.suggested_shares:
            state = {**state,
                     "suggested_shares":   _pg.suggested_shares,
                     "risk_usd":           _pg.risk_usd,
                     "position_size_usd":  _pg.position_size_usd}
    except Exception as _pg_err:
        print(f"⚠️  [PortfolioGuard] Error (skipping): {_pg_err}")

    # ── Insider signal — SEC EDGAR Form 4 (6-hour cache) ─────────────────────
    _insider_result = {"insider_signal": "NEUTRAL", "insider_score": 0,
                       "key_alerts": [], "_source": "skipped"}
    try:
        from agents import insider_agent as _ia
        _insider_result = _ia.get_signal(ticker)
        _isig   = _insider_result["insider_signal"]
        _iscore = _insider_result["insider_score"]
        if _isig != "NEUTRAL":
            _iemoji = "🟢" if _isig == "BULLISH" else "🔴"
            print(f"{_iemoji} [DecisionAgent] Insider {_isig} for {ticker} (score={_iscore:+d})")
            for _ialert in _insider_result.get("key_alerts", [])[:2]:
                print(f"   {_ialert}")
    except Exception as _ie:
        print(f"⚠️  [DecisionAgent] Insider agent error (skipping): {_ie}")
    state = {**state, "insider_signal": _insider_result}

    print(f"🧠 [DecisionAgent] Analyzing {ticker} @ ${price:.4f}...")

    # Capture deterministic score before LLM runs
    score_bd     = state.get("score_breakdown", {})
    det_score    = score_bd.get("final_score") or score_bd.get("raw_score") or 0

    try:
        result       = analyze_market(state)
        model_signal = result["signal"]
        confidence   = result["confidence"]
        threshold    = _get_threshold(state)

        # ── Log dual-call debate scores ───────────────────────────────────────
        if result.get("bull_score") is not None:
            print(
                f"   🥊 Bull ({result['bull_score']}) vs Bear ({result['bear_score']}) "
                f"→ net {result['net_score']:+d}"
            )

        # ── Regime gate: convert BUY → HOLD in BEAR/PANIC markets ───────────
        if model_signal == "BUY" and not _buy_ok:
            decision_delta = f"BUY BLOCKED by regime gate: {_buy_reason}"
            signal           = "HOLD"
            result["signal"] = "HOLD"
            should_alert     = False
            print(f"📊 [DecisionAgent] BUY → HOLD (regime gate: {_buy_reason})")

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

        # ── Insider signal adjustment ─────────────────────────────────────────
        _ins = state.get("insider_signal", {})
        if isinstance(_ins, dict):
            _ins_score = _ins.get("insider_score", 0)
            if model_signal == "BUY" and _ins_score <= -40:
                decision_delta   = f"BUY BLOCKED by insider cluster sell (score={_ins_score:+d})"
                signal           = "HOLD"
                result["signal"] = "HOLD"
                should_alert     = False
                print(f"📋 [DecisionAgent] BUY → HOLD (insider cluster sell, score={_ins_score:+d})")
            elif model_signal == "BUY" and _ins_score <= -25:
                _prev_conf = confidence
                confidence = max(0, confidence - 10)
                if _prev_conf != confidence:
                    print(f"📋 [DecisionAgent] Insider BEARISH: confidence "
                          f"{_prev_conf} → {confidence} (score={_ins_score:+d})")

        # ── 3-Timeframe agreement gate ────────────────────────────────────────
        _tfa = state.get("timeframe_agreement", {})
        if isinstance(_tfa, dict):
            if model_signal == "BUY" and not _tfa.get("buy_confirmed", True):
                _tfa_str = (
                    f"daily={(_tfa.get('daily') or {}).get('signal','?')} "
                    f"1h={(_tfa.get('hourly') or {}).get('signal','?')} "
                    f"15m={(_tfa.get('intraday') or {}).get('signal','?')}"
                )
                decision_delta   = f"BUY BLOCKED by 3TF disagreement ({_tfa_str})"
                signal           = "HOLD"
                result["signal"] = "HOLD"
                should_alert     = False
                print(f"📊 [DecisionAgent] BUY → HOLD (3TF: {_tfa_str})")
            elif model_signal == "SELL" and not _tfa.get("sell_confirmed", True):
                _tfa_str = (
                    f"daily={(_tfa.get('daily') or {}).get('signal','?')} "
                    f"1h={(_tfa.get('hourly') or {}).get('signal','?')} "
                    f"15m={(_tfa.get('intraday') or {}).get('signal','?')}"
                )
                decision_delta   = f"SELL BLOCKED by 3TF disagreement ({_tfa_str})"
                signal           = "HOLD"
                result["signal"] = "HOLD"
                should_alert     = False
                print(f"📊 [DecisionAgent] SELL → HOLD (3TF: {_tfa_str})")

        # ── SELL confidence floor (symmetric — news override cannot lower it) ────
        if model_signal == "SELL" and confidence < SELL_CONFIDENCE_THRESHOLD:
            decision_delta = (
                f"SELL@{confidence} BLOCKED: below SELL floor {SELL_CONFIDENCE_THRESHOLD} "
                f"(det_score={det_score})"
            )
            signal           = "HOLD"
            result["signal"] = "HOLD"
            should_alert     = False
            print(
                f"⚠️  [DecisionAgent] SELL confidence {confidence} < {SELL_CONFIDENCE_THRESHOLD} "
                f"→ HOLD  (det_score={det_score})"
            )

        # ── General confidence threshold ──────────────────────────────────────
        elif model_signal in ("BUY", "SELL") and confidence < threshold:
            decision_delta = (
                f"model={model_signal}@{confidence} BLOCKED by threshold {threshold} "
                f"(det_score={det_score})"
            )
            signal           = "HOLD"
            result["signal"] = "HOLD"
            should_alert     = False
            print(
                f"⚠️  [DecisionAgent] Confidence {confidence}/100 < threshold {threshold} "
                f"— overriding {model_signal} → HOLD  (det_score={det_score})"
            )

        # ── Minimum absolute score gate — blocks weak/negative det_scores ─────
        elif model_signal in ("BUY", "SELL") and abs(det_score) < MIN_ALERT_SCORE:
            decision_delta = (
                f"{model_signal}@{confidence} BLOCKED: abs(det_score)={abs(det_score)} "
                f"< min {MIN_ALERT_SCORE}"
            )
            signal           = "HOLD"
            result["signal"] = "HOLD"
            should_alert     = False
            print(
                f"⚠️  [DecisionAgent] abs(det_score)={abs(det_score)} < {MIN_ALERT_SCORE} "
                f"— blocking {model_signal} → HOLD"
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
            _bsc = result.get("bull_score", 0)
            _rsc = result.get("bear_score", 0)
            _net = result.get("net_score", 0)
            decision_delta = (
                f"Bull({_bsc}) Bear({_rsc}) net={_net:+d} "
                f"→ {model_signal}@{confidence} thr={threshold}"
            )
            signal       = model_signal
            should_alert = signal in ("BUY", "SELL")
            emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
            print(
                f"{emoji} [DecisionAgent] {signal}  "
                f"confidence={confidence}/100  Bull({_bsc}) Bear({_rsc}) net={_net:+d}  "
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
            # LangSmith run ID — stored with the signal for loss-flagging
            "_langsmith_run_id": get_current_run_id(),
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
