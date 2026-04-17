"""
decision_validator.py — LangGraph node: hard validation rules that override
Claude's decision. Runs after decide, before assess_risk.

Each rule is logged individually for traceability.
All structural fixes (entry zone, stop loss) are applied regardless of signal.
"""


def validator_node(state: dict) -> dict:
    signal     = state.get("signal", "HOLD")
    confidence = state.get("confidence", 0)
    rsi        = state.get("rsi", 50.0)
    atr        = state.get("atr", 0.0)
    price      = state.get("current_price", state.get("price", 0.0))
    agreement  = state.get("agreement_score", 100.0)
    raw_news   = state.get("raw_news", [])
    rvol       = state.get("volume_spike_ratio", state.get("volume_ratio", 1.0))
    gap_info   = state.get("gap_info", {})
    gap_pct    = gap_info.get("gap_pct", 0.0) if isinstance(gap_info, dict) else 0.0
    regime     = (state.get("market_regime") or {}).get("regime", "UNKNOWN")

    entry_low  = state.get("entry_low",  0.0) or 0.0
    entry_high = state.get("entry_high", 0.0) or 0.0
    stop_loss  = state.get("stop_loss",  0.0) or 0.0
    stop_pct   = state.get("stop_pct",   0.0) or 0.0

    _atr = atr if atr > 0 else (price * 0.015 if price > 0 else 0.0)

    overrides: list = []

    # ── Rule 1 — RSI overbought gate ───────────────────────────────────────────
    if rsi > 73 and signal == "BUY":
        overrides.append(f"RSI overbought {rsi:.1f}")
        signal = "HOLD"
        print(f"🛑 [Validator] HOLD override: RSI overbought {rsi:.1f}")

    # ── Rule 2 — Panic gate (no shorting in panic markets) ─────────────────────
    if regime == "PANIC" and signal == "SELL":
        overrides.append("Market panic — no short selling")
        signal = "HOLD"
        print("🛑 [Validator] HOLD override: Market panic — no short selling")

    # ── Rule 3 — Entry zone gate (structural fix, not a veto) ─────────────────
    if price > 0 and _atr > 0:
        if entry_low <= 0 or entry_high <= 0 or entry_low >= entry_high:
            entry_low  = round(price - 0.5 * _atr, 4)
            entry_high = round(price + 0.5 * _atr, 4)
            print(
                f"⚠️  [Validator] Fixed invalid entry zone "
                f"→ ${entry_low:.2f} – ${entry_high:.2f}"
            )

    # ── Rule 4 — Stop loss gate (structural fix) ───────────────────────────────
    if price > 0 and _atr > 0:
        if stop_loss <= 0 or stop_loss >= price:
            stop_loss = round(price - 2.0 * _atr, 4)
            print(f"⚠️  [Validator] Fixed invalid stop loss → ${stop_loss:.2f}")
        stop_pct = (stop_loss - price) / price * 100
        if abs(stop_pct) < 0.5:
            stop_loss = round(price - 2.0 * _atr, 4)
            stop_pct  = (stop_loss - price) / price * 100
            print(
                f"⚠️  [Validator] Fixed zero stop pct "
                f"→ ${stop_loss:.2f} ({stop_pct:.1f}%)"
            )

    # ── Rule 5 — Agreement gate ────────────────────────────────────────────────
    if agreement < 55 and signal != "HOLD":
        overrides.append(f"Low agent agreement {agreement:.0f}%")
        signal = "HOLD"
        print(f"🛑 [Validator] HOLD override: Low agent agreement {agreement:.0f}%")

    # ── Rule 6 — Catalyst gate (BUY needs news OR unusual volume) ─────────────
    has_news = len(raw_news) > 0
    if not has_news and rvol < 1.8 and signal == "BUY":
        overrides.append("No catalyst: no news + normal volume")
        signal = "HOLD"
        print("🛑 [Validator] HOLD override: No catalyst: no news + normal volume")

    # ── Rule 7 — Chasing gate (extended gap without volume confirmation) ────────
    if gap_pct > 10.0 and rvol < 3.0 and signal == "BUY":
        overrides.append(f"Chasing extended gap ({gap_pct:.1f}%) without volume")
        signal = "HOLD"
        print(
            f"🛑 [Validator] HOLD override: "
            f"Chasing extended gap ({gap_pct:.1f}%) without volume"
        )

    # ── Rule 8 — Confidence gate ───────────────────────────────────────────────
    threshold_used      = state.get("threshold_used", 65)
    effective_threshold = max(threshold_used, 52)
    if confidence < effective_threshold and signal in ("BUY", "SELL"):
        overrides.append(
            f"Confidence {confidence} below effective threshold {effective_threshold}"
        )
        signal = "HOLD"
        print(
            f"🛑 [Validator] Rule 8: conf {confidence} "
            f"< threshold {effective_threshold} → HOLD"
        )

    validator_passed = len(overrides) == 0
    if validator_passed and signal in ("BUY", "SELL"):
        print(f"✅ [Validator] All rules passed — {signal} signal cleared")

    # Rebuild entry_zone string in sync with fixed floats
    new_entry_zone = f"${entry_low:.2f} - ${entry_high:.2f}" if entry_low > 0 else state.get("entry_zone", "")

    return {
        **state,
        "signal":              signal,
        "entry_zone":          new_entry_zone,
        "entry_low":           entry_low,
        "entry_high":          entry_high,
        "stop_loss":           round(stop_loss, 4),
        "stop_pct":            round(stop_pct, 1),
        "final_signal":        signal,
        "validator_passed":    validator_passed,
        "validator_overrides": overrides,
        # Re-evaluate should_alert: only True if signal survived all rules
        "should_alert":        signal in ("BUY", "SELL") and validator_passed,
    }
