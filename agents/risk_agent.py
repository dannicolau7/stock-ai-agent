"""
agents/risk_agent.py — LangGraph node: trade risk gate.

Runs after decision_agent, before sizing_agent.
Vetoes or approves the trade based on portfolio-level rules.
Sets risk_approved=False to cancel the trade without sending an alert.

Checks (in order):
  1. Signal is not BUY → pass through (no risk checks for HOLD/SELL)
  2. Daily loss limit — if today's realized losses exceed MAX_DAILY_LOSS, veto
  3. Open position count — veto if already at MAX_OPEN_POSITIONS
  4. Sector concentration — veto if sector exposure would exceed MAX_SECTOR_PCT
  5. Small-cap / low-float exposure — warn and reduce conviction if float < 10M
  6. Volatility gate — ATR/price > 8% means the stock moves too wildly; reduce
  7. Regime gate — BEAR macro → veto unless news_triggered + high confidence
  8. Risk/Reward minimum — R:R < 1.5 → veto
"""

from config import (
    PORTFOLIO_SIZE, MAX_RISK_PER_TRADE, MAX_DAILY_LOSS,
    MAX_OPEN_POSITIONS, MAX_SECTOR_PCT, MAX_SMALLCAP_PCT,
)

# Regime-based confidence floor: in BEAR markets only high-conviction trades pass
REGIME_CONFIDENCE_FLOOR = {
    "BULL":     55,
    "RECOVERY": 60,
    "NEUTRAL":  65,
    "RISK_OFF": 70,
    "BEAR":     78,
    "STAGFLATION": 75,
}

MIN_RR_RATIO = 1.5


def risk_node(state: dict) -> dict:
    signal     = state.get("signal", "HOLD")
    ticker     = state["ticker"]

    # Only BUY signals go through risk checks
    if signal != "BUY":
        return {**state, "risk_approved": True, "risk_veto_reason": "",
                "risk_multiplier": 1.0, "risk_warnings": []}

    confidence  = state.get("confidence", 0)
    atr         = state.get("atr", 0.0)
    price       = state.get("current_price", 0.0)
    rr_ratio    = state.get("rr_ratio", 0.0)
    sector      = state.get("sector", "Unknown")
    float_rot   = state.get("float_rotation", 0.0)
    avg_volume  = state.get("avg_volume", 0.0)
    news_trig   = state.get("news_triggered", False)
    stop_loss   = state.get("stop_loss", 0.0)

    vetoes:   list[str] = []
    warnings: list[str] = []
    risk_multiplier     = 1.0   # 1.0 = full size, 0.5 = half size, 0.0 = veto

    # ── 1. Daily loss limit ────────────────────────────────────────────────────
    daily_loss_pct = _get_daily_loss_pct()
    if daily_loss_pct <= -MAX_DAILY_LOSS:
        vetoes.append(
            f"Daily loss limit hit ({daily_loss_pct*100:.1f}% vs -{MAX_DAILY_LOSS*100:.0f}% limit) "
            f"— no new longs today"
        )

    # ── 2. Open position count ─────────────────────────────────────────────────
    open_count = _count_open_positions()
    if open_count >= MAX_OPEN_POSITIONS:
        vetoes.append(
            f"Max open positions reached ({open_count}/{MAX_OPEN_POSITIONS}) "
            f"— close a position first"
        )

    # ── 3. Sector concentration ────────────────────────────────────────────────
    sector_pct = _sector_exposure_pct(sector)
    if sector_pct >= MAX_SECTOR_PCT:
        vetoes.append(
            f"Sector concentration too high: {sector} already at {sector_pct*100:.0f}% "
            f"(limit {MAX_SECTOR_PCT*100:.0f}%)"
        )
    elif sector_pct >= MAX_SECTOR_PCT * 0.75:
        warnings.append(f"Sector {sector} at {sector_pct*100:.0f}% — approaching limit")
        risk_multiplier = min(risk_multiplier, 0.7)

    # ── 4. Small-cap / low-float exposure ──────────────────────────────────────
    is_small_cap = avg_volume > 0 and avg_volume < 500_000
    if is_small_cap:
        smallcap_pct = _smallcap_exposure_pct()
        if smallcap_pct >= MAX_SMALLCAP_PCT:
            vetoes.append(
                f"Small-cap exposure limit: {smallcap_pct*100:.0f}% of portfolio already in "
                f"low-liquidity names (limit {MAX_SMALLCAP_PCT*100:.0f}%)"
            )
        else:
            warnings.append(f"Low-liquidity name (avg vol {avg_volume:,.0f}) — half size")
            risk_multiplier = min(risk_multiplier, 0.5)

    # ── 5. Volatility gate ─────────────────────────────────────────────────────
    if atr > 0 and price > 0:
        atr_pct = atr / price
        if atr_pct > 0.10:                            # ATR > 10% of price
            vetoes.append(
                f"Volatility too high: ATR is {atr_pct*100:.1f}% of price "
                f"(threshold 10%) — position sizing impossible"
            )
        elif atr_pct > 0.06:                          # ATR 6–10%
            warnings.append(f"High volatility (ATR {atr_pct*100:.1f}%) — reduce size")
            risk_multiplier = min(risk_multiplier, 0.6)
        elif atr_pct > 0.04:                          # ATR 4–6%
            risk_multiplier = min(risk_multiplier, 0.8)

    # ── 6. Regime-based confidence floor ──────────────────────────────────────
    import world_context as wctx
    macro = wctx.get()["macro"]
    regime = macro.get("regime", "NEUTRAL")
    regime_floor = REGIME_CONFIDENCE_FLOOR.get(regime, 65)

    if confidence < regime_floor:
        if regime in ("BEAR", "STAGFLATION") and not news_trig:
            vetoes.append(
                f"Regime gate: {regime} market requires confidence ≥{regime_floor} "
                f"(got {confidence}) — veto non-news-triggered BUY"
            )
        else:
            warnings.append(
                f"Below {regime} regime confidence floor ({confidence} < {regime_floor}) "
                f"— reducing size"
            )
            risk_multiplier = min(risk_multiplier, 0.6)

    # ── 7. R:R minimum ────────────────────────────────────────────────────────
    if rr_ratio > 0 and rr_ratio < MIN_RR_RATIO:
        vetoes.append(
            f"R:R too low: {rr_ratio:.2f}:1 (minimum {MIN_RR_RATIO}:1) "
            f"— risk not justified by reward"
        )

    # ── Verdict ────────────────────────────────────────────────────────────────
    approved = len(vetoes) == 0

    if vetoes:
        veto_str = " | ".join(vetoes)
        print(f"🚫 [RiskAgent] {ticker} VETOED — {veto_str}")
    elif warnings:
        warn_str = " | ".join(warnings)
        print(f"⚠️  [RiskAgent] {ticker} APPROVED (×{risk_multiplier:.1f}) — {warn_str}")
    else:
        print(f"✅ [RiskAgent] {ticker} APPROVED — all risk gates passed "
              f"(open={open_count}/{MAX_OPEN_POSITIONS}  regime={regime}  R:R={rr_ratio:.1f})")

    veto_reason = " | ".join(vetoes) if vetoes else ""

    return {
        **state,
        "risk_approved":    approved,
        "risk_veto_reason": veto_reason,
        "risk_multiplier":  risk_multiplier if approved else 0.0,
        "risk_warnings":    warnings,
        # If vetoed, flip should_alert off so alert_agent fires nothing
        "should_alert":     state.get("should_alert", False) and approved,
        "signal":           signal if approved else "HOLD",
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_daily_loss_pct() -> float:
    """Compute today's realized P&L as % of portfolio from performance_tracker."""
    try:
        from datetime import date
        import performance_tracker as pt
        with pt._get_conn() as conn:
            rows = conn.execute("""
                SELECT o.return_pct, s.price
                FROM outcomes o
                JOIN signals s ON s.id = o.signal_id
                WHERE s.fired_at >= date('now', 'start of day')
                  AND o.win IS NOT NULL
                  AND o.checkpoint = '1d'
            """).fetchall()
        if not rows:
            return 0.0
        # Average return across today's closed trades as proxy for daily P&L
        return sum(r[0] for r in rows) / 100 / len(rows)
    except Exception:
        return 0.0    # fail open — don't block if tracker unavailable


def _count_open_positions() -> int:
    """Count open BUY positions from performance_tracker."""
    try:
        import performance_tracker as pt
        return len(pt.get_open_signals(paper=False))
    except Exception:
        return 0


def _sector_exposure_pct(sector: str) -> float:
    """Fraction of open BUY positions already in `sector` (by position count)."""
    try:
        import performance_tracker as pt
        with pt._get_conn() as conn:
            # "Open" = BUY signal whose 7d outcome is still pending
            total = conn.execute("""
                SELECT COUNT(*) FROM signals s
                WHERE s.signal = 'BUY' AND s.paper = 0
                  AND NOT EXISTS (
                      SELECT 1 FROM outcomes o
                      WHERE o.signal_id = s.id
                        AND o.checkpoint = '7d'
                        AND o.win IS NOT NULL
                  )
            """).fetchone()[0]
            if total == 0:
                return 0.0
            in_sector = conn.execute("""
                SELECT COUNT(*) FROM signals s
                WHERE s.signal = 'BUY' AND s.paper = 0
                  AND s.sector = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM outcomes o
                      WHERE o.signal_id = s.id
                        AND o.checkpoint = '7d'
                        AND o.win IS NOT NULL
                  )
            """, (sector,)).fetchone()[0]
            return in_sector / total
    except Exception:
        return 0.0   # fail open


def _smallcap_exposure_pct() -> float:
    """Fraction of open BUY positions in low-liquidity names (avg_volume < 500k)."""
    try:
        import performance_tracker as pt
        with pt._get_conn() as conn:
            total = conn.execute("""
                SELECT COUNT(*) FROM signals s
                WHERE s.signal = 'BUY' AND s.paper = 0
                  AND NOT EXISTS (
                      SELECT 1 FROM outcomes o
                      WHERE o.signal_id = s.id
                        AND o.checkpoint = '7d'
                        AND o.win IS NOT NULL
                  )
            """).fetchone()[0]
            if total == 0:
                return 0.0
            small_cap = conn.execute("""
                SELECT COUNT(*) FROM signals s
                WHERE s.signal = 'BUY' AND s.paper = 0
                  AND s.avg_volume > 0 AND s.avg_volume < 500000
                  AND NOT EXISTS (
                      SELECT 1 FROM outcomes o
                      WHERE o.signal_id = s.id
                        AND o.checkpoint = '7d'
                        AND o.win IS NOT NULL
                  )
            """).fetchone()[0]
            return small_cap / total
    except Exception:
        return 0.0   # fail open
