"""
signal_aggregator.py — LangGraph node: aggregates all signals from DataAgent,
NewsAgent, and TechAgent into a consensus picture before DecisionAgent.

Runs after parallel_analyze (news + tech), before decide.

If agreement_pct < agreement_min (regime-adaptive, default 55) sets
skip_claude=True and returns HOLD immediately, saving a Claude API call
for genuinely mixed setups.
"""

import json
import os
import time
from datetime import datetime, timezone

from intelligence_hub import hub
from langsmith import traceable
from utils.tracing import annotate_run

# ── Tomorrow-watchlist cache (refresh every 5 min) ─────────────────────────────
_twl_tickers:     set   = set()
_twl_loaded_at:   float = 0.0
_TWL_TTL_S:       float = 300.0
_TWL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "tomorrow_watchlist.json")


def _get_tomorrow_tickers() -> set:
    global _twl_tickers, _twl_loaded_at
    if time.time() - _twl_loaded_at < _TWL_TTL_S:
        return _twl_tickers
    try:
        if os.path.exists(_TWL_PATH):
            with open(_TWL_PATH) as f:
                data = json.load(f)
            _twl_tickers = {s["ticker"] for s in data.get("setups", []) if s.get("ticker")}
        else:
            _twl_tickers = set()
    except Exception:
        _twl_tickers = set()
    _twl_loaded_at = time.time()
    return _twl_tickers


# ── Signal → hub weight key mapping ───────────────────────────────────────────
# Used to apply reflection weights to signal scores in aggregator_node.
_SIG_TO_HUB_WEIGHT: dict = {
    "rsi_very_oversold":  "rsi_oversold",
    "rsi_oversold":       "rsi_oversold",
    "rsi_very_overbought":"rsi_oversold",
    "rsi_overbought":     "rsi_oversold",
    "macd_bullish_cross": "macd",
    "macd_bearish_cross": "macd",
    "ema_golden_cross":   "ema_cross",
    "ema_death_cross":    "ema_cross",
    "price_above_vwap":   "vwap",
    "price_below_vwap":   "vwap",
    "volume_spike_3x":    "volume_spike",
    "volume_spike_2x":    "volume_spike",
    "gap_up_confirmed":   "gap_up",
    "gap_down_confirmed": "gap_up",
    "at_support":         "sr_level",
    "breakout":           "sr_level",
    "at_resistance":      "sr_level",
    "breakdown":          "sr_level",
    "obv_accumulation":   "obv",
    "obv_distribution":   "obv",
    "fresh_news_catalyst":"news_sentiment",
    "edgar_filing":       "edgar",
    "stoch_rsi_buy":      "stoch_rsi",
}


# ── Signal derivations ─────────────────────────────────────────────────────────

def _macd_signal(macd: dict) -> str:
    h = macd.get("histogram", 0.0) if isinstance(macd, dict) else 0.0
    return "bullish_cross" if h > 0 else "bearish_cross" if h < 0 else "neutral"


def _vwap_signal(price: float, vwap: float) -> str:
    if vwap <= 0 or price <= 0:
        return "UNKNOWN"
    return "ABOVE" if price > vwap else "BELOW"


def _bb_position(price: float, bollinger: dict) -> str:
    if not isinstance(bollinger, dict):
        return "INSIDE_BB"
    upper = bollinger.get("upper", 0.0)
    lower = bollinger.get("lower", 0.0)
    if upper > 0 and price > upper:
        return "ABOVE_BB"
    if lower > 0 and price < lower:
        return "BELOW_BB"
    return "INSIDE_BB"


def _sr_signal(price: float, support: float, resistance: float) -> str:
    if support <= 0 or resistance <= 0 or price <= 0:
        return "NONE"
    if abs(price - support) / price < 0.02:
        return "AT_SUPPORT"
    if abs(price - resistance) / price < 0.02:
        return "AT_RESISTANCE"
    if price > resistance * 1.01:
        return "BREAKOUT"
    if price < support * 0.99:
        return "BREAKDOWN"
    return "NONE"


def _news_age_hours(raw_news: list) -> float:
    """Age of most recent news item in hours. Returns 999 if no news."""
    if not raw_news:
        return 999.0
    now = datetime.now(timezone.utc)
    min_age = 999.0
    for item in raw_news:
        ts = (
            item.get("published_utc")
            or item.get("published_at")
            or item.get("datetime")
            or item.get("date", "")
        )
        if not ts:
            continue
        try:
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_h = (now - dt).total_seconds() / 3600
            min_age = min(min_age, age_h)
        except Exception:
            continue
    return min_age


def _pattern_name(p) -> str:
    """Extract uppercase pattern name from dict or string."""
    if isinstance(p, dict):
        return p.get("pattern", "").upper()
    return str(p).upper()


# ── Main node ──────────────────────────────────────────────────────────────────

@traceable(name="signal_aggregator", tags=["pipeline", "aggregation"])
def aggregator_node(state: dict) -> dict:
    annotate_run(state)
    # ── Load hub: reflection weights + regime thresholds ───────────────────────
    hub_weights    = hub.get_reflection_weights()
    thresholds     = hub.get_regime_thresholds()
    agreement_min  = thresholds.get("agreement_min", 55)

    # ── EDGAR 8-K major catalyst override ──────────────────────────────────────
    # Must run BEFORE agreement_pct check so 8-K events never skip Claude.
    has_edgar_8k = (
        bool(state.get("has_edgar_filing", False)) and
        state.get("edgar_filing_type", "") == "8-K"
    )
    if has_edgar_8k:
        print(f"🚨 [Aggregator] EDGAR 8-K detected for {state.get('ticker','?')} "
              f"— major catalyst override (Claude will NOT be skipped)")

    ticker       = state.get("ticker", "?")
    price        = state.get("current_price", 0.0)
    rsi          = state.get("rsi", 50.0)
    macd         = state.get("macd", {})
    ema_stack    = state.get("ema_stack", {})
    vwap         = state.get("vwap", 0.0)
    bollinger    = state.get("bollinger", {})
    volume_ratio = state.get("volume_spike_ratio", state.get("volume_ratio", 1.0))
    gap_info     = state.get("gap_info", {})
    support      = state.get("support", 0.0)
    resistance   = state.get("resistance", 0.0)
    patterns     = state.get("patterns", [])
    smart_money  = state.get("smart_money", "NEUTRAL")
    raw_news     = state.get("raw_news", [])
    market_regime = state.get("market_regime", {})
    sector_mom   = state.get("sector_momentum", {})

    # Derived intermediate signals
    macd_sig = _macd_signal(macd)
    ema_sig  = ema_stack.get("alignment", "MIXED") if isinstance(ema_stack, dict) else "MIXED"
    vwap_sig = _vwap_signal(price, vwap)
    bb_pos   = _bb_position(price, bollinger)
    gap_sig  = gap_info.get("signal", "NEUTRAL") if isinstance(gap_info, dict) else "NEUTRAL"
    sr_sig   = _sr_signal(price, support, resistance)
    has_news = len(raw_news) > 0
    news_age = _news_age_hours(raw_news)
    has_edgar = bool(state.get("has_edgar_filing", False))
    regime   = market_regime.get("regime", "UNKNOWN") if isinstance(market_regime, dict) else "UNKNOWN"
    sector_s = sector_mom.get("signal", "NEUTRAL") if isinstance(sector_mom, dict) else "NEUTRAL"
    stoch_rsi = state.get("stoch_rsi_signal", "NONE")

    pattern_names = [_pattern_name(p) for p in patterns]

    bullish_signals: list = []
    bearish_signals: list = []

    # ── BULLISH signals ────────────────────────────────────────────────────────
    if rsi < 30:
        bullish_signals.append(("rsi_very_oversold", 1.0))
    elif rsi < 40:
        bullish_signals.append(("rsi_oversold", 0.9))

    if macd_sig == "bullish_cross":
        bullish_signals.append(("macd_bullish_cross", 0.85))

    if ema_sig == "GOLDEN":
        bullish_signals.append(("ema_golden_cross", 0.8))

    if vwap_sig == "ABOVE":
        bullish_signals.append(("price_above_vwap", 0.75))

    if volume_ratio > 3.0:
        bullish_signals.append(("volume_spike_3x", 0.95))
    elif volume_ratio > 2.0:
        bullish_signals.append(("volume_spike_2x", 0.85))

    if gap_sig == "GAP_UP_CONFIRMED":
        bullish_signals.append(("gap_up_confirmed", 0.8))

    if sr_sig == "AT_SUPPORT":
        bullish_signals.append(("at_support", 0.75))
    elif sr_sig == "BREAKOUT":
        bullish_signals.append(("breakout", 0.9))

    for pname in pattern_names:
        if pname in ("HAMMER", "BULL_ENGULFING", "BULLISH_ENGULFING", "MORNING_STAR",
                     "PIERCING_LINE", "DRAGONFLY_DOJI"):
            bullish_signals.append((f"candle_{pname.lower()}", 0.7))
            break  # one candle is enough

    if smart_money == "ACCUMULATION":
        bullish_signals.append(("obv_accumulation", 0.75))

    if has_news and news_age < 2:
        bullish_signals.append(("fresh_news_catalyst", 0.95))

    if has_edgar:
        bullish_signals.append(("edgar_filing", 0.95))

    if regime == "CALM":
        bullish_signals.append(("calm_regime", 0.7))

    if sector_s == "STRONG_TAILWIND":
        bullish_signals.append(("sector_tailwind", 0.7))

    if stoch_rsi == "BUY_CROSS":
        bullish_signals.append(("stoch_rsi_buy", 0.8))

    # Pre-identified by yesterday's EOD scanner → modest bullish prior
    if ticker in _get_tomorrow_tickers():
        bullish_signals.append(("in_tomorrow_watchlist", 0.65))

    # ── BEARISH signals ────────────────────────────────────────────────────────
    if rsi > 80:
        bearish_signals.append(("rsi_very_overbought", 1.0))
    elif rsi > 70:
        bearish_signals.append(("rsi_overbought", 0.9))

    if macd_sig == "bearish_cross":
        bearish_signals.append(("macd_bearish_cross", 0.85))

    if ema_sig == "DEATH":
        bearish_signals.append(("ema_death_cross", 0.8))

    if vwap_sig == "BELOW":
        bearish_signals.append(("price_below_vwap", 0.75))

    if bb_pos == "ABOVE_BB":
        bearish_signals.append(("above_bollinger_band", 0.7))

    if gap_sig == "GAP_DOWN_CONFIRMED":
        bearish_signals.append(("gap_down_confirmed", 0.8))

    if sr_sig == "AT_RESISTANCE":
        bearish_signals.append(("at_resistance", 0.7))
    elif sr_sig == "BREAKDOWN":
        bearish_signals.append(("breakdown", 0.9))

    for pname in pattern_names:
        if pname in ("SHOOTING_STAR", "BEAR_ENGULFING", "BEARISH_ENGULFING",
                     "EVENING_STAR", "GRAVESTONE_DOJI", "DARK_CLOUD_COVER"):
            bearish_signals.append((f"candle_{pname.lower()}", 0.7))
            break

    if smart_money == "DISTRIBUTION":
        bearish_signals.append(("obv_distribution", 0.75))

    if regime in ("FEAR", "PANIC"):
        bearish_signals.append(("fear_panic_regime", 0.9))

    if sector_s == "STRONG_HEADWIND":
        bearish_signals.append(("sector_headwind", 0.7))

    # ── Apply reflection weights to signal scores ──────────────────────────────
    def _apply_weights(signals: list) -> list:
        out = []
        for name, base_w in signals:
            hub_key = _SIG_TO_HUB_WEIGHT.get(name)
            if hub_key is None and name.startswith("candle_"):
                hub_key = "pattern"
            mult = hub_weights.get(hub_key, 1.0) if hub_key else 1.0
            out.append((name, round(base_w * mult, 4)))
        return out

    bullish_signals = _apply_weights(bullish_signals)
    bearish_signals = _apply_weights(bearish_signals)

    # ── Calculate consensus ────────────────────────────────────────────────────
    bull_score = sum(w for _, w in bullish_signals)
    bear_score = sum(w for _, w in bearish_signals)
    total      = bull_score + bear_score

    if total == 0:
        agreement_pct = 50.0
        consensus     = "NEUTRAL"
    else:
        leading       = max(bull_score, bear_score)
        agreement_pct = (leading / total) * 100
        consensus     = "BULLISH" if bull_score >= bear_score else "BEARISH"

    # Highest-weight signal for logging
    all_leading = bullish_signals if bull_score >= bear_score else bearish_signals
    top_signal  = max(all_leading, key=lambda x: x[1])[0] if all_leading else "none"

    print(
        f"📊 [Aggregator] {ticker}: "
        f"{len(bullish_signals)} bull signals vs {len(bearish_signals)} bear signals  "
        f"(bull={bull_score:.2f}  bear={bear_score:.2f}  top={top_signal})"
    )
    proceed_msg = (f"proceeding to Claude"
                   if agreement_pct >= agreement_min
                   else f"HOLD (low agreement, regime={thresholds.get('regime','?')} min={agreement_min}%)")
    print(f"📊 [Aggregator] Agreement: {agreement_pct:.0f}% {consensus} → {proceed_msg}")

    updates: dict = {
        "bullish_signals":   bullish_signals,
        "bearish_signals":   bearish_signals,
        "agreement_score":   round(agreement_pct, 1),
        "consensus":         consensus,
        "signal_count_bull": len(bullish_signals),
        "signal_count_bear": len(bearish_signals),
        "major_catalyst":    has_edgar_8k,
        "skip_claude":       False,
        "skip_reason":       "",
    }

    if agreement_pct < agreement_min and not has_edgar_8k:
        print(
            f"⚖️  [Aggregator] Agents disagree ({agreement_pct:.0f}% < {agreement_min}% min) → "
            f"HOLD, skipping Claude call"
        )
        updates.update({
            "signal":      "HOLD",
            "confidence":  0,
            "skip_reason": f"Agents disagree: agreement {agreement_pct:.0f}% < {agreement_min}% ({thresholds.get('regime','?')} regime)",
            "skip_claude": True,
        })
    elif agreement_pct < agreement_min and has_edgar_8k:
        print(
            f"⚖️  [Aggregator] Low agreement ({agreement_pct:.0f}%) but "
            f"EDGAR 8-K present — proceeding to Claude anyway"
        )

    return {**state, **updates}
