"""
analyzer.py — calls Claude with full market context + adaptive weights from
self_learner.py. Returns structured signal: BUY/SELL/HOLD with confidence,
entry zone, targets, stop loss, and reasoning.

Unique scoring framework includes:
  OBV / Smart Money Divergence / EMA Stack / Float Rotation /
  Social Velocity / Sector Momentum / Catalyst Timing /
  Pre-Market Gap / Earnings Proximity / Self-Learning weights
"""

import json
import anthropic
from langsmith import traceable
from config import ANTHROPIC_API_KEY
from self_learner import get_weight_adjustments, get_summary as sl_summary
import world_context as wctx

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


@traceable(name="claude_analyzer", tags=["claude", "llm"])
def analyze_market(context: dict) -> dict:
    ticker         = context.get("ticker", "UNKNOWN")
    price          = context.get("current_price", 0.0)
    prev_close     = context.get("prev_close", price)
    volume         = context.get("volume", 0)
    avg_volume     = context.get("avg_volume", 0)
    rsi            = context.get("rsi", 50.0)
    macd           = context.get("macd", {})
    bollinger      = context.get("bollinger", {})
    atr            = context.get("atr", 0.0)
    support        = context.get("support", 0.0)
    resistance     = context.get("resistance", 0.0)
    volume_spike   = context.get("volume_spike", False)
    vol_ratio      = context.get("volume_spike_ratio", 1.0)
    vwap           = context.get("vwap", 0.0)
    obv            = context.get("obv", 0.0)
    smart_money    = context.get("smart_money", "NEUTRAL")
    ema_stack      = context.get("ema_stack", {})
    float_rot      = context.get("float_rotation", 0.0)
    sector_m       = context.get("sector_momentum", {})
    timing         = context.get("timing", {})
    gap_info       = context.get("gap_info", {})
    earnings_info  = context.get("earnings_info", {})
    market_regime  = context.get("market_regime", {})
    rel_str        = context.get("relative_strength", {})
    score_bd       = context.get("score_breakdown", {})
    intra_rsi      = context.get("intraday_rsi", 50.0)
    sr_levels      = context.get("sr_levels", {})
    news_sentiment = context.get("news_sentiment", "NEUTRAL")
    sentiment_score= context.get("sentiment_score", 50)
    news_summary   = context.get("news_summary", "")
    social_vel     = context.get("social_velocity", {})
    patterns       = context.get("patterns", [])

    day_change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

    # ── Self-learning weights ──────────────────────────────────────────────────
    weights = get_weight_adjustments()
    w_macd        = weights.get("macd", 1.0)
    w_rsi_bounce  = weights.get("rsi_bounce", 1.0)
    w_rsi_mom     = weights.get("rsi_momentum", 1.0)
    w_vwap        = weights.get("vwap", 1.0)
    w_volume      = weights.get("volume", 1.0)
    w_bb          = weights.get("bollinger", 1.0)
    w_sentiment   = weights.get("sentiment", 1.0)
    w_support     = weights.get("support", 1.0)
    w_ema         = weights.get("ema_stack", 1.0)
    w_smd         = weights.get("smart_money", 1.0)
    w_float       = weights.get("float_rot", 1.0)
    w_gap         = weights.get("gap", 1.0)

    # ── Build prompt context strings ───────────────────────────────────────────
    bb_upper = bollinger.get("upper", 0)
    bb_mid   = bollinger.get("middle", 0)
    bb_lower = bollinger.get("lower", 0)
    bb_bw    = bollinger.get("bandwidth", 0)
    bb_pos   = (
        "ABOVE_UPPER (overbought)" if price > bb_upper > 0
        else "BELOW_LOWER (oversold)" if price < bb_lower > 0
        else "UPPER_HALF" if price > bb_mid > 0
        else "LOWER_HALF"
    )

    vwap_line = ""
    if vwap > 0:
        pos = "ABOVE VWAP ✅ (bullish)" if price >= vwap else "BELOW VWAP ⚠️ (bearish)"
        vwap_line = f"VWAP:           ${vwap:.4f}  →  {pos}\n"

    ema_line = ""
    if ema_stack.get("ema9"):
        ema_line = (
            f"EMA Stack:      9=${ema_stack['ema9']:.4f}  "
            f"21=${ema_stack['ema21']:.4f}  50=${ema_stack['ema50']:.4f}  "
            f"→ {ema_stack['alignment']}\n"
        )

    earnings_line = ""
    e_risk = earnings_info.get("earnings_risk", "none")
    e_days = earnings_info.get("days_to_earnings", 999)
    e_date = earnings_info.get("earnings_date", "")
    if e_risk != "none":
        earnings_line = (
            f"⚠️  EARNINGS IN {e_days} DAYS ({e_date}) — risk={e_risk.upper()}\n"
        )

    # Earnings confidence cap
    earnings_cap = 100
    if e_days <= 3:
        earnings_cap = 55
    elif e_days <= 7:
        earnings_cap = 65
    elif e_days <= 14:
        earnings_cap = 75

    # Timing multiplier
    t_mult  = timing.get("multiplier", 1.0)
    t_win   = timing.get("window", "unknown")

    # Float rotation label
    float_label = ""
    if float_rot > 100:
        float_label = f"🚀 EXTREME ({float_rot:.0f}% of float)"
    elif float_rot > 50:
        float_label = f"🔥 MAJOR ({float_rot:.0f}% of float)"
    elif float_rot > 20:
        float_label = f"elevated ({float_rot:.0f}% of float)"
    elif float_rot > 0:
        float_label = f"{float_rot:.0f}% of float"

    # Pattern string
    patterns_str = (
        ",  ".join(f"{p['pattern']} ({p['confidence']:.0%}) — {p['description']}"
                   for p in patterns)
        if patterns else "None detected"
    )

    # Pre-computed score breakdown string
    fired_str  = "  ".join(f"{s[0]} {s[1]}" for s in score_bd.get("fired", []))
    missed_str = ", ".join(score_bd.get("missed", [])[:6])
    score_line = (
        f"Pre-computed raw={score_bd.get('raw_score',0)} "
        f"× {score_bd.get('timing_mult',1.0)} timing "
        f"= {score_bd.get('final_score',0)} pts\n"
        f"  Fired: {fired_str or 'none'}\n"
        f"  Missed: {missed_str or 'none'}"
    )

    # Multi-level S/R string
    sr_str = (
        f"  5-day:  support ${sr_levels.get('support_5d',0):.4f}  "
        f"resist ${sr_levels.get('resistance_5d',0):.4f}\n"
        f"  10-day: support ${sr_levels.get('support_10d',0):.4f}  "
        f"resist ${sr_levels.get('resistance_10d',0):.4f}\n"
        f"  20-day: support ${sr_levels.get('support_20d',0):.4f}  "
        f"resist ${sr_levels.get('resistance_20d',0):.4f}"
    ) if sr_levels else f"  Support ${support:.4f}  Resistance ${resistance:.4f}"

    prompt = f"""You are an elite quantitative momentum trader with a unique multi-signal scoring system.
Analyze {ticker} and return a precise trading recommendation.

=== MARKET CONTEXT ===
Market regime:  SPY {market_regime.get('regime','?')} (EMA5={market_regime.get('ema5',0):.2f} vs EMA20={market_regime.get('ema20',0):.2f})
SPY today:      {market_regime.get('spy_day_chg',0):+.2f}%
Relative str:   {ticker} {rel_str.get('label','n/a')}
Pre-Market Gap: {gap_info.get('label', 'n/a')}
{earnings_line}
=== PRICE ACTION ===
Price:          ${price:.4f}
Prev Close:     ${prev_close:.4f}
Day Change:     {day_change_pct:+.2f}%

=== TECHNICAL INDICATORS ===
RSI daily(14):  {rsi:.1f}{'  ⚠️ OVERBOUGHT' if rsi > 70 else '  ⚠️ OVERSOLD' if rsi < 30 else '  ✅ IDEAL (30-65)' if 30 <= rsi <= 65 else ''}
RSI intraday:   {intra_rsi:.1f} (15-min bars)
MACD Histogram: {macd.get('histogram', 0):+.6f}{'  🟢 BULLISH' if macd.get('histogram', 0) > 0 else '  🔴 BEARISH'}
BB Position:    {bb_pos}  (bw={bb_bw:.4f})
ATR(14):        ${atr:.4f}
{vwap_line}{ema_line}OBV:            {obv:,.0f}
Smart Money:    {smart_money}  ({'🟢 hidden accumulation' if smart_money == 'ACCUMULATION' else '🔴 hidden distribution' if smart_money == 'DISTRIBUTION' else 'aligned'})

=== KEY LEVELS (MULTI-TIMEFRAME) ===
{sr_str}

=== VOLUME & FLOAT ===
Volume ratio:   {vol_ratio:.1f}x avg{'  🔥 SPIKE' if volume_spike else ''}
Float rotation: {float_label if float_label else 'low'}

=== SECTOR & TIMING ===
Sector ETF ({sector_m.get('etf','SPY')}): {sector_m.get('change_pct', 0):+.2f}%  →  {sector_m.get('signal','NEUTRAL')}
Market window:  {t_win}  (score multiplier ×{t_mult})

=== MULTI-SOURCE SENTIMENT ===
Combined score: {news_sentiment}  ({sentiment_score}/100)
Social velocity:{social_vel.get('label', 'n/a')}
Summary:        {news_summary}

=== UNIQUE SCORING FRAMEWORK (apply self-learned weights shown) ===
Score each active signal, multiply by its weight, then apply timing multiplier:

BULLISH signals:
  MACD bullish histogram:           +{round(20*w_macd)} pts  (weight ×{w_macd})
  RSI 30–50 bounce zone:            +{round(20*w_rsi_bounce)} pts  (weight ×{w_rsi_bounce})
  RSI 50–65 momentum zone:          +{round(10*w_rsi_mom)} pts  (weight ×{w_rsi_mom})
  Price above VWAP:                 +{round(20*w_vwap)} pts  (weight ×{w_vwap})
  EMA stack BULLISH (9>21>50):      +{round(15*w_ema)} pts  (weight ×{w_ema})
  Volume spike ≥2×:                 +{round(15*w_volume)} pts  (weight ×{w_volume})
  BB oversold (below lower band):   +{round(15*w_bb)} pts  (weight ×{w_bb})
  Smart money ACCUMULATION:         +{round(20*w_smd)} pts  (weight ×{w_smd})
  Float rotation >50%:              +{round(20*w_float)} pts  (weight ×{w_float})
  Sentiment BULLISH (≥60):          +{round(15*w_sentiment)} pts  (weight ×{w_sentiment})
  Social velocity surging (≥3×):    +15 pts
  Pre-market gap +2–10% (bullish):  +{round(10*w_gap)} pts  (weight ×{w_gap})
  Sector momentum BULLISH:          +10 pts
  Price near support (≤3%):         +{round(10*w_support)} pts  (weight ×{w_support})
  Day change +2% to +8%:            +10 pts

BEARISH / PENALTY signals:
  MACD bearish histogram:           -{round(20*w_macd)} pts
  RSI >70 overbought:               -20 pts
  EMA stack BEARISH (9<21<50):      -{round(15*w_ema)} pts
  Price below VWAP:                 -{round(15*w_vwap)} pts
  Smart money DISTRIBUTION:         -{round(20*w_smd)} pts
  Sentiment BEARISH (≤40):          -{round(15*w_sentiment)} pts
  Pre-market gap >10% (fade risk):  -15 pts
  Sector momentum BEARISH:          -10 pts
  Day change >+10% (extended):      -10 pts
  Float rotation >100% AND no news: -10 pts (manipulation risk)

Apply timing multiplier ×{t_mult} to final score.

=== DETECTED CHART PATTERNS ===
{patterns_str}

{wctx.build_prompt_section()}

=== PRE-COMPUTED SCORE (validate and adjust if needed) ===
{score_line}

IMPORTANT RULES:
- If volume ratio < 0.5×, cap confidence at 60 (no conviction)
- If earnings ≤{e_days} days away, cap confidence at {earnings_cap}
- If market regime is BEAR and signal is BUY, reduce confidence by 15
- Score ≥ 60 after timing → BUY, confidence 65–100
- Score 30–59 → HOLD
- Score < 30 or net negative → SELL or HOLD
- Entry zone = tightest 5-day support to current price
- Stop loss = 5-day support OR price − ATR, whichever is closer to price
- T1 = nearest resistance, T2 = +8–10% from entry, T3 = +18–22% from entry
- Risk/Reward must be ≥ 1.5:1 to issue a BUY

TRADE HORIZON RULES (pick exactly one):
- "intraday": RSI > 65 on daily (already extended), price at/above daily resistance, EMA stack mixed or bearish, pure intraday RVOL spike with no multi-day setup, or earnings ≤ 3 days away → take profits before the close
- "swing": EMA stack BULLISH, MACD histogram just turned positive or strengthening, RSI 40–65 with room to run, price breaking out with volume, clear sector tailwind → 2–5 day hold, trail stop daily
- "position": Major catalyst (earnings beat, FDA, partnership), RSI bouncing from deeply oversold (<35) on high volume, bullish alignment on daily AND weekly timeframe, strong sector rotation → week+ hold

Respond ONLY with this exact JSON (no markdown fences):
{{
  "signal": "BUY" or "SELL" or "HOLD",
  "confidence": <integer 0-100>,
  "entry_zone": "<e.g. '$1.85 - $1.93'>",
  "targets": [<T1 float>, <T2 float>, <T3 float>],
  "stop_loss": <float>,
  "reasoning": "<2-3 sentences: which signals fired, market regime, key risk>",
  "action_plan": "<specific step-by-step: when to enter, where to add, when to exit, what invalidates the trade>",
  "trade_horizon": "intraday" or "swing" or "position",
  "horizon_reasoning": "<1 sentence: the single most important factor that determined the timeframe>"
}}"""

    try:
        client   = _get_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text  = parts[1][4:] if parts[1].startswith("json") else parts[1]
        text = text.strip()

        # Robust JSON extraction — find the first {...} block
        if not text.startswith("{"):
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

        result          = json.loads(text)
        signal          = str(result.get("signal", "HOLD")).upper()
        confidence      = max(0, min(100, int(result.get("confidence", 0))))
        targets         = [float(t) for t in result.get("targets", [])]
        stop_loss       = float(result.get("stop_loss", round(price * 0.95, 4)))
        trade_horizon   = str(result.get("trade_horizon", "swing")).lower()
        horizon_reason  = str(result.get("horizon_reasoning", ""))
        if trade_horizon not in ("intraday", "swing", "position"):
            trade_horizon = "swing"

        # Hard-apply earnings cap
        if confidence > earnings_cap:
            confidence = earnings_cap
            print(f"⚠️  [Analyzer] Earnings cap applied → capped at {earnings_cap}")

        # Circuit breaker — suppress BUY on extreme fear / market selloff
        if signal == "BUY":
            try:
                from circuit_breaker import check_market
                spy_chg = context.get("market_regime", {}).get("spy_day_chg")
                cb = check_market(spy_day_chg=spy_chg)
                if not cb["safe"]:
                    print(f"🚫 [Analyzer] Circuit breaker triggered: {cb['reason']}")
                    signal     = "HOLD"
                    confidence = 0
                    result["reasoning"] = (
                        f"⚠️ Circuit breaker active: {cb['reason']}. "
                        f"BUY suppressed — wait for safer market conditions."
                    )
            except Exception as _cb_err:
                print(f"⚠️  [Analyzer] Circuit breaker check failed (fail-open): {_cb_err}")

        # Compute R:R ratio
        try:
            entry_str  = str(result.get("entry_zone", "")).replace("$", "")
            parts      = entry_str.split("-")
            entry_mid  = (float(parts[0].strip()) + float(parts[1].strip())) / 2
            t1         = targets[0] if targets else price * 1.05
            risk       = entry_mid - stop_loss
            reward     = t1 - entry_mid
            rr_ratio   = round(reward / risk, 2) if risk > 0 else 0.0
        except Exception:
            rr_ratio = 0.0

        print(f"   📊 R:R ratio = {rr_ratio:.2f}:1")

        print(f"   ⏱️  Trade horizon: {trade_horizon.upper()} — {horizon_reason[:60]}")

        return {
            "signal":           signal,
            "confidence":       confidence,
            "entry_zone":       str(result.get("entry_zone", f"${price:.4f}")),
            "targets":          targets,
            "stop_loss":        stop_loss,
            "reasoning":        str(result.get("reasoning", "")),
            "action_plan":      str(result.get("action_plan", "")),
            "rr_ratio":         rr_ratio,
            "trade_horizon":    trade_horizon,
            "horizon_reasoning": horizon_reason,
        }

    except json.JSONDecodeError as e:
        print(f"⚠️  [Analyzer] JSON parse error: {e}")
        return _fallback(price)
    except Exception as e:
        print(f"❌ [Analyzer] Claude API error: {e}")
        return _fallback(price)


def _fallback(price: float) -> dict:
    return {
        "signal":            "HOLD",
        "confidence":        0,
        "entry_zone":        f"${price:.4f}",
        "targets":           [round(price * 1.05, 4), round(price * 1.10, 4), round(price * 1.20, 4)],
        "stop_loss":         round(price * 0.95, 4),
        "reasoning":         "Analysis unavailable — defaulting to HOLD.",
        "trade_horizon":     "swing",
        "horizon_reasoning": "",
    }
