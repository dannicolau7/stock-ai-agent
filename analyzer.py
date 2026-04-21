"""
analyzer.py — calls Claude with full market context using a dual bull/bear debate.

Two adversarial Claude calls run in parallel:
  - Bull analyst: finds every reason to BUY (rates conviction 0-100)
  - Bear analyst: finds every reason to AVOID (rates risk 0-100)

Decision: net = bull_score - bear_score
  net ≥ +30 → BUY  (confidence = bull_score)
  net ≤ -30 → SELL (confidence = bear_score)
  otherwise → HOLD

Unique scoring framework includes:
  OBV / Smart Money Divergence / EMA Stack / Float Rotation /
  Social Velocity / Sector Momentum / Catalyst Timing /
  Pre-Market Gap / Earnings Proximity / Self-Learning weights
"""

import json
from concurrent.futures import ThreadPoolExecutor

import anthropic
from langsmith import traceable

from config import ANTHROPIC_API_KEY
from self_learner import get_weight_adjustments
import world_context as wctx
from intelligence_hub import hub

_client = None

# Gap between bull and bear required to trigger a signal
DEBATE_THRESHOLD = 30


def _enforce_target_spacing(targets: list, price: float, min_gap: float = 0.03) -> list:
    """
    Ensure T1, T2, T3 are strictly increasing with at least min_gap (3%) between each.
    If Claude returns duplicates or inverted values, rebuild T2/T3 from T1.
    """
    if not targets:
        return [round(price * 1.05, 4), round(price * 1.10, 4), round(price * 1.20, 4)]

    t1 = targets[0] if targets[0] > price else price * 1.05

    if len(targets) > 1 and targets[1] >= t1 * (1 + min_gap):
        t2 = targets[1]
    else:
        t2 = round(t1 * (1 + max(min_gap, 0.08)), 4)

    if len(targets) > 2 and targets[2] >= t2 * (1 + min_gap):
        t3 = targets[2]
    else:
        t3 = round(t2 * (1 + max(min_gap, 0.10)), 4)

    return [round(t1, 4), round(t2, 4), round(t3, 4)]


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _call_claude_raw(prompt: str, max_tokens: int = 600) -> dict:
    """
    Single blocking Claude API call → parsed JSON dict.
    Returns {} on any error (timeout, parse failure, API error).
    """
    try:
        client   = _get_client()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Strip markdown fences
        if text.startswith("```"):
            parts = text.split("```")
            text  = parts[1][4:] if parts[1].startswith("json") else parts[1]
        text = text.strip()
        # Find first {...} block
        if not text.startswith("{"):
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
        return json.loads(text)
    except Exception as e:
        print(f"⚠️  [Analyzer] Claude call failed: {e}")
        return {}


def _build_context_block(
    context: dict,
    *,
    earnings_cap: int,
    portfolio_section: str,
    pre_section: str,
    regime_section: str,
    weights: dict,
) -> str:
    """
    Build the shared technical/fundamental context string used by both bull and bear prompts.
    Returns everything from SECTION 1 through SELF-LEARNED SIGNAL WEIGHTS.
    """
    ticker         = context.get("ticker", "UNKNOWN")
    price          = context.get("current_price", 0.0)
    prev_close     = context.get("prev_close", price)
    volume_spike   = context.get("volume_spike", False)
    vol_ratio      = context.get("volume_spike_ratio", 1.0)
    rsi            = context.get("rsi", 50.0)
    macd           = context.get("macd", {})
    bollinger      = context.get("bollinger", {})
    atr            = context.get("atr", 0.0)
    support        = context.get("support", 0.0)
    resistance     = context.get("resistance", 0.0)
    vwap           = context.get("vwap", 0.0)
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

    # Self-learning weight labels
    w_macd       = weights.get("macd", 1.0)
    w_rsi_bounce = weights.get("rsi_bounce", 1.0)
    w_rsi_mom    = weights.get("rsi_momentum", 1.0)
    w_vwap       = weights.get("vwap", 1.0)
    w_volume     = weights.get("volume", 1.0)
    w_bb         = weights.get("bollinger", 1.0)
    w_sentiment  = weights.get("sentiment", 1.0)
    w_support    = weights.get("support", 1.0)
    w_ema        = weights.get("ema_stack", 1.0)
    w_smd        = weights.get("smart_money", 1.0)
    w_float      = weights.get("float_rot", 1.0)
    w_gap        = weights.get("gap", 1.0)

    # Bollinger position
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

    t_mult = timing.get("multiplier", 1.0)
    t_win  = timing.get("window", "unknown")

    float_label = ""
    if float_rot > 100:
        float_label = f"🚀 EXTREME ({float_rot:.0f}% of float)"
    elif float_rot > 50:
        float_label = f"🔥 MAJOR ({float_rot:.0f}% of float)"
    elif float_rot > 20:
        float_label = f"elevated ({float_rot:.0f}% of float)"
    elif float_rot > 0:
        float_label = f"{float_rot:.0f}% of float"

    patterns_str = (
        ",  ".join(f"{p['pattern']} ({p['confidence']:.0%}) — {p['description']}"
                   for p in patterns)
        if patterns else "None detected"
    )

    fired_str  = "  ".join(f"{s[0]} {s[1]}" for s in score_bd.get("fired", []))
    missed_str = ", ".join(score_bd.get("missed", [])[:6])
    score_line = (
        f"Pre-computed raw={score_bd.get('raw_score',0)} "
        f"× {score_bd.get('timing_mult',1.0)} timing "
        f"= {score_bd.get('final_score',0)} pts\n"
        f"  Fired: {fired_str or 'none'}\n"
        f"  Missed: {missed_str or 'none'}"
    )

    sr_str = (
        f"  5-day:  support ${sr_levels.get('support_5d',0):.4f}  "
        f"resist ${sr_levels.get('resistance_5d',0):.4f}\n"
        f"  10-day: support ${sr_levels.get('support_10d',0):.4f}  "
        f"resist ${sr_levels.get('resistance_10d',0):.4f}\n"
        f"  20-day: support ${sr_levels.get('support_20d',0):.4f}  "
        f"resist ${sr_levels.get('resistance_20d',0):.4f}"
    ) if sr_levels else f"  Support ${support:.4f}  Resistance ${resistance:.4f}"

    bull_sigs = context.get("bullish_signals", [])
    bear_sigs = context.get("bearish_signals", [])
    agreement = context.get("agreement_score", 0.0)
    consensus = context.get("consensus", "NEUTRAL")

    def _sig_lines(sigs: list) -> str:
        return "\n".join(f"  • {n} (weight {w:.2f})" for n, w in sigs) or "  (none)"

    conflict_lines = ""
    if consensus == "BULLISH" and bear_sigs:
        conflict_lines = "\n".join(f"  ⚠️  {n} (weight {w:.2f})" for n, w in bear_sigs)
    elif consensus == "BEARISH" and bull_sigs:
        conflict_lines = "\n".join(f"  ⚠️  {n} (weight {w:.2f})" for n, w in bull_sigs)

    try:
        ctx = wctx.get()
        macro_regime   = ctx.get("macro", {}).get("regime", "UNKNOWN")
        breadth_health = ctx.get("breadth", {}).get("health", "UNKNOWN")
        vix_str        = str(ctx.get("macro", {}).get("vix", "N/A"))
    except Exception:
        macro_regime   = market_regime.get("regime", "UNKNOWN")
        breadth_health = "UNKNOWN"
        vix_str        = "N/A"

    news_catalyst_str = "No recent news"
    if news_summary:
        news_catalyst_str = f"{news_summary[:120]}"
    if context.get("has_edgar_filing"):
        news_catalyst_str = f"EDGAR filing detected. {news_catalyst_str}"

    all_dominant = bull_sigs if consensus in ("BULLISH", "NEUTRAL") else bear_sigs
    top_sig_name = max(all_dominant, key=lambda x: x[1])[0] if all_dominant else "none"

    return f"""{portfolio_section}{pre_section}{regime_section}
=== SECTION 1 — CONSENSUS PICTURE ===
Signal aggregator found:

BULLISH signals ({len(bull_sigs)}):
{_sig_lines(bull_sigs)}

BEARISH signals ({len(bear_sigs)}):
{_sig_lines(bear_sigs)}

Agreement score: {agreement:.0f}% {consensus}
Strongest signal: {top_sig_name}

=== SECTION 2 — KEY CONFLICTS TO RESOLVE ===
{"These signals conflict with the consensus and need your judgment:" if conflict_lines else "No significant conflicts — consensus is clean."}
{conflict_lines if conflict_lines else ""}

=== SECTION 3 — MARKET CONTEXT ===
Market regime:  SPY {market_regime.get('regime','?')} (EMA5={market_regime.get('ema5',0):.2f} vs EMA20={market_regime.get('ema20',0):.2f})
SPY today:      {market_regime.get('spy_day_chg',0):+.2f}%
VIX:            {vix_str}
Breadth:        {breadth_health}
Macro bias:     {macro_regime}
Relative str:   {ticker} {rel_str.get('label','n/a')}
{earnings_line}
=== SECTION 4 — CATALYST CHECK ===
News catalyst:   {news_catalyst_str}
News sentiment:  {news_sentiment} ({sentiment_score}/100)
Social velocity: {social_vel.get('label', 'n/a')}
Volume confirm:  {vol_ratio:.1f}x average{'  🔥 SPIKE' if volume_spike else ''}
Float rotation:  {float_label if float_label else 'low'}

=== SUPPORTING TECHNICALS ===
Price:          ${price:.4f}  (prev close ${prev_close:.4f}, day {day_change_pct:+.2f}%)
RSI daily(14):  {rsi:.1f}{'  ⚠️ OVERBOUGHT' if rsi > 70 else '  ⚠️ OVERSOLD' if rsi < 30 else ''}
RSI intraday:   {intra_rsi:.1f} (15-min)
MACD Histogram: {macd.get('histogram', 0):+.6f}{'  🟢' if macd.get('histogram', 0) > 0 else '  🔴'}
BB Position:    {bb_pos}  (bw={bb_bw:.4f})
ATR(14):        ${atr:.4f}
{vwap_line}{ema_line}Smart Money:    {smart_money}
Pre-Market Gap: {gap_info.get('label', 'n/a')}
Sector ETF ({sector_m.get('etf','SPY')}): {sector_m.get('change_pct', 0):+.2f}%  →  {sector_m.get('signal','NEUTRAL')}
Market window:  {t_win}  (score multiplier ×{t_mult})

=== KEY S/R LEVELS ===
{sr_str}

=== CHART PATTERNS ===
{patterns_str}

=== PRE-COMPUTED SCORE ===
{score_line}

{wctx.build_prompt_section()}

=== SELF-LEARNED SIGNAL WEIGHTS (reference) ===
MACD ×{w_macd}  RSI-bounce ×{w_rsi_bounce}  RSI-mom ×{w_rsi_mom}  VWAP ×{w_vwap}
EMA ×{w_ema}  Volume ×{w_volume}  BB ×{w_bb}  SmartMoney ×{w_smd}
Float ×{w_float}  Sentiment ×{w_sentiment}  Gap ×{w_gap}  Support ×{w_support}"""


def _bull_prompt(context_block: str, ticker: str, price: float,
                 e_days: int, earnings_cap: int) -> str:
    """Focused bull-case prompt — returns JSON with trade levels + bull conviction score."""
    return f"""You are a momentum trader making the BULL CASE for this trade.
Your job: find EVERY reason this stock is a strong BUY right now.
Be specific — cite exact RSI values, MACD direction, volume ratio, catalyst, chart pattern.
Ignore bearish signals entirely; your role is to argue FOR the trade.

{context_block}

=== YOUR MISSION ===
Build the BULL CASE for {ticker} at ${price:.4f}.
Rate your conviction 0–100 where 100 = screaming buy, 0 = no bull case at all.

RULES:
- Earnings ≤{e_days} days away → cap bull_score at {earnings_cap}
- R/R must be ≥ 1.2:1 or set bull_score ≤ 50
- Trade horizon: "intraday" (pure momentum, ≤1 day) | "swing" (2–5d) | "position" (1w+)

Respond ONLY with this exact JSON (no markdown fences):
{{
  "bull_score": <int 0-100>,
  "bull_summary": "<2 sentences: core thesis + key catalyst>",
  "bull_factors": ["<factor1>", "<factor2>", "<factor3>"],
  "entry_low": <float>,
  "entry_high": <float>,
  "stop_loss": <float>,
  "stop_pct": <float>,
  "target_1": <float>,
  "target_2": <float>,
  "trade_horizon": "intraday" or "swing" or "position",
  "horizon_reasoning": "<1 sentence: key factor for timeframe>",
  "action_plan": "<step-by-step: entry, targets, exit, what invalidates>",
  "top_3_signals": ["<signal1>", "<signal2>", "<signal3>"]
}}"""


def _bear_prompt(context_block: str, ticker: str, price: float) -> str:
    """Focused bear-case prompt — returns JSON with risk score only (no trade levels)."""
    return f"""You are a skeptical risk manager making the BEAR CASE against this trade.
Your job: find EVERY reason to AVOID this stock right now.
Be harsh. Be a devil's advocate. Find every red flag, headwind, and risk.
Ignore bullish signals entirely; your role is to argue AGAINST the trade.

{context_block}

=== YOUR MISSION ===
Build the BEAR CASE against {ticker} at ${price:.4f}.
Rate the risk level 0–100 where 100 = extremely dangerous, definitely avoid.

Respond ONLY with this exact JSON (no markdown fences):
{{
  "bear_score": <int 0-100>,
  "bear_summary": "<2 sentences: core risk thesis + biggest red flag>",
  "bear_factors": ["<risk1>", "<risk2>", "<risk3>"],
  "main_risk": "<1 sentence: single biggest risk>"
}}"""


@traceable(name="claude_analyzer", tags=["claude", "llm"])
def analyze_market(context: dict) -> dict:
    # ── Skip-Claude fast path (aggregator disagreement) ────────────────────────
    if context.get("skip_claude"):
        skip_reason = context.get("skip_reason", "agents disagree")
        print(f"⏭️  [Analyzer] Skipping Claude — {skip_reason}")
        return _fallback(context.get("current_price", 0.0))

    ticker        = context.get("ticker", "UNKNOWN")
    price         = context.get("current_price", 0.0)
    atr           = context.get("atr", 0.0)
    earnings_info = context.get("earnings_info", {})
    timing        = context.get("timing", {})
    vol_ratio     = context.get("volume_spike_ratio", 1.0)

    # ── Self-learning weights ──────────────────────────────────────────────────
    weights = get_weight_adjustments()

    # ── Earnings confidence cap ────────────────────────────────────────────────
    e_days       = earnings_info.get("days_to_earnings", 999)
    earnings_cap = 100
    if e_days <= 3:
        earnings_cap = 55
    elif e_days <= 7:
        earnings_cap = 65
    elif e_days <= 14:
        earnings_cap = 75

    # ── Hub context sections ───────────────────────────────────────────────────
    portfolio = hub.get_portfolio_context(ticker)
    if portfolio.get("already_open"):
        portfolio_section = (
            f"\n=== PORTFOLIO CONTEXT ===\n"
            f"⚠️  {ticker} ALREADY IN PORTFOLIO — adding more would double exposure.\n"
            f"Open positions ({portfolio['open_count']}): {', '.join(portfolio['open_tickers'][:5])}\n"
        )
    elif portfolio.get("open_count", 0) > 0:
        portfolio_section = (
            f"\n=== PORTFOLIO CONTEXT ===\n"
            f"Open positions: {portfolio['open_count']} "
            f"({', '.join(portfolio['open_tickers'][:3])})\n"
        )
    else:
        portfolio_section = ""

    tomorrow_setup = hub.get_tomorrow_setup(ticker)
    if tomorrow_setup:
        pre_section = (
            f"\n=== EOD PRE-IDENTIFICATION ===\n"
            f"✅ {ticker} was flagged by yesterday's EOD scanner:\n"
            f"  Setup type: {tomorrow_setup.get('setup_type', '?')}\n"
            f"  EOD score: {tomorrow_setup.get('score', '?')}\n"
            f"  Reason: {tomorrow_setup.get('reason', 'n/a')}\n"
        )
    else:
        pre_section = ""

    thresholds = hub.get_regime_thresholds()
    regime_section = (
        f"\n=== ACTIVE REGIME THRESHOLDS ({thresholds.get('regime', 'NORMAL')}) ===\n"
        f"RSI oversold < {thresholds['rsi_oversold']}  |  overbought > {thresholds['rsi_overbought']}\n"
        f"Volume spike min: {thresholds['volume_spike_min']}×  |  "
        f"Agreement min: {thresholds['agreement_min']}%  |  "
        f"Confidence cap: {thresholds['confidence_cap']}\n"
    )

    # ── Build shared context block ─────────────────────────────────────────────
    context_block = _build_context_block(
        context,
        earnings_cap=earnings_cap,
        portfolio_section=portfolio_section,
        pre_section=pre_section,
        regime_section=regime_section,
        weights=weights,
    )

    bull_p = _bull_prompt(context_block, ticker, price, e_days, earnings_cap)
    bear_p = _bear_prompt(context_block, ticker, price)

    # ── Parallel dual Claude calls ─────────────────────────────────────────────
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_bull = ex.submit(_call_claude_raw, bull_p, 700)
            fut_bear = ex.submit(_call_claude_raw, bear_p, 400)
            bull_raw = fut_bull.result(timeout=90)
            bear_raw = fut_bear.result(timeout=90)
    except Exception as e:
        print(f"❌ [Analyzer] Parallel call error: {e}")
        return _fallback(price)

    # ── Parse scores ───────────────────────────────────────────────────────────
    bull_score   = max(0, min(100, int(bull_raw.get("bull_score", 0))))
    bear_score   = max(0, min(100, int(bear_raw.get("bear_score", 50))))  # cautious default
    bull_summary = str(bull_raw.get("bull_summary", ""))[:120]
    bear_summary = str(bear_raw.get("bear_summary", ""))[:120]
    net_score    = bull_score - bear_score
    print(f"   🟢 Bull {bull_score}  🔴 Bear {bear_score}  net {net_score:+d}")

    # ── Decision ───────────────────────────────────────────────────────────────
    if net_score >= DEBATE_THRESHOLD:
        signal, confidence = "BUY",  bull_score
    elif -net_score >= DEBATE_THRESHOLD:
        signal, confidence = "SELL", bear_score
    else:
        signal, confidence = "HOLD", 0

    # ── ATR fallback reference ─────────────────────────────────────────────────
    _fb_atr = atr if atr > 0 else price * 0.015

    # ── Entry / stop / targets (from bull call for BUY; ATR fallback otherwise) ─
    if signal == "BUY" and bull_raw:
        ez_low        = float(bull_raw.get("entry_low",  0) or 0)
        ez_high       = float(bull_raw.get("entry_high", 0) or 0)
        stop_loss     = float(bull_raw.get("stop_loss",  0) or 0)
        t1            = float(bull_raw.get("target_1",   0) or 0)
        t2            = float(bull_raw.get("target_2",   0) or 0)
        trade_horizon = str(bull_raw.get("trade_horizon", "swing")).lower()
        horizon_reason= str(bull_raw.get("horizon_reasoning", ""))
        action_plan   = str(bull_raw.get("action_plan", ""))
        top_3_signals = [str(s) for s in bull_raw.get("top_3_signals", [])][:3]
    else:
        ez_low = ez_high = stop_loss = t1 = t2 = 0.0
        trade_horizon  = "swing"
        horizon_reason = ""
        action_plan    = ""
        top_3_signals  = []

    # ── Validate / fix entry zone ──────────────────────────────────────────────
    if ez_low <= 0 or ez_high <= 0 or ez_low >= ez_high:
        print(f"⚠️  [Analyzer] Entry zone invalid — recalculating from ATR")
        ez_low  = round(price - 0.5 * _fb_atr, 2)
        ez_high = round(price + 0.5 * _fb_atr, 2)
    entry_zone = f"${ez_low:.2f} - ${ez_high:.2f}"

    # ── Validate / fix stop loss ───────────────────────────────────────────────
    if stop_loss <= 0:
        stop_loss = round(price - 2.0 * _fb_atr, 4)
    if atr > 0:
        min_stop = round(price - atr, 4)
        if stop_loss > min_stop:
            print(f"⚠️  [Analyzer] Stop widened: ${stop_loss:.4f} → ${min_stop:.4f} (1× ATR floor)")
            stop_loss = min_stop
    if stop_loss >= price:
        stop_loss = round(price - 2.0 * _fb_atr, 4)
        print(f"⚠️  [Analyzer] Stop was ≥ price — fixed to ${stop_loss:.4f}")
    stop_pct = round((stop_loss - price) / price * 100, 1) if price > 0 else 0.0
    if abs(stop_pct) < 0.5:
        stop_loss = round(price - 2.0 * _fb_atr, 4)
        stop_pct  = round((stop_loss - price) / price * 100, 1)
        print(f"⚠️  [Analyzer] Stop pct < 0.5% — fixed to ${stop_loss:.4f} ({stop_pct:.1f}%)")

    # ── Validate / fix targets ─────────────────────────────────────────────────
    raw_targets = [t1, t2, round(t2 * 1.10, 4) if t2 > 0 else 0.0]
    targets = _enforce_target_spacing(raw_targets, price)

    # ── Trade horizon ──────────────────────────────────────────────────────────
    if trade_horizon not in ("intraday", "swing", "position"):
        trade_horizon = "swing"

    # ── Earnings confidence cap ────────────────────────────────────────────────
    if confidence > earnings_cap:
        confidence = earnings_cap
        print(f"⚠️  [Analyzer] Earnings cap applied → capped at {earnings_cap}")

    # ── Volume cap ────────────────────────────────────────────────────────────
    if vol_ratio > 0 and vol_ratio < 0.5:
        confidence = min(confidence, 60)

    # ── After-hours BUY suppression ───────────────────────────────────────────
    t_mult     = timing.get("multiplier", 1.0)
    _has_edgar = context.get("has_edgar_filing", False)
    _is_8k     = context.get("edgar_filing_type", "") == "8-K"
    _news_trig = context.get("news_triggered", False)
    _major_cat = context.get("major_catalyst", False)
    _suppress  = (signal == "BUY" and t_mult == 0.0)
    _bypass    = (_has_edgar and _is_8k) or (_news_trig and _major_cat)

    if _suppress and not _bypass:
        print(f"🔕 [Analyzer] After-hours BUY suppressed (timing_mult=0, no major catalyst)")
        signal     = "HOLD"
        confidence = 0
    elif _suppress and _bypass:
        cat_label = context.get("edgar_filing_type", "catalyst")
        print(f"🚨 [Analyzer] Major catalyst ({cat_label}) — bypassing after-hours suppression")

    # ── Circuit breaker ────────────────────────────────────────────────────────
    if signal == "BUY":
        try:
            from circuit_breaker import check_market
            spy_chg = context.get("market_regime", {}).get("spy_day_chg")
            cb = check_market(spy_day_chg=spy_chg)
            if not cb["safe"]:
                print(f"🚫 [Analyzer] Circuit breaker triggered: {cb['reason']}")
                signal     = "HOLD"
                confidence = 0
        except Exception as _cb_err:
            print(f"⚠️  [Analyzer] Circuit breaker check failed (fail-open): {_cb_err}")

    # ── R:R ratio ─────────────────────────────────────────────────────────────
    try:
        entry_mid = (ez_low + ez_high) / 2
        t1_val    = targets[0] if targets else price * 1.05
        risk      = entry_mid - stop_loss
        reward    = t1_val - entry_mid
        rr_ratio  = round(reward / risk, 2) if risk > 0 else 0.0
    except Exception:
        rr_ratio = 0.0

    print(f"   📊 R:R ratio = {rr_ratio:.2f}:1")
    if horizon_reason:
        print(f"   ⏱️  Trade horizon: {trade_horizon.upper()} — {horizon_reason[:60]}")

    # ── Combined reasoning ─────────────────────────────────────────────────────
    reasoning = (
        f"🟢 Bull ({bull_score}): {bull_summary}\n"
        f"🔴 Bear ({bear_score}): {bear_summary}\n"
        f"Decision: {signal} (net {net_score:+d})"
    )

    return {
        "signal":            signal,
        "confidence":        confidence,
        "entry_zone":        entry_zone,
        "entry_low":         round(ez_low, 4),
        "entry_high":        round(ez_high, 4),
        "targets":           targets,
        "stop_loss":         stop_loss,
        "stop_pct":          stop_pct,
        "reasoning":         reasoning,
        "action_plan":       action_plan,
        "rr_ratio":          rr_ratio,
        "trade_horizon":     trade_horizon,
        "horizon_reasoning": horizon_reason,
        "main_risk":         str(bear_raw.get("main_risk", "")),
        "top_3_signals":     top_3_signals,
        # ── Dual-call debate scores ────────────────────────────────────────────
        "bull_score":        bull_score,
        "bear_score":        bear_score,
        "bull_summary":      bull_summary,
        "bear_summary":      bear_summary,
        "net_score":         net_score,
    }


def _fallback(price: float) -> dict:
    return {
        "signal":            "HOLD",
        "confidence":        0,
        "entry_zone":        f"${price:.4f}",
        "entry_low":         price,
        "entry_high":        price,
        "targets":           [round(price * 1.05, 4), round(price * 1.10, 4), round(price * 1.20, 4)],
        "stop_loss":         round(price * 0.95, 4),
        "stop_pct":          -5.0,
        "reasoning":         "Analysis unavailable — defaulting to HOLD.",
        "action_plan":       "",
        "rr_ratio":          0.0,
        "trade_horizon":     "swing",
        "horizon_reasoning": "",
        "main_risk":         "",
        "top_3_signals":     [],
        "bull_score":        0,
        "bear_score":        50,
        "bull_summary":      "",
        "bear_summary":      "",
        "net_score":         -50,
    }
