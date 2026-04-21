"""
agents/pattern_agent.py — "Beaten-down + catalyst" recovery pattern detector.

Primary use case: stocks like HIMS that sold off 30–60% from peak, then gap
up on news (FDA, partnership, earnings beat) — best risk/reward for
short-duration swing trades.

detect_recovery_pattern(ticker, current_price, ohlcv_30d) → dict:
  recovery_score    int   0-100 — how clean the beaten-down + catalyst setup is
  exhaustion_score  int   0-100 — risk of reversal (sell-the-news, overbought)
  pattern_valid     bool  True when either score ≥ 60
  pattern_type      str   "RECOVERY" | "EXHAUSTION" | "MIXED" | "NONE"
  entry_zone        list  [low, high] — suggested limit-order range
  target_price      float — first price target (~4× ATR above entry)
  stop_loss         float — hard stop (~2× ATR below entry)
  hold_days         int   2 for RECOVERY, 1 for EXHAUSTION
  reasoning         list  — human-readable explanation of each scored component
"""

import math
import numpy as np
from langsmith import traceable

RECOVERY_THRESHOLD   = 60
EXHAUSTION_THRESHOLD = 60


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)


def _calc_bollinger_upper(closes: np.ndarray, period: int = 20) -> float:
    if len(closes) < period:
        return _safe_float(closes[-1]) if len(closes) else 0.0
    recent = closes[-period:]
    return float(np.mean(recent)) + 2.0 * float(np.std(recent, ddof=0))


def _calc_atr_from_bars(bars: list, period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h  = _safe_float(bars[i].get("h",  bars[i].get("high",  0)))
        l  = _safe_float(bars[i].get("l",  bars[i].get("low",   0)))
        pc = _safe_float(bars[i-1].get("c", bars[i-1].get("close", 0)))
        if h > 0 and l > 0 and pc > 0:
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    recent = trs[-period:]
    return round(sum(recent) / len(recent), 4) if recent else 0.0


# ── Recovery score (beaten-down + catalyst) ────────────────────────────────────

def _recovery_score(
    prior_30d_return: float,
    dist_from_52w_high_pct: float,
    rsi: float,
    gap_up_pct: float,
    rel_vol: float,
) -> tuple[int, list[str]]:
    """
    Score a beaten-down + catalyst setup.
    Maximum raw total = 100 (25+20+20+20+15).
    """
    score   = 0
    reasons = []

    # 1. Stock was beaten down (sold off before today)
    if prior_30d_return < -10:
        score += 25
        reasons.append(f"Beaten down {prior_30d_return:.1f}% in 30d (+25)")
    elif prior_30d_return < -5:
        score += 10
        reasons.append(f"Partially sold off {prior_30d_return:.1f}% in 30d (+10)")

    # 2. Far from 52-week high — still in accumulation zone, not a top
    if dist_from_52w_high_pct > 20:
        score += 20
        reasons.append(f"{dist_from_52w_high_pct:.0f}% below 52w high — room to recover (+20)")
    elif dist_from_52w_high_pct > 10:
        score += 10
        reasons.append(f"{dist_from_52w_high_pct:.0f}% below 52w high (+10)")

    # 3. RSI not overbought (still has room to run)
    if rsi < 50:
        score += 20
        reasons.append(f"RSI {rsi:.0f} — oversold, momentum room (+20)")
    elif rsi < 65:
        score += 20
        reasons.append(f"RSI {rsi:.0f} — not overbought (+20)")
    elif rsi < 70:
        score += 10
        reasons.append(f"RSI {rsi:.0f} — approaching overbought (+10)")

    # 4. Catalyst gap-up today (5–20% is the sweet spot)
    if 5 <= gap_up_pct <= 20:
        score += 20
        reasons.append(f"Gap up {gap_up_pct:.1f}% — ideal catalyst range (+20)")
    elif 2 <= gap_up_pct < 5:
        score += 10
        reasons.append(f"Gap up {gap_up_pct:.1f}% — moderate catalyst (+10)")
    elif gap_up_pct > 20:
        score += 5
        reasons.append(f"Gap up {gap_up_pct:.1f}% — extended, partial credit (+5)")

    # 5. Volume confirms institutional participation
    if rel_vol >= 2.0:
        score += 15
        reasons.append(f"RVOL {rel_vol:.1f}x — volume confirming move (+15)")
    elif rel_vol >= 1.5:
        score += 8
        reasons.append(f"RVOL {rel_vol:.1f}x — moderate volume (+8)")

    return min(score, 100), reasons


# ── Exhaustion score (extended / sell-the-news risk) ──────────────────────────

def _exhaustion_score(
    prior_30d_return: float,
    rsi: float,
    price: float,
    bb_upper: float,
    volumes: np.ndarray,
) -> tuple[int, list[str]]:
    """
    Score the risk that a stock is extended and likely to reverse.
    Maximum raw total = 90 (30+25+20+15+10 with insider proxy omitted).
    Capped at 100.
    """
    score   = 0
    reasons = []

    # 1. Extended run — sell-the-news risk
    if prior_30d_return > 50:
        score += 30
        reasons.append(f"Up {prior_30d_return:.0f}% in 30d — likely exhausted (+30)")
    elif prior_30d_return > 25:
        score += 15
        reasons.append(f"Up {prior_30d_return:.0f}% in 30d — extended (+15)")

    # 2. Overbought RSI
    if rsi > 80:
        score += 25
        reasons.append(f"RSI {rsi:.0f} — severely overbought (+25)")
    elif rsi > 70:
        score += 25
        reasons.append(f"RSI {rsi:.0f} — overbought (+25)")

    # 3. Price above upper Bollinger Band
    if bb_upper > 0 and price > bb_upper:
        score += 20
        reasons.append(f"Above upper BB ${bb_upper:.2f} — extended (+20)")

    # 4. Volume declining 3 consecutive days (distribution)
    if len(volumes) >= 3:
        last3 = volumes[-3:]
        if last3[-1] < last3[-2] < last3[-3] and last3[-3] > 0:
            score += 15
            reasons.append("Volume declining 3 days — distribution signal (+15)")

    # 5. Insider selling proxy: needs SEC EDGAR Form 4 (not available on free tier)
    #    Left as 0 — override from caller if you have the data.

    return min(score, 100), reasons


# ── Entry / target / stop levels ──────────────────────────────────────────────

def _compute_levels(
    price: float,
    pattern_type: str,
    atr: float,
) -> dict:
    """
    ATR-based entry zone, target, and stop.
    Falls back to fixed % when ATR is unavailable.
    """
    if atr <= 0 or not math.isfinite(atr):
        atr = price * 0.02   # 2% proxy

    if pattern_type == "RECOVERY":
        entry_low  = round(price * 0.99,  2)
        entry_high = round(price * 1.01,  2)
        target     = round(price + 4 * atr, 2)   # ~4× ATR upside
        stop       = round(price - 2 * atr, 2)   # ~2× ATR downside
        hold_days  = 2
    elif pattern_type == "EXHAUSTION":
        entry_low  = round(price * 0.98,  2)
        entry_high = round(price * 1.00,  2)
        target     = round(price * 0.95,  2)     # 5% fade target
        stop       = round(price * 1.03,  2)     # 3% stop above entry
        hold_days  = 1
    else:  # MIXED or NONE
        entry_low  = round(price * 0.98, 2)
        entry_high = round(price * 1.02, 2)
        target     = round(price * 1.10, 2)
        stop       = round(price * 0.95, 2)
        hold_days  = 2

    return {
        "entry_zone":   [entry_low, entry_high],
        "target_price": target,
        "stop_loss":    stop,
        "hold_days":    hold_days,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

@traceable(name="pattern_agent", tags=["pipeline", "patterns"])
def detect_recovery_pattern(
    ticker: str,
    current_price: float,
    ohlcv_30d: list[dict],
    *,
    rel_vol: float = 1.0,
    gap_up_pct: float | None = None,
    high_52w: float | None = None,
) -> dict:
    """
    Detect the "beaten-down + catalyst" recovery pattern.

    Parameters
    ----------
    ticker        : stock symbol (for logging only)
    current_price : latest trade price
    ohlcv_30d     : list of OHLCV bar dicts with keys h/high, l/low, c/close,
                    v/volume (20–30 daily bars expected)
    rel_vol       : relative volume ratio (today / avg daily), default 1.0
    gap_up_pct    : today's gap from prior close (%); computed from bars if None
    high_52w      : 52-week high; falls back to max high in ohlcv_30d if None

    Returns
    -------
    dict:
      recovery_score    int
      exhaustion_score  int
      pattern_valid     bool
      pattern_type      str   "RECOVERY" | "EXHAUSTION" | "MIXED" | "NONE"
      entry_zone        list[float]
      target_price      float
      stop_loss         float
      hold_days         int
      reasoning         list[str]
    """
    _empty = {
        "recovery_score":   0,
        "exhaustion_score": 0,
        "pattern_valid":    False,
        "pattern_type":     "NONE",
        "entry_zone":       [round(current_price * 0.98, 2),
                             round(current_price * 1.02, 2)],
        "target_price":     round(current_price * 1.10, 2),
        "stop_loss":        round(current_price * 0.95, 2),
        "hold_days":        2,
        "reasoning":        ["Insufficient data"],
    }

    if not ohlcv_30d or len(ohlcv_30d) < 5:
        return _empty

    try:
        # ── Extract arrays ─────────────────────────────────────────────────────
        closes  = np.array([
            _safe_float(b.get("c", b.get("close", 0))) for b in ohlcv_30d
        ], dtype=float)
        highs   = np.array([
            _safe_float(b.get("h", b.get("high", 0)))  for b in ohlcv_30d
        ], dtype=float)
        volumes = np.array([
            _safe_float(b.get("v", b.get("volume", 0))) for b in ohlcv_30d
        ], dtype=float)

        if closes[-1] <= 0 or not np.any(closes > 0):
            return _empty

        # ── Core metrics ───────────────────────────────────────────────────────
        price_30d_ago  = _safe_float(closes[0], current_price)
        prior_30d_ret  = ((current_price - price_30d_ago) / price_30d_ago * 100
                          if price_30d_ago > 0 else 0.0)

        valid_highs    = highs[highs > 0]
        _52w_high      = high_52w or (float(np.max(valid_highs))
                                      if len(valid_highs) else current_price)
        dist_pct       = ((_52w_high - current_price) / _52w_high * 100
                          if _52w_high > current_price else 0.0)

        rsi            = _calc_rsi(closes)
        bb_upper       = _calc_bollinger_upper(closes)

        if gap_up_pct is None:
            prev_close = _safe_float(closes[-2] if len(closes) >= 2 else closes[-1])
            gap_up_pct = ((current_price - prev_close) / prev_close * 100
                          if prev_close > 0 else 0.0)

        atr = _calc_atr_from_bars(ohlcv_30d)

        # ── Score both patterns ────────────────────────────────────────────────
        rec_score, rec_reasons = _recovery_score(
            prior_30d_ret, dist_pct, rsi, gap_up_pct, rel_vol
        )
        exh_score, exh_reasons = _exhaustion_score(
            prior_30d_ret, rsi, current_price, bb_upper, volumes
        )

        # ── Determine dominant pattern ─────────────────────────────────────────
        rec_valid = rec_score >= RECOVERY_THRESHOLD
        exh_valid = exh_score >= EXHAUSTION_THRESHOLD

        if rec_valid and exh_valid:
            pattern_type = "MIXED"
        elif rec_valid:
            pattern_type = "RECOVERY"
        elif exh_valid:
            pattern_type = "EXHAUSTION"
        else:
            pattern_type = "NONE"

        pattern_valid = pattern_type != "NONE"
        levels        = _compute_levels(current_price, pattern_type, atr)
        reasoning     = rec_reasons + exh_reasons or [
            f"Below threshold (rec={rec_score} exh={exh_score})"
        ]

        print(
            f"🔷 [PatternAgent] {ticker}  "
            f"rec={rec_score}  exh={exh_score}  → {pattern_type}  "
            f"valid={pattern_valid}"
        )

        return {
            "recovery_score":   rec_score,
            "exhaustion_score": exh_score,
            "pattern_valid":    pattern_valid,
            "pattern_type":     pattern_type,
            **levels,
            "reasoning":        reasoning,
        }

    except Exception as e:
        print(f"❌ [PatternAgent] {ticker} error: {e}")
        return _empty


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import yfinance as yf

    ticker = sys.argv[1] if len(sys.argv) > 1 else "HIMS"

    df = yf.Ticker(ticker).history(period="35d", interval="1d")
    if df.empty:
        print(f"No data for {ticker}")
        sys.exit(1)

    bars = [
        {"h": row.High, "l": row.Low, "c": row.Close, "v": row.Volume}
        for _, row in df.iterrows()
    ]
    price      = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2])
    gap_pct    = (price - prev_close) / prev_close * 100
    avg_vol    = float(df["Volume"].mean())
    rvol       = float(df["Volume"].iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

    result = detect_recovery_pattern(
        ticker, price, bars,
        rel_vol=rvol,
        gap_up_pct=gap_pct,
    )

    print(f"\n=== PatternAgent: {ticker} @ ${price:.2f} ===")
    print(f"  Pattern:  {result['pattern_type']}  (valid={result['pattern_valid']})")
    print(f"  Recovery score:   {result['recovery_score']}")
    print(f"  Exhaustion score: {result['exhaustion_score']}")
    print(f"  Entry zone: ${result['entry_zone'][0]} – ${result['entry_zone'][1]}")
    print(f"  Target:     ${result['target_price']}")
    print(f"  Stop:       ${result['stop_loss']}")
    print(f"  Hold:       {result['hold_days']}d")
    print(f"\nReasoning:")
    for r in result["reasoning"]:
        print(f"  • {r}")
