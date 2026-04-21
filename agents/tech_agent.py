"""
tech_agent.py — LangGraph node: RSI, MACD, Bollinger Bands, ATR, VWAP,
OBV, Smart Money Divergence, Float Rotation, EMA Stack (9/21/50),
Sector Momentum, Catalyst Timing, Pre-Market Gap Intelligence.
"""

import math
import numpy as np
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import yfinance as yf
from langsmith import traceable
from utils.tracing import annotate_run

EST = ZoneInfo("America/New_York")

# ── Sector ETF map ─────────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Technology":          "XLK",
    "Communication":       "XLC",
    "Consumer Cyclical":   "XLY",
    "Consumer Defensive":  "XLP",
    "Healthcare":          "XLV",
    "Financials":          "XLF",
    "Industrials":         "XLI",
    "Basic Materials":     "XLB",
    "Real Estate":         "XLRE",
    "Utilities":           "XLU",
    "Energy":              "XLE",
    "Semiconductor":       "SOXX",
    "Biotechnology":       "XBI",
    "AI":                  "AIQ",
}


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _sanitize(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """
    Forward-fill NaN / Inf values in a price or volume array.

    Uses the preceding finite value so indicator calculations see a smooth
    series rather than zeros (which would distort RSI / MACD / OBV).
    Falls back to `fill` only at position 0 when there is no prior value.
    """
    result = arr.copy()
    for i in range(len(result)):
        if not np.isfinite(result[i]):
            result[i] = result[i - 1] if i > 0 and np.isfinite(result[i - 1]) else fill
    return result


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    k      = 2 / (period + 1)
    result = [float(arr[0])]
    for v in arr[1:]:
        result.append(float(v) * k + result[-1] * (1 - k))
    return np.array(result)


def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    if not np.isfinite(avg_gain) or not np.isfinite(avg_loss):
        return 50.0
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0   # no price movement — RSI undefined, use neutral
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _calc_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9) -> dict:
    if len(closes) < slow + sig:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
    macd_line   = _ema(closes, fast) - _ema(closes, slow)
    signal_line = _ema(macd_line, sig)
    histogram   = macd_line - signal_line
    return {
        "macd":      round(float(macd_line[-1]),   6),
        "signal":    round(float(signal_line[-1]), 6),
        "histogram": round(float(histogram[-1]),   6),
    }


def _calc_bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> dict:
    if len(closes) < period:
        last = float(closes[-1]) if len(closes) else 0.0
        return {"upper": last, "middle": last, "lower": last, "bandwidth": 0.0}
    recent = closes[-period:]
    middle = float(np.mean(recent))
    std    = float(np.std(recent, ddof=0))
    upper  = round(middle + num_std * std, 6)
    lower  = round(middle - num_std * std, 6)
    middle = round(middle, 6)
    bw     = round((upper - lower) / middle if middle else 0.0, 6)
    return {"upper": upper, "middle": middle, "lower": lower, "bandwidth": bw}


def _calc_atr(bars: list, period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, len(bars)):
        h  = float(bars[i]["h"])
        l  = float(bars[i]["l"])
        pc = float(bars[i - 1]["c"])
        if math.isfinite(h) and math.isfinite(l) and math.isfinite(pc):
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    recent = trs[-period:]
    return round(sum(recent) / len(recent), 6) if recent else 0.0


def _calc_vwap(intraday_bars: list) -> float:
    """VWAP = Σ(typical_price × volume) / Σ(volume) from today's intraday bars."""
    if not intraday_bars:
        return 0.0
    try:
        cum_pv = cum_v = 0.0
        for b in intraday_bars:
            h, l, c = float(b["h"]), float(b["l"]), float(b["c"])
            vol = float(b["v"])
            if not (math.isfinite(h) and math.isfinite(l) and
                    math.isfinite(c) and math.isfinite(vol)):
                continue
            cum_pv += (h + l + c) / 3 * vol
            cum_v  += vol
        return round(cum_pv / cum_v, 6) if cum_v > 0 else 0.0
    except Exception:
        return 0.0


def _calc_obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """On-Balance Volume — cumulative volume direction indicator."""
    obv    = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + float(volumes[i]))
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - float(volumes[i]))
        else:
            obv.append(obv[-1])
    return np.array(obv)


def _smart_money_divergence(closes: np.ndarray, volumes: np.ndarray,
                             lookback: int = 5) -> str:
    """
    Detects hidden accumulation or distribution over the last `lookback` bars.
    ACCUMULATION: price down but OBV up  → institutions buying quietly (bullish)
    DISTRIBUTION: price up but OBV down → institutions selling into rally (bearish)
    NEUTRAL:      price and OBV agree
    """
    if len(closes) < lookback + 1:
        return "NEUTRAL"
    obv          = _calc_obv(closes, volumes)
    price_change = float(closes[-1]) - float(closes[-lookback])
    obv_change   = float(obv[-1])   - float(obv[-lookback])
    if price_change < 0 and obv_change > 0:
        return "ACCUMULATION"   # bullish hidden signal
    if price_change > 0 and obv_change < 0:
        return "DISTRIBUTION"   # bearish hidden signal
    return "NEUTRAL"


def _ema_stack(closes: np.ndarray) -> dict:
    """
    EMA 9 / 21 / 50 stack.
    Returns alignment label and individual EMA values.
    BULLISH:  EMA9 > EMA21 > EMA50
    BEARISH:  EMA9 < EMA21 < EMA50
    MIXED:    anything else
    """
    if len(closes) < 50:
        return {"alignment": "MIXED", "ema9": 0.0, "ema21": 0.0, "ema50": 0.0}
    ema9  = float(_ema(closes, 9)[-1])
    ema21 = float(_ema(closes, 21)[-1])
    ema50 = float(_ema(closes, 50)[-1])
    if ema9 > ema21 > ema50:
        alignment = "BULLISH"
    elif ema9 < ema21 < ema50:
        alignment = "BEARISH"
    else:
        alignment = "MIXED"
    return {
        "alignment": alignment,
        "ema9":  round(ema9,  4),
        "ema21": round(ema21, 4),
        "ema50": round(ema50, 4),
    }


def _fetch_hourly_bars(ticker: str) -> list[dict]:
    """
    Fetch last 5 days of 1-hour bars via yfinance.
    Returns list of {"o","h","l","c","v"} dicts, or [] on failure.
    """
    try:
        df = yf.Ticker(ticker).history(period="5d", interval="1h")
        if df.empty:
            return []
        bars = []
        for ts, row in df.iterrows():
            bars.append({
                "o": float(row["Open"]),
                "h": float(row["High"]),
                "l": float(row["Low"]),
                "c": float(row["Close"]),
                "v": float(row["Volume"]),
            })
        return bars
    except Exception:
        return []


def _timeframe_signal(closes: np.ndarray, label: str) -> dict:
    """
    Summarise one timeframe as BULLISH / BEARISH / NEUTRAL.

    Criteria — BULLISH (all 3 must hold):
      • MACD histogram > 0
      • RSI ≤ 70  (not overbought)
      • Price > MA20 (short-term uptrend)

    Criteria — BEARISH (all 3 must hold):
      • MACD histogram < 0
      • RSI ≥ 30  (not oversold)
      • Price < MA20

    Anything else → NEUTRAL.
    """
    if len(closes) < 5:
        return {"rsi": 50.0, "macd_hist": 0.0, "ma20": 0.0, "signal": "NEUTRAL", "tf": label}

    rsi       = _calc_rsi(closes)
    macd_d    = _calc_macd(closes)
    hist      = macd_d["histogram"]
    ma20      = float(np.mean(closes[-min(20, len(closes)):])) if len(closes) >= 2 else float(closes[-1])
    price     = float(closes[-1])

    if hist > 0 and rsi <= 70 and price > ma20:
        sig = "BULLISH"
    elif hist < 0 and rsi >= 30 and price < ma20:
        sig = "BEARISH"
    else:
        sig = "NEUTRAL"

    return {
        "rsi":       round(rsi, 1),
        "macd_hist": round(hist, 6),
        "ma20":      round(ma20, 4),
        "price":     round(price, 4),
        "signal":    sig,
        "tf":        label,
    }


def _float_rotation_pct(ticker: str, today_volume: float) -> float:
    """
    Float rotation % = (today_volume / float_shares) × 100.
    >50%  = major event (whole float trading hands)
    >100% = extreme squeeze / breakout
    Returns 0.0 on failure.
    """
    try:
        info   = yf.Ticker(ticker).info
        float_shares = info.get("floatShares") or info.get("sharesOutstanding") or 0
        if float_shares > 0 and today_volume > 0:
            return round((today_volume / float_shares) * 100, 2)
    except Exception:
        pass
    return 0.0


def _sector_momentum(sector: str) -> dict:
    """
    Fetch today's % change for the sector ETF.
    Returns {"etf": str, "change_pct": float, "signal": "BULLISH"|"BEARISH"|"NEUTRAL"}
    """
    etf = SECTOR_ETFS.get(sector, "SPY")
    try:
        tk    = yf.Ticker(etf)
        fi    = tk.fast_info
        price = float(fi.last_price or 0)
        prev  = float(fi.previous_close or price)
        chg   = ((price - prev) / prev * 100) if prev else 0.0
        signal = "BULLISH" if chg >= 1.0 else "BEARISH" if chg <= -1.0 else "NEUTRAL"
        return {"etf": etf, "change_pct": round(chg, 2), "signal": signal}
    except Exception:
        return {"etf": etf, "change_pct": 0.0, "signal": "NEUTRAL"}


def _catalyst_timing_multiplier() -> dict:
    """
    Returns a score multiplier and label based on current EST time.
    Power hour (9:30–10:30): ×1.3   — strongest momentum window
    Midday   (10:30–14:00): ×1.0   — normal
    Afternoon(14:00–15:30): ×1.1   — late momentum possible
    Last hour(15:30–16:00): ×0.7   — avoid (traps/reversals common)
    Pre/after market:        ×0.6
    """
    now   = datetime.now(EST).time()
    open_ = dtime(9, 30)
    ph_end= dtime(10, 30)
    mid   = dtime(14, 0)
    late  = dtime(15, 30)
    close = dtime(16, 0)

    if open_ <= now < ph_end:
        return {"multiplier": 1.3, "window": "power hour 🔥"}
    elif ph_end <= now < mid:
        return {"multiplier": 1.0, "window": "midday"}
    elif mid <= now < late:
        return {"multiplier": 1.1, "window": "afternoon momentum"}
    elif late <= now < close:
        return {"multiplier": 0.7, "window": "last hour ⚠️"}
    else:
        # After-hours / pre-market: multiplier = 0 so swing BUYs are killed entirely.
        # News-triggered signals (spike/edgar/yf-news) bypass this via news_triggered flag.
        return {"multiplier": 0.0, "window": "after hours 🔕"}


def _premarket_gap(bars: list, premarket_price: float = 0.0) -> dict:
    """
    Calculates gap % between pre-market price and prior close.
    Gap >5% with news  → likely continuation (buy)
    Gap >10% no news   → likely fade (avoid or short)
    Gap <0             → gap down (bearish)
    Returns {"gap_pct": float, "signal": str, "label": str}
    """
    if not bars or premarket_price <= 0:
        return {"gap_pct": 0.0, "signal": "NEUTRAL", "label": "no pre-market data"}
    prior_close = float(bars[-1]["c"])
    if prior_close <= 0 or not math.isfinite(prior_close):
        return {"gap_pct": 0.0, "signal": "NEUTRAL", "label": "no prior close"}
    gap_pct = ((premarket_price - prior_close) / prior_close) * 100
    if gap_pct > 10:
        signal, label = "FADE_RISK", f"gap +{gap_pct:.1f}% — likely fade"
    elif gap_pct > 5:
        signal, label = "BULLISH",   f"gap +{gap_pct:.1f}% — continuation likely"
    elif gap_pct > 2:
        signal, label = "BULLISH",   f"gap +{gap_pct:.1f}% — moderate gap up"
    elif gap_pct < -5:
        signal, label = "BEARISH",   f"gap {gap_pct:.1f}% — gap down"
    elif gap_pct < -2:
        signal, label = "BEARISH",   f"gap {gap_pct:.1f}% — moderate gap down"
    else:
        signal, label = "NEUTRAL",   f"gap {gap_pct:.1f}% — flat open"
    return {"gap_pct": round(gap_pct, 2), "signal": signal, "label": label}


def _multi_level_sr(bars: list) -> dict:
    """
    Multi-timeframe support/resistance using 5, 10, and 20 most recent bars.
    Uses the tightest (most recent) levels as primary S/R — much more actionable
    than pulling the min/max over 90 days.
    """
    result = {}
    for days, label in [(5, "5d"), (10, "10d"), (20, "20d")]:
        recent = bars[-min(days, len(bars)):]
        if recent:
            highs = [float(b["h"]) for b in recent if math.isfinite(float(b["h"]))]
            lows  = [float(b["l"]) for b in recent if math.isfinite(float(b["l"]))]
            if highs:
                result[f"resistance_{label}"] = round(max(highs), 4)
            if lows:
                result[f"support_{label}"] = round(min(lows), 4)
    # Primary levels = tightest (5-day)
    result["support"]    = result.get("support_5d",    result.get("support_10d",    0.0))
    result["resistance"] = result.get("resistance_5d", result.get("resistance_10d", 0.0))
    return result


def _market_regime() -> dict:
    """
    Check SPY EMA5 vs EMA20 to determine overall market direction.
    BULL: EMA5 > EMA20 (uptrend)
    BEAR: EMA5 < EMA20 (downtrend)
    """
    try:
        df        = yf.Ticker("SPY").history(period="30d", interval="1d")
        spy_arr   = df["Close"].values.astype(float)
        spy_price = float(spy_arr[-1])
        spy_prev  = float(spy_arr[-2])
        day_chg   = (spy_price - spy_prev) / spy_prev * 100
        ema5      = float(_ema(spy_arr, 5)[-1])
        ema20     = float(_ema(spy_arr, 20)[-1])
        regime    = "BULL" if ema5 > ema20 else "BEAR"
        return {
            "regime":      regime,
            "spy_price":   round(spy_price, 2),
            "spy_day_chg": round(day_chg, 2),
            "ema5":        round(ema5, 2),
            "ema20":       round(ema20, 2),
        }
    except Exception:
        return {"regime": "UNKNOWN", "spy_price": 0.0, "spy_day_chg": 0.0,
                "ema5": 0.0, "ema20": 0.0}


def _relative_strength(price: float, prev_close: float) -> dict:
    """
    Compare stock's day change vs SPY day change.
    Positive RS = outperforming market (bullish).
    """
    try:
        fi       = yf.Ticker("SPY").fast_info
        spy_p    = float(fi.last_price or 0)
        spy_prev = float(fi.previous_close or spy_p)
        stk_chg  = (price - prev_close) / prev_close * 100 if prev_close else 0
        spy_chg  = (spy_p - spy_prev)   / spy_prev  * 100 if spy_prev   else 0
        rs       = round(stk_chg - spy_chg, 2)
        if rs > 2:
            label = f"outperforming SPY by +{rs:.1f}% 💪"
        elif rs < -2:
            label = f"underperforming SPY by {rs:.1f}% ⚠️"
        else:
            label = f"in line with SPY ({rs:+.1f}%)"
        return {"rs_vs_spy": rs, "label": label, "spy_chg": round(spy_chg, 2)}
    except Exception:
        return {"rs_vs_spy": 0.0, "label": "n/a", "spy_chg": 0.0}


def _intraday_rsi(intraday_bars: list) -> float:
    """RSI(14) calculated from 15-min intraday bars — more responsive than daily."""
    if len(intraday_bars) < 5:
        return 50.0
    closes = _sanitize(np.array([float(b["c"]) for b in intraday_bars], dtype=float))
    return _calc_rsi(closes, period=min(14, len(closes) - 1))


def _compute_score_breakdown(state: dict, weights: dict) -> dict:
    """
    Pre-compute the signal score in Python for full transparency.
    Returns {"raw_score", "final_score", "timing_mult", "fired": [(label, pts)], "missed": [label]}
    """
    price       = state.get("current_price", 0)
    prev        = state.get("prev_close", price)
    day_chg     = (price - prev) / prev * 100 if prev else 0
    rsi         = state.get("rsi", 50)
    macd_hist   = (state.get("macd") or {}).get("histogram", 0)
    vwap        = state.get("vwap", 0)
    bb_lower    = (state.get("bollinger") or {}).get("lower", 0)
    support     = state.get("support", 0)
    vol_spike   = state.get("volume_spike", False)
    vol_ratio   = state.get("volume_spike_ratio", 1)
    smd         = state.get("smart_money", "NEUTRAL")
    ema_align   = (state.get("ema_stack") or {}).get("alignment", "MIXED")
    float_rot   = state.get("float_rotation", 0)
    sent_score  = state.get("sentiment_score", 50)
    velocity    = (state.get("social_velocity") or {}).get("velocity", 1)
    gap_sig     = (state.get("gap_info") or {}).get("signal", "NEUTRAL")
    sector_sig  = (state.get("sector_momentum") or {}).get("signal", "NEUTRAL")
    t_mult      = (state.get("timing") or {}).get("multiplier", 1.0)

    fired  = []
    missed = []
    score  = 0

    def _check(condition, label, pts, positive=True):
        nonlocal score
        if condition:
            score += pts if positive else -pts
            fired.append((label, f"+{pts}" if positive else f"-{pts}"))
        else:
            missed.append(label)

    # Bullish
    _check(macd_hist > 0,              "MACD bullish",           round(20 * weights.get("macd", 1.0)))
    _check(30 <= rsi <= 50,            f"RSI {rsi:.0f} bounce",  round(20 * weights.get("rsi_bounce", 1.0)))
    _check(50 < rsi <= 65,             f"RSI {rsi:.0f} momentum",round(10 * weights.get("rsi_momentum", 1.0)))
    _check(vwap > 0 and price >= vwap, "Above VWAP",             round(20 * weights.get("vwap", 1.0)))
    _check(ema_align == "BULLISH",     "EMA stack bullish",      round(15 * weights.get("ema_stack", 1.0)))
    _check(vol_spike,                  f"Volume {vol_ratio:.1f}x",round(15 * weights.get("volume", 1.0)))
    _check(bb_lower > 0 and price < bb_lower, "BB oversold",     round(15 * weights.get("bollinger", 1.0)))
    _check(smd == "ACCUMULATION",      "Smart $ accumulating",   round(20 * weights.get("smart_money", 1.0)))
    _check(float_rot > 50,             f"Float rot {float_rot:.0f}%", round(20 * weights.get("float_rot", 1.0)))
    _check(sent_score >= 60,           "Bullish sentiment",      round(15 * weights.get("sentiment", 1.0)))
    _check(velocity >= 3,              "Social velocity surge",  15)
    _check(gap_sig == "BULLISH",       "Bullish gap",            round(10 * weights.get("gap", 1.0)))
    _check(sector_sig == "BULLISH",    "Sector bullish",         10)
    _check(support > 0 and abs(price - support) / price <= 0.03, "Near support", round(10 * weights.get("support", 1.0)))
    _check(2 <= day_chg <= 8,          f"Day +{day_chg:.1f}%",  10)

    # Bearish (penalties)
    _check(macd_hist < 0,              "MACD bearish",           round(20 * weights.get("macd", 1.0)),  False)
    _check(rsi > 70,                   f"RSI {rsi:.0f} overbought", 20,                                 False)
    _check(ema_align == "BEARISH",     "EMA stack bearish",      round(15 * weights.get("ema_stack", 1.0)), False)
    _check(vwap > 0 and price < vwap,  "Below VWAP",             round(15 * weights.get("vwap", 1.0)), False)
    _check(smd == "DISTRIBUTION",      "Smart $ distributing",   round(20 * weights.get("smart_money", 1.0)), False)
    _check(sent_score <= 40,           "Bearish sentiment",      round(15 * weights.get("sentiment", 1.0)), False)
    _check(gap_sig == "FADE_RISK",     "Gap fade risk",          15,                                    False)
    _check(sector_sig == "BEARISH",    "Sector bearish",         10,                                    False)
    _check(day_chg > 10,               f"Extended +{day_chg:.1f}%", 10,                                False)
    _check(float_rot > 100 and sent_score < 60, "Float w/o catalyst", 10,                              False)

    final = round(score * t_mult)
    return {
        "raw_score":   score,
        "timing_mult": t_mult,
        "final_score": final,
        "fired":       fired,
        "missed":      missed,
    }


def _find_sr(bars: list, lookback: int = 20) -> dict:
    """Legacy fallback — prefer _multi_level_sr."""
    recent = bars[-min(lookback, len(bars)):]
    highs  = [b["h"] for b in recent]
    lows   = [b["l"] for b in recent]
    return {
        "support":    round(min(lows), 6),
        "resistance": round(max(highs), 6),
    }


# ── LangGraph node ─────────────────────────────────────────────────────────────

@traceable(name="tech_agent", tags=["pipeline", "indicators"])
def tech_node(state: dict) -> dict:
    annotate_run(state)
    bars              = state.get("bars", [])
    intraday_bars     = state.get("intraday_bars", [])
    ticker            = state["ticker"]
    current_volume    = state.get("volume", 0.0)
    avg_volume        = state.get("avg_volume", 0.0)
    sector            = state.get("sector", "Technology")
    premarket_price   = state.get("premarket_price", 0.0)

    print(f"📊 [TechAgent] Calculating indicators for {ticker}  ({len(bars)} bars)...")

    try:
        if len(bars) < 5:
            raise ValueError(f"Not enough bars: {len(bars)}")

        closes  = _sanitize(np.array([float(b["c"]) for b in bars], dtype=float))
        volumes = _sanitize(np.array([float(b.get("v", 0)) for b in bars], dtype=float))

        # ── Core indicators ───────────────────────────────────────────────────
        rsi        = _calc_rsi(closes)
        intra_rsi  = _intraday_rsi(intraday_bars)

        # ── Stochastic RSI ────────────────────────────────────────────────────
        try:
            from ta.momentum import StochRSIIndicator
            import pandas as pd

            close_series = pd.Series([float(b.get("c", b.get("close", 0)))
                                      for b in bars])

            if len(close_series) >= 14:
                stoch = StochRSIIndicator(
                    close=close_series,
                    window=14,
                    smooth1=3,
                    smooth2=3,
                )
                k_series = stoch.stochrsi_k()
                d_series = stoch.stochrsi_d()

                last_k = float(k_series.iloc[-1]) if not k_series.empty else 0.5
                last_d = float(d_series.iloc[-1]) if not d_series.empty else 0.5
                prev_k = float(k_series.iloc[-2]) if len(k_series) > 1 else last_k
                prev_d = float(d_series.iloc[-2]) if len(d_series) > 1 else last_d

                if last_k < 0.20:
                    stoch_signal = "OVERSOLD"
                elif last_k > 0.80:
                    stoch_signal = "OVERBOUGHT"
                elif prev_k <= prev_d and last_k > last_d and last_k < 0.50:
                    stoch_signal = "BUY_CROSS"
                elif prev_k >= prev_d and last_k < last_d and last_k > 0.50:
                    stoch_signal = "SELL_CROSS"
                else:
                    stoch_signal = "NEUTRAL"
            else:
                last_k, last_d = 0.5, 0.5
                stoch_signal   = "NEUTRAL"

        except Exception as _stoch_err:
            print(f"⚠️  [TechAgent] StochRSI error: {_stoch_err}")
            last_k, last_d = 0.5, 0.5
            stoch_signal   = "NEUTRAL"

        print(f"   Stoch RSI: k={last_k:.3f} d={last_d:.3f} → {stoch_signal}")

        macd       = _calc_macd(closes)
        bollinger  = _calc_bollinger(closes)
        atr        = _calc_atr(bars)
        sr         = _multi_level_sr(bars)
        vwap       = _calc_vwap(intraday_bars)

        # ── Unique indicators ─────────────────────────────────────────────────
        obv_arr    = _calc_obv(closes, volumes)
        obv_now    = round(float(obv_arr[-1]), 0)
        smd        = _smart_money_divergence(closes, volumes)
        ema_st     = _ema_stack(closes)
        float_rot  = _float_rotation_pct(ticker, current_volume)
        sector_m   = _sector_momentum(sector)
        timing     = _catalyst_timing_multiplier()
        gap_info   = _premarket_gap(bars, premarket_price)
        mkt_regime = _market_regime()
        rel_str    = _relative_strength(
            float(closes[-1]), state.get("prev_close", float(closes[-1]))
        )

        # ── Volume spike (regime-adaptive threshold) ──────────────────────────
        try:
            from intelligence_hub import hub as _hub
            _vol_min = _hub.get_regime_thresholds().get("volume_spike_min", 2.0)
        except Exception:
            _vol_min = 2.0
        vol_ratio    = (current_volume / avg_volume) if avg_volume > 0 else 1.0
        volume_spike = vol_ratio >= _vol_min

        price  = float(closes[-1])
        bb_tag = (
            "ABOVE_BB🔴" if price > bollinger["upper"]
            else "BELOW_BB🟢" if price < bollinger["lower"]
            else "inside_BB"
        )

        # ── Chart pattern detection ───────────────────────────────────────────
        highs_arr = _sanitize(np.array([float(b["h"]) for b in bars], dtype=float))
        lows_arr  = _sanitize(np.array([float(b["l"]) for b in bars], dtype=float))
        try:
            from pattern_detector import detect_patterns
            patterns = detect_patterns(closes, highs_arr, lows_arr, volumes)
        except Exception:
            patterns = []
        if patterns:
            pat_names = ", ".join(p["pattern"] for p in patterns)
            print(f"   🔷 Patterns: {pat_names}")

        # ── 3-Timeframe agreement ─────────────────────────────────────────────
        # Daily (existing closes), Hourly (fresh fetch), 15-min (intraday_bars)
        tf_daily    = _timeframe_signal(closes, "daily")
        hourly_bars = _fetch_hourly_bars(ticker)
        if hourly_bars:
            h_closes  = _sanitize(np.array([float(b["c"]) for b in hourly_bars], dtype=float))
            tf_hourly = _timeframe_signal(h_closes, "1h")
        else:
            tf_hourly = {"rsi": 50.0, "macd_hist": 0.0, "ma20": 0.0, "price": 0.0,
                         "signal": "NEUTRAL", "tf": "1h"}

        if intraday_bars:
            i_closes   = _sanitize(np.array([float(b["c"]) for b in intraday_bars], dtype=float))
            tf_intraday = _timeframe_signal(i_closes, "15m")
        else:
            tf_intraday = {"rsi": 50.0, "macd_hist": 0.0, "ma20": 0.0, "price": 0.0,
                           "signal": "NEUTRAL", "tf": "15m"}

        sigs           = [tf_daily["signal"], tf_hourly["signal"], tf_intraday["signal"]]
        buy_confirmed  = all(s == "BULLISH" for s in sigs)
        sell_confirmed = all(s == "BEARISH" for s in sigs)
        if buy_confirmed:
            tf_agreement = "ALL_BULL"
        elif sell_confirmed:
            tf_agreement = "ALL_BEAR"
        else:
            tf_agreement = "MIXED"

        print(
            f"   3TF: daily={tf_daily['signal']} "
            f"1h={tf_hourly['signal']} "
            f"15m={tf_intraday['signal']} "
            f"→ {tf_agreement}"
        )

        timeframe_agreement = {
            "daily":          tf_daily,
            "hourly":         tf_hourly,
            "intraday":       tf_intraday,
            "agreement":      tf_agreement,
            "buy_confirmed":  buy_confirmed,
            "sell_confirmed": sell_confirmed,
        }

        # ── Pre-compute score breakdown (for transparency) ────────────────────
        from self_learner import get_weight_adjustments
        partial_state = {
            **state,
            "rsi": rsi, "macd": macd, "bollinger": bollinger,
            "vwap": vwap, "volume_spike": volume_spike,
            "volume_spike_ratio": round(vol_ratio, 2),
            "smart_money": smd, "ema_stack": ema_st,
            "float_rotation": float_rot, "sector_momentum": sector_m,
            "timing": timing, "gap_info": gap_info,
            "support": sr["support"], "resistance": sr["resistance"],
        }
        score_bd = _compute_score_breakdown(partial_state, get_weight_adjustments())

        print(
            f"✅ [TechAgent] RSI={rsi:.1f} (intra={intra_rsi:.1f})  "
            f"MACD={macd['histogram']:+.6f}  BB={bb_tag}  ATR={atr:.4f}\n"
            f"   VWAP=${vwap:.4f} ({'above' if price>=vwap>0 else 'below' if vwap>0 else 'n/a'})  "
            f"EMA={ema_st['alignment']}  SMD={smd}  Vol={vol_ratio:.1f}x{'🔥' if volume_spike else ''}\n"
            f"   OBV={obv_now:,.0f}  Float={float_rot:.1f}%  "
            f"Sector({sector_m['etf']})={sector_m['change_pct']:+.2f}%\n"
            f"   Market={mkt_regime['regime']} (SPY {mkt_regime['spy_day_chg']:+.2f}%)  "
            f"RS={rel_str['label']}  Timing={timing['window']}\n"
            f"   Score: raw={score_bd['raw_score']} × {score_bd['timing_mult']} "
            f"= {score_bd['final_score']}  "
            f"[{len(score_bd['fired'])} signals fired]"
        )

        return {
            **state,
            "rsi":                rsi,
            "intraday_rsi":       intra_rsi,
            "stoch_rsi":          round(last_k, 3),
            "stoch_rsi_k":        round(last_k, 3),
            "stoch_rsi_d":        round(last_d, 3),
            "stoch_rsi_signal":   stoch_signal,
            "macd":               macd,
            "bollinger":          bollinger,
            "atr":                atr,
            "support":            sr["support"],
            "resistance":         sr["resistance"],
            "sr_levels":          sr,
            "volume_spike":       volume_spike,
            "volume_spike_ratio": round(vol_ratio, 2),
            "vwap":               vwap,
            "obv":                obv_now,
            "smart_money":        smd,
            "ema_stack":          ema_st,
            "float_rotation":     float_rot,
            "sector_momentum":    sector_m,
            "timing":             timing,
            "gap_info":           gap_info,
            "market_regime":      mkt_regime,
            "relative_strength":  rel_str,
            "score_breakdown":    score_bd,
            "patterns":           patterns,
            "timeframe_agreement": timeframe_agreement,
        }

    except Exception as e:
        print(f"❌ [TechAgent] Error: {e}")
        price = state.get("current_price", 0.0)
        return {
            **state,
            "rsi":                50.0,
            "intraday_rsi":       50.0,
            "stoch_rsi":          0.5,
            "stoch_rsi_k":        0.5,
            "stoch_rsi_d":        0.5,
            "stoch_rsi_signal":   "NEUTRAL",
            "macd":               {"macd": 0.0, "signal": 0.0, "histogram": 0.0},
            "bollinger":          {"upper": price, "middle": price, "lower": price, "bandwidth": 0.0},
            "atr":                0.0,
            "support":            0.0,
            "resistance":         0.0,
            "sr_levels":          {},
            "volume_spike":       False,
            "volume_spike_ratio": 1.0,
            "vwap":               0.0,
            "obv":                0.0,
            "smart_money":        "NEUTRAL",
            "ema_stack":          {"alignment": "MIXED", "ema9": 0.0, "ema21": 0.0, "ema50": 0.0},
            "float_rotation":     0.0,
            "sector_momentum":    {"etf": "SPY", "change_pct": 0.0, "signal": "NEUTRAL"},
            "timing":             {"multiplier": 1.0, "window": "unknown"},
            "gap_info":           {"gap_pct": 0.0, "signal": "NEUTRAL", "label": ""},
            "market_regime":      {"regime": "UNKNOWN", "spy_price": 0.0, "spy_day_chg": 0.0},
            "relative_strength":  {"rs_vs_spy": 0.0, "label": "n/a", "spy_chg": 0.0},
            "score_breakdown":    {"raw_score": 0, "final_score": 0, "timing_mult": 1.0,
                                   "fired": [], "missed": []},
            "patterns":           [],
            "timeframe_agreement": {"daily": {}, "hourly": {}, "intraday": {},
                                    "agreement": "MIXED",
                                    "buy_confirmed": False, "sell_confirmed": False},
        }

run_tech_agent = tech_node
