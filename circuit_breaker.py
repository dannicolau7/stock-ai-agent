"""
circuit_breaker.py — Market-level risk filter.

Prevents BUY signals on days with extreme fear (high VIX) or sharp
market-wide selloffs (SPY down sharply intraday).

check_market(spy_day_chg) → {safe, reason, vix, spy_chg}

Design principles:
  - Fails open: if VIX fetch errors, safe=True (never silently block signals)
  - 15-minute cache: avoids repeated downloads across ticker scans
  - SPY change is passed in from market_regime (already computed) to
    avoid a second yfinance fetch; falls back to fetching if not supplied
  - Thresholds are env-configurable so you can tighten/loosen without code changes
"""

import os
import time

import yfinance as yf
from dotenv import load_dotenv

load_dotenv(override=True)

VIX_THRESHOLD      = float(os.getenv("VIX_THRESHOLD",      "25"))
SPY_DROP_THRESHOLD = float(os.getenv("SPY_DROP_THRESHOLD", "-1.5"))

_CACHE_TTL = 900   # 15 minutes

_cache_ts:     float = 0.0
_cache_result: dict  = {}


def _fetch_vix() -> float:
    """Download latest VIX close from yfinance. Returns 0.0 on any error."""
    try:
        df = yf.download("^VIX", period="2d", interval="1d",
                         progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1].item())
    except Exception:
        pass
    return 0.0


def _fetch_spy_chg() -> float:
    """Compute SPY 1-day % change. Returns 0.0 on any error."""
    try:
        df = yf.download("SPY", period="2d", interval="1d",
                         progress=False, auto_adjust=True)
        if df is not None and len(df) >= 2:
            prev  = float(df["Close"].iloc[-2].item())
            last  = float(df["Close"].iloc[-1].item())
            if prev > 0:
                return round((last / prev - 1) * 100, 2)
    except Exception:
        pass
    return 0.0


def check_market(spy_day_chg: float = None) -> dict:
    """
    Check whether market conditions are safe to issue BUY signals.

    Parameters
    ----------
    spy_day_chg : float, optional
        SPY intraday % change already computed by market_regime().
        If supplied, skips the SPY download.

    Returns
    -------
    dict with keys:
        safe      : bool   — True = OK to trade, False = suppress BUYs
        reason    : str    — human-readable explanation
        vix       : float  — latest VIX reading
        spy_chg   : float  — SPY day change %
    """
    global _cache_ts, _cache_result

    # Return cached result if fresh
    if _cache_result and (time.time() - _cache_ts) < _CACHE_TTL:
        # Re-apply live spy_chg if caller supplied a fresher value
        if spy_day_chg is not None:
            merged = dict(_cache_result)
            merged["spy_chg"] = spy_day_chg
            if spy_day_chg <= SPY_DROP_THRESHOLD:
                merged["safe"]   = False
                merged["reason"] = (
                    f"SPY {spy_day_chg:+.1f}% (drops > {SPY_DROP_THRESHOLD}% — "
                    f"market-wide selloff)"
                )
            elif _cache_result["safe"] is False and "VIX" in _cache_result["reason"]:
                pass   # VIX trigger still active
            else:
                merged["safe"]   = True
                merged["reason"] = "OK"
            return merged
        return _cache_result

    # Fetch VIX — fail open: if fetch throws, treat as safe
    try:
        vix = _fetch_vix()
    except Exception:
        vix = 0.0

    # SPY change: use supplied value or fetch fresh — fail open on error
    try:
        spy = spy_day_chg if spy_day_chg is not None else _fetch_spy_chg()
    except Exception:
        spy = 0.0

    # Evaluate
    if vix > 0 and vix > VIX_THRESHOLD:
        result = {
            "safe":    False,
            "reason":  f"VIX {vix:.1f} > {VIX_THRESHOLD:.0f} (fear too high — avoid longs)",
            "vix":     vix,
            "spy_chg": spy,
        }
    elif spy <= SPY_DROP_THRESHOLD:
        result = {
            "safe":    False,
            "reason":  (
                f"SPY {spy:+.1f}% (drops > {SPY_DROP_THRESHOLD}% — "
                f"market-wide selloff)"
            ),
            "vix":     vix,
            "spy_chg": spy,
        }
    else:
        result = {
            "safe":    True,
            "reason":  "OK",
            "vix":     vix,
            "spy_chg": spy,
        }

    _cache_ts     = time.time()
    _cache_result = result
    return result


if __name__ == "__main__":
    status = check_market()
    icon   = "✅" if status["safe"] else "🚫"
    print(f"{icon} Market status: {status['reason']}")
    print(f"   VIX:     {status['vix']:.1f}  (threshold {VIX_THRESHOLD:.0f})")
    print(f"   SPY chg: {status['spy_chg']:+.2f}%  (threshold {SPY_DROP_THRESHOLD:+.1f}%)")
