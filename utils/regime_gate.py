"""
utils/regime_gate.py — Market regime classifier and trade gate.

Runs at 9:15 AM ET (and on-demand). Determines whether the overall market
environment permits BUY signals and adjusts position sizing accordingly.

Public API:
  get_regime()           → RegimeState (cached, refreshes every 30 min)
  check_buy_allowed()    → (allowed: bool, reason: str)
  apply_regime_penalty(ticker, sector, score) → adjusted_score
  regime_header()        → one-line string for WhatsApp alert headers

Regime definitions:
  BULL   SPY > 20MA  AND VIX < 20            → full size, normal operation
  NEUTRAL SPY > 200MA AND VIX 20–30          → 50% size, tighter stops
  BEAR   SPY < 20MA  OR  VIX > 30            → BUY signals disabled
  PANIC  SPY < 200MA AND VIX > 35            → ALL signals disabled, alert sent
"""

import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
import numpy as np

_EST = ZoneInfo("America/New_York")

# ── Regime thresholds ──────────────────────────────────────────────────────────

VIX_BULL_MAX   = 20.0
VIX_NEUTRAL_MAX = 30.0
VIX_PANIC_MIN  = 35.0

SECTOR_PENALTY_PCT    = 2.0   # sector ETF down > this → add penalty to BUY score
SECTOR_PENALTY_POINTS = 15    # points deducted from BUY score

CACHE_TTL_S = 30 * 60   # refresh at most every 30 minutes

# Sector ETF map (matches tech_agent.py)
_SECTOR_ETFS: dict[str, str] = {
    "Technology":        "XLK",
    "Financials":        "XLF",
    "Energy":            "XLE",
    "Healthcare":        "XLV",
    "Consumer Cyclical": "XLY",
    "Industrials":       "XLI",
    "Basic Materials":   "XLB",
    "Real Estate":       "XLRE",
    "Utilities":         "XLU",
    "Communication":     "XLC",
    "Consumer Defensive":"XLP",
    "Semiconductor":     "SOXX",
    "Biotechnology":     "XBI",
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    regime:       str     = "UNKNOWN"   # BULL | NEUTRAL | BEAR | PANIC
    spy_price:    float   = 0.0
    spy_ma20:     float   = 0.0
    spy_ma200:    float   = 0.0
    spy_trend:    str     = "—"         # "above 20MA" | "below 20MA" | etc.
    vix:          float   = 0.0
    sector_changes: dict  = field(default_factory=dict)  # etf → day_chg_pct
    updated_at:   datetime | None = None

    @property
    def buy_allowed(self) -> bool:
        return self.regime in ("BULL", "NEUTRAL")

    @property
    def all_allowed(self) -> bool:
        return self.regime != "PANIC"

    @property
    def position_scale(self) -> float:
        """Multiplier for position sizing: 1.0 = full, 0.5 = half."""
        return {"BULL": 1.0, "NEUTRAL": 0.5, "BEAR": 0.0, "PANIC": 0.0}.get(
            self.regime, 1.0
        )


# ── EMA helper ────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    k      = 2 / (period + 1)
    result = [float(arr[0])]
    for v in arr[1:]:
        result.append(float(v) * k + result[-1] * (1 - k))
    return np.array(result)


# ── Data fetchers ──────────────────────────────────────────────────────────────

def _fetch_spy_data() -> dict:
    """Fetch SPY price, 20-day MA, 200-day MA via yfinance."""
    try:
        df = yf.Ticker("SPY").history(period="220d", interval="1d")
        if df.empty or len(df) < 20:
            return {}
        closes = df["Close"].values.astype(float)
        price  = float(closes[-1])
        ma20   = float(_ema(closes, 20)[-1])
        ma200  = float(_ema(closes, 200)[-1]) if len(closes) >= 200 else float(np.mean(closes))
        return {
            "price": round(price, 2),
            "ma20":  round(ma20,  2),
            "ma200": round(ma200, 2),
        }
    except Exception as e:
        print(f"⚠️  [RegimeGate] SPY fetch failed: {e}")
        return {}


def _fetch_vix() -> float:
    """Fetch current VIX level via yfinance."""
    try:
        fi = yf.Ticker("^VIX").fast_info
        return round(float(fi.last_price or fi.previous_close or 0), 2)
    except Exception as e:
        print(f"⚠️  [RegimeGate] VIX fetch failed: {e}")
        return 0.0


def _fetch_sector_changes() -> dict[str, float]:
    """
    Fetch today's % change for each sector ETF.
    Returns {etf_symbol: day_change_pct}.
    """
    changes: dict[str, float] = {}
    etfs = list(set(_SECTOR_ETFS.values()))
    for etf in etfs:
        try:
            fi   = yf.Ticker(etf).fast_info
            px   = float(fi.last_price or 0)
            prev = float(fi.previous_close or px)
            chg  = (px - prev) / prev * 100 if prev > 0 else 0.0
            changes[etf] = round(chg, 2)
        except Exception:
            changes[etf] = 0.0
    return changes


# ── Regime classification ─────────────────────────────────────────────────────

def _classify_regime(spy_price: float, ma20: float, ma200: float, vix: float) -> str:
    """
    Determine market regime from SPY vs MAs and VIX.

    PANIC  : SPY below 200MA AND VIX > 35
    BEAR   : SPY below 20MA  OR  VIX > 30
    NEUTRAL: SPY above 200MA AND VIX 20–30
    BULL   : SPY above 20MA  AND VIX < 20
    """
    if vix <= 0 or spy_price <= 0:
        return "UNKNOWN"

    below_200 = spy_price < ma200
    below_20  = spy_price < ma20

    if below_200 and vix > VIX_PANIC_MIN:
        return "PANIC"
    if below_20 or vix > VIX_NEUTRAL_MAX:
        return "BEAR"
    if vix > VIX_BULL_MAX:
        return "NEUTRAL"
    return "BULL"


def _spy_trend_label(price: float, ma20: float, ma200: float) -> str:
    if price > ma20 > ma200:
        return f"above 20MA & 200MA"
    if price > ma200 and price < ma20:
        return f"below 20MA (${ma20:.0f}), above 200MA"
    if price < ma200:
        return f"below 200MA (${ma200:.0f}) ⚠️"
    return "mixed"


# ── Cache ─────────────────────────────────────────────────────────────────────

_cache_lock  = threading.Lock()
_cached: RegimeState | None = None
_cached_at:  float          = 0.0   # epoch seconds


def _refresh() -> RegimeState:
    """Fetch all data and build a fresh RegimeState."""
    spy  = _fetch_spy_data()
    vix  = _fetch_vix()
    sect = _fetch_sector_changes()

    spy_price = spy.get("price", 0.0)
    ma20      = spy.get("ma20",  0.0)
    ma200     = spy.get("ma200", 0.0)

    regime    = _classify_regime(spy_price, ma20, ma200, vix)
    trend     = _spy_trend_label(spy_price, ma20, ma200)

    state = RegimeState(
        regime         = regime,
        spy_price      = spy_price,
        spy_ma20       = ma20,
        spy_ma200      = ma200,
        spy_trend      = trend,
        vix            = vix,
        sector_changes = sect,
        updated_at     = datetime.now(_EST),
    )

    print(
        f"📊 [RegimeGate] {regime}  "
        f"SPY=${spy_price:.2f} ({trend})  "
        f"VIX={vix:.1f}  "
        f"size_scale={state.position_scale:.0%}"
    )

    # PANIC alert
    if regime == "PANIC":
        _send_panic_alert(state)

    return state


def get_regime(*, force_refresh: bool = False) -> RegimeState:
    """
    Return current RegimeState from cache (refreshes after CACHE_TTL_S seconds).
    Thread-safe.
    """
    global _cached, _cached_at
    with _cache_lock:
        age = _time.monotonic() - _cached_at
        if force_refresh or _cached is None or age >= CACHE_TTL_S:
            _cached    = _refresh()
            _cached_at = _time.monotonic()
        return _cached


# ── Buy gate ──────────────────────────────────────────────────────────────────

def check_buy_allowed(regime: RegimeState | None = None) -> tuple[bool, str]:
    """
    Return (allowed, reason).
    Use this before firing any BUY alert.
    """
    r = regime or get_regime()
    if r.regime == "PANIC":
        return False, f"Market PANIC — VIX {r.vix:.1f}, SPY below 200MA. All signals halted."
    if r.regime == "BEAR":
        return False, f"Market BEAR — BUY signals disabled (SPY trend: {r.spy_trend}, VIX {r.vix:.1f})."
    if r.regime == "UNKNOWN":
        return True, "Regime unknown — allowing through (data unavailable)"
    return True, f"Market {r.regime} — BUY allowed (VIX {r.vix:.1f})"


# ── Sector penalty ────────────────────────────────────────────────────────────

def apply_regime_penalty(
    ticker: str,
    sector: str,
    score: int,
    regime: RegimeState | None = None,
) -> tuple[int, list[str]]:
    """
    Adjust a BUY score based on regime and sector ETF performance.

    Returns (adjusted_score, [list of applied penalties]).
    Penalties:
      - Sector ETF down > SECTOR_PENALTY_PCT today → -SECTOR_PENALTY_POINTS
      - NEUTRAL regime → -10 (reduced conviction environment)
    """
    r        = regime or get_regime()
    penalties: list[str] = []
    adj      = score

    # Sector ETF check
    etf = _SECTOR_ETFS.get(sector)
    if etf:
        chg = r.sector_changes.get(etf, 0.0)
        if chg <= -SECTOR_PENALTY_PCT:
            adj -= SECTOR_PENALTY_POINTS
            penalties.append(
                f"Sector {etf} {chg:+.1f}% today "
                f"(-{SECTOR_PENALTY_POINTS}pts)"
            )

    # NEUTRAL regime penalty
    if r.regime == "NEUTRAL":
        adj -= 10
        penalties.append("NEUTRAL regime (-10pts)")

    if adj != score:
        print(
            f"📊 [RegimeGate] {ticker} score {score}→{adj}  "
            f"penalties: {'; '.join(penalties)}"
        )

    return adj, penalties


# ── Alert header ──────────────────────────────────────────────────────────────

def regime_header(regime: RegimeState | None = None) -> str:
    """
    One-line string for WhatsApp alert headers.
    Example: "Market: BULL | VIX: 14.2 | SPY: above 20MA & 200MA"
    """
    r = regime or get_regime()
    return f"Market: {r.regime} | VIX: {r.vix:.1f} | SPY: {r.spy_trend}"


# ── PANIC alert ───────────────────────────────────────────────────────────────

_panic_alerted: bool = False   # fire once per process lifetime


def _send_panic_alert(state: RegimeState) -> None:
    global _panic_alerted
    if _panic_alerted:
        return
    _panic_alerted = True
    try:
        from alerts import send_whatsapp
        now_str = datetime.now(tz=_EST).strftime("%I:%M %p ET")
        msg = (
            f"🚨 MARKET HALT — PANIC REGIME\n\n"
            f"SPY ${state.spy_price:.2f} (below 200MA ${state.spy_ma200:.0f})\n"
            f"VIX {state.vix:.1f} — extreme fear\n\n"
            f"All BUY and SELL signals disabled.\n"
            f"Monitor only — do not trade.\n"
            f"⏰ {now_str}"
        )
        send_whatsapp(msg)
        print("🚨 [RegimeGate] PANIC alert sent via WhatsApp")
    except Exception as e:
        print(f"⚠️  [RegimeGate] PANIC alert failed: {e}")


# ── Convenience: run at 9:15 AM ───────────────────────────────────────────────

def morning_regime_check() -> RegimeState:
    """
    Force-refresh regime data. Call from scheduler at 9:15 AM ET.
    Sends PANIC alert if warranted. Returns current RegimeState.
    """
    state = get_regime(force_refresh=True)
    print(
        f"\n📊 [RegimeGate] 9:15 AM check — {state.regime}\n"
        f"   SPY  ${state.spy_price:.2f}  "
        f"20MA ${state.spy_ma20:.2f}  200MA ${state.spy_ma200:.2f}\n"
        f"   VIX  {state.vix:.1f}\n"
        f"   Sectors: "
        + "  ".join(
            f"{etf} {chg:+.1f}%"
            for etf, chg in sorted(state.sector_changes.items())
            if abs(chg) >= 0.5
        )
    )
    return state


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== RegimeGate standalone test ===\n")
    state = morning_regime_check()

    print(f"\nRegime:        {state.regime}")
    print(f"SPY:           ${state.spy_price:.2f}  (20MA ${state.spy_ma20:.2f} / 200MA ${state.spy_ma200:.2f})")
    print(f"Trend:         {state.spy_trend}")
    print(f"VIX:           {state.vix:.1f}")
    print(f"Position scale:{state.position_scale:.0%}")
    print(f"BUY allowed:   {state.buy_allowed}")

    buy_ok, reason = check_buy_allowed(state)
    print(f"\ncheck_buy_allowed: {buy_ok}  — {reason}")

    header = regime_header(state)
    print(f"\nAlert header:  {header}")

    print("\nSector ETF changes today:")
    for etf, chg in sorted(state.sector_changes.items(), key=lambda x: x[1]):
        bar = "▼" if chg < -SECTOR_PENALTY_PCT else ("▲" if chg > SECTOR_PENALTY_PCT else " ")
        print(f"  {bar} {etf:6s}  {chg:+.2f}%")

    # Penalty test
    score_adj, notes = apply_regime_penalty("NVDA", "Technology", 80, state)
    print(f"\nPenalty test (NVDA, Tech, score=80) → {score_adj}  notes={notes}")
