import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
import yfinance as yf
from polygon_feed import get_daily_bars, get_current_price, get_news, get_previous_close, get_ticker_details


def _fetch_intraday_bars(ticker: str) -> list:
    """Fetch today's 15-min intraday bars via yfinance."""
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="15m")
        if df.empty:
            return []
        bars = []
        for ts, row in df.iterrows():
            bars.append({
                "t": int(ts.timestamp() * 1000),
                "o": float(row["Open"]),
                "h": float(row["High"]),
                "l": float(row["Low"]),
                "c": float(row["Close"]),
                "v": float(row["Volume"]),
            })
        return bars
    except Exception as e:
        print(f"⚠️  [DataAgent] Intraday bars error: {e}")
        return []


def _fetch_premarket_price(ticker: str) -> float:
    """Fetch latest pre-market price via yfinance (prePost=True)."""
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1m", prepost=True)
        if df.empty:
            return 0.0
        return float(df["Close"].iloc[-1])
    except Exception:
        return 0.0


def get_premarket_data(ticker: str) -> dict:
    """
    Rich pre-market snapshot using yfinance.info + 1-min bars.
    Returns: premarket_price, premarket_change_pct, premarket_volume,
             premarket_high, premarket_low, is_premarket_active.
    """
    result = {
        "premarket_price":      0.0,
        "premarket_change_pct": 0.0,
        "premarket_volume":     0,
        "premarket_high":       0.0,
        "premarket_low":        0.0,
        "is_premarket_active":  False,
    }
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        pm_price = float(info.get("preMarketPrice")  or 0.0)
        pm_vol   = int  (info.get("preMarketVolume") or 0)
        pm_chg   = float(info.get("preMarketChangePercent") or 0.0)
        # yfinance sometimes returns the ratio (0.05 = 5%) vs already-pct
        if pm_chg != 0 and abs(pm_chg) < 1:
            pm_chg *= 100

        result["premarket_price"]      = pm_price
        result["premarket_volume"]     = pm_vol
        result["premarket_change_pct"] = round(pm_chg, 2)
        result["is_premarket_active"]  = pm_price > 0

        # High/low from 1-min bars
        hist = stock.history(period="1d", interval="1m", prepost=True)
        if not hist.empty:
            from zoneinfo import ZoneInfo
            from datetime import time as _dtime
            _ET = ZoneInfo("America/New_York")
            if hist.index.tz is None:
                hist.index = hist.index.tz_localize("UTC").tz_convert(_ET)
            else:
                hist.index = hist.index.tz_convert(_ET)
            pre = hist[hist.index.time < _dtime(9, 30)]
            if not pre.empty:
                result["premarket_high"] = float(pre["High"].max())
                result["premarket_low"]  = float(pre["Low"].min())
    except Exception as e:
        print(f"⚠️  [DataAgent] get_premarket_data({ticker}) error: {e}")
    return result


def _fetch_earnings_info(ticker: str) -> dict:
    """
    Returns days until next earnings and whether it's within danger zone.
    Uses yfinance calendar.
    """
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None or cal.empty:
            return {"days_to_earnings": 999, "earnings_risk": "none", "earnings_date": ""}

        # calendar columns vary — try common keys
        date_col = None
        for col in ["Earnings Date", "Earnings Dates", 0]:
            try:
                val = cal[col] if isinstance(col, str) else cal.iloc[:, col]
                if val is not None:
                    date_col = val
                    break
            except Exception:
                continue

        if date_col is None:
            return {"days_to_earnings": 999, "earnings_risk": "none", "earnings_date": ""}

        # Get the first upcoming date
        today = datetime.now(timezone.utc).date()
        for raw_date in (date_col if hasattr(date_col, '__iter__') else [date_col]):
            try:
                if hasattr(raw_date, 'date'):
                    d = raw_date.date()
                else:
                    d = datetime.strptime(str(raw_date)[:10], "%Y-%m-%d").date()
                days = (d - today).days
                if days >= 0:
                    if days <= 3:
                        risk = "HIGH"
                    elif days <= 7:
                        risk = "MEDIUM"
                    elif days <= 14:
                        risk = "LOW"
                    else:
                        risk = "none"
                    return {
                        "days_to_earnings": days,
                        "earnings_risk": risk,
                        "earnings_date": str(d),
                    }
            except Exception:
                continue
    except Exception:
        pass
    return {"days_to_earnings": 999, "earnings_risk": "none", "earnings_date": ""}


def _fetch_sector(ticker: str) -> str:
    """Get sector from yfinance.info."""
    try:
        info   = yf.Ticker(ticker).info
        sector = info.get("sector") or info.get("industry") or "Technology"
        return sector
    except Exception:
        return "Technology"


def run_data_agent(state: dict) -> dict:
    ticker = state.get("ticker", "")
    print(f"📡 [DataAgent] Fetching data for {ticker}...")

    try:
        # Current price — record fetch time so alert_agent can re-fetch if stale
        price = get_current_price(ticker)
        price_fetched_at = datetime.now(timezone.utc).isoformat()
        print(f"   💰 Price: ${price}  [fetched {datetime.now().strftime('%H:%M:%S')}]")

        # Daily bars (90 days for technical analysis)
        bars = get_daily_bars(ticker, days=90)
        print(f"   📊 Got {len(bars)} daily bars")

        # Intraday 15-min bars for VWAP
        intraday_bars = _fetch_intraday_bars(ticker)
        print(f"   📈 Got {len(intraday_bars)} intraday bars (15m)")

        # Pre-market price for gap detection
        premarket_price = _fetch_premarket_price(ticker)

        # Previous close details
        prev = get_previous_close(ticker)

        # News
        news = get_news(ticker, limit=8)
        print(f"   📰 Got {len(news)} news articles")

        # Company details + sector
        details = get_ticker_details(ticker)
        sector  = _fetch_sector(ticker)

        # Earnings proximity
        earnings_info = _fetch_earnings_info(ticker)
        if earnings_info["earnings_risk"] != "none":
            print(f"   ⚠️  Earnings in {earnings_info['days_to_earnings']}d "
                  f"({earnings_info['earnings_date']}) — risk={earnings_info['earnings_risk']}")
        else:
            print(f"   📅 No near-term earnings risk")

        # Volume analysis
        avg_volume = 0.0
        if bars:
            volumes    = [b.get("v", 0) for b in bars[-30:]]
            avg_volume = sum(volumes) / len(volumes) if volumes else 0.0

        # Current session volume — sum today's intraday bars, NOT yesterday's aggregate.
        # prev.get("v") is Polygon's /prev endpoint (previous-day close bar), so using it
        # as "current volume" was always wrong. Falls back to 0 rather than propagating
        # stale data into volume_spike / liquidity / sizing checks.
        if intraday_bars:
            current_volume = float(sum(b.get("v", 0) for b in intraday_bars))
        else:
            current_volume = 0.0   # pre-market or closed — no session data yet

        volume_ratio = round(current_volume / avg_volume, 2) if avg_volume else 0
        print(f"   📦 Volume ratio: {volume_ratio}x average  |  Sector: {sector}")

        state.update({
            "current_price":    price,
            "price_fetched_at": price_fetched_at,
            "bars":             bars,
            "intraday_bars":    intraday_bars,
            "premarket_price":  premarket_price,
            "prev_close":       float(prev.get("c", price)),
            "raw_news":         news,
            "ticker_details":   details,
            "sector":           sector,
            "earnings_info":    earnings_info,
            "avg_volume":       round(avg_volume, 2),
            "volume":           float(current_volume),
            "volume_ratio":     volume_ratio,
            "error":            None,
        })
        print(f"✅ [DataAgent] Done")

    except Exception as e:
        print(f"❌ [DataAgent] Error: {e}")
        state["error"] = str(e)

    return state


data_node = run_data_agent  # alias for graph.py


if __name__ == "__main__":
    result = run_data_agent({"ticker": "BZAI"})
    for k, v in result.items():
        if k not in ("bars", "intraday_bars", "raw_news", "ticker_details"):
            print(f"  {k}: {v}")
