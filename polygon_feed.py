import time
import requests
import yfinance as yf
from datetime import datetime, timedelta
from config import POLYGON_API_KEY

BASE_URL = "https://api.polygon.io"


def _get(path: str, params: dict = None, retries: int = 3) -> dict:
    p = {"apiKey": POLYGON_API_KEY}
    if params:
        p.update(params)
    for attempt in range(retries):
        r = requests.get(f"{BASE_URL}{path}", params=p, timeout=15)
        if r.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"⏳ [Polygon] Rate limited — waiting {wait}s (attempt {attempt+1}/{retries})...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()  # raise after exhausting retries
    return {}


def get_daily_bars(ticker: str, days: int = 60) -> list:
    """Get OHLCV daily bars — free tier"""
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    data = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
        {"adjusted": "true", "sort": "asc", "limit": 120},
    )
    return data.get("results", [])


def get_previous_close(ticker: str) -> dict:
    """Get previous day close — free tier"""
    data = _get(f"/v2/aggs/ticker/{ticker}/prev")
    results = data.get("results", [])
    return results[0] if results else {}


def get_current_price(ticker: str) -> float:
    """
    Returns real-time price via yfinance.
    Falls back to Polygon previous close if yfinance fails.
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info["last_price"]
        if price and float(price) > 0:
            print(f"💰 Live price via yfinance: ${float(price):.4f}")
            return float(price)
        raise ValueError("yfinance returned zero/null price")
    except Exception as e:
        print(f"⚠️  yfinance failed ({e}) — falling back to Polygon...")

    prev  = get_previous_close(ticker)
    price = float(prev.get("c", 0))
    print(f"💰 Fallback price via Polygon: ${price:.4f}")
    return price


def get_intraday_bars(ticker: str, interval: str = "5m", period: str = "1d") -> list:
    """
    Returns intraday OHLCV bars via yfinance.
    Each bar is a dict with keys: t (ms timestamp), o, h, l, c, v
    Compatible with the same format used by get_daily_bars().

    interval: '1m' '2m' '5m' '15m' '30m' '60m' '90m'
    period:   '1d' '5d' '1mo'
    """
    try:
        df = yf.Ticker(ticker).history(interval=interval, period=period)
        if df.empty:
            print(f"⚠️  [yfinance] No intraday data returned for {ticker}")
            return []
        bars = []
        for ts, row in df.iterrows():
            bars.append({
                "t": int(ts.timestamp() * 1000),
                "o": round(float(row["Open"]),   6),
                "h": round(float(row["High"]),   6),
                "l": round(float(row["Low"]),    6),
                "c": round(float(row["Close"]),  6),
                "v": round(float(row["Volume"]), 2),
            })
        print(f"📊 [yfinance] {len(bars)} intraday bars ({interval}, {period}) for {ticker}")
        return bars
    except Exception as e:
        print(f"❌ [yfinance] get_intraday_bars failed: {e}")
        return []


def get_news(ticker: str, limit: int = 10) -> list:
    """Get news articles — free tier"""
    data = _get("/v2/reference/news", {
        "ticker": ticker,
        "limit": limit,
        "order": "desc"
    })
    return data.get("results", [])


def get_ticker_details(ticker: str) -> dict:
    """Get company details — free tier"""
    data = _get(f"/v3/reference/tickers/{ticker}")
    return data.get("results", {})


if __name__ == "__main__":
    ticker = "BZAI"
    print(f"\n🔑 Using Polygon key: {POLYGON_API_KEY[:8]}...")

    print(f"\n📊 Testing previous close price...")
    price = get_current_price(ticker)
    print(f"   {ticker} price: ${price}")

    print(f"\n📈 Testing daily bars (last 5 days)...")
    bars = get_daily_bars(ticker, days=5)
    print(f"   Got {len(bars)} bars")
    if bars:
        last = bars[-1]
        print(f"   Last bar → O:{last.get('o')} H:{last.get('h')} L:{last.get('l')} C:{last.get('c')} V:{last.get('v')}")

    print(f"\n📰 Testing news feed...")
    news = get_news(ticker, limit=3)
    print(f"   Got {len(news)} articles")
    for n in news:
        print(f"   → {n.get('published_utc','')[:10]} | {n.get('title','')[:60]}")

    print(f"\n🏢 Testing ticker details...")
    details = get_ticker_details(ticker)
    print(f"   Company: {details.get('name')}")
    print(f"   Market cap: {details.get('market_cap')}")

    print(f"\n⏱️  Testing intraday bars (5min, today)...")
    intraday = get_intraday_bars(ticker, interval="5m", period="1d")
    print(f"   Got {len(intraday)} 5-min bars")
    if intraday:
        first = intraday[0]
        last  = intraday[-1]
        print(f"   First → {datetime.fromtimestamp(first['t']/1000).strftime('%H:%M')}  C:{first['c']}")
        print(f"   Last  → {datetime.fromtimestamp(last['t']/1000).strftime('%H:%M')}  C:{last['c']}")

    print(f"\n✅ Polygon feed working!")
