import requests
import yfinance as yf
from datetime import datetime, timedelta
from config import POLYGON_API_KEY

BASE_URL = "https://api.polygon.io"


def _get(path: str, params: dict = None) -> dict:
    p = {"apiKey": POLYGON_API_KEY}
    if params:
        p.update(params)
    r = requests.get(f"{BASE_URL}{path}", params=p, timeout=15)
    r.raise_for_status()
    return r.json()


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

    print(f"\n✅ Polygon feed working!")
