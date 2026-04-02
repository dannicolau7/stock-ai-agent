import requests
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


def get_snapshot(ticker: str) -> dict:
    data = _get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
    return data.get("ticker", {})


def get_daily_bars(ticker: str, days: int = 60) -> list:
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
    data = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
        {"adjusted": "true", "sort": "asc", "limit": 120},
    )
    return data.get("results", [])


def get_news(ticker: str, limit: int = 10) -> list:
    data = _get("/v2/reference/news", {"ticker": ticker, "limit": limit, "order": "desc"})
    return data.get("results", [])


def get_current_price(ticker: str) -> float:
    snap = get_snapshot(ticker)
    price = snap.get("lastTrade", {}).get("p", 0)
    if not price:
        price = snap.get("day", {}).get("c", 0)
    return float(price)
