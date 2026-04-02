import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polygon_feed import get_daily_bars, get_current_price, get_news, get_previous_close, get_ticker_details

def run_data_agent(state: dict) -> dict:
    ticker = state.get("ticker", "BZAI")
    print(f"📡 [DataAgent] Fetching data for {ticker}...")

    try:
        # Current price
        price = get_current_price(ticker)
        print(f"   💰 Price: ${price}")

        # Daily bars (90 days for technical analysis)
        bars = get_daily_bars(ticker, days=90)
        print(f"   📊 Got {len(bars)} daily bars")

        # Previous close details
        prev = get_previous_close(ticker)

        # News
        news = get_news(ticker, limit=8)
        print(f"   📰 Got {len(news)} news articles")

        # Company details
        details = get_ticker_details(ticker)

        # Volume analysis
        avg_volume = 0
        if bars:
            volumes = [b.get("v", 0) for b in bars[-30:]]
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
        current_volume = prev.get("v", 0)
        volume_ratio = round(current_volume / avg_volume, 2) if avg_volume else 0
        print(f"   📦 Volume ratio: {volume_ratio}x average")

        state.update({
            "price":         price,
            "bars":          bars,
            "prev_close":    prev,
            "news":          news,
            "ticker_details": details,
            "avg_volume":    round(avg_volume, 2),
            "current_volume": current_volume,
            "volume_ratio":  volume_ratio,
            "error":         None,
        })
        print(f"✅ [DataAgent] Done")

    except Exception as e:
        print(f"❌ [DataAgent] Error: {e}")
        state["error"] = str(e)

    return state


if __name__ == "__main__":
    result = run_data_agent({"ticker": "BZAI"})
    for k, v in result.items():
        if k not in ("bars", "news", "ticker_details"):
            print(f"  {k}: {v}")
