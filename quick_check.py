import yfinance as yf
import sys

# ── Add or remove tickers here anytime ──
DEFAULT_TICKERS = ["BZAI", "AAPL", "NVDA", "TSLA", "META"]

def check_stock(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info['last_price']
        prev_close = stock.fast_info['previous_close']
        change = price - prev_close
        pct = (change / prev_close) * 100
        arrow = "▲" if change >= 0 else "▼"
        color = "🟢" if change >= 0 else "🔴"

        print(f"\n{'='*60}")
        print(f"{color} {ticker}: ${price:.2f}  {arrow} {pct:.2f}%  (prev: ${prev_close:.2f})")
        print(f"{'='*60}")

        # News
        print(f"📰 Latest News:")
        try:
            news = stock.news
            if not news:
                print("   No news found")
            for i, n in enumerate(news[:3]):
                content = n.get("content", {})
                title = content.get("title", "No title")
                date = content.get("pubDate", "")[:10]
                print(f"   {i+1}. [{date}] {title}")
        except Exception as e:
            print(f"   Could not fetch news: {e}")

    except Exception as e:
        print(f"❌ {ticker}: Error — {e}")

def main():
    # Use command line tickers if provided
    # Otherwise use DEFAULT_TICKERS list above
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS

    print(f"\n🚀 Stock Quick Check")
    print(f"📋 Checking {len(tickers)} stocks: {', '.join(tickers)}")

    for ticker in tickers:
        check_stock(ticker.upper())

    print(f"\n✅ Done!")

if __name__ == "__main__":
    main()
