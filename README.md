# Stock AI Agent

> AI-powered stock monitoring agent that combines technical analysis, social sentiment, and Claude AI to generate real-time BUY / SELL / HOLD signals with SMS and push notification alerts.

**Version:** v1.0 &nbsp;|&nbsp; **Status:** Active Development

---

## 1. Project Overview

### What it does

Stock AI Agent is a fully automated stock monitoring system that runs a multi-agent pipeline every 5 minutes during market hours. For each cycle it:

1. Fetches real-time price and OHLCV bars (yfinance + Polygon)
2. Scrapes Reddit for social sentiment across r/wallstreetbets, r/stocks, r/investing
3. Calculates RSI, MACD, Bollinger Bands, volume spikes, support and resistance
4. Passes all context to Claude AI which scores a confidence level (0–100)
5. Fires a BUY or SELL alert via SMS (Twilio) + push notification (Pushover) **only** when confidence ≥ 65
6. Logs every signal to `signals_log.csv` for accuracy tracking
7. Streams live data to a browser dashboard with candlestick charts

### Who it's for

- Retail traders who want AI-assisted signal generation on small/mid cap stocks
- Developers learning LangGraph multi-agent patterns with real financial data
- Anyone who wants to build on top of a working, free-tier signal pipeline

### Current Version: v1.0

---

## 2. Tech Stack

| Layer | Tool | Purpose | Cost |
|---|---|---|---|
| Agent orchestration | LangGraph + LangChain | Multi-agent pipeline, state management | Free |
| AI reasoning | Claude API (claude-sonnet-4-6) | Signal interpretation, confidence scoring | Pay-per-use |
| Historical data | Polygon.io | Daily OHLCV bars, news, ticker details | Free tier |
| Real-time data | yfinance | Live price, intraday 5-min candles | Free |
| Social sentiment | Reddit JSON API | WSB / stocks / investing post sentiment | Free, no auth |
| SMS alerts | Twilio | BUY/SELL signal SMS delivery | Free trial |
| Push notifications | Pushover | iOS/Android push notifications | $5 one-time |
| Dashboard server | FastAPI + uvicorn | REST API + HTML serving | Free |
| Chart UI | TradingView Lightweight Charts | Candlestick chart, RSI panel, markers | Free |

---

## 3. Project Structure

```
stock-ai-agent/
│
├── main.py                  # Entry point — monitoring loop, FastAPI server, market hours
├── config.py                # Loads all API keys from .env via python-dotenv
├── graph.py                 # Builds and compiles the LangGraph pipeline
├── analyzer.py              # Calls Claude API with full market context → structured signal
├── polygon_feed.py          # Polygon.io + yfinance data layer (bars, price, news)
├── alerts.py                # Twilio SMS + Pushover push notification senders
├── logger.py                # Appends every BUY/SELL signal to signals_log.csv
├── backtest.py              # Rule-based backtester — same logic as live agent
├── watchlist_manager.py     # Persist tickers to watchlist.json between sessions
├── watchlist.json           # Saved ticker watchlist
├── quick_check.py           # Instant price + news snapshot for any ticker(s)
├── requirements.txt         # Python dependencies
├── .env                     # API keys (gitignored)
├── .env.example             # Template for .env setup
│
├── agents/
│   ├── data_agent.py        # Node 1 — fetches price, bars, volume, news from Polygon/yfinance
│   ├── news_agent.py        # Node 2 — Reddit sentiment scoring (no API key required)
│   ├── tech_agent.py        # Node 3 — RSI, MACD, Bollinger Bands, ATR, volume spike
│   ├── decision_agent.py    # Node 4 — calls analyzer.py, gates on confidence >= 65
│   └── alert_agent.py       # Node 5 — fires SMS + push, respects paper trading mode
│
└── dashboard/
    └── index.html           # Live trading terminal — candlesticks, RSI, signal log, news
```

### Agent Pipeline Flow

```
fetch_data → analyze_news → analyze_tech → decide → alert
    │              │              │            │         │
 price/bars    Reddit score    RSI/MACD/BB  Claude AI  SMS+Push
```

---

## 4. Signal Sources & Quality

| Signal | Source | Cost | Built | Reliability |
|---|---|---|---|---|
| RSI (14) | Calculated from bars | Free | ✅ | ⭐⭐⭐⭐ |
| MACD (12/26/9) | Calculated from bars | Free | ✅ | ⭐⭐⭐⭐ |
| Volume spike (>2× avg) | yfinance / Polygon | Free | ✅ | ⭐⭐⭐⭐⭐ |
| Bollinger Bands (20, 2σ) | Calculated from bars | Free | ✅ | ⭐⭐⭐ |
| Support / Resistance | 20-bar high/low | Free | ✅ | ⭐⭐⭐⭐ |
| News sentiment | Polygon news + Reddit | Free | ✅ | ⭐⭐⭐ |
| Social sentiment | Reddit (WSB/stocks/investing) | Free | ✅ | ⭐⭐ |
| Options flow | Unusual Whales | $50/mo | ❌ Not yet | ⭐⭐⭐⭐⭐ |
| Dark pool prints | Unusual Whales | $50/mo | ❌ Not yet | ⭐⭐⭐⭐⭐ |
| Insider trading | SEC EDGAR | Free | ❌ Not yet | ⭐⭐⭐⭐⭐ |
| Real-time news | Benzinga | $50/mo | ❌ Not yet | ⭐⭐⭐⭐ |

---

## 5. Confidence Scoring System

Each cycle Claude scores the combined signal confluence from 0–100. The signal only fires if the score reaches the threshold.

| Signal Component | Bullish Condition | Points |
|---|---|---|
| RSI | < 30 (oversold) | +30 |
| RSI | 30–40 range | +15 |
| MACD | Histogram crosses positive (bullish crossover) | +25 |
| MACD | Histogram positive (bullish drift) | +10 |
| Bollinger Bands | Price below lower band | +20 |
| Volume spike | Current volume > 2× 20-bar average | +15 |

Bearish scoring is mirrored (RSI > 70, MACD crosses negative, price above upper BB, volume spike down).

```
Score 80–100  →  Very high conviction   → BUY or SELL fires
Score 65–79   →  Clear directional bias → BUY or SELL fires
Score 50–64   →  Mixed signals          → Forced to HOLD
Score 0–49    →  Opposing signals       → Forced to HOLD

Threshold: score >= 65 required to send alert
```

Claude's final reasoning is included in the alert and logged to `signals_log.csv`.

---

## 6. API Keys Required

| Variable | Where to Get It | Cost |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Pay-per-use (already have it) |
| `POLYGON_API_KEY` | [polygon.io/dashboard](https://polygon.io/dashboard) | Free tier |
| `TWILIO_ACCOUNT_SID` | [twilio.com/console](https://twilio.com/console) | Free trial |
| `TWILIO_AUTH_TOKEN` | [twilio.com/console](https://twilio.com/console) | Free trial |
| `TWILIO_FROM_NUMBER` | Twilio console → Phone Numbers | Free trial number |
| `TWILIO_TO_NUMBER` | Your personal phone number | — |
| `PUSHOVER_APP_TOKEN` | [pushover.net](https://pushover.net) → Your Apps | $5 one-time |
| `PUSHOVER_USER_KEY` | [pushover.net](https://pushover.net) → Settings | $5 one-time |

### `.env` setup

```bash
cp .env.example .env
# then fill in your keys:
nano .env
```

```env
ANTHROPIC_API_KEY=sk-ant-...
POLYGON_API_KEY=...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
TWILIO_TO_NUMBER=+1xxxxxxxxxx
PUSHOVER_APP_TOKEN=...
PUSHOVER_USER_KEY=...
TICKER=BZAI
MONITOR_INTERVAL=300
```

---

## 7. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify setup

```bash
# Test API keys load correctly
python3 config.py

# Test Polygon + yfinance connection
python3 polygon_feed.py

# Quick price and news snapshot (default tickers)
python3 quick_check.py

# Quick check specific tickers
python3 quick_check.py BZAI AAPL NVDA
```

### Run the agent

```bash
# Paper trading — full pipeline, no real SMS/push sent
python3 main.py --ticker BZAI AWRE LTRX BBAI SOUN --paper

# Live monitoring — fires real alerts
python3 main.py --ticker BZAI AWRE LTRX BBAI SOUN

# Monitor tickers from watchlist.json (no --ticker flag needed)
python3 main.py --paper

# Custom scan interval (every 2 minutes)
python3 main.py --ticker BZAI AWRE LTRX BBAI SOUN --interval 120

# Custom dashboard port
python3 main.py --ticker BZAI AWRE LTRX BBAI SOUN --port 8080
```

### Watchlist management

```bash
python3 main.py --add NVDA        # add ticker to watchlist.json
python3 main.py --remove NVDA     # remove ticker from watchlist.json
python3 main.py --list            # print current watchlist
```

### Backtesting

```bash
# Backtest last 30 days
python3 backtest.py --ticker BZAI --days 30

# Backtest 90 days, 10-bar forward window
python3 backtest.py --ticker AAPL --days 90 --forward 10
```

### Dashboard

Once `main.py` is running, open your browser:

```
http://localhost:8000
```

The dashboard auto-polls every 30 seconds for new signals. Tabs: **Signal** · **History** · **News** · **Watchlist**

---

## 8. Alert Format

Every BUY or SELL signal is sent as both an SMS and a push notification in this format:

```
🟢 BUY — BZAI
Price:      $1.7900
Entry Zone: $1.75 - $1.85
Targets:    $1.95 / $2.10 / $2.35
Stop Loss:  $1.62
RSI:        28.4
MACD Hist:  +0.002341

RSI oversold at 28 with MACD bullish crossover confirming
momentum reversal. Volume spike 3.2x average suggests
institutional accumulation near the $1.65 support level.
```

```
🔴 SELL — BZAI
Price:      $1.9900
Entry Zone: $1.95 - $2.00
Targets:    $1.78 / $1.65 / $1.50
Stop Loss:  $2.14
RSI:        74.1
MACD Hist:  -0.003812

RSI overbought at 74 with MACD bearish crossover. Price
rejected at resistance with elevated volume confirming
distribution. Stop above recent high at $2.14.
```

HOLD signals are **never** sent — alerts only fire when confidence ≥ 65.

---

## 9. Data Sources Explained

### Polygon.io (free tier)

- **What it provides:** Adjusted daily OHLCV bars (up to 2 years), previous day close, company details, news headlines
- **What it doesn't provide on free tier:** Real-time quotes, intraday bars, options data, WebSocket streaming
- **Rate limit:** 5 API calls/minute on free tier
- **Used for:** Historical bars for RSI/MACD/BB calculation, news feed, ticker metadata

### yfinance (free, no key)

- **What it provides:** Real-time last price (`fast_info['last_price']`), intraday bars (1m, 5m, 15m, 30m, 1h), fundamentals
- **Why we use it:** Polygon free tier only gives previous close — yfinance fills the gap with a live price
- **Used for:** `get_current_price()` (primary), `get_intraday_bars()` (5-min candles for dashboard)
- **Limitation:** Unofficial API, can occasionally be rate-limited; Polygon is the fallback

### Reddit (free, no auth)

- **Subreddits monitored:** r/wallstreetbets · r/stocks · r/investing
- **Endpoint:** `https://www.reddit.com/r/{sub}/search.json?q={ticker}&sort=new&limit=10`
- **Scoring method:** Count positive vs negative keyword hits across all post titles
  - **Positive keywords:** buy, bull, bullish, moon, calls, surge, up, gain, profit, long, rocket, rally, breakout, pump, squeeze, rip, green, wins, winner, growth
  - **Negative keywords:** sell, bear, bearish, puts, crash, down, loss, short, dump, drop, falling, tank, red, baghold, bagholder, bankrupt, fraud, collapse, bust, rekt
  - **Score formula:** `(positive_hits / total_hits) × 100`
  - BULLISH ≥ 60 · BEARISH ≤ 40 · NEUTRAL otherwise
- **Limitation:** Small-cap tickers may have few posts; score is a rough proxy, not professional sentiment

---

## 10. Roadmap

### v1.0 — Current ✅

- [x] Multi-agent LangGraph pipeline (data → news → tech → decision → alert)
- [x] RSI, MACD, Bollinger Bands, volume spike, support/resistance
- [x] Reddit social sentiment (r/WSB, r/stocks, r/investing) — no API key
- [x] Claude AI confidence scoring with 65-threshold gate
- [x] SMS alerts via Twilio
- [x] Push notifications via Pushover
- [x] FastAPI dashboard with TradingView candlestick chart
- [x] Paper trading mode
- [x] `quick_check.py` — instant price/news snapshot
- [x] Signal logger → `signals_log.csv`
- [x] Watchlist persistence (`watchlist.json`)
- [x] Market hours awareness (9:30 AM – 4:00 PM EST)
- [x] Daily report at 4:30 PM EST

### v1.1 — Next 🔜

- [ ] Intraday 5-min bars on dashboard (yfinance)
- [ ] Backtesting module with win rate stats
- [ ] SEC EDGAR insider trading signal (free)
- [ ] Signal accuracy tracking (actual vs predicted)
- [ ] Multi-ticker monitoring loop

### v1.2 — Planned 📋

- [ ] Pre-market news scanner (8:00 AM SMS summary)
- [ ] Market scanner — top movers each morning
- [ ] Multi-ticker watchlist on dashboard UI
- [ ] Email report option (SendGrid)

### v2.0 — Advanced 🚀

- [ ] Options flow via Unusual Whales ($50/mo)
- [ ] Dark pool prints via Unusual Whales ($50/mo)
- [ ] Real-time news via Benzinga ($50/mo)
- [ ] Upgrade to Polygon paid tier (WebSocket streaming)
- [ ] Full LangGraph refactor with parallel node execution
- [ ] Self-improving prompts based on signal accuracy history

---

## 11. Important Disclaimers

> ⚠️ **This software is for educational and research purposes only.**

- **Not financial advice.** Nothing in this project constitutes investment advice, a recommendation to buy or sell any security, or a solicitation of any kind.
- **Paper trade first.** Always run in `--paper` mode for a minimum of 30 days before considering any live use. Understand the signal accuracy on your specific tickers before trusting any output.
- **Past performance does not guarantee future results.** A high backtest win rate on historical data does not mean the strategy will perform the same going forward.
- **Never risk money you cannot afford to lose.** Algorithmic trading systems can and do produce losing trades, streaks of losses, and complete failures in certain market conditions.
- **The AI makes mistakes.** Claude is a language model, not a licensed financial analyst. Its reasoning can be incorrect, incomplete, or confidently wrong.
- **You are responsible for your own trading decisions.** The authors of this software accept no liability for financial losses incurred through its use.

---

## 12. Author

**Dan Nicolau**
Senior QA Engineer → AI QA Architect

- GitHub: [github.com/dannicolau7](https://github.com/dannicolau7)

---

*Built with Claude Sonnet 4.6, LangGraph, and a lot of coffee.*
