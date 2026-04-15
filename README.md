# Argus — Agentic Market Intelligence

> **An autonomous trading agent that perceives, reasons, acts, and learns.**
>
> 15 specialized agents run in parallel 24/7 — perceiving geopolitics, macro regimes, earnings catalysts, market breadth, social flow, SEC filings, and price spikes. A shared world model synthesizes all signals into real-time BUY / SELL / HOLD decisions via Claude AI. A feedback loop tracks every outcome, reflects daily on what worked, and adjusts its own behavior — no human intervention required.
>
> *Perceive → Reason → Act → Observe → Learn → Repeat better.*

**Version:** v4.0 &nbsp;|&nbsp; **Status:** Active Development

---

## 1. Project Overview

### What it does

Argus runs 15 background agents simultaneously to ensure no opportunity is missed — whether from your watchlist, macro rotation, news catalysts, or autonomous discovery.

#### Discovery Engine 1 — Watchlist Monitor (every 5 minutes, market hours)

Continuously monitors your saved tickers every 5 minutes during market hours (9:30 AM – 4:00 PM EST). For each cycle it:

1. Fetches real-time price and OHLCV bars (yfinance + Polygon)
2. Scores social sentiment (StockTwits bull/bear ratio + StockTwits velocity)
3. Calculates RSI, MACD, EMA stack, volume, ATR, VWAP, support and resistance
4. Detects chart patterns algorithmically (bull flag, double bottom, ascending triangle, cup & handle, breakout)
5. Classifies the latest news headline into a catalyst category
6. Computes relative strength vs SPY and QQQ across 1d / 5d / 20d
7. Detects the best-matching setup pattern (gap-and-go, breakout, pullback, oversold bounce)
8. Builds a 4-layer score: context + setup (with pattern bonus) + execution − risk penalties
9. Checks the VIX/SPY circuit breaker — suppresses BUY signals when market is too dangerous
10. Passes everything to Claude AI which decides BUY / SELL / HOLD with confidence and trade horizon
11. Fires a WhatsApp alert via Twilio when confidence ≥ 65; deduplicates so each position only alerts once

#### Discovery Engine 2 — Morning Market Scanner (7:45 AM EST daily)

Scans all ~6,000 US stocks every morning before the open to find the single best opportunity of the day. No watchlist needed — it finds stocks you've never heard of.

1. Applies Gate 0: dollar volume ≥ $100k (eliminates illiquid garbage)
2. Checks the circuit breaker — if VIX ≥ 25 or SPY drops ≥ 1.5%, skips the scan for the day
3. Runs the full 4-layer scoring pipeline (including pattern bonus) on every survivor
4. Ranks by total score and selects the top candidate
5. Sends a pre-market WhatsApp summary with entry zone, targets, stop, and trade horizon
6. If nothing qualifies, sends a "no pick today" WhatsApp so you always know the scan ran

#### Discovery Engine 3 — 24/7 News Watcher (every 5 minutes, around the clock)

Monitors news for **all** US stocks — not just your watchlist — by polling the Polygon news API every 5 minutes, day and night. When a new article appears on any ticker:

1. Quick gate: price $0.50–$50, average volume ≥ 50k (prevents garbage)
2. 4-hour cooldown per ticker (no alert storms)
3. Classifies the headline (FDA approval, earnings beat, dilution offering, etc.)
4. If bullish catalyst: runs the full pipeline and sends a WhatsApp alert

### Who it's for

- Retail traders who want AI-assisted signal generation on small/mid cap stocks
- Developers learning LangGraph multi-agent patterns with real financial data
- Anyone who wants a 24/7 news-driven discovery system without paying for premium data

---

## 2. Tech Stack

| Layer | Tool | Purpose | Cost |
|---|---|---|---|
| Agent orchestration | LangGraph + LangChain | Multi-agent pipeline, state management | Free |
| AI reasoning | Claude API (claude-sonnet-4-6) | Signal interpretation, confidence scoring, trade horizon | Pay-per-use |
| Historical data | Polygon.io | Daily OHLCV bars, news, ticker details | Free tier |
| Real-time data | yfinance | Live price, intraday 5-min candles, benchmarks, pre-market | Free |
| Social sentiment | StockTwits API | Bull/bear ratio + message velocity | Free, no auth |
| Pattern detection | `pattern_detector.py` | 5 chart patterns from raw OHLCV (algorithmic, no ML) | Free (local) |
| Backtesting | `backtester.py` | Walk-forward rule-based backtest on historical bars | Free (local) |
| Circuit breaker | `circuit_breaker.py` | VIX + SPY drop guard (suppresses BUY in danger zones) | Free (local) |
| News classification | `features/news_classifier.py` | Phrase-level catalyst detection | Free (local) |
| Relative strength | `features/relative_strength.py` | Multi-horizon RS vs SPY + QQQ | Free (local) |
| Setup detection | `setups/` (4 modules) | Gap-and-go, breakout, pullback, bounce | Free (local) |
| Self-learning | `self_learner.py` | Tracks signal win rates, adjusts weights | Free (local) |
| WhatsApp alerts | Twilio WhatsApp API | BUY/SELL signal delivery | Free sandbox |
| Push notifications | Pushover | iOS/Android push notifications | $5 one-time |
| Dashboard server | FastAPI + uvicorn | REST API + HTML serving | Free |
| Chart UI | TradingView Lightweight Charts | Candlestick chart, RSI panel, markers | Free |

---

## 3. Project Structure

```
argus/
│
├── main.py                    # Entry point — 15 async agent tasks, FastAPI server
├── config.py                  # API keys from .env
├── graph.py                   # LangGraph pipeline (fetch → news → tech → decide → alert)
├── analyzer.py                # Claude Sonnet signal analysis with full world context
├── world_context.py           # Thread-safe shared state — all agents write here
├── polygon_feed.py            # Polygon.io + yfinance data layer
├── alerts.py                  # Twilio WhatsApp + Pushover senders
├── logger.py                  # Signals log (CSV)
├── watchlist_manager.py       # Persist tickers to watchlist.json
├── circuit_breaker.py         # VIX + SPY safety gate
├── pattern_detector.py        # Chart pattern detection (5 patterns)
├── market_scanner.py          # Morning scan — ~6k stocks → best of day
├── scheduler.py               # Daily event scheduler (3 AM – 11 PM)
├── quick_check.py             # Instant snapshot for any ticker
│
├── ── Intelligence Agents (always-on background loops) ──
├── geo_watcher.py             # Geopolitical RSS → sector bias (every 30min)
├── macro_watcher.py           # Yields, VIX, DXY, oil, gold → regime (every 60min)
├── earnings_watcher.py        # Earnings calendar → catalysts + beat rates (every 4h)
├── breadth_watcher.py         # 11 sectors vs 20d MA → market health (every 30min)
├── social_watcher.py          # StockTwits, Congress trades, unusual options (every 60min)
├── news_watcher.py            # Polygon + Yahoo Finance news → pipeline trigger (every 90s)
├── spike_watcher.py           # Price + volume spike detection on ~200 tickers (every 60s)
├── edgar_watcher.py           # SEC 8-K RSS → filing catalyst detection (every 60s)
│
├── ── Agentic Feedback Loop ──
├── performance_tracker.py     # Logs every alert; checks 1d/3d/7d outcomes (SQLite)
├── portfolio_agent.py         # Open position monitor; fires EXIT alerts (every 5min)
├── reflection_agent.py        # Daily 4:15 PM review; adjusts confidence thresholds
├── discovery_agent.py         # Macro-driven sector discovery; builds dynamic scan list (every 4h)
│
├── agents/
│   ├── data_agent.py          # Node 1 — price, bars, volume, news
│   ├── news_agent.py          # Node 2 — sentiment + news classification
│   ├── tech_agent.py          # Node 3 — RSI, MACD, EMA stack, ATR, patterns
│   ├── decision_agent.py      # Node 4 — Claude signal + adaptive confidence threshold
│   └── alert_agent.py         # Node 5 — WhatsApp + push + performance logging
│
├── data/
│   ├── performance.db         # SQLite — signal history + outcomes
│   ├── learnings.json         # Reflection agent accumulated insights
│   └── discovery_log.jsonl    # Discovery agent history
│
├── setups/                    # Setup detectors (gap-and-go, breakout, pullback, bounce)
├── features/                  # Relative strength, news classifier
├── dashboard/index.html       # Live terminal — Signal, History, News, Watchlist, Portfolio
├── requirements.txt
├── .env / .env.example
└── watchlist.json
```

### Agent Pipeline Flow

```
fetch_data → analyze_news → analyze_tech → decide → alert
    │              │              │            │         │
 price/bars   StockTwits     RSI/MACD/EMA  Claude AI  WhatsApp
              RS vs SPY/QQQ  setup detect   4-layer    +Pushover
              news classify  pattern detect  score      (deduped)
                             VWAP/OBV       horizon
```

---

## 4. Signal Sources & Quality

| Signal | Source | Cost | Status | Reliability |
|---|---|---|---|---|
| RSI (14) | Calculated from daily bars | Free | ✅ | ⭐⭐⭐⭐ |
| MACD (12/26/9) | Calculated from daily bars | Free | ✅ | ⭐⭐⭐⭐ |
| EMA Stack (9/21/50) | Calculated from daily bars | Free | ✅ | ⭐⭐⭐⭐ |
| Volume / RVOL | yfinance / Polygon | Free | ✅ | ⭐⭐⭐⭐⭐ |
| ATR | Calculated from OHLCV | Free | ✅ | ⭐⭐⭐⭐ |
| VWAP | Calculated from intraday bars | Free | ✅ | ⭐⭐⭐⭐ |
| Support / Resistance | 20-bar high/low | Free | ✅ | ⭐⭐⭐⭐ |
| Relative Strength (1d/5d/20d) | vs SPY + QQQ via yfinance | Free | ✅ | ⭐⭐⭐⭐⭐ |
| Chart patterns (5) | pattern_detector.py (algorithmic) | Free | ✅ | ⭐⭐⭐⭐ |
| Trade horizon | Claude AI (intraday/swing/position) | Pay-per-use | ✅ | ⭐⭐⭐⭐ |
| VIX circuit breaker | yfinance ^VIX (15-min cache) | Free | ✅ | ⭐⭐⭐⭐⭐ |
| Pre-market gap | yfinance prepost=True (1-min bars) | Free | ✅ | ⭐⭐⭐⭐ |
| Sector rotation | 11 SPDR ETFs via yfinance | Free | ✅ | ⭐⭐⭐⭐ |
| Social sentiment | StockTwits bull/bear + velocity | Free | ✅ | ⭐⭐⭐ |
| News classification | Polygon headlines (local NLP) | Free | ✅ | ⭐⭐⭐⭐ |
| Setup pattern | gap_and_go / breakout / pullback / bounce | Free | ✅ | ⭐⭐⭐⭐ |
| Self-learned weights | CSV signal win-rate tracker | Free | ✅ | ⭐⭐⭐ |
| Options flow | Unusual Whales | $50/mo | ❌ Not yet | ⭐⭐⭐⭐⭐ |
| Dark pool prints | Unusual Whales | $50/mo | ❌ Not yet | ⭐⭐⭐⭐⭐ |
| Insider trading | SEC EDGAR | Free | ❌ Not yet | ⭐⭐⭐⭐⭐ |

---

## 5. Scoring System

Scores are built in four independent layers. There is no fixed maximum — a perfect setup with multiple pattern bonuses can score 165+, a weak one 30.

### Layer 1 — Context Score (market fit)

| Signal | Condition | Points |
|---|---|---|
| Relative strength | RS composite ≥ +12% vs SPY+QQQ | +30 |
| Relative strength | RS composite ≥ +7% | +22 |
| Relative strength | RS composite ≥ +3% | +12 |
| Dollar volume | ≥ $5M/day | +15 |
| Dollar volume | ≥ $1M/day | +8 |
| Bullish catalyst | FDA approval, earnings beat, contract win | +10 to +25 |
| Small cap | Market cap ≤ $500M | +10 |
| Price | < $10 (momentum-friendly) | +5 |
| Relative strength | RS composite ≤ -7% (lagging market) | −20 |

### Layer 2 — Setup Score (pattern quality + chart pattern bonus)

All qualifying setups are scored; the **highest-scoring** setup wins (not the first match).

| Setup | Hard Filters | Max Score |
|---|---|---|
| Gap and Go | Gap ≥ 3%, RVOL ≥ 3x, RSI ≤ 67 | ~120 |
| Oversold Bounce | RSI ≤ 35, RVOL ≥ 2x, within 5% of 20-bar support | ~110 |
| Breakout | Price ≥ 20-day high, RVOL ≥ 2x, RSI 50–67 | ~115 |
| First Pullback | EMA9 > EMA21 > EMA50, near EMA9, RSI 38–55 | ~90 |
| General | No pattern — RVOL and news bonus only | ~50 |

**Chart pattern bonus** (additive on top of setup score):

| Pattern | Bonus |
|---|---|
| Bull Flag | +15 pts |
| Cup & Handle | +15 pts |
| Double Bottom | +12 pts |
| Ascending Triangle | +12 pts |
| Breakout (20-day high + vol ≥ 1.5×) | +8 pts |

### Layer 3 — Execution Score (technical confirmation only)

| Signal | Condition | Points |
|---|---|---|
| MACD | Bullish crossover (hist crosses 0 from below) | +20 |
| MACD | Histogram positive | +8 |
| EMA cross | EMA9 crosses above EMA21 | +15 |
| EMA cross | EMA9 > EMA21 (aligned bullish) | +6 |
| Fib support | Price within 2% of 38.2% or 61.8% fib | +12 |
| Price support | Within 3% of 20-bar low | +10 |

### Layer 4 — Risk Penalty

| Signal | Condition | Penalty |
|---|---|---|
| Bearish catalyst | Dilution, FDA rejection, earnings miss | −15 to −25 |
| Overbought | RSI ≥ 75 | −20 |
| Overbought | RSI ≥ 68 | −10 |

### Alert threshold

```
VIX < 25 AND SPY drop < 1.5%  (circuit breaker safe)
  AND  Total score ≥ 68
  AND  Claude confidence ≥ 65
  AND  ticker NOT already in active position  (dedup guard)
→  WhatsApp alert fires
Otherwise  →  HOLD (no alert sent)
```

Score breakdown is shown in every WhatsApp alert:

```
Score: 142 (ctx=45 setup=67 exec=50 risk=-20)
```

---

## 6. Chart Pattern Detection

`pattern_detector.py` scans the last 30 bars of daily OHLCV data algorithmically — no machine learning, no API calls, fully deterministic.

| Pattern | Detection Logic | Confidence |
|---|---|---|
| **Bull Flag** | Pole: ≥8% gain over 3-5 bars. Flag: last 5 bars range < 3%, volume declining | 0.0–1.0 |
| **Double Bottom** | Two lows within 3% of each other in last 30 bars, recovery ≥5% between | 0.0–1.0 |
| **Ascending Triangle** | Last 15 bars: flat resistance (highs within 2%), rising lows (positive slope) | 0.0–1.0 |
| **Cup & Handle** | 15–30 bar U-shape near prior high, followed by 3–5 bar handle < 50% of cup depth | 0.0–1.0 |
| **Breakout** | Price crosses 20-day high today, volume ≥ 1.5× average | 0.0–1.0 |

Patterns appear as colored pills on the **Signal tab** of the dashboard, and are included in the prompt Claude receives for final reasoning.

---

## 7. Trade Horizon Prediction

Claude classifies every BUY signal into one of three horizons:

| Horizon | Meaning | Typical Hold |
|---|---|---|
| **Intraday** | RSI > 65 on daily, at resistance, mixed EMA stack, or earnings ≤ 3 days | Exit before close |
| **Swing** | EMA stack bullish, MACD just turned positive, RSI 40–65 | 2–5 days |
| **Position** | Major catalyst, RSI bouncing from < 35 on high volume | 1+ week |

The horizon and one-sentence reason appear in the WhatsApp alert and on the Signal tab dashboard badge.

---

## 8. VIX + SPY Circuit Breaker

`circuit_breaker.py` acts as a market-wide safety gate. It is checked:

- At the start of every `scan_best_of_day()` call — aborts the entire scan if unsafe
- In `analyzer.py` after Claude's response — overrides a BUY to HOLD with a warning message

**Thresholds (configurable in `.env`):**

| Variable | Default | Meaning |
|---|---|---|
| `VIX_THRESHOLD` | 25 | VIX above this → circuit breaker triggers |
| `SPY_DROP_THRESHOLD` | -1.5 | SPY intraday drop % below this → triggers |

Results are cached for 15 minutes to avoid redundant downloads across multiple ticker scans.

---

## 9. Pre-Market Gap Scanner

`premarket_scanner.py` runs at **8:30 AM EST** every trading day via the scheduler. It finds stocks that have already moved significantly before the market opens.

- Data: yfinance 1-minute bars with `prepost=True`
- Filters: gap ≥ 3%, pre-market volume ≥ 30k, price $0.50–$100
- Parallel fetches: 10 concurrent workers via `ThreadPoolExecutor`
- Sends top 3 gaps to WhatsApp; stores top 10 in the `/api/premarket` endpoint

```
⚡ Pre-Market Gaps (8:30 AM ET)
BZAI  +8.5%  $2.18  vol 2.3M
NVDA  +4.1%  $142.30  vol 890k
SMCI  +3.2%  $28.40  vol 540k
💡 Confirm with volume at open
```

---

## 10. Backtester

`backtester.py` runs a **walk-forward rule-based backtest** on 1–2 years of historical daily bars. It uses the same quantitative scoring as the live scanner — no Claude API, so it can process thousands of bars cheaply.

### Logic

- Scores each bar using RSI, MACD histogram, EMA alignment, volume ratio, gap %, Bollinger, and support
- Triggers a simulated BUY at the **next bar's open** when score ≥ 60 (no lookahead bias)
- Exit conditions: stop loss (−5% or −1 ATR), target (+8%), or 5-bar timeout
- Tracks: win rate, avg gain %, avg loss %, total P&L %, max drawdown %, trade list

### Run via API

```
POST /api/backtest
{"tickers": ["AAPL", "TSLA"], "period": "1y"}
→ {win_rate, avg_gain_pct, avg_loss_pct, total_pnl_pct, max_drawdown_pct, total_trades, trades[]}
```

Results are shown in the **Performance tab** of the dashboard.

---

## 11. API Keys Required

| Variable | Where to Get It | Cost |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Pay-per-use |
| `POLYGON_API_KEY` | [polygon.io/dashboard](https://polygon.io/dashboard) | Free tier |
| `TWILIO_ACCOUNT_SID` | [twilio.com/console](https://twilio.com/console) | Free sandbox |
| `TWILIO_AUTH_TOKEN` | [twilio.com/console](https://twilio.com/console) | Free sandbox |
| `TWILIO_FROM_NUMBER` | Twilio console → WhatsApp Sandbox | `whatsapp:+14155238886` |
| `TWILIO_TO_NUMBER` | Your WhatsApp number | e.g. `whatsapp:+40...` |
| `PUSHOVER_APP_TOKEN` | [pushover.net](https://pushover.net) → Your Apps | $5 one-time (optional) |
| `PUSHOVER_USER_KEY` | [pushover.net](https://pushover.net) → Settings | $5 one-time (optional) |

### `.env` setup

```bash
cp .env.example .env
nano .env
```

```env
ANTHROPIC_API_KEY=sk-ant-...
POLYGON_API_KEY=...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=whatsapp:+14155238886
TWILIO_TO_NUMBER=whatsapp:+40xxxxxxxxx
PUSHOVER_APP_TOKEN=...        # optional
PUSHOVER_USER_KEY=...         # optional
MONITOR_INTERVAL=300
VIX_THRESHOLD=25              # optional — default 25
SPY_DROP_THRESHOLD=-1.5       # optional — default -1.5
```

---

## 12. How to Run

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

# Quick price and news snapshot
python3 quick_check.py BZAI AAPL NVDA
```

### Run the agent

```bash
# Paper trading — full pipeline, no real WhatsApp alerts sent
python3 main.py --ticker BZAI AWRE LTRX BBAI SOUN --paper

# Live monitoring — fires real WhatsApp alerts
python3 main.py --ticker BZAI AWRE LTRX BBAI SOUN

# Monitor tickers from watchlist.json (no --ticker flag needed)
python3 main.py --paper

# Custom scan interval (every 2 minutes)
python3 main.py --ticker BZAI --interval 120

# Custom dashboard port
python3 main.py --ticker BZAI --port 8080
```

`main.py` starts all three engines automatically:
- Watchlist monitoring loop (every `MONITOR_INTERVAL` seconds, market hours)
- Daily scheduler (3:00 AM through 11:00 PM, 11 timed events)
- 24/7 news watcher (every 5 minutes, all clocks)

### Watchlist management

```bash
python3 main.py --add NVDA        # add ticker to watchlist.json
python3 main.py --remove NVDA     # remove ticker from watchlist.json
python3 main.py --list            # print current watchlist
```

### Run the morning scanner manually

```bash
# Scan all ~6,000 US stocks and show the best pick (paper — no alert)
python3 market_scanner.py --paper

# Live run — sends WhatsApp if a strong pick is found
python3 market_scanner.py
```

### Backtesting

```bash
# Via API (while main.py is running)
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "TSLA"], "period": "1y"}'

# Or use the backtest tab in the dashboard UI
```

### Dashboard

Once `main.py` is running, open your browser:

```
http://localhost:8000
```

**Tabs:**
- **Signal** — live signal card with confidence gauge, horizon badge, chart patterns, entry/targets/stop
- **History** — all past BUY/SELL signals with prices and confidence
- **News** — latest Polygon news headlines for monitored tickers
- **Watchlist** — add / remove tickers without restarting
- **Performance** — win rate, P&L stats, VIX/circuit breaker status, sector rotation heatmap, pre-market gaps, backtest runner

---

## 13. Alert Format

Every BUY or SELL signal is sent as a WhatsApp message:

```
🟢 BUY — BZAI  [gap_and_go]
Price:      $1.7900
Entry Zone: $1.75 – $1.85
Targets:    $1.95 / $2.10 / $2.35
Stop Loss:  $1.62
RSI:        34.2  |  RVOL: 4.8x
Score: 142 (ctx=45 setup=67 exec=50 risk=-20)
Horizon:    Swing (2–5 days) — EMA stack bullish, MACD just crossed positive
Signals: gap +8.3%, RVOL 4.8x 🔥, RS +9.2% vs mkt, earnings beat 🟢

RSI oversold with gap-and-go setup on earnings beat catalyst.
RVOL 4.8x confirms institutional entry near the $1.65 support.
```

HOLD signals are **never** sent. Duplicate BUY alerts for the same open position are **suppressed** — you get one alert when the signal fires, not one every 5 minutes.

---

## 14. Daily Scheduler

The scheduler runs 11 timed events each trading day (Mon–Fri). The 3 AM overnight scan runs every night including weekends.

| Time (EST) | Event |
|---|---|
| 3:00 AM | Overnight news scan — WhatsApp if score ≥ 85 |
| 7:45 AM | Best-of-Day selection — full scanner → WhatsApp pick (or "no pick today") |
| 8:30 AM | Pre-market gap scanner — top gap movers → WhatsApp |
| 9:25 AM | "Market opens in 5 min" alert with top watch ticker |
| 9:30 AM | Live monitoring starts (handled by `main.py` monitoring loop) |
| 4:00 PM | Market closed + open positions summary → WhatsApp |
| 4:30 PM | Daily performance report → WhatsApp |
| 6:00 PM | Earnings scan — WhatsApp if score ≥ 75 |
| 8:00 PM | Evening news scan — WhatsApp if score ≥ 85 |
| 10:00 PM | Final scan — WhatsApp if score ≥ 85 |
| 11:00 PM | Good night summary → graceful shutdown |

---

## 15. News Classification

Headlines from the Polygon news API are classified into 10 categories using phrase-level matching (not single words — to prevent false positives like "cloud offering" triggering dilution).

| Category | Example Phrases | Score Adj |
|---|---|---|
| fda_approval | "FDA approves", "FDA clearance", "breakthrough designation" | +25 |
| earnings_beat | "beats estimates", "raises guidance", "record revenue" | +25 |
| contract_win | "wins contract", "awarded contract", "defense contract" | +20 |
| partnership | "strategic partnership", "joint venture", "licensing agreement" | +15 |
| upgrade | "upgrades to buy", "price target raised", "initiates buy" | +10 |
| general | No match | 0 |
| downgrade | "downgrades to sell", "price target cut" | −10 |
| earnings_miss | "misses estimates", "lowered guidance", "swings to loss" | −20 |
| offering_dilution | "secondary offering", "private placement", "prospectus supplement" | −25 |
| fda_rejection | "FDA rejects", "complete response letter", "refuse to file" | −25 |

---

## 16. Self-Learner

`self_learner.py` reads `best_picks_log.csv` after the market closes and updates signal weights based on which signals were present in winning vs losing picks.

**Tracked signals:**

| Signal | CSV column | Notes |
|---|---|---|
| Volume spike | `rvol` | RVOL ≥ 3x tagged as volume signal |
| RSI bounce | `rsi` | RSI ≤ 35 at entry |
| RSI momentum | `rsi` | RSI 40–55 in trend |
| MACD cross | `macd_cross` | 1 = bullish crossover at entry |
| EMA stack | `ema_cross` | 1 = EMA9 > EMA21 > EMA50 |
| Gap | `gap_pct` | Gap ≥ 3% |
| Bullish news | `news_category` | Any BULLISH_CATEGORIES hit |

Weights are loaded at scanner startup and adjust the scoring multipliers on each signal over time.

---

## 17. Data Sources Explained

### Polygon.io (free tier)

- **Provides:** Adjusted daily OHLCV bars (up to 2 years), previous-day close, company details, news headlines
- **Rate limit:** 5 API calls/minute
- **Used for:** Historical bars (RSI/MACD/EMA), news feed, ticker metadata, all-stock news polling (news watcher)

### yfinance (free, no key)

- **Provides:** Real-time last price, intraday bars (1m/5m/15m/1h), SPY/QQQ benchmark closes, pre-market bars, sector ETF data, VIX
- **Used for:** Live price, dashboard candles, RS computation, pre-market gaps, sector heatmap, circuit breaker VIX
- **Limitation:** Unofficial API, occasional rate-limiting; Polygon is the fallback for price

### StockTwits (free, no key)

- **Endpoint:** `https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json`
- **Provides:** Bull/bear ratio (% of messages tagged bullish vs bearish), message count velocity
- **Scoring:** Bull ratio ≥ 65% → +15 pts; velocity ≥ 2x → additional +10 pts
- **Limitation:** No message text — ratio and count only; thin coverage on very small caps

---

## 18. Roadmap

### v1.0 — Shipped ✅

- [x] Multi-agent LangGraph pipeline (data → news → tech → decision → alert)
- [x] RSI, MACD, Bollinger Bands, volume spike, support/resistance
- [x] Claude AI confidence scoring with 65-threshold gate
- [x] WhatsApp alerts via Twilio
- [x] Push notifications via Pushover
- [x] FastAPI dashboard with TradingView candlestick chart
- [x] Paper trading mode
- [x] Signal logger → `signals_log.csv`
- [x] Watchlist persistence (`watchlist.json`)
- [x] Market hours awareness (9:30 AM – 4:00 PM EST)
- [x] Daily report at 4:30 PM EST

### v2.0 — Shipped ✅

- [x] Multi-ticker monitoring loop
- [x] Morning market scanner (7:45 AM, ~6,000 stocks)
- [x] 24/7 news watcher — discovers unknown stocks from news events
- [x] StockTwits sentiment (replaces Reddit)
- [x] 4-layer scoring: context + setup + execution − risk penalty
- [x] 4 named setup patterns with individual hard filters and scoring
- [x] Best-match setup detection (all qualifying setups scored, highest wins)
- [x] Relative strength vs SPY + QQQ (1d / 5d / 20d weighted composite)
- [x] News classification into 10 catalyst categories (phrase-level)
- [x] Self-learner — tracks 7 signal win rates across CSV history
- [x] Forward return accuracy (actual historical close, not spot price)
- [x] Dollar volume Gate 0 (eliminates illiquid stocks before scoring)
- [x] Score breakdown in every WhatsApp alert
- [x] Daily scheduler with 10 timed events

### v3.0 — Current ✅

- [x] Trade horizon prediction — Claude classifies every BUY as intraday / swing / position
- [x] Chart pattern detection — 5 patterns (bull flag, double bottom, ascending triangle, cup & handle, breakout) with score bonus
- [x] Pre-market gap scanner — 8:30 AM WhatsApp with top gap movers
- [x] Walk-forward backtester — rule-based, no Claude API, via `/api/backtest`
- [x] Sector rotation heatmap — 11 SPDR ETFs, color-coded by day %, on Performance tab
- [x] VIX + SPY circuit breaker — suppresses BUY signals when market is in danger
- [x] Performance dashboard tab — win rate, P&L stats, recent picks, VIX gauge
- [x] Duplicate alert dedup guard — one WhatsApp per open position, not one per scan cycle
- [x] "No pick today" WhatsApp when morning scanner finds nothing qualifying

### v4.0 — Planned 📋

- [ ] Alpaca paper/live order execution — auto-place market orders from BUY signals
- [ ] Position sizing — risk-based share calculation (e.g. 2% account risk per trade)
- [ ] Global rate limiter shared across all three discovery engines
- [ ] SEC EDGAR insider trading signal (free)
- [ ] Options flow via Unusual Whales ($50/mo)
- [ ] Dark pool prints via Unusual Whales ($50/mo)
- [ ] Parallel node execution in LangGraph pipeline
- [ ] Email digest option (SendGrid)

---

## 19. Important Disclaimers

> ⚠️ **This software is for educational and research purposes only.**

- **Not financial advice.** Nothing in this project constitutes investment advice, a recommendation to buy or sell any security, or a solicitation of any kind.
- **Paper trade first.** Always run in `--paper` mode for a minimum of 30 days before considering any live use.
- **Past performance does not guarantee future results.** A high backtest win rate on historical data does not mean the strategy will perform the same going forward.
- **Never risk money you cannot afford to lose.** Algorithmic trading systems can and do produce losing trades, streaks of losses, and complete failures in certain market conditions.
- **The AI makes mistakes.** Claude is a language model, not a licensed financial analyst. Its reasoning can be incorrect, incomplete, or confidently wrong.
- **You are responsible for your own trading decisions.** The authors of this software accept no liability for financial losses incurred through its use.

---

## 20. Author

**Dan Nicolau**
Senior QA Engineer → AI QA Architect

- GitHub: [github.com/dannicolau7](https://github.com/dannicolau7)

---

*Built with Claude Sonnet 4.6, LangGraph, and a lot of coffee.*
