"""
news_agent.py — Multi-source sentiment:
  1. StockTwits   — crowd sentiment (free, no key, built for stocks)
  2. CNN Markets  — RSS headlines filtered by ticker
  3. Polygon news — already fetched by data_agent (raw_news in state)
  4. Claude Haiku — reads all headlines and returns a sentiment score + summary

Final score = StockTwits 30% + Claude-on-headlines 70%
"""

import json
import re
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import anthropic
from config import ANTHROPIC_API_KEY
from langsmith import traceable
from utils.tracing import annotate_run

# ── Per-source TTL caches ──────────────────────────────────────────────────────
# Avoids re-hitting the same external API on every scan cycle for the same ticker.
# Key: ticker str  →  Value: (fetched_at float, result)

_ST_CACHE:     dict = {}   # StockTwits — 5 min TTL
_REDDIT_CACHE: dict = {}   # Reddit     — 15 min TTL
_RSS_CACHE:    dict = {}   # RSS feeds  — 10 min TTL

_ST_TTL     = 5  * 60
_REDDIT_TTL = 15 * 60
_RSS_TTL    = 10 * 60


# ── StockTwits ─────────────────────────────────────────────────────────────────

def _social_velocity(messages: list) -> dict:
    """
    Measures how fast StockTwits mentions are accelerating.
    Compares messages in last 1h vs prior hours from the same batch.
    Returns {"velocity": float, "label": str}
    velocity > 3× = 🚀 surging
    velocity > 1.5× = 📈 accelerating
    velocity < 0.5× = 📉 fading
    """
    if not messages:
        return {"velocity": 1.0, "label": "no data"}
    try:
        now      = datetime.now(timezone.utc)
        last_1h  = 0
        prior    = 0
        for m in messages:
            created = m.get("created_at", "")
            if not created:
                continue
            pub = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_h = (now - pub).total_seconds() / 3600
            if age_h <= 1:
                last_1h += 1
            elif age_h <= 6:
                prior   += 1
        # Normalise prior to per-hour rate
        prior_per_h = prior / 5 if prior > 0 else 0.1
        velocity    = round(last_1h / prior_per_h, 2) if prior_per_h > 0 else 1.0
        if velocity >= 3:
            label = f"🚀 surging ({velocity:.1f}×)"
        elif velocity >= 1.5:
            label = f"📈 accelerating ({velocity:.1f}×)"
        elif velocity <= 0.5:
            label = f"📉 fading ({velocity:.1f}×)"
        else:
            label = f"➡️ steady ({velocity:.1f}×)"
        return {"velocity": velocity, "label": label}
    except Exception:
        return {"velocity": 1.0, "label": "unknown"}


def _fetch_stocktwits(ticker: str) -> tuple:
    cached = _ST_CACHE.get(ticker)
    if cached and (time.time() - cached[0]) < _ST_TTL:
        return cached[1]
    """
    Pull the latest 30 messages from StockTwits for ticker.
    Returns (sentiment, score_0_100, bull_count, bear_count, post_count).
    """
    try:
        r = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            timeout=10,
            headers={"User-Agent": "argus/1.0"},
        )
        if r.status_code == 403:
            # StockTwits public API now requires authentication — suppress silently
            return "NEUTRAL", 50, 0, 0, 0, {"velocity": 1.0, "label": "API blocked (403)"}
        if r.status_code != 200 or not r.text.strip():
            return "NEUTRAL", 50, 0, 0, 0, {"velocity": 1.0, "label": "no data"}
        messages = r.json().get("messages", [])
        bull = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment", {}) and
               m["entities"]["sentiment"].get("basic") == "Bullish"
        )
        bear = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment", {}) and
               m["entities"]["sentiment"].get("basic") == "Bearish"
        )
        velocity = _social_velocity(messages)
        total    = bull + bear
        if total == 0:
            return "NEUTRAL", 50, 0, 0, len(messages), velocity
        score     = round((bull / total) * 100)
        sentiment = "BULLISH" if score >= 60 else "BEARISH" if score <= 40 else "NEUTRAL"
        result = sentiment, score, bull, bear, len(messages), velocity
        _ST_CACHE[ticker] = (time.time(), result)
        return result
    except Exception as e:
        print(f"⚠️  [NewsAgent] StockTwits error: {e}")
        return "NEUTRAL", 50, 0, 0, 0, {"velocity": 1.0, "label": "no data"}


# ── CNN Markets RSS ────────────────────────────────────────────────────────────

# ── Reddit (extra signal, kept for small-cap chatter) ─────────────────────────

SUBREDDITS = ["wallstreetbets", "stocks", "investing", "pennystocks"]
REDDIT_HEADERS = {"User-Agent": "argus/1.0 (research tool)"}


def _fetch_reddit_headlines(ticker: str) -> list:
    """Return up to 8 post titles mentioning ticker from Reddit."""
    cached = _REDDIT_CACHE.get(ticker)
    if cached and (time.time() - cached[0]) < _REDDIT_TTL:
        return cached[1]
    titles = []
    for sub in SUBREDDITS:
        try:
            r = requests.get(
                f"https://www.reddit.com/r/{sub}/search.json",
                params={"q": ticker, "sort": "new", "limit": 10, "restrict_sr": "on"},
                headers=REDDIT_HEADERS,
                timeout=8,
            )
            children = r.json().get("data", {}).get("children", [])
            for c in children:
                t = c.get("data", {}).get("title", "")
                if t:
                    titles.append(t)
        except Exception:
            pass
        if len(titles) >= 8:
            break
    result = titles[:8]
    _REDDIT_CACHE[ticker] = (time.time(), result)
    return result


# ── CNN Markets RSS ────────────────────────────────────────────────────────────

CNN_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.investing.com/rss/news.rss",
]


def _fetch_market_headlines(ticker: str) -> list:
    """
    Parse financial RSS feeds (Yahoo Finance, MarketWatch) for ticker headlines.
    Returns list of headline strings (up to 5).
    """
    cached = _RSS_CACHE.get(ticker)
    if cached and (time.time() - cached[0]) < _RSS_TTL:
        return cached[1]
    ticker_upper = ticker.upper()
    headlines    = []
    feeds        = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    ]
    for feed_url in feeds:
        try:
            r = requests.get(feed_url, timeout=10,
                             headers={"User-Agent": "argus/1.0"},
                             verify=True)
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
            for item in root.iter("item"):
                title = item.findtext("title") or ""
                desc  = item.findtext("description") or ""
                text  = (title + " " + desc).upper()
                if ticker_upper in text or "MARKET" in text or "STOCK" in text:
                    if title and title not in headlines:
                        headlines.append(title)
            if len(headlines) >= 5:
                break
        except Exception as e:
            print(f"⚠️  [NewsAgent] RSS error ({feed_url.split('/')[2]}): {e}")
    result = headlines[:5]
    _RSS_CACHE[ticker] = (time.time(), result)
    return result


# ── Claude Haiku sentiment ─────────────────────────────────────────────────────

def _claude_sentiment(ticker: str, headlines: list) -> tuple:
    """
    Ask Claude Haiku to rate sentiment from all collected headlines.
    Returns (sentiment, score_0_100, summary_str).
    """
    if not headlines:
        return "NEUTRAL", 50, "No recent news found."

    lines = "\n".join(f"- {h}" for h in headlines[:8])
    prompt = (
        f"Analyze these recent headlines for stock {ticker}.\n\n"
        f"{lines}\n\n"
        f"Consider market context, company news, and sector trends.\n"
        f"Respond ONLY with this exact JSON (no markdown fences):\n"
        f'{{"sentiment":"BULLISH" or "BEARISH" or "NEUTRAL",'
        f'"score":<integer 0-100 where 100=extremely bullish>,'
        f'"summary":"<one sentence: key catalyst or risk driving this sentiment>"}}'
    )
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp   = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text  = parts[1][4:] if parts[1].startswith("json") else parts[1]
        result    = json.loads(text.strip())
        sentiment = str(result.get("sentiment", "NEUTRAL")).upper()
        score     = max(0, min(100, int(result.get("score", 50))))
        summary   = str(result.get("summary", ""))
        return sentiment, score, summary
    except Exception as e:
        print(f"⚠️  [NewsAgent] Claude sentiment error: {e}")
        return "NEUTRAL", 50, ""


# ── LangGraph node ─────────────────────────────────────────────────────────────

@traceable(name="news_agent", tags=["pipeline", "news", "llm"])
def news_node(state: dict) -> dict:
    ticker   = state["ticker"]
    annotate_run(state)
    raw_news = state.get("raw_news", [])

    print(f"📰 [NewsAgent] Analyzing sentiment for {ticker}...")

    # 1. StockTwits crowd sentiment + social velocity
    st_sentiment, st_score, bull, bear, n_posts, velocity = _fetch_stocktwits(ticker)
    if n_posts > 0:
        print(f"   📱 StockTwits: {st_sentiment}  {bull}🐂 {bear}🐻  ({n_posts} posts)  "
              f"score={st_score}/100  velocity={velocity['label']}")
    else:
        print(f"   📱 StockTwits: no posts found")

    # 2. Collect headlines — Polygon news already in state + RSS
    polygon_headlines = []
    for n in raw_news:
        t = n.get("title") or (n.get("content") or {}).get("title") or ""
        if t:
            polygon_headlines.append(t)
    if polygon_headlines:
        print(f"   📡 Polygon news: {len(polygon_headlines)} ticker-specific articles")

    market_headlines = _fetch_market_headlines(ticker)
    if market_headlines:
        print(f"   📺 Market news: {len(market_headlines)} headlines (Yahoo/MarketWatch)")
    else:
        print(f"   📺 Market news: no headlines found")

    reddit_headlines = _fetch_reddit_headlines(ticker)
    if reddit_headlines:
        print(f"   🤖 Reddit: {len(reddit_headlines)} posts ({', '.join(f'r/{s}' for s in SUBREDDITS)})")
    else:
        print(f"   🤖 Reddit: no posts found")

    all_headlines = polygon_headlines + market_headlines + reddit_headlines

    # 3. Claude sentiment on all headlines
    cl_sentiment, cl_score, cl_summary = _claude_sentiment(ticker, all_headlines)
    if all_headlines:
        print(f"   🤖 Claude news: {cl_sentiment}  score={cl_score}/100")
        if cl_summary:
            print(f"   📌 {cl_summary}")
    else:
        print(f"   🤖 Claude news: no headlines to analyze")

    # 4. Combined score — StockTwits 30%, Claude 70%
    if all_headlines and n_posts > 0:
        combined_score = round(st_score * 0.30 + cl_score * 0.70)
    elif all_headlines:
        combined_score = cl_score
    elif n_posts > 0:
        combined_score = st_score
    else:
        combined_score = 50

    if combined_score >= 60:
        final_sentiment = "BULLISH"
    elif combined_score <= 40:
        final_sentiment = "BEARISH"
    else:
        final_sentiment = "NEUTRAL"

    # Build summary string
    parts = []
    if n_posts > 0:
        parts.append(f"StockTwits {bull}🐂/{bear}🐻 ({n_posts} posts)")
    if cl_summary:
        parts.append(cl_summary)
    if not parts:
        parts.append(f"No strong sentiment signal for {ticker}.")
    summary = " | ".join(parts)

    emoji = "📈" if final_sentiment == "BULLISH" else "📉" if final_sentiment == "BEARISH" else "➡️"
    print(f"✅ [NewsAgent] {emoji} {final_sentiment}  combined={combined_score}/100")

    return {
        **state,
        "news_sentiment":   final_sentiment,
        "sentiment_score":  combined_score,
        "news_summary":     summary,
        "social_velocity":  velocity,
    }
