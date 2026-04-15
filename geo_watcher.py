"""
geo_watcher.py — Geopolitical & macro news intelligence agent.

Polls Reuters, AP, and Google News RSS feeds every 30 minutes.
Uses Claude Haiku (cost-efficient) to batch-analyze headlines for:
  - Which sectors/ETFs are affected
  - Direction (bullish/bearish) and magnitude
  - Overall market bias
  - Key risk themes

Results stored in world_context.geo and injected into every Claude
analyzer prompt, giving the signal engine full geopolitical awareness.

No API keys required — pure RSS feeds + existing Anthropic key.
"""

import asyncio
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import requests
import anthropic

from config import ANTHROPIC_API_KEY
import world_context as wctx

GEO_INTERVAL = 30 * 60   # 30 minutes between full RSS sweeps

# RSS feeds — mix of business news and geopolitical/world news
RSS_FEEDS = [
    ("Reuters Business",  "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters World",     "https://feeds.reuters.com/Reuters/worldNews"),
    ("AP Business",       "https://apnews.com/hub/business/rss"),
    ("AP Politics",       "https://apnews.com/hub/politics/rss"),
    ("Google Economy",    "https://news.google.com/rss/search?q=economy+federal+reserve+interest+rates&hl=en-US&gl=US&ceid=US:en"),
    ("Google Geopolitics","https://news.google.com/rss/search?q=geopolitics+war+trade+sanctions+market+impact&hl=en-US&gl=US&ceid=US:en"),
    ("Google Sectors",    "https://news.google.com/rss/search?q=oil+energy+semiconductor+pharma+FDA+stock+market&hl=en-US&gl=US&ceid=US:en"),
]

_seen_guids: set = set()
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ── RSS parsing ────────────────────────────────────────────────────────────────

def _parse_feed(url: str, label: str) -> list:
    """Fetch and parse an RSS 2.0 or Atom feed. Returns list of article dicts."""
    try:
        r = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "stock-ai-agent/3.0 (market research)"},
        )
        if r.status_code != 200:
            return []

        root = ET.fromstring(r.text)

        # Try RSS 2.0 <item> first, then Atom <entry>
        items = root.findall(".//item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        articles = []
        for item in items[:15]:
            title = (
                item.findtext("title")
                or item.findtext("{http://www.w3.org/2005/Atom}title")
                or ""
            ).strip()
            guid = (
                item.findtext("guid")
                or item.findtext("link")
                or item.findtext("{http://www.w3.org/2005/Atom}id")
                or ""
            ).strip()

            if not title or not guid or guid in _seen_guids:
                continue

            articles.append({"title": title, "guid": guid, "source": label})

        return articles

    except Exception as e:
        print(f"⚠️  [GeoAgent] RSS error ({label}): {e}")
        return []


def _collect_new_articles() -> list:
    """Fetch all RSS feeds, return only unseen articles, mark as seen."""
    all_articles = []
    for label, url in RSS_FEEDS:
        arts = _parse_feed(url, label)
        all_articles.extend(arts)

    # Deduplicate by guid across feeds, then mark seen
    seen_this_run = set()
    new_articles  = []
    for art in all_articles:
        if art["guid"] not in seen_this_run:
            seen_this_run.add(art["guid"])
            new_articles.append(art)

    for art in new_articles:
        _seen_guids.add(art["guid"])

    # Trim seen set to prevent unbounded growth
    if len(_seen_guids) > 10_000:
        _seen_guids.clear()

    return new_articles


# ── Claude Haiku analysis ──────────────────────────────────────────────────────

def _analyze_with_claude(articles: list) -> dict:
    """
    Send a batch of headlines to Claude Haiku for market impact analysis.
    Uses Haiku (not Sonnet) for cost efficiency — runs every 30 min.
    Returns structured geo intelligence dict.
    """
    if not articles:
        return {}

    headlines_text = "\n".join(
        f"{i+1}. [{a['source']}] {a['title']}"
        for i, a in enumerate(articles[:20])
    )

    prompt = f"""You are a senior market strategist at a hedge fund. Analyze these news headlines for their impact on US financial markets.

HEADLINES (last 30 minutes):
{headlines_text}

Instructions:
1. Identify headlines with MEANINGFUL market impact (skip noise/minor stories)
2. For each meaningful headline, identify affected sectors using ETF tickers (XLK, XLE, XLF, XLV, XLY, XLP, XLI, XLRE, XLB, XLU, XLC) or stock tickers if specific
3. Determine direction: BULLISH or BEARISH for those sectors
4. Rate magnitude: low / medium / high

Then provide overall market assessment.

Respond ONLY with valid JSON, no markdown:
{{
  "events": [
    {{
      "headline": "short version of headline",
      "sectors": ["XLE", "CVX"],
      "direction": "BULLISH",
      "magnitude": "high",
      "impact": "one sentence explaining the market impact"
    }}
  ],
  "overall_bias": "NEUTRAL",
  "hot_sectors": ["XLE", "XLV"],
  "cold_sectors": ["XLK"],
  "risk_summary": "two sentences summarizing key risks and opportunities right now"
}}

If no headlines have meaningful market impact, return: {{"events": [], "overall_bias": "NEUTRAL", "hot_sectors": [], "cold_sectors": [], "risk_summary": "No major market-moving events in current news cycle."}}"""

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if "```" in text:
            text = text.split("```")[-2] if text.count("```") >= 2 else text
            text = text.lstrip("json").strip()

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        print(f"⚠️  [GeoAgent] JSON parse error: {e}")
        return {}
    except Exception as e:
        print(f"⚠️  [GeoAgent] Claude error: {e}")
        return {}


# ── Single sweep ───────────────────────────────────────────────────────────────

def _run_sweep() -> bool:
    """Fetch all feeds, analyze with Claude, update world_context. Returns True if updated."""
    print(f"🌍 [GeoAgent] Sweeping {len(RSS_FEEDS)} RSS feeds...")
    articles = _collect_new_articles()

    if not articles:
        print(f"🌍 [GeoAgent] No new articles")
        return False

    print(f"🌍 [GeoAgent] {len(articles)} new headline(s) — analyzing with Claude Haiku...")
    result = _analyze_with_claude(articles)

    if not result:
        return False

    events       = result.get("events", [])
    overall_bias = result.get("overall_bias", "NEUTRAL")
    hot_sectors  = result.get("hot_sectors", [])
    cold_sectors = result.get("cold_sectors", [])
    risk_summary = result.get("risk_summary", "")

    wctx.update_geo({
        "events":       events,
        "overall_bias": overall_bias,
        "hot_sectors":  hot_sectors,
        "cold_sectors": cold_sectors,
        "risk_summary": risk_summary,
    })

    bias_icon = "🟢" if overall_bias == "BULLISH" else "🔴" if overall_bias == "BEARISH" else "🟡"
    print(f"🌍 [GeoAgent] {bias_icon} Bias={overall_bias}  "
          f"Hot={','.join(hot_sectors) or 'none'}  Cold={','.join(cold_sectors) or 'none'}")
    print(f"🌍 [GeoAgent] {len(events)} market-moving event(s) extracted")
    if risk_summary:
        print(f"🌍 [GeoAgent] {risk_summary[:100]}")

    return True


# ── Main loop ──────────────────────────────────────────────────────────────────

async def geo_watcher_loop():
    """
    Async loop started in main.py lifespan.
    Runs immediately at startup, then every 30 minutes.
    """
    print(f"🌍 [GeoAgent] Started — sweeping every {GEO_INTERVAL//60}min")

    loop = asyncio.get_running_loop()

    # Seed seen GUIDs at startup so we don't flood old articles on first run
    print("🌍 [GeoAgent] Seeding RSS history...")
    for label, url in RSS_FEEDS:
        arts = _parse_feed(url, label)
        for a in arts:
            _seen_guids.add(a["guid"])
    print(f"🌍 [GeoAgent] Seeded {len(_seen_guids)} existing articles. Running first analysis...")

    # Run immediately at startup with all seeded articles for initial world context
    # (process all current headlines once to build initial context)
    _seen_guids.clear()
    await loop.run_in_executor(None, _run_sweep)

    while True:
        try:
            await asyncio.sleep(GEO_INTERVAL)
            await loop.run_in_executor(None, _run_sweep)
        except asyncio.CancelledError:
            print("🌍 [GeoAgent] Stopped.")
            break
        except Exception as e:
            print(f"❌ [GeoAgent] Loop error: {e}")
            await asyncio.sleep(300)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== GeoAgent Standalone Test ===\n")
    _run_sweep()
    ctx = wctx.get()
    geo = ctx["geo"]
    print(f"\nOverall Bias: {geo['overall_bias']}")
    print(f"Hot sectors:  {geo['hot_sectors']}")
    print(f"Cold sectors: {geo['cold_sectors']}")
    print(f"\nEvents ({len(geo['events'])}):")
    for e in geo["events"]:
        icon = "🟢" if e["direction"] == "BULLISH" else "🔴"
        print(f"  {icon} [{','.join(e['sectors'])}] {e['headline'][:70]}")
        print(f"     {e['impact']}")
    print(f"\nRisk Summary: {geo['risk_summary']}")
    print(f"\n--- Prompt section preview ---")
    print(wctx.build_prompt_section())
