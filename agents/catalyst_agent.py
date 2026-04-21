"""
agents/catalyst_agent.py — Claude-based news catalyst classifier for swing trading.

classify_catalyst(ticker, news_items) → dict:
  catalyst_type    str   REGULATORY | PARTNERSHIP | ANALYST_UPGRADE |
                         ANALYST_DOWNGRADE | LAWSUIT_FRAUD | CEO_DEPARTURE |
                         EARNINGS | GENERAL
  catalyst_weight  int   -90 to +90  (signed; EARNINGS always 0)
  summary          str   one sentence
  is_tradeable     bool  True only when abs(weight) >= 60 and type != EARNINGS
  direction        str   "bullish" | "bearish" | "neutral"

Results are cached for 1 hour keyed by (ticker, sorted headline fingerprint).
Falls back to features.news_classifier on any Claude error.
"""

import hashlib
import json
import threading
from datetime import datetime

import anthropic
from langsmith import traceable

from config import ANTHROPIC_API_KEY

# ── Catalyst type → default unsigned magnitude ─────────────────────────────────
# Direction is determined by Claude; sign is applied in _apply_direction().
CATALYST_WEIGHTS: dict[str, int] = {
    "REGULATORY":        90,   # FDA approval/review/policy — signed by direction
    "PARTNERSHIP":       75,
    "ANALYST_UPGRADE":   60,
    "ANALYST_DOWNGRADE": -60,  # inherently negative
    "LAWSUIT_FRAUD":     -85,  # inherently negative
    "CEO_DEPARTURE":     -80,  # inherently negative
    "EARNINGS":           0,   # flagged as SKIP
    "GENERAL":           20,
}

_NEGATIVE_BY_DEFAULT = {"ANALYST_DOWNGRADE", "LAWSUIT_FRAUD", "CEO_DEPARTURE"}
_INHERENTLY_NEUTRAL  = {"EARNINGS"}

MIN_TRADEABLE_WEIGHT = 60   # abs(weight) must reach this for is_tradeable=True
CACHE_TTL_S          = 3_600   # 1 hour

_SYSTEM_PROMPT = (
    "You are a strict news catalyst classifier for swing trading. "
    "Your job is to find reasons to AVOID trades, not hype them. "
    "Only mark is_tradeable=True if catalyst is confirmed news, "
    "not speculation or rumor. Return JSON only."
)

_VALID_TYPES = set(CATALYST_WEIGHTS.keys())

# ── Cache ──────────────────────────────────────────────────────────────────────

_cache: dict      = {}
_cache_lock       = threading.Lock()

# ── Anthropic client ───────────────────────────────────────────────────────────

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_title(item: dict) -> str:
    """Handle both Polygon (top-level 'title') and yfinance ('content.title') formats."""
    return (
        item.get("title")
        or (item.get("content") or {}).get("title")
        or ""
    ).strip()


def _cache_key(ticker: str, headlines: list[str]) -> str:
    blob = ticker + "|" + "|".join(sorted(headlines))
    return hashlib.md5(blob.encode()).hexdigest()


def _apply_direction(catalyst_type: str, direction: str) -> int:
    """
    Return the signed catalyst_weight.
    Inherently negative types ignore direction; REGULATORY flips on bearish.
    """
    base = CATALYST_WEIGHTS.get(catalyst_type, 20)
    if catalyst_type in _NEGATIVE_BY_DEFAULT:
        return min(base, -abs(base))   # always negative
    if catalyst_type in _INHERENTLY_NEUTRAL:
        return 0
    # REGULATORY and GENERAL are direction-sensitive
    if direction == "bearish":
        return -abs(base)
    return abs(base)


def _fallback_classification(ticker: str, headlines: list[str]) -> dict:
    """
    Rule-based fallback using features.news_classifier when Claude is unavailable.
    Takes the strongest-scoring headline.
    """
    from features.news_classifier import classify, score as nc_score, label as nc_label

    # Map news_classifier categories → catalyst_type
    _nc_map = {
        "fda_approval":          ("REGULATORY",        "bullish"),
        "fda_rejection":         ("REGULATORY",        "bearish"),
        "earnings_beat":         ("EARNINGS",          "bullish"),
        "earnings_miss":         ("EARNINGS",          "bearish"),
        "contract_win":          ("PARTNERSHIP",       "bullish"),
        "partnership":           ("PARTNERSHIP",       "bullish"),
        "upgrade":               ("ANALYST_UPGRADE",   "bullish"),
        "downgrade":             ("ANALYST_DOWNGRADE", "bearish"),
        "offering_dilution":     ("GENERAL",           "bearish"),
        "ceo_departure":         ("CEO_DEPARTURE",     "bearish"),
        "sec_investigation":     ("LAWSUIT_FRAUD",     "bearish"),
        "class_action_lawsuit":  ("LAWSUIT_FRAUD",     "bearish"),
        "fraud_allegations":     ("LAWSUIT_FRAUD",     "bearish"),
        "bankruptcy_risk":       ("LAWSUIT_FRAUD",     "bearish"),
        "general":               ("GENERAL",           "neutral"),
    }

    best_cat, best_score, best_hl = "general", 0, headlines[0] if headlines else ""
    for hl in headlines:
        cat = classify(hl)
        s   = abs(nc_score(cat))
        if s > best_score:
            best_cat, best_score, best_hl = cat, s, hl

    catalyst_type, direction = _nc_map.get(best_cat, ("GENERAL", "neutral"))
    weight      = _apply_direction(catalyst_type, direction)
    is_tradeable = (
        abs(weight) >= MIN_TRADEABLE_WEIGHT
        and catalyst_type != "EARNINGS"
    )
    return {
        "catalyst_type":   catalyst_type,
        "catalyst_weight": weight,
        "summary":         f"{nc_label(best_cat)}: {best_hl[:80]}",
        "is_tradeable":    is_tradeable,
        "direction":       direction,
        "_source":         "fallback",
    }


# ── Claude classification ──────────────────────────────────────────────────────

def _ask_claude(ticker: str, headlines: list[str]) -> dict | None:
    """
    Call Claude Haiku with the catalyst classification prompt.
    Returns parsed dict or None on failure.
    """
    headlines_text = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines[:10]))

    prompt = f"""Classify the most significant news catalyst for {ticker} from these headlines.

HEADLINES:
{headlines_text}

Choose the SINGLE best catalyst type from:
REGULATORY, PARTNERSHIP, ANALYST_UPGRADE, ANALYST_DOWNGRADE, LAWSUIT_FRAUD, CEO_DEPARTURE, EARNINGS, GENERAL

Rules:
- REGULATORY covers FDA decisions, SEC policy, government rulings, clinical trial results
- LAWSUIT_FRAUD covers class actions, fraud allegations, SEC/DOJ investigations, bankruptcy
- CEO_DEPARTURE covers CEO/CFO/COO resignations, firings, sudden leadership changes
- EARNINGS covers any scheduled or surprise earnings release — always mark SKIP
- If multiple headlines, pick the highest-impact one
- is_tradeable must be False for rumors, speculation, or EARNINGS
- direction must reflect actual market impact, not just headline framing

Respond ONLY with valid JSON, no markdown:
{{
  "catalyst_type": "<TYPE>",
  "catalyst_weight": <integer from -90 to 90>,
  "summary": "<one sentence: what happened and why it matters>",
  "is_tradeable": <true|false>,
  "direction": "<bullish|bearish|neutral>"
}}"""

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if "```" in text:
            parts = text.split("```")
            text  = parts[1].lstrip("json").strip() if len(parts) >= 3 else text
        result = json.loads(text)

        # Validate and normalise
        cat = result.get("catalyst_type", "GENERAL").upper()
        if cat not in _VALID_TYPES:
            cat = "GENERAL"

        direction = result.get("direction", "neutral").lower()
        if direction not in ("bullish", "bearish", "neutral"):
            direction = "neutral"

        # Use Claude's weight if in valid range, otherwise derive from type+direction
        raw_weight = result.get("catalyst_weight", None)
        if isinstance(raw_weight, int) and -90 <= raw_weight <= 90:
            weight = raw_weight
        else:
            weight = _apply_direction(cat, direction)

        # Override is_tradeable with our own logic (Claude may hallucinate)
        is_tradeable = (abs(weight) >= MIN_TRADEABLE_WEIGHT and cat != "EARNINGS")
        # Still allow Claude to set False (stricter) — never let it set True for EARNINGS
        if cat == "EARNINGS":
            is_tradeable = False
        elif not result.get("is_tradeable", True):
            is_tradeable = False   # respect Claude's skepticism

        return {
            "catalyst_type":   cat,
            "catalyst_weight": weight,
            "summary":         str(result.get("summary", ""))[:200],
            "is_tradeable":    is_tradeable,
            "direction":       direction,
            "_source":         "claude",
        }

    except json.JSONDecodeError as e:
        print(f"⚠️  [CatalystAgent] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"⚠️  [CatalystAgent] Claude error: {e}")
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

@traceable(name="catalyst_agent", tags=["pipeline", "catalyst", "llm"])
def classify_catalyst(ticker: str, news_items: list[dict]) -> dict:
    """
    Classify the most significant news catalyst from news_items.

    news_items: list of Polygon or yfinance news dicts
      (Polygon: {"title": ...}, yfinance: {"content": {"title": ...}})

    Returns:
      catalyst_type    str
      catalyst_weight  int   -90 to +90
      summary          str
      is_tradeable     bool
      direction        str   "bullish" | "bearish" | "neutral"
      _source          str   "claude" | "fallback" | "cache"
    """
    _empty = {
        "catalyst_type":   "GENERAL",
        "catalyst_weight": 20,
        "summary":         "No news available.",
        "is_tradeable":    False,
        "direction":       "neutral",
        "_source":         "empty",
    }

    if not news_items:
        return _empty

    headlines = [_extract_title(n) for n in news_items]
    headlines = [h for h in headlines if h]
    if not headlines:
        return _empty

    key = _cache_key(ticker, headlines)

    # ── Cache lookup ─────────────────────────────────────────────────────────
    with _cache_lock:
        cached = _cache.get(key)
        if cached:
            age_s = (datetime.now() - cached["cached_at"]).total_seconds()
            if age_s < CACHE_TTL_S:
                result = dict(cached["result"])
                result["_source"] = "cache"
                return result

    # ── Claude classification ─────────────────────────────────────────────────
    print(f"🔬 [CatalystAgent] Classifying {len(headlines)} headline(s) for {ticker}...")
    result = _ask_claude(ticker, headlines)

    # ── Fallback ──────────────────────────────────────────────────────────────
    if result is None:
        result = _fallback_classification(ticker, headlines)
        print(f"🔬 [CatalystAgent] Fallback → {result['catalyst_type']} "
              f"w={result['catalyst_weight']} tradeable={result['is_tradeable']}")
    else:
        tradeable_str = "✅ TRADEABLE" if result["is_tradeable"] else "⛔ skip"
        print(f"🔬 [CatalystAgent] {result['catalyst_type']} "
              f"w={result['catalyst_weight']:+d} {result['direction']} "
              f"— {tradeable_str}")

    # ── Cache store ───────────────────────────────────────────────────────────
    with _cache_lock:
        _cache[key] = {"result": result, "cached_at": datetime.now()}

    # Trim cache to prevent unbounded growth
    with _cache_lock:
        if len(_cache) > 500:
            oldest = sorted(_cache, key=lambda k: _cache[k]["cached_at"])[:100]
            for k in oldest:
                del _cache[k]

    return result


def invalidate_cache(ticker: str | None = None) -> None:
    """
    Clear cached results. Pass ticker to clear only that ticker's entries,
    or None to wipe the entire cache.
    """
    with _cache_lock:
        if ticker is None:
            _cache.clear()
        else:
            to_del = [k for k in _cache if ticker.upper() in k]
            for k in to_del:
                del _cache[k]


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    from polygon_feed import get_news
    items = get_news(t, limit=5)

    print(f"\n=== CatalystAgent standalone test: {t} ===")
    print(f"Headlines:")
    for i, n in enumerate(items):
        print(f"  {i+1}. {_extract_title(n)}")

    result = classify_catalyst(t, items)
    print(f"\nResult:")
    for k, v in result.items():
        print(f"  {k}: {v}")
