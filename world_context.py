"""
world_context.py — Shared global market intelligence context.

All background intelligence agents (geo, macro, earnings, breadth, social)
write their findings here. The analyzer.py reads this and injects it into
every Claude prompt, giving Claude full awareness of the world environment
when deciding on individual stock signals.

Thread-safe: all reads/writes go through a lock.
"""

import threading
from datetime import datetime

_lock = threading.Lock()

_ctx: dict = {
    "geo": {
        "events":       [],       # [{headline, sectors, direction, magnitude, impact}]
        "overall_bias": "NEUTRAL",
        "hot_sectors":  [],       # sectors to favor
        "cold_sectors": [],       # sectors to avoid
        "risk_summary": "",
        "updated_at":   None,
    },
    "macro": {
        "regime":       "UNKNOWN",   # BULL / BEAR / STAGFLATION / RECOVERY / NEUTRAL
        "fed_stance":   "UNKNOWN",   # HAWKISH / DOVISH / NEUTRAL / PAUSED
        "yield_10y":    0.0,
        "yield_2y":     0.0,
        "yield_curve":  0.0,         # 10y - 2y spread in bps
        "vix":          0.0,
        "dxy":          0.0,         # dollar index
        "oil":          0.0,         # crude oil price
        "gold":         0.0,
        "bias":         "NEUTRAL",
        "summary":      "",
        "updated_at":   None,
    },
    "earnings": {
        "upcoming":     [],       # [{ticker, days, direction, avg_move_pct, beat_rate}]
        "hot_plays":    [],       # pre-earnings setups worth watching
        "updated_at":   None,
    },
    "breadth": {
        "health":            "UNKNOWN",  # STRONG / HEALTHY / NEUTRAL / WEAKENING / BEARISH
        "vix":               0.0,
        "vix_term_structure": "UNKNOWN", # CONTANGO (calm) / BACKWARDATION (fear)
        "pct_above_200ma":   0.0,        # % of SPY holdings above 200-day MA (proxy)
        "ad_ratio":          0.0,        # advance/decline ratio
        "summary":           "",
        "updated_at":        None,
    },
    "social": {
        "trending":      [],      # [{ticker, mentions, change_pct, source}]
        "congress_buys": [],      # [{ticker, politician, amount, date}]
        "unusual_opts":  [],      # [{ticker, call_put_ratio, description}]
        "updated_at":    None,
    },
}


# ── Writers (one per agent) ────────────────────────────────────────────────────

def update_geo(data: dict):
    with _lock:
        _ctx["geo"].update(data)
        _ctx["geo"]["updated_at"] = datetime.now().isoformat()


def update_macro(data: dict):
    with _lock:
        _ctx["macro"].update(data)
        _ctx["macro"]["updated_at"] = datetime.now().isoformat()


def update_earnings(data: dict):
    with _lock:
        _ctx["earnings"].update(data)
        _ctx["earnings"]["updated_at"] = datetime.now().isoformat()


def update_breadth(data: dict):
    with _lock:
        _ctx["breadth"].update(data)
        _ctx["breadth"]["updated_at"] = datetime.now().isoformat()


def update_social(data: dict):
    with _lock:
        _ctx["social"].update(data)
        _ctx["social"]["updated_at"] = datetime.now().isoformat()


# ── Reader ─────────────────────────────────────────────────────────────────────

def get() -> dict:
    with _lock:
        import copy
        return copy.deepcopy(_ctx)


# ── Prompt injection ───────────────────────────────────────────────────────────

def build_prompt_section() -> str:
    """
    Returns a formatted string ready to inject into Claude's analyzer prompt.
    Only includes sections that have been populated (non-None updated_at).
    """
    ctx   = get()
    lines = []

    # ── Geopolitical ──────────────────────────────────────────────────────────
    geo = ctx["geo"]
    if geo["updated_at"] and (geo["events"] or geo["risk_summary"]):
        lines.append("=== GEOPOLITICAL & MACRO NEWS INTELLIGENCE ===")
        lines.append(f"Global Bias: {geo['overall_bias']}")
        if geo["risk_summary"]:
            lines.append(f"Risk Theme: {geo['risk_summary']}")
        if geo["hot_sectors"]:
            lines.append(f"Favored Sectors: {', '.join(geo['hot_sectors'])}")
        if geo["cold_sectors"]:
            lines.append(f"Avoid Sectors:   {', '.join(geo['cold_sectors'])}")
        for e in geo["events"][:4]:
            direction_icon = "🟢" if e.get("direction") == "BULLISH" else "🔴" if e.get("direction") == "BEARISH" else "🟡"
            lines.append(
                f"  {direction_icon} {e.get('headline','')[:80]}\n"
                f"     → {e.get('impact','')}  |  Sectors: {', '.join(e.get('sectors',[]))}"
            )
        age = _age_str(geo["updated_at"])
        lines.append(f"  (geo data {age})")

    # ── Macro ──────────────────────────────────────────────────────────────────
    macro = ctx["macro"]
    if macro["updated_at"] and macro["regime"] != "UNKNOWN":
        lines.append("=== MACRO ENVIRONMENT ===")
        lines.append(
            f"Regime: {macro['regime']}  |  Fed: {macro['fed_stance']}  |  Bias: {macro['bias']}"
        )
        if macro["yield_10y"]:
            curve_label = "INVERTED ⚠️" if macro["yield_curve"] < 0 else "NORMAL"
            lines.append(
                f"Yields: 10y={macro['yield_10y']:.2f}%  2y={macro['yield_2y']:.2f}%  "
                f"Curve={macro['yield_curve']:+.0f}bp ({curve_label})"
            )
        parts = []
        if macro["vix"]:   parts.append(f"VIX={macro['vix']:.1f}")
        if macro["dxy"]:   parts.append(f"DXY={macro['dxy']:.1f}")
        if macro["oil"]:   parts.append(f"Oil=${macro['oil']:.1f}")
        if macro["gold"]:  parts.append(f"Gold=${macro['gold']:.0f}")
        if parts:
            lines.append("Markets: " + "  ".join(parts))
        if macro["summary"]:
            lines.append(f"Assessment: {macro['summary']}")

    # ── Market Breadth ─────────────────────────────────────────────────────────
    breadth = ctx["breadth"]
    if breadth["updated_at"] and breadth["health"] != "UNKNOWN":
        rotation = breadth.get("rotation", "")
        lines.append("=== MARKET BREADTH ===")
        health_line = f"Health: {breadth['health']}"
        if rotation:
            health_line += f"  |  Rotation: {rotation}"
        health_line += f"  |  VIX: {breadth['vix']:.1f}  ({breadth['vix_term_structure']})"
        lines.append(health_line)
        if breadth["pct_above_200ma"]:
            lines.append(f"% above 20-MA: {breadth['pct_above_200ma']:.0f}%  |  A/D ratio: {breadth['ad_ratio']:.1f}")
        if breadth.get("leading_sectors"):
            lines.append(f"Leading sectors: {', '.join(breadth['leading_sectors'])}")
        if breadth.get("lagging_sectors"):
            lines.append(f"Lagging sectors: {', '.join(breadth['lagging_sectors'])}")
        if breadth["summary"]:
            lines.append(f"Note: {breadth['summary']}")

    # ── Upcoming Earnings ──────────────────────────────────────────────────────
    earnings = ctx["earnings"]
    if earnings["updated_at"] and earnings["upcoming"]:
        tone     = earnings.get("earnings_tone", "")
        tone_pfx = f"  Earnings Tone: {tone}" if tone else ""
        lines.append("=== UPCOMING EARNINGS (MARKET MOVERS) ===")
        if tone_pfx:
            lines.append(tone_pfx)
        for e in earnings["upcoming"][:5]:
            icon = "🟢" if e.get("direction") == "BULLISH" else "🔴" if e.get("direction") == "BEARISH" else "🟡"
            lines.append(
                f"  {icon} {e['ticker']} in {e['days']}d  "
                f"avg move ±{e.get('avg_move_pct', 0):.1f}%  "
                f"beat rate {e.get('beat_rate', 0):.0f}%"
            )
        if earnings.get("hot_plays"):
            hp = earnings["hot_plays"][0]
            lines.append(f"  Key play: {hp['ticker']} — {hp.get('thesis', '')[:80]}")
        if earnings.get("summary"):
            lines.append(f"  {earnings['summary'][:150]}")

    # ── Social & Alternative Data ──────────────────────────────────────────────
    social = ctx["social"]
    if social["updated_at"]:
        alt_lines = []
        if social["trending"]:
            top = social["trending"][:5]
            alt_lines.append(
                "Trending: " +
                ", ".join(t["ticker"] for t in top)
            )
        if social["congress_buys"]:
            cb = social["congress_buys"][0]
            alt_lines.append(
                f"Congress buy: {cb['ticker']} ${cb.get('amount','?')} — {cb.get('politician','?')}"
            )
        if social["unusual_opts"]:
            for uo in social["unusual_opts"][:2]:
                bias_icon = "🟢" if uo.get("bias") == "BULLISH" else "🔴"
                alt_lines.append(f"Unusual options: {bias_icon} {uo['ticker']} — {uo['description'][:70]}")
        if social.get("summary"):
            alt_lines.append(f"Alt-data note: {social['summary'][:120]}")
        if alt_lines:
            lines.append("=== ALTERNATIVE DATA ===")
            lines.extend(alt_lines)

    return "\n".join(lines) if lines else ""


def _age_str(iso_str: str | None) -> str:
    if not iso_str:
        return "unknown age"
    try:
        dt      = datetime.fromisoformat(iso_str)
        minutes = int((datetime.now() - dt).total_seconds() / 60)
        if minutes < 60:
            return f"{minutes}min ago"
        return f"{minutes // 60}h ago"
    except Exception:
        return "recently"
