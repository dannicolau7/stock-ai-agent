"""
reflection_agent.py — Daily self-improvement agent.

Runs once per day at 4:15 PM ET (15 minutes after market close).
Reviews all completed signals from the past 30 days, asks Claude Sonnet
to find patterns in what worked and what didn't, then:

  1. Updates a `learnings.json` file with actionable rules
  2. Adjusts confidence thresholds in world_context (fed into every future prompt)
  3. Sends a daily WhatsApp summary to the user with win rate + top insight

This is the "trader journaling" step — the agent reflects on its own
decisions and improves over time.
"""

import asyncio
import json
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic

from config import ANTHROPIC_API_KEY
import performance_tracker as pt
import world_context as wctx
from alerts import send_whatsapp

LEARNINGS_PATH = Path(__file__).parent / "data" / "learnings.json"
ET = ZoneInfo("America/New_York")

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ── Load / save learnings ─────────────────────────────────────────────────────

def load_learnings() -> dict:
    LEARNINGS_PATH.parent.mkdir(exist_ok=True)
    if LEARNINGS_PATH.exists():
        try:
            return json.loads(LEARNINGS_PATH.read_text())
        except Exception:
            pass
    return {
        "updated_at":          None,
        "confidence_adj":      0,    # +/- adjustment to base threshold (65)
        "avoid_conditions":    [],   # conditions to avoid signaling
        "favor_conditions":    [],   # conditions that correlate with wins
        "insights":            [],   # free-text insights (last 5)
        "all_time_win_rate":   None,
        "last_30d_win_rate":   None,
    }


def save_learnings(data: dict):
    LEARNINGS_PATH.parent.mkdir(exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    LEARNINGS_PATH.write_text(json.dumps(data, indent=2))


# ── Core reflection ───────────────────────────────────────────────────────────

def _run_reflection(paper: bool = False) -> dict:
    """Ask Claude Sonnet to analyze signal history and extract learnings."""
    print("🧠 [Reflection] Running daily signal review...")

    stats_30 = pt.get_stats(lookback_days=30, paper=paper)
    stats_7  = pt.get_stats(lookback_days=7,  paper=paper)

    if stats_30.get("total", 0) < 3:
        print("🧠 [Reflection] Not enough signals yet (need ≥3). Skipping.")
        return {}

    learnings    = load_learnings()
    prior_insight = learnings.get("insights", [])[-1] if learnings.get("insights") else "None yet"

    # Build detailed signal table for Claude
    with pt._get_conn() as conn:
        rows = conn.execute("""
            SELECT s.ticker, s.signal, s.price, s.confidence, s.macro_regime,
                   s.macro_bias, s.geo_bias, s.market_health, s.rsi, s.volume_spike,
                   s.sentiment, s.trade_horizon, s.news_triggered, s.fired_at,
                   o.checkpoint, o.return_pct, o.win
            FROM signals s
            JOIN outcomes o ON o.signal_id = s.id
            WHERE s.fired_at >= date('now', '-30 days')
              AND s.paper = ?
              AND o.win IS NOT NULL
              AND o.checkpoint = '3d'
            ORDER BY s.fired_at DESC
        """, (int(paper),)).fetchall()

    if not rows:
        print("🧠 [Reflection] No completed 3d outcomes yet.")
        return {}

    signal_lines = []
    for r in rows:
        outcome = "WIN ✅" if r["win"] else "LOSS ❌"
        signal_lines.append(
            f"{r['ticker']} {r['signal']} conf={r['confidence']} "
            f"macro={r['macro_regime']} health={r['market_health']} "
            f"rsi={r['rsi']:.0f} volspike={r['volume_spike']} "
            f"sentiment={r['sentiment']} horizon={r['trade_horizon']} "
            f"news_trig={r['news_triggered']} → {r['return_pct']:+.1f}% {outcome}"
        )

    prompt = f"""You are a quantitative trading strategist reviewing the past 30 days of stock signals.

PERFORMANCE SUMMARY:
- Last 30 days: {stats_30['win_rate']}% win rate  ({stats_30['wins']}/{stats_30['total']} signals)
- Last 7 days:  {stats_7.get('win_rate', 'N/A')}% win rate
- Avg 3d return: {stats_30.get('by_checkpoint', {}).get('3d', {}).get('avg_ret', 0):+.2f}%
- By macro regime: {json.dumps(stats_30.get('by_regime', {}), indent=2)}

LAST PRIOR INSIGHT:
{prior_insight}

SIGNAL LOG (last 30 days, 3d outcomes):
{chr(10).join(signal_lines)}

Instructions:
1. Identify the 2-3 most important patterns that separate winners from losers
2. List conditions to AVOID signaling (e.g. "don't signal in BEAR macro + low confidence")
3. List conditions that FAVOR winners (e.g. "volume spike + BULL macro → highest win rate")
4. Recommend a confidence threshold adjustment (+5 = be more selective, -5 = be more aggressive)
5. Write one sharp actionable insight for tomorrow

Respond ONLY with valid JSON:
{{
  "patterns": ["pattern 1", "pattern 2"],
  "avoid_conditions": ["BEAR macro + news_triggered=False", "RSI > 75 + no volume spike"],
  "favor_conditions": ["BULL macro + volume spike + sentiment BULLISH", "health=STRONG + conf>70"],
  "confidence_adj": 0,
  "insight": "one sharp sentence for tomorrow's trading",
  "summary": "2-3 sentence summary of what the agent learned this week"
}}"""

    try:
        response = _get_client().messages.create(
            model="claude-sonnet-4-6",    # use Sonnet for reflection (more nuanced)
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if "```" in text:
            parts = text.split("```")
            text  = parts[1].lstrip("json").strip() if len(parts) >= 3 else text
        result = json.loads(text)

        # Update and save learnings
        learnings["last_30d_win_rate"]  = stats_30["win_rate"]
        learnings["confidence_adj"]     = int(result.get("confidence_adj", 0))
        learnings["avoid_conditions"]   = result.get("avoid_conditions", [])
        learnings["favor_conditions"]   = result.get("favor_conditions", [])
        insights = learnings.get("insights", [])
        if result.get("insight"):
            insights.append(result["insight"])
            learnings["insights"] = insights[-5:]   # keep last 5
        save_learnings(learnings)

        # Inject adjusted threshold into world_context for decision_agent to read
        wctx.update_macro({"confidence_adj": int(result.get("confidence_adj", 0))})

        print(f"🧠 [Reflection] Win rate: {stats_30['win_rate']}%  Threshold adj: {result.get('confidence_adj', 0):+d}")
        for p in result.get("patterns", []):
            print(f"   📌 {p}")
        if result.get("insight"):
            print(f"   💡 {result['insight']}")

        return result

    except Exception as e:
        print(f"⚠️  [Reflection] Claude error: {e}")
        return {}


def _send_daily_summary(result: dict, stats: dict, paper: bool = False):
    """Send daily performance WhatsApp to user."""
    if paper:
        return

    win_rate  = stats.get("win_rate", "?")
    total     = stats.get("total", 0)
    avg_ret   = stats.get("by_checkpoint", {}).get("3d", {}).get("avg_ret", 0)
    insight   = result.get("insight", "")
    summary   = result.get("summary", "")

    best      = stats.get("best_trades", [])
    worst     = stats.get("worst_trades", [])
    best_str  = f"{best[0]['ticker']} {best[0]['ret']:+.1f}%" if best else "—"
    worst_str = f"{worst[0]['ticker']} {worst[0]['ret']:+.1f}%" if worst else "—"

    body = (
        f"📊 *Daily Signal Review*\n"
        f"Win rate (30d): *{win_rate}%*  ({total} signals)\n"
        f"Avg 3d return: *{avg_ret:+.2f}%*\n"
        f"Best: {best_str}  |  Worst: {worst_str}\n\n"
        f"💡 *Insight:* {insight}\n\n"
        f"_{summary}_"
    )

    try:
        send_whatsapp(body)
        print("🧠 [Reflection] Daily summary sent via WhatsApp")
    except Exception as e:
        print(f"⚠️  [Reflection] WhatsApp failed: {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────

async def reflection_agent_loop(paper: bool = False):
    """
    Runs once per day at 4:15 PM ET (after market close).
    On first startup, runs immediately if after 4 PM.
    """
    print("🧠 [Reflection] Started — will run daily at 4:15 PM ET")

    while True:
        try:
            now_et = datetime.now(ET)
            target = now_et.replace(hour=16, minute=15, second=0, microsecond=0)
            if now_et >= target:
                target = target.replace(day=target.day + 1)

            wait_s = (target - now_et).total_seconds()
            print(f"🧠 [Reflection] Next review in {wait_s/3600:.1f}h (at 4:15 PM ET)")
            await asyncio.sleep(wait_s)

            loop   = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, _run_reflection, paper)
            if result:
                stats = pt.get_stats(lookback_days=30, paper=paper)
                _send_daily_summary(result, stats, paper=paper)

        except asyncio.CancelledError:
            print("🧠 [Reflection] Stopped.")
            break
        except Exception as e:
            print(f"❌ [Reflection] Loop error: {e}")
            await asyncio.sleep(3600)


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Reflection Agent Standalone Test ===\n")
    print("Learnings file:", LEARNINGS_PATH)

    learnings = load_learnings()
    print(f"Prior win rate: {learnings.get('last_30d_win_rate', 'none')}")
    print(f"Conf adj: {learnings.get('confidence_adj', 0):+d}")
    print(f"Avoid: {learnings.get('avoid_conditions', [])}")
    print(f"Favor: {learnings.get('favor_conditions', [])}")
    print(f"Insights: {learnings.get('insights', [])}")

    print("\nRunning reflection (needs ≥3 completed signals to do anything)...")
    result = _run_reflection(paper=False)
    if result:
        print(f"\nResult: {json.dumps(result, indent=2)}")
    else:
        print("Skipped — not enough signal history yet.")
        print("(This is expected on a fresh install)")
