"""
agents/eval_agent.py — Weekly performance evaluation & learning agent.

Runs every Sunday at 7 PM EST. Reads the last 50 closed trades, sends
them to Claude for adversarial analysis, proposes weight adjustments, and
requests manual approval before applying any changes.

Public API
----------
  run_weekly_eval()              — main entry (called by scheduler)
  approve_learnings(code=None)   — apply pending weight adjustments
  reject_learnings(code=None)    — discard pending proposal
  get_pending_approval()         — return pending proposal dict (or None)

Approval flow
-------------
  1. run_weekly_eval() writes data/weekly_learnings.json  (status="pending")
  2. WhatsApp message sent with 6-char code + two URLs:
       /api/eval/approve/<code>   → approve_learnings(code)
       /api/eval/reject/<code>    → reject_learnings(code)
  3. On approval: weight_adjustments EMA-blended into data/learnings.json
                  IntelligenceHub weights refreshed
  4. Next eval fills in after_win_rate for the prior entry in history[]

Improvement tracking
--------------------
  weekly_learnings.json  →  history: list of weekly snapshots:
    { week_ending, before_win_rate, after_win_rate,
      applied, narrative_summary, weight_adjustments }
  after_win_rate is filled in by the following week's eval run.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import anthropic

from config import ANTHROPIC_API_KEY
from alerts import send_whatsapp
import performance_tracker as pt

ET = ZoneInfo("America/New_York")

_WEEKLY_PATH  = Path(__file__).parent.parent / "data" / "weekly_learnings.json"
_LEARNINGS_PATH = Path(__file__).parent.parent / "data" / "learnings.json"

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ── Persistence helpers ────────────────────────────────────────────────────────

def _load_weekly() -> dict:
    if _WEEKLY_PATH.exists():
        try:
            return json.loads(_WEEKLY_PATH.read_text())
        except Exception:
            pass
    return {"history": [], "pending": None}


def _save_weekly(data: dict) -> None:
    _WEEKLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _WEEKLY_PATH.write_text(json.dumps(data, indent=2))


def _load_learnings() -> dict:
    if _LEARNINGS_PATH.exists():
        try:
            return json.loads(_LEARNINGS_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_learnings(data: dict) -> None:
    _LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    _LEARNINGS_PATH.write_text(json.dumps(data, indent=2))


# ── Trade data ─────────────────────────────────────────────────────────────────

def _get_closed_trades(n: int = 50) -> list[dict]:
    """
    Return up to n most recent fully-closed trades (7d outcome filled).
    Includes: ticker, signal, confidence, price, stop_loss, rsi, volume_spike,
    sentiment, sector, trade_horizon, macro_regime, news_triggered, reasoning,
    fired_at, return_pct, win.
    """
    try:
        with pt._get_conn() as conn:
            rows = conn.execute("""
                SELECT
                    s.ticker, s.signal, s.confidence, s.price,
                    s.stop_loss, s.rsi, s.volume_spike, s.sentiment,
                    s.sector, s.trade_horizon, s.macro_regime,
                    s.news_triggered, s.reasoning, s.fired_at,
                    o.return_pct, o.win
                FROM signals s
                JOIN outcomes o ON o.signal_id = s.id
                WHERE s.paper = 0
                  AND o.checkpoint = '7d'
                  AND o.win IS NOT NULL
                ORDER BY s.fired_at DESC
                LIMIT ?
            """, (n,)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"⚠️  [EvalAgent] Trade query failed: {e}")
        return []


# ── Claude analysis ────────────────────────────────────────────────────────────

_VALID_SIGNALS = {
    "macd", "rsi_bounce", "rsi_momentum", "rsi_oversold",
    "volume_spike", "ema_stack", "edgar_filing", "news_sentiment",
    "reddit", "vwap", "bollinger", "float_rot", "sentiment",
    "smart_money", "support", "gap", "bullish_news",
}

def _build_trade_table(trades: list[dict]) -> str:
    """Format trades as a compact table for the Claude prompt."""
    lines = [
        "ticker | sig | conf | rsi | vol_spike | macro | sector | horizon | "
        "news | return% | win"
    ]
    for t in trades:
        rsi   = f"{t.get('rsi') or 0:.0f}"
        ret   = f"{t.get('return_pct') or 0:+.1f}%"
        win   = "WIN" if t.get("win") else "LOSS"
        lines.append(
            f"{t['ticker']} | {t['signal']} | {t['confidence']} | "
            f"{rsi} | {'Y' if t.get('volume_spike') else 'N'} | "
            f"{t.get('macro_regime','?')} | {t.get('sector','?')} | "
            f"{t.get('trade_horizon','?')} | "
            f"{'Y' if t.get('news_triggered') else 'N'} | "
            f"{ret} | {win}"
        )
    return "\n".join(lines)


def _compute_win_rates_by_dimension(trades: list[dict]) -> dict:
    """Compute win rates per RSI zone, confidence bucket, macro, sector, etc."""
    def _wr(bucket):
        t = bucket["total"]
        return round(bucket["wins"] / t * 100, 1) if t else None

    dims: dict[str, dict[str, dict]] = {
        "rsi_zone":    defaultdict(lambda: {"wins": 0, "total": 0}),
        "conf_bucket": defaultdict(lambda: {"wins": 0, "total": 0}),
        "macro":       defaultdict(lambda: {"wins": 0, "total": 0}),
        "sector":      defaultdict(lambda: {"wins": 0, "total": 0}),
        "horizon":     defaultdict(lambda: {"wins": 0, "total": 0}),
        "volume":      defaultdict(lambda: {"wins": 0, "total": 0}),
        "news":        defaultdict(lambda: {"wins": 0, "total": 0}),
    }

    for t in trades:
        w   = int(bool(t.get("win")))
        rsi = float(t.get("rsi") or 50)
        conf = int(t.get("confidence") or 0)

        rsi_z  = "oversold(<30)" if rsi < 30 else "bounce(30-50)" if rsi < 50 else "momentum(50-65)" if rsi < 65 else "overbought(65+)"
        conf_b = "<50" if conf < 50 else "50-65" if conf < 65 else "65-80" if conf < 80 else "80+"
        macro  = (t.get("macro_regime") or "UNKNOWN").upper()
        sector = t.get("sector") or "UNKNOWN"
        horiz  = t.get("trade_horizon") or "swing"
        vol_k  = "spike" if t.get("volume_spike") else "normal"
        news_k = "news_triggered" if t.get("news_triggered") else "no_news"

        for key, bucket_key in [
            ("rsi_zone", rsi_z), ("conf_bucket", conf_b), ("macro", macro),
            ("sector", sector), ("horizon", horiz), ("volume", vol_k),
            ("news", news_k),
        ]:
            dims[key][bucket_key]["total"] += 1
            dims[key][bucket_key]["wins"]  += w

    return {
        dim: {k: {"win_rate": _wr(v), "total": v["total"]} for k, v in buckets.items()}
        for dim, buckets in dims.items()
    }


def _analyze_with_claude(trades: list[dict], stats_30: dict) -> dict:
    """
    Send trades + stats to Claude; returns structured analysis dict.
    Returns {} on any failure.
    """
    if not trades:
        print("⚠️  [EvalAgent] No trades to analyze.")
        return {}

    trade_table  = _build_trade_table(trades)
    dim_analysis = _compute_win_rates_by_dimension(trades)

    # Identify high/low performers for Claude's context
    wins   = [t for t in trades if t.get("win")]
    losses = [t for t in trades if not t.get("win")]
    win_rate_pct = round(len(wins) / len(trades) * 100, 1) if trades else 0

    valid_signal_list = ", ".join(sorted(_VALID_SIGNALS))

    prompt = f"""You are a quantitative trading strategist performing a weekly post-mortem.

=== PERFORMANCE SUMMARY ===
Total closed trades: {len(trades)}
Win rate: {win_rate_pct}%
Wins: {len(wins)}  |  Losses: {len(losses)}
30-day stats: {json.dumps(stats_30, indent=2)}

=== WIN RATE BY DIMENSION ===
{json.dumps(dim_analysis, indent=2)}

=== FULL TRADE LOG ({len(trades)} trades, newest first) ===
{trade_table}

=== REASONING EXCERPTS (last 10 losses) ===
{chr(10).join(f"- {t['ticker']} {t['signal']}: {str(t.get('reasoning',''))[:120]}" for t in losses[:10])}

=== YOUR MISSION ===
Analyze these {len(trades)} closed trades and identify:
1. Which signal types have high vs low win rates
2. Which market conditions (macro regime, sector, RSI zone) predict wins/losses
3. What patterns to start watching that are absent but implied by the data
4. What currently-tracked patterns consistently fail and should be removed
5. Suggested weight multipliers for each signal (0.5–1.5; 1.0 = neutral)

Valid signals for weight_adjustments: {valid_signal_list}

RULES:
- weight_adjustments: only signals with ≥3 samples; range 0.5–1.5
- patterns_to_add: specific, actionable pattern names (e.g. "opening_range_breakout_with_volume")
- patterns_to_remove: only patterns you see consistently losing across multiple tickers
- Be data-driven — cite specific win rates, not generalities
- narrative_summary: exactly 3 sentences: what worked, what failed, key adjustment

Respond ONLY with valid JSON (no markdown fences):
{{
  "weight_adjustments": {{
    "macd": 1.2,
    "rsi_bounce": 0.8,
    "...": "..."
  }},
  "patterns_to_add": [
    "pattern_name_snake_case"
  ],
  "patterns_to_remove": [
    "pattern_name_snake_case"
  ],
  "insights": [
    "actionable insight 1",
    "actionable insight 2",
    "actionable insight 3"
  ],
  "avoid_conditions": [
    "specific condition string"
  ],
  "favor_conditions": [
    "specific condition string"
  ],
  "confidence_adj": 0,
  "narrative_summary": "3-sentence summary..."
}}"""

    try:
        print("🧠 [EvalAgent] Calling Claude for weekly analysis...")
        response = _get_client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Strip optional ``` fences
        if "```" in text:
            parts = text.split("```")
            text  = parts[1].lstrip("json").strip() if len(parts) >= 3 else text
        # Find first { … }
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        result = json.loads(text)

        # Sanitise weight values — must be in [0.5, 1.5] and valid signal names
        raw_weights = result.get("weight_adjustments", {})
        clean_weights = {}
        for sig, val in raw_weights.items():
            if sig in _VALID_SIGNALS:
                try:
                    clean_weights[sig] = round(max(0.5, min(1.5, float(val))), 3)
                except (TypeError, ValueError):
                    pass
        result["weight_adjustments"] = clean_weights
        return result

    except json.JSONDecodeError as e:
        print(f"⚠️  [EvalAgent] JSON parse error: {e} — raw: {text[:200]}")
        return {}
    except Exception as e:
        print(f"❌ [EvalAgent] Claude call failed: {e}")
        return {}


# ── Approval ───────────────────────────────────────────────────────────────────

def get_pending_approval() -> dict | None:
    """Return the pending approval proposal, or None if nothing is pending."""
    data = _load_weekly()
    p    = data.get("pending")
    if not p or p.get("status") != "pending":
        return None
    return p


def approve_learnings(code: str | None = None) -> dict:
    """
    Apply the pending weight adjustments to learnings.json (EMA blend 80/20).
    Returns {"ok": True, "applied": {...}} or {"ok": False, "reason": "..."}.
    """
    data    = _load_weekly()
    pending = data.get("pending")

    if not pending or pending.get("status") != "pending":
        return {"ok": False, "reason": "No pending proposal found."}

    if code is not None and pending.get("approval_code") != str(code).upper():
        return {"ok": False, "reason": f"Invalid approval code '{code}'."}

    analysis = pending.get("analysis", {})
    new_weights = analysis.get("weight_adjustments", {})

    # Load current learnings and EMA-blend weights (80% old, 20% new)
    learnings = _load_learnings()
    old_weights = learnings.get("signal_weights", {})
    blended = {}
    all_signals = set(old_weights) | set(new_weights)
    for sig in all_signals:
        old = old_weights.get(sig, 1.0)
        new = new_weights.get(sig, old)
        blended[sig] = round(old * 0.8 + new * 0.2, 4)

    learnings["signal_weights"]    = blended
    learnings["last_eval_at"]      = datetime.now().isoformat()
    learnings["eval_win_rate"]     = pending.get("before_win_rate")
    learnings["avoid_conditions"]  = analysis.get("avoid_conditions", learnings.get("avoid_conditions", []))
    learnings["favor_conditions"]  = analysis.get("favor_conditions", learnings.get("favor_conditions", []))

    # Add insights (keep last 5)
    insights = learnings.get("insights", [])
    for ins in analysis.get("insights", []):
        if ins and ins not in insights:
            insights.append(ins)
    learnings["insights"] = insights[-5:]

    # Confidence adjustment
    conf_adj = int(analysis.get("confidence_adj", 0))
    if conf_adj:
        learnings["confidence_adj"] = conf_adj

    _save_learnings(learnings)

    # Refresh IntelligenceHub weights
    try:
        from intelligence_hub import hub
        hub.update_reflection_weights(blended)
        print(f"🔄 [EvalAgent] Hub weights refreshed ({len(blended)} signals)")
    except Exception as e:
        print(f"⚠️  [EvalAgent] Hub weight refresh failed: {e}")

    # Update pending state
    pending["status"]      = "approved"
    pending["decided_at"]  = datetime.now().isoformat()
    data["pending"]        = pending

    # Mark corresponding history entry as applied
    for entry in data.get("history", []):
        if entry.get("approval_code") == pending.get("approval_code"):
            entry["applied"]    = True
            entry["decided_at"] = pending["decided_at"]
            break

    _save_weekly(data)

    print(f"✅ [EvalAgent] Approved — {len(blended)} weights applied to learnings.json")
    _send_decision_alert("approved", pending, blended)
    return {"ok": True, "applied": blended}


def reject_learnings(code: str | None = None) -> dict:
    """
    Discard the pending proposal without applying changes.
    """
    data    = _load_weekly()
    pending = data.get("pending")

    if not pending or pending.get("status") != "pending":
        return {"ok": False, "reason": "No pending proposal found."}

    if code is not None and pending.get("approval_code") != str(code).upper():
        return {"ok": False, "reason": f"Invalid approval code '{code}'."}

    pending["status"]     = "rejected"
    pending["decided_at"] = datetime.now().isoformat()
    data["pending"]       = pending

    for entry in data.get("history", []):
        if entry.get("approval_code") == pending.get("approval_code"):
            entry["applied"]    = False
            entry["decided_at"] = pending["decided_at"]
            break

    _save_weekly(data)
    print("🚫 [EvalAgent] Proposal rejected — no weights changed.")
    _send_decision_alert("rejected", pending, {})
    return {"ok": True, "rejected": True}


# ── Improvement tracking ───────────────────────────────────────────────────────

def _fill_prior_after_win_rate(data: dict, current_win_rate: float) -> None:
    """
    Fill in after_win_rate for the most recent approved history entry
    whose after_win_rate is still None.
    """
    for entry in reversed(data.get("history", [])):
        if entry.get("applied") and entry.get("after_win_rate") is None:
            decided_at = entry.get("decided_at")
            if decided_at:
                applied_date = datetime.fromisoformat(decided_at).date()
                days_since   = (date.today() - applied_date).days
                # Only fill in if at least 14 days have passed (enough closed trades)
                if days_since >= 14:
                    entry["after_win_rate"] = current_win_rate
                    delta = round(current_win_rate - (entry.get("before_win_rate") or 0), 1)
                    entry["win_rate_delta"] = delta
                    print(
                        f"📈 [EvalAgent] Prior adjustment impact: "
                        f"{entry['before_win_rate']}% → {current_win_rate}% "
                        f"({delta:+.1f}%)"
                    )
            break


# ── WhatsApp messages ──────────────────────────────────────────────────────────

_SEP = "─" * 35


def _send_approval_request(
    analysis: dict,
    code: str,
    before_win_rate: float,
    trade_count: int,
) -> None:
    n_weights = len(analysis.get("weight_adjustments", {}))
    n_add     = len(analysis.get("patterns_to_add", []))
    n_remove  = len(analysis.get("patterns_to_remove", []))
    summary   = analysis.get("narrative_summary", "")[:300]

    # Format top weight changes
    weight_lines = []
    for sig, val in sorted(
        analysis.get("weight_adjustments", {}).items(),
        key=lambda x: abs(x[1] - 1.0), reverse=True
    )[:5]:
        arrow = "↑" if val >= 1.1 else ("↓" if val <= 0.9 else "→")
        weight_lines.append(f"  {arrow} {sig}: {val:.2f}")

    top_insights = analysis.get("insights", [])[:2]
    insight_lines = "\n".join(f"• {ins[:100]}" for ins in top_insights)

    msg = (
        f"🤖 Argus Weekly Eval — {date.today()}\n"
        f"{_SEP}\n"
        f"Trades reviewed: {trade_count}\n"
        f"Current win rate: {before_win_rate:.1f}%\n"
        f"Proposed changes: {n_weights} weight adj, "
        f"+{n_add} patterns, -{n_remove} patterns\n"
        f"{_SEP}\n"
        f"Summary:\n{summary}\n"
        f"{_SEP}\n"
        f"Top weight changes:\n"
        f"{chr(10).join(weight_lines) or '  (none)'}\n"
        f"{_SEP}\n"
        f"Key insights:\n{insight_lines or '(none)'}\n"
        f"{_SEP}\n"
        f"Approval code: {code}\n"
        f"To APPROVE: GET /api/eval/approve/{code}\n"
        f"To REJECT:  GET /api/eval/reject/{code}\n"
        f"\nOr run: python3 -c \"from agents.eval_agent import approve_learnings; approve_learnings('{code}')\""
    )
    send_whatsapp(msg)
    print(f"📱 [EvalAgent] Approval request sent. Code: {code}")


def _send_decision_alert(decision: str, pending: dict, blended: dict) -> None:
    icon   = "✅" if decision == "approved" else "🚫"
    action = "APPROVED — weights applied" if decision == "approved" else "REJECTED — no changes"
    before = pending.get("before_win_rate", 0)

    msg = (
        f"{icon} Argus Eval {action.upper()}\n"
        f"Code: {pending.get('approval_code','?')}\n"
        f"Win rate at time of eval: {before:.1f}%\n"
    )
    if blended:
        top = sorted(blended.items(), key=lambda x: abs(x[1] - 1.0), reverse=True)[:3]
        msg += "Applied:\n" + "\n".join(f"  {s}: {v:.2f}" for s, v in top)
    send_whatsapp(msg)


# ── Main entry point ───────────────────────────────────────────────────────────

def run_weekly_eval() -> dict:
    """
    Full weekly evaluation pipeline. Designed to be called every Sunday 7 PM EST.
    Returns the analysis dict (or {} on failure).
    """
    print("\n" + "═" * 55)
    print("📊 [EvalAgent] Starting weekly evaluation...")
    print("═" * 55)

    # Load weekly state + fill in prior after_win_rate if ready
    data     = _load_weekly()
    stats_30 = pt.get_stats(lookback_days=30)
    current_win_rate = float(stats_30.get("win_rate") or 0)
    _fill_prior_after_win_rate(data, current_win_rate)

    # Guard: don't run if already ran this week
    last_run = data.get("pending", {}).get("requested_at") if data.get("pending") else None
    if last_run:
        last_run_date = datetime.fromisoformat(last_run).date()
        days_ago      = (date.today() - last_run_date).days
        if days_ago < 6 and data.get("pending", {}).get("status") == "pending":
            print(f"⏭️  [EvalAgent] Already ran {days_ago}d ago and approval is pending. Skipping.")
            return {}

    # Fetch closed trades
    trades = _get_closed_trades(n=50)
    if len(trades) < 10:
        msg = (
            f"⚠️ Argus Weekly Eval skipped — only {len(trades)} closed trades "
            f"(need ≥10). Will retry next Sunday."
        )
        print(f"⚠️  [EvalAgent] {msg}")
        send_whatsapp(msg)
        return {}

    print(f"📋 [EvalAgent] Loaded {len(trades)} closed trades  "
          f"(win rate {current_win_rate:.1f}%)")

    # Claude analysis
    analysis = _analyze_with_claude(trades, stats_30)
    if not analysis:
        err_msg = "❌ Argus Weekly Eval failed — Claude analysis returned no data."
        send_whatsapp(err_msg)
        return {}

    # Log key findings
    print(f"   Weight adjustments: {len(analysis.get('weight_adjustments', {}))}")
    print(f"   Patterns to add:    {len(analysis.get('patterns_to_add', []))}")
    print(f"   Patterns to remove: {len(analysis.get('patterns_to_remove', []))}")
    print(f"   Summary: {analysis.get('narrative_summary','')[:100]}...")

    # Generate approval code
    code = secrets.token_hex(3).upper()   # e.g. "A3F7B2"

    # Build history entry (after_win_rate filled in by next run)
    history_entry = {
        "week_ending":        str(date.today()),
        "before_win_rate":    current_win_rate,
        "after_win_rate":     None,
        "win_rate_delta":     None,
        "trade_count":        len(trades),
        "applied":            False,
        "decided_at":         None,
        "approval_code":      code,
        "narrative_summary":  analysis.get("narrative_summary", ""),
        "weight_adjustments": analysis.get("weight_adjustments", {}),
    }
    data.setdefault("history", []).append(history_entry)

    # Build pending proposal
    data["pending"] = {
        "status":          "pending",
        "approval_code":   code,
        "requested_at":    datetime.now().isoformat(),
        "decided_at":      None,
        "before_win_rate": current_win_rate,
        "trade_count":     len(trades),
        "analysis":        analysis,
    }

    _save_weekly(data)
    print(f"💾 [EvalAgent] Saved to {_WEEKLY_PATH.name}  (code={code})")

    # Refresh LangSmith eval dataset from losing trades
    try:
        from utils.tracing import create_eval_dataset
        create_eval_dataset(n=50)
    except Exception as _ds_err:
        print(f"⚠️  [EvalAgent] LangSmith dataset refresh failed: {_ds_err}")

    # Send WhatsApp approval request
    _send_approval_request(analysis, code, current_win_rate, len(trades))

    print("═" * 55)
    return analysis
