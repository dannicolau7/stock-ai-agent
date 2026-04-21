"""
utils/tracing.py — Shared LangSmith tracing utilities for Argus.

All agent nodes import from here so there's one place to change behaviour.

Designed to be zero-cost when tracing is disabled (LANGCHAIN_API_KEY not set
or LANGCHAIN_TRACING_V2 != "true") — every public function is a no-op in
that case and never raises.

Public API
----------
  annotate_run(state)             — patch active run with ticker/signal/regime
  get_current_run_id()            — return active LangSmith run ID or None
  flag_outcome(run_id, ...)       — add score=0 (loss) or score=1 (win) feedback
  create_eval_dataset(n)          — build/refresh 'argus-losing-trades' dataset
  is_enabled()                    — True when LangSmith is active
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# ── Lazy enable check ─────────────────────────────────────────────────────────

def is_enabled() -> bool:
    return (
        bool(os.getenv("LANGCHAIN_API_KEY"))
        and os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )


# ── Metadata / tag builders ────────────────────────────────────────────────────

def build_tags(state: dict) -> list[str]:
    """Return dynamic tags for a pipeline run, derived from state."""
    tags: list[str] = []
    ticker = state.get("ticker", "")
    if ticker:
        tags.append(f"ticker:{ticker}")
    signal = state.get("signal", "HOLD")
    if signal and signal != "HOLD":
        tags.append(f"signal:{signal}")
    regime = (state.get("market_regime") or {}).get("regime", "")
    if regime:
        tags.append(f"regime:{regime}")
    if state.get("news_triggered"):
        tags.append("news_trigger")
    if state.get("has_edgar_filing"):
        tags.append("edgar_trigger")
    horizon = state.get("trade_horizon", "")
    if horizon:
        tags.append(f"horizon:{horizon}")
    return tags


def build_metadata(state: dict) -> dict:
    """Return structured metadata for a LangSmith run."""
    regime = state.get("market_regime") or {}
    score  = (state.get("score_breakdown") or {}).get("final_score", 0)
    return {
        "ticker":        state.get("ticker", ""),
        "signal_type":   state.get("signal", "HOLD"),
        "confidence":    state.get("confidence", 0),
        "timestamp":     state.get("timestamp", ""),
        "market_regime": regime.get("regime", ""),
        "sector":        state.get("sector", ""),
        "trade_horizon": state.get("trade_horizon", ""),
        "det_score":     score,
        "bull_score":    state.get("bull_score", 0),
        "bear_score":    state.get("bear_score", 0),
        "rsi":           state.get("rsi", 0),
        "volume_ratio":  state.get("volume_spike_ratio", 0),
        "agreement":     state.get("agreement_score", 0),
    }


# ── Active-run annotation ──────────────────────────────────────────────────────

def annotate_run(state: dict) -> None:
    """
    Patch the currently-active LangSmith run with dynamic metadata and tags.
    Safe to call from any @traceable context — silently no-ops if not enabled
    or if called outside a trace.
    """
    if not is_enabled():
        return
    try:
        from langsmith.run_helpers import get_current_run_tree
        run = get_current_run_tree()
        if run is None:
            return
        # RunTree.patch() sends a PATCH to the LangSmith API in the background
        run.patch(
            metadata=build_metadata(state),
            tags=build_tags(state),
        )
    except Exception:
        pass   # never crash the pipeline over observability


def get_current_run_id() -> str | None:
    """Return the active LangSmith run ID (UUID string), or None."""
    if not is_enabled():
        return None
    try:
        from langsmith.run_helpers import get_current_run_tree
        run = get_current_run_tree()
        return str(run.id) if run else None
    except Exception:
        return None


# ── Outcome feedback ──────────────────────────────────────────────────────────

def flag_outcome(
    run_id:      str | None,
    ticker:      str,
    return_pct:  float,
    checkpoint:  str,
    *,
    win:         bool,
) -> None:
    """
    Add a LangSmith feedback record to the run that generated this trade.

    score=1.0 → WIN (green in dashboard)
    score=0.0 → LOSS (red, automatically surfaces for review)
    """
    if not is_enabled() or not run_id:
        return
    try:
        from langsmith import Client
        client  = Client()
        score   = 1.0 if win else 0.0
        outcome = "WIN" if win else "LOSS"
        client.create_feedback(
            run_id=run_id,
            key="trade_outcome",
            score=score,
            comment=f"{outcome} {checkpoint}: {return_pct:+.1f}% on {ticker}",
        )
        icon = "✅" if win else "🔴"
        print(
            f"{icon} [LangSmith] Feedback → run {run_id[:8]}… "
            f"{outcome} {return_pct:+.1f}% ({ticker} {checkpoint})"
        )
    except Exception as e:
        print(f"⚠️  [LangSmith] feedback failed: {e}")


# ── Eval dataset ──────────────────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent.parent / "data" / "performance.db"
_DATASET_NAME = "argus-losing-trades"


def create_eval_dataset(n: int = 50) -> str | None:
    """
    Create / refresh the LangSmith dataset 'argus-losing-trades' from the
    last n losing closed trades (7d checkpoint).

    Each example:
      inputs  = trade state snapshot at decision time
      outputs = {"return_pct": -X.X, "win": False,
                 "ideal_signal": "HOLD",
                 "langsmith_run_id": "..."}

    Returns the dataset URL or None on failure.
    """
    if not is_enabled():
        print("⚠️  [LangSmith] Tracing disabled — skipping eval dataset creation.")
        return None

    # ── Query losing trades ───────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT
                s.ticker, s.signal, s.confidence, s.price, s.stop_loss,
                s.rsi, s.volume_spike, s.sentiment, s.sector,
                s.trade_horizon, s.macro_regime, s.macro_bias,
                s.news_triggered, s.reasoning, s.fired_at,
                s.langsmith_run_id,
                o.return_pct, o.win
            FROM signals s
            JOIN outcomes o ON o.signal_id = s.id
            WHERE s.paper = 0
              AND o.checkpoint = '7d'
              AND o.win = 0
            ORDER BY s.fired_at DESC
            LIMIT ?
        """, (n,)).fetchall()
        conn.close()
    except Exception as e:
        print(f"❌ [LangSmith] DB query for eval dataset failed: {e}")
        return None

    if not rows:
        print("⚠️  [LangSmith] No losing trades found for eval dataset.")
        return None

    # ── Create or find dataset ────────────────────────────────────────────────
    try:
        from langsmith import Client
        client = Client()

        dataset = None
        try:
            dataset = client.create_dataset(
                _DATASET_NAME,
                description=(
                    f"Last {n} losing trades — auto-generated by Argus eval_agent. "
                    "Use to test prompt improvements against real failure cases."
                ),
            )
            print(f"📋 [LangSmith] Created new dataset '{_DATASET_NAME}'")
        except Exception:
            # Already exists — find it
            matches = list(client.list_datasets(dataset_name=_DATASET_NAME))
            dataset = matches[0] if matches else None

        if dataset is None:
            print(f"❌ [LangSmith] Could not create or locate dataset '{_DATASET_NAME}'")
            return None

        # Clear stale examples before re-populating
        try:
            old = list(client.list_examples(dataset_id=dataset.id))
            for ex in old:
                client.delete_example(ex.id)
        except Exception:
            pass

        # Insert fresh examples
        for r in rows:
            inputs = {
                "ticker":         r["ticker"],
                "signal":         r["signal"],
                "confidence":     r["confidence"],
                "price":          r["price"],
                "stop_loss":      r["stop_loss"] or 0,
                "rsi":            r["rsi"] or 50,
                "volume_spike":   bool(r["volume_spike"]),
                "sentiment":      r["sentiment"] or "NEUTRAL",
                "sector":         r["sector"] or "",
                "trade_horizon":  r["trade_horizon"] or "swing",
                "macro_regime":   r["macro_regime"] or "",
                "macro_bias":     r["macro_bias"] or "",
                "news_triggered": bool(r["news_triggered"]),
                "reasoning":      (r["reasoning"] or "")[:500],
                "fired_at":       r["fired_at"],
                "langsmith_run_id": r["langsmith_run_id"] or "",
            }
            outputs = {
                "return_pct":   r["return_pct"],
                "win":          False,
                "outcome":      f"LOSS {r['return_pct']:+.1f}%",
                "ideal_signal": "HOLD",   # conservative: if it lost, HOLD was safer
            }
            client.create_example(
                inputs=inputs,
                outputs=outputs,
                dataset_id=dataset.id,
            )

        url = f"https://smith.langchain.com/datasets/{dataset.id}"
        print(
            f"📊 [LangSmith] Eval dataset '{_DATASET_NAME}' refreshed: "
            f"{len(rows)} examples → {url}"
        )
        return url

    except Exception as e:
        print(f"❌ [LangSmith] create_eval_dataset failed: {e}")
        return None
