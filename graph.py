"""
graph.py — builds and compiles the full LangGraph pipeline.
Flow: fetch_data → analyze_news → analyze_tech → decide
      → assess_risk → size_position → check_execution → alert

Includes a lightweight audit runner for validating whether each stage
materially changes the trade state.
"""

from datetime import datetime, timezone
from typing import TypedDict
import copy

from langgraph.graph import StateGraph, END
from langsmith import traceable

import concurrent.futures

from agents.data_agent         import data_node
from agents.news_agent         import news_node
from agents.tech_agent         import tech_node
from agents.signal_aggregator  import aggregator_node
from agents.decision_agent     import decision_node
from agents.decision_validator import validator_node
from agents.risk_agent         import risk_node
from agents.sizing_agent       import sizing_node
from agents.execution_agent    import execution_node
from agents.alert_agent        import alert_node


# ── Parallel news + tech node ──────────────────────────────────────────────────
# Runs news_node and tech_node simultaneously in threads, then merges their
# unique outputs. Cuts data-collection wall-time from ~30s to ~10s.

_NEWS_KEYS = frozenset({
    "news_sentiment", "sentiment_score", "news_summary", "social_velocity",
})
_TECH_KEYS = frozenset({
    "rsi", "intraday_rsi", "stoch_rsi", "stoch_rsi_k", "stoch_rsi_d",
    "stoch_rsi_signal", "macd", "bollinger", "atr", "support", "resistance",
    "sr_levels", "volume_spike", "volume_spike_ratio", "vwap", "obv", "smart_money",
    "ema_stack", "float_rotation", "sector_momentum", "timing", "gap_info",
    "market_regime", "relative_strength", "score_breakdown", "patterns",
})


@traceable(name="parallel_analyze", tags=["pipeline", "parallel"])
def parallel_analyze_node(state: dict) -> dict:
    """Runs news_node and tech_node concurrently, then merges their outputs."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as exe:
        fut_news = exe.submit(news_node, dict(state))
        fut_tech = exe.submit(tech_node, dict(state))
        news_result = fut_news.result()
        tech_result = fut_tech.result()

    # Only pull keys each node is responsible for — avoids stale-state overwrites
    news_delta = {k: v for k, v in news_result.items() if k in _NEWS_KEYS}
    tech_delta = {k: v for k, v in tech_result.items() if k in _TECH_KEYS}
    return {**state, **news_delta, **tech_delta}


# ── Shared state schema ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Config / input
    ticker:        str
    timestamp:     str
    pass  # paper_trading removed

    # Data node
    current_price:    float
    price_fetched_at: str
    prev_close:       float
    volume:         float
    avg_volume:     float
    volume_ratio:   float
    bars:           list
    raw_news:       list
    ticker_details: dict
    error:          str

    # News node
    news_sentiment:  str
    sentiment_score: float
    news_summary:    str

    # Data node extras
    sector:            str
    premarket_price:   float
    earnings_info:     dict
    intraday_bars:     list

    # Tech node
    rsi:               float
    intraday_rsi:      float
    stoch_rsi:         float
    stoch_rsi_k:       float
    stoch_rsi_d:       float
    stoch_rsi_signal:  str
    macd:              dict
    bollinger:         dict
    atr:               float
    support:           float
    resistance:        float
    sr_levels:         dict
    volume_spike:      bool
    volume_spike_ratio: float
    vwap:              float
    obv:               float
    smart_money:       str
    ema_stack:         dict
    float_rotation:    float
    sector_momentum:   dict
    timing:            dict
    gap_info:          dict
    market_regime:     dict
    relative_strength: dict
    score_breakdown:   dict

    # News node extras
    social_velocity:   dict

    # Tech node extras (pattern detection)
    patterns: list

    # Decision node
    signal:            str
    confidence:        float
    entry_zone:        str
    targets:           list
    stop_loss:         float
    reasoning:         str
    action_plan:       str
    rr_ratio:          float
    should_alert:      bool
    trade_horizon:     str
    horizon_reasoning: str

    # Dedup guard — set True if ticker already has an active position
    already_alerted: bool

    # Set True when pipeline is triggered by a news/spike/EDGAR event (lowers threshold to 55)
    news_triggered: bool

    # Risk node
    risk_approved:    bool
    risk_veto_reason: str
    risk_multiplier:  float
    risk_warnings:    list

    # Sizing node
    position_size_pct: float
    position_size_usd: float
    max_shares:        int
    risk_dollars:      float
    scale_in:          bool
    size_reasoning:    str

    # Execution node
    executable:        bool
    execution_reason:  str
    order_type:        str

    # Alert node
    alert_sent:        bool
    alert_reason_code: str

    # Decision audit — carried through pipeline for traceability
    model_signal:     str
    model_confidence: float
    threshold_used:   int
    decision_delta:   str

    # Convenience price/zone aliases (set by decision_node, fixed by validator)
    price:      float
    entry_low:  float
    entry_high: float
    stop_pct:   float

    # Signal aggregator
    has_edgar_filing:   bool
    bullish_signals:    list
    bearish_signals:    list
    agreement_score:    float
    consensus:          str
    signal_count_bull:  int
    signal_count_bear:  int
    skip_claude:        bool
    skip_reason:        str

    # Decision validator
    final_signal:        str
    validator_passed:    bool
    validator_overrides: list

    # Extra Claude fields
    main_risk:     str
    top_3_signals: list


# ── Audit helpers ───────────────────────────────────────────────────────────────

PIPELINE_NODES = [
    ("fetch_data",        data_node),
    ("parallel_analyze",  parallel_analyze_node),   # news + tech concurrently
    ("aggregate_signals", aggregator_node),
    ("decide",            decision_node),
    ("validate_decision", validator_node),
    ("assess_risk",       risk_node),
    ("size_position",     sizing_node),
    ("check_execution",   execution_node),
    ("alert",             alert_node),
]


def _audit_snapshot(state: dict) -> dict:
    return {
        "ticker": state.get("ticker"),
        "current_price": state.get("current_price"),
        "news_sentiment": state.get("news_sentiment"),
        "sentiment_score": state.get("sentiment_score"),
        "rsi": state.get("rsi"),
        "volume_spike_ratio": state.get("volume_spike_ratio"),
        "score_breakdown": copy.deepcopy(state.get("score_breakdown")),
        "signal": state.get("signal"),
        "confidence": state.get("confidence"),
        "entry_zone": state.get("entry_zone"),
        "targets": copy.deepcopy(state.get("targets")),
        "stop_loss": state.get("stop_loss"),
        "should_alert": state.get("should_alert"),
        "risk_approved": state.get("risk_approved"),
        "risk_veto_reason": state.get("risk_veto_reason"),
        "risk_multiplier": state.get("risk_multiplier"),
        "risk_warnings": copy.deepcopy(state.get("risk_warnings")),
        "position_size_pct": state.get("position_size_pct"),
        "position_size_usd": state.get("position_size_usd"),
        "max_shares": state.get("max_shares"),
        "scale_in": state.get("scale_in"),
        "executable": state.get("executable"),
        "execution_reason": state.get("execution_reason"),
        "order_type": state.get("order_type"),
        "alert_sent": state.get("alert_sent"),
        "alert_reason_code": state.get("alert_reason_code"),
        "model_signal": state.get("model_signal"),
        "model_confidence": state.get("model_confidence"),
        "threshold_used": state.get("threshold_used"),
        "decision_delta": state.get("decision_delta"),
        # Fields added to capture validator, aggregator, and catalyst data
        "final_signal": state.get("final_signal"),
        "validator_passed": state.get("validator_passed"),
        "validator_overrides": copy.deepcopy(state.get("validator_overrides")),
        "agreement_score": state.get("agreement_score"),
        "consensus": state.get("consensus"),
        "skip_claude": state.get("skip_claude"),
        "skip_reason": state.get("skip_reason"),
        "stoch_rsi_signal": state.get("stoch_rsi_signal"),
        "stoch_rsi": state.get("stoch_rsi"),
        "major_catalyst": state.get("major_catalyst"),
        "pre_identified": state.get("pre_identified"),
        "pre_identified_reason": state.get("pre_identified_reason"),
        "has_edgar_filing": state.get("has_edgar_filing"),
        "edgar_filing_type": state.get("edgar_filing_type"),
        "signal_count_bull": state.get("signal_count_bull"),
        "signal_count_bear": state.get("signal_count_bear"),
        "bullish_signals": copy.deepcopy(state.get("bullish_signals")),
        "bearish_signals": copy.deepcopy(state.get("bearish_signals")),
        "market_regime": state.get("market_regime"),
        "circuit_breaker_active": state.get("circuit_breaker_active"),
    }


def _diff_snapshots(before: dict, after: dict) -> dict:
    changed = {}
    for key, value in after.items():
        if before.get(key) != value:
            changed[key] = {"before": before.get(key), "after": value}
    return changed


def run_pipeline_audit(initial_state: dict) -> dict:
    """
    Run the pipeline sequentially and capture state deltas after each node.

    This provides a concrete answer to "is the agentic part working?" by
    showing whether each stage contributes meaningful changes.
    """
    state = copy.deepcopy(initial_state)
    initial = _audit_snapshot(state)
    steps = []

    for step_name, node in PIPELINE_NODES:
        before = _audit_snapshot(state)
        state = node(state)
        after = _audit_snapshot(state)
        steps.append({
            "step": step_name,
            "changed": _diff_snapshots(before, after),
            "snapshot": after,
        })

    return {
        "initial": initial,
        "steps": steps,
        "final": _audit_snapshot(state),
        "material_steps": [step["step"] for step in steps if step["changed"]],
    }


# ── Graph factory ──────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("fetch_data",        data_node)
    g.add_node("parallel_analyze",  parallel_analyze_node)
    g.add_node("aggregate_signals", aggregator_node)
    g.add_node("decide",            decision_node)
    g.add_node("validate_decision", validator_node)
    g.add_node("assess_risk",       risk_node)
    g.add_node("size_position",     sizing_node)
    g.add_node("check_execution",   execution_node)
    g.add_node("alert",             alert_node)

    g.set_entry_point("fetch_data")
    g.add_edge("fetch_data",        "parallel_analyze")
    g.add_edge("parallel_analyze",  "aggregate_signals")
    g.add_edge("aggregate_signals", "decide")
    g.add_edge("decide",            "validate_decision")
    g.add_edge("validate_decision", "assess_risk")
    g.add_edge("assess_risk",       "size_position")
    g.add_edge("size_position",     "check_execution")
    g.add_edge("check_execution",   "alert")
    g.add_edge("alert",             END)

    return g.compile()


# ── Compiled singleton ─────────────────────────────────────────────────────────

GRAPH = build_graph()


def make_initial_state(ticker: str) -> AgentState:
    return AgentState(
        ticker=ticker,
        timestamp=datetime.now(timezone.utc).isoformat(),
        current_price=0.0,
        price_fetched_at="",
        prev_close=0.0,
        volume=0.0,
        avg_volume=0.0,
        bars=[],
        raw_news=[],
        ticker_details={},
        volume_ratio=0.0,
        error="",
        news_sentiment="NEUTRAL",
        sentiment_score=50.0,
        news_summary="",
        rsi=50.0,
        stoch_rsi=0.5,
        stoch_rsi_k=0.5,
        stoch_rsi_d=0.5,
        stoch_rsi_signal="NEUTRAL",
        macd={"macd": 0.0, "signal": 0.0, "histogram": 0.0},
        bollinger={"upper": 0.0, "middle": 0.0, "lower": 0.0, "bandwidth": 0.0},
        atr=0.0,
        support=0.0,
        resistance=0.0,
        sector="Technology",
        premarket_price=0.0,
        earnings_info={"days_to_earnings": 999, "earnings_risk": "none", "earnings_date": ""},
        intraday_bars=[],
        volume_spike=False,
        volume_spike_ratio=1.0,
        vwap=0.0,
        obv=0.0,
        intraday_rsi=50.0,
        sr_levels={},
        smart_money="NEUTRAL",
        ema_stack={"alignment": "MIXED", "ema9": 0.0, "ema21": 0.0, "ema50": 0.0},
        float_rotation=0.0,
        sector_momentum={"etf": "SPY", "change_pct": 0.0, "signal": "NEUTRAL"},
        timing={"multiplier": 1.0, "window": "unknown"},
        gap_info={"gap_pct": 0.0, "signal": "NEUTRAL", "label": ""},
        market_regime={"regime": "UNKNOWN", "spy_price": 0.0, "spy_day_chg": 0.0},
        relative_strength={"rs_vs_spy": 0.0, "label": "n/a", "spy_chg": 0.0},
        score_breakdown={"raw_score": 0, "final_score": 0, "timing_mult": 1.0, "fired": [], "missed": []},
        social_velocity={"velocity": 1.0, "label": "no data"},
        patterns=[],
        signal="HOLD",
        confidence=0.0,
        entry_zone="",
        targets=[],
        stop_loss=0.0,
        reasoning="",
        action_plan="",
        rr_ratio=0.0,
        should_alert=False,
        trade_horizon="swing",
        horizon_reasoning="",
        already_alerted=False,
        news_triggered=False,
        # Risk node
        risk_approved=True,
        risk_veto_reason="",
        risk_multiplier=1.0,
        risk_warnings=[],
        # Sizing node
        position_size_pct=0.0,
        position_size_usd=0.0,
        max_shares=0,
        risk_dollars=0.0,
        scale_in=False,
        size_reasoning="",
        # Execution node
        executable=False,
        execution_reason="",
        order_type="NONE",
        alert_sent=False,
        alert_reason_code="not_evaluated",
        model_signal="HOLD",
        model_confidence=0.0,
        threshold_used=0,
        decision_delta="",
        # Convenience aliases
        price=0.0,
        entry_low=0.0,
        entry_high=0.0,
        stop_pct=0.0,
        # Signal aggregator
        has_edgar_filing=False,
        bullish_signals=[],
        bearish_signals=[],
        agreement_score=0.0,
        consensus="NEUTRAL",
        signal_count_bull=0,
        signal_count_bear=0,
        skip_claude=False,
        skip_reason="",
        # Decision validator
        final_signal="HOLD",
        validator_passed=False,
        validator_overrides=[],
        # Extra Claude fields
        main_risk="",
        top_3_signals=[],
    )
