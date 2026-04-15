"""
graph.py — builds and compiles the full LangGraph pipeline.
Flow: fetch_data → analyze_news → analyze_tech → decide
      → assess_risk → size_position → check_execution → alert
"""

from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.data_agent      import data_node
from agents.news_agent      import news_node
from agents.tech_agent      import tech_node
from agents.decision_agent  import decision_node
from agents.risk_agent      import risk_node
from agents.sizing_agent    import sizing_node
from agents.execution_agent import execution_node
from agents.alert_agent     import alert_node


# ── Shared state schema ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Config / input
    ticker:        str
    timestamp:     str
    paper_trading: bool

    # Data node
    current_price:  float
    prev_close:     float
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
    alert_sent: bool


# ── Graph factory ──────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("fetch_data",      data_node)
    g.add_node("analyze_news",    news_node)
    g.add_node("analyze_tech",    tech_node)
    g.add_node("decide",          decision_node)
    g.add_node("assess_risk",     risk_node)
    g.add_node("size_position",   sizing_node)
    g.add_node("check_execution", execution_node)
    g.add_node("alert",           alert_node)

    g.set_entry_point("fetch_data")
    g.add_edge("fetch_data",      "analyze_news")
    g.add_edge("analyze_news",    "analyze_tech")
    g.add_edge("analyze_tech",    "decide")
    g.add_edge("decide",          "assess_risk")
    g.add_edge("assess_risk",     "size_position")
    g.add_edge("size_position",   "check_execution")
    g.add_edge("check_execution", "alert")
    g.add_edge("alert",           END)

    return g.compile()


# ── Compiled singleton ─────────────────────────────────────────────────────────

GRAPH = build_graph()


def make_initial_state(ticker: str, paper_trading: bool = False) -> AgentState:
    return AgentState(
        ticker=ticker,
        timestamp=datetime.now(timezone.utc).isoformat(),
        paper_trading=paper_trading,
        current_price=0.0,
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
    )
