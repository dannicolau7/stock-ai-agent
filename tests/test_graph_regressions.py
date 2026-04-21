"""
tests/test_graph_regressions.py — focused coverage for the newer graph path.

These tests target the aggregator / validator / hub interactions that are not
covered by the older system tests.
"""

from unittest.mock import MagicMock

import pytest

import graph


def _base_state(ticker: str = "AAPL") -> dict:
    state = graph.make_initial_state(ticker)
    state.update({
        "current_price": 100.0,
        "price": 100.0,
        "timestamp": "2026-04-16T13:30:00+00:00",
    })
    return state


class TestSignalAggregator:
    def test_parallel_analyze_preserves_stoch_rsi_fields(self, monkeypatch):
        monkeypatch.setattr(graph, "news_node", lambda state: {
            **state,
            "news_sentiment": "BULLISH",
            "sentiment_score": 77.0,
            "news_summary": "fresh catalyst",
            "social_velocity": {"velocity": 2.0, "label": "rising"},
        })
        monkeypatch.setattr(graph, "tech_node", lambda state: {
            **state,
            "rsi": 61.0,
            "stoch_rsi": 0.84,
            "stoch_rsi_k": 0.84,
            "stoch_rsi_d": 0.73,
            "stoch_rsi_signal": "OVERBOUGHT",
            "macd": {"histogram": 0.3},
            "bollinger": {"upper": 1.0, "middle": 0.5, "lower": 0.2, "bandwidth": 0.1},
            "atr": 2.0,
            "support": 95.0,
            "resistance": 105.0,
            "sr_levels": {},
            "volume_spike": True,
            "volume_spike_ratio": 2.1,
            "vwap": 100.0,
            "obv": 1.0,
            "smart_money": "BULLISH",
            "ema_stack": {"alignment": "BULLISH"},
            "float_rotation": 0.5,
            "sector_momentum": {"signal": "BULLISH"},
            "timing": {"multiplier": 1.2, "window": "open"},
            "gap_info": {"signal": "NEUTRAL", "gap_pct": 0.0},
            "market_regime": {"regime": "BULL"},
            "relative_strength": {"rs_vs_spy": 1.2, "label": "strong", "spy_chg": 0.4},
            "score_breakdown": {"raw_score": 70, "final_score": 84, "timing_mult": 1.2, "fired": [], "missed": []},
            "patterns": [],
        })

        result = graph.parallel_analyze_node(_base_state("AMD"))

        assert result["stoch_rsi_signal"] == "OVERBOUGHT"
        assert result["stoch_rsi_k"] == 0.84
        assert result["news_sentiment"] == "BULLISH"

    def test_edgar_8k_prevents_skip_claude_even_when_agreement_is_low(self, monkeypatch):
        from agents import signal_aggregator as agg

        monkeypatch.setattr(
            agg.hub,
            "get_reflection_weights",
            lambda: {},
        )
        monkeypatch.setattr(
            agg.hub,
            "get_regime_thresholds",
            lambda: {"agreement_min": 80, "regime": "BEAR"},
        )

        state = _base_state("MSFT")
        state.update({
            "has_edgar_filing": True,
            "edgar_filing_type": "8-K",
            "rsi": 50.0,
            "macd": {"histogram": 0.0},
            "ema_stack": {"alignment": "MIXED"},
            "vwap": 0.0,
            "bollinger": {"upper": 0.0, "lower": 0.0},
            "volume_spike_ratio": 1.0,
            "gap_info": {"signal": "NEUTRAL", "gap_pct": 0.0},
            "patterns": [],
            "smart_money": "NEUTRAL",
            "raw_news": [],
            "market_regime": {"regime": "UNKNOWN"},
            "sector_momentum": {"signal": "NEUTRAL"},
        })

        result = agg.aggregator_node(state)

        assert result["major_catalyst"] is True
        assert result["skip_claude"] is False


class TestDecisionValidator:
    def test_news_triggered_signal_below_65_should_not_be_forced_to_hold(self):
        from agents.decision_validator import validator_node

        state = _base_state("NVDA")
        state.update({
            "signal": "BUY",
            "confidence": 60,
            "rsi": 45.0,
            "atr": 2.0,
            "agreement_score": 90.0,
            "raw_news": [{"title": "Fresh catalyst"}],
            "volume_spike_ratio": 2.5,
            "gap_info": {"gap_pct": 2.0},
            "market_regime": {"regime": "BULL"},
            "entry_low": 99.0,
            "entry_high": 101.0,
            "stop_loss": 96.0,
            "stop_pct": -4.0,
            "news_triggered": True,
        })

        result = validator_node(state)

        assert result["signal"] == "BUY"


class TestDecisionHubInteraction:
    def test_decision_stage_should_not_mark_hub_alert_early(self, monkeypatch):
        from agents import decision_agent as da

        monkeypatch.setattr(da.hub, "was_alerted_today", lambda ticker: False)
        monkeypatch.setattr(da.hub, "get_portfolio_context", lambda ticker: {"already_open": False})
        monkeypatch.setattr(da.hub, "get_regime_thresholds", lambda: {"confidence_cap": 100, "regime": "BULL"})

        marked = {"called": False}

        def _mark_alerted(ticker, signal):
            marked["called"] = True

        monkeypatch.setattr(da.hub, "mark_alerted", _mark_alerted)
        monkeypatch.setattr(da, "analyze_market", lambda state: {
            "signal": "BUY",
            "confidence": 70,
            "entry_zone": "$99 - $101",
            "entry_low": 99.0,
            "entry_high": 101.0,
            "targets": [105.0, 110.0, 120.0],
            "stop_loss": 96.0,
            "stop_pct": -4.0,
            "reasoning": "test",
            "action_plan": "",
            "rr_ratio": 2.0,
            "trade_horizon": "swing",
            "horizon_reasoning": "",
            "main_risk": "",
            "top_3_signals": [],
            "bull_score": 70,
            "bear_score": 35,
            "net_score": 35,
            "bull_summary": "Strong momentum with volume catalyst",
            "bear_summary": "Minor overhead resistance near target",
        })

        import performance_tracker as pt
        monkeypatch.setattr(pt, "record_decision_audit", lambda state: None)

        state = _base_state("AAPL")
        state["score_breakdown"] = {"final_score": 70, "raw_score": 70}

        da.decision_node(state)

        assert marked["called"] is False


class TestAlertAgent:
    def test_hub_marked_only_after_successful_delivery(self, monkeypatch):
        from agents import alert_agent as aa
        import performance_tracker as pt
        from intelligence_hub import hub

        monkeypatch.setattr(aa, "_fresh_price", lambda ticker, fetched_at, current: current)
        monkeypatch.setattr(pt, "is_alert_fired", lambda *args, **kwargs: False)
        monkeypatch.setattr(pt, "mark_alert_fired", lambda *args, **kwargs: None)
        monkeypatch.setattr(pt, "record_signal", lambda *args, **kwargs: None)
        monkeypatch.setattr(aa, "send_alert", lambda **kwargs: True)

        marked = {"count": 0}
        monkeypatch.setattr(hub, "mark_alerted", lambda ticker, signal: marked.__setitem__("count", marked["count"] + 1))

        state = _base_state("TSLA")
        state.update({
            "signal": "BUY",
            "confidence": 80,
            "should_alert": True,
            "entry_zone": "$99 - $101",
            "targets": [105.0, 110.0],
            "stop_loss": 96.0,
            "reasoning": "test",
            "trade_horizon": "swing",
        })

        result = aa.alert_node(state)

        assert result["alert_sent"] is True
        assert result["alert_reason_code"] == "sent"
        assert marked["count"] == 1
