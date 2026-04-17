"""
tests/test_pipeline_audit.py — validates the real end-to-end pipeline.

Uses GRAPH.invoke() directly so the test exercises real node logic,
not mocked states.
"""


def test_pipeline_audit_real():
    from graph import make_initial_state, GRAPH

    state = make_initial_state("AAPL")
    state["paper_trading"] = True

    result = GRAPH.invoke(state)

    critical_fields = [
        "signal", "confidence", "agreement_score",
        "consensus", "validator_passed",
        "stoch_rsi_signal", "risk_approved",
        "should_alert", "alert_sent",
    ]

    missing = [f for f in critical_fields if f not in result]

    assert not missing, f"Pipeline missing fields: {missing}"
    assert result["signal"] in ("BUY", "SELL", "HOLD")
    assert result.get("stoch_rsi_signal") not in (None, "NONE", "")
    assert "validator_passed" in result

    print("✅ Real pipeline audit passed")
    print(f"   Signal: {result['signal']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Agreement: {result.get('agreement_score')}")
    print(f"   Stoch RSI: {result.get('stoch_rsi_signal')}")
    print(f"   Validator: {result.get('validator_passed')}")
