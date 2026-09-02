from ai.agents import AgentContext
from ai.stage5_engine import Stage5DecisionEngine
from app.ml.adaptive_intelligence import AdaptiveIntelligence, AdaptiveObservation


def state(**overrides):
    base = {
        "symbol": "BTCUSDT", "usable": True, "data_quality": 1.0,
        "price_structure": {"trend": 0.8, "last_return": 0.01},
        "timeframes": {"5m": {"return": 0.02}},
        "volatility": {"realized": 0.003},
        "volume": {"ratio": 1.4},
        "order_flow": {"aggressive_buy_ratio": 0.72},
        "derivatives": {"funding_rate": 0.0001, "open_interest_change": 0.02},
        "liquidity": {"spread_bps": 3},
        "regime": {"label": "TRENDING_UP", "market_risk": 0.15},
        "news": {"risk": 0.1, "sentiment": 0.5, "credibility": 0.9},
    }
    base.update(overrides)
    return base


def predictive(**overrides):
    p = {"probabilities": {"long": 0.86, "short": 0.08}, "uncertainty": 0.05, "model_version": "champion", "horizon": 6}
    p.update(overrides)
    return p


def test_canonical_market_state_adapter():
    c = AgentContext.from_market_state(state())
    assert c.symbol == "BTCUSDT"
    assert c.buying_pressure > c.selling_pressure
    assert c.market_regime == "trending_up"


def test_stage5_returns_auditable_direction_when_all_gates_pass():
    engine = Stage5DecisionEngine(AdaptiveIntelligence())
    result = engine.evaluate(market_state=state(), predictive=predictive())
    assert result.action == "LONG"
    assert result.execution_candidate is True
    assert result.evidence["council"]
    assert result.evidence["scenarios"]["probabilities"]
    assert result.evidence["adversarial"]
    assert result.what_would_change


def test_risk_veto_is_absolute():
    engine = Stage5DecisionEngine(AdaptiveIntelligence())
    result = engine.evaluate(market_state=state(), predictive=predictive(), risk_vetoes=["portfolio_risk_limit"])
    assert result.action == "WAIT"
    assert result.execution_candidate is False
    assert "portfolio_risk_limit" in result.vetoes


def test_low_data_quality_abstains():
    engine = Stage5DecisionEngine(AdaptiveIntelligence())
    result = engine.evaluate(market_state=state(data_quality=0.4, usable=False), predictive=predictive())
    assert result.action == "WAIT"
    assert "insufficient_data_quality" in result.vetoes


def test_material_cross_layer_contradiction_abstains():
    engine = Stage5DecisionEngine(AdaptiveIntelligence())
    bearish = state(regime={"label": "TRENDING_DOWN", "market_risk": 0.2}, order_flow={"aggressive_buy_ratio": 0.2})
    result = engine.evaluate(market_state=bearish, predictive=predictive())
    assert result.action == "WAIT"
    assert result.execution_candidate is False
    assert "material_contradiction" in result.vetoes or "adversarial_block" in result.vetoes


def test_adaptive_agent_weights_are_used_when_evidence_exists():
    adaptive = AdaptiveIntelligence()
    for i in range(60):
        adaptive.add_observation(AdaptiveObservation("BTCUSDT", "agent:flow", "LONG", 0.8, 0.01, str(i), "trending_up", 6))
        adaptive.add_observation(AdaptiveObservation("BTCUSDT", "agent:news", "LONG", 0.6, -0.01, str(i), "trending_up", 6))
    engine = Stage5DecisionEngine(adaptive)
    result = engine.evaluate(market_state=state(), predictive=predictive())
    assert result.evidence["council"]["reasons"]
    assert "Specialist weights" in " ".join(result.evidence["council"]["reasons"])


def test_decision_quality_is_independent_of_trade_pnl():
    engine = Stage5DecisionEngine(AdaptiveIntelligence())
    result = engine.evaluate(market_state=state(), predictive=predictive())
    engine.record_outcome(decision=result, realized_return=0.01)
    quality = engine.decision_quality()
    assert quality.samples == 1
    assert quality.directional_accuracy == 1.0
    assert quality.mean_return == 0.01


def test_abstention_prevents_execution():
    adaptive = AdaptiveIntelligence()
    engine = Stage5DecisionEngine(adaptive)
    result = engine.evaluate(market_state=state(), predictive={"probabilities": {"long": 0.51, "short": 0.49}, "uncertainty": 0.01, "model_version": "new", "horizon": 6})
    assert result.action == "WAIT"
    assert result.execution_candidate is False
    assert "adaptive_abstention" in result.vetoes


def test_stage5_has_no_execution_authority():
    status = Stage5DecisionEngine(AdaptiveIntelligence()).status()
    assert status["execution_authority"] is False
    assert status["risk_vetoes_absolute"] is True
    assert status["production_model_mutation"] is False
