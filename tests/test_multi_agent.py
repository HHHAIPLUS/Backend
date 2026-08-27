from ai.agents import AgentAction, AgentContext, IntelligenceCouncil


def context(**overrides):
    data = dict(symbol="BTCUSDT", momentum=.6, trend_strength=.8, buying_pressure=.8, selling_pressure=.2, volatility=.2, liquidity_stress=.1, news_risk=.1, news_sentiment=.6, news_credibility=.9, funding_bias=.1, open_interest_change=.2, correlation_risk=.1, market_regime="trending_up", position_side="long", unrealized_return=.02, thesis_integrity=.9)
    data.update(overrides)
    return AgentContext(**data)


def test_council_runs_specialists_and_returns_auditable_votes():
    decision = IntelligenceCouncil().deliberate(context())
    assert len(decision.agents) == 9
    assert any(a.agent_id == "adversarial" for a in decision.agents)
    assert -1 <= decision.score <= 1
    assert 0 <= decision.confidence <= 1


def test_extreme_news_can_trigger_risk_off_veto():
    decision = IntelligenceCouncil().deliberate(context(news_risk=.95, news_sentiment=-.9, news_credibility=.95))
    assert "extreme_news_risk" in decision.veto_flags
    assert decision.action == AgentAction.RISK_OFF


def test_disagreement_reduces_confidence():
    decision = IntelligenceCouncil().deliberate(context(momentum=.05, buying_pressure=.95, selling_pressure=.05, news_sentiment=-.95, news_risk=.05))
    assert decision.disagreement >= 0
    assert decision.confidence <= 1
