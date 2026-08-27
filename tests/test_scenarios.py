from ai.scenario_engine import ScenarioEngine, ScenarioKind, ScenarioRequest


def base(**overrides):
    data = dict(symbol="BTCUSDT", horizon_minutes=60, momentum=0.7, trend_strength=0.8,
                buying_pressure=0.85, selling_pressure=0.15, volatility=0.25,
                liquidity_stress=0.10, news_risk=0.10, news_sentiment=0.6,
                market_risk=0.15, thesis_integrity=0.8)
    data.update(overrides)
    return ScenarioRequest(**data)


def test_probabilities_form_a_distribution():
    report = ScenarioEngine().generate(base())
    assert len(report.scenarios) == 5
    assert abs(sum(x.probability for x in report.scenarios) - 1) < 1e-9


def test_bullish_conditions_favor_bullish_scenario():
    report = ScenarioEngine().generate(base())
    assert report.dominant_scenario == ScenarioKind.BULLISH_CONTINUATION


def test_risk_off_conditions_favor_risk_off():
    report = ScenarioEngine().generate(base(momentum=-0.7, trend_strength=0.6, buying_pressure=0.1,
                                           selling_pressure=0.95, volatility=0.8, liquidity_stress=0.95,
                                           news_risk=0.9, news_sentiment=-0.9, market_risk=0.95,
                                           thesis_integrity=0.15))
    assert report.dominant_scenario == ScenarioKind.DISORDERLY_RISK_OFF


def test_uncertainty_is_exposed():
    report = ScenarioEngine().generate(base(momentum=0, trend_strength=0.2, buying_pressure=0.5,
                                             selling_pressure=0.5, volatility=0.5, liquidity_stress=0.2,
                                             news_risk=0.2, news_sentiment=0, market_risk=0.2,
                                             thesis_integrity=0.5))
    assert 0 <= report.uncertainty <= 1
    assert report.recommendation
