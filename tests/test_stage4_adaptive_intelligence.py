from app.ml.adaptive_intelligence import AdaptiveIntelligence, AdaptiveObservation


def obs(i, model="champion", regime="trend", confidence=.8, ret=.01, feature=0.0):
    return AdaptiveObservation(
        symbol="BTCUSDT", model_version=model, action="LONG", confidence=confidence,
        realized_return=ret, observed_at=f"2026-01-01T00:{i:02d}:00+00:00",
        regime=regime, horizon=6, expected_probability=confidence,
        features={"trend_strength": feature, "volatility_proxy": .02},
    )


def test_regime_and_model_reliability_are_learned():
    engine = AdaptiveIntelligence()
    for i in range(40):
        engine.add_observation(obs(i, ret=.01 if i % 4 else -.002))
    report = engine.report()
    assert report.samples == 40
    assert report.regime_reliability["trend"]["samples"] == 40
    assert report.model_reliability["champion"]["samples"] == 40
    assert 0 < report.learned_weights["champion"] <= 1


def test_confidence_is_reduced_for_unfamiliar_state():
    engine = AdaptiveIntelligence()
    for i in range(60):
        engine.add_observation(obs(i, feature=0.0))
    adapted = engine.adapt_confidence(.9, model_version="champion", regime="trend", features={"trend_strength": 20.0, "volatility_proxy": .02})
    assert adapted["unfamiliar"] is True
    assert adapted["adjusted_confidence"] < adapted["raw_confidence"]


def test_concept_and_feature_drift_are_detected():
    engine = AdaptiveIntelligence()
    for i in range(120):
        engine.add_observation(obs(i, feature=0.0, ret=.01))
    for i in range(120, 240):
        engine.add_observation(obs(i, feature=5.0, ret=-.01))
    drift = engine.drift_report(window=100)
    assert drift["max_psi"] > 0
    assert drift["concept_drift"] is True


def test_challenger_stays_quarantined_until_evaluated_and_can_be_promoted():
    engine = AdaptiveIntelligence()
    candidate = engine.create_candidate("champion-v1", "challenger-v2", "validated experiment", {})
    assert candidate.status == "quarantined"
    champion = [0.001] * 120
    challenger = [0.002] * 120
    result = engine.evaluate_challenger(candidate.candidate_id, champion, challenger, regimes=["trend"] * 120)
    assert result.status == "promotion_eligible"
    assert result.evidence["promoted"] is True


def test_challenger_is_rejected_when_not_better():
    engine = AdaptiveIntelligence()
    candidate = engine.create_candidate("champion-v1", "challenger-v2", "validated experiment", {})
    champion = [0.002] * 120
    challenger = [0.001] * 120
    result = engine.evaluate_challenger(candidate.candidate_id, champion, challenger)
    assert result.status == "rejected"
