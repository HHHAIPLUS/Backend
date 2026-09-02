from app.ml.adaptive_conditions import conditioned_performance, adapt_confidence
from app.ml.adaptive_intelligence import AdaptiveIntelligence, AdaptiveObservation


def test_model_regime_horizon_conditioning():
    engine = AdaptiveIntelligence()
    for i in range(35):
        engine.add_observation(AdaptiveObservation('BTCUSDT','v1','LONG',.8,.01,f'2026-01-01T00:{i:02d}:00+00:00','trend',6,.8,{'trend_strength':.5}))
    result = conditioned_performance(engine)
    assert result['v1|trend|6']['samples'] == 35
    adapted = adapt_confidence(engine,.9,model_version='v1',regime='trend',horizon=6,features={'trend_strength':.5})
    assert adapted['conditioned'] is True
    assert adapted['conditioned_samples'] == 35
