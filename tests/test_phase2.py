from datetime import datetime, timezone

from ai.models import FeatureVector, MarketRegime
from ai.predictor import BaselinePredictor


def test_baseline_predictor_returns_probabilities():
    features = FeatureVector(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        features={"return_1": 0.001, "range_pct": 0.01},
    )
    prediction = BaselinePredictor().predict(features, MarketRegime.TRENDING_UP)

    total = (
        prediction.long_probability
        + prediction.short_probability
        + prediction.no_trade_probability
    )
    assert abs(total - 1.0) < 1e-9
    assert 0 <= prediction.confidence <= 1
