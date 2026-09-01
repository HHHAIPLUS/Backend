from app.ml.historical_context import REQUIRED_CONTEXT, make_context
from app.ml.features import build_model_features


def test_historical_context_values_reach_model_features():
    context = make_context(
        "2026-01-01T00:00:00Z",
        {name: (i + 1) / 100 for i, name in enumerate(REQUIRED_CONTEXT)},
    )
    candles = [[1704067200000, 100, 101, 99, 100, 1000], [1704067500000, 100, 102, 99, 101, 1100]]
    features = build_model_features(candles, context=context)
    assert features["order_book_imbalance"] == 0.01
    assert features["funding_rate"] == 0.02
    assert features["open_interest_change"] == 0.03
    assert features["news_risk"] == 0.04
    assert features["news_sentiment"] == 0.05
    assert features["liquidity_stress"] == 0.06
