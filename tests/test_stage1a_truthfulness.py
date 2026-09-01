from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.ml.features import build_model_features
from app.ml.validation import evaluate_predictions, walk_forward


def _candles(count: int = 60) -> list[list[float]]:
    base = 100.0
    rows: list[list[float]] = []
    for i in range(count):
        close = base + i * 0.1
        rows.append([
            1_700_000_000_000 + i * 300_000,
            close - 0.05,
            close + 0.10,
            close - 0.10,
            close,
            1000.0 + i,
        ])
    return rows


def test_walk_forward_is_strictly_chronological() -> None:
    rows = [
        {"observed_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()},
        {"observed_at": datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc).isoformat()},
        {"observed_at": datetime(2026, 1, 1, 0, 2, tzinfo=timezone.utc).isoformat()},
        {"observed_at": datetime(2026, 1, 1, 0, 3, tzinfo=timezone.utc).isoformat()},
    ]
    folds = walk_forward(rows, min_train=2, test_size=1, step=1)
    assert folds
    for fold in folds:
        assert fold.train[-1]["observed_at"] < fold.test[0]["observed_at"]


def test_evaluation_keeps_prediction_count_separate_from_trade_count() -> None:
    result = evaluate_predictions([
        (1, 1, 0.01),
        (0, 0, 0.0),
        (-1, 1, -0.02),
    ])
    assert result["predictions"] == 3
    assert result["trades"] == 2
    assert result["class_support"] == {"-1": 1, "0": 1, "1": 1}


def test_feature_contract_is_exact_and_finite() -> None:
    features = build_model_features(_candles())
    expected = {
        "return_1", "range_pct", "volume_change", "order_book_imbalance",
        "funding_rate", "open_interest_change", "news_risk", "news_sentiment",
        "volatility_proxy", "trend_strength", "momentum", "liquidity_stress",
    }
    assert set(features) == expected
    assert all(value == value for value in features.values())
    assert all(abs(value) != float("inf") for value in features.values())


def test_model_features_do_not_require_future_candles() -> None:
    candles = _candles()
    earlier = build_model_features(candles[:40])
    later = build_model_features(candles[:41])
    assert earlier != later


def test_feature_builder_rejects_no_data_only_by_neutral_contract() -> None:
    # Empty candle input is allowed by the compatibility boundary, but it must
    # produce a complete finite feature vector rather than a partial vector.
    features = build_model_features([])
    assert len(features) == 12
    assert all(value == 0.0 for value in features.values())


@pytest.mark.parametrize("bad_rows", [None, []])
def test_empty_data_contract_is_explicit(bad_rows) -> None:
    features = build_model_features(bad_rows)
    assert len(features) == 12
