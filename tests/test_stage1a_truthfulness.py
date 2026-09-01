from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.ml.features import build_model_features
from app.ml.validation import evaluate_predictions, validate_observations, walk_forward


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


def test_walk_forward_rejects_unsorted_observations() -> None:
    rows = [
        {"observed_at": "2026-01-01T00:01:00+00:00"},
        {"observed_at": "2026-01-01T00:00:00+00:00"},
    ]
    with pytest.raises(ValueError, match="chronological"):
        validate_observations(rows)


def test_walk_forward_rejects_duplicate_observations() -> None:
    rows = [
        {"observed_at": "2026-01-01T00:00:00+00:00"},
        {"observed_at": "2026-01-01T00:00:00+00:00"},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_observations(rows)


def test_evaluation_keeps_prediction_count_separate_from_trade_count() -> None:
    result = evaluate_predictions([
        (1, 1, 0.01),
        (0, 0, 0.0),
        (-1, 1, -0.02),
        (1, 1, 0.0),
    ])
    assert result["predictions"] == 4
    assert result["trades"] == 3
    assert result["avg_return"] == pytest.approx(-0.0033333333333333335)
    assert result["class_support"] == {"-1": 1, "0": 1, "1": 2}


def test_evaluation_rejects_non_finite_trade_return() -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        evaluate_predictions([(1, 1, float("nan"))])


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


def test_empty_feature_contract_is_explicit() -> None:
    features = build_model_features([])
    assert len(features) == 12
    assert all(value == 0.0 for value in features.values())


@pytest.mark.parametrize("bad_rows", [None, []])
def test_empty_data_contract_is_explicit(bad_rows) -> None:
    features = build_model_features(bad_rows)
    assert len(features) == 12
