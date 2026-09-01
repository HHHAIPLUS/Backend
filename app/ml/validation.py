from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any


@dataclass
class Fold:
    train: list[dict]
    test: list[dict]


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("Each validation row must contain a non-empty observed_at timestamp.")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid observed_at timestamp: {value!r}") from exc


def validate_observations(rows: list[dict]) -> None:
    """Validate the temporal integrity of supervised observations.

    Rows must have unique, parseable timestamps.  Silent timestamp repair or
    sorting is deliberately avoided here because silently changing ordering
    can hide data-quality or leakage problems.
    """
    previous: datetime | None = None
    seen: set[datetime] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every validation row must be a mapping.")
        current = _timestamp(row.get("observed_at"))
        if current in seen:
            raise ValueError(f"Duplicate observed_at timestamp: {current.isoformat()}")
        if previous is not None and current <= previous:
            raise ValueError("Validation rows must be strictly chronological with unique timestamps.")
        seen.add(current)
        previous = current


def walk_forward(rows: list[dict], min_train: int = 300, test_size: int = 100, step: int = 100) -> list[Fold]:
    """Create chronological walk-forward folds without future leakage."""
    if min_train <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("min_train, test_size and step must be greater than zero.")
    if not rows:
        return []

    # Do not silently sort here.  The dataset builder is responsible for
    # chronological ordering; validation must fail loudly if that contract is
    # violated rather than masking a timestamp/data-quality defect.
    validate_observations(rows)

    folds: list[Fold] = []
    end = min_train
    while end + test_size <= len(rows):
        train = rows[:end]
        test = rows[end:end + test_size]
        if _timestamp(train[-1]["observed_at"]) >= _timestamp(test[0]["observed_at"]):
            raise ValueError("Walk-forward fold contains temporal overlap or leakage.")
        folds.append(Fold(train=train, test=test))
        end += step
    return folds


def evaluate_predictions(predictions: list[tuple[int, int, float]]) -> dict:
    """Evaluate classification plus simulated directional-return metrics.

    Tuple format: (actual_label, predicted_label, realized_return).
    A prediction with a non-zero predicted direction is a simulated trade,
    including a zero-return trade.  Zero predicted direction is abstention.
    """
    if not predictions:
        return {
            "trades": 0, "predictions": 0, "accuracy": 0.0,
            "balanced_accuracy": 0.0, "avg_return": 0.0, "total_return": 0.0,
            "class_recall": {"-1": 0.0, "0": 0.0, "1": 0.0},
            "class_support": {"-1": 0, "0": 0, "1": 0},
        }

    total = len(predictions)
    correct = sum(actual == predicted for actual, predicted, _ in predictions)

    trade_returns: list[float] = []
    for _, predicted, result in predictions:
        if int(predicted) == 0:
            continue
        numeric = float(result)
        if not isfinite(numeric):
            raise ValueError("Non-finite simulated return encountered during evaluation.")
        trade_returns.append(numeric)

    class_recall: dict[str, float] = {}
    class_support: dict[str, int] = {}
    for class_id in (-1, 0, 1):
        actual_rows = [(a, p) for a, p, _ in predictions if a == class_id]
        support = len(actual_rows)
        class_support[str(class_id)] = support
        class_recall[str(class_id)] = (
            sum(a == p for a, p in actual_rows) / support if support else 0.0
        )

    return {
        "trades": len(trade_returns),
        "predictions": total,
        "accuracy": correct / total,
        "balanced_accuracy": sum(class_recall.values()) / 3.0,
        "avg_return": sum(trade_returns) / len(trade_returns) if trade_returns else 0.0,
        "total_return": sum(trade_returns),
        "class_recall": class_recall,
        "class_support": class_support,
    }
