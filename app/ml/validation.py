from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass
class Fold:
    train: list[dict]
    test: list[dict]


def walk_forward(rows: list[dict], min_train: int = 300, test_size: int = 100, step: int = 100) -> list[Fold]:
    """Create chronological walk-forward folds without future leakage."""
    if min_train <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("min_train, test_size and step must be greater than zero.")
    ordered = sorted(rows, key=lambda r: r["observed_at"])
    folds: list[Fold] = []
    end = min_train
    while end + test_size <= len(ordered):
        folds.append(Fold(train=ordered[:end], test=ordered[end:end + test_size]))
        end += step
    return folds


def evaluate_predictions(predictions: list[tuple[int, int, float]]) -> dict:
    """Evaluate classification plus simulated directional-return metrics.

    Tuple format: (actual_label, predicted_label, realized_return).
    A zero realized return represents an abstained/non-directional simulation.
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
    trade_returns = [
        float(result) for _, _, result in predictions
        if isfinite(float(result)) and float(result) != 0.0
    ]

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
