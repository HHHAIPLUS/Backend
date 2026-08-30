from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fold:
    train: list[dict]
    test: list[dict]


def walk_forward(
    rows: list[dict],
    min_train: int = 300,
    test_size: int = 100,
    step: int = 100,
) -> list[Fold]:
    """
    Create chronological walk-forward validation folds.

    No future rows are allowed into a training set.
    """

    if min_train <= 0:
        raise ValueError("min_train must be greater than zero.")

    if test_size <= 0:
        raise ValueError("test_size must be greater than zero.")

    if step <= 0:
        raise ValueError("step must be greater than zero.")

    ordered = sorted(
        rows,
        key=lambda r: r["observed_at"],
    )

    folds: list[Fold] = []

    end = min_train

    while end + test_size <= len(ordered):
        folds.append(
            Fold(
                train=ordered[:end],
                test=ordered[end:end + test_size],
            )
        )

        end += step

    return folds


def evaluate_predictions(
    predictions: list[tuple[int, int, float]],
) -> dict:
    """
    Evaluate three-class directional predictions.

    Classes:
        -1 = short
         0 = flat
         1 = long

    Balanced accuracy is calculated across ALL three classes.
    If a class has no actual samples, its recall is reported as
    zero rather than silently excluding that class.
    """

    if not predictions:
        return {
            "trades": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "avg_return": 0.0,
            "class_recall": {
                "-1": 0.0,
                "0": 0.0,
                "1": 0.0,
            },
            "class_support": {
                "-1": 0,
                "0": 0,
                "1": 0,
            },
        }

    total = len(predictions)

    correct = sum(
        1
        for actual, predicted, _ in predictions
        if actual == predicted
    )

    returns = [
        float(result)
        for _, _, result in predictions
    ]

    classes = (-1, 0, 1)

    class_recall: dict[str, float] = {}
    class_support: dict[str, int] = {}

    for class_id in classes:
        actual_rows = [
            (actual, predicted)
            for actual, predicted, _
            in predictions
            if actual == class_id
        ]

        support = len(actual_rows)

        class_support[str(class_id)] = support

        if support == 0:
            class_recall[str(class_id)] = 0.0
            continue

        class_correct = sum(
            1
            for actual, predicted
            in actual_rows
            if actual == predicted
        )

        class_recall[str(class_id)] = (
            class_correct / support
        )

    balanced_accuracy = sum(
        class_recall.values()
    ) / len(classes)

    return {
        "trades": total,
        "accuracy": correct / total,
        "balanced_accuracy": balanced_accuracy,
        "avg_return": sum(returns) / total,
        "class_recall": class_recall,
        "class_support": class_support,
    }