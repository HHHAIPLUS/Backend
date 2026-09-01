"""HHHAI Stage 1A: end-to-end training-data trust gate.

Stage 1A has one job: prevent an untrustworthy historical dataset from
becoming a model that can ever be promoted. This module combines source
capability, point-in-time context, dataset assembly, integrity auditing and
walk-forward validation into one decision object.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

from app.ml.historical_context import REQUIRED_CONTEXT, HistoricalContext, validate_context_timestamp
from app.ml.historical_dataset import HistoricalExample, assemble_point_in_time_examples
from app.ml.dataset_integrity import audit_training_rows, DatasetIntegrityError
from app.ml.validation import walk_forward, evaluate_predictions
from app.ml.predictive import FEATURES
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Stage1AReport:
    status: str
    rows_requested: int
    rows_accepted: int
    rows_deferred: int
    coverage: dict[str, float]
    missing_context: dict[str, int]
    validation: dict[str, Any]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coverage(examples: Sequence[HistoricalExample], requested: int) -> dict[str, float]:
    return {name: (len(examples) / requested if requested else 0.0) for name in REQUIRED_CONTEXT}


def _examples_as_rows(examples: Sequence[HistoricalExample]) -> list[dict[str, Any]]:
    return [
        {
            "observed_at": e.observed_at,
            "features": e.features,
            "label": e.label,
            "outcome_return": e.outcome_return,
            "context_available": {name: True for name in REQUIRED_CONTEXT},
            "provenance": e.provenance,
        }
        for e in examples
    ]


def run_stage1a(
    candles: Sequence[Any],
    contexts: Mapping[str, HistoricalContext],
    *,
    horizon: int = 6,
    threshold: float = 0.0025,
    min_context_coverage: float = 0.98,
    min_train: int = 300,
    test_size: int = 100,
    step: int = 100,
) -> Stage1AReport:
    """Run every Stage 1A gate and return one auditable result.

    This function never fills unavailable historical context. It also never
    promotes a model merely because accuracy is high: chronology, coverage,
    class diversity and out-of-sample return are all required.
    """
    blockers: list[str] = []
    missing = {name: 0 for name in REQUIRED_CONTEXT}
    try:
        examples, deferred = assemble_point_in_time_examples(candles, contexts, horizon=horizon, threshold=threshold)
    except Exception as exc:
        return Stage1AReport("REJECTED", len(candles), 0, len(candles), {}, missing, {}, (f"dataset assembly failed: {exc}",))

    requested = max(0, len(candles) - 24 - horizon)
    for row in deferred:
        for name in row.get("missing_context", []):
            if name in missing:
                missing[name] += 1
    coverage = _coverage(examples, requested)
    for name, ratio in coverage.items():
        if ratio < min_context_coverage:
            blockers.append(f"{name} coverage {ratio:.2%} < {min_context_coverage:.2%}")

    rows = _examples_as_rows(examples)
    if not rows:
        blockers.append("no complete point-in-time training examples")
        return Stage1AReport("REJECTED", requested, 0, len(deferred), coverage, missing, {}, tuple(blockers))

    try:
        audit = audit_training_rows(rows, required_context_features=REQUIRED_CONTEXT)
    except DatasetIntegrityError as exc:
        blockers.append(f"integrity audit failed: {exc}")
        return Stage1AReport("REJECTED", requested, len(rows), len(deferred), coverage, missing, {"audit": str(exc)}, tuple(blockers))

    folds = walk_forward(rows, min_train=min_train, test_size=test_size, step=step)
    if not folds:
        blockers.append("insufficient rows for walk-forward validation")
        return Stage1AReport("REJECTED", requested, len(rows), len(deferred), coverage, missing, {"audit": audit}, tuple(blockers))

    predictions: list[tuple[int, int, float]] = []
    fold_count = 0
    for fold in folds:
        train_y = [int(r["label"]) for r in fold.train]
        if len(set(train_y)) < 3:
            continue
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])
        X = [[float(r["features"].get(k, 0.0)) for k in FEATURES] for r in fold.train]
        Xt = [[float(r["features"].get(k, 0.0)) for k in FEATURES] for r in fold.test]
        model.fit(X, train_y)
        pred = model.predict(Xt)
        for row, prediction in zip(fold.test, pred):
            predictions.append((int(row["label"]), int(prediction), float(row["outcome_return"])))
        fold_count += 1

    metrics = evaluate_predictions(predictions)
    validation = {"audit": audit, "folds": fold_count, "predictions": len(predictions), "metrics": metrics}
    if fold_count == 0:
        blockers.append("no valid three-class training folds")
    if metrics.get("balanced_accuracy", 0.0) < 0.50:
        blockers.append("balanced accuracy below 0.50")
    if metrics.get("average_return", 0.0) <= 0.0:
        blockers.append("average simulated return is not positive")
    if blockers:
        return Stage1AReport("REJECTED", requested, len(rows), len(deferred), coverage, missing, validation, tuple(blockers))
    return Stage1AReport("READY_FOR_STAGE_1B", requested, len(rows), len(deferred), coverage, missing, validation, ())
