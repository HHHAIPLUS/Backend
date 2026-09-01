"""Point-in-time historical dataset assembly for HHHAI Stage 1A.

This module provides the safe assembly boundary. It does not fabricate
historical context. A training example is created only when all required
context fields have been observed for the same timestamp; otherwise the
example is rejected/deferred for enrichment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from app.ml.historical_context import REQUIRED_CONTEXT, HistoricalContext, merge_context, validate_context_timestamp
from app.ml.features import build_model_features


@dataclass(frozen=True)
class HistoricalExample:
    observed_at: str
    features: dict[str, float]
    label: int
    outcome_return: float
    provenance: dict[str, str]


def _candle_to_mapping(row: Any) -> dict[str, float | str]:
    if isinstance(row, Mapping):
        observed_at = row.get("observed_at")
        if observed_at is None:
            observed_at = row.get("timestamp")
        return {
            "observed_at": str(observed_at),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
    if len(row) < 6:
        raise ValueError("Candle row must contain timestamp, OHLC and volume")
    return {
        "observed_at": str(row[0]),
        "open": float(row[1]), "high": float(row[2]),
        "low": float(row[3]), "close": float(row[4]),
        "volume": float(row[5]),
    }


def assemble_point_in_time_examples(
    candles: Iterable[Any],
    contexts: Mapping[str, HistoricalContext],
    horizon: int = 6,
    threshold: float = 0.0025,
) -> tuple[list[HistoricalExample], list[dict[str, Any]]]:
    """Build only examples whose context is complete and point-in-time safe.

    Returns (accepted_examples, deferred_rows). Deferred rows are not silently
    converted to zeros; they contain the timestamp and missing context fields
    needed by a later enrichment job.
    """
    candle_rows = [_candle_to_mapping(row) for row in candles]
    candle_rows.sort(key=lambda row: validate_context_timestamp(str(row["observed_at"])))
    if horizon <= 0 or threshold <= 0:
        raise ValueError("horizon and threshold must be positive")

    accepted: list[HistoricalExample] = []
    deferred: list[dict[str, Any]] = []
    lookback = 24
    for i in range(lookback, len(candle_rows) - horizon):
        current = candle_rows[i]
        observed_at = str(current["observed_at"])
        context = contexts.get(observed_at)
        if context is None:
            deferred.append({"observed_at": observed_at, "missing_context": list(REQUIRED_CONTEXT), "reason": "context_not_found"})
            continue
        if validate_context_timestamp(context.observed_at) != validate_context_timestamp(observed_at):
            raise ValueError(f"Context timestamp mismatch at {observed_at}")
        if not context.is_complete():
            deferred.append({"observed_at": observed_at, "missing_context": list(context.missing()), "reason": "incomplete_context"})
            continue

        future_close = float(candle_rows[i + horizon]["close"])
        current_close = float(current["close"])
        future_return = future_close / current_close - 1.0
        label = 1 if future_return > threshold else -1 if future_return < -threshold else 0
        window = candle_rows[i - lookback:i + 1]
        features = build_model_features(window, context=context)
        provenance = {name: context.sources.get(name, "historical_context") for name in REQUIRED_CONTEXT}
        accepted.append(HistoricalExample(observed_at, features, label, future_return, provenance))

    return accepted, deferred
