"""Orchestration for point-in-time historical enrichment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.ml.historical_context import REQUIRED_CONTEXT, HistoricalContext
from app.ml.historical_dataset import assemble_point_in_time_examples, HistoricalExample

@dataclass(frozen=True)
class EnrichmentReport:
    total_observations: int
    accepted_observations: int
    deferred_observations: int
    coverage: dict[str, float]
    missing_reasons: dict[str, int]
    production_ready: bool


def enrich_dataset(candles: list[Any], contexts: Mapping[str, HistoricalContext], *, horizon: int = 6, threshold: float = 0.0025, minimum_coverage: float = 0.95) -> tuple[list[HistoricalExample], EnrichmentReport]:
    examples, deferred = assemble_point_in_time_examples(candles, contexts, horizon=horizon, threshold=threshold)
    total = len(examples) + len(deferred)
    accepted_ratio = len(examples) / total if total else 0.0
    coverage = {name: accepted_ratio for name in REQUIRED_CONTEXT}
    reasons: dict[str, int] = {}
    for row in deferred:
        reason = str(row.get("reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    production_ready = bool(total and accepted_ratio >= minimum_coverage and all(v >= minimum_coverage for v in coverage.values()))
    return examples, EnrichmentReport(total, len(examples), len(deferred), coverage, reasons, production_ready)
