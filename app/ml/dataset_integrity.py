"""Truthfulness checks for supervised trading datasets.

This module deliberately does not manufacture historical order-book, funding,
open-interest, news, or liquidity values.  If those fields were not actually
observed at the example timestamp, the dataset is marked incomplete instead
of pretending that zero means neutral market information.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Iterable

from app.ml.predictive import FEATURES


CONTEXT_FEATURES = {
    "order_book_imbalance",
    "funding_rate",
    "open_interest_change",
    "news_risk",
    "news_sentiment",
    "liquidity_stress",
}


@dataclass(frozen=True)
class DatasetAudit:
    rows: int
    ordered: bool
    unique_timestamps: bool
    finite_features: bool
    complete_context_rows: int
    incomplete_context_rows: int
    leakage_suspected: bool

    @property
    def production_ready(self) -> bool:
        return (
            self.rows > 0
            and self.ordered
            and self.unique_timestamps
            and self.finite_features
            and self.incomplete_context_rows == 0
            and not self.leakage_suspected
        )


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("observed_at must be an ISO-8601 string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def audit_dataset(rows: Iterable[dict[str, Any]]) -> DatasetAudit:
    rows = list(rows)
    timestamps: list[datetime] = []
    finite_features = True
    complete = 0
    incomplete = 0

    for row in rows:
        timestamps.append(_timestamp(row.get("observed_at")))
        features = row.get("features")
        if not isinstance(features, dict):
            finite_features = False
            incomplete += 1
            continue

        if any(
            key not in features or not _finite(features[key])
            for key in FEATURES
        ):
            finite_features = False

        # Presence is the minimum provenance contract. A value of 0.0 is not
        # evidence that the market was neutral; callers must supply a separate
        # provenance map when using context features historically.
        provenance = row.get("feature_provenance", {})
        context_observed = all(
            bool(provenance.get(key)) for key in CONTEXT_FEATURES
        ) if isinstance(provenance, dict) else False

        if context_observed:
            complete += 1
        else:
            incomplete += 1

    ordered = all(a < b for a, b in zip(timestamps, timestamps[1:]))
    unique = len(timestamps) == len(set(timestamps))

    # A simple structural leakage guard: labels/outcomes must not be used as
    # feature names. The actual future-return construction is audited separately
    # because this module intentionally cannot infer exchange candle boundaries.
    leakage_suspected = any(
        isinstance(row.get("features"), dict)
        and any(
            key in row["features"]
            for key in ("label", "outcome_return", "future_return", "target")
        )
        for row in rows
    )

    return DatasetAudit(
        rows=len(rows),
        ordered=ordered,
        unique_timestamps=unique,
        finite_features=finite_features,
        complete_context_rows=complete,
        incomplete_context_rows=incomplete,
        leakage_suspected=leakage_suspected,
    )


def require_production_ready(rows: Iterable[dict[str, Any]]) -> DatasetAudit:
    audit = audit_dataset(rows)
    if not audit.production_ready:
        raise ValueError(
            "Historical dataset is not production-ready: "
            f"ordered={audit.ordered}, unique_timestamps={audit.unique_timestamps}, "
            f"finite_features={audit.finite_features}, "
            f"complete_context_rows={audit.complete_context_rows}/{audit.rows}, "
            f"leakage_suspected={audit.leakage_suspected}"
        )
    return audit
