"""Truthfulness checks for supervised trading datasets and OHLCV history.

The audit is deliberately conservative: it rejects malformed candles, duplicate
or non-chronological observations, interval gaps, non-finite model features,
future-looking targets inside features, and any model feature whose historical
value lacks provenance. It never turns unavailable historical context into a
fake neutral zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Iterable, Sequence

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
class CandleAudit:
    rows: int
    ordered: bool
    unique_timestamps: bool
    interval_ms: int | None
    interval_consistent: bool
    duplicate_timestamps: int
    gaps: int
    malformed_rows: int
    invalid_ohlc: int
    invalid_volume: int
    future_or_open_candles: int
    finite: bool
    production_ready: bool
    reason: str = ""

@dataclass(frozen=True)
class DatasetAudit:
    rows: int
    ordered: bool
    unique_timestamps: bool
    finite_features: bool
    complete_context_rows: int
    incomplete_context_rows: int
    leakage_suspected: bool
    candle_metadata_valid: bool = True
    production_ready: bool = False
    reason: str = ""

def _timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError("timestamp must be positive")
        seconds = float(value) / 1000.0 if float(value) >= 10_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds, timezone.utc)
    if not isinstance(value, str):
        raise ValueError("observed_at must be an ISO-8601 string")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("observed_at must include timezone information")
    return dt.astimezone(timezone.utc)

def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def _candle_values(row: Any) -> tuple[int, float, float, float, float, float] | None:
    if isinstance(row, dict):
        keys = ("timestamp", "open", "high", "low", "close", "volume")
        if not all(key in row for key in keys):
            return None
        raw = [row[key] for key in keys]
    elif isinstance(row, (list, tuple)) and len(row) >= 6:
        raw = list(row[:6])
    else:
        return None
    try:
        return int(float(raw[0])), *(float(x) for x in raw[1:6])
    except (TypeError, ValueError):
        return None

def audit_klines(
    rows: Sequence[Any],
    *,
    interval_ms: int,
    now_ms: int | None = None,
    allow_current_open_candle: bool = False,
) -> CandleAudit:
    """Audit raw candles before they can enter supervised training.

    The last/current candle is rejected by default because its OHLC values can
    still change. This keeps the training set composed of closed observations.
    """
    rows = list(rows)
    now_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    timestamps: list[int] = []
    malformed = invalid_ohlc = invalid_volume = future_open = 0
    finite = True

    for row in rows:
        parsed = _candle_values(row)
        if parsed is None:
            malformed += 1
            finite = False
            continue
        ts, op, hi, lo, cl, vol = parsed
        timestamps.append(ts)
        if not all(_finite(x) for x in parsed):
            finite = False
        if min(op, hi, lo, cl) <= 0 or hi < max(op, cl) or lo > min(op, cl) or hi < lo:
            invalid_ohlc += 1
        if vol < 0:
            invalid_volume += 1
        # A candle is considered closed only after the next interval begins.
        if ts + interval_ms > now_ms:
            future_open += 1

    ordered = all(a < b for a, b in zip(timestamps, timestamps[1:]))
    unique = len(timestamps) == len(set(timestamps))
    duplicate_count = len(timestamps) - len(set(timestamps))
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    consistent = bool(deltas) and all(delta == interval_ms for delta in deltas)
    gaps = sum(1 for delta in deltas if delta != interval_ms)

    ready = bool(
        timestamps
        and malformed == 0
        and invalid_ohlc == 0
        and invalid_volume == 0
        and finite
        and ordered
        and unique
        and consistent
        and (allow_current_open_candle or future_open == 0)
    )
    reason = "" if ready else (
        f"malformed={malformed}, invalid_ohlc={invalid_ohlc}, invalid_volume={invalid_volume}, "
        f"ordered={ordered}, unique={unique}, interval_consistent={consistent}, gaps={gaps}, "
        f"future_or_open={future_open}, finite={finite}"
    )
    return CandleAudit(
        rows=len(rows), ordered=ordered, unique_timestamps=unique,
        interval_ms=interval_ms, interval_consistent=consistent,
        duplicate_timestamps=duplicate_count, gaps=gaps,
        malformed_rows=malformed, invalid_ohlc=invalid_ohlc,
        invalid_volume=invalid_volume, future_or_open_candles=future_open,
        finite=finite, production_ready=ready, reason=reason,
    )

def audit_dataset(rows: Iterable[dict[str, Any]]) -> DatasetAudit:
    rows = list(rows)
    timestamps: list[datetime] = []
    finite_features = True
    complete = 0
    incomplete = 0
    candle_metadata_valid = True

    required_context = set(FEATURES).intersection(CONTEXT_FEATURES)

    for row in rows:
        try:
            timestamps.append(_timestamp(row.get("observed_at")))
        except (TypeError, ValueError):
            finite_features = False
            continue

        features = row.get("features")
        if not isinstance(features, dict):
            finite_features = False
            incomplete += 1
            continue

        if any(key not in features or not _finite(features[key]) for key in FEATURES):
            finite_features = False

        provenance = row.get("feature_provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}

        # Only features actually used by the predictive model require historical
        # provenance. Context features outside FEATURES may legitimately remain
        # live-only and must not block an OHLCV-only training dataset.
        missing_model_context = [key for key in required_context if not provenance.get(key)]
        if missing_model_context:
            incomplete += 1
        else:
            complete += 1

        candle = row.get("candle")
        if candle is not None and _candle_values(candle) is None:
            candle_metadata_valid = False

    ordered = all(a < b for a, b in zip(timestamps, timestamps[1:]))
    unique = len(timestamps) == len(set(timestamps))
    leakage_suspected = any(
        isinstance(row.get("features"), dict)
        and any(key in row["features"] for key in ("label", "outcome_return", "future_return", "target"))
        for row in rows
    )

    ready = bool(
        rows and ordered and unique and finite_features and
        candle_metadata_valid and incomplete == 0 and not leakage_suspected
    )
    return DatasetAudit(
        rows=len(rows), ordered=ordered, unique_timestamps=unique,
        finite_features=finite_features, complete_context_rows=complete,
        incomplete_context_rows=incomplete, leakage_suspected=leakage_suspected,
        candle_metadata_valid=candle_metadata_valid, production_ready=ready,
        reason="" if ready else "dataset integrity requirements were not satisfied",
    )

def require_production_ready(rows: Iterable[dict[str, Any]]) -> DatasetAudit:
    audit = audit_dataset(rows)
    if not audit.production_ready:
        raise ValueError(
            "Historical dataset is not production-ready: "
            f"ordered={audit.ordered}, unique_timestamps={audit.unique_timestamps}, "
            f"finite_features={audit.finite_features}, "
            f"complete_context_rows={audit.complete_context_rows}/{audit.rows}, "
            f"candle_metadata_valid={audit.candle_metadata_valid}, "
            f"leakage_suspected={audit.leakage_suspected}"
        )
    return audit
