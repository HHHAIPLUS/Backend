from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

REQUIRED_CONTEXT = ("order_book_imbalance", "funding_rate", "open_interest_change", "news_risk", "news_sentiment", "liquidity_stress")

@dataclass(frozen=True)
class HistoricalContext:
    observed_at: str
    values: Mapping[str, float]
    available: frozenset[str]
    sources: Mapping[str, str]
    def is_complete(self) -> bool: return all(name in self.available for name in REQUIRED_CONTEXT)
    def missing(self) -> tuple[str, ...]: return tuple(name for name in REQUIRED_CONTEXT if name not in self.available)
    def as_features(self) -> dict[str, float]:
        if not self.is_complete(): raise ValueError(f"Incomplete historical context at {self.observed_at}: {', '.join(self.missing())}")
        return {name: float(self.values[name]) for name in REQUIRED_CONTEXT}

def validate_context_timestamp(observed_at: str | int | float) -> datetime:
    if isinstance(observed_at, (int, float)):
        value = float(observed_at)
        if value <= 0: raise ValueError("Historical timestamps must be positive")
        seconds = value / 1000.0 if value >= 10_000_000_000 else value
        return datetime.fromtimestamp(seconds, timezone.utc)
    text = str(observed_at).strip()
    if text.isdigit(): return validate_context_timestamp(int(text))
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None: raise ValueError("Historical context timestamps must include timezone information")
    return dt.astimezone(timezone.utc)

def make_context(observed_at: str | int | float, values: Mapping[str, Any], sources: Mapping[str, str] | None = None) -> HistoricalContext:
    dt = validate_context_timestamp(observed_at); normalized: dict[str, float] = {}; available: set[str] = set(); source_map = dict(sources or {})
    for name in REQUIRED_CONTEXT:
        value = values.get(name)
        if value is None: continue
        try: numeric = float(value)
        except (TypeError, ValueError) as exc: raise ValueError(f"Historical context {name} is not numeric") from exc
        if numeric != numeric or numeric in (float("inf"), float("-inf")): raise ValueError(f"Historical context {name} is not finite")
        normalized[name] = numeric; available.add(name)
    return HistoricalContext(dt.isoformat(), normalized, frozenset(available), source_map)

def merge_context(*contexts: HistoricalContext) -> HistoricalContext:
    if not contexts: raise ValueError("At least one context record is required")
    timestamps = {validate_context_timestamp(c.observed_at) for c in contexts}
    if len(timestamps) != 1: raise ValueError("Cannot merge historical context from different timestamps")
    values: dict[str, float] = {}; available: set[str] = set(); sources: dict[str, str] = {}
    for context in contexts:
        for name in context.available:
            if name in available and values[name] != context.values[name]: raise ValueError(f"Conflicting historical values for {name}")
            values[name] = float(context.values[name]); available.add(name)
            if name in context.sources: sources[name] = context.sources[name]
    return HistoricalContext(contexts[0].observed_at, values, frozenset(available), sources)
