from __future__ import annotations

from statistics import mean
from typing import Any

from app.ml.adaptive_intelligence import AdaptiveIntelligence


def conditioned_performance(engine: AdaptiveIntelligence) -> dict[str, dict[str, float]]:
    """Learn model reliability at the model/regime/horizon level."""
    groups: dict[str, list[Any]] = {}
    for row in engine.observations:
        key = f"{row.model_version}|{row.regime}|{row.horizon}"
        groups.setdefault(key, []).append(row)
    result: dict[str, dict[str, float]] = {}
    for key, rows in groups.items():
        wins = sum(r.realized_return > 0 for r in rows)
        result[key] = {
            "samples": float(len(rows)),
            "win_rate": float((wins + 1) / (len(rows) + 2)),
            "avg_return": float(mean(r.realized_return for r in rows)),
            "confidence_reliability": float(mean(r.confidence if r.realized_return > 0 else 1 - r.confidence for r in rows)),
        }
    return result


def adapt_confidence(engine: AdaptiveIntelligence, confidence: float, *, model_version: str, regime: str, horizon: int, features: dict[str, float] | None = None) -> dict[str, Any]:
    conditions = conditioned_performance(engine)
    exact = conditions.get(f"{model_version}|{regime}|{horizon}")
    if exact and exact["samples"] >= engine.MIN_RELIABILITY_SAMPLES:
        reliability = exact
    else:
        fallback = engine.adapt_confidence(confidence, model_version=model_version, regime=regime, features=features)
        return {**fallback, "horizon": horizon, "conditioned_samples": 0 if not exact else exact["samples"], "conditioned": bool(exact and exact["samples"] >= engine.MIN_RELIABILITY_SAMPLES)}
    raw = max(0.0, min(1.0, float(confidence)))
    unfamiliar = engine.unfamiliar_state(features or {}) if features else {"unfamiliar": False, "max_zscore": 0.0}
    adjusted = raw * (0.5 + 0.5 * reliability["confidence_reliability"])
    if unfamiliar.get("unfamiliar"): adjusted *= 0.75
    return {"raw_confidence": raw, "adjusted_confidence": max(0.0, min(1.0, adjusted)), "reliability_samples": reliability["samples"], "conditioned_samples": reliability["samples"], "conditioned": True, "horizon": horizon, "unfamiliar": unfamiliar}
