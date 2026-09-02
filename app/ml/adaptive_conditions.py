from __future__ import annotations
from statistics import mean
from typing import Any
from app.ml.adaptive_intelligence import AdaptiveIntelligence

def conditioned_performance(engine: AdaptiveIntelligence) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Any]] = {}
    for row in engine.observations: groups.setdefault(f"{row.model_version}|{row.regime}|{row.horizon}", []).append(row)
    result: dict[str, dict[str, float]] = {}
    for key, rows in groups.items():
        wins = sum(r.realized_return > 0 for r in rows)
        result[key] = {"samples": float(len(rows)), "win_rate": float((wins + 1) / (len(rows) + 2)), "avg_return": float(mean(r.realized_return for r in rows)), "confidence_reliability": float(mean(r.confidence if r.realized_return > 0 else 1 - r.confidence for r in rows))}
    return result

def adapt_confidence(engine: AdaptiveIntelligence, confidence: float, *, model_version: str, regime: str, horizon: int, features: dict[str, float] | None = None) -> dict[str, Any]:
    exact = conditioned_performance(engine).get(f"{model_version}|{regime}|{horizon}")
    if exact and exact["samples"] >= engine.MIN_RELIABILITY_SAMPLES:
        raw = max(0.0, min(1.0, float(confidence))); unfamiliar = engine.unfamiliar_state(features or {}) if features else {"unfamiliar": False, "max_zscore": 0.0}; adjusted = raw * (0.5 + 0.5 * exact["confidence_reliability"])
        if unfamiliar.get("unfamiliar"): adjusted *= 0.75
        return {"raw_confidence": raw, "adjusted_confidence": max(0.0, min(1.0, adjusted)), "reliability_samples": exact["samples"], "conditioned_samples": exact["samples"], "conditioned": True, "horizon": horizon, "unfamiliar": unfamiliar}
    fallback = engine.adapt_confidence(confidence, model_version=model_version, regime=regime, features=features)
    return {**fallback, "horizon": horizon, "conditioned_samples": 0 if not exact else exact["samples"], "conditioned": False}

def learned_prediction_weights(engine: AdaptiveIntelligence, model_predictions: dict[str, dict[str, float]], regime: str, horizon: int) -> dict[str, float]:
    """Evidence-learned fusion weights; caller must still pass promotion gates."""
    perf = conditioned_performance(engine); scores: dict[str, float] = {}
    for model in model_predictions:
        evidence = perf.get(f"{model}|{regime}|{horizon}")
        if evidence and evidence["samples"] >= engine.MIN_WEIGHT_SAMPLES:
            scores[model] = max(-1.0, min(1.0, evidence["avg_return"] * 100.0)) + 0.5 * (evidence["confidence_reliability"] - 0.5)
    if not scores: return {}
    import math
    logits = {k: math.exp(3.0 * v) for k, v in scores.items()}; total = sum(logits.values())
    return {k: logits[k] / total for k in logits}
