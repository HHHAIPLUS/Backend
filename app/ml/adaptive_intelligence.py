from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from math import exp, isfinite, log
from statistics import mean, pstdev
from typing import Any, Iterable
from uuid import uuid4

@dataclass(frozen=True)
class AdaptiveObservation:
    symbol: str
    model_version: str
    action: str
    confidence: float
    realized_return: float
    observed_at: str
    regime: str = "unknown"
    horizon: int = 6
    expected_probability: float | None = None
    features: dict[str, float] | None = None

@dataclass
class AdaptiveReport:
    samples: int
    regime_reliability: dict[str, dict[str, float]]
    model_reliability: dict[str, dict[str, float]]
    calibration: dict[str, float]
    drift: dict[str, Any]
    familiarity: dict[str, Any]
    learned_weights: dict[str, float]
    abstention_threshold: float
    generated_at: str

@dataclass
class AdaptiveCandidate:
    candidate_id: str
    champion_version: str
    challenger_version: str
    status: str
    reason: str
    created_at: str
    evidence: dict[str, Any]

class AdaptiveIntelligence:
    """Bounded, evidence-driven adaptation layer.

    It learns reliability from completed observations, detects drift/unfamiliar
    states and proposes model/threshold changes. It never mutates a production
    model artifact; promotion remains an explicit validation operation.
    """
    MIN_RELIABILITY_SAMPLES = 30
    MIN_WEIGHT_SAMPLES = 50
    DEFAULT_ABSTENTION = 0.60

    def __init__(self, max_observations: int = 10000):
        self.observations: list[AdaptiveObservation] = []
        self.candidates: list[AdaptiveCandidate] = []
        self.reference: dict[str, tuple[float, float]] = {}
        self.max_observations = max_observations

    def add_observation(self, observation: AdaptiveObservation) -> None:
        if not isfinite(observation.realized_return): raise ValueError("realized_return must be finite")
        if not 0 <= observation.confidence <= 1: raise ValueError("confidence must be between 0 and 1")
        self.observations.append(observation)
        if len(self.observations) > self.max_observations: del self.observations[: len(self.observations) - self.max_observations]
        self._refresh_reference()

    def ingest_records(self, records: Iterable[Any]) -> int:
        added = 0
        for record in records:
            outcome = getattr(record, "outcome_return", None)
            if outcome is None: continue
            features = getattr(record, "features", {}) or {}
            regime = str(features.get("market_regime", features.get("regime", "unknown")))
            horizon = int(features.get("horizon", 6) or 6)
            expected = features.get("expected_probability")
            self.add_observation(AdaptiveObservation(symbol=str(getattr(record, "symbol", "unknown")), model_version=str(getattr(record, "model_version", "unknown")), action=str(getattr(record, "action", "NO_TRADE")).upper(), confidence=float(getattr(record, "confidence", 0.0)), realized_return=float(outcome), observed_at=str(getattr(record, "created_at", datetime.now(timezone.utc).isoformat())), regime=regime, horizon=horizon, expected_probability=float(expected) if expected is not None else None, features={k: float(v) for k, v in features.items() if isinstance(v, (int, float)) and isfinite(float(v))}))
            added += 1
        return added

    def _refresh_reference(self) -> None:
        values: dict[str, list[float]] = {}
        for obs in self.observations:
            for key, value in (obs.features or {}).items():
                if isfinite(value): values.setdefault(key, []).append(value)
        self.reference = {k: (mean(v), pstdev(v) or 1e-9) for k, v in values.items() if len(v) >= 10}

    @staticmethod
    def _reliability(rows: list[AdaptiveObservation]) -> dict[str, float]:
        if not rows: return {"samples": 0.0, "win_rate": 0.0, "avg_return": 0.0, "confidence_reliability": 0.0}
        wins = sum(r.realized_return > 0 for r in rows)
        win_rate = (wins + 1) / (len(rows) + 2)
        confidence_reliability = mean((r.confidence if r.realized_return > 0 else 1 - r.confidence) for r in rows)
        return {"samples": float(len(rows)), "win_rate": float(win_rate), "avg_return": float(mean(r.realized_return for r in rows)), "confidence_reliability": float(confidence_reliability)}

    def regime_performance(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[AdaptiveObservation]] = {}
        for row in self.observations: groups.setdefault(row.regime, []).append(row)
        return {key: self._reliability(rows) for key, rows in groups.items()}

    def model_performance(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[AdaptiveObservation]] = {}
        for row in self.observations: groups.setdefault(row.model_version, []).append(row)
        return {key: self._reliability(rows) for key, rows in groups.items()}

    def calibration_drift(self, window: int = 200) -> dict[str, float]:
        rows = [r for r in self.observations if r.expected_probability is not None][-window:]
        if not rows: return {"samples": 0.0, "brier": 0.0, "ece": 0.0, "drift": 0.0}
        brier = mean((r.expected_probability - (1.0 if r.realized_return > 0 else 0.0)) ** 2 for r in rows)
        bins: dict[int, list[AdaptiveObservation]] = {}
        for r in rows: bins.setdefault(min(9, int(r.expected_probability * 10)), []).append(r)
        ece = sum(len(group) / len(rows) * abs(mean(r.expected_probability for r in group) - mean(r.realized_return > 0 for r in group)) for group in bins.values())
        baseline = self.observations[:-len(rows)] if len(self.observations) > len(rows) else []
        base_rows = [r for r in baseline if r.expected_probability is not None][-window:]
        base_brier = mean((r.expected_probability - (1.0 if r.realized_return > 0 else 0.0)) ** 2 for r in base_rows) if base_rows else brier
        return {"samples": float(len(rows)), "brier": float(brier), "ece": float(ece), "drift": float(brier - base_brier)}

    @staticmethod
    def _psi(reference: list[float], current: list[float], bins: int = 10) -> float:
        if len(reference) < 20 or len(current) < 20: return 0.0
        ref_mean, ref_sd = mean(reference), pstdev(reference)
        if ref_sd < 1e-12: return 10.0 if abs(mean(current) - ref_mean) > 1e-9 else 0.0
        quantiles = sorted(reference)
        edges = sorted(set(quantiles[int(i * (len(quantiles) - 1) / bins)] for i in range(bins + 1)))
        if len(edges) < 2: return 10.0 if abs(mean(current) - ref_mean) > 1e-9 else 0.0
        def counts(values):
            out = [0] * (len(edges) - 1)
            for value in values:
                index = next((i for i in range(len(edges) - 1) if edges[i] <= value < edges[i + 1]), len(out) - 1)
                out[index] += 1
            return [max(c / len(values), 1e-6) for c in out]
        a, b = counts(reference), counts(current)
        return float(sum((x - y) * log(x / y) for x, y in zip(a, b)))

    def drift_report(self, window: int = 200) -> dict[str, Any]:
        recent = self.observations[-window:]
        prior = self.observations[:-window] if len(self.observations) > window else []
        feature_drift: dict[str, float] = {}
        mean_shifts: dict[str, float] = {}
        for key in set(self.reference):
            a = [o.features[key] for o in prior if o.features and key in o.features][-max(window * 2, 200):]
            b = [o.features[key] for o in recent if o.features and key in o.features]
            if a and b:
                feature_drift[key] = self._psi(a, b)
                mean_shifts[key] = abs(mean(b) - mean(a)) / max(pstdev(a), 1e-9)
        recent_win = mean(o.realized_return > 0 for o in recent) if recent else 0.0
        prior_win = mean(o.realized_return > 0 for o in prior) if prior else recent_win
        return {"feature_psi": feature_drift, "feature_mean_shift_z": mean_shifts, "max_psi": max(feature_drift.values(), default=0.0), "max_mean_shift_z": max(mean_shifts.values(), default=0.0), "recent_win_rate": recent_win, "prior_win_rate": prior_win, "concept_drift": abs(recent_win - prior_win) >= 0.15 and len(recent) >= 50}

    def unfamiliar_state(self, features: dict[str, float], threshold: float = 3.0) -> dict[str, Any]:
        zscores: dict[str, float] = {}
        for key, value in features.items():
            if key not in self.reference or not isfinite(float(value)): continue
            mu, sd = self.reference[key]
            zscores[key] = abs((float(value) - mu) / sd)
        max_z = max(zscores.values(), default=0.0)
        return {"unfamiliar": max_z >= threshold, "max_zscore": max_z, "outlier_features": [k for k, z in zscores.items() if z >= threshold], "reference_features": len(self.reference)}

    def learned_model_weights(self) -> dict[str, float]:
        perf = self.model_performance()
        eligible = {k: v for k, v in perf.items() if v["samples"] >= self.MIN_WEIGHT_SAMPLES}
        if not eligible: return {k: 1.0 / max(1, len(perf)) for k in perf}
        scores = {k: max(-1.0, min(1.0, v["avg_return"] * 100)) + 0.5 * (v["confidence_reliability"] - 0.5) for k, v in eligible.items()}
        logits = {k: exp(3.0 * s) for k, s in scores.items()}
        total = sum(logits.values())
        return {k: logits[k] / total for k in logits}

    def learned_abstention_threshold(self) -> float:
        rows = [r for r in self.observations if r.action != "NO_TRADE"]
        if len(rows) < self.MIN_RELIABILITY_SAMPLES: return self.DEFAULT_ABSTENTION
        best, best_score = self.DEFAULT_ABSTENTION, -float("inf")
        for threshold in [0.50 + i * 0.02 for i in range(21)]:
            selected = [r for r in rows if r.confidence >= threshold]
            if len(selected) < self.MIN_RELIABILITY_SAMPLES: continue
            losses = [abs(r.realized_return) for r in selected if r.realized_return < 0]
            score = mean(r.realized_return for r in selected) - 0.25 * mean(losses) if losses else mean(r.realized_return for r in selected)
            if score > best_score: best, best_score = threshold, score
        return round(best, 2)

    def adapt_confidence(self, confidence: float, *, model_version: str, regime: str, features: dict[str, float] | None = None) -> dict[str, Any]:
        confidence = max(0.0, min(1.0, float(confidence)))
        model_rows = [r for r in self.observations if r.model_version == model_version and r.regime == regime]
        model_rows = model_rows if len(model_rows) >= self.MIN_RELIABILITY_SAMPLES else [r for r in self.observations if r.model_version == model_version]
        reliability = self._reliability(model_rows) if model_rows else {"confidence_reliability": 0.5, "samples": 0}
        unfamiliar = self.unfamiliar_state(features or {}) if features else {"unfamiliar": False, "max_zscore": 0.0}
        adjusted = confidence * (0.5 + 0.5 * reliability.get("confidence_reliability", 0.5))
        if unfamiliar.get("unfamiliar"): adjusted *= 0.75
        return {"raw_confidence": confidence, "adjusted_confidence": max(0.0, min(1.0, adjusted)), "reliability_samples": reliability.get("samples", 0), "unfamiliar": unfamiliar}

    def create_candidate(self, champion_version: str, challenger_version: str, reason: str, evidence: dict[str, Any]) -> AdaptiveCandidate:
        candidate = AdaptiveCandidate(str(uuid4()), champion_version, challenger_version, "quarantined", reason, datetime.now(timezone.utc).isoformat(), evidence)
        self.candidates.append(candidate)
        return candidate

    def evaluate_challenger(self, candidate_id: str, champion_returns: list[float], challenger_returns: list[float], champion_predictions: list[int] | None = None, challenger_predictions: list[int] | None = None, regimes: list[str] | None = None) -> AdaptiveCandidate:
        candidate = next((c for c in self.candidates if c.candidate_id == candidate_id), None)
        if candidate is None: raise KeyError("Adaptive candidate not found")
        if len(champion_returns) != len(challenger_returns) or len(champion_returns) < 100: raise ValueError("Champion/challenger comparison requires matched samples >= 100")
        import numpy as np
        from app.ml.model_validation import paired_bootstrap_ci
        a, b = np.asarray(champion_returns, dtype=float), np.asarray(challenger_returns, dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError("Champion/challenger returns must be finite")
        ci = paired_bootstrap_ci(b, a)
        lo = float(ci["ci_low"]); hi = float(ci["ci_high"])
        stable = True; regime_results: dict[str, float] = {}
        if regimes and len(regimes) == len(a):
            for regime in sorted(set(regimes)):
                vals = (b - a)[np.asarray(regimes) == regime]
                if len(vals) >= 30:
                    regime_results[regime] = float(np.mean(vals)); stable &= regime_results[regime] > -0.0005
        prediction_ok = (champion_predictions is None and challenger_predictions is None) or (len(champion_predictions) == len(challenger_predictions) == len(a))
        promoted = bool(ci.get("valid") and lo > 0 and float(np.mean(b - a)) > 0 and stable and prediction_ok)
        candidate.status = "promotion_eligible" if promoted else "rejected"
        candidate.evidence = {**candidate.evidence, "mean_delta": float(np.mean(b - a)), "bootstrap_ci": [lo, hi], "regime_delta": regime_results, "stable": stable, "promoted": promoted}
        return candidate

    def report(self) -> AdaptiveReport:
        return AdaptiveReport(len(self.observations), self.regime_performance(), self.model_performance(), self.calibration_drift(), self.drift_report(), self.unfamiliar_state({}), self.learned_model_weights(), self.learned_abstention_threshold(), datetime.now(timezone.utc).isoformat())

    def status(self) -> dict[str, Any]:
        report = self.report()
        return {"engine": "stage4-adaptive-intelligence-v1", "observations": report.samples, "models_tracked": len(report.model_reliability), "regimes_tracked": len(report.regime_reliability), "candidate_count": len(self.candidates), "quarantined_candidates": sum(c.status == "quarantined" for c in self.candidates), "learned_weights": report.learned_weights, "abstention_threshold": report.abstention_threshold, "production_self_modification": False, "execution_authority": False}

    def snapshot(self) -> dict[str, Any]:
        return {"status": self.status(), "report": asdict(self.report()), "candidates": [asdict(c) for c in self.candidates[-50:]]}

adaptive_intelligence = AdaptiveIntelligence()
