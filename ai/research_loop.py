from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from statistics import mean, pstdev
from typing import Any, Callable, Iterable
from uuid import uuid4


@dataclass(frozen=True)
class ResearchSnapshot:
    record_id: str
    symbol: str
    action: str
    model_version: str
    regime: str
    horizon: int
    confidence: float
    expected_probability: float | None
    features: dict[str, float]
    thesis: str
    created_at: str
    realized_return: float
    execution_return: float | None = None
    slippage: float = 0.0
    fees: float = 0.0


@dataclass
class ResearchCandidate:
    candidate_id: str
    base_model: str
    hypothesis: str
    status: str
    created_at: str
    lineage: dict[str, Any]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    candidate_id: str
    samples: int
    average_return: float
    win_rate: float
    max_drawdown: float
    net_return: float
    mean_delta_vs_champion: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    walk_forward: dict[str, Any]
    out_of_sample: dict[str, Any]
    stress: dict[str, Any]
    reproducibility_fingerprint: str
    passed: bool


class ResearchLoop:
    """Evidence-only research loop.

    Completed decisions become immutable research snapshots. Candidate ideas are
    quarantined and evaluated by deterministic historical replay, chronological
    walk-forward/OOS splits and robustness checks. Nothing in this class can
    mutate production models or submit exchange orders.
    """

    VERSION = "stage7-research-loop-v1"
    MIN_OOS = 30
    MIN_WALK_FORWARD = 3
    BOOTSTRAPS = 1000

    def __init__(self, max_snapshots: int = 20000):
        self.snapshots: list[ResearchSnapshot] = []
        self.candidates: list[ResearchCandidate] = []
        self.experiments: list[ExperimentResult] = []
        self.max_snapshots = max_snapshots

    @staticmethod
    def _directional_return(snapshot: ResearchSnapshot) -> float:
        action = snapshot.action.upper()
        if action == "SHORT":
            return -snapshot.realized_return
        if action == "NO_TRADE":
            return 0.0
        return snapshot.realized_return

    @staticmethod
    def _equity_metrics(returns: list[float]) -> tuple[float, float, float]:
        if not returns:
            return 0.0, 0.0, 0.0
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for value in returns:
            equity *= 1.0 + value
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
        return mean(returns), sum(returns), max_dd

    @staticmethod
    def _paired_bootstrap(delta: list[float], seed: int = 17) -> tuple[float, float]:
        if len(delta) < 2:
            return 0.0, 0.0
        # Deterministic LCG keeps this dependency-light and reproducible.
        state = seed & 0x7FFFFFFF
        samples: list[float] = []
        n = len(delta)
        for _ in range(1000):
            total = 0.0
            for _ in range(n):
                state = (1103515245 * state + 12345) & 0x7FFFFFFF
                total += delta[state % n]
            samples.append(total / n)
        samples.sort()
        return samples[25], samples[975]

    def add_snapshot(self, snapshot: ResearchSnapshot) -> None:
        if not snapshot.record_id:
            raise ValueError("record_id is required")
        if not 0 <= snapshot.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if snapshot.horizon < 1:
            raise ValueError("horizon must be positive")
        if not all(isinstance(v, (int, float)) for v in snapshot.features.values()):
            raise ValueError("features must be numeric")
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda x: x.created_at)
        if len(self.snapshots) > self.max_snapshots:
            del self.snapshots[: len(self.snapshots) - self.max_snapshots]

    def ingest_decisions(self, decisions: Iterable[dict[str, Any]], outcomes: dict[str, dict[str, Any]]) -> int:
        added = 0
        for row in decisions:
            rid = str(row.get("record_id") or row.get("id") or "")
            outcome = outcomes.get(rid)
            if not rid or not outcome:
                continue
            payload = row.get("payload") or row
            realized = outcome.get("realized_return", outcome.get("payload", {}).get("realized_return"))
            if realized is None:
                continue
            features = payload.get("features") or {}
            self.add_snapshot(ResearchSnapshot(
                record_id=rid,
                symbol=str(payload.get("symbol", row.get("symbol", "unknown"))),
                action=str(payload.get("action", "NO_TRADE")).upper(),
                model_version=str(payload.get("model_version", "unknown")),
                regime=str(features.get("market_regime", features.get("regime", "unknown"))),
                horizon=int(features.get("horizon", 6) or 6),
                confidence=float(payload.get("confidence", 0.0)),
                expected_probability=float(features["expected_probability"]) if features.get("expected_probability") is not None else None,
                features={k: float(v) for k, v in features.items() if isinstance(v, (int, float))},
                thesis=str(payload.get("thesis", "")),
                created_at=str(row.get("created_at", payload.get("created_at", datetime.now(timezone.utc).isoformat()))),
                realized_return=float(realized),
                execution_return=float(outcome.get("execution_return")) if outcome.get("execution_return") is not None else None,
                slippage=float(outcome.get("slippage", 0.0) or 0.0),
                fees=float(outcome.get("fees", 0.0) or 0.0),
            ))
            added += 1
        return added

    def attribute_errors(self) -> dict[str, Any]:
        rows = self.snapshots
        prediction_errors = [r for r in rows if r.expected_probability is not None]
        calibration = mean((r.expected_probability - (1.0 if self._directional_return(r) > 0 else 0.0)) ** 2 for r in prediction_errors) if prediction_errors else 0.0
        regimes: dict[str, list[ResearchSnapshot]] = {}
        models: dict[str, list[ResearchSnapshot]] = {}
        for row in rows:
            regimes.setdefault(row.regime, []).append(row)
            models.setdefault(row.model_version, []).append(row)
        return {
            "snapshots": len(rows),
            "prediction_errors": sum(self._directional_return(r) <= 0 for r in rows if r.action != "NO_TRADE"),
            "calibration_brier": calibration,
            "regime_errors": {k: sum(self._directional_return(r) <= 0 for r in v) for k, v in regimes.items()},
            "model_errors": {k: sum(self._directional_return(r) <= 0 for r in v) for k, v in models.items()},
            "weak_signals": sorted(((k, abs(mean([self._directional_return(r) for r in v]))) for k, v in models.items()), key=lambda x: x[1])[:10],
        }

    def generate_candidates(self) -> list[ResearchCandidate]:
        errors = self.attribute_errors()
        candidates: list[ResearchCandidate] = []
        if errors["snapshots"] >= self.MIN_OOS and errors["calibration_brier"] > 0.20:
            candidates.append(self.propose("calibration", "recalibrate probability/confidence mapping", "Observed calibration error exceeds 0.20."))
        for regime, count in errors["regime_errors"].items():
            if count >= 10:
                candidates.append(self.propose("production", f"regime-conditioned abstention for {regime}", f"Repeated adverse outcomes in regime {regime}."))
        return candidates

    def propose(self, base_model: str, hypothesis: str, reason: str) -> ResearchCandidate:
        fingerprint = self._fingerprint([base_model, hypothesis, reason])
        candidate = ResearchCandidate(str(uuid4()), base_model, hypothesis, "quarantined", datetime.now(timezone.utc).isoformat(), {"research_version": self.VERSION, "hypothesis_fingerprint": fingerprint}, {"reason": reason})
        self.candidates.append(candidate)
        return candidate

    @staticmethod
    def _fingerprint(parts: Iterable[Any]) -> str:
        return sha256("|".join(str(x) for x in parts).encode()).hexdigest()

    def _candidate_actions(self, rows: list[ResearchSnapshot], candidate: ResearchCandidate) -> list[str]:
        actions = [r.action.upper() for r in rows]
        # Only deterministic, data-driven candidate transformations are allowed.
        if candidate.hypothesis.startswith("regime-conditioned abstention"):
            regime = candidate.hypothesis.rsplit(" ", 1)[-1]
            return ["NO_TRADE" if r.regime == regime else a for r, a in zip(rows, actions)]
        if candidate.hypothesis.startswith("recalibrate"):
            return ["NO_TRADE" if r.confidence < 0.60 else a for r, a in zip(rows, actions)]
        return actions

    def _returns_for_actions(self, rows: list[ResearchSnapshot], actions: list[str]) -> list[float]:
        out: list[float] = []
        for row, action in zip(rows, actions):
            if action == "NO_TRADE":
                out.append(0.0)
            elif action == "SHORT":
                out.append(-row.realized_return - abs(row.slippage) - abs(row.fees))
            else:
                out.append(row.realized_return - abs(row.slippage) - abs(row.fees))
        return out

    def _walk_forward(self, rows: list[ResearchSnapshot], candidate: ResearchCandidate) -> dict[str, Any]:
        if len(rows) < self.MIN_WALK_FORWARD * 20:
            return {"valid": False, "folds": 0, "reason": "insufficient chronological samples"}
        folds = min(5, max(3, len(rows) // 50))
        fold_size = len(rows) // folds
        scores = []
        for i in range(folds):
            start = i * fold_size
            end = len(rows) if i == folds - 1 else (i + 1) * fold_size
            test = rows[start:end]
            if not test:
                continue
            actions = self._candidate_actions(test, candidate)
            values = self._returns_for_actions(test, actions)
            scores.append(mean(values))
        return {"valid": len(scores) >= 3, "folds": len(scores), "fold_average_returns": scores, "positive_folds": sum(x > 0 for x in scores)}

    def _stress(self, returns: list[float]) -> dict[str, Any]:
        if not returns:
            return {"valid": False}
        shocks = [-0.0005, -0.001, -0.002]
        stressed = {str(s): mean([x + s for x in returns]) for s in shocks}
        return {"valid": True, "mean_under_shock": stressed, "worst": min(stressed.values())}

    def evaluate(self, candidate_id: str) -> ExperimentResult:
        candidate = next((c for c in self.candidates if c.candidate_id == candidate_id), None)
        if not candidate:
            raise KeyError("Research candidate not found")
        rows = list(self.snapshots)
        if len(rows) < self.MIN_OOS:
            raise ValueError(f"at least {self.MIN_OOS} completed snapshots are required")
        champion = self._returns_for_actions(rows, [r.action.upper() for r in rows])
        candidate_returns = self._returns_for_actions(rows, self._candidate_actions(rows, candidate))
        split = max(self.MIN_OOS, int(len(rows) * 0.70))
        split = min(split, len(rows) - self.MIN_OOS)
        oos_rows = rows[split:]
        oos_champion = self._returns_for_actions(oos_rows, [r.action.upper() for r in oos_rows])
        oos_candidate = self._returns_for_actions(oos_rows, self._candidate_actions(oos_rows, candidate))
        delta = [b - a for a, b in zip(oos_champion, oos_candidate)]
        ci_low, ci_high = self._paired_bootstrap(delta)
        _, net, dd = self._equity_metrics(oos_candidate)
        _, _, champion_dd = self._equity_metrics(oos_champion)
        wf = self._walk_forward(rows[:split], candidate)
        stress = self._stress(oos_candidate)
        economic = mean(oos_candidate) > mean(oos_champion) and net > sum(oos_champion)
        passed = len(oos_rows) >= self.MIN_OOS and ci_low > 0 and economic and dd <= champion_dd + 0.02 and wf.get("valid", False) and stress.get("worst", -1) > -0.01
        fingerprint = self._fingerprint([self.VERSION, candidate.candidate_id, [r.record_id for r in rows], candidate.hypothesis])
        result = ExperimentResult(str(uuid4()), candidate.candidate_id, len(oos_rows), mean(oos_candidate), sum(x > 0 for x in oos_candidate) / len(oos_candidate), dd, net, mean(delta), ci_low, ci_high, wf, {"valid": True, "samples": len(oos_rows), "candidate_average": mean(oos_candidate), "champion_average": mean(oos_champion)}, stress, fingerprint, passed)
        self.experiments.append(result)
        candidate.status = "promotion_eligible" if passed else "rejected"
        candidate.evidence = {**candidate.evidence, "experiment_id": result.experiment_id, "passed": passed, "oos_samples": len(oos_rows), "bootstrap_ci": [ci_low, ci_high], "fingerprint": fingerprint}
        return result

    def snapshot(self) -> dict[str, Any]:
        return {"engine": self.VERSION, "snapshots": len(self.snapshots), "candidates": len(self.candidates), "quarantined": sum(c.status == "quarantined" for c in self.candidates), "experiments": len(self.experiments), "production_self_modification": False, "execution_authority": False, "latest_experiment": asdict(self.experiments[-1]) if self.experiments else None}
