from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from ai.evaluation import EvaluationResult, evaluate_outcomes
from ai.model_registry import ModelRegistry
from ai.models import TradeOutcome
from ai.learning import CandidatePromotionPolicy, can_promote


@dataclass
class DecisionRecord:
    record_id: str
    symbol: str
    action: str
    thesis: str
    features: dict[str, float]
    confidence: float
    model_version: str
    created_at: str
    outcome_return: float | None = None
    outcome_profitable: bool | None = None


@dataclass
class LearningCandidate:
    candidate_id: str
    base_model: str
    proposed_change: str
    reason: str
    status: str = "quarantined"
    created_at: str = ""


class ControlledLearningEngine:
    """Records outcomes and proposes changes without self-modifying production code.

    Candidates are quarantined until independently evaluated and promoted through
    an explicit gate. This engine never receives exchange execution authority.
    """
    model_version = "phase13-controlled-learning-v1"

    def __init__(self, max_records: int = 2000):
        self.records: list[DecisionRecord] = []
        self.candidates: list[LearningCandidate] = []
        self.registry = ModelRegistry()
        self.registry.register("adaptive-intelligence", "production", "production")
        self.max_records = max_records

    def record_decision(self, *, symbol: str, action: str, thesis: str,
                        features: dict[str, float], confidence: float,
                        model_version: str = "production") -> DecisionRecord:
        row = DecisionRecord(str(uuid4()), symbol, action, thesis, features,
                             max(0.0, min(1.0, confidence)), model_version,
                             datetime.now(timezone.utc).isoformat())
        self.records.append(row)
        if len(self.records) > self.max_records:
            self.records.pop(0)
        return row

    def record_outcome(self, record_id: str, realized_return: float) -> DecisionRecord:
        for row in self.records:
            if row.record_id == record_id:
                row.outcome_return = realized_return
                row.outcome_profitable = realized_return > 0
                return row
        raise KeyError("Decision record not found")

    def evaluate_completed(self) -> EvaluationResult:
        outcomes = [
            TradeOutcome(trade_id=r.record_id, model_version=r.model_version, predicted_side=r.action, realized_return=r.outcome_return, was_profitable=bool(r.outcome_profitable), timestamp=datetime.fromisoformat(r.created_at))
            for r in self.records if r.outcome_return is not None
        ]
        return evaluate_outcomes(outcomes)

    def propose_candidate(self, *, base_model: str, proposed_change: str,
                          reason: str) -> LearningCandidate:
        candidate = LearningCandidate(
            candidate_id=str(uuid4()), base_model=base_model,
            proposed_change=proposed_change, reason=reason,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.candidates.append(candidate)
        return candidate

    def evaluate_candidate(self, candidate_id: str, result: EvaluationResult,
                           policy: CandidatePromotionPolicy | None = None) -> LearningCandidate:
        candidate = next((c for c in self.candidates if c.candidate_id == candidate_id), None)
        if not candidate:
            raise KeyError("Candidate not found")
        policy = policy or CandidatePromotionPolicy()
        candidate.status = "promotion_eligible" if can_promote(result, policy) else "rejected"
        return candidate

    def status(self) -> dict[str, Any]:
        result = self.evaluate_completed()
        return {
            "engine": self.model_version,
            "records": len(self.records),
            "completed_outcomes": result.trades,
            "win_rate": result.win_rate,
            "average_return": result.average_return,
            "candidates": len(self.candidates),
            "quarantined_candidates": sum(c.status == "quarantined" for c in self.candidates),
            "production_self_modification": False,
            "execution_authority": False,
        }

    def snapshot(self) -> dict[str, Any]:
        return {"status": self.status(), "candidates": [asdict(c) for c in self.candidates[-20:]]}
