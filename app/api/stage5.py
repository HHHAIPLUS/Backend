from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.ml.adaptive_intelligence import AdaptiveObservation, adaptive_intelligence
from ai.stage5_engine import Stage5DecisionEngine

router = APIRouter(prefix="/api/decision", tags=["stage5-advanced-decision"])
_engine = Stage5DecisionEngine(adaptive_intelligence)


class DecisionRequest(BaseModel):
    market_state: dict[str, Any]
    predictive: dict[str, Any]
    risk_vetoes: list[str] = Field(default_factory=list)
    position_side: str | None = None
    unrealized_return: float = 0.0
    thesis_integrity: float = Field(default=0.5, ge=0, le=1)


class OutcomeRequest(BaseModel):
    decision: dict[str, Any]
    realized_return: float


@router.get("/status")
def status():
    return _engine.status()


@router.post("/evaluate")
def evaluate(request: DecisionRequest):
    return _engine.evaluate(**request.model_dump()).as_dict()


@router.post("/outcome")
def outcome(request: OutcomeRequest):
    from ai.advanced_decision_engine import AdvancedDecision
    _engine.record_outcome(decision=AdvancedDecision(**request.decision), realized_return=request.realized_return)
    return _engine.decision_quality().as_dict()
