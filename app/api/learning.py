from fastapi import APIRouter
from pydantic import BaseModel, Field
from ai.self_learning import ControlledLearningEngine
from app.persistence.repository import record_decision as persist_decision, record_outcome as persist_outcome, record_adaptive_observation
from app.persistence.supabase import store
from app.api.adaptive import adaptive
from app.ml.adaptive_intelligence import AdaptiveObservation
from dataclasses import asdict
from uuid import uuid4

async def hydrate_learning():
    if not store.configured: return
    try:
        rows=await store.select("decision_records", {"select":"id,payload,created_at","order":"created_at.desc","limit":"2000"})
        for item in reversed(rows):
            payload=item.get("payload") or {}
            if not payload.get("record_id"): continue
            from ai.self_learning import DecisionRecord
            row=DecisionRecord(**payload)
            if not any(x.record_id==row.record_id for x in _engine.records): _engine.records.append(row)
    except Exception:
        return

router = APIRouter(prefix="/api/learning", tags=["controlled-learning"])
_engine = ControlledLearningEngine()

class DecisionRequest(BaseModel):
    symbol: str
    action: str
    thesis: str
    features: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    model_version: str = "production"

class OutcomeRequest(BaseModel):
    record_id: str
    realized_return: float

class CandidateRequest(BaseModel):
    base_model: str
    proposed_change: str
    reason: str

@router.get("/status")
def learning_status():
    return _engine.status()

@router.get("/journal")
def learning_journal(limit: int = 20):
    rows = _engine.records[-max(1, min(limit, 100)):]
    return {"records": [r.__dict__ for r in reversed(rows)]}

@router.get("/candidates")
def learning_candidates():
    return _engine.snapshot()

@router.post("/decisions")
async def record_decision(request: DecisionRequest):
    row=_engine.record_decision(**request.model_dump())
    if store.configured:
        try: await persist_decision(row.symbol, row.__dict__)
        except Exception: pass
    return row.__dict__

@router.post("/outcomes")
async def record_outcome(request: OutcomeRequest):
    row=_engine.record_outcome(request.record_id, request.realized_return)
    features=row.features or {}
    observation=AdaptiveObservation(symbol=row.symbol, model_version=row.model_version, action=row.action.upper(), confidence=row.confidence, realized_return=request.realized_return, observed_at=row.created_at, regime=str(features.get("market_regime", features.get("regime", "unknown"))), horizon=int(features.get("horizon", 6) or 6), expected_probability=features.get("expected_probability"), features=features)
    adaptive.add_observation(observation)
    if store.configured:
        try: await persist_outcome(request.record_id, row.__dict__)
        except Exception: pass
        try: await record_adaptive_observation({'id':str(uuid4()), **asdict(observation)})
        except Exception: pass
    return row.__dict__

@router.post("/candidates")
def propose_candidate(request: CandidateRequest):
    return _engine.propose_candidate(**request.model_dump()).__dict__
