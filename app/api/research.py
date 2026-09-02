from __future__ import annotations

from dataclasses import asdict
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ai.research_loop import ResearchLoop
from app.persistence.supabase import store
from app.persistence.repository import record_event

router = APIRouter(prefix="/api/research", tags=["stage7-research"])
research = ResearchLoop()


class SnapshotRequest(BaseModel):
    record_id: str
    symbol: str
    action: str
    model_version: str
    regime: str = "unknown"
    horizon: int = Field(default=6, ge=1)
    confidence: float = Field(ge=0, le=1)
    expected_probability: float | None = Field(default=None, ge=0, le=1)
    features: dict[str, float] = Field(default_factory=dict)
    thesis: str = ""
    created_at: str
    realized_return: float
    execution_return: float | None = None
    slippage: float = 0.0
    fees: float = 0.0


class CandidateRequest(BaseModel):
    base_model: str
    hypothesis: str
    reason: str


@router.get("/status")
def status():
    return research.snapshot()


@router.get("/errors")
def errors():
    return research.attribute_errors()


@router.get("/candidates")
def candidates():
    return {"candidates": [asdict(x) for x in research.candidates]}


@router.get("/experiments")
def experiments():
    return {"experiments": [asdict(x) for x in research.experiments[-50:]]}


@router.post("/snapshots")
async def add_snapshot(request: SnapshotRequest):
    from ai.research_loop import ResearchSnapshot
    row = ResearchSnapshot(**request.model_dump())
    research.add_snapshot(row)
    if store.configured:
        try:
            await store.insert("research_snapshots", asdict(row))
        except Exception:
            pass
    return asdict(row)


@router.post("/candidates")
def create_candidate(request: CandidateRequest):
    return asdict(research.propose(**request.model_dump()))


@router.post("/run/{candidate_id}")
async def run_experiment(candidate_id: str):
    result = research.evaluate(candidate_id)
    if store.configured:
        try:
            await store.insert("research_experiments", asdict(result))
            await record_event("stage7_research_experiment", {"experiment_id": result.experiment_id, "candidate_id": candidate_id, "passed": result.passed})
        except Exception:
            pass
    return asdict(result)


@router.post("/auto-generate")
def auto_generate():
    return {"candidates": [asdict(x) for x in research.generate_candidates()]}
