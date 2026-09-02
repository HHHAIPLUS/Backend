from __future__ import annotations

from dataclasses import asdict
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ai.research_loop import ResearchLoop, ResearchSnapshot, ResearchCandidate
from ai.research_governance import ResearchGovernance
from app.persistence.supabase import store
from app.persistence.repository import record_event

router = APIRouter(prefix="/api/research", tags=["stage7-research"])
research = ResearchLoop()
governance = ResearchGovernance()

class SnapshotRequest(BaseModel):
    record_id: str; symbol: str; action: str; model_version: str; regime: str = "unknown"
    horizon: int = Field(default=6, ge=1); confidence: float = Field(ge=0, le=1)
    expected_probability: float | None = Field(default=None, ge=0, le=1)
    features: dict[str, float] = Field(default_factory=dict); thesis: str = ""; created_at: str
    realized_return: float; execution_return: float | None = None; slippage: float = 0.0; fees: float = 0.0

class CandidateRequest(BaseModel): base_model: str; hypothesis: str; reason: str
class ApprovalRequest(BaseModel): approved_by: str; rollback_target: str

async def hydrate_research() -> None:
    if not store.configured: return
    try:
        rows = await store.select("research_snapshots", {"select": "*", "order": "created_at.asc", "limit": "20000"})
        for row in rows:
            payload = {k: row.get(k) for k in ResearchSnapshot.__dataclass_fields__}; payload["features"] = payload.get("features") or {}
            research.add_snapshot(ResearchSnapshot(**payload))
        candidates = await store.select("research_candidates", {"select": "*", "order": "created_at.asc", "limit": "1000"})
        for row in candidates:
            research.candidates.append(ResearchCandidate(str(row["id"]), row["base_model"], row["hypothesis"], row["status"], row["created_at"], row.get("lineage") or {}, row.get("evidence") or {}))
    except Exception: return

@router.get("/status")
def status(): return research.snapshot()
@router.get("/errors")
def errors(): return research.attribute_errors()
@router.get("/candidates")
def candidates(): return {"candidates": [asdict(x) for x in research.candidates]}
@router.get("/experiments")
def experiments(): return {"experiments": [asdict(x) for x in research.experiments[-50:]]}

@router.post("/snapshots")
async def add_snapshot(request: SnapshotRequest):
    row = ResearchSnapshot(**request.model_dump()); research.add_snapshot(row)
    if store.configured:
        try: await store.upsert("research_snapshots", asdict(row), "record_id")
        except Exception: pass
    return asdict(row)

@router.post("/candidates")
async def create_candidate(request: CandidateRequest):
    candidate = research.propose(**request.model_dump())
    if store.configured:
        try: await store.insert("research_candidates", {"id": candidate.candidate_id, "base_model": candidate.base_model, "hypothesis": candidate.hypothesis, "status": candidate.status, "created_at": candidate.created_at, "lineage": candidate.lineage, "evidence": candidate.evidence})
        except Exception: pass
    return asdict(candidate)

@router.post("/run/{candidate_id}")
async def run_experiment(candidate_id: str):
    result = research.evaluate(candidate_id)
    if store.configured:
        try: await store.insert("research_experiments", asdict(result)); await record_event("stage7_research_experiment", {"experiment_id": result.experiment_id, "candidate_id": candidate_id, "passed": result.passed})
        except Exception: pass
    return asdict(result)

@router.post("/approve/{candidate_id}")
async def approve_candidate(candidate_id: str, request: ApprovalRequest):
    candidate = next((x for x in research.candidates if x.candidate_id == candidate_id), None)
    if candidate is None: raise KeyError("Research candidate not found")
    approval = governance.approve(candidate, request.approved_by, request.rollback_target)
    if store.configured:
        try: await store.upsert("research_candidates", {"id": candidate.candidate_id, "base_model": candidate.base_model, "hypothesis": candidate.hypothesis, "status": candidate.status, "created_at": candidate.created_at, "lineage": candidate.lineage, "evidence": candidate.evidence}, "id")
        except Exception: pass
    return asdict(approval)

@router.post("/auto-generate")
def auto_generate(): return {"candidates": [asdict(x) for x in research.generate_candidates()]}
