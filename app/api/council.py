from fastapi import APIRouter
from pydantic import BaseModel
from ai.agents import AgentContext, IntelligenceCouncil
from ai.agent_memory import AgentMemory

router = APIRouter(prefix="/api/council", tags=["multi-agent-intelligence"])
_council = IntelligenceCouncil()
_memory = AgentMemory()

class DeliberationRequest(BaseModel):
    context: AgentContext

@router.get("/status")
def council_status():
    return {
        "engine": "HHHAI Multi-Agent Intelligence Council",
        "phase": 8,
        "agents": [{"id": a.agent_id, "name": a.name, "weight": a.weight} for a in _council.definitions],
        "decision_authority": "risk_engine_after_council",
        "execution_authority": False,
        "adversarial_agent": True,
        "continuous_learning": False,
        "live_market_feeds": False,
    }

@router.post("/deliberate")
def deliberate(request: DeliberationRequest):
    decision = _council.deliberate(request.context)
    _memory.record(decision)
    return decision.model_dump(mode="json")

@router.get("/recent")
def recent(limit: int = 10):
    limit = max(1, min(limit, 50))
    return {"items": [x.model_dump(mode="json") for x in _memory.recent(limit)]}
