from fastapi import APIRouter
from pydantic import BaseModel, Field
from ai.adaptive_engine import AdaptivePositionEngine
from ai.adaptive_models import MarketObservation, PositionSnapshot
from app.ml.adaptive_intelligence import AdaptiveIntelligence, AdaptiveObservation
from app.persistence.repository import record_adaptive_candidate, record_adaptive_observation, load_adaptive_observations, update_adaptive_candidate
from app.persistence.supabase import store
from dataclasses import asdict
from datetime import datetime, timezone

router = APIRouter(prefix='/api/adaptive', tags=['adaptive-intelligence'])
_engine = AdaptivePositionEngine()
adaptive = AdaptiveIntelligence()

class RegisterPositionRequest(BaseModel):
    position: PositionSnapshot
    thesis: str
class EvaluatePositionRequest(BaseModel):
    position: PositionSnapshot
    market: MarketObservation
class ObservationRequest(BaseModel):
    symbol: str
    model_version: str
    action: str
    confidence: float = Field(ge=0, le=1)
    realized_return: float
    observed_at: str
    regime: str = 'unknown'
    horizon: int = Field(default=6, ge=1)
    expected_probability: float | None = Field(default=None, ge=0, le=1)
    features: dict[str, float] = Field(default_factory=dict)
class CandidateRequest(BaseModel):
    champion_version: str
    challenger_version: str
    reason: str
    evidence: dict = Field(default_factory=dict)
class CandidateEvaluationRequest(BaseModel):
    candidate_id: str
    champion_returns: list[float]
    challenger_returns: list[float]
    regimes: list[str] | None = None

async def hydrate_adaptive():
    if not store.configured:
        return
    try:
        rows = await load_adaptive_observations()
        for row in rows:
            adaptive.add_observation(AdaptiveObservation(
                symbol=row['symbol'], model_version=row['model_version'], action=row['action'],
                confidence=float(row['confidence']), realized_return=float(row['realized_return']),
                observed_at=row['observed_at'], regime=row.get('regime', 'unknown'),
                horizon=int(row.get('horizon', 6)), expected_probability=row.get('expected_probability'),
                features=row.get('features') or {},
            ))
    except Exception:
        return

@router.get('/status')
def adaptive_status():
    return {'engine':'Autonomous Adaptive Intelligence','mode':'decision_only','fixed_take_profit_required':False,'continuous_re_evaluation':True,'execution_authority':False,'risk_authority':'backend_risk_engine','learning':adaptive.status()}

@router.get('/intelligence/report')
def intelligence_report():
    return asdict(adaptive.report())

@router.post('/intelligence/observations')
async def add_intelligence_observation(request: ObservationRequest):
    observation = AdaptiveObservation(**request.model_dump())
    adaptive.add_observation(observation)
    if store.configured:
        try:
            await record_adaptive_observation({'id': __import__('uuid').uuid4().__str__(), **asdict(observation)})
        except Exception:
            pass
    return asdict(observation)

@router.post('/intelligence/candidates')
async def create_intelligence_candidate(request: CandidateRequest):
    candidate = adaptive.create_candidate(**request.model_dump())
    if store.configured:
        try:
            await record_adaptive_candidate({'id': candidate.candidate_id, 'champion_version':candidate.champion_version, 'challenger_version':candidate.challenger_version, 'status':candidate.status, 'reason':candidate.reason, 'evidence':candidate.evidence, 'created_at':candidate.created_at})
        except Exception:
            pass
    return asdict(candidate)

@router.post('/intelligence/candidates/evaluate')
async def evaluate_intelligence_candidate(request: CandidateEvaluationRequest):
    candidate = adaptive.evaluate_challenger(**request.model_dump())
    if store.configured:
        try:
            await update_adaptive_candidate(candidate.candidate_id, {'status':candidate.status, 'evidence':candidate.evidence, 'evaluated_at':datetime.now(timezone.utc).isoformat()})
        except Exception:
            pass
    return asdict(candidate)

@router.post('/positions/register')
def register_position(request: RegisterPositionRequest):
    return _engine.register_position(request.position, request.thesis).model_dump(mode='json')

@router.post('/positions/evaluate')
def evaluate_position(request: EvaluatePositionRequest):
    return _engine.evaluate(request.position, request.market).model_dump(mode='json')
