from fastapi import APIRouter
from pydantic import BaseModel
from ai.adaptive_engine import AdaptivePositionEngine
from ai.adaptive_models import MarketObservation, PositionSnapshot

router = APIRouter(prefix='/api/adaptive', tags=['adaptive-intelligence'])
_engine = AdaptivePositionEngine()

class RegisterPositionRequest(BaseModel):
    position: PositionSnapshot
    thesis: str
class EvaluatePositionRequest(BaseModel):
    position: PositionSnapshot
    market: MarketObservation

@router.get('/status')
def adaptive_status():
    return {'engine':'Autonomous Adaptive Intelligence','mode':'decision_only','fixed_take_profit_required':False,'continuous_re_evaluation':True,'execution_authority':False,'risk_authority':'backend_risk_engine'}

@router.post('/positions/register')
def register_position(request: RegisterPositionRequest):
    return _engine.register_position(request.position, request.thesis).model_dump(mode='json')

@router.post('/positions/evaluate')
def evaluate_position(request: EvaluatePositionRequest):
    return _engine.evaluate(request.position, request.market).model_dump(mode='json')
