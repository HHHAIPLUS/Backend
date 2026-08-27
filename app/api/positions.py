from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ai.adaptive_engine import AdaptivePositionEngine
from ai.adaptive_models import MarketObservation, PositionSnapshot
from ai.position_management import PositionManager

router = APIRouter(prefix='/api/positions', tags=['autonomous-position-management'])
adaptive = AdaptivePositionEngine()
manager = PositionManager()

class RegisterRequest(BaseModel):
    position: PositionSnapshot
    thesis: str
    quantity: float = Field(gt=0)

class ReviewRequest(BaseModel):
    position: PositionSnapshot
    market: MarketObservation
    thesis: str = 'Existing position thesis'
    quantity: float = Field(default=1.0, gt=0)

@router.get('/management/status')
def management_status():
    return {
        'engine': 'Autonomous Position Management',
        'continuous_review': True,
        'fixed_take_profit_required': False,
        'dynamic_profit_protection': True,
        'partial_reduction': True,
        'emergency_exit_detection': True,
        'execution_authority': False,
        'risk_authority': 'backend_risk_engine',
        'open_positions_tracked': len(manager.positions),
    }

@router.post('/register')
def register(request: RegisterRequest):
    adaptive.register_position(request.position, request.thesis)
    return manager.register(request.position, request.quantity).model_dump(mode='json')

@router.post('/review')
def review(request: ReviewRequest):
    if request.position.symbol not in manager.positions:
        adaptive.register_position(request.position, request.thesis)
        manager.register(request.position, request.quantity)
    adaptive_decision = adaptive.evaluate(request.position, request.market)
    management_decision = manager.review(request.position, request.market, adaptive_decision)
    return {
        'adaptive_decision': adaptive_decision.model_dump(mode='json'),
        'management_decision': management_decision.model_dump(mode='json'),
        'position': manager.positions[request.position.symbol].model_dump(mode='json'),
    }

@router.get('/active')
def active_positions():
    return {'positions': [p.model_dump(mode='json') for p in manager.snapshot()] }

@router.get('/worker-reviews')
def worker_reviews():
    from app.services.monitor_worker import monitor
    return {'reviews': monitor.position_reviews[-50:]}
