from fastapi import APIRouter
from pydantic import BaseModel,Field
from ai.trade_optimizer import TradeCandidate,MarketRegime,TradeOptimizer
router=APIRouter(prefix='/api/trade-optimizer',tags=['trade-optimization']); optimizer=TradeOptimizer()
class CandidateIn(BaseModel):
 symbol:str; side:str; entry:float=Field(gt=0); target:float=Field(gt=0); invalidation:float=Field(gt=0); probability_of_success:float=Field(ge=0,le=1); regime_fit:float=Field(ge=0,le=1); confirmation:float=Field(ge=0,le=1); timing_quality:float=Field(ge=0,le=1); liquidity_score:float=Field(ge=0,le=1); news_risk:float=Field(ge=0,le=1)
class RegimeIn(BaseModel):
 name:str; trend_strength:float=0; volatility:float=0; liquidity:float=Field(default=1,ge=0,le=1)
class EvaluateRequest(BaseModel): candidate:CandidateIn; regime:RegimeIn
@router.get('/status')
def status(): return {'engine':'HHHAI Trading Intelligence Optimization I','capabilities':['expected value','dynamic risk/reward','regime-aware selection','trade-quality filtering','entry timing'],'execution_authority':False,'live_trading':False}
@router.post('/evaluate')
def evaluate(request:EvaluateRequest): return optimizer.evaluate(TradeCandidate(**request.candidate.model_dump()),MarketRegime(**request.regime.model_dump()))
