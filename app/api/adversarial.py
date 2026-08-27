from fastapi import APIRouter
from pydantic import BaseModel, Field
from ai.adversarial import AdversarialEngine
router=APIRouter(prefix='/api/adversarial',tags=['adversarial-intelligence'])
engine=AdversarialEngine()
class ChallengeRequest(BaseModel):
    symbol:str
    proposed_action:str
    context:dict[str,float|str|None]=Field(default_factory=dict)
@router.get('/status')
def status():
    return {'engine':'Adversarial Intelligence','mode':'challenge_only','execution_authority':False,'blocking_enabled':True}
@router.post('/challenge')
def challenge(req:ChallengeRequest):
    return engine.evaluate(symbol=req.symbol,proposed_action=req.proposed_action,context=req.context).model_dump(mode='json')
