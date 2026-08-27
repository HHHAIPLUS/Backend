from fastapi import APIRouter, Header, HTTPException
from ai.capital_guard import CapitalGuard,RiskSnapshot
from app.services.safety import kill_switch
from app.api.admin import require_admin
router=APIRouter(prefix='/api/capital',tags=['capital-safety']); guard=CapitalGuard()
@router.get('/status')
def status(): return {'engine':'HHHAI Capital Guard','kill_switch':kill_switch.gate(),'execution_authority':False,'live_trading':False,'policy':guard.policy.__dict__}
@router.post('/evaluate')
def evaluate(payload:dict):
 if kill_switch.enabled:return {'decision':'emergency_stop','reasons':[kill_switch.reason or 'Kill switch engaged.'],'execution_authority':False}
 return guard.evaluate(RiskSnapshot(**payload))
@router.post('/kill-switch/engage')
def engage(reason='Manual emergency stop',x_hhhai_admin_token:str|None=Header(default=None)):
 require_admin(x_hhhai_admin_token); kill_switch.engage(reason); return kill_switch.gate()
@router.post('/kill-switch/reset')
def reset(x_hhhai_admin_token:str|None=Header(default=None)):
 require_admin(x_hhhai_admin_token); kill_switch.reset(); return kill_switch.gate()
