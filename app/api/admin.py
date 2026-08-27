from fastapi import APIRouter, Header, HTTPException
from app.core.security import constant_time_equal, live_trading_enabled
from app.services.safety import kill_switch
import os
router = APIRouter(prefix='/api/admin', tags=['admin'])

def require_admin(token: str | None):
    expected = os.getenv('HHHAI_ADMIN_TOKEN', '')
    if not expected: raise HTTPException(503, 'Admin authentication is not configured')
    if not constant_time_equal(token or '', expected): raise HTTPException(401, 'Unauthorized')

@router.get('/security-status')
def security_status(x_hhhai_admin_token: str | None = Header(default=None)):
    require_admin(x_hhhai_admin_token)
    return {'authenticated': True, 'live_trading_enabled': live_trading_enabled()}

@router.post('/emergency-stop')
def emergency_stop(x_hhhai_admin_token: str | None = Header(default=None)):
    require_admin(x_hhhai_admin_token)
    kill_switch.engage('Authenticated admin emergency stop')
    return {'status': 'engaged', 'trading_allowed': False, 'kill_switch': kill_switch.gate()}

@router.get('/scenario-status')
def scenario_status(x_hhhai_admin_token: str | None = Header(default=None)):
    require_admin(x_hhhai_admin_token)
    from ai.scenario_engine import ScenarioEngine
    return {
        'phase': 11,
        'engine': ScenarioEngine.model_version,
        'multiple_futures': True,
        'uncertainty_tracking': True,
        'execution_authority': False,
    }
