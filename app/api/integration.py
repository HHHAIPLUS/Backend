from fastapi import APIRouter, HTTPException, Header
from app.api.admin import require_admin
from app.core.config import settings
from app.persistence.supabase import store
from app.exchanges.factory import adapters
router=APIRouter(prefix='/api/integration',tags=['integration'])
@router.get('/status')
async def integration_status():
    return {'backend':'connected','supabase':{'configured':store.configured,'persistence':store.configured},'binance':{'configured':bool(settings.binance_api_key and settings.binance_api_secret),'testnet':settings.binance_testnet},'bitget':{'configured':bool(settings.bitget_api_key and settings.bitget_api_secret and settings.bitget_passphrase),'testnet':settings.bitget_testnet},'ai_pipeline':'multi-agent + scenario + adversarial + validated-model gate','risk_engine':'fail_closed','testnet_execution':settings.testnet_trading_enabled,'live_execution':settings.live_trading_enabled,'production_ready':False}
@router.get('/{exchange}/account')
async def exchange_account(exchange:str, x_hhhai_admin_token:str|None=Header(default=None)):
    require_admin(x_hhhai_admin_token)
    if exchange not in {'binance','bitget'}: raise HTTPException(404,'Unsupported exchange')
    try: return await adapters()[exchange].get_account_status()
    except Exception as e: raise HTTPException(503,str(e))
@router.get('/{exchange}/ticker/{symbol}')
async def exchange_ticker(exchange:str,symbol:str):
    if exchange not in {'binance','bitget'}: raise HTTPException(404,'Unsupported exchange')
    try: return await adapters()[exchange].get_ticker(symbol)
    except Exception as e: raise HTTPException(503,str(e))
