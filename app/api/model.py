from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from app.api.admin import require_admin
from app.ml.predictive import predictive_model
from app.ml.bootstrap import fetch_binance_klines, build_dataset, validate_and_promote
from app.ml.model_persistence import persist_model

router=APIRouter(prefix='/api/model',tags=['predictive-model'])

class BootstrapRequest(BaseModel):
    symbol: str = Field(default='BTCUSDT', min_length=5, max_length=30)
    interval: str = '5m'
    limit: int = Field(default=1500, ge=500, le=1500)
    horizon: int = Field(default=6, ge=1, le=24)
    threshold: float = Field(default=0.0025, gt=0, lt=0.1)

@router.get('/status')
def status():
    return {'version':predictive_model.version,'trained':predictive_model.model is not None,'execution_gate':predictive_model.model is not None,'artifact':str(predictive_model.model_path)}

@router.post('/bootstrap')
def bootstrap(request: BootstrapRequest, x_hhhai_admin_token: str | None = Header(default=None)):
    require_admin(x_hhhai_admin_token)
    try:
        raw = fetch_binance_klines(request.symbol, request.interval, request.limit)
        rows = build_dataset(raw, request.horizon, request.threshold)
        if len(rows) < 500:
            raise HTTPException(422, f'Historical dataset contains only {len(rows)} usable rows; at least 500 are required.')
        result = validate_and_promote(rows, version=f'bootstrap-{request.symbol.upper()}-{request.interval}')
        if result.get("status") == "PROMOTED":
            import asyncio
            asyncio.run(persist_model(result.get("metrics")))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f'Model bootstrap failed: {type(exc).__name__}: {exc}')
