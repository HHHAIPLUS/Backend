from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.admin import require_admin
from app.ml.predictive import predictive_model
from app.ml.bootstrap import fetch_historical_klines, build_dataset, validate_and_promote
from app.ml.model_persistence import persist_model
from app.core.config import settings

router = APIRouter(prefix="/api/model", tags=["predictive-model"])


class BootstrapRequest(BaseModel):
    symbol: str = Field(default="BTCUSDT", min_length=5, max_length=30)
    interval: str = "5m"
    limit: int = Field(default=5000, ge=500, le=10000)
    horizon: int = Field(default=6, ge=1, le=24)
    threshold: float = Field(default=0.0025, gt=0, lt=0.1)


@router.get("/status")
def status():
    # A trained model is not execution authority. Live execution remains an
    # independently gated capability and is disabled by default.
    return {
        "version": predictive_model.version,
        "trained": predictive_model.model is not None,
        "model_ready": predictive_model.model is not None,
        "execution_gate": False,
        "live_trading_enabled": bool(settings.live_trading_enabled),
        "testnet_trading_enabled": bool(settings.testnet_trading_enabled),
        "autotrading_enabled": bool(settings.hhhai_autotrading_enabled),
        "artifact": str(predictive_model.model_path),
    }


@router.post("/bootstrap")
def bootstrap(
    request: BootstrapRequest | None = None,
    symbol: str = Query(default="BTCUSDT", min_length=5, max_length=30),
    interval: str = Query(default="5m"),
    limit: int = Query(default=5000, ge=500, le=10000),
    horizon: int = Query(default=6, ge=1, le=24),
    threshold: float = Query(default=0.0025, gt=0, lt=0.1),
    x_hhhai_admin_token: str | None = Header(default=None),
):
    """Build and validate a candidate model; never authorizes live execution."""
    require_admin(x_hhhai_admin_token)
    request = request or BootstrapRequest(symbol=symbol, interval=interval, limit=limit, horizon=horizon, threshold=threshold)
    try:
        effective_limit = max(5000, request.limit)
        raw, provider = fetch_historical_klines(symbol=request.symbol, interval=request.interval, limit=effective_limit)
        rows = build_dataset(raw, request.horizon, request.threshold)
        if len(rows) < 500:
            raise HTTPException(status_code=422, detail=f"Historical dataset contains only {len(rows)} usable rows; at least 500 are required.")
        result = validate_and_promote(rows, version=f"bootstrap-{request.symbol.upper()}-{request.interval}")
        result.update({"data_provider": provider, "symbol": request.symbol.upper(), "interval": request.interval, "requested_candles": request.limit, "effective_candles": effective_limit, "training_rows": len(rows)})
        if result.get("status") == "PROMOTED":
            import asyncio
            asyncio.run(persist_model(result.get("metrics")))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model bootstrap failed: {type(exc).__name__}: {exc}")
