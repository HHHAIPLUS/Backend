from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.admin import require_admin
from app.ml.predictive import predictive_model
from app.ml.predictive_brain import PredictiveBrain
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


brain = PredictiveBrain()


@router.get("/status")
def status():
    # A trained model is not execution authority. Live execution remains an
    # independently gated capability and is disabled by default.
    brain_manifest = brain.manifest()
    return {
        "version": brain.version if brain.bundle is not None else predictive_model.version,
        "trained": brain.bundle is not None or predictive_model.model is not None,
        "brain_ready": brain.bundle is not None,
        "model_ready": predictive_model.model is not None,
        "execution_gate": False,
        "live_trading_enabled": bool(settings.live_trading_enabled),
        "testnet_trading_enabled": bool(settings.testnet_trading_enabled),
        "autotrading_enabled": bool(settings.hhhai_autotrading_enabled),
        "artifact": str(brain.artifact_path if brain.bundle is not None else predictive_model.model_path),
        "brain_manifest": brain_manifest,
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
    """Build, validate and evaluate the predictive brain; never authorizes live execution."""
    require_admin(x_hhhai_admin_token)
    request = request or BootstrapRequest(symbol=symbol, interval=interval, limit=limit, horizon=horizon, threshold=threshold)
    try:
        effective_limit = max(5000, request.limit)
        raw, provider = fetch_historical_klines(symbol=request.symbol, interval=request.interval, limit=effective_limit)
        rows = build_dataset(raw, request.horizon, request.threshold)
        if len(rows) < 500:
            raise HTTPException(status_code=422, detail=f"Historical dataset contains only {len(rows)} usable rows; at least 500 are required.")

        result = validate_and_promote(rows, version=f"bootstrap-{request.symbol.upper()}-{request.interval}")
        brain_report = brain.train(rows, version=f"brain-{request.symbol.upper()}-{request.interval}")
        result["predictive_brain"] = {
            "status": brain_report.status,
            "version": brain_report.version,
            "metrics": brain_report.metrics,
            "reason": brain_report.reason,
            "artifact": brain_report.artifact,
        }
        # A direction-only legacy promotion is no longer sufficient for Stage 3.
        # If the advanced brain cannot beat the honest baseline, it remains unpromoted.
        if brain_report.status != "PROMOTED":
            result["status"] = "REJECTED"
            result["reason"] = "Predictive brain promotion gate failed: " + brain_report.reason
        else:
            result["status"] = "PROMOTED"
            result["version"] = brain_report.version

        result.update({"data_provider": provider, "symbol": request.symbol.upper(), "interval": request.interval, "requested_candles": request.limit, "effective_candles": effective_limit, "training_rows": len(rows)})
        if result.get("status") == "PROMOTED":
            import asyncio
            asyncio.run(persist_model(result.get("metrics")))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model bootstrap failed: {type(exc).__name__}: {exc}")
