from fastapi import APIRouter, Header, HTTPException

from app.api.admin import require_admin
from app.ml.predictive_brain import predictive_brain
from app.ml.phase2_authority import authoritative_config, fetch_authoritative_dataset
from app.ml.model_persistence import persist_brain
from app.core.config import settings

router = APIRouter(prefix="/api/model", tags=["predictive-model"])


brain = predictive_brain


@router.get("/status")
def status():
    # A trained model is not execution authority. Live execution remains an
    # independently gated capability and is disabled by default.
    brain_manifest = brain.manifest()
    return {
        "version": brain.version if brain.bundle is not None else "untrained",
        "trained": brain.bundle is not None,
        "brain_ready": brain.bundle is not None,
        "model_ready": brain.bundle is not None,
        "execution_gate": False,
        "live_trading_enabled": bool(settings.live_trading_enabled),
        "testnet_trading_enabled": bool(settings.testnet_trading_enabled),
        "autotrading_enabled": bool(settings.hhhai_autotrading_enabled),
        "artifact": str(brain.artifact_path),
        "brain_manifest": brain_manifest,
    }


@router.post("/bootstrap")
def bootstrap(
    x_hhhai_admin_token: str | None = Header(default=None),
):
    """Run the single authoritative Phase 2 brain pipeline; never authorizes live execution."""
    require_admin(x_hhhai_admin_token)
    try:
        rows, data_meta = fetch_authoritative_dataset()
        if len(rows) < 500:
            raise HTTPException(status_code=422, detail=f"Historical dataset contains only {len(rows)} usable rows; at least 500 are required.")

        version = f"brain-{data_meta['symbol']}-{data_meta['interval']}"
        brain_report = brain.train(rows, version=version)
        result = {
            "status": brain_report.status,
            "version": brain_report.version,
            "metrics": brain_report.metrics,
            "reason": brain_report.reason,
            "artifact": brain_report.artifact,
            "data": data_meta,
            "authoritative_config": authoritative_config(),
        }
        if brain_report.status == "PROMOTED":
            import asyncio
            persisted = asyncio.run(persist_brain(brain_report.metrics))
            if not persisted:
                result["status"] = "REJECTED"
                result["reason"] = "Brain passed OOS gates but its authoritative artifact could not be persisted."
            else:
                result["persisted_artifact"] = "predictive_brain"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model bootstrap failed: {type(exc).__name__}: {exc}")
