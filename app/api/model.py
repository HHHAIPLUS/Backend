from fastapi import APIRouter

from app.ml.predictive_brain import predictive_brain
from app.core.config import settings

router = APIRouter(prefix="/api/model", tags=["predictive-model"])

brain = predictive_brain


@router.get("/status")
def status():
    """Expose the production brain state without providing a second training path."""
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
        "training_authority": "github-actions-phase2-authoritative",
    }
