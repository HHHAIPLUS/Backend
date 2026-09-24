from __future__ import annotations

import base64
import io

import joblib

from app.ml.predictive_brain import predictive_brain
from app.persistence.supabase import store


async def persist_brain(metrics: dict | None = None):
    if not store.configured or predictive_brain.bundle is None:
        return False
    buffer = io.BytesIO()
    joblib.dump(predictive_brain.bundle, buffer)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    manifest = predictive_brain.manifest() or {}
    if manifest.get("promotion", {}).get("promoted") is not True:
        return False
    await store.upsert(
        "model_artifacts",
        {
            "name": "predictive_brain",
            "version": predictive_brain.version,
            "artifact": {
                "encoding": "joblib-base64",
                "data": encoded,
                "schema_version": manifest.get("schema_version"),
            },
            "metrics": metrics or manifest.get("metrics") or {},
        },
        "version",
    )
    return True


async def hydrate_model():
    """Hydrate only the promoted predictive brain; no legacy model is execution authority."""
    if not store.configured:
        return False
    try:
        row = await store.latest(
            "model_artifacts",
            {
                "select": "*",
                "name": "eq.predictive_brain",
                "order": "created_at.desc",
                "limit": "1",
            },
        )
        if not row:
            return False
        artifact = row.get("artifact") or {}
        if artifact.get("encoding") != "joblib-base64":
            return False
        data = base64.b64decode(str(artifact["data"]))
        bundle = joblib.load(io.BytesIO(data))
        if bundle.get("schema_version") != 4:
            return False
        if bundle.get("promotion", {}).get("promoted") is not True:
            return False
        if bundle.get("feature_hash") is None or not bundle.get("features"):
            return False
        predictive_brain.bundle = bundle
        predictive_brain.version = str(
            row.get("version") or bundle.get("version") or "persisted"
        )
        return True
    except Exception:
        return False
