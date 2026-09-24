from __future__ import annotations

import base64
import hashlib
import io

import joblib

from app.ml.predictive_brain import predictive_brain
from app.persistence.supabase import store


async def persist_brain(metrics: dict | None = None):
    """Persist the exact promoted bundle and verify the stored row before succeeding."""
    if not store.configured or predictive_brain.bundle is None:
        return False

    manifest = predictive_brain.manifest() or {}
    if manifest.get("promotion", {}).get("promoted") is not True:
        return False

    buffer = io.BytesIO()
    joblib.dump(predictive_brain.bundle, buffer)
    raw = buffer.getvalue()
    encoded = base64.b64encode(raw).decode("ascii")
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    version = str(predictive_brain.version)

    await store.upsert(
        "model_artifacts",
        {
            "name": "predictive_brain",
            "version": version,
            "artifact": {
                "encoding": "joblib-base64",
                "data": encoded,
                "sha256": artifact_sha256,
                "schema_version": manifest.get("schema_version"),
                "version": version,
            },
            "metrics": metrics or manifest.get("metrics") or {},
        },
        "version",
    )

    # A successful HTTP upsert is not enough. Read the exact row back and
    # verify that Supabase contains the same promoted bytes and version.
    row = await store.latest(
        "model_artifacts",
        {
            "select": "*",
            "name": "eq.predictive_brain",
            "version": f"eq.{version}",
            "limit": "1",
        },
    )
    if not row or str(row.get("version")) != version:
        return False

    stored = row.get("artifact") or {}
    if stored.get("encoding") != "joblib-base64":
        return False
    stored_data = base64.b64decode(str(stored.get("data") or ""))
    if hashlib.sha256(stored_data).hexdigest() != artifact_sha256:
        return False
    if stored.get("sha256") != artifact_sha256:
        return False

    stored_bundle = joblib.load(io.BytesIO(stored_data))
    if stored_bundle.get("schema_version") != manifest.get("schema_version"):
        return False
    if stored.get("version") != version:
        return False
    if stored_bundle.get("promotion", {}).get("promoted") is not True:
        return False
    if stored_bundle.get("feature_hash") != manifest.get("feature_hash"):
        return False
    return True


async def hydrate_model():
    """Hydrate only the exact promoted predictive brain persisted in Supabase."""
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
        if artifact.get("sha256") != hashlib.sha256(data).hexdigest():
            return False

        bundle = joblib.load(io.BytesIO(data))
        if bundle.get("schema_version") != 4:
            return False
        if bundle.get("promotion", {}).get("promoted") is not True:
            return False
        if bundle.get("feature_hash") is None or not bundle.get("features"):
            return False

        row_version = str(row.get("version") or "")
        if not row_version or stored.get("version") != row_version:
            return False

        predictive_brain.bundle = bundle
        predictive_brain.version = row_version
        return True
    except Exception:
        return False
