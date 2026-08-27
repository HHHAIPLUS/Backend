from __future__ import annotations
from app.ml.predictive import predictive_model
from app.persistence.supabase import store

async def persist_model(metrics: dict | None = None):
    if not store.configured or predictive_model.model is None:
        return
    artifact = predictive_model.artifact()
    if not artifact:
        return
    await store.upsert('model_artifacts', {
        'name': 'direction_model',
        'version': predictive_model.version,
        'artifact': artifact,
        'metrics': metrics or {},
    }, 'version')

async def hydrate_model():
    if not store.configured:
        return False
    try:
        row = await store.latest('model_artifacts', {'select':'*','order':'created_at.desc','limit':'1'})
        if not row:
            return False
        predictive_model.load_compact_artifact(row['artifact'])
        return True
    except Exception:
        return False
