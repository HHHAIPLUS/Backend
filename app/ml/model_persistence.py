from __future__ import annotations
import base64
import io
import joblib
from app.ml.predictive import predictive_model
from app.ml.predictive_brain import predictive_brain
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

async def persist_brain(metrics: dict | None = None):
    if not store.configured or predictive_brain.bundle is None:
        return False
    buffer=io.BytesIO()
    joblib.dump(predictive_brain.bundle, buffer)
    encoded=base64.b64encode(buffer.getvalue()).decode("ascii")
    manifest=predictive_brain.manifest() or {}
    await store.upsert('model_artifacts', {
        'name':'predictive_brain',
        'version':predictive_brain.version,
        'artifact':{'encoding':'joblib-base64','data':encoded,'schema_version':manifest.get('schema_version')},
        'metrics':metrics or manifest.get('metrics') or {},
    }, 'version')
    return True

async def hydrate_model():
    if not store.configured:
        return False
    try:
        row=await store.latest('model_artifacts', {'select':'*','name':'eq.direction_model','order':'created_at.desc','limit':'1'})
        if row:
            predictive_model.load_compact_artifact(row['artifact'])
    except Exception:
        pass
    try:
        row=await store.latest('model_artifacts', {'select':'*','name':'eq.predictive_brain','order':'created_at.desc','limit':'1'})
        if not row:
            return False
        artifact=row.get('artifact') or {}
        if artifact.get('encoding')!='joblib-base64':
            return False
        data=base64.b64decode(str(artifact['data']))
        bundle=joblib.load(io.BytesIO(data))
        if bundle.get('feature_hash') is None or bundle.get('features') != predictive_brain.bundle.get('features') if predictive_brain.bundle else False:
            pass
        predictive_brain.bundle=bundle
        predictive_brain.version=str(row.get('version') or bundle.get('version') or 'persisted')
        return True
    except Exception:
        return False
