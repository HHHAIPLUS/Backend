from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib, json, platform
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.predictive import FEATURES

MODEL_FAMILIES = ("logistic_regression", "extra_trees", "hist_gradient_boosting")
HORIZONS = (1, 3, 6, 12)
COST_RATE = 0.0008

@dataclass
class BrainReport:
    status: str
    version: str
    metrics: dict[str, Any]
    reason: str
    artifact: str | None = None

def _feature_hash() -> str:
    return hashlib.sha256("|".join(FEATURES).encode()).hexdigest()

def _x(rows):
    return np.asarray([[float(r.get("features", {}).get(k, 0.0) or 0.0) for k in FEATURES] for r in rows], dtype=float)

def _future_return(rows, horizon):
    values=[]
    for row in rows:
        by=row.get("outcome_return_by_horizon", {})
        value=by.get(str(horizon), by.get(horizon))
        if value is None and horizon == int(row.get("outcome_horizon", 6)): value=row.get("outcome_return")
        if value is None: raise ValueError(f"Missing point-in-time outcome for horizon {horizon}")
        values.append(float(value))
    return np.asarray(values, dtype=float)

def _direction_target(values, threshold=COST_RATE):
    return np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))

def _classifier(family):
    if family == "logistic_regression": return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])
    if family == "extra_trees": return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    if family == "hist_gradient_boosting": return HistGradientBoostingClassifier(max_iter=250, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    raise ValueError(f"Unknown model family: {family}")

def _regressor(family):
    if family in ("logistic_regression", "extra_trees"): return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if family == "hist_gradient_boosting": return HistGradientBoostingRegressor(max_iter=250, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    raise ValueError(f"Unknown model family: {family}")

def _metrics(y, pred, probs, classes, returns):
    traded=pred != 0
    net=returns*np.where(pred==1,1.0,np.where(pred==-1,-1.0,0.0))-np.where(traded,COST_RATE,0.0)
    mapping={int(c):i for i,c in enumerate(classes)}
    if all(c in mapping for c in (-1,0,1)):
        ordered=np.column_stack([probs[:,mapping[-1]],probs[:,mapping[0]],probs[:,mapping[1]]]); truth=np.column_stack([(y==-1),(y==0),(y==1)]).astype(float)
        brier=float(np.mean(np.sum((ordered-truth)**2,axis=1)))
    else: brier=float("nan")
    equity=np.cumsum(net); peak=np.maximum.accumulate(np.r_[0.0,equity]); dd=float(np.max(peak[1:]-equity)) if len(equity) else 0.0
    return {"samples":int(len(y)),"trades":int(traded.sum()),"trade_rate":float(traded.mean()),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"precision_macro":float(precision_score(y,pred,average="macro",zero_division=0)),"recall_macro":float(recall_score(y,pred,average="macro",zero_division=0)),"avg_net_return":float(net.mean()),"avg_trade_net_return":float(net[traded].mean()) if traded.any() else 0.0,"total_net_return":float(net.sum()),"max_drawdown":dd,"calibration_brier":brier,"mean_confidence":float(np.max(probs,axis=1).mean())}

class PredictiveBrain:
    """Multi-model predictive ensemble with baseline comparison, uncertainty and abstention."""
    def __init__(self, artifact_dir="artifacts"):
        self.path=Path(artifact_dir); self.path.mkdir(parents=True,exist_ok=True); self.artifact_path=self.path/"predictive_brain.joblib"; self.manifest_path=self.path/"predictive_brain_manifest.json"; self.bundle=None; self.version="untrained"; self._load()
    def _load(self):
        if not self.artifact_path.exists() or not self.manifest_path.exists(): return
        try:
            m=json.loads(self.manifest_path.read_text()); b=joblib.load(self.artifact_path)
            if m.get("features")!=FEATURES or b.get("feature_hash")!=_feature_hash(): return
            self.bundle=b; self.version=str(m["version"])
        except Exception: self.bundle=None; self.version="untrained"
    def _horizon_eval(self,xtr,xte,rtr,rte):
        out={}
        for h in HORIZONS:
            try:
                a=_future_return(rtr,h); b=_future_return(rte,h); ya=_direction_target(a); yb=_direction_target(b)
                if len(set(ya.tolist()))<3 or len(set(yb.tolist()))<3: raise ValueError("three classes required")
                m=_classifier("logistic_regression"); m.fit(xtr,ya); out[str(h)]=_metrics(yb,m.predict(xte),m.predict_proba(xte),m.classes_,b)
            except Exception as exc: out[str(h)]={"status":"UNAVAILABLE","reason":str(exc)}
        return out
    def train(self,rows,version="brain-v1",test_fraction=.2):
        if len(rows)<600: return BrainReport("REJECTED",version,{"rows":len(rows)},"At least 600 point-in-time rows are required.")
        rows=sorted(rows,key=lambda r:str(r.get("observed_at",""))); x=_x(rows); y=_future_return(rows,6); d=_direction_target(y); split=int(len(rows)*(1-test_fraction))
        if split<300 or len(rows)-split<100: return BrainReport("REJECTED",version,{},"Chronological train/test split is too small.")
        xtr,xte=x[:split],x[split:]; dtr,dte=d[:split],d[split:]; rtr,rte=y[:split],y[split:]
        if len(set(dtr.tolist()))<3 or len(set(dte.tolist()))<3: return BrainReport("REJECTED",version,{},"All three direction classes are required in train and test.")
        scores={}; candidates=[]
        for family in MODEL_FAMILIES:
            try:
                m=_classifier(family); m.fit(xtr,dtr); p=m.predict(xte); pr=m.predict_proba(xte); s=_metrics(dte,p,pr,m.classes_,rte); scores[family]=s; candidates.append((s["avg_net_return"],s["balanced_accuracy"],family,m))
            except Exception as exc: scores[family]={"error":f"{type(exc).__name__}: {exc}"}
        base=scores.get("logistic_regression"); complex_models=[c for c in candidates if c[2]!="logistic_regression"]
        if not base or not complex_models: return BrainReport("REJECTED",version,{"families":scores},"Complete model-family evaluation was not possible.")
        best=max(complex_models,key=lambda c:(c[0],c[1]))
        if best[0]<=float(base.get("avg_net_return",-1e99)) or best[1]<=float(base.get("balanced_accuracy",0.0)):
            return BrainReport("REJECTED",version,{"families":scores,"best_candidate":best[2]},"Complex model did not beat the honest logistic baseline on both net return and balanced accuracy.")
        family,best_model=best[2],best[3]
        er=_regressor(family); er.fit(xtr,rtr); dn=_regressor(family); dn.fit(xtr,np.minimum(rtr,0)); vol=_regressor(family); vol.fit(xtr,np.abs(rtr))
        rv=np.abs(rtr); rt=np.where(rv>np.quantile(rv,.66),2,np.where(rv>np.quantile(rv,.33),1,0)); rm=_classifier(family); rm.fit(xtr,rt)
        meta_x=np.full((len(xtr),6),np.nan); start=max(150,len(xtr)//3); step=max(50,(len(xtr)-start)//3)
        for end in range(start,len(xtr),step):
            stop=min(end+step,len(xtr)); fm=_classifier(family); fm.fit(xtr[:end],dtr[:end]); fe=_regressor(family); fe.fit(xtr[:end],rtr[:end]); fd=_regressor(family); fd.fit(xtr[:end],np.minimum(rtr[:end],0)); fv=_regressor(family); fv.fit(xtr[:end],np.abs(rtr[:end])); meta_x[end:stop]=np.column_stack([fm.predict_proba(xtr[end:stop]),fe.predict(xtr[end:stop]),fd.predict(xtr[end:stop]),fv.predict(xtr[end:stop])])
        valid=np.isfinite(meta_x).all(axis=1)
        if valid.sum()<100: return BrainReport("REJECTED",version,{"families":scores},"Insufficient out-of-fold meta-training samples.")
        mm=LogisticRegression(max_iter=1000,random_state=42); mm.fit(meta_x[valid],dtr[valid]); hm=self._horizon_eval(xtr,xte,rows[:split],rows[split:])
        bundle={"direction_model":best_model,"expected_return_model":er,"downside_model":dn,"volatility_model":vol,"regime_model":rm,"meta_model":mm,"family":family,"feature_hash":_feature_hash(),"features":FEATURES,"cost_rate":COST_RATE,"horizons":HORIZONS,"horizon_metrics":hm,"sequence_model_evaluation":{"status":"NOT_REQUIRED","reason":"Current canonical dataset is tabular; sequence modeling is deferred until sequence coverage and sample volume justify it."}}
        tmp=self.artifact_path.with_suffix(".tmp"); joblib.dump(bundle,tmp); tmp.replace(self.artifact_path); manifest={"version":version,"features":FEATURES,"feature_hash":_feature_hash(),"family":family,"metrics":{"families":scores,"promoted":family,"horizons":hm},"cost_rate":COST_RATE,"horizons":HORIZONS,"python":platform.python_version()}; self.manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)); self.bundle=bundle; self.version=version
        return BrainReport("PROMOTED",version,manifest["metrics"],"Ensemble beat the logistic baseline on chronological out-of-sample net return and balanced accuracy.",str(self.artifact_path))
    def predict(self,features):
        if self.bundle is None: return {"trained":False,"abstain":True,"version":self.version,"decision":"NO_TRADE","reason":"No promoted predictive brain artifact is available."}
        x=np.asarray([[float(features.get(k,0.0) or 0.0) for k in FEATURES]],dtype=float); dm=self.bundle["direction_model"]; bp=dm.predict_proba(x)[0]; er=float(self.bundle["expected_return_model"].predict(x)[0]); dn=float(self.bundle["downside_model"].predict(x)[0]); vo=float(self.bundle["volatility_model"].predict(x)[0]); rg=int(self.bundle["regime_model"].predict(x)[0]); mf=np.concatenate([bp,np.asarray([er,dn,vo])])[None,:]; mp=self.bundle["meta_model"].predict_proba(mf)[0]; mc=list(self.bundle["meta_model"].classes_); probs={"short":float(mp[mc.index(-1)]) if -1 in mc else 0.0,"flat":float(mp[mc.index(0)]) if 0 in mc else 0.0,"long":float(mp[mc.index(1)]) if 1 in mc else 0.0}; direction=max(probs,key=probs.get); edge=er-self.bundle["cost_rate"]; uncertainty=1-max(probs.values()); abstain=direction=="flat" or max(probs.values())<.55 or edge<=0 or not np.isfinite([er,dn,vo]).all(); return {"trained":True,"abstain":abstain,"version":self.version,"decision":"NO_TRADE" if abstain else direction.upper(),"probabilities":probs,"expected_return":er,"expected_edge_after_cost":edge,"downside":dn,"volatility":vo,"regime":rg,"uncertainty":float(uncertainty),"model_family":self.bundle["family"]}
    def manifest(self):
        if not self.manifest_path.exists(): return None
        try: return json.loads(self.manifest_path.read_text())
        except Exception: return None
