from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib, json, platform
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.predictive import FEATURES
from app.ml.model_validation import promotion_gate

MODEL_FAMILIES = ("logistic_regression", "extra_trees", "hist_gradient_boosting")
HORIZONS = (1, 3, 6, 12)
COST_RATE = 0.0008
ARTIFACT_SCHEMA = 2

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
    x=np.asarray([[float(r.get("features", {}).get(k, 0.0) or 0.0) for k in FEATURES] for r in rows],dtype=float)
    if not np.isfinite(x).all(): raise ValueError("Predictive features contain non-finite values.")
    return x

def _future_return(rows,horizon):
    values=[]
    for row in rows:
        by=row.get("outcome_return_by_horizon",{})
        value=by.get(str(horizon),by.get(horizon))
        if value is None and horizon==int(row.get("outcome_horizon",6)): value=row.get("outcome_return")
        if value is None: raise ValueError(f"Missing point-in-time outcome for horizon {horizon}")
        values.append(float(value))
    result=np.asarray(values,dtype=float)
    if not np.isfinite(result).all(): raise ValueError("Outcome returns contain non-finite values.")
    return result

def _direction_target(values,threshold=COST_RATE):
    return np.where(values>threshold,1,np.where(values<-threshold,-1,0))

def _classifier(family):
    if family=="logistic_regression": return Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42))])
    if family=="extra_trees": return ExtraTreesClassifier(n_estimators=120,min_samples_leaf=10,class_weight="balanced",random_state=42,n_jobs=-1)
    if family=="hist_gradient_boosting": return HistGradientBoostingClassifier(max_iter=180,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    raise ValueError(f"Unknown model family: {family}")

def _regressor(family):
    if family in ("logistic_regression","extra_trees"): return ExtraTreesRegressor(n_estimators=120,min_samples_leaf=10,random_state=42,n_jobs=-1)
    if family=="hist_gradient_boosting": return HistGradientBoostingRegressor(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    raise ValueError(f"Unknown model family: {family}")

def _net_returns(returns,pred):
    traded=pred!=0
    return returns*np.where(pred==1,1.0,np.where(pred==-1,-1.0,0.0))-np.where(traded,COST_RATE,0.0)

def _metrics(y,pred,probs,classes,returns):
    net=_net_returns(returns,pred); traded=pred!=0; mapping={int(c):i for i,c in enumerate(classes)}
    if all(c in mapping for c in (-1,0,1)):
        ordered=np.column_stack([probs[:,mapping[-1]],probs[:,mapping[0]],probs[:,mapping[1]]]); truth=np.column_stack([(y==-1),(y==0),(y==1)]).astype(float); brier=float(np.mean(np.sum((ordered-truth)**2,axis=1)))
    else: brier=float("nan")
    equity=np.cumsum(net); peak=np.maximum.accumulate(np.r_[0.0,equity]); dd=float(np.max(peak[1:]-equity)) if len(equity) else 0.0
    side={}
    for label,name in ((-1,"short"),(1,"long")):
        mask=pred==label; side[name]={"samples":int(mask.sum()),"precision":float(precision_score(y[mask],pred[mask],labels=[label],average="micro",zero_division=0)) if mask.any() else 0.0,"avg_net_return":float(net[mask].mean()) if mask.any() else 0.0}
    return {"samples":int(len(y)),"trades":int(traded.sum()),"trade_rate":float(traded.mean()),"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"precision_macro":float(precision_score(y,pred,average="macro",zero_division=0)),"recall_macro":float(recall_score(y,pred,average="macro",zero_division=0)),"avg_net_return":float(net.mean()),"avg_trade_net_return":float(net[traded].mean()) if traded.any() else 0.0,"total_net_return":float(net.sum()),"max_drawdown":dd,"calibration_brier":brier,"mean_confidence":float(np.max(probs,axis=1).mean()),"long":side["long"],"short":side["short"]}

def _calibrate(model,x_cal,y_cal):
    return CalibratedClassifierCV(FrozenEstimator(model),method="sigmoid").fit(x_cal,y_cal)

class PredictiveBrain:
    """Predictive brain with honest baseline, untouched OOS promotion, calibration, meta-learning and abstention."""
    def __init__(self,artifact_dir="artifacts"):
        self.path=Path(artifact_dir); self.path.mkdir(parents=True,exist_ok=True); self.artifact_path=self.path/"predictive_brain.joblib"; self.manifest_path=self.path/"predictive_brain_manifest.json"; self.bundle=None; self.version="untrained"; self._load()
    def _load(self):
        if not self.artifact_path.exists() or not self.manifest_path.exists(): return
        try:
            m=json.loads(self.manifest_path.read_text()); b=joblib.load(self.artifact_path)
            if m.get("schema_version")!=ARTIFACT_SCHEMA or m.get("features")!=FEATURES or b.get("feature_hash")!=_feature_hash(): raise ValueError("Predictive brain artifact schema or feature fingerprint mismatch")
            if m.get("promotion",{}).get("promoted") is not True: raise ValueError("Artifact is not a promoted candidate")
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
        if len(rows)<800: return BrainReport("REJECTED",version,{"rows":len(rows)},"At least 800 point-in-time rows are required for independent selection, calibration and OOS testing.")
        rows=sorted(rows,key=lambda r:str(r.get("observed_at",""))); x=_x(rows); n=len(rows)
        # Split by unique timestamps, never by arbitrary rows. This is required for
        # multi-symbol datasets so candles from the same market time cannot straddle
        # train and OOS partitions.
        timestamps=[str(r.get("observed_at","")) for r in rows]
        unique_times=sorted(set(timestamps))
        if len(unique_times)<20: return BrainReport("REJECTED",version,{},"Not enough unique timestamps for chronological evaluation.")
        cutoff_pos=max(1,min(len(unique_times)-1,int(len(unique_times)*(1-test_fraction))))
        cutoff_time=unique_times[cutoff_pos]
        test_start=next(i for i,t in enumerate(timestamps) if t>=cutoff_time)
        pre=x[:test_start]; xte=x[test_start:]
        if len(xte)<100 or len(pre)<600: return BrainReport("REJECTED",version,{},"Chronological train/validation/calibration/test partitions are too small.")
        pre_times=sorted(set(timestamps[:test_start]))
        select_cutoff=pre_times[max(1,min(len(pre_times)-1,int(len(pre_times)*.75)))]
        select_end=next(i for i,t in enumerate(timestamps[:test_start]) if t>=select_cutoff)
        xfit,xval=pre[:select_end],pre[select_end:]
        horizon_selection={}
        for h in HORIZONS:
            try:
                rh=_future_return(rows[:test_start],h); yh=_direction_target(rh); yhfit,yhval=yh[:select_end],yh[select_end:]
                if len(yhval)<100 or len(set(yhfit.tolist()))<3 or len(set(yhval.tolist()))<3: continue
                hm=_classifier("logistic_regression"); hm.fit(xfit,yhfit); hp=hm.predict(xval); hpr=hm.predict_proba(xval); hs=_metrics(yhval,hp,hpr,hm.classes_,rh[select_end:]); horizon_selection[str(h)] = hs
            except Exception as exc: horizon_selection[str(h)]={"error":f"{type(exc).__name__}: {exc}"}
        viable=[(float(v.get("avg_trade_net_return",-1e99)),float(v.get("balanced_accuracy",0.0)),int(h)) for h,v in horizon_selection.items() if "error" not in v and int(v.get("trades",0))>=100]
        if not viable: return BrainReport("REJECTED",version,{"horizon_selection":horizon_selection}, "No horizon produced enough validation trades for selection.")
        chosen_horizon=max(viable,key=lambda z:(z[0],z[1]))[2]
        returns=_future_return(rows,chosen_horizon); d=_direction_target(returns)
        pre_r=returns[:test_start]; rte=returns[test_start:]; pre_y=d[:test_start]; dte=d[test_start:]
        if len(set(dte.tolist()))<3: return BrainReport("REJECTED",version,{"chosen_horizon":chosen_horizon}, "Untouched OOS test period must contain all three direction classes.")
        xfit,xval=pre[:select_end],pre[select_end:]; yfit,yval=pre_y[:select_end],pre_y[select_end:]
        if len(xval)<100 or len(set(yfit.tolist()))<3 or len(set(yval.tolist()))<3: return BrainReport("REJECTED",version,{},"Model-selection validation partition is insufficient.")
        validation_scores={}; candidates=[]
        for family in MODEL_FAMILIES:
            try:
                m=_classifier(family); m.fit(xfit,yfit); p=m.predict(xval); pr=m.predict_proba(xval); s=_metrics(yval,p,pr,m.classes_,pre_r[select_end:]); validation_scores[family]=s; candidates.append((s["avg_net_return"],s["balanced_accuracy"],family))
            except Exception as exc: validation_scores[family]={"error":f"{type(exc).__name__}: {exc}"}
        base=validation_scores.get("logistic_regression")
        if not base or not candidates: return BrainReport("REJECTED",version,{"validation_families":validation_scores},"Complete model-family evaluation was not possible.")
        best=max(candidates,key=lambda c:(c[0],c[1])); family=best[2]
        if family != "logistic_regression" and (best[0] <= float(base.get("avg_net_return",-1e99)) or best[1] < float(base.get("balanced_accuracy",0.0))):
            family="logistic_regression"
        cal_start=select_end; x_model=pre[:cal_start]; y_model=pre_y[:cal_start]; x_cal=pre[cal_start:]; y_cal=pre_y[cal_start:]
        if len(x_cal)<100 or len(set(y_model.tolist()))<3 or len(set(y_cal.tolist()))<3: return BrainReport("REJECTED",version,{},"Calibration partition is insufficient.")
        direction_raw=_classifier(family); direction_raw.fit(x_model,y_model); direction=_calibrate(direction_raw,x_cal,y_cal)
        baseline_raw=_classifier("logistic_regression"); baseline_raw.fit(x_model,y_model); baseline=_calibrate(baseline_raw,x_cal,y_cal)
        cal_pred=direction.predict(x_cal); cal_prob=direction.predict_proba(x_cal)
        confidence=np.max(cal_prob,axis=1)
        selection_threshold=0.55; selection_score=-float("inf")
        for threshold in np.arange(0.45,0.81,0.02):
            selected=cal_pred.copy(); selected[confidence < threshold]=0
            traded=selected!=0
            if int(traded.sum())<100: continue
            net=_net_returns(y_cal*0.0 + returns[cal_start:test_start],selected)
            score=float(net.sum()) - 0.5*float(np.max(np.maximum.accumulate(np.r_[0.0,np.cumsum(net)])[1:]-np.cumsum(net)))
            if score>selection_score: selection_score=score; selection_threshold=float(threshold)
        candidate_pred=direction.predict(xte); candidate_prob=direction.predict_proba(xte)
        candidate_pred[candidate_prob.max(axis=1) < selection_threshold]=0
        baseline_pred=baseline.predict(xte); baseline_prob=baseline.predict_proba(xte)
        candidate_metrics=_metrics(dte,candidate_pred,candidate_prob,direction.classes_,rte); baseline_metrics=_metrics(dte,baseline_pred,baseline_prob,baseline.classes_,rte)
        er=_regressor(family); er.fit(pre,pre_r); dn=_regressor(family); dn.fit(pre,np.minimum(pre_r,0)); vol=_regressor(family); vol.fit(pre,np.abs(pre_r)); rv=np.abs(pre_r); rq=np.quantile(rv,[.33,.66]); regime_target=np.where(rv>rq[1],2,np.where(rv>rq[0],1,0)); rm=_classifier(family); rm.fit(pre,regime_target); am=HistGradientBoostingClassifier(max_iter=150,random_state=46).fit(pre,(np.abs(pre_r)<=COST_RATE).astype(int))
        meta_rows=[]; meta_y=[]; starts=max(250,len(pre)//3); step=max(100,(len(pre)-starts)//3)
        for end in range(starts,len(pre),step):
            stop=min(end+step,len(pre))
            if stop<=end: continue
            fm=_classifier(family); fm.fit(pre[:end],pre_y[:end]); fr=_regressor(family); fr.fit(pre[:end],pre_r[:end]); fd=_regressor(family); fd.fit(pre[:end],np.minimum(pre_r[:end],0)); fv=_regressor(family); fv.fit(pre[:end],np.abs(pre_r[:end]))
            meta_rows.append(np.column_stack([fm.predict_proba(pre[end:stop]),fr.predict(pre[end:stop]),fd.predict(pre[end:stop]),fv.predict(pre[end:stop])]))
            meta_y.extend(pre_y[end:stop].tolist())
        if not meta_rows or sum(len(a) for a in meta_rows)<100: return BrainReport("REJECTED",version,{},"Insufficient true out-of-fold meta-training samples.")
        meta_x=np.vstack(meta_rows); meta_y=np.asarray(meta_y,dtype=int); mm=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42); mm.fit(meta_x,meta_y)
        fm_oos=_classifier(family); fm_oos.fit(pre,pre_y); fr_oos=_regressor(family); fr_oos.fit(pre,pre_r); fd_oos=_regressor(family); fd_oos.fit(pre,np.minimum(pre_r,0)); fv_oos=_regressor(family); fv_oos.fit(pre,np.abs(pre_r))
        meta_oos=np.column_stack([fm_oos.predict_proba(xte),fr_oos.predict(xte),fd_oos.predict(xte),fv_oos.predict(xte)])
        candidate_prob=mm.predict_proba(meta_oos); candidate_pred=mm.predict(meta_oos); candidate_pred[candidate_prob.max(axis=1)<selection_threshold]=0
        candidate_metrics=_metrics(dte,candidate_pred,candidate_prob,mm.classes_,rte)
        gate=promotion_gate(_net_returns(rte,candidate_pred),_net_returns(rte,baseline_pred),candidate_metrics["balanced_accuracy"],baseline_metrics["balanced_accuracy"],candidate_metrics["max_drawdown"],baseline_metrics["max_drawdown"])
        absolute_gate={"enough_samples":candidate_metrics["trades"]>=100,"accuracy_ok":candidate_metrics["accuracy"]>=0.52,"balanced_accuracy_ok":candidate_metrics["balanced_accuracy"]>=0.50,"positive_trade_expectancy":candidate_metrics["avg_trade_net_return"]>0.0,"positive_total_net_return":candidate_metrics["total_net_return"]>0.0,"drawdown_ok":candidate_metrics["max_drawdown"]<=0.15}
        if not all(absolute_gate.values()): return BrainReport("REJECTED",version,{"validation_families":validation_scores,"baseline_oos":baseline_metrics,"candidate_oos":candidate_metrics,"promotion":gate,"absolute_gate":absolute_gate},"Candidate did not clear the untouched OOS absolute safety gate.")
        horizon_metrics=self._horizon_eval(pre,xte,rows[:test_start],rows[test_start:])
        bundle={"schema_version":ARTIFACT_SCHEMA,"direction_model":direction,"baseline_model":baseline,"expected_return_model":er,"downside_model":dn,"volatility_model":vol,"regime_model":rm,"abstention_model":am,"meta_model":mm,"family":family,"decision_threshold":selection_threshold,"feature_hash":_feature_hash(),"features":FEATURES,"cost_rate":COST_RATE,"horizons":HORIZONS,"chosen_horizon":chosen_horizon,"horizon_selection":horizon_selection,"horizon_metrics":horizon_metrics,"oos_metrics":{"candidate":candidate_metrics,"baseline":baseline_metrics},"promotion":gate,"sequence_model_evaluation":{"status":"NOT_REQUIRED","reason":"Canonical Stage 3 data is tabular and the current sample/coverage does not justify sequence-model complexity; revisit when temporal sequence coverage and sample volume materially increase."}}
        tmp=self.artifact_path.with_suffix(".tmp"); joblib.dump(bundle,tmp); tmp.replace(self.artifact_path); manifest={"schema_version":ARTIFACT_SCHEMA,"version":version,"features":FEATURES,"feature_hash":_feature_hash(),"family":family,"metrics":{"validation_families":validation_scores,"candidate_oos":candidate_metrics,"baseline_oos":baseline_metrics,"horizons":horizon_metrics},"promotion":gate,"cost_rate":COST_RATE,"horizons":HORIZONS,"chosen_horizon":chosen_horizon,"horizon_selection":horizon_selection,"python":platform.python_version()}; self.manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)); self.bundle=bundle; self.version=version
        return BrainReport("PROMOTED",version,manifest["metrics"],"Candidate cleared independent selection, calibrated untouched OOS evaluation and paired statistical promotion gates.",str(self.artifact_path))
    def predict(self,features):
        if self.bundle is None: return {"trained":False,"abstain":True,"version":self.version,"decision":"NO_TRADE","reason":"No promoted predictive brain artifact is available."}
        x=np.asarray([[float(features.get(k,0.0) or 0.0) for k in FEATURES]],dtype=float)
        if not np.isfinite(x).all(): return {"trained":True,"abstain":True,"version":self.version,"decision":"NO_TRADE","reason":"Non-finite predictive features."}
        dm=self.bundle["direction_model"]; bp=dm.predict_proba(x)[0]; er=float(self.bundle["expected_return_model"].predict(x)[0]); dn=max(0.0,float(self.bundle["downside_model"].predict(x)[0])); vo=max(0.0,float(self.bundle["volatility_model"].predict(x)[0])); rg=int(self.bundle["regime_model"].predict(x)[0]); am=float(self.bundle["abstention_model"].predict_proba(x)[0][1]); mf=np.concatenate([bp,np.asarray([er,dn,vo])])[None,:]; mp=self.bundle["meta_model"].predict_proba(mf)[0]; mc=list(self.bundle["meta_model"].classes_); probs={"short":float(mp[mc.index(-1)]) if -1 in mc else 0.0,"flat":float(mp[mc.index(0)]) if 0 in mc else 0.0,"long":float(mp[mc.index(1)]) if 1 in mc else 0.0}; direction=max(probs,key=probs.get); edge=er-self.bundle["cost_rate"]; threshold=float(self.bundle.get("decision_threshold",0.55)); uncertainty=float(1-max(probs.values())); abstain=am>=.60 or direction=="flat" or max(probs.values())<threshold or edge<=0 or not np.isfinite([er,dn,vo]).all(); return {"trained":True,"abstain":abstain,"version":self.version,"decision":"NO_TRADE" if abstain else direction.upper(),"probabilities":probs,"expected_return":er,"expected_edge_after_cost":edge,"downside":dn,"volatility":vo,"regime":rg,"uncertainty":uncertainty,"abstention_probability":am,"model_family":self.bundle["family"]}
    def manifest(self):
        if not self.manifest_path.exists(): return None
        try: return json.loads(self.manifest_path.read_text())
        except Exception: return None


predictive_brain = PredictiveBrain()
