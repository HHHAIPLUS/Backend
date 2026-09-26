from __future__ import annotations
import json, os, math, time
from datetime import datetime, timezone
import httpx, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from app.ml.features import FEATURES, build_model_features

SYMBOL="BTCUSDT"; INTERVAL="1H"; N=30000; COST=0.0014
FINAL_HOLDOUT=2200; HORIZON=1; GAP=HORIZON
MIN_TRADES=100

def fetch():
    rows=[]; end=None
    with httpx.Client(timeout=30, trust_env=False, headers={"User-Agent":"HHHAI/phase2"}) as c:
        while len(rows)<N:
            p={"category":"USDT-FUTURES","symbol":SYMBOL,"interval":INTERVAL,"limit":200}
            if end is not None: p["endTime"]=end
            data=c.get("https://api.bitget.com/api/v3/market/history-candles",params=p).json().get("data",[])
            if not data: raise RuntimeError("Bitget returned no candles")
            batch=[(int(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])) for r in data if len(r)>=6]
            rows.extend(batch)
            new_end=min(r[0] for r in batch)-1
            if end is not None and new_end>=end: raise RuntimeError("pagination stalled")
            end=new_end
            time.sleep(.05)
    rows=sorted({r[0]:r for r in rows}.values())[-N:]
    if len(rows)!=N: raise RuntimeError(f"expected {N} candles, got {len(rows)}")
    for a,b in zip(rows,rows[1:]):
        if b[0]-a[0] != 3600000: raise RuntimeError("candle gap/overlap")
        if not (a[3] <= min(a[1],a[4]) and a[2] >= max(a[1],a[4]) and a[1]>0 and a[4]>0 and a[5]>=0): raise RuntimeError("invalid OHLCV")
    return rows

def dataset(rows):
    X=[]; y=[]; ret=[]; ts=[]
    for i in range(168,len(rows)-HORIZON):
        window=[{"timestamp":r[0],"open":r[1],"high":r[2],"low":r[3],"close":r[4],"volume":r[5]} for r in rows[i-168:i+1]]
        X.append([build_model_features(window).get(k,0.0) for k in FEATURES])
        r=rows[i+HORIZON][4]/rows[i][4]-1.0
        y.append(1 if r>COST else -1 if r<-COST else 0); ret.append(r); ts.append(rows[i][0])
    return np.asarray(X,float),np.asarray(y),np.asarray(ret,float),np.asarray(ts)

def trade_metrics(y,p,r):
    pos=p!=0
    net=np.where(pos,p*r-COST,0.0)
    trades=int(pos.sum()); total=float(net.sum())
    curve=np.cumsum(net); peak=np.maximum.accumulate(curve); dd=float(np.max(peak-curve)) if len(curve) else 0.0
    avg=float(net[pos].mean()) if trades else 0.0
    wins=int((net[pos]>0).sum()) if trades else 0
    gross_profit=float(net[net>0].sum()); gross_loss=float(-net[net<0].sum())
    return {"trades":trades,"trade_rate":float(pos.mean()),"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"avg_net_return":avg,"total_net_return":total,"max_drawdown":dd,"profit_factor":(gross_profit/gross_loss if gross_loss>0 else None),"wins":wins}

def model(family):
    if family=="logistic":
        return Pipeline([("scale",StandardScaler()),("m",LogisticRegression(max_iter=1800,class_weight="balanced",C=0.5,random_state=42))])
    return RandomForestClassifier(n_estimators=180,min_samples_leaf=12,max_features="sqrt",class_weight="balanced_subsample",random_state=42,n_jobs=1)

def predict_trade(m,X):
    pr=m.predict_proba(X); cls=m.classes_
    out=np.zeros(len(X),dtype=int)
    # fixed, predeclared cost-aware qualification; no final-holdout tuning
    for j,c in enumerate(cls):
        if c==-1: out[pr[:,j]>=0.55]=-1
        if c==1: out[pr[:,j]>=0.55]=1
    # require directional probability to beat flat and cost-aware confidence
    for i in range(len(out)):
        if out[i] != 0:
            j=list(cls).index(out[i])
            flat=j if False else None
            other=np.delete(pr[i],j)
            if pr[i,j] < 0.55 or pr[i,j] <= other.max(): out[i]=0
    return out

def main():
    rows=fetch(); X,y,r,ts=dataset(rows)
    split=len(X)-FINAL_HOLDOUT
    dev_end=split
    # Development walk-forward windows. Candidate selection uses only these windows.
    windows=[(max(0,dev_end-10000),dev_end-1800,1800),(max(0,dev_end-11800),dev_end-3600,1800),(max(0,dev_end-13600),dev_end-5400,1800),(max(0,dev_end-15400),dev_end-7200,1800)]
    scores=[]
    for family in ("logistic","rf"):
        fold=[]
        for a,b,t in windows:
            tr=np.arange(a,b-GAP); te=np.arange(b,min(b+t,dev_end))
            m=model(family); m.fit(X[tr],y[tr]); p=predict_trade(m,X[te])
            fold.append(trade_metrics(y[te],p,r[te]))
        scores.append((family,fold))
    # Frozen selection rule: median total net return, then median drawdown, then median balanced accuracy.
    def key(item):
        fold=item[1]
        return (float(np.median([z["total_net_return"] for z in fold])),
                -float(np.median([z["max_drawdown"] for z in fold])),
                float(np.median([z["balanced_accuracy"] for z in fold])))
    chosen=max(scores,key=key)[0]
    # Freeze architecture/parameters before touching final holdout.
    train=np.arange(0,split-GAP); final=np.arange(split,len(X))
    m=model(chosen); m.fit(X[train],y[train]); p=predict_trade(m,X[final])
    fm=trade_metrics(y[final],p,r[final])
    result={"status":"PASS" if fm["trades"]>=MIN_TRADES and fm["total_net_return"]>0 and fm["avg_net_return"]>0 and fm["max_drawdown"]<=0.15 else "FAIL",
            "data":{"candles":len(rows),"dataset_rows":len(X),"symbol":SYMBOL,"interval":INTERVAL},
            "design":{"horizon":HORIZON,"cost_rate":COST,"gap":GAP,"final_holdout_rows":FINAL_HOLDOUT,"final_holdout_start":datetime.fromtimestamp(ts[split]/1000,tz=timezone.utc).isoformat(),"selection":"development-only walk-forward median economics"},
            "development":{"candidates":[{"family":a,"folds":b} for a,b in scores],"chosen_family":chosen},
            "final_holdout":fm,
            "causal_checks":{"chronological":True,"no_future_features":True,"target_horizon":HORIZON,"holdout_untouched_during_selection":True}}
    os.makedirs("phase2_evidence",exist_ok=True)
    open("phase2_evidence/phase2_adaptive_report.json","w").write(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS": raise SystemExit(1)

if __name__=="__main__": main()
