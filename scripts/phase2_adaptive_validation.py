from __future__ import annotations
import json, os, math, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone
import httpx, numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from app.ml.features import FEATURES, build_model_features

SYMBOL="BTCUSDT"; INTERVAL="1H"; N=30000; COST=0.0014
FINAL_HOLDOUT=2200; HORIZON=1; GAP=HORIZON
MIN_TRADES=100

def fetch():
    rows=[]; end_ms=int(time.time()*1000)//3600000*3600000
    with httpx.Client(timeout=30, trust_env=False, headers={"User-Agent":"HHHAI/phase2"}) as client:
        while len(rows)<N:
            start_ms=end_ms-(199*3600000)
            p={"symbol":SYMBOL,"productType":"USDT-FUTURES","granularity":INTERVAL,"limit":200,"startTime":start_ms,"endTime":end_ms}
            resp=client.get("https://api.bitget.com/api/v2/mix/market/history-candles",params=p)
            resp.raise_for_status(); payload=resp.json()
            if payload.get("code") not in (None,"00000"): raise RuntimeError(f"Bitget candles error: {payload}")
            data=payload.get("data",[])
            if not data: raise RuntimeError(f"Bitget returned no historical candles for {start_ms}..{end_ms}; payload={payload}")
            batch=[(int(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])) for r in data if len(r)>=6]
            rows.extend(batch)
            oldest=min(r[0] for r in batch)
            if oldest>=end_ms: raise RuntimeError("candle pagination stalled")
            end_ms=oldest
            time.sleep(.06)
    rows=sorted({r[0]:r for r in rows}.values())[-N:]
    if len(rows)!=N: raise RuntimeError(f"expected {N} candles, got {len(rows)}")
    for a,b in zip(rows,rows[1:]):
        if b[0]-a[0] != 3600000: raise RuntimeError(f"candle gap/overlap at {a[0]}->{b[0]} delta={b[0]-a[0]}")
        if not (a[3] <= min(a[1],a[4]) and a[2] >= max(a[1],a[4]) and a[1]>0 and a[4]>0 and a[5]>=0): raise RuntimeError("invalid OHLCV")
    return rows

HORIZONS=(1,3,6,12)
FINAL_HOLDOUT=2200
MAX_HORIZON=max(HORIZONS)

def dataset(rows):
    X=[]; returns={h:[] for h in HORIZONS}; ts=[]
    for i in range(168,len(rows)-MAX_HORIZON):
        window=[{"timestamp":r[0],"open":r[1],"high":r[2],"low":r[3],"close":r[4],"volume":r[5]} for r in rows[i-168:i+1]]
        X.append([build_model_features(window).get(k,0.0) for k in FEATURES])
        for h in HORIZONS:
            returns[h].append(rows[i+h][4]/rows[i][4]-1.0)
        ts.append(rows[i][0])
    return np.asarray(X,float),{h:np.asarray(v,float) for h,v in returns.items()},np.asarray(ts)

def trade_metrics(y,p,r):
    pos=p!=0
    net=np.where(pos,p*r-COST,0.0)
    trades=int(pos.sum()); total=float(net.sum())
    curve=np.cumsum(net); peak=np.maximum.accumulate(curve); dd=float(np.max(peak-curve)) if len(curve) else 0.0
    avg=float(net[pos].mean()) if trades else 0.0
    wins=int((net[pos]>0).sum()) if trades else 0
    gross_profit=float(net[net>0].sum()); gross_loss=float(-net[net<0].sum())
    return {"trades":trades,"trade_rate":float(pos.mean()),"accuracy":float(accuracy_score(y,p)),
            "balanced_accuracy":float(balanced_accuracy_score(y,p)),"avg_net_return":avg,
            "total_net_return":total,"max_drawdown":dd,
            "profit_factor":(gross_profit/gross_loss if gross_loss>0 else None),"wins":wins}

def model(family):
    if family=="logistic":
        return Pipeline([("scale",StandardScaler()),("m",LogisticRegression(max_iter=1800,class_weight="balanced",C=0.5,random_state=42))])
    if family=="ridge":
        return Pipeline([("scale",StandardScaler()),("m",Ridge(alpha=1.0))])
    raise ValueError(f"unknown model family: {family}")

def classifier_predictions(m,X,threshold):
    pr=m.predict_proba(X); cls=m.classes_
    out=np.zeros(len(X),dtype=int)
    for j,c in enumerate(cls):
        if c==-1: out[pr[:,j]>=threshold]=-1
        elif c==1: out[pr[:,j]>=threshold]=1
    for i in range(len(out)):
        if out[i] != 0:
            j=list(cls).index(out[i])
            if pr[i,j] < threshold or pr[i,j] <= np.delete(pr[i],j).max(): out[i]=0
    return out

def ridge_predictions(m,X,threshold):
    pred=np.asarray(m.predict(X),dtype=float)
    return np.where(pred>=threshold,1,np.where(pred<=-threshold,-1,0)).astype(int)

def momentum_predictions(X,threshold):
    idx=FEATURES.index("momentum")
    score=X[:,idx]
    return np.where(score>=threshold,1,np.where(score<=-threshold,-1,0)).astype(int)

def main():
    rows=fetch()
    X,returns,ts=dataset(rows)
    split=len(X)-FINAL_HOLDOUT
    dev_end=split
    windows=[
        (max(0,dev_end-10000),dev_end-1800,1800),
        (max(0,dev_end-11800),dev_end-3600,1800),
        (max(0,dev_end-13600),dev_end-5400,1800),
        (max(0,dev_end-15400),dev_end-7200,1800),
    ]
    MIN_DEVELOPMENT_TRADES=100
    candidates=[]
    for horizon in HORIZONS:
        r=returns[horizon]
        y=np.where(r>COST,1,np.where(r<-COST,-1,0)).astype(int)
        specs=[
            ("logistic","classifier",(0.35,0.40,0.45,0.50,0.55)),
            ("ridge","ridge",(0.0015,0.0020,0.0025,0.0030,0.0040)),
            ("momentum","momentum",(0.05,0.10,0.15,0.20,0.25)),
        ]
        for family,kind,thresholds in specs:
            fold_by_threshold={float(t):[] for t in thresholds}
            for a,b,t in windows:
                tr=np.arange(a,b-HORIZON if horizon==1 else a) if False else np.arange(a,b-horizon)
                te=np.arange(b,min(b+t,dev_end))
                if len(tr)<500 or len(te)==0: continue
                if kind=="momentum":
                    for th in thresholds:
                        p=momentum_predictions(X[te],float(th))
                        fold_by_threshold[float(th)].append(trade_metrics(y[te],p,r[te]))
                else:
                    m=model(family)
                    m.fit(X[tr],y[tr] if kind=="classifier" else r[tr])
                    for th in thresholds:
                        if kind=="classifier":
                            p=classifier_predictions(m,X[te],float(th))
                        else:
                            p=ridge_predictions(m,X[te],float(th))
                        fold_by_threshold[float(th)].append(trade_metrics(y[te],p,r[te]))
            for th,fold in fold_by_threshold.items():
                eligible=len(fold)==len(windows) and min(z["trades"] for z in fold)>=MIN_DEVELOPMENT_TRADES
                candidates.append({"horizon":horizon,"family":family,"threshold":th,"folds":fold,"eligible":eligible})
    eligible=[x for x in candidates if x["eligible"]]
    if not eligible:
        raise RuntimeError("No predeclared candidate satisfies the minimum development trade coverage")
    def key(item):
        fold=item["folds"]
        return (
            float(np.median([z["avg_net_return"] for z in fold])),
            float(np.mean([z["total_net_return"]>0 for z in fold])),
            float(np.median([z["total_net_return"] for z in fold])),
            -float(np.median([z["max_drawdown"] for z in fold])),
            float(np.median([z["balanced_accuracy"] for z in fold])),
        )
    chosen=max(eligible,key=key)
    horizon=chosen["horizon"]; family=chosen["family"]; threshold=chosen["threshold"]; r=returns[horizon]
    y=np.where(r>COST,1,np.where(r<-COST, -1,0)).astype(int)
    train=np.arange(0,split-horizon)
    final=np.arange(split,len(X))
    if family=="momentum":
        p=momentum_predictions(X[final],threshold)
    else:
        m=model(family)
        m.fit(X[train],y[train] if family=="logistic" else r[train])
        p=classifier_predictions(m,X[final],threshold) if family=="logistic" else ridge_predictions(m,X[final],threshold)
    fm=trade_metrics(y[final],p,r[final])
    result={
        "status":"PASS" if fm["trades"]>=100 and fm["total_net_return"]>0 and fm["avg_net_return"]>0 and fm["max_drawdown"]<=0.15 else "FAIL",
        "data":{"candles":len(rows),"dataset_rows":len(X),"symbol":SYMBOL,"interval":INTERVAL},
        "design":{"horizons":list(HORIZONS),"chosen_horizon":horizon,"cost_rate":COST,"final_holdout_rows":FINAL_HOLDOUT,
                  "final_holdout_start":datetime.fromtimestamp(ts[split]/1000,tz=timezone.utc).isoformat(),
                  "selection":"development-only chronological walk-forward selection with fixed coverage gate"},
        "development":{"min_trades_per_fold":MIN_DEVELOPMENT_TRADES,"eligible_candidates":len(eligible),
                       "chosen":{"horizon":horizon,"family":family,"threshold":threshold},
                       "candidates":candidates},
        "final_holdout":fm,
        "causal_checks":{"chronological":True,"no_future_features":True,"target_horizon":horizon,
                         "holdout_untouched_during_selection":True}
    }
    os.makedirs("phase2_evidence",exist_ok=True)
    open("phase2_evidence/phase2_adaptive_report.json","w").write(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS": raise SystemExit(1)

if __name__=="__main__": main()
