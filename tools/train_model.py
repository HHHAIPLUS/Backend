from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from app.ml.predictive import FEATURES
from app.ml.validation import walk_forward

def load(path):
    rows=[]
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            rows.append({'observed_at':r['observed_at'],'features':{k:float(r.get(k,0) or 0) for k in FEATURES},'label':int(r['label']),'outcome_return':float(r.get('outcome_return',0) or 0)})
    return rows

def vec(r): return [r['features'].get(k,0.0) for k in FEATURES]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('csv'); ap.add_argument('--version',required=True); ap.add_argument('--min-train',type=int,default=500); ap.add_argument('--test-size',type=int,default=100); ap.add_argument('--step',type=int,default=100); ap.add_argument('--artifact-dir',default='artifacts'); args=ap.parse_args()
    rows=load(args.csv); folds=walk_forward(rows,args.min_train,args.test_size,args.step)
    if not folds: raise SystemExit('Not enough chronological data for walk-forward validation.')
    fold_metrics=[]; all_preds=[]
    for fold in folds:
        model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=500,multi_class='auto'))]); X=[vec(r) for r in fold.train]; y=[r['label'] for r in fold.train]; model.fit(X,y)
        pred=model.predict([vec(r) for r in fold.test]); probs=model.predict_proba([vec(r) for r in fold.test]); classes=list(model.classes_)
        returns=[]
        for r,p,prob in zip(fold.test,pred,probs):
            # Trade only when model is sufficiently confident; otherwise flat.
            confidence=float(max(prob)); trade_return=r['outcome_return'] if confidence>=.55 and int(p)==r['label'] else (-abs(r['outcome_return']) if confidence>=.55 else 0.0)
            returns.append(trade_return); all_preds.append((r['label'],int(p),trade_return))
        fold_metrics.append({'train':len(fold.train),'test':len(fold.test),'accuracy':float(np.mean(np.array(pred)==np.array([r['label'] for r in fold.test]))),'balanced_accuracy':float(balanced_accuracy_score([r['label'] for r in fold.test],pred)),'avg_return':float(np.mean(returns))})
    metrics={'folds':len(fold_metrics),'accuracy':float(np.mean([x['accuracy'] for x in fold_metrics])),'balanced_accuracy':float(np.mean([x['balanced_accuracy'] for x in fold_metrics])),'average_return':float(np.mean([x['avg_return'] for x in fold_metrics])),'folds_detail':fold_metrics}
    if metrics['accuracy']<0.52 or metrics['balanced_accuracy']<0.50 or metrics['average_return']<=0: raise SystemExit(json.dumps({'status':'REJECTED','metrics':metrics},indent=2))
    model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=500,multi_class='auto'))]); model.fit([vec(r) for r in rows],[r['label'] for r in rows])
    out=Path(args.artifact_dir); out.mkdir(parents=True,exist_ok=True); (out/'direction_model.json').write_text(json.dumps({'version':args.version,'X':[vec(r) for r in rows],'y':[r['label'] for r in rows],'metrics':metrics}))
    (out/'direction_model_report.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps({'status':'PROMOTED','version':args.version,'metrics':metrics},indent=2))
if __name__=='__main__': main()
