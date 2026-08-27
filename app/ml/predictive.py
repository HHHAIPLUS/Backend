from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['return_1','range_pct','volume_change','order_book_imbalance','funding_rate','open_interest_change','news_risk','news_sentiment','volatility_proxy','trend_strength','momentum','liquidity_stress']

@dataclass
class ModelReport:
    trained: bool
    version: str
    metrics: dict
    reason: str

class PredictiveModel:
    def __init__(self, artifact_dir='artifacts'):
        self.path=Path(artifact_dir); self.path.mkdir(parents=True, exist_ok=True)
        self.model_path=self.path/'direction_model.json'
        self.model=None; self.version='untrained'
        self._load()

    def _load(self):
        if self.model_path.exists():
            data=json.loads(self.model_path.read_text())
            self.version=data['version']
            self.model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=500))])
            self.model.fit(np.array(data['X']), np.array(data['y']))

    def vector(self, features: dict) -> list[float]:
        return [float(features.get(k,0.0) or 0.0) for k in FEATURES]

    def predict(self, features: dict) -> dict:
        if self.model is None:
            return {'trained':False,'abstain':True,'version':self.version,'probabilities':{'short':0.0,'flat':1.0,'long':0.0},'reason':'No validated model artifact is available.'}
        x=np.array([self.vector(features)])
        p=self.model.predict_proba(x)[0]
        classes=list(self.model.classes_)
        probs={str(c):float(v) for c,v in zip(classes,p)}
        return {'trained':True,'abstain':False,'version':self.version,'probabilities':{'short':probs.get('-1',0.0),'flat':probs.get('0',0.0),'long':probs.get('1',0.0)}}

    def artifact(self) -> dict | None:
        if self.model is None:
            return None
        return {'version': self.version, 'X': self.model.named_steps['scale'].inverse_transform(self.model.named_steps['scale'].transform(np.array([]).reshape(0, len(FEATURES)))).tolist() if False else [], 'y': [], 'coef': self.model.named_steps['clf'].coef_.tolist(), 'intercept': self.model.named_steps['clf'].intercept_.tolist(), 'classes': self.model.named_steps['clf'].classes_.tolist(), 'mean': self.model.named_steps['scale'].mean_.tolist(), 'scale': self.model.named_steps['scale'].scale_.tolist()}

    def load_compact_artifact(self, data: dict) -> None:
        self.version = data['version']
        scale = StandardScaler()
        scale.mean_ = np.asarray(data['mean'], dtype=float)
        scale.scale_ = np.asarray(data['scale'], dtype=float)
        scale.var_ = scale.scale_ ** 2
        scale.n_features_in_ = len(FEATURES)
        clf = LogisticRegression(max_iter=500)
        clf.classes_ = np.asarray(data['classes'])
        clf.coef_ = np.asarray(data['coef'], dtype=float)
        clf.intercept_ = np.asarray(data['intercept'], dtype=float)
        clf.n_features_in_ = len(FEATURES)
        clf.n_iter_ = np.asarray([1])
        self.model = Pipeline([('scale', scale), ('clf', clf)])

    def train(self, rows: list[dict], version: str, min_rows: int = 500):
        if len(rows)<min_rows: return ModelReport(False, self.version, {}, f'Need at least {min_rows} labeled examples; received {len(rows)}.')
        X=np.array([self.vector(r['features']) for r in rows]); y=np.array([int(r['label']) for r in rows])
        if len(set(y.tolist()))<3: return ModelReport(False,self.version,{},'Training data must contain short, flat and long labels.')
        # Caller is expected to provide walk-forward validation metrics before promotion.
        model=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=500))])
        model.fit(X,y)
        self.model=model; self.version=version
        self.model_path.write_text(json.dumps({'version':version,'X':X.tolist(),'y':y.tolist()}))
        return ModelReport(True,version,{},'Candidate model trained. Promotion still requires independent walk-forward gates.')

predictive_model=PredictiveModel()
