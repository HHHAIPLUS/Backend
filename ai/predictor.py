from __future__ import annotations
from datetime import datetime, timezone
from ai.models import FeatureVector, MarketRegime, Prediction
from app.ml.predictive import predictive_model

class BaselinePredictor:
    model_version='deterministic-baseline-0.2'
    def predict(self, features:FeatureVector, regime:MarketRegime)->Prediction:
        # Research/simulation fallback only. Production execution must use a validated model.
        momentum=features.features.get('return_1',0.0); volatility=abs(features.features.get('range_pct',0.0))
        long_probability=short_probability=0.34; no_trade_probability=0.32
        if momentum>0: long_probability+=min(momentum*10,.15); short_probability-=min(momentum*5,.07)
        elif momentum<0: short_probability+=min(abs(momentum)*10,.15); long_probability-=min(abs(momentum)*5,.07)
        if volatility>.05 or regime==MarketRegime.HIGH_VOLATILITY: no_trade_probability+=.15; long_probability-=.075; short_probability-=.075
        vals=[max(0,long_probability),max(0,short_probability),max(0,no_trade_probability)]; total=sum(vals); vals=[v/total for v in vals]
        return Prediction(symbol=features.symbol,timestamp=datetime.now(timezone.utc),long_probability=vals[0],short_probability=vals[1],no_trade_probability=vals[2],confidence=max(vals),regime=regime,model_version=self.model_version,rationale=['Research/simulation baseline only.'])

class ProductionPredictor:
    """Validated-model gate. Abstains when no independently validated artifact is available."""
    def predict(self, features:FeatureVector, regime:MarketRegime)->Prediction:
        r=predictive_model.predict(features.features)
        if r['abstain']:
            return Prediction(symbol=features.symbol,timestamp=datetime.now(timezone.utc),long_probability=0,short_probability=0,no_trade_probability=1,confidence=1,regime=regime,model_version='untrained',rationale=[r['reason']])
        p=r['probabilities']; confidence=max(p.values())
        return Prediction(symbol=features.symbol,timestamp=datetime.now(timezone.utc),long_probability=p['long'],short_probability=p['short'],no_trade_probability=p['flat'],confidence=confidence,regime=regime,model_version=r['version'],rationale=['Validated model artifact loaded.'])
