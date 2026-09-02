from __future__ import annotations

from datetime import datetime, timezone

from ai.models import FeatureVector, MarketRegime, Prediction
from app.ml.predictive import predictive_model
from app.ml.predictive_brain import PredictiveBrain
from app.ml.adaptive_intelligence import adaptive_intelligence


class BaselinePredictor:
    model_version = "deterministic-baseline-0.2"

    def predict(self, features: FeatureVector, regime: MarketRegime) -> Prediction:
        momentum = features.features.get("return_1", 0.0)
        volatility = abs(features.features.get("range_pct", 0.0))
        long_probability = short_probability = 0.34
        no_trade_probability = 0.32
        if momentum > 0:
            long_probability += min(momentum * 10, 0.15)
            short_probability -= min(momentum * 5, 0.07)
        elif momentum < 0:
            short_probability += min(abs(momentum) * 10, 0.15)
            long_probability -= min(abs(momentum) * 5, 0.07)
        if volatility > 0.05 or regime == MarketRegime.HIGH_VOLATILITY:
            no_trade_probability += 0.15
            long_probability -= 0.075
            short_probability -= 0.075
        vals = [max(0, long_probability), max(0, short_probability), max(0, no_trade_probability)]
        total = sum(vals)
        vals = [v / total for v in vals]
        return Prediction(symbol=features.symbol, timestamp=datetime.now(timezone.utc), long_probability=vals[0], short_probability=vals[1], no_trade_probability=vals[2], confidence=max(vals), regime=regime, model_version=self.model_version, rationale=["Research/simulation baseline only."])


class ProductionPredictor:
    """Production predictor: promoted Stage 3 brain first, with bounded Stage 4 reliability adaptation."""

    def __init__(self) -> None:
        self.brain = PredictiveBrain()

    def _adaptive(self, probability: float, model_version: str, regime: MarketRegime, features: dict) -> dict:
        return adaptive_intelligence.adapt_confidence(probability, model_version=model_version, regime=regime.value, features=features)

    def predict(self, features: FeatureVector, regime: MarketRegime) -> Prediction:
        brain = self.brain.predict(features.features)
        if brain.get("trained"):
            p = brain["probabilities"]
            raw = max(p.values())
            adaptation = self._adaptive(raw, brain["version"], regime, features.features)
            adjusted = adaptation["adjusted_confidence"]
            # Convert reduced confidence into explicit no-trade mass rather than
            # altering the learned directional ranking.
            no_trade = max(p["flat"], 1.0 - adjusted)
            directional_scale = max(0.0, 1.0 - no_trade)
            side_total = p["long"] + p["short"]
            long_probability = p["long"] / side_total * directional_scale if side_total else 0.0
            short_probability = p["short"] / side_total * directional_scale if side_total else 0.0
            return Prediction(
                symbol=features.symbol,
                timestamp=datetime.now(timezone.utc),
                long_probability=long_probability,
                short_probability=short_probability,
                no_trade_probability=no_trade,
                confidence=max(long_probability, short_probability, no_trade),
                regime=regime,
                model_version=brain["version"],
                rationale=[
                    "Promoted Stage 3 predictive brain with Stage 4 reliability adaptation.",
                    f"decision={brain['decision']}",
                    f"expected_return={brain.get('expected_return', 0.0):.6f}",
                    f"expected_edge_after_cost={brain.get('expected_edge_after_cost', 0.0):.6f}",
                    f"downside_risk={brain.get('downside', 0.0):.6f}",
                    f"uncertainty={brain.get('uncertainty', 1.0):.4f}",
                    f"abstention_probability={brain.get('abstention_probability', 1.0):.4f}",
                    f"adaptive_confidence={adjusted:.4f}",
                    f"adaptive_reliability_samples={adaptation['reliability_samples']}",
                    f"unfamiliar_state={adaptation['unfamiliar'].get('unfamiliar', False)}",
                ],
            )

        r = predictive_model.predict(features.features)
        if r["abstain"]:
            return Prediction(symbol=features.symbol, timestamp=datetime.now(timezone.utc), long_probability=0, short_probability=0, no_trade_probability=1, confidence=1, regime=regime, model_version="untrained", rationale=[r["reason"]])
        p = r["probabilities"]
        adaptation = self._adaptive(max(p.values()), r["version"], regime, features.features)
        no_trade = max(p["flat"], 1.0 - adaptation["adjusted_confidence"])
        side_total = p["long"] + p["short"]
        scale = max(0.0, 1.0 - no_trade)
        return Prediction(symbol=features.symbol, timestamp=datetime.now(timezone.utc), long_probability=p["long"] / side_total * scale if side_total else 0.0, short_probability=p["short"] / side_total * scale if side_total else 0.0, no_trade_probability=no_trade, confidence=max(no_trade, scale), regime=regime, model_version=r["version"], rationale=["Validated Logistic Regression baseline fallback with Stage 4 reliability adaptation.", f"adaptive_confidence={adaptation['adjusted_confidence']:.4f}", f"adaptive_reliability_samples={adaptation['reliability_samples']}"])
