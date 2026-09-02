from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss


@dataclass
class EnsemblePrediction:
    short: float
    flat: float
    long: float
    expected_return: float
    downside_risk: float
    volatility: float
    regime: float
    uncertainty: float
    model_agreement: float
    abstain: bool


class PredictiveEnsemble:
    """Small, reproducible multi-head ensemble for directional futures research.

    The logistic model remains the auditable baseline. Tree models provide
    nonlinear candidates; regressors estimate return, downside and volatility.
    Every head is trained only on rows supplied by the chronological trainer.
    """

    def __init__(self) -> None:
        self.baseline: Pipeline | None = None
        self.direction: CalibratedClassifierCV | None = None
        self.return_model: HistGradientBoostingRegressor | None = None
        self.risk_model: HistGradientBoostingRegressor | None = None
        self.vol_model: HistGradientBoostingRegressor | None = None
        self.regime_model: HistGradientBoostingClassifier | None = None
        self.abstention_model: HistGradientBoostingClassifier | None = None
        self.feature_names: list[str] = []
        self.version = "untrained"
        self.baseline_log_loss: float | None = None
        self.validation_metrics: dict[str, Any] = {}

    @staticmethod
    def _matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
        x = np.asarray([[float(r["features"].get(k, 0.0) or 0.0) for k in feature_names] for r in rows], dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Training features contain non-finite values.")
        return x

    @staticmethod
    def _labels(rows: list[dict]) -> np.ndarray:
        return np.asarray([int(r["label"]) for r in rows], dtype=int)

    @staticmethod
    def _returns(rows: list[dict]) -> np.ndarray:
        values = np.asarray([float(r["outcome_return"]) for r in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Training outcomes contain non-finite returns.")
        return values

    def fit(self, train_rows: list[dict], feature_names: list[str], version: str) -> None:
        if len(train_rows) < 500:
            raise ValueError("Predictive ensemble requires at least 500 rows.")
        y = self._labels(train_rows)
        if set(y.tolist()) != {-1, 0, 1}:
            raise ValueError("Training data must contain -1, 0 and 1 labels.")
        x = self._matrix(train_rows, feature_names)
        returns = self._returns(train_rows)

        self.feature_names = list(feature_names)
        self.version = version

        self.baseline = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        ])
        self.baseline.fit(x, y)

        # Calibration uses a held-out tail, preserving chronology.
        split = max(100, int(len(train_rows) * 0.8))
        if split >= len(train_rows):
            split = len(train_rows) - 1
        base_for_calibration = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            random_state=42
        )
        base_for_calibration.fit(x[:split], y[:split])
        self.direction = CalibratedClassifierCV(base_for_calibration, method="sigmoid", cv="prefit")
        self.direction.fit(x[split:], y[split:])

        self.return_model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42
        ).fit(x, returns)
        downside = np.maximum(0.0, -returns)
        self.risk_model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=43
        ).fit(x, downside)
        self.vol_model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=44
        ).fit(x, np.abs(returns))

        regime = np.asarray([
            1 if float(r["features"].get("trend_strength", 0.0) or 0.0) > 0.5 else
            2 if abs(float(r["features"].get("volatility_proxy", 0.0) or 0.0)) > 0.05 else 0
            for r in train_rows
        ], dtype=int)
        self.regime_model = HistGradientBoostingClassifier(max_iter=150, random_state=45).fit(x, regime)
        # The abstention target is conservative: low absolute future return is
        # treated as no-trade, but the final gate also checks uncertainty.
        abstain = (np.abs(returns) <= max(1e-8, float(np.quantile(np.abs(returns), 0.30)))).astype(int)
        self.abstention_model = HistGradientBoostingClassifier(max_iter=150, random_state=46).fit(x, abstain)

    def predict(self, features: dict[str, Any]) -> EnsemblePrediction:
        if not all([self.baseline, self.direction, self.return_model, self.risk_model, self.vol_model, self.regime_model, self.abstention_model]):
            return EnsemblePrediction(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, True)
        x = np.asarray([[float(features.get(k, 0.0) or 0.0) for k in self.feature_names]], dtype=float)
        if not np.isfinite(x).all():
            return EnsemblePrediction(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, True)
        p_base = self.baseline.predict_proba(x)[0]
        p = self.direction.predict_proba(x)[0]
        classes = [int(v) for v in self.direction.classes_]
        mapping = {c: float(v) for c, v in zip(classes, p)}
        # Blend calibrated nonlinear direction with the logistic baseline.
        base_classes = [int(v) for v in self.baseline.classes_]
        bm = {c: float(v) for c, v in zip(base_classes, p_base)}
        short = 0.65 * mapping.get(-1, 0.0) + 0.35 * bm.get(-1, 0.0)
        flat = 0.65 * mapping.get(0, 0.0) + 0.35 * bm.get(0, 0.0)
        long = 0.65 * mapping.get(1, 0.0) + 0.35 * bm.get(1, 0.0)
        total = short + flat + long
        short, flat, long = short / total, flat / total, long / total
        expected = float(self.return_model.predict(x)[0])
        risk = max(0.0, float(self.risk_model.predict(x)[0]))
        volatility = max(0.0, float(self.vol_model.predict(x)[0]))
        regime = float(self.regime_model.predict_proba(x).max())
        abstain_probability = float(self.abstention_model.predict_proba(x)[0][1])
        # Entropy-like uncertainty plus model disagreement.
        entropy = -sum(v * np.log(max(v, 1e-12)) for v in (short, flat, long)) / np.log(3.0)
        disagreement = abs(float(mapping.get(1, 0.0) - bm.get(1, 0.0))) + abs(float(mapping.get(-1, 0.0) - bm.get(-1, 0.0)))
        uncertainty = min(1.0, 0.7 * entropy + 0.3 * disagreement)
        agreement = max(0.0, 1.0 - disagreement)
        abstain = abstain_probability >= 0.60 or uncertainty >= 0.70 or not all(isfinite(v) for v in (expected, risk, volatility))
        return EnsemblePrediction(short, flat, long, expected, risk, volatility, regime, uncertainty, agreement, abstain)

    def evaluate_baseline(self, rows: list[dict]) -> dict[str, float]:
        if not rows or self.baseline is None:
            return {"log_loss": float("inf"), "mean_expected_return": 0.0}
        x = self._matrix(rows, self.feature_names)
        y = self._labels(rows)
        probabilities = self.baseline.predict_proba(x)
        return {
            "log_loss": float(log_loss(y, probabilities, labels=[-1, 0, 1])),
            "mean_expected_return": float(np.mean(self._returns(rows))),
        }
