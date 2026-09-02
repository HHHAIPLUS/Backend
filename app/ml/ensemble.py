from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    """Multi-head predictive ensemble for futures research and production gating."""

    def __init__(self, artifact_path: str = "artifacts/predictive_ensemble.joblib") -> None:
        self.artifact_path = Path(artifact_path)
        self.baseline = None
        self.direction = None
        self.return_model = None
        self.risk_model = None
        self.vol_model = None
        self.regime_model = None
        self.abstention_model = None
        self.feature_names: list[str] = []
        self.version = "untrained"
        self.validation_metrics: dict[str, Any] = {}
        self._load()

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

    def _load(self) -> None:
        if not self.artifact_path.exists():
            return
        try:
            payload = joblib.load(self.artifact_path)
            if payload.get("schema_version") != 2:
                raise ValueError("Unsupported ensemble artifact schema.")
            for name in ("baseline", "direction", "return_model", "risk_model", "vol_model", "regime_model", "abstention_model"):
                setattr(self, name, payload[name])
            self.feature_names = list(payload["feature_names"])
            self.version = str(payload["version"])
            self.validation_metrics = dict(payload.get("validation_metrics", {}))
        except Exception:
            self.baseline = self.direction = None
            self.return_model = self.risk_model = self.vol_model = None
            self.regime_model = self.abstention_model = None
            self.feature_names = []
            self.version = "untrained"
            self.validation_metrics = {}

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

        self.baseline = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))]).fit(x, y)

        # Fit the nonlinear direction estimator on the earlier portion and
        # calibrate it on a strictly later holdout. FrozenEstimator is the
        # supported replacement for the removed cv="prefit" API in sklearn 1.9.
        split = int(len(train_rows) * 0.80)
        split = max(300, min(split, len(train_rows) - 100))
        nonlinear = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42).fit(x[:split], y[:split])
        calibration = FrozenEstimator(nonlinear)
        self.direction = CalibratedClassifierCV(calibration, method="sigmoid").fit(x[split:], y[split:])

        self.return_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42).fit(x, returns)
        self.risk_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=43).fit(x, np.maximum(0.0, -returns))
        self.vol_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=44).fit(x, np.abs(returns))
        regime = np.asarray([1 if float(r["features"].get("trend_strength", 0.0) or 0.0) > 0.5 else 2 if abs(float(r["features"].get("volatility_proxy", 0.0) or 0.0)) > 0.05 else 0 for r in train_rows], dtype=int)
        self.regime_model = HistGradientBoostingClassifier(max_iter=150, random_state=45).fit(x, regime)
        cutoff = max(1e-8, float(np.quantile(np.abs(returns), 0.30)))
        self.abstention_model = HistGradientBoostingClassifier(max_iter=150, random_state=46).fit(x, (np.abs(returns) <= cutoff).astype(int))

    def save(self, validation_metrics: dict[str, Any]) -> None:
        if self.baseline is None or self.direction is None:
            raise ValueError("Cannot save an untrained ensemble.")
        self.validation_metrics = dict(validation_metrics)
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"schema_version": 2, "version": self.version, "feature_names": self.feature_names, "baseline": self.baseline, "direction": self.direction, "return_model": self.return_model, "risk_model": self.risk_model, "vol_model": self.vol_model, "regime_model": self.regime_model, "abstention_model": self.abstention_model, "validation_metrics": self.validation_metrics}, self.artifact_path)

    def predict(self, features: dict[str, Any]) -> EnsemblePrediction:
        models = (self.baseline, self.direction, self.return_model, self.risk_model, self.vol_model, self.regime_model, self.abstention_model)
        if any(model is None for model in models):
            return EnsemblePrediction(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, True)
        x = np.asarray([[float(features.get(k, 0.0) or 0.0) for k in self.feature_names]], dtype=float)
        if not np.isfinite(x).all():
            return EnsemblePrediction(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, True)
        base_p = self.baseline.predict_proba(x)[0]
        direction_p = self.direction.predict_proba(x)[0]
        bm = {int(c): float(v) for c, v in zip(self.baseline.classes_, base_p)}
        dm = {int(c): float(v) for c, v in zip(self.direction.classes_, direction_p)}
        short = 0.65 * dm.get(-1, 0.0) + 0.35 * bm.get(-1, 0.0)
        flat = 0.65 * dm.get(0, 0.0) + 0.35 * bm.get(0, 0.0)
        long = 0.65 * dm.get(1, 0.0) + 0.35 * bm.get(1, 0.0)
        total = short + flat + long
        short, flat, long = short / total, flat / total, long / total
        expected = float(self.return_model.predict(x)[0])
        risk = max(0.0, float(self.risk_model.predict(x)[0]))
        volatility = max(0.0, float(self.vol_model.predict(x)[0]))
        regime = float(self.regime_model.predict_proba(x).max())
        abstain_probability = float(self.abstention_model.predict_proba(x)[0][1])
        entropy = -sum(v * np.log(max(v, 1e-12)) for v in (short, flat, long)) / np.log(3.0)
        disagreement = abs(dm.get(1, 0.0) - bm.get(1, 0.0)) + abs(dm.get(-1, 0.0) - bm.get(-1, 0.0))
        uncertainty = min(1.0, 0.7 * entropy + 0.3 * disagreement)
        agreement = max(0.0, 1.0 - disagreement)
        abstain = abstain_probability >= 0.60 or uncertainty >= 0.70 or not all(isfinite(v) for v in (expected, risk, volatility))
        return EnsemblePrediction(short, flat, long, expected, risk, volatility, regime, uncertainty, agreement, abstain)

    def evaluate(self, rows: list[dict]) -> dict[str, float]:
        if not rows or self.direction is None:
            return {"accuracy": 0.0, "balanced_accuracy": 0.0, "log_loss": float("inf"), "avg_return": 0.0, "trades": 0.0}
        x = self._matrix(rows, self.feature_names)
        y = self._labels(rows)
        p = self.direction.predict_proba(x)
        classes = [int(c) for c in self.direction.classes_]
        predicted = np.asarray([classes[i] for i in np.argmax(p, axis=1)], dtype=int)
        returns = self._returns(rows)
        signed = np.asarray([r if pred == 1 else -r if pred == -1 else 0.0 for r, pred in zip(returns, predicted)])
        traded = predicted != 0
        return {"accuracy": float(accuracy_score(y, predicted)), "balanced_accuracy": float(balanced_accuracy_score(y, predicted)), "log_loss": float(log_loss(y, p, labels=classes)), "avg_return": float(np.mean(signed[traded])) if traded.any() else 0.0, "trades": float(traded.sum())}


predictive_ensemble = PredictiveEnsemble()
