from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "return_1", "range_pct", "volume_change", "order_book_imbalance",
    "funding_rate", "open_interest_change", "news_risk", "news_sentiment",
    "volatility_proxy", "trend_strength", "momentum", "liquidity_stress",
]


@dataclass
class ModelReport:
    trained: bool
    version: str
    metrics: dict
    reason: str


class PredictiveModel:
    """Transparent Logistic Regression baseline for Stage 3 comparison.

    It remains deliberately simple so that every more complex candidate has
    an honest, reproducible benchmark. Production routing may use the
    multi-head ensemble only after the independent validation gate promotes it.
    """

    def __init__(self, artifact_dir: str = "artifacts"):
        self.path = Path(artifact_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.model_path = self.path / "direction_model.json"
        self.model = None
        self.version = "untrained"
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            data = json.loads(self.model_path.read_text())
            if all(key in data for key in ("version", "coef", "intercept", "classes", "mean", "scale")):
                self.load_compact_artifact(data)
                return
            if "X" in data and "y" in data:
                X = np.asarray(data["X"], dtype=float)
                y = np.asarray(data["y"], dtype=int)
                if X.ndim != 2 or X.shape[1] != len(FEATURES) or len(X) != len(y) or len(set(y.tolist())) < 3:
                    raise ValueError("Stored model artifact contains invalid training data.")
                model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
                model.fit(X, y)
                self.model, self.version = model, data["version"]
                return
            raise ValueError("Unrecognized predictive model artifact format.")
        except Exception:
            self.model = None
            self.version = "untrained"

    def vector(self, features: dict) -> list[float]:
        return [float(features.get(key, 0.0) or 0.0) for key in FEATURES]

    def predict(self, features: dict) -> dict:
        required = ("return_1", "range_pct", "volume_change", "volatility_proxy", "trend_strength", "momentum")
        missing = [key for key in required if key not in features]
        if missing:
            return {"trained": self.model is not None, "abstain": True, "version": self.version, "probabilities": {"short": 0.0, "flat": 1.0, "long": 0.0}, "reason": "Predictive model requires candle-derived features; missing: " + ", ".join(missing)}
        if self.model is None:
            return {"trained": False, "abstain": True, "version": self.version, "probabilities": {"short": 0.0, "flat": 1.0, "long": 0.0}, "reason": "No validated model artifact is available."}
        x = np.asarray([self.vector(features)], dtype=float)
        probabilities = self.model.predict_proba(x)[0]
        probability_map = {str(class_id): float(probability) for class_id, probability in zip(self.model.classes_, probabilities)}
        return {"trained": True, "abstain": False, "version": self.version, "probabilities": {"short": probability_map.get("-1", 0.0), "flat": probability_map.get("0", 0.0), "long": probability_map.get("1", 0.0)}}

    def artifact(self) -> dict | None:
        if self.model is None:
            return None
        scaler = self.model.named_steps["scale"]
        classifier = self.model.named_steps["clf"]
        return {"version": self.version, "features": FEATURES, "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(), "classes": classifier.classes_.tolist(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}

    def load_compact_artifact(self, data: dict) -> None:
        required = ("version", "coef", "intercept", "classes", "mean", "scale")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError("Compact model artifact is missing: " + ", ".join(missing))
        stored_features = data.get("features")
        if stored_features is not None and list(stored_features) != FEATURES:
            raise ValueError("Model artifact feature ordering does not match the current model.")
        mean = np.asarray(data["mean"], dtype=float)
        scale_values = np.asarray(data["scale"], dtype=float)
        coef = np.asarray(data["coef"], dtype=float)
        intercept = np.asarray(data["intercept"], dtype=float)
        classes = np.asarray(data["classes"])
        if len(mean) != len(FEATURES) or len(scale_values) != len(FEATURES) or coef.shape[-1] != len(FEATURES):
            raise ValueError("Model artifact has an invalid feature count.")
        if len(classes) != 3:
            raise ValueError("Predictive model must contain exactly three classes: -1, 0 and 1.")
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_, scaler.var_, scaler.n_features_in_ = mean, scale_values, scale_values ** 2, len(FEATURES)
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        classifier.classes_, classifier.coef_, classifier.intercept_, classifier.n_features_in_, classifier.n_iter_ = classes, coef, intercept, len(FEATURES), np.asarray([1])
        self.model = Pipeline([("scale", scaler), ("clf", classifier)])
        self.version = str(data["version"])

    def train(self, rows: list[dict], version: str, min_rows: int = 500) -> ModelReport:
        if len(rows) < min_rows:
            return ModelReport(False, self.version, {}, f"Need at least {min_rows} labeled examples; received {len(rows)}.")
        X = np.asarray([self.vector(row["features"]) for row in rows], dtype=float)
        y = np.asarray([int(row["label"]) for row in rows], dtype=int)
        if set(y.tolist()) != {-1, 0, 1}:
            return ModelReport(False, self.version, {}, "Training data must contain all three labels: -1, 0 and 1.")
        model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
        model.fit(X, y)
        self.model, self.version = model, version
        artifact = self.artifact()
        if artifact is None:
            return ModelReport(False, self.version, {}, "Failed to create model artifact.")
        self.model_path.write_text(json.dumps(artifact, indent=2))
        return ModelReport(True, version, {}, "Candidate baseline model trained and compact artifact persisted.")


predictive_model = PredictiveModel()
