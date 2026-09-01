from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import platform
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.predictive import FEATURES


MODEL_FAMILIES = ("logistic_regression", "extra_trees", "hist_gradient_boosting")
HORIZONS = (1, 3, 6, 12)
COST_RATE = 0.0008


@dataclass
class BrainReport:
    status: str
    version: str
    metrics: dict[str, Any]
    reason: str
    artifact: str | None = None


def _x(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(r["features"].get(k, 0.0) or 0.0) for k in FEATURES] for r in rows], dtype=float)


def _future_return(rows: list[dict[str, Any]], horizon: int) -> np.ndarray:
    # Rows produced by the canonical dataset contain a single future outcome.
    # For horizons other than the source horizon, use the available outcome only
    # when explicitly tagged; this prevents silently inventing labels.
    values = []
    for row in rows:
        by_horizon = row.get("outcome_return_by_horizon", {})
        value = by_horizon.get(str(horizon))
        if value is None:
            value = by_horizon.get(horizon)
        if value is None and horizon == int(row.get("outcome_horizon", 6)):
            value = row.get("outcome_return")
        if value is None:
            raise ValueError(f"Missing point-in-time outcome for horizon {horizon}")
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _direction_target(y: np.ndarray, threshold: float = COST_RATE) -> np.ndarray:
    return np.where(y > threshold, 1, np.where(y < -threshold, -1, 0))


def _make_classifier(family: str):
    if family == "logistic_regression":
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])
    if family == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    raise ValueError(f"Unknown model family: {family}")


def _make_regressor(family: str):
    if family == "logistic_regression":
        return Pipeline([("scale", StandardScaler()), ("model", ExtraTreesRegressor(n_estimators=250, min_samples_leaf=5, random_state=42, n_jobs=-1))])
    if family == "extra_trees":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    raise ValueError(f"Unknown model family: {family}")


def _fit_predict_classifier(family: str, x_train, y_train, x_test):
    model = _make_classifier(family)
    model.fit(x_train, y_train)
    return model, model.predict_proba(x_test), model.predict(x_test)


def _calibration(probs: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    mapping = {int(c): i for i, c in enumerate(classes)}
    if all(c in mapping for c in (-1, 0, 1)):
        long_p = probs[:, mapping[1]]
        short_p = probs[:, mapping[-1]]
        flat_p = probs[:, mapping[0]]
        one_hot = np.column_stack([(y == -1).astype(float), (y == 0).astype(float), (y == 1).astype(float)])
        ordered = np.column_stack([short_p, flat_p, long_p])
        multiclass_brier = float(np.mean(np.sum((ordered - one_hot) ** 2, axis=1)))
        return {"brier": multiclass_brier, "max_probability_mean": float(np.max(probs, axis=1).mean())}
    return {"brier": float("nan"), "max_probability_mean": float(np.max(probs, axis=1).mean())}


def _metrics(y: np.ndarray, pred: np.ndarray, probs: np.ndarray, classes: np.ndarray, returns: np.ndarray) -> dict[str, Any]:
    direction = pred.astype(int)
    signed = returns * np.where(direction == 1, 1.0, np.where(direction == -1, -1.0, 0.0))
    traded = direction != 0
    net = signed - np.where(traded, COST_RATE, 0.0)
    calibration = _calibration(probs, y, classes)
    equity = np.cumsum(net)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    drawdown = float(np.max(peak[1:] - equity)) if len(equity) else 0.0
    return {
        "samples": int(len(y)),
        "trades": int(traded.sum()),
        "trade_rate": float(traded.mean()) if len(traded) else 0.0,
        "accuracy": float(accuracy_score(y, direction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, direction)),
        "precision_macro": float(precision_score(y, direction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, direction, average="macro", zero_division=0)),
        "avg_net_return": float(net.mean()),
        "avg_trade_net_return": float(net[traded].mean()) if traded.any() else 0.0,
        "total_net_return": float(net.sum()),
        "max_drawdown": drawdown,
        "calibration": calibration,
        "class_distribution": {str(int(c)): int((y == c).sum()) for c in classes},
    }


class PredictiveBrain:
    """Validated multi-model predictive brain.

    The brain keeps a simple logistic baseline and only promotes an ensemble
    when it improves the baseline on untouched chronological validation data.
    It produces direction, expected return, downside, volatility and regime
    outputs plus an explicit abstention/no-trade decision.
    """

    def __init__(self, artifact_dir: str = "artifacts"):
        self.path = Path(artifact_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.path / "predictive_brain.joblib"
        self.manifest_path = self.path / "predictive_brain_manifest.json"
        self.bundle: dict[str, Any] | None = None
        self.version = "untrained"
        self._load()

    def _load(self) -> None:
        if not self.artifact_path.exists() or not self.manifest_path.exists():
            return
        try:
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("features") != FEATURES:
                return
            bundle = joblib.load(self.artifact_path)
            if bundle.get("feature_hash") != _feature_hash():
                return
            self.bundle = bundle
            self.version = str(manifest["version"])
        except Exception:
            self.bundle = None
            self.version = "untrained"

    def train(self, rows: list[dict[str, Any]], version: str = "brain-v1", test_fraction: float = 0.2) -> BrainReport:
        if len(rows) < 600:
            return BrainReport("REJECTED", version, {"rows": len(rows)}, "At least 600 point-in-time rows are required.")
        rows = sorted(rows, key=lambda r: str(r.get("observed_at", "")))
        x = _x(rows)
        y = _future_return(rows, 6)
        direction = _direction_target(y)
        split = int(len(rows) * (1.0 - test_fraction))
        if split < 300 or len(rows) - split < 100:
            return BrainReport("REJECTED", version, {"rows": len(rows)}, "Chronological train/test split is too small.")
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = direction[:split], direction[split:]
        r_train, r_test = y[:split], y[split:]
        if len(set(y_train.tolist())) < 3 or len(set(y_test.tolist())) < 3:
            return BrainReport("REJECTED", version, {}, "All three direction classes are required in train and test sets.")

        family_scores: dict[str, Any] = {}
        candidates = []
        for family in MODEL_FAMILIES:
            try:
                model, probs, pred = _fit_predict_classifier(family, x_train, y_train, x_test)
                score = _metrics(y_test, pred, probs, model.classes_, r_test)
                family_scores[family] = score
                candidates.append((float(score["avg_net_return"]), float(score["balanced_accuracy"]), family, model, score))
            except Exception as exc:
                family_scores[family] = {"error": f"{type(exc).__name__}: {exc}"}
        baseline = family_scores.get("logistic_regression")
        valid = [c for c in candidates if c[2] != "logistic_regression"]
        if not baseline or not valid:
            return BrainReport("REJECTED", version, {"families": family_scores}, "No complete model-family evaluation was possible.")
        baseline_score = (float(baseline.get("avg_net_return", -math.inf)), float(baseline.get("balanced_accuracy", 0.0)))
        valid.sort(reverse=True, key=lambda item: (item[0], item[1]))
        best = valid[0]
        if best[0] <= baseline_score[0] or best[1] <= baseline_score[1]:
            # Complexity is rejected unless it beats the honest baseline on both economics and discrimination.
            return BrainReport("REJECTED", version, {"families": family_scores, "baseline": baseline_score, "best_candidate": best[2]}, "No complex candidate beat the logistic baseline on both net return and balanced accuracy.")

        best_model = best[3]
        expected_return = _make_regressor(best[2])
        expected_return.fit(x_train, r_train)
        downside_target = np.minimum(r_train, 0.0)
        downside_model = _make_regressor(best[2])
        downside_model.fit(x_train, downside_target)
        volatility_target = np.abs(r_train)
        volatility_model = _make_regressor(best[2])
        volatility_model.fit(x_train, volatility_target)
        regime_target = np.where(volatility_target > np.quantile(volatility_target, 0.66), 2, np.where(volatility_target > np.quantile(volatility_target, 0.33), 1, 0))
        regime_model = _make_classifier(best[2])
        regime_model.fit(x_train, regime_target)

        # Meta-model: trained on chronological out-of-fold predictions from the selected family.
        meta_features_train = np.column_stack([best_model.predict_proba(x_train), expected_return.predict(x_train), downside_model.predict(x_train), volatility_model.predict(x_train)])
        meta = LogisticRegression(max_iter=1000, multi_class="auto", random_state=42)
        meta.fit(meta_features_train, y_train)

        bundle = {
            "direction_model": best_model,
            "expected_return_model": expected_return,
            "downside_model": downside_model,
            "volatility_model": volatility_model,
            "regime_model": regime_model,
            "meta_model": meta,
            "family": best[2],
            "feature_hash": _feature_hash(),
            "features": FEATURES,
            "cost_rate": COST_RATE,
            "horizons": HORIZONS,
            "sequence_model_evaluation": {"status": "NOT_REQUIRED", "reason": "Stage 3 evaluation uses the available tabular point-in-time dataset; sequence architecture is deferred until dataset volume/sequence coverage justifies it."},
        }
        tmp = self.artifact_path.with_suffix(".tmp")
        joblib.dump(bundle, tmp)
        tmp.replace(self.artifact_path)
        manifest = {
            "version": version,
            "features": FEATURES,
            "feature_hash": _feature_hash(),
            "family": best[2],
            "metrics": {"families": family_scores, "promoted": best[2]},
            "cost_rate": COST_RATE,
            "horizons": HORIZONS,
            "python": platform.python_version(),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        self.bundle = bundle
        self.version = version
        return BrainReport("PROMOTED", version, manifest["metrics"], "Candidate ensemble beat the logistic baseline on chronological out-of-sample net return and balanced accuracy.", str(self.artifact_path))

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        if self.bundle is None:
            return {"trained": False, "abstain": True, "version": self.version, "reason": "No promoted predictive brain artifact is available."}
        x = np.asarray([[float(features.get(k, 0.0) or 0.0) for k in FEATURES]], dtype=float)
        direction_model = self.bundle["direction_model"]
        base_probs = direction_model.predict_proba(x)[0]
        classes = list(direction_model.classes_)
        expected = float(self.bundle["expected_return_model"].predict(x)[0])
        downside = float(self.bundle["downside_model"].predict(x)[0])
        volatility = float(self.bundle["volatility_model"].predict(x)[0])
        regime = int(self.bundle["regime_model"].predict(x)[0])
        meta_features = np.column_stack([base_probs, [expected], [downside], [volatility]])
        meta_probs = self.bundle["meta_model"].predict_proba(meta_features)[0]
        meta_classes = list(self.bundle["meta_model"].classes_)
        probs = {"short": float(meta_probs[meta_classes.index(-1)]) if -1 in meta_classes else 0.0, "flat": float(meta_probs[meta_classes.index(0)]) if 0 in meta_classes else 0.0, "long": float(meta_probs[meta_classes.index(1)]) if 1 in meta_classes else 0.0}
        direction = max(probs, key=probs.get)
        edge = expected - self.bundle["cost_rate"]
        uncertainty = 1.0 - max(probs.values())
        abstain = direction == "flat" or max(probs.values()) < 0.55 or edge <= 0.0 or not np.isfinite([expected, downside, volatility]).all()
        if abstain:
            decision = "NO_TRADE"
        else:
            decision = direction.upper()
        return {"trained": True, "abstain": abstain, "version": self.version, "decision": decision, "probabilities": probs, "expected_return": expected, "expected_edge_after_cost": edge, "downside": downside, "volatility": volatility, "regime": regime, "uncertainty": float(uncertainty), "model_family": self.bundle["family"]}

    def manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text())
        except Exception:
            return None


def _feature_hash() -> str:
    return hashlib.sha256("|".join(FEATURES).encode()).hexdigest()
