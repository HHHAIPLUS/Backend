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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
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
        return ExtraTreesRegressor(n_estimators=250, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if family == "extra_trees":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
    if family == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    raise ValueError(f"Unknown model family: {family}")

def _metrics(y: np.ndarray, pred: np.ndarray, probs: np.ndarray, classes: np.ndarray, returns: np.ndarray) -> dict[str, Any]:
    direction = pred.astype(int)
    signed = returns * np.where(direction == 1, 1.0, np.where(direction == -1, -1.0, 0.0))
    traded = direction != 0
    net = signed - np.where(traded, COST_RATE, 0.0)
    mapping = {int(c): i for i, c in enumerate(classes)}
    if all(c in mapping for c in (-1, 0, 1)):
        ordered = np.column_stack([probs[:, mapping[-1]], probs[:, mapping[0]], probs[:, mapping[1]]])
        truth = np.column_stack([(y == -1).astype(float), (y == 0).astype(float), (y == 1).astype(float)])
        brier = float(np.mean(np.sum((ordered - truth) ** 2, axis=1)))
    else:
        brier = float("nan")
    equity = np.cumsum(net)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    drawdown = float(np.max(peak[1:] - equity)) if len(equity) else 0.0
    return {
        "samples": int(len(y)), "trades": int(traded.sum()), "trade_rate": float(traded.mean()) if len(traded) else 0.0,
        "accuracy": float(accuracy_score(y, direction)), "balanced_accuracy": float(balanced_accuracy_score(y, direction)),
        "precision_macro": float(precision_score(y, direction, average="macro", zero_division=0)), "recall_macro": float(recall_score(y, direction, average="macro", zero_division=0)),
        "avg_net_return": float(net.mean()), "avg_trade_net_return": float(net[traded].mean()) if traded.any() else 0.0,
        "total_net_return": float(net.sum()), "max_drawdown": drawdown, "calibration_brier": brier,
        "mean_confidence": float(np.max(probs, axis=1).mean()), "class_distribution": {str(int(c)): int((y == c).sum()) for c in classes},
    }

def _feature_hash() -> str:
    return hashlib.sha256("|".join(FEATURES).encode()).hexdigest()

class PredictiveBrain:
    """Validated multi-model predictive brain with an explicit no-trade gate."""
    def __init__(self, artifact_dir: str = "artifacts"):
        self.path = Path(artifact_dir); self.path.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.path / "predictive_brain.joblib"
        self.manifest_path = self.path / "predictive_brain_manifest.json"
        self.bundle: dict[str, Any] | None = None; self.version = "untrained"; self._load()

    def _load(self) -> None:
        if not self.artifact_path.exists() or not self.manifest_path.exists(): return
        try:
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("features") != FEATURES: return
            bundle = joblib.load(self.artifact_path)
            if bundle.get("feature_hash") != _feature_hash(): return
            self.bundle = bundle; self.version = str(manifest["version"])
        except Exception:
            self.bundle = None; self.version = "untrained"

    def _evaluate_horizons(self, x_train, x_test, rows_train, rows_test) -> dict[str, Any]:
        results = {}
        for horizon in HORIZONS:
            try:
                train_y = _future_return(rows_train, horizon); test_y = _future_return(rows_test, horizon)
                train_d = _direction_target(train_y); test_d = _direction_target(test_y)
                if len(set(train_d.tolist())) < 3 or len(set(test_d.tolist())) < 3: raise ValueError("three direction classes are required")
                model = _make_classifier("logistic_regression"); model.fit(x_train, train_d); pred = model.predict(x_test); probs = model.predict_proba(x_test)
                results[str(horizon)] = _metrics(test_d, pred, probs, model.classes_, test_y)
            except Exception as exc:
                results[str(horizon)] = {"status": "UNAVAILABLE", "reason": str(exc)}
        return results

    def train(self, rows: list[dict[str, Any]], version: str = "brain-v1", test_fraction: float = 0.2) -> BrainReport:
        if len(rows) < 600: return BrainReport("REJECTED", version, {"rows": len(rows)}, "At least 600 point-in-time rows are required.")
        rows = sorted(rows, key=lambda r: str(r.get("observed_at", ""))); x = _x(rows); y = _future_return(rows, 6); direction = _direction_target(y)
        split = int(len(rows) * (1.0 - test_fraction)); x_train, x_test = x[:split], x[split:]; y_train, y_test = direction[:split], direction[split:]; r_train, r_test = y[:split], y[split:]
        if split < 300 or len(rows) - split < 100: return BrainReport("REJECTED", version, {}, "Chronological train/test split is too small.")
        if len(set(y_train.tolist())) < 3 or len(set(y_test.tolist())) < 3: return BrainReport("REJECTED", version, {}, "All three direction classes are required in train and test sets.")
        family_scores = {}; candidates = []
        for family in MODEL_FAMILIES:
            try:
                model = _make_classifier(family); model.fit(x_train, y_train); probs = model.predict_proba(x_test); pred = model.predict(x_test); score = _metrics(y_test, pred, probs, model.classes_, r_test)
                family_scores[family] = score; candidates.append((score["avg_net_return"], score["balanced_accuracy"], family, model, score))
            except Exception as exc: family_scores[family] = {"error": f"{type(exc).__name__}: {exc}"}
        baseline = family_scores.get("logistic_regression"); valid = [c for c in candidates if c[2] != "logistic_regression"]
        if not baseline or not valid: return BrainReport("REJECTED", version, {"families": family_scores}, "No complete model-family evaluation was possible.")
        valid.sort(reverse=True, key=lambda item: (item[0], item[1])); best = valid[0]
        if best[0] <= float(baseline.get("avg_net_return", -math.inf)) or best[1] <= float(baseline.get("balanced_accuracy", 0.0)):
            return BrainReport("REJECTED", version, {"families": family_scores, "baseline": baseline, "best_candidate": best[2]}, "No complex candidate beat the logistic baseline on both net return and balanced accuracy.")
        best_model = best[3]
        expected_return = _make_regressor(best[2]); expected_return.fit(x_train, r_train)
        downside_model = _make_regressor(best[2]); downside_model.fit(x_train, np.minimum(r_train, 0.0))
        volatility_model = _make_regressor(best[2]); volatility_model.fit(x_train, np.abs(r_train))
        vol = np.abs(r_train); regime_target = np.where(vol > np.quantile(vol, .66), 2, np.where(vol > np.quantile(vol, .33), 1, 0)); regime_model = _make_classifier(best[2]); regime_model.fit(x_train, regime_target)
        # Chronological OOF meta-features: no in-sample predictions are used for the meta learner.
        meta = np.full((len(x_train), 6), np.nan); start = max(150, len(x_train) // 3); step = max(50, (len(x_train) - start) // 3)
        for end in range(start, len(x_train), step):
            stop = min(end + step, len(x_train)); fold_model = _make_classifier(best[2]); fold_model.fit(x_train[:end], y_train[:end]); fold_er = _make_regressor(best[2]); fold_er.fit(x_train[:end], r_train[:end]); fold_dn = _make_regressor(best[2]); fold_dn.fit(x_train[:end], np.minimum(r_train[:end], 0.0)); fold_v = _make_regressor(best[2]); fold_v.fit(x_train[:end], np.abs(r_train[:end])); meta[end:stop, :] = np.column_stack([fold_model.predict_proba(x_train[end:stop]), fold_er.predict(x_train[end:stop]), fold_dn.predict(x_train[end:stop]), fold_v.predict(x_train[end:stop])])
        valid_meta = np.isfinite(meta).all(axis=1)
        if valid_meta.sum() < 100: return BrainReport("REJECTED", version, {"families": family_scores}, "Insufficient out-of-fold meta-training samples.")
        meta_model = LogisticRegression(max_iter=1000, random_state=42); meta_model.fit(meta[valid_meta], y_train[valid_meta])
        horizon_metrics = self._evaluate_horizons(x_train, x_test, rows[:split], rows[split:])
        bundle = {"direction_model": best_model, "expected_return_model": expected_return, "downside_model": downside_model, "volatility_model": volatility_model, "regime_model": regime_model, "meta_model": meta_model, "family": best[2], "feature_hash": _feature_hash(), "features": FEATURES, "cost_rate": COST_RATE, "horizons": HORIZONS, "horizon_metrics": horizon_metrics, "sequence_model_evaluation": {"status": "NOT_REQUIRED", "reason": "Available Stage 3 dataset is tabular point-in-time data; sequence architecture is deferred until sequence coverage and sample volume justify it."}}
        tmp = self.artifact_path.with_suffix(".tmp"); joblib.dump(bundle, tmp); tmp.replace(self.artifact_path)
        manifest = {"version": version, "features": FEATURES, "feature_hash": _feature_hash(), "family": best[2], "metrics": {"families": family_scores, "promoted": best[2], "horizons": horizon_metrics}, "cost_rate": COST_RATE, "horizons": HORIZONS, "python": platform.python_version()}
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)); self.bundle = bundle; self.version = version
        return BrainReport("PROMOTED", version, manifest["metrics"], "Candidate ensemble beat the logistic baseline on chronological out-of-sample net return and balanced accuracy.", str(self.artifact_path))

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        if self.bundle is None: return {"trained": False, "abstain": True, "version": self.version, "decision": "NO_TRADE", "reason": "No promoted predictive brain artifact is available."}
        x = np.asarray([[float(features.get(k, 0.0) or 0.0) for k in FEATURES]], dtype=float); dm = self.bundle["direction_model"]; base_probs = dm.predict_proba(x)[0]; er = float(self.bundle["expected_return_model"].predict(x)[0]); dn = float(self.bundle["downside_model"].predict(x)[0]); vol = float(self.bundle["volatility_model"].predict(x)[0]); regime = int(self.bundle["regime_model"].predict(x)[0]); mf = np.column_stack([base_probs, [er], [dn], [vol]]); mp = self.bundle["meta_model"].predict_proba(mf)[0]; mc = list(self.bundle["meta_model"].classes_); probs = {"short": float(mp[mc.index(-1)]) if -1 in mc else 0.0, "flat": float(mp[mc.index(0)]) if 0 in mc else 0.0, "long": float(mp[mc.index(1)]) if 1 in mc else 0.0}; direction = max(probs, key=probs.get); edge = er - self.bundle["cost_rate"]; uncertainty = 1.0 - max(probs.values()); abstain = direction == "flat" or max(probs.values()) < .55 or edge <= 0.0 or not np.isfinite([er, dn, vol]).all(); return {"trained": True, "abstain": abstain, "version": self.version, "decision": "NO_TRADE" if abstain else direction.upper(), "probabilities": probs, "expected_return": er, "expected_edge_after_cost": edge, "downside": dn, "volatility": vol, "regime": regime, "uncertainty": float(uncertainty), "model_family": self.bundle["family"]}

    def manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists(): return None
        try: return json.loads(self.manifest_path.read_text())
        except Exception: return None
