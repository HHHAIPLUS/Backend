from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import platform
import os
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    VotingClassifier,
    RandomForestRegressor,
)
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler

from app.ml.features import FEATURES
from app.ml.model_validation import promotion_gate

MODEL_FAMILIES = (
    "logistic_regression",
    "binary_logistic_selective",
    "trend_regime",
    "binary_xgb_selective",
    "long_only_xgboost",
    "short_only_xgboost",
    "xgboost",
    "xgboost_directional_weighted",
    "blended_directional",
    "return_weighted_xgboost",
    "logistic_regression_unweighted",
    "logistic_regression_directional",
    "extra_trees",
    "random_forest_unweighted",
    "random_forest_balanced",
    "hist_gradient_boosting",
    "hist_gradient_boosting_balanced",
    "soft_voting",
    "random_forest",
    "gaussian_nb",
    "sgd_logistic",
)
RETURN_BASE_FAMILIES = (
    "ridge",
    "extra_trees_regressor",
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
    "xgboost_regressor",
)
RETURN_SIGNAL_THRESHOLDS = (0.0005, 0.0008, 0.0010, 0.0012, 0.0014, 0.0018, 0.0022, 0.0030, 0.0040, 0.0050)
RETURN_FAMILIES = tuple(
    f"{family}@{threshold:.4f}"
    for family in RETURN_BASE_FAMILIES
    for threshold in RETURN_SIGNAL_THRESHOLDS
)
HORIZONS = (1, 3, 6, 12)
_configured_label_threshold = (
    os.getenv("HHHAI_PHASE2_FIXED_LABEL_THRESHOLD", "").strip()
    or os.getenv("HHHAI_BRAIN_LABEL_THRESHOLD", "").strip()
)
if _configured_label_threshold:
    LABEL_THRESHOLDS = (float(_configured_label_threshold),)
else:
    LABEL_THRESHOLDS = (0.0010, 0.0015, 0.0020, 0.0025)
LABEL_MODE = os.getenv("HHHAI_PHASE2_LABEL_MODE", "fixed").strip().lower()
COST_RATE = 0.0014
ARTIFACT_SCHEMA = 4
MAX_LABEL_HORIZON = max(HORIZONS)
FIXED_HORIZON = int(os.getenv("HHHAI_PHASE2_FIXED_HORIZON", "0") or "0")
MIN_OOS_TRADES = 100
EXECUTION_PROFILES = ("confidence", "trend", "volatility", "momentum", "long_only")


class BinaryDirectionalClassifier(ClassifierMixin, BaseEstimator):
    """Binary long/short learner; abstention supplies the third no-trade class."""
    def __init__(self):
        self.model_ = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1800, class_weight="balanced", random_state=42)),
        ])
        self.classes_ = np.asarray([-1, 1], dtype=int)

    def fit(self, x, y, sample_weight=None):
        self.model_.fit(x, np.asarray(y, dtype=int), **({"model__sample_weight": sample_weight} if sample_weight is not None else {}))
        return self

    def predict(self, x):
        return self.model_.predict(x).astype(int)

    def predict_proba(self, x):
        return np.asarray(self.model_.predict_proba(x), dtype=float)

class BlendedDirectionalClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.lr_=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=2200,class_weight='balanced',C=0.7,random_state=42))])
        self.xgb_=XGBClassifier(n_estimators=420,max_depth=3,learning_rate=0.025,subsample=0.82,colsample_bytree=0.82,min_child_weight=10,reg_alpha=0.20,reg_lambda=4.0,objective='multi:softprob',num_class=3,eval_metric='mlogloss',tree_method='hist',n_jobs=1,random_state=44)
        self.classes_=np.asarray([-1,0,1],dtype=int)
    def fit(self,x,y,sample_weight=None):
        y=np.asarray(y,dtype=int); mapping={-1:0,0:1,1:2}; enc=np.asarray([mapping[int(v)] for v in y],dtype=int)
        w=_balanced_weights(y) if sample_weight is None else np.asarray(sample_weight,dtype=float)
        self.lr_.fit(x,y,model__sample_weight=w); self.xgb_.fit(x,enc,sample_weight=w); return self
    def predict_proba(self,x):
        return 0.45*np.asarray(self.lr_.predict_proba(x),dtype=float)+0.55*np.asarray(self.xgb_.predict_proba(x),dtype=float)
    def predict(self,x):
        return self.classes_[np.argmax(self.predict_proba(x),axis=1)]


def _balanced_weights(y):
    y = np.asarray(y, dtype=int)
    labels, counts = np.unique(y, return_counts=True)
    total = float(len(y))
    k = float(len(labels))
    weights = {int(label): total / max(k * float(count), 1.0) for label, count in zip(labels, counts)}
    return np.asarray([weights[int(label)] for label in y], dtype=float)


class SideOnlyXGBClassifier(ClassifierMixin, BaseEstimator):
    """Binary directional learner exposed as long/flat or short/flat probabilities."""
    def __init__(self, side: int):
        self.side = int(side)
        self.other = -self.side
        self.model_ = XGBClassifier(
            n_estimators=260, max_depth=4, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=10,
            reg_alpha=0.10, reg_lambda=3.0, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=42,
        )
        self.classes_ = np.asarray([-1, 0, 1], dtype=int)

    def fit(self, x, y, sample_weight=None):
        target = (np.asarray(y, dtype=int) == self.side).astype(int)
        weights = _balanced_weights(target) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.model_.fit(x, target, sample_weight=weights)
        return self

    def predict_proba(self, x):
        p = np.asarray(self.model_.predict_proba(x), dtype=float)[:, 1]
        out = np.zeros((len(p), 3), dtype=float)
        out[:, 1] = 1.0 - p
        out[:, 2 if self.side == 1 else 0] = p
        return out

    def predict(self, x):
        p = self.predict_proba(x)[:, 2 if self.side == 1 else 0]
        # Lower entry boundary is fixed at 0.35; calibration still controls confidence/abstention.
        return np.where(p >= 0.35, self.side, 0).astype(int)


class BinaryXGBSelectiveClassifier(ClassifierMixin, BaseEstimator):
    """Binary long/short XGBoost trained only on actionable labels; low confidence is flat."""
    def __init__(self):
        self.model_ = XGBClassifier(
            n_estimators=320, max_depth=3, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=12,
            reg_alpha=0.15, reg_lambda=4.0, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=42,
        )
        self.classes_ = np.asarray([-1, 0, 1], dtype=int)

    def fit(self, x, y, sample_weight=None):
        y = np.asarray(y, dtype=int)
        mask = y != 0
        target = (y[mask] == 1).astype(int)
        weights = _balanced_weights(target) if sample_weight is None else np.asarray(sample_weight, dtype=float)[mask]
        self.model_.fit(np.asarray(x)[mask], target, sample_weight=weights)
        return self

    def predict_proba(self, x):
        p = np.asarray(self.model_.predict_proba(x), dtype=float)[:, 1]
        out = np.column_stack([1.0-p, np.zeros(len(p)), p])
        return out

    def predict(self, x):
        p = self.predict_proba(x)[:, 2]
        return np.where(p >= 0.5, 1, -1).astype(int)


class TrendRegimeClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic, point-in-time trend/regime baseline used as a registered candidate."""
    def __init__(self, strength=0.0, momentum=0.0, ema=0.0):
        self.strength = float(strength)
        self.momentum = float(momentum)
        self.ema = float(ema)
        self.classes_ = np.asarray([-1, 0, 1], dtype=int)

    def fit(self, x, y, sample_weight=None):
        return self

    def _pred(self, x):
        ts = x[:, FEATURES.index("trend_strength_72")]
        mom = x[:, FEATURES.index("momentum")]
        ema = x[:, FEATURES.index("ema_gap_24_72")]
        long_mask = (ts > self.strength) & (mom > self.momentum) & (ema > self.ema)
        short_mask = (ts < -self.strength) & (mom < -self.momentum) & (ema < -self.ema)
        return np.where(long_mask, 1, np.where(short_mask, -1, 0)).astype(int)

    def predict(self, x):
        return self._pred(np.asarray(x, dtype=float))

    def predict_proba(self, x):
        pred = self.predict(x)
        out = np.full((len(pred), 3), 0.05, dtype=float)
        out[:, 1] = 0.90
        out[pred == -1] = (0.90, 0.05, 0.05)
        out[pred == 1] = (0.05, 0.05, 0.90)
        return out


class DirectionalWeightedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Three-class XGBoost that explicitly penalizes missed directional moves."""
    def __init__(self):
        self.model_ = XGBClassifier(
            n_estimators=320, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=10,
            reg_alpha=0.10, reg_lambda=3.0, objective="multi:softprob",
            num_class=3, eval_metric="mlogloss", tree_method="hist",
            n_jobs=1, random_state=43,
        )
        self.classes_ = np.asarray([-1, 0, 1], dtype=int)

    def fit(self, x, y, sample_weight=None):
        mapping = {-1: 0, 0: 1, 1: 2}
        y_arr = np.asarray(y, dtype=int)
        encoded = np.asarray([mapping[int(v)] for v in y_arr], dtype=int)
        base = np.asarray([2.0 if v != 0 else 0.5 for v in y_arr], dtype=float)
        weights = base if sample_weight is None else base * np.asarray(sample_weight, dtype=float)
        self.model_.fit(x, encoded, sample_weight=weights)
        return self

    def predict(self, x):
        encoded = np.asarray(self.model_.predict(x), dtype=int)
        return self.classes_[encoded]

    def predict_proba(self, x):
        return np.asarray(self.model_.predict_proba(x), dtype=float)


class ReturnWeightedXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost direction learner that emphasizes economically meaningful moves."""
    def __init__(self):
        self.model_ = XGBClassifier(
            n_estimators=420, max_depth=3, learning_rate=0.025,
            subsample=0.82, colsample_bytree=0.82, min_child_weight=8,
            reg_alpha=0.15, reg_lambda=4.0, objective="multi:softprob",
            num_class=3, eval_metric="mlogloss", tree_method="hist",
            n_jobs=1, random_state=47,
        )
        self.classes_ = np.asarray([-1, 0, 1], dtype=int)

    def fit(self, x, y, sample_weight=None):
        mapping = {-1: 0, 0: 1, 1: 2}
        y_arr = np.asarray(y, dtype=int)
        encoded = np.asarray([mapping[int(v)] for v in y_arr], dtype=int)
        base = _balanced_weights(y_arr)
        weights = base if sample_weight is None else base * np.asarray(sample_weight, dtype=float)
        self.model_.fit(x, encoded, sample_weight=weights)
        return self

    def predict(self, x):
        encoded = np.asarray(self.model_.predict(x), dtype=int)
        return self.classes_[encoded]

    def predict_proba(self, x):
        return np.asarray(self.model_.predict_proba(x), dtype=float)


class XGBDirectionalClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost wrapper that preserves the {-1, 0, 1} public class contract."""
    def __init__(self):
        self.model_ = XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.80,
            colsample_bytree=0.80,
            min_child_weight=12,
            reg_alpha=0.10,
            reg_lambda=3.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=1,
            random_state=42,
        )
        self.classes_ = np.asarray([-1, 0, 1], dtype=int)

    def fit(self, x, y, sample_weight=None):
        mapping = {-1: 0, 0: 1, 1: 2}
        encoded = np.asarray([mapping[int(v)] for v in y], dtype=int)
        weights = _balanced_weights(encoded) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.model_.fit(x, encoded, sample_weight=weights)
        return self

    def predict(self, x):
        encoded = np.asarray(self.model_.predict(x), dtype=int)
        return self.classes_[encoded]

    def predict_proba(self, x):
        return np.asarray(self.model_.predict_proba(x), dtype=float)


@dataclass
class BrainReport:
    status: str
    version: str
    metrics: dict[str, Any]
    reason: str
    artifact: str | None = None


def _feature_hash() -> str:
    return hashlib.sha256("|".join(FEATURES).encode()).hexdigest()


def _x(rows):
    x = np.asarray(
        [[float(r.get("features", {}).get(k, 0.0) or 0.0) for k in FEATURES] for r in rows],
        dtype=float,
    )
    if not np.isfinite(x).all():
        raise ValueError("Predictive features contain non-finite values.")
    return x


TARGET_RETURN_FIELD = os.getenv("HHHAI_PHASE2_TARGET_RETURN", "close")
TRAIN_FILTER_MULTIPLIER = max(1.0, float(os.getenv("HHHAI_PHASE2_TRAIN_FILTER_MULTIPLIER", "1.0")))

def _economic_sample_weights(returns, threshold):
    """Emphasize larger moves while capping outlier influence."""
    r = np.abs(np.asarray(returns, dtype=float))
    scale = max(float(threshold), 1e-6)
    return np.sqrt(np.clip(r / scale, 0.5, 4.0))


def _training_filter_mask(returns, threshold):
    """Keep sufficiently large training outcomes for filtered-label learning."""
    if TRAIN_FILTER_MULTIPLIER <= 1.0:
        return np.ones(len(returns), dtype=bool)
    r = np.asarray(returns, dtype=float)
    mask = np.abs(r) >= float(threshold) * TRAIN_FILTER_MULTIPLIER
    return mask



def _future_return(rows, horizon):
    values = []
    for row in rows:
        by = row.get("barrier_return_by_horizon", {}) if TARGET_RETURN_FIELD == "barrier" else row.get("outcome_return_by_horizon", {})
        value = by.get(str(horizon), by.get(horizon))
        if value is None and horizon == int(row.get("outcome_horizon", 6)):
            value = row.get("outcome_return")
        if value is None:
            raise ValueError(f"Missing point-in-time outcome for horizon {horizon}")
        values.append(float(value))
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Outcome returns contain non-finite values.")
    return result


def _label_bounds(values):
    values = np.asarray(values, dtype=float)
    if LABEL_MODE != "quantile":
        return None
    if len(values) < 30:
        raise ValueError("Not enough training returns for quantile labels.")
    q1, q2 = np.quantile(values, [0.33, 0.67])
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 >= q2:
        raise ValueError("Quantile label boundaries are invalid.")
    return float(q1), float(q2)


def _direction_target(values, threshold=COST_RATE, bounds=None):
    values = np.asarray(values, dtype=float)
    if LABEL_MODE == "quantile":
        if bounds is None:
            bounds = _label_bounds(values)
        low, high = bounds
        return np.where(values > high, 1, np.where(values < low, -1, 0))
    return np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))


def _classifier(family):
    if family == "trend_regime":
        return TrendRegimeClassifier(strength=0.0, momentum=0.0, ema=0.0)
    if family == "binary_xgb_selective":
        return BinaryXGBSelectiveClassifier()
    if family == "binary_logistic_selective":
        return BinaryDirectionalClassifier()
    if family == "long_only_xgboost":
        return SideOnlyXGBClassifier(1)
    if family == "short_only_xgboost":
        return SideOnlyXGBClassifier(-1)
    if family == "xgboost":
        return XGBDirectionalClassifier()
    if family == "xgboost_directional_weighted":
        return DirectionalWeightedXGBClassifier()
    if family == "blended_directional":
        return BlendedDirectionalClassifier()
    if family == "return_weighted_xgboost":
        return ReturnWeightedXGBClassifier()
    if family == "logistic_regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        ])
    if family == "logistic_regression_unweighted":
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight=None, random_state=42))])
    if family == "logistic_regression_directional":
        return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1800, class_weight={-1: 2.0, 0: 0.5, 1: 2.0}, random_state=42))])
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=60, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=1
        )
    if family == "random_forest_unweighted":
        return RandomForestClassifier(
            n_estimators=80, min_samples_leaf=12, max_features="sqrt",
            class_weight=None, random_state=42, n_jobs=1
        )
    if family == "random_forest_balanced":
        return RandomForestClassifier(
            n_estimators=80, min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced_subsample", random_state=42, n_jobs=1
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=140, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=42
        )
    if family == "hist_gradient_boosting_balanced":
        return HistGradientBoostingClassifier(
            max_iter=140, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=42
        )
    if family == "soft_voting":
        return VotingClassifier(
            estimators=[
                ("lr", Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))])),
                ("et", ExtraTreesClassifier(n_estimators=60, min_samples_leaf=10, class_weight="balanced", random_state=42, n_jobs=1)),
                ("rf", RandomForestClassifier(n_estimators=70, min_samples_leaf=10, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=1)),
            ],
            voting="soft",
            weights=[2, 1, 1],
            flatten_transform=True,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=120, min_samples_leaf=12, max_features="sqrt",
            class_weight="balanced_subsample", random_state=42, n_jobs=1
        )
    if family == "gaussian_nb":
        return GaussianNB(var_smoothing=1e-8)
    if family == "sgd_logistic":
        return Pipeline([(
            "scale", StandardScaler(),
        ), (
            "model", SGDClassifier(loss="log_loss", alpha=1e-4, class_weight="balanced",
                                   max_iter=2500, tol=1e-4, random_state=42, early_stopping=False,
                                   validation_fraction=0.12, n_iter_no_change=20),
        )])
    raise ValueError(f"Unknown model family: {family}")


def _regressor(family):
    base = family.split("@", 1)[0]
    if base == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])
    if base == "extra_trees_regressor":
        return ExtraTreesRegressor(n_estimators=60, min_samples_leaf=10, random_state=42, n_jobs=1)
    if base == "random_forest_regressor":
        return RandomForestRegressor(n_estimators=80, min_samples_leaf=12, max_features="sqrt", random_state=42, n_jobs=1)
    if base == "hist_gradient_boosting_regressor":
        return HistGradientBoostingRegressor(
            max_iter=140, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False, random_state=42
        )
    if base == "xgboost_regressor":
        return XGBRegressor(
            n_estimators=320, max_depth=4, learning_rate=0.03,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=12,
            reg_alpha=0.10, reg_lambda=3.0, objective="reg:squarederror",
            eval_metric="rmse", tree_method="hist", n_jobs=1, random_state=42,
        )
    raise ValueError(f"Unknown return model family: {family}")

def _return_signal_threshold(family):
    if "@" not in family:
        return COST_RATE
    return float(family.rsplit("@", 1)[1])


def _net_returns(returns, pred):
    """Apply returns to held positions and charge cost only when position changes."""
    returns = np.asarray(returns, dtype=float)
    pred = np.asarray(pred, dtype=int)
    position = np.where(pred == 1, 1.0, np.where(pred == -1, -1.0, 0.0))
    prev = np.r_[0.0, position[:-1]]
    gross = position * returns
    change = np.abs(position - prev)
    costs = (COST_RATE / 2.0) * change
    if len(costs) and position[-1] != 0.0:
        costs[-1] += COST_RATE / 2.0
    return gross - costs


def _apply_execution_profile(pred, x, profile):
    """Apply one of five pre-registered execution filters using only point-in-time features."""
    pred = np.asarray(pred, dtype=int).copy()
    if profile == "confidence":
        return pred
    if profile == "trend":
        trend = np.asarray(x, dtype=float)[:, FEATURES.index("trend_strength_72")]
        pred[(pred == 1) & (trend <= 0.0)] = 0
        pred[(pred == -1) & (trend >= 0.0)] = 0
    elif profile == "volatility":
        vol = np.asarray(x, dtype=float)[:, FEATURES.index("volatility_72")]
        # Avoid the highest-volatility tail where hourly execution costs and reversals are largest.
        cutoff = float(np.quantile(vol, 0.85)) if len(vol) else 1.0
        pred[vol > cutoff] = 0
    elif profile == "momentum":
        momentum = np.asarray(x, dtype=float)[:, FEATURES.index("momentum")]
        pred[(pred == 1) & (momentum <= 0.0)] = 0
        pred[(pred == -1) & (momentum >= 0.0)] = 0
    elif profile == "long_only":
        pred[pred == -1] = 0
    return pred

def _apply_regime_filter(pred, x, threshold):
    """Filter directional signals using the precomputed medium-term trend."""
    pred = np.asarray(pred, dtype=int).copy()
    if threshold is None:
        return pred
    idx = FEATURES.index("ema_gap_24_72")
    gap = np.asarray(x, dtype=float)[:, idx]
    pred[(pred == 1) & (gap <= float(threshold))] = 0
    pred[(pred == -1) & (gap >= -float(threshold))] = 0
    return pred


def _metrics(y, pred, probs, classes, returns):
    net = _net_returns(returns, pred)
    traded = pred != 0
    mapping = {int(c): i for i, c in enumerate(classes)}
    if all(c in mapping for c in (-1, 0, 1)):
        ordered = np.column_stack([probs[:, mapping[-1]], probs[:, mapping[0]], probs[:, mapping[1]]])
        truth = np.column_stack([(y == -1), (y == 0), (y == 1)]).astype(float)
        brier = float(np.mean(np.sum((ordered - truth) ** 2, axis=1)))
    else:
        brier = float("nan")
    equity = np.cumsum(net)
    peak = np.maximum.accumulate(np.r_[0.0, equity])
    dd = float(np.max(peak[1:] - equity)) if len(equity) else 0.0
    pred_counts = np.bincount(pred + 1, minlength=3).astype(float)
    prediction_class_fractions = (pred_counts / max(1, len(pred))).tolist()
    side = {}
    for label, name in ((-1, "short"), (1, "long")):
        mask = pred == label
        side[name] = {
            "samples": int(mask.sum()),
            "precision": float(
                precision_score(y[mask], pred[mask], labels=[label], average="micro", zero_division=0)
            ) if mask.any() else 0.0,
            "avg_net_return": float(net[mask].mean()) if mask.any() else 0.0,
        }
    return {
        "samples": int(len(y)),
        "trades": int(traded.sum()),
        "trade_rate": float(traded.mean()),
        "prediction_class_fractions": prediction_class_fractions,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "avg_net_return": float(net.mean()),
        "avg_trade_net_return": float(net[traded].mean()) if traded.any() else 0.0,
        "total_net_return": float(net.sum()),
        "max_drawdown": dd,
        "calibration_brier": brier,
        "mean_confidence": float(np.max(probs, axis=1).mean()),
        "long": side["long"],
        "short": side["short"],
    }


def _regression_signal(expected, min_edge):
    expected = np.asarray(expected, dtype=float)
    signal = np.where(expected > min_edge, 1, np.where(expected < -min_edge, -1, 0))
    return signal.astype(int)


def _purged_time_splits(rows, test_fraction=.20):
    """Return train/validation/calibration/OOS index ranges with label embargoes.

    Every supervised label looks forward up to MAX_LABEL_HORIZON candles. Rows
    immediately before a boundary are therefore purged so their future outcome
    cannot cross into the next partition.
    """
    timestamps = [str(r.get("observed_at", "")) for r in rows]
    unique_times = sorted(set(timestamps))
    if len(unique_times) < 40:
        raise ValueError("Not enough unique timestamps for chronological evaluation.")
    fixed_start = os.getenv("PHASE2_OOS_START", "").strip()
    fixed_end = os.getenv("PHASE2_OOS_END", "").strip()
    if fixed_start:
        if not fixed_end or not (unique_times[0] <= fixed_start < fixed_end <= unique_times[-1]):
            raise ValueError("Configured historical OOS window is outside available data.")
        oos_start_time = fixed_start
        oos_start = next(i for i, t in enumerate(timestamps) if t >= oos_start_time)
        oos_end = next((i for i, t in enumerate(timestamps) if t >= fixed_end), len(rows))
    else:
        oos_time_pos = int(len(unique_times) * (1.0 - test_fraction))
        oos_time_pos = max(1, min(len(unique_times) - 1, oos_time_pos))
        oos_start_time = unique_times[oos_time_pos]
        oos_start = next(i for i, t in enumerate(timestamps) if t >= oos_start_time)
        oos_end = len(rows)

    pre_times = sorted(set(timestamps[:oos_start]))
    if len(pre_times) < 30:
        raise ValueError("Not enough pre-OOS timestamps for train/validation/calibration.")
    val_time_pos = int(len(pre_times) * .67)
    val_time_pos = max(1, min(len(pre_times) - 2, val_time_pos))
    cal_time_pos = int(len(pre_times) * .84)
    cal_time_pos = max(val_time_pos + 1, min(len(pre_times) - 1, cal_time_pos))

    val_start_time = pre_times[val_time_pos]
    cal_start_time = pre_times[cal_time_pos]
    val_start = next(i for i, t in enumerate(timestamps[:oos_start]) if t >= val_start_time)
    cal_start = next(i for i, t in enumerate(timestamps[:oos_start]) if t >= cal_start_time)

    # Purge the tail of every partition. Since the dataset is single-symbol
    # for production Phase 2, this is a candle-count embargo. For multi-symbol
    # data, timestamps are grouped, so an entire timestamp remains together.
    purge = MAX_LABEL_HORIZON
    train_end = max(0, val_start - purge)
    val_end = max(val_start, cal_start - purge)
    cal_end = max(cal_start, oos_start - purge)

    if train_end < 600 or val_end - val_start < 100 or cal_end - cal_start < 100 or oos_end - oos_start < 100:
        raise ValueError("Purged train/validation/calibration/OOS partitions are too small.")

    return {
        "train": (0, train_end),
        "validation": (val_start, val_end),
        "calibration": (cal_start, cal_end),
        "oos": (oos_start, oos_end),
        "purge_rows": purge,
        "timestamps": {
            "train_end": timestamps[train_end - 1],
            "validation_start": timestamps[val_start],
            "validation_end": timestamps[val_end - 1],
            "calibration_start": timestamps[cal_start],
            "calibration_end": timestamps[cal_end - 1],
            "oos_start": timestamps[oos_start],
            "oos_end": timestamps[oos_end - 1],
        },
    }


def _slice(a, bounds):
    return a[bounds[0]:bounds[1]]


def _calibrate(model, x_cal, y_cal):
    # Calibration is a separate pre-OOS stage. Balance the calibration
    # objective so a flat-heavy target prior cannot turn a useful directional
    # learner into an almost-always-flat predictor merely through probability
    # recalibration. The calibration rows themselves remain strictly
    # chronological and disjoint from train/validation/OOS.
    weights = _balanced_weights(y_cal)
    return CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(
        x_cal, y_cal, sample_weight=weights
    )


class PredictiveBrain:
    """Predictive brain with strict chronological selection, calibration and untouched OOS promotion."""

    def __init__(self, artifact_dir="artifacts"):
        self.path = Path(artifact_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.artifact_path = self.path / "predictive_brain.joblib"
        self.manifest_path = self.path / "predictive_brain_manifest.json"
        self.bundle = None
        self.version = "untrained"
        self._load()

    def _load(self):
        if not self.artifact_path.exists() or not self.manifest_path.exists():
            return
        try:
            manifest = json.loads(self.manifest_path.read_text())
            bundle = joblib.load(self.artifact_path)
            if manifest.get("schema_version") != ARTIFACT_SCHEMA:
                raise ValueError("Predictive brain artifact schema mismatch")
            if manifest.get("features") != FEATURES or bundle.get("feature_hash") != _feature_hash():
                raise ValueError("Predictive brain feature fingerprint mismatch")
            if manifest.get("promotion", {}).get("promoted") is not True:
                raise ValueError("Artifact is not a promoted candidate")
            self.bundle = bundle
            self.version = str(manifest["version"])
        except Exception:
            self.bundle = None
            self.version = "untrained"

    def train(self, rows, version="brain-v1", test_fraction=.20):
        if len(rows) < 1200:
            return BrainReport(
                "REJECTED", version, {"rows": len(rows)},
                "At least 1200 point-in-time rows are required for independent train, validation, calibration and OOS testing."
            )
        rows = sorted(rows, key=lambda r: str(r.get("observed_at", "")))
        x = _x(rows)
        n = len(rows)
        splits = _purged_time_splits(rows, test_fraction)
        tr = splits["train"]
        va = splits["validation"]
        ca = splits["calibration"]
        oo = splits["oos"]

        # Selection uses two chronological validation folds inside the pre-OOS
        # history. This reduces dependence on one favorable validation slice while
        # keeping the final OOS period completely untouched.
        selection_end = va[1]
        purge = MAX_LABEL_HORIZON
        fold1_train_end = max(600, int(selection_end * 0.58))
        fold1_val_start = fold1_train_end + purge
        fold1_val_end = int(selection_end * 0.75)
        fold2_train_end = fold1_val_end
        fold2_val_start = fold2_train_end + purge
        fold2_val_end = selection_end
        selection_folds = [
            ((0, fold1_train_end), (fold1_val_start, fold1_val_end)),
            ((0, fold2_train_end), (fold2_val_start, fold2_val_end)),
        ]
        selection_folds = [
            (train_bounds, val_bounds)
            for train_bounds, val_bounds in selection_folds
            if val_bounds[1] - val_bounds[0] >= 100
        ]
        if len(selection_folds) < 2:
            return BrainReport("REJECTED", version, {}, "Multiple chronological validation folds are required for model selection.")

        def aggregate_scores(scores):
            if not scores:
                return {"status": "UNAVAILABLE"}
            numeric = ("accuracy", "balanced_accuracy", "avg_net_return", "avg_trade_net_return", "trade_rate")
            out = {key: float(np.mean([float(s[key]) for s in scores])) for key in numeric if key in scores[0]}
            out["samples"] = int(sum(int(s.get("samples", 0)) for s in scores))
            out["trades"] = int(sum(int(s.get("trades", 0)) for s in scores))
            if all("prediction_class_fractions" in s for s in scores):
                out["prediction_class_fractions"] = [
                    float(np.mean([float(s["prediction_class_fractions"][i]) for s in scores]))
                    for i in range(3)
                ]
            out["total_net_return"] = float(sum(float(s.get("total_net_return", 0.0)) for s in scores))
            out["max_drawdown"] = float(max(float(s.get("max_drawdown", 0.0)) for s in scores))
            out["folds"] = scores
            return out

        horizon_selection = {}
        horizons_to_test = (FIXED_HORIZON,) if FIXED_HORIZON in HORIZONS else HORIZONS
        for h in horizons_to_test:
            full_returns = _future_return(rows, h)
            for threshold in LABEL_THRESHOLDS:
                fold_scores = []
                for train_bounds, val_bounds in selection_folds:
                    train_returns = _slice(full_returns, train_bounds)
                    val_returns = _slice(full_returns, val_bounds)
                    label_bounds = _label_bounds(train_returns)
                    y_train_fold = _direction_target(train_returns, threshold, label_bounds)
                    y_val_fold = _direction_target(val_returns, threshold, label_bounds)
                    if len(set(y_train_fold.tolist())) < 3 or len(set(y_val_fold.tolist())) < 3:
                        continue
                    horizon_models = (
                        _classifier("logistic_regression"),
                        _classifier("return_weighted_xgboost"),
                    )
                    horizon_fold_candidates = []
                    for model in horizon_models:
                        if isinstance(model, ReturnWeightedXGBClassifier):
                            model.fit(
                                _slice(x, train_bounds),
                                y_train_fold,
                                sample_weight=_economic_sample_weights(train_returns, float(threshold)),
                            )
                        else:
                            model.fit(_slice(x, train_bounds), y_train_fold)
                        pred = model.predict(_slice(x, val_bounds))
                        probs = model.predict_proba(_slice(x, val_bounds))
                        horizon_fold_candidates.append(_metrics(y_val_fold, pred, probs, model.classes_, val_returns))
                    fold_scores.append(max(
                        horizon_fold_candidates,
                        key=lambda z: (
                            float(z.get("balanced_accuracy", -1e9)),
                            float(z.get("avg_trade_net_return", -1e9)),
                            float(z.get("accuracy", -1e9)),
                        ),
                    ))
                if fold_scores:
                    # Reject validation targets that are dominated by one class.
                    # This prevents a high overall accuracy from coming mainly
                    # from predicting the flat class and is determined entirely
                    # before the untouched OOS period.
                    validation_labels = np.concatenate([
                        _direction_target(
                            _slice(full_returns, val_bounds),
                            threshold,
                            _label_bounds(_slice(full_returns, train_bounds)),
                        )
                        for train_bounds, val_bounds in selection_folds
                    ])
                    counts = np.bincount(validation_labels + 1, minlength=3).astype(float)
                    fractions = counts / max(1, len(validation_labels))
                    if float(fractions.min()) < 0.10:
                        continue
                    horizon_selection[f"{h}:{threshold:.4f}"] = {
                        **aggregate_scores(fold_scores), "horizon": h, "label_threshold": threshold,
                        "validation_class_fractions": fractions.tolist(),
                    }

        viable = [
            v for v in horizon_selection.values()
            if int(v.get("trades", 0)) >= MIN_OOS_TRADES
            and np.isfinite(float(v.get("avg_trade_net_return", 0.0)))
        ]
        if not viable:
            return BrainReport("REJECTED", version, {"horizon_selection": horizon_selection},
                               "No horizon/label threshold produced enough validation trades across the chronological selection folds.")

        # Select the supervised target/horizon primarily by validation
        # classification quality because the untouched OOS gate requires both
        # accuracy and balanced accuracy. Trade expectancy remains the
        # validation-only secondary criterion; OOS is never used here.
        def _selection_key(v):
            # Fixed, pre-OOS selection hierarchy: require meaningful validation
            # classification and economic quality when available; otherwise
            # fail over to the strongest balanced classification result.
            bal = float(v.get("balanced_accuracy", -1e99))
            acc = float(v.get("accuracy", -1e99))
            exp = float(v.get("avg_trade_net_return", -1e99))
            economically_viable = bal >= 0.50 and acc >= 0.52 and exp > 0.0
            total = float(v.get("total_net_return", -1e99))
            dd = float(v.get("max_drawdown", 1e99))
            return (
                1 if economically_viable else 0,
                bal,
                acc,
                exp if economically_viable else -1e99,
                total if economically_viable else -1e99,
                -dd if economically_viable else -1e99,
                float(v.get("trade_rate", 0.0)),
            )

        chosen = max(viable, key=_selection_key)
        chosen_horizon = int(chosen["horizon"])
        chosen_threshold = float(chosen["label_threshold"])

        returns = _future_return(rows, chosen_horizon)
        r_train = _slice(returns, tr)
        final_label_bounds = _label_bounds(r_train)
        y = _direction_target(returns, chosen_threshold, final_label_bounds)
        y_train = _slice(y, tr)
        y_val = _slice(y, va)
        y_cal = _slice(y, ca)
        y_oos = _slice(y, oo)
        r_val = _slice(returns, va)
        r_cal = _slice(returns, ca)
        r_oos = _slice(returns, oo)

        if any(len(set(part.tolist())) < 3 for part in (y_train, y_val, y_cal, y_oos)):
            return BrainReport("REJECTED", version, {"chosen_horizon": chosen_horizon, "chosen_threshold": chosen_threshold},
                               "Every chronological partition must contain long, flat and short classes.")

        # Model-family and direction selection are validation-only across the
        # same chronological folds. OOS is not inspected during selection.
        validation_scores = {}
        candidates = []
        # Recent-market adaptation is evaluated only on the pre-OOS validation
        # folds.  A rolling training window is a legitimate time-series
        # hyperparameter: it lets the model forget stale regimes without ever
        # looking at the untouched OOS period.
        TRAIN_WINDOW_CANDIDATES = (0, 1500, 2500, 4000)
        for family in MODEL_FAMILIES:
            for train_window in TRAIN_WINDOW_CANDIDATES:
                fold_scores = []
                inverse_fold_scores = []
                for base_train_bounds, val_bounds in selection_folds:
                    train_end = base_train_bounds[1]
                    train_start = max(0, train_end - train_window) if train_window else base_train_bounds[0]
                    train_bounds = (train_start, train_end)
                    y_train_fold = _slice(y, train_bounds)
                    y_val_fold = _slice(y, val_bounds)
                    model = _classifier(family)
                    x_train_fold = _slice(x, train_bounds)
                    r_train_fold = _slice(returns, train_bounds)
                    filter_mask = _training_filter_mask(r_train_fold, float(chosen_threshold))
                    if filter_mask.sum() >= 300 and len(set(y_train_fold[filter_mask].tolist())) == 3 and family != "return_weighted_xgboost":
                        x_train_fold = x_train_fold[filter_mask]
                        y_train_fold = y_train_fold[filter_mask]
                    if family == "binary_logistic_selective":
                        binary_mask = y_train_fold != 0
                        x_train_fold = x_train_fold[binary_mask]
                        y_train_fold = y_train_fold[binary_mask]
                    if family == "return_weighted_xgboost":
                        model.fit(
                            x_train_fold, y_train_fold,
                            sample_weight=_economic_sample_weights(r_train_fold, float(chosen_threshold)),
                        )
                    elif family == "hist_gradient_boosting_balanced":
                        counts = np.bincount(y_train_fold + 1, minlength=3).astype(float)
                        weights = np.asarray([1.0 / max(counts[label + 1], 1.0) for label in y_train_fold])
                        weights *= len(weights) / max(weights.sum(), 1e-12)
                        model.fit(x_train_fold, y_train_fold, sample_weight=weights)
                    else:
                        model.fit(x_train_fold, y_train_fold)
                    pred = model.predict(_slice(x, val_bounds))
                    probs = model.predict_proba(_slice(x, val_bounds))
                    fold_scores.append(_metrics(y_val_fold, pred, probs, model.classes_, _slice(returns, val_bounds)))
                    inv_pred = np.where(pred == 1, -1, np.where(pred == -1, 1, 0))
                    inv_probs = probs.copy()
                    mapping = {int(c): i for i, c in enumerate(model.classes_)}
                    if all(c in mapping for c in (-1, 0, 1)):
                        inv_probs[:, mapping[-1]], inv_probs[:, mapping[1]] = (
                            probs[:, mapping[1]], probs[:, mapping[-1]]
                        )
                    inverse_fold_scores.append(
                        _metrics(y_val_fold, inv_pred, inv_probs, model.classes_, _slice(returns, val_bounds))
                    )
                score = aggregate_scores(fold_scores)
                inv_score = aggregate_scores(inverse_fold_scores)
                key = f"{family}@window={train_window}"
                validation_scores[key] = score
                validation_scores[key + "_inverse"] = inv_score
                if score.get("status") != "UNAVAILABLE":
                    candidates.append((score["avg_trade_net_return"], score["balanced_accuracy"], family, False, train_window))
                if inv_score.get("status") != "UNAVAILABLE":
                    candidates.append((inv_score["avg_trade_net_return"], inv_score["balanced_accuracy"], family, True, train_window))

        for family in RETURN_FAMILIES:
            for train_window in TRAIN_WINDOW_CANDIDATES:
                fold_scores = []
                for base_train_bounds, val_bounds in selection_folds:
                    train_end = base_train_bounds[1]
                    train_start = max(0, train_end - train_window) if train_window else base_train_bounds[0]
                    train_bounds = (train_start, train_end)
                    reg = _regressor(family)
                    reg.fit(_slice(x, train_bounds), _slice(returns, train_bounds))
                    expected = reg.predict(_slice(x, val_bounds))
                    pred = _regression_signal(expected, _return_signal_threshold(family))
                    probs = np.column_stack([
                        np.where(pred == -1, 0.90, 0.05),
                        np.where(pred == 0, 0.90, 0.05),
                        np.where(pred == 1, 0.90, 0.05),
                    ])
                    fold_scores.append(_metrics(_slice(y, val_bounds), pred, probs, np.array([-1, 0, 1]), _slice(returns, val_bounds)))
                score = aggregate_scores(fold_scores)
                key = f"{family}@window={train_window}"
                validation_scores[key] = score
                if score.get("status") != "UNAVAILABLE":
                    candidates.append((score["avg_trade_net_return"], score["balanced_accuracy"], family, False, train_window))

        # Return regressors remain useful diagnostics, but they must not be
        # eligible to become the Phase 2 directional brain. Their current
        # probability vectors are synthetic (0.90/0.05/0.05), so treating them
        # as calibrated three-class classifiers can manufacture misleading
        # confidence and allow a low-trade regression to win model selection.
        # Phase 2 requires genuine directional classification evidence.
        candidates = [candidate for candidate in candidates if candidate[2] in MODEL_FAMILIES]

        if not candidates:
            return BrainReport("REJECTED", version, {"validation_families": validation_scores},
                               "No valid model family was evaluated.")

        # Model selection is validation-only. Prefer candidates that already
        # demonstrate positive validation expectancy and sufficient coverage.
        def _candidate_score(c):
            family_name = c[2]
            window = int(c[4])
            key = f"{family_name}@window={window}" + ("_inverse" if c[3] else "")
            score = validation_scores[key]
            bal = float(score.get("balanced_accuracy", -1e99))
            acc = float(score.get("accuracy", -1e99))
            exp = float(score.get("avg_trade_net_return", -1e99))
            total = float(score.get("total_net_return", -1e99))
            dd = float(score.get("max_drawdown", 1e99))
            predicted_fractions = score.get("prediction_class_fractions", [0.0, 0.0, 0.0])
            directional_coverage_ok = float(predicted_fractions[0]) >= 0.02 and float(predicted_fractions[2]) >= 0.02
            classification_ok = bal >= 0.50 and acc >= 0.52
            economic_ok = exp > 0.0 and total > 0.0 and dd <= 0.15
            return (
                1 if classification_ok else 0,
                1 if economic_ok else 0,
                1 if directional_coverage_ok else 0,
                bal,
                acc,
                exp,
                total,
                -dd,
                float(score.get("trade_rate", 0.0)),
                -window,
            )

        fixed_family = os.getenv("HHHAI_PHASE2_FIXED_MODEL_FAMILY", "").strip()
        fixed_window = int(os.getenv("HHHAI_PHASE2_FIXED_TRAIN_WINDOW", "0") or "0")
        if fixed_family:
            if fixed_family not in MODEL_FAMILIES:
                return BrainReport("REJECTED", version, {"validation_families": validation_scores},
                                   "Phase 2 fixed model family must be a genuine directional classifier.")
            fixed = [candidate for candidate in candidates if candidate[2] == fixed_family and candidate[4] == fixed_window]
            if not fixed:
                return BrainReport("REJECTED", version, {"validation_families": validation_scores},
                                   f"Requested fixed model family/window unavailable: {fixed_family}@{fixed_window}")
            best = fixed[0]
        else:
            best = max(candidates, key=_candidate_score)
        family = best[2]
        invert_direction = bool(best[3])
        train_window = int(best[4])

        # Fit the selected family on train+validation, then calibrate on a
        # completely separate calibration period. OOS remains untouched.
        fit_end = va[1]
        fit_start = max(0, fit_end - train_window) if train_window else 0
        fit_bounds = (fit_start, fit_end)
        x_fit = _slice(x, fit_bounds)
        y_fit = _slice(y, fit_bounds)
        if family in MODEL_FAMILIES:
            raw_direction = _classifier(family)
            fit_mask = _training_filter_mask(_slice(returns, fit_bounds), float(chosen_threshold))
            if fit_mask.sum() >= 300 and len(set(y_fit[fit_mask].tolist())) == 3 and family != "return_weighted_xgboost":
                x_fit_direction = x_fit[fit_mask]
                y_fit_direction = y_fit[fit_mask]
            else:
                x_fit_direction = x_fit
                y_fit_direction = y_fit
            if family == "binary_logistic_selective":
                binary_mask = y_fit_direction != 0
                x_fit_direction = x_fit_direction[binary_mask]
                y_fit_direction = y_fit_direction[binary_mask]
            if family == "return_weighted_xgboost":
                raw_direction.fit(
                    x_fit_direction, y_fit_direction,
                    sample_weight=_economic_sample_weights(
                        _slice(returns, fit_bounds), float(chosen_threshold)
                    ),
                )
            elif family == "hist_gradient_boosting_balanced":
                counts = np.bincount(y_fit_direction + 1, minlength=3).astype(float)
                weights = np.asarray([1.0 / max(counts[label + 1], 1.0) for label in y_fit_direction])
                weights *= len(weights) / max(weights.sum(), 1e-12)
                raw_direction.fit(x_fit_direction, y_fit_direction, sample_weight=weights)
            else:
                raw_direction.fit(x_fit_direction, y_fit_direction)

            # Pre-OOS collapse guard: a model selected on older validation
            # history can become dominated by the flat class after the final
            # train+validation fit. Detect that using the untouched-free
            # calibration period and switch to a class-balanced logistic
            # learner only when the selected model produces less than 2%
            # long or short signals. This is model-stability protection, not
            # an OOS tuning rule.
            direction_fallback_family = None
            if family in MODEL_FAMILIES:
                raw_cal_pred = np.asarray(raw_direction.predict(_slice(x, ca)), dtype=int)
                raw_counts = np.bincount(raw_cal_pred + 1, minlength=3).astype(float)
                raw_fractions = raw_counts / max(1, len(raw_cal_pred))
                if float(raw_fractions[0]) < 0.02 or float(raw_fractions[2]) < 0.02:
                    fallback = _classifier("logistic_regression")
                    fallback.fit(x_fit_direction, y_fit_direction)
                    fallback_pred = np.asarray(fallback.predict(_slice(x, ca)), dtype=int)
                    fallback_counts = np.bincount(fallback_pred + 1, minlength=3).astype(float)
                    fallback_fractions = fallback_counts / max(1, len(fallback_pred))
                    if float(fallback_fractions[0]) >= 0.02 and float(fallback_fractions[2]) >= 0.02:
                        raw_direction = fallback
                        direction_fallback_family = "logistic_regression"

            baseline_raw = _classifier("logistic_regression")
            baseline_raw.fit(x_fit, y_fit)
            # The fixed soft-voting ensemble already averages calibrated
            # component probabilities structurally; wrapping the whole voter in
            # a second post-hoc calibrator can collapse its directional signal.
            # Keep the pre-registered ensemble probabilities intact.
            direction = raw_direction if family in ("soft_voting", "long_only_xgboost", "short_only_xgboost") else _calibrate(raw_direction, _slice(x, ca), y_cal)
            baseline = _calibrate(baseline_raw, _slice(x, ca), y_cal)
        else:
            # Regression candidates are fit on train+validation and use the
            # calibration set only to choose a confidence/abstention threshold.
            raw_reg = _regressor(family)
            # Regression candidates predict the actual future return target.
            raw_reg.fit(x_fit, _slice(returns, fit_bounds))
            direction = raw_reg
            baseline_raw = _classifier("logistic_regression")
            baseline_raw.fit(x_fit, y_fit)
            baseline = _calibrate(baseline_raw, _slice(x, ca), y_cal)

        # Threshold selection is calibration-only. It must retain a meaningful
        # trade sample and cannot inspect OOS.
        cal_pred, cal_prob = self._predict_selected(direction, _slice(x, ca), invert_direction, family)
        confidence = np.max(cal_prob, axis=1)
        selection_threshold = 0.30
        threshold_candidates = []
        regime_thresholds = (None, 0.0, 0.0005, 0.0010, 0.0020, 0.0040)
        # Calibration coverage is a model-selection constraint, not an OOS
        # promotion gate. Requiring 5% of the entire calibration window can
        # reject otherwise valid selective models before their untouched OOS
        # performance is even measured. Keep a small, stable floor here while
        # the authoritative OOS gate remains >= MIN_OOS_TRADES.
        min_cal_trades = max(50, int(len(y_cal) * 0.02))
        min_cal_trade_rate = 0.005
        for threshold in np.arange(0.30, 0.71, 0.02):
            selected_base = cal_pred.copy()
            if family in MODEL_FAMILIES:
                selected_base[confidence < threshold] = 0
            for profile in EXECUTION_PROFILES:
              for regime_threshold in regime_thresholds:
                selected = _apply_execution_profile(
                    selected_base, _slice(x, ca), profile
                )
                selected = _apply_regime_filter(
                    selected, _slice(x, ca), regime_threshold
                )
                traded = selected != 0
                trade_count = int(traded.sum())
                trade_rate = trade_count / max(1, len(selected))
                if trade_count < min_cal_trades or trade_rate < min_cal_trade_rate:
                    continue
                net = _net_returns(r_cal, selected)
                equity = np.cumsum(net)
                peak = np.maximum.accumulate(np.r_[0.0, equity])
                drawdown = float(np.max(peak[1:] - equity)) if len(equity) else 0.0
                avg_trade = float(net[traded].mean())
                cal_metrics = _metrics(y_cal, selected, cal_prob, np.array([-1, 0, 1]), r_cal)
                threshold_candidates.append((
                    float(avg_trade),
                    float(net.sum()),
                    -float(drawdown),
                    float(cal_metrics["balanced_accuracy"]),
                    float(cal_metrics["accuracy"]),
                    float(trade_rate),
                    float(threshold),
                    regime_threshold,
                    profile,
                ))

        if family in MODEL_FAMILIES and not threshold_candidates:
            return BrainReport(
                "REJECTED", version, {"chosen_horizon": chosen_horizon, "chosen_threshold": chosen_threshold},
                "Calibration produced no decision threshold with the required minimum trade coverage."
            )
        regime_filter_threshold = None
        execution_profile = "confidence"
        if threshold_candidates:
            # Economic execution is selected only on the separate calibration
            # period. Prefer positive total net return and controlled drawdown,
            # then expectancy and classification quality. OOS remains untouched.
            def _calibration_key(c):
                avg_trade, total, neg_dd, bal, acc, trade_rate, threshold, regime, profile = c
                dd = -float(neg_dd)
                classification_ok = float(bal) >= 0.50 and float(acc) >= 0.52
                economic_ok = float(total) > 0.0 and dd <= 0.15
                return (
                    1 if classification_ok else 0,
                    1 if economic_ok else 0,
                    float(bal),
                    float(acc),
                    float(total),
                    float(avg_trade),
                    float(neg_dd),
                    float(trade_rate),
                )
            best_calibration = max(threshold_candidates, key=_calibration_key)
            selection_threshold = best_calibration[6]
            regime_filter_threshold = best_calibration[7]
            execution_profile = best_calibration[8]

        candidate_pred, candidate_prob = self._predict_selected(direction, _slice(x, oo), invert_direction, family)
        baseline_pred = baseline.predict(_slice(x, oo))
        baseline_prob = baseline.predict_proba(_slice(x, oo))
        if family in MODEL_FAMILIES:
            candidate_pred[candidate_prob.max(axis=1) < selection_threshold] = 0
        else:
            # _predict_selected already converts the expected return into a
            # {-1,0,1} signal. Do not threshold the discrete signal again.
            candidate_pred = candidate_pred.astype(int)
        candidate_pred = _apply_execution_profile(candidate_pred, _slice(x, oo), execution_profile)
        candidate_pred = _apply_regime_filter(candidate_pred, _slice(x, oo), regime_filter_threshold)

        candidate_metrics = _metrics(y_oos, candidate_pred, candidate_prob, np.array([-1, 0, 1]), r_oos)
        baseline_metrics = _metrics(y_oos, baseline_pred, baseline_prob, baseline.classes_, r_oos)
        # Research-only diagnostic: evaluate all five frozen execution profiles
        # on this already-observed development OOS period. These diagnostics
        # are never used by promotion or model selection.
        execution_oos_profiles = {}
        for profile in EXECUTION_PROFILES:
            profile_pred = self._predict_selected(direction, _slice(x, oo), invert_direction, family)[0]
            if family in MODEL_FAMILIES:
                profile_pred[candidate_prob.max(axis=1) < selection_threshold] = 0
            profile_pred = _apply_execution_profile(profile_pred, _slice(x, oo), profile)
            profile_pred = _apply_regime_filter(profile_pred, _slice(x, oo), regime_filter_threshold)
            execution_oos_profiles[profile] = _metrics(
                y_oos, profile_pred, candidate_prob, np.array([-1, 0, 1]), r_oos
            )

        # Statistical comparison is calculated once, after all choices are
        # frozen. It is evidence, never a tuning signal.
        gate = promotion_gate(
            _net_returns(r_oos, candidate_pred),
            _net_returns(r_oos, baseline_pred),
            candidate_metrics["balanced_accuracy"],
            baseline_metrics["balanced_accuracy"],
            candidate_metrics["max_drawdown"],
            baseline_metrics["max_drawdown"],
            min_samples=MIN_OOS_TRADES,
        )
        absolute_gate = {
            "enough_samples": candidate_metrics["trades"] >= MIN_OOS_TRADES,
            "accuracy_ok": candidate_metrics["accuracy"] >= 0.52,
            "balanced_accuracy_ok": candidate_metrics["balanced_accuracy"] >= 0.50,
            "positive_trade_expectancy": candidate_metrics["avg_trade_net_return"] > 0.0,
            "positive_total_net_return": candidate_metrics["total_net_return"] > 0.0,
            "drawdown_ok": candidate_metrics["max_drawdown"] <= 0.15,
        }

        metrics = {
            "validation_families": validation_scores,
            "baseline_oos": baseline_metrics,
            "candidate_oos": candidate_metrics,
            "promotion": gate,
            "absolute_gate": absolute_gate,
            "split_evidence": splits,
            "chosen_horizon": chosen_horizon,
            "chosen_label_threshold": chosen_threshold,
            "train_window": train_window,
            "decision_threshold": selection_threshold,
            "regime_filter_threshold": regime_filter_threshold,
            "execution_profile": execution_profile,
            "execution_oos_profiles": execution_oos_profiles,
            "cost_rate": COST_RATE,
            "label_mode": LABEL_MODE,
            "label_bounds": final_label_bounds,
        }
        if not all(absolute_gate.values()):
            return BrainReport("REJECTED", version, metrics,
                               "Candidate did not clear the untouched OOS absolute safety gate.")

        horizon_metrics = self._horizon_eval(
            _slice(x, tr), _slice(x, oo), rows[tr[0]:tr[1]], rows[oo[0]:], chosen_threshold
        )
        bundle = {
            "schema_version": ARTIFACT_SCHEMA,
            "direction_model": direction,
            "baseline_model": baseline,
            "expected_return_model": direction if family not in MODEL_FAMILIES else self._fit_return_model(x_fit, _slice(returns, fit_bounds), family),
            "downside_model": self._fit_return_model(x_fit, np.minimum(_slice(returns, fit_bounds), 0.0), family if family in RETURN_FAMILIES else "ridge"),
            "volatility_model": self._fit_return_model(x_fit, np.abs(_slice(returns, fit_bounds)), family if family in RETURN_FAMILIES else "ridge"),
            "regime_model": self._fit_regime_model(x_fit, _slice(returns, fit_bounds), family),
            "abstention_model": None,
            "meta_model": None,
            "family": family,
            "direction_inverted": invert_direction,
            "decision_threshold": selection_threshold,
            "regime_filter_threshold": regime_filter_threshold,
            "execution_profile": execution_profile,
            "feature_hash": _feature_hash(),
            "features": FEATURES,
            "cost_rate": COST_RATE,
            "label_mode": LABEL_MODE,
            "label_bounds": final_label_bounds,
            "horizons": HORIZONS,
            "chosen_horizon": chosen_horizon,
            "label_threshold": chosen_threshold,
            "horizon_selection": horizon_selection,
            "horizon_metrics": horizon_metrics,
            "oos_metrics": {"candidate": candidate_metrics, "baseline": baseline_metrics},
            "promotion": gate,
            "split_evidence": splits,
            "sequence_model_evaluation": {
                "status": "NOT_REQUIRED",
                "reason": "Tabular OHLCV feature coverage is the verified Phase 2 scope; sequence complexity is deferred to a later robustness phase."
            },
        }
        tmp = self.artifact_path.with_suffix(".tmp")
        joblib.dump(bundle, tmp)
        tmp.replace(self.artifact_path)
        manifest = {
            "schema_version": ARTIFACT_SCHEMA,
            "version": version,
            "features": FEATURES,
            "feature_hash": _feature_hash(),
            "family": family,
            "direction_inverted": invert_direction,
            "metrics": metrics | {"horizons": horizon_metrics},
            "promotion": gate,
            "cost_rate": COST_RATE,
            "horizons": HORIZONS,
            "chosen_horizon": chosen_horizon,
            "horizon_selection": horizon_selection,
            "python": platform.python_version(),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        self.bundle = bundle
        self.version = version
        return BrainReport(
            "PROMOTED", version, manifest["metrics"],
            "Candidate cleared independent selection, calibration, untouched OOS absolute gates and paired statistical evidence.",
            str(self.artifact_path),
        )

    def manifest(self) -> dict[str, Any] | None:
        """Return the persisted manifest only for a currently promoted artifact."""
        if not self.manifest_path.exists() or self.bundle is None:
            return None
        try:
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("schema_version") != ARTIFACT_SCHEMA:
                return None
            if manifest.get("promotion", {}).get("promoted") is not True:
                return None
            return manifest
        except Exception:
            return None

    @staticmethod
    def _fit_return_model(x, target, family):
        if family not in RETURN_FAMILIES:
            family = "ridge"
        model = _regressor(family)
        model.fit(x, np.asarray(target, dtype=float))
        return model

    @staticmethod
    def _fit_regime_model(x, returns, family):
        from sklearn.dummy import DummyClassifier
        rv = np.abs(np.asarray(returns, dtype=float))
        q = np.quantile(rv, [0.33, 0.66])
        target = np.where(rv > q[1], 2, np.where(rv > q[0], 1, 0))
        if len(np.unique(target)) < 2:
            model = DummyClassifier(strategy="most_frequent")
        elif family in MODEL_FAMILIES:
            model = _classifier(family)
        else:
            model = _classifier("logistic_regression")
        model.fit(x, target)
        return model

    @staticmethod
    def _predict_selected(model, x, invert_direction, family):
        if family in MODEL_FAMILIES:
            probs = model.predict_proba(x)
            if family == "binary_logistic_selective":
                raw = model.predict(x).astype(int)
                classes = np.asarray(model.classes_, dtype=int)
                mapping = {int(v): i for i, v in enumerate(classes)}
                directional = np.column_stack([probs[:, mapping[-1]], probs[:, mapping[1]]])
                confidence = np.max(directional, axis=1)
                pred = np.where(raw == 1, 1, -1).astype(int)
                # Flat probability is the residual uncertainty; calibration threshold
                # later converts low-confidence directional calls into NO_TRADE.
                probs = np.column_stack([directional[:, 0], 1.0 - confidence, directional[:, 1]])
                return pred, probs
            if family == "soft_voting":
                # Use the calibrated probability argmax for the ensemble.
                # VotingClassifier.predict() can collapse to the flat class even
                # when the directional probability is the strongest class.
                classes = np.asarray(getattr(model, "classes_", [-1, 0, 1]), dtype=int)
                pred = classes[np.argmax(probs, axis=1)].astype(int)
            else:
                pred = model.predict(x).astype(int)
            if invert_direction:
                pred = np.where(pred == 1, -1, np.where(pred == -1, 1, 0))
                mapping = {int(c): i for i, c in enumerate(model.classes_)}
                if all(c in mapping for c in (-1, 0, 1)):
                    probs[:, mapping[-1]], probs[:, mapping[1]] = probs[:, mapping[1]], probs[:, mapping[-1]]
            return pred, probs
        expected = np.asarray(model.predict(x), dtype=float)
        pred = _regression_signal(expected, _return_signal_threshold(family))
        probs = np.column_stack([
            np.where(pred == -1, 0.90, 0.05),
            np.where(pred == 0, 0.90, 0.05),
            np.where(pred == 1, 0.90, 0.05),
        ])
        return pred, probs

    def _horizon_eval(self, x_train, x_oos, train_rows, oos_rows, threshold):
        out = {}
        for h in HORIZONS:
            try:
                a = _future_return(train_rows, h)
                b = _future_return(oos_rows, h)
                ya = _direction_target(a, threshold)
                yb = _direction_target(b, threshold)
                if len(set(ya.tolist())) < 3 or len(set(yb.tolist())) < 3:
                    raise ValueError("three classes required")
                if len(x_train) != len(ya):
                    # x_train corresponds to the same chronological train
                    # partition as train_rows; fail closed rather than silently
                    # evaluating mismatched samples.
                    raise ValueError("Horizon evaluation feature/label length mismatch")
                m = _classifier("logistic_regression")
                m.fit(x_train, ya)
                out[str(h)] = _metrics(yb, m.predict(x_oos), m.predict_proba(x_oos), m.classes_, b)
            except Exception as exc:
                out[str(h)] = {"status": "UNAVAILABLE", "reason": str(exc)}
        return out

    def predict(self, features):
        if self.bundle is None:
            return {
                "trained": False, "abstain": True, "version": self.version,
                "decision": "NO_TRADE",
                "reason": "No promoted predictive brain artifact is available.",
            }
        x = np.asarray(
            [[float(features.get(k, 0.0) or 0.0) for k in FEATURES]], dtype=float
        )
        if not np.isfinite(x).all():
            return {
                "trained": True, "abstain": True, "version": self.version,
                "decision": "NO_TRADE", "reason": "Non-finite predictive features.",
            }
        family = self.bundle["family"]
        model = self.bundle["direction_model"]
        if family in MODEL_FAMILIES:
            pred, probs_arr = self._predict_selected(model, x, bool(self.bundle.get("direction_inverted", False)), family)
            probs = {
                "short": float(probs_arr[0, 0]),
                "flat": float(probs_arr[0, 1]),
                "long": float(probs_arr[0, 2]),
            }
            direction = max(probs, key=probs.get)
            confidence = max(probs.values())
            threshold = float(self.bundle.get("decision_threshold", 0.55))
            abstain = direction == "flat" or confidence < threshold
            expected_model = self.bundle["expected_return_model"]
            er = float(expected_model.predict(x)[0])
        else:
            er = float(model.predict(x)[0])
            pred = _regression_signal([er], COST_RATE)
            probs = {
                "short": 0.90 if pred[0] == -1 else 0.05,
                "flat": 0.90 if pred[0] == 0 else 0.05,
                "long": 0.90 if pred[0] == 1 else 0.05,
            }
            direction = max(probs, key=probs.get)
            confidence = max(probs.values())
            threshold = float(self.bundle.get("decision_threshold", 0.55))
            abstain = direction == "flat" or confidence < threshold

        downside = max(0.0, float(self.bundle["downside_model"].predict(x)[0]))
        volatility = max(0.0, float(self.bundle["volatility_model"].predict(x)[0]))
        regime = int(self.bundle["regime_model"].predict(x)[0])
        edge = er - COST_RATE
        abstain = bool(abstain or edge <= 0.0 or not np.isfinite([er, downside, volatility]).all())
        return {
            "trained": True,
            "abstain": abstain,
            "version": self.version,
            "decision": "NO_TRADE" if abstain else direction.upper(),
            "probabilities": probs,
            "expected_return": er,
            "expected_edge_after_cost": edge,
            "downside": downside,
            "volatility": volatility,
            "regime": regime,
            "uncertainty": float(1.0 - confidence),
            "abstention_probability": float(1.0 - confidence),
            "model_family": family,
            "direction_inverted": bool(self.bundle.get("direction_inverted", False)),
        }

predictive_brain = PredictiveBrain()
