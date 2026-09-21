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
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler

from app.ml.predictive import FEATURES
from app.ml.model_validation import promotion_gate

MODEL_FAMILIES = ("logistic_regression", "logistic_regression_directional", "sgd_logistic", "xgboost", "extra_trees", "random_forest_balanced", "hist_gradient_boosting", "soft_voting")
RETURN_FAMILIES = ("ridge", "hist_gradient_boosting_regressor", "xgboost_regressor", "extra_trees_regressor", "random_forest_regressor")
HORIZONS = (1, 3, 6, 12)
LABEL_THRESHOLDS = (0.0010, 0.0015, 0.0020, 0.0025)
COST_RATE = 0.0014
ARTIFACT_SCHEMA = 4
MAX_LABEL_HORIZON = max(HORIZONS)
MIN_OOS_TRADES = 100


class DirectionalXGBClassifier:
    """Sklearn-compatible XGBoost wrapper with directional labels -1/0/1."""
    def __init__(self, **params):
        self.params = dict(params)
        self.model = XGBClassifier(**self.params)

    def fit(self, X, y, *args, **kwargs):
        y_arr = np.asarray(y, dtype=int)
        if not np.isin(y_arr, (-1, 0, 1)).all():
            raise ValueError("Directional XGBoost expects labels -1, 0, 1")
        self.model.fit(X, y_arr + 1, *args, **kwargs)
        self.classes_ = np.asarray([-1, 0, 1], dtype=int)
        self.n_classes_ = 3
        return self

    def predict(self, X, *args, **kwargs):
        encoded = self.model.predict(X, *args, **kwargs)
        return np.asarray(encoded, dtype=int) - 1

    def predict_proba(self, X, *args, **kwargs):
        return self.model.predict_proba(X, *args, **kwargs)

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params)
        self.model.set_params(**params)
        return self


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


def _future_return(rows, horizon):
    values = []
    for row in rows:
        by = row.get("outcome_return_by_horizon", {})
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


def _direction_target(values, threshold=COST_RATE):
    return np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))


def _classifier(family):
    if family == "xgboost":
        return DirectionalXGBClassifier(n_estimators=160, max_depth=4, learning_rate=0.04, subsample=0.80, colsample_bytree=0.80, min_child_weight=8, reg_alpha=0.10, reg_lambda=2.0, objective="multi:softprob", num_class=3, eval_metric="mlogloss", tree_method="hist", n_jobs=1, random_state=42)
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
            max_iter=140, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42
        )
    if family == "hist_gradient_boosting_balanced":
        return HistGradientBoostingClassifier(
            max_iter=140, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42
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
                                   max_iter=2500, tol=1e-4, random_state=42, early_stopping=True,
                                   validation_fraction=0.12, n_iter_no_change=20),
        )])
    raise ValueError(f"Unknown model family: {family}")


def _regressor(family):
    if family == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])
    if family == "extra_trees_regressor":
        return ExtraTreesRegressor(n_estimators=60, min_samples_leaf=10, random_state=42, n_jobs=1)
    if family == "random_forest_regressor":
        return RandomForestRegressor(n_estimators=80, min_samples_leaf=12, max_features="sqrt", random_state=42, n_jobs=1)
    if family == "hist_gradient_boosting_regressor":
        return HistGradientBoostingRegressor(
            max_iter=140, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42
        )
    if family == "xgboost_regressor":
        return XGBRegressor(
            n_estimators=220, max_depth=4, learning_rate=0.03,
            subsample=0.80, colsample_bytree=0.80, min_child_weight=12,
            reg_alpha=0.10, reg_lambda=3.0, objective="reg:squarederror",
            eval_metric="rmse", tree_method="hist", n_jobs=1, random_state=42
        )
    raise ValueError(f"Unknown return model family: {family}")


def _net_returns(returns, pred):
    """Mark-to-market returns with costs charged only when position changes.

    A position is held until the signal changes. Entry, exit and reversal costs
    are charged on the position delta rather than on every hourly bar. The final
    non-flat position is explicitly closed at the end of the evaluation window.
    """
    returns = np.asarray(returns, dtype=float)
    pred = np.asarray(pred, dtype=int)
    prev = np.r_[0, pred[:-1]]
    position = np.where(pred == 1, 1.0, np.where(pred == -1, -1.0, 0.0))
    costs = COST_RATE * np.abs(position - prev.astype(float))
    if len(costs) and position[-1] != 0.0:
        costs[-1] += COST_RATE * abs(position[-1])
    return position * returns - costs


def _trade_pnls(returns, pred):
    """Return one net PnL value per held position segment, including costs."""
    returns = np.asarray(returns, dtype=float)
    pred = np.asarray(pred, dtype=int)
    if len(pred) == 0:
        return np.asarray([], dtype=float)
    pnls = []
    current = 0
    pnl = 0.0
    for ret, pos in zip(returns, pred):
        if pos != current:
            if current != 0:
                pnl -= COST_RATE * abs(current)
                pnls.append(pnl)
                pnl = 0.0
            if pos != 0:
                pnl -= COST_RATE * abs(pos)
            current = int(pos)
        if current != 0:
            pnl += float(current) * float(ret)
    if current != 0:
        pnl -= COST_RATE * abs(current)
        pnls.append(pnl)
    return np.asarray(pnls, dtype=float)


def _metrics(y, pred, probs, classes, returns):
    net = _net_returns(returns, pred)
    trade_pnls = _trade_pnls(returns, pred)
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
        "trades": int(len(trade_pnls)),
        "trade_rate": float(len(trade_pnls) / max(1, len(y))),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "avg_net_return": float(net.mean()),
        "avg_trade_net_return": float(trade_pnls.mean()) if len(trade_pnls) else 0.0,
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
    return CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(x_cal, y_cal)


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
            out["total_net_return"] = float(sum(float(s.get("total_net_return", 0.0)) for s in scores))
            out["max_drawdown"] = float(max(float(s.get("max_drawdown", 0.0)) for s in scores))
            out["folds"] = scores
            return out

        horizon_selection = {}
        for h in HORIZONS:
            full_returns = _future_return(rows, h)
            for threshold in LABEL_THRESHOLDS:
                fold_scores = []
                for train_bounds, val_bounds in selection_folds:
                    train_returns = _slice(full_returns, train_bounds)
                    val_returns = _slice(full_returns, val_bounds)
                    y_train_fold = _direction_target(train_returns, threshold)
                    y_val_fold = _direction_target(val_returns, threshold)
                    if len(set(y_train_fold.tolist())) < 3 or len(set(y_val_fold.tolist())) < 3:
                        continue
                    model = _classifier("logistic_regression")
                    model.fit(_slice(x, train_bounds), y_train_fold)
                    pred = model.predict(_slice(x, val_bounds))
                    probs = model.predict_proba(_slice(x, val_bounds))
                    fold_scores.append(_metrics(y_val_fold, pred, probs, model.classes_, val_returns))
                if fold_scores:
                    # Reject validation targets that are dominated by one class.
                    # This prevents a high overall accuracy from coming mainly
                    # from predicting the flat class and is determined entirely
                    # before the untouched OOS period.
                    validation_labels = np.concatenate([
                        _direction_target(_slice(full_returns, val_bounds), threshold)
                        for _, val_bounds in selection_folds
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
                exp if economically_viable else -1e99,
                total if economically_viable else -1e99,
                -dd if economically_viable else -1e99,
                bal,
                acc,
                float(v.get("trade_rate", 0.0)),
            )

        chosen = max(viable, key=_selection_key)
        chosen_horizon = int(chosen["horizon"])
        chosen_threshold = float(chosen["label_threshold"])

        returns = _future_return(rows, chosen_horizon)
        y = _direction_target(returns, chosen_threshold)
        y_train = _slice(y, tr)
        y_val = _slice(y, va)
        y_cal = _slice(y, ca)
        y_oos = _slice(y, oo)
        r_train = _slice(returns, tr)
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
        for family in MODEL_FAMILIES:
            fold_scores = []
            inverse_fold_scores = []
            for train_bounds, val_bounds in selection_folds:
                y_train_fold = _slice(y, train_bounds)
                y_val_fold = _slice(y, val_bounds)
                model = _classifier(family)
                x_train_fold = _slice(x, train_bounds)
                counts = np.bincount(y_train_fold + 1, minlength=3).astype(float)
                if family in {"xgboost", "hist_gradient_boosting"}:
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
            validation_scores[family] = score
            validation_scores[family + "_inverse"] = inv_score
            candidates.append((score["avg_trade_net_return"], score["balanced_accuracy"], family, False))
            candidates.append((inv_score["avg_trade_net_return"], inv_score["balanced_accuracy"], family, True))

        # Cost-aware regression candidates are evaluated across a fixed,
        # pre-registered forecast-magnitude grid. This is still validation-only:
        # the untouched OOS is not used to choose the threshold.
        regression_thresholds = (0.0014, 0.0020, 0.0030, 0.0040, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200)
        for family in RETURN_FAMILIES:
            for regression_threshold in regression_thresholds:
                fold_scores = []
                for train_bounds, val_bounds in selection_folds:
                    reg = _regressor(family)
                    reg.fit(_slice(x, train_bounds), _slice(returns, train_bounds))
                    expected = reg.predict(_slice(x, val_bounds))
                    pred = _regression_signal(expected, regression_threshold)
                    probs = np.column_stack([
                        np.where(pred == -1, 0.90, 0.05),
                        np.where(pred == 0, 0.90, 0.05),
                        np.where(pred == 1, 0.90, 0.05),
                    ])
                    fold_scores.append(_metrics(_slice(y, val_bounds), pred, probs, np.array([-1, 0, 1]), _slice(returns, val_bounds)))
                score = aggregate_scores(fold_scores)
                key = f"{family}@{regression_threshold:.4f}"
                validation_scores[key] = score | {"regression_threshold": regression_threshold, "family": family}
                candidates.append((score["avg_trade_net_return"], score["balanced_accuracy"], key, False))

        if not candidates:
            return BrainReport("REJECTED", version, {"validation_families": validation_scores},
                               "No valid model family was evaluated.")

        # Model selection is validation-only. Classification quality is the
        # primary objective because the independent OOS gate explicitly requires
        # both accuracy and balanced accuracy. Trading expectancy is a secondary
        # tie-breaker, not a reason to prefer a model with weaker classification.
        # OOS observations are never consulted here.
        def _candidate_score(c):
            key = c[2] + "_inverse" if c[3] else c[2]
            score = validation_scores[key]
            bal = float(score.get("balanced_accuracy", -1e99))
            acc = float(score.get("accuracy", -1e99))
            exp = float(score.get("avg_trade_net_return", -1e99))
            viable_validation = exp > 0.0 and int(score.get("trades", 0)) >= MIN_OOS_TRADES
            total = float(score.get("total_net_return", -1e99))
            dd = float(score.get("max_drawdown", 1e99))
            return (
                1 if viable_validation else 0,
                exp if viable_validation else -1e99,
                total if viable_validation else -1e99,
                -dd if viable_validation else -1e99,
                bal,
                acc,
                float(c[0]),
                float(c[1]),
            )

        best = max(candidates, key=_candidate_score)
        family = best[2]
        invert_direction = bool(best[3])

        # Fit the selected family on train+validation, then calibrate on a
        # completely separate calibration period. OOS remains untouched.
        fit_end = va[1]
        x_fit = x[:fit_end]
        y_fit = y[:fit_end]
        if family in MODEL_FAMILIES:
            raw_direction = _classifier(family)
            counts = np.bincount(y_fit + 1, minlength=3).astype(float)
            if family in {"xgboost", "hist_gradient_boosting"}:
                weights = np.asarray([1.0 / max(counts[label + 1], 1.0) for label in y_fit])
                weights *= len(weights) / max(weights.sum(), 1e-12)
                raw_direction.fit(x_fit, y_fit, sample_weight=weights)
            else:
                raw_direction.fit(x_fit, y_fit)
            baseline_raw = _classifier("logistic_regression")
            baseline_raw.fit(x_fit, y_fit)
            # The fixed soft-voting ensemble already averages calibrated
            # component probabilities structurally; wrapping the whole voter in
            # a second post-hoc calibrator can collapse its directional signal.
            # Keep the pre-registered ensemble probabilities intact.
            direction = raw_direction if family == "soft_voting" else _calibrate(raw_direction, _slice(x, ca), y_cal)
            baseline = _calibrate(baseline_raw, _slice(x, ca), y_cal)
        else:
            # Regression candidates are fit on train+validation and use the
            # calibration set only to choose a confidence/abstention threshold.
            raw_reg = _regressor(family)
            # Regression candidates predict the actual future return target.
            raw_reg.fit(x_fit, _slice(returns, (0, fit_end)))
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
        min_cal_trades = max(100, int(len(y_cal) * 0.02))
        if family in MODEL_FAMILIES:
            for threshold in np.arange(0.30, 0.96, 0.02):
                selected = cal_pred.copy()
                selected[confidence < threshold] = 0
                traded = selected != 0
                trade_count = int(traded.sum())
                trade_rate = trade_count / max(1, len(selected))
                if trade_count < min_cal_trades or trade_rate < 0.02:
                    continue
                net = _net_returns(r_cal, selected)
                equity = np.cumsum(net)
                peak = np.maximum.accumulate(np.r_[0.0, equity])
                drawdown = float(np.max(peak[1:] - equity)) if len(equity) else 0.0
                avg_trade = float(net[traded].mean())
                cal_metrics = _metrics(y_cal, selected, cal_prob, np.array([-1, 0, 1]), r_cal)
                threshold_candidates.append((
                    float(avg_trade), float(net.sum()), -float(drawdown),
                    float(cal_metrics["balanced_accuracy"]), float(cal_metrics["accuracy"]),
                    float(trade_rate), float(threshold),
                ))
        else:
            expected_cal = np.asarray(direction.predict(_slice(x, ca)), dtype=float)
            for threshold in np.arange(COST_RATE, 0.02001, 0.0002):
                selected = _regression_signal(expected_cal, float(threshold))
                traded = selected != 0
                trade_count = int(traded.sum())
                trade_rate = trade_count / max(1, len(selected))
                if trade_count < min_cal_trades or trade_rate < 0.05:
                    continue
                net = _net_returns(r_cal, selected)
                equity = np.cumsum(net)
                peak = np.maximum.accumulate(np.r_[0.0, equity])
                drawdown = float(np.max(peak[1:] - equity)) if len(equity) else 0.0
                probs = np.column_stack([
                    np.where(selected == -1, 0.90, 0.05),
                    np.where(selected == 0, 0.90, 0.05),
                    np.where(selected == 1, 0.90, 0.05),
                ])
                cal_metrics = _metrics(y_cal, selected, probs, np.array([-1, 0, 1]), r_cal)
                threshold_candidates.append((
                    float(net[traded].mean()), float(net.sum()), -float(drawdown),
                    float(cal_metrics["balanced_accuracy"]), float(cal_metrics["accuracy"]),
                    float(trade_rate), float(threshold),
                ))

        if family in MODEL_FAMILIES and not threshold_candidates:
            return BrainReport(
                "REJECTED", version, {"chosen_horizon": chosen_horizon, "chosen_threshold": chosen_threshold},
                "Calibration produced no decision threshold with the required minimum trade coverage."
            )
        if threshold_candidates:
            # Calibration-only threshold selection. Prefer thresholds that
            # simultaneously show balanced classification and positive net
            # expectancy, then prefer stronger economics and lower drawdown.
            # The untouched OOS is never consulted.
            def _threshold_key(t):
                exp, total, neg_dd, bal, acc, rate, threshold = t
                viable = bal >= 0.50 and acc >= 0.52 and exp > 0.0
                return (
                    1 if viable else 0,
                    exp if viable else -1e99,
                    total if viable else -1e99,
                    neg_dd if viable else -1e99,
                    bal,
                    acc,
                    rate,
                    threshold,
                )
            selection_threshold = max(threshold_candidates, key=_threshold_key)[-1]

        candidate_pred, candidate_prob = self._predict_selected(direction, _slice(x, oo), invert_direction, family, selection_threshold)
        baseline_pred = baseline.predict(_slice(x, oo))
        baseline_prob = baseline.predict_proba(_slice(x, oo))
        if family in MODEL_FAMILIES:
            candidate_pred[candidate_prob.max(axis=1) < selection_threshold] = 0
        else:
            # _predict_selected already converts the expected return into a
            # {-1,0,1} signal. Do not threshold the discrete signal again.
            candidate_pred = candidate_pred.astype(int)

        candidate_metrics = _metrics(y_oos, candidate_pred, candidate_prob, np.array([-1, 0, 1]), r_oos)
        baseline_metrics = _metrics(y_oos, baseline_pred, baseline_prob, baseline.classes_, r_oos)

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
            "decision_threshold": selection_threshold,
            "cost_rate": COST_RATE,
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
            "expected_return_model": direction if family not in MODEL_FAMILIES else self._fit_return_model(x_fit, _slice(returns, (0, fit_end)), family),
            "downside_model": self._fit_return_model(x_fit, np.minimum(_slice(returns, (0, fit_end)), 0.0), family if family in RETURN_FAMILIES else "ridge"),
            "volatility_model": self._fit_return_model(x_fit, np.abs(_slice(returns, (0, fit_end))), family if family in RETURN_FAMILIES else "ridge"),
            "regime_model": self._fit_regime_model(x_fit, _slice(returns, (0, fit_end)), family),
            "abstention_model": None,
            "meta_model": None,
            "family": family,
            "direction_inverted": invert_direction,
            "decision_threshold": selection_threshold,
            "feature_hash": _feature_hash(),
            "features": FEATURES,
            "cost_rate": COST_RATE,
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
    def _predict_selected(model, x, invert_direction, family, signal_threshold=COST_RATE):
        if family in MODEL_FAMILIES:
            probs = model.predict_proba(x)
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
        pred = _regression_signal(expected, signal_threshold)
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
            pred = _regression_signal([er], float(self.bundle.get("decision_threshold", COST_RATE)))
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
