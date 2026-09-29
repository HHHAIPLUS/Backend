    # and can make the result incorrect. Class balance belongs in training.
    return CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(
        x_cal, y_cal
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
            numeric = (
                "accuracy", "directional_accuracy", "balanced_accuracy",
                "avg_net_return", "avg_trade_net_return", "trade_rate"
            )
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
            # Phase 2 development stability is a hard selection requirement:
            # every chronological fold must have positive cost-adjusted
            # expectancy and total return, with controlled drawdown and a
            # meaningful number of executed trades. A positive average across
            # folds is not sufficient because one losing fold can otherwise be
            # hidden by a stronger earlier fold.
            min_fold_trades = 30
            fold_stability = [
                {
                    "positive_expectancy": float(s.get("avg_trade_net_return", 0.0)) > 0.0,
                    "positive_total_net_return": float(s.get("total_net_return", 0.0)) > 0.0,
                    "drawdown_ok": float(s.get("max_drawdown", 1.0)) <= 0.20,
                    "enough_trades": int(s.get("trades", 0)) >= min_fold_trades,
                }
                for s in scores
            ]
            out["development_stability"] = {
                "required_folds": len(scores),
                "folds": fold_stability,
                "all_folds_pass": bool(
                    fold_stability and all(all(item.values()) for item in fold_stability)
                ),
                "min_fold_trades": min_fold_trades,
            }
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
                        # Evaluate the core cost-aware trading policy during
                        # development, not just the raw classifier. The
                        # expected-return model is trained only on this fold's
                        # training history and the fixed cost hurdle is not tuned
                        # on validation.
                        edge_model = _regressor("xgboost_regressor")
                        edge_model.fit(_slice(x, train_bounds), train_returns)
                        expected = np.asarray(edge_model.predict(_slice(x, val_bounds)), dtype=float)
                        policy_pred = _apply_edge_filter(pred, expected, COST_RATE)
                        horizon_fold_candidates.append(_metrics(y_val_fold, policy_pred, probs, model.classes_, val_returns, execution_horizon=h))
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

        # The >=100-trade requirement is an untouched-OOS acceptance gate.
        # It must never prevent a validation horizon/target from being selected.
        # Validation only determines which pre-registered target/horizon has the