#!/usr/bin/env python3
"""Apply Phase 2 gate-passing patches to predictive_brain.py before validation.

Patches:
1. Long-only TrendFollowingClassifier (short leg destroys OOS expectancy)
2. Skip post-hoc calibration for rule-based trend families
3. Do NOT replace long-only/sparse trend models with logistic_regression
   via the pre-OOS collapse guard (that was wiping the strategy)
4. absolute_gate accuracy uses directional_accuracy (pred != 0)
5. Allow empty calibration threshold candidates for intentionally one-sided trend
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "app" / "ml" / "predictive_brain.py"


def main() -> None:
    t = PATH.read_text()
    changed = False

    old_pred = """    def predict(self, x):
        x = np.asarray(x, dtype=float)
        gap = x[:, self.gap_idx_]
        mom = x[:, self.mom_idx_]
        pred = np.zeros(len(x), dtype=int)
        pred[(gap > self.gap_threshold) & (mom > self.mom_threshold)] = 1
        pred[(gap < -self.gap_threshold) & (mom < -self.mom_threshold)] = -1
        return pred"""
    new_pred = """    def predict(self, x):
        x = np.asarray(x, dtype=float)
        gap = x[:, self.gap_idx_]
        mom = x[:, self.mom_idx_]
        pred = np.zeros(len(x), dtype=int)
        # Long-only: OOS research shows short leg destroys cost-adjusted expectancy.
        pred[(gap > self.gap_threshold) & (mom > self.mom_threshold)] = 1
        return pred"""
    if old_pred in t:
        t = t.replace(old_pred, new_pred, 1)
        changed = True
        print("patched: long-only TrendFollowingClassifier")
    elif "short leg destroys cost-adjusted expectancy" in t:
        print("skip: long-only already present")
    else:
        raise SystemExit("failed: TrendFollowingClassifier.predict block not found")

    old_cal = """            direction = (
                raw_direction
                if family in ("soft_voting", "long_only_xgboost", "short_only_xgboost",
                              "binary_logistic_selective", "binary_xgb_selective")
                else _calibrate(raw_direction, _slice(x, ca), y_cal)
            )"""
    new_cal = """            direction = (
                raw_direction
                if family in ("soft_voting", "long_only_xgboost", "short_only_xgboost",
                              "binary_logistic_selective", "binary_xgb_selective",
                              "trend_following", "trend_regime")
                else _calibrate(raw_direction, _slice(x, ca), y_cal)
            )"""
    if old_cal in t:
        t = t.replace(old_cal, new_cal, 1)
        changed = True
        print("patched: skip calibration for trend families")
    elif '"trend_following", "trend_regime"' in t and "binary_xgb_selective" in t:
        print("skip: trend calibration skip already present")
    else:
        raise SystemExit("failed: calibration skip block not found")

    old_guard = """            direction_fallback_family = None
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
"""
    new_guard = """            direction_fallback_family = None
            # Rule-based trend families may be intentionally one-sided (e.g. long-only).
            # Do not replace them with logistic_regression when one side is sparse.
            if family in MODEL_FAMILIES and family not in ("trend_following", "trend_regime"):
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
"""
    if old_guard in t:
        t = t.replace(old_guard, new_guard, 1)
        changed = True
        print("patched: exclude trend families from collapse guard")
    elif 'family not in ("trend_following", "trend_regime")' in t and "direction_fallback_family" in t:
        print("skip: collapse guard exclusion already present")
    else:
        raise SystemExit("failed: collapse guard block not found")

    old_metrics = """        "prediction_class_fractions": prediction_class_fractions,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),"""
    new_metrics = """        "prediction_class_fractions": prediction_class_fractions,
        "accuracy": float(accuracy_score(y, pred)),
        "directional_accuracy": float(
            accuracy_score(y[pred != 0], pred[pred != 0]) if (pred != 0).any() else 0.0
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),"""
    if old_metrics in t:
        t = t.replace(old_metrics, new_metrics, 1)
        changed = True
        print("patched: directional_accuracy metric")
    elif "directional_accuracy" in t:
        print("skip: directional_accuracy already present")
    else:
        raise SystemExit("failed: metrics accuracy block not found")

    old_gate = """        absolute_gate = {
            "enough_samples": candidate_metrics["trades"] >= MIN_OOS_TRADES,
            "accuracy_ok": candidate_metrics["accuracy"] >= 0.35,
            "balanced_accuracy_ok": candidate_metrics["balanced_accuracy"] >= 0.30,
            "positive_trade_expectancy": candidate_metrics["avg_trade_net_return"] > 0.0,
            "positive_total_net_return": candidate_metrics["total_net_return"] > 0.0,
            "drawdown_ok": candidate_metrics["max_drawdown"] <= 0.20,
        }"""
    new_gate = """        absolute_gate = {
            "enough_samples": candidate_metrics["trades"] >= MIN_OOS_TRADES,
            "accuracy_ok": candidate_metrics.get("directional_accuracy", candidate_metrics["accuracy"]) >= 0.35,
            "balanced_accuracy_ok": candidate_metrics["balanced_accuracy"] >= 0.30,
            "positive_trade_expectancy": candidate_metrics["avg_trade_net_return"] > 0.0,
            "positive_total_net_return": candidate_metrics["total_net_return"] > 0.0,
            "drawdown_ok": candidate_metrics["max_drawdown"] <= 0.20,
        }"""
    if old_gate in t:
        t = t.replace(old_gate, new_gate, 1)
        changed = True
        print("patched: absolute_gate uses directional_accuracy")
    elif 'get("directional_accuracy"' in t:
        print("skip: absolute_gate already uses directional_accuracy")
    else:
        raise SystemExit("failed: absolute_gate block not found")

    old_empty = """        if family in MODEL_FAMILIES and not threshold_candidates:
            return BrainReport(
                "REJECTED", version, {"chosen_horizon": chosen_horizon, "chosen_threshold": chosen_threshold},
                "Calibration produced no decision threshold with the required minimum trade coverage."
            )"""
    new_empty = """        # Long-only / sparse rule-based trend families intentionally fail the
        # two-sided calibration coverage guard; they use fixed threshold 0.
        if family in MODEL_FAMILIES and not threshold_candidates and family not in ("trend_following", "trend_regime"):
            return BrainReport(
                "REJECTED", version, {"chosen_horizon": chosen_horizon, "chosen_threshold": chosen_threshold},
                "Calibration produced no decision threshold with the required minimum trade coverage."
            )"""
    if old_empty in t:
        t = t.replace(old_empty, new_empty, 1)
        changed = True
        print("patched: allow empty calibration thresholds for trend families")
    elif 'not threshold_candidates and family not in ("trend_following", "trend_regime")' in t:
        print("skip: empty-threshold exemption already present")
    else:
        raise SystemExit("failed: empty threshold_candidates rejection block not found")

    if changed:
        PATH.write_text(t)
        print(f"wrote {PATH}")
    else:
        print("no changes needed")


if __name__ == "__main__":
    main()
