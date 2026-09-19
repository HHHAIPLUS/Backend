from __future__ import annotations

import numpy as np

from app.ml.model_validation import paired_bootstrap_ci, promotion_gate
from app.ml.predictive_brain import _direction_target, _metrics, _purged_time_splits


def test_phase2_direction_labels_are_cost_aware():
    values = np.array([-0.004, -0.001, 0.0, 0.001, 0.004], dtype=float)
    labels = _direction_target(values, 0.0015)
    assert labels.tolist() == [-1, 0, 0, 0, 1]


def test_phase2_metrics_include_net_cost_and_trade_count():
    y = np.array([-1, 0, 1, 1], dtype=int)
    pred = np.array([-1, 0, 1, -1], dtype=int)
    probs = np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.7,.1,.2]], dtype=float)
    returns = np.array([.004, 0.0, .004, .004], dtype=float)
    metrics = _metrics(y, pred, probs, np.array([-1,0,1]), returns)
    assert metrics["samples"] == 4
    assert metrics["trades"] == 3
    assert metrics["avg_trade_net_return"] < .004
    assert metrics["total_net_return"] < 0
    assert metrics["total_net_return"] < float(np.sum(returns * np.where(pred == 1, 1.0, np.where(pred == -1, -1.0, 0.0))))


def test_phase2_paired_bootstrap_rejects_invalid_small_sample():
    result = paired_bootstrap_ci(np.ones(20), np.zeros(20))
    assert result["valid"] is False


def test_phase2_promotion_gate_never_promotes_failed_statistical_gate():
    candidate = np.full(100, 0.0001)
    baseline = np.full(100, 0.0002)
    result = promotion_gate(candidate, baseline, .55, .50, .01, .01)
    assert result["promoted"] is False


def test_phase2_split_is_chronological_and_purged():
    rows = [{"observed_at": f"2026-01-{(i // 24) + 1:02d}T{(i % 24):02d}:00:00+00:00"} for i in range(24 * 60)]
    split = _purged_time_splits(rows)
    tr = split["train"]
    va = split["validation"]
    ca = split["calibration"]
    oo = split["oos"]
    assert tr[1] <= va[0] - split["purge_rows"]
    assert va[1] <= ca[0] - split["purge_rows"]
    assert ca[1] <= oo[0] - split["purge_rows"]
    assert rows[tr[1] - 1]["observed_at"] < rows[va[0]]["observed_at"]
    assert rows[va[1] - 1]["observed_at"] < rows[ca[0]]["observed_at"]
    assert rows[ca[1] - 1]["observed_at"] < rows[oo[0]]["observed_at"]


def test_phase2_promotion_gate_counts_trades_not_flat_samples():
    candidate = np.r_[np.zeros(1000), np.full(99, 0.001)]
    baseline = np.r_[np.zeros(1000), np.full(99, 0.0005)]
    result = promotion_gate(candidate, baseline, .60, .50, .01, .01)
    assert result["trade_count"] == 99
    assert result["enough_samples"] is False
