from __future__ import annotations

import numpy as np

from app.ml.model_validation import paired_bootstrap_ci, promotion_gate
from app.ml.predictive_brain import _direction_target, _metrics


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

# Phase 2 final verification trigger.
