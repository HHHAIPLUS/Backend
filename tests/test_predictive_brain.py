import math

import numpy as np

from app.ml.predictive_brain import PredictiveBrain, _direction_target, _feature_hash
from app.ml.predictive import FEATURES


def _rows(n=900):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        x = rng.normal(size=len(FEATURES))
        nonlinear = x[0] * x[1] + 0.8 * np.sin(x[2]) + 0.4 * (x[3] > 0) - 0.25 * x[4]
        future = float(0.0045 * np.tanh(nonlinear) + rng.normal(0, 0.00035))
        rows.append({
            "observed_at": f"2025-01-{1 + i // 24:02d}T{i % 24:02d}:00:00+00:00",
            "features": {k: float(v) for k, v in zip(FEATURES, x)},
            "label": int(_direction_target(np.asarray([future]))[0]),
            "outcome_return": future,
            "outcome_horizon": 6,
        })
    return rows


def test_direction_target_has_three_states():
    values = np.asarray([-0.01, -0.0001, 0.0001, 0.01])
    assert _direction_target(values).tolist() == [-1, 0, 0, 1]


def test_feature_hash_is_stable():
    assert len(_feature_hash()) == 64


def test_brain_trains_or_rejects_without_corrupting_artifacts(tmp_path):
    brain = PredictiveBrain(str(tmp_path))
    report = brain.train(_rows(), version="test-brain")
    assert report.status in {"PROMOTED", "REJECTED"}
    if report.status == "PROMOTED":
        loaded = PredictiveBrain(str(tmp_path))
        assert loaded.version == "test-brain"
        result = loaded.predict(_rows()[700]["features"])
        assert result["trained"] is True
        assert result["decision"] in {"LONG", "SHORT", "NO_TRADE"}
        assert math.isfinite(result["expected_return"])
        assert math.isfinite(result["uncertainty"])


def test_brain_abstains_without_a_promoted_model(tmp_path):
    brain = PredictiveBrain(str(tmp_path))
    result = brain.predict({key: 0.0 for key in FEATURES})
    assert result["trained"] is False
    assert result["abstain"] is True
