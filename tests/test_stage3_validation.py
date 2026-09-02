import numpy as np

from app.ml.model_validation import paired_bootstrap_ci, promotion_gate
from app.ml.predictive_brain import PredictiveBrain


def test_paired_bootstrap_requires_matched_oos_samples():
    candidate = np.full(120, 0.002)
    baseline = np.zeros(120)
    result = paired_bootstrap_ci(candidate, baseline, samples=300)
    assert result["valid"] is True
    assert result["ci_low"] > 0


def test_promotion_gate_rejects_statistically_unproven_candidate():
    candidate = np.zeros(120)
    baseline = np.zeros(120)
    gate = promotion_gate(candidate, baseline, 0.60, 0.60, 0.0, 0.0)
    assert gate["promoted"] is False
    assert gate["statistically_positive"] is False


def test_unpromoted_brain_artifact_fails_closed(tmp_path):
    brain = PredictiveBrain(str(tmp_path))
    brain.manifest_path.write_text('{"schema_version": 2, "version": "candidate", "features": [], "promotion": {"promoted": false}}')
    brain.artifact_path.write_bytes(b"invalid")
    reloaded = PredictiveBrain(str(tmp_path))
    assert reloaded.bundle is None
    assert reloaded.version == "untrained"
