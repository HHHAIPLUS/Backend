import numpy as np

from app.ml.ensemble import PredictiveEnsemble
from app.ml.predictive import FEATURES


def _rows(n=600):
    rows = []
    for i in range(n):
        x = (i % 30) / 30.0
        direction = 1 if i % 3 == 0 else -1 if i % 3 == 1 else 0
        rows.append({
            "features": {name: (x if name == "momentum" else (0.02 * direction if name == "return_1" else 0.01 * (i % 5))) for name in FEATURES},
            "label": direction,
            "outcome_return": 0.004 * direction + 0.0005 * (i % 4),
        })
    return rows


def test_stage3_ensemble_trains_all_heads(tmp_path):
    rows = _rows()
    model = PredictiveEnsemble(str(tmp_path / "ensemble.joblib"))
    model.fit(rows, FEATURES, "stage3-test")
    assert model.baseline is not None
    assert model.direction is not None
    assert model.return_model is not None
    assert model.risk_model is not None
    assert model.vol_model is not None
    assert model.regime_model is not None
    assert model.abstention_model is not None

    prediction = model.predict(rows[-1]["features"])
    probabilities = [prediction.short, prediction.flat, prediction.long]
    assert all(np.isfinite(probabilities))
    assert abs(sum(probabilities) - 1.0) < 1e-8
    assert 0.0 <= prediction.uncertainty <= 1.0
    assert 0.0 <= prediction.model_agreement <= 1.0


def test_stage3_ensemble_artifact_round_trip(tmp_path):
    rows = _rows()
    path = tmp_path / "ensemble.joblib"
    first = PredictiveEnsemble(str(path))
    first.fit(rows, FEATURES, "stage3-test")
    first.save({"gate": "test"})

    second = PredictiveEnsemble(str(path))
    assert second.version == "stage3-test"
    prediction = second.predict(rows[-1]["features"])
    assert np.isfinite(prediction.expected_return)
    assert np.isfinite(prediction.downside_risk)
    assert np.isfinite(prediction.volatility)
