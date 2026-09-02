from datetime import datetime, timedelta, timezone

import pytest

from ai.research_loop import ResearchLoop, ResearchSnapshot


def make_rows(n=120):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        high = i % 3 != 0
        ret = 0.002 if high else -0.004
        rows.append(ResearchSnapshot(
            record_id=f"r{i}", symbol="BTCUSDT", action="LONG", model_version="champion-v1",
            regime="trend" if i % 2 else "range", horizon=6,
            confidence=0.85 if high else 0.40,
            expected_probability=0.85 if high else 0.60,
            features={"momentum": float(i % 10)}, thesis="continuation",
            created_at=(start + timedelta(minutes=i)).isoformat(), realized_return=ret,
            slippage=0.0001, fees=0.0001,
        ))
    return rows


def test_snapshot_ingestion_and_error_attribution():
    engine = ResearchLoop()
    for row in make_rows(40): engine.add_snapshot(row)
    report = engine.attribute_errors()
    assert report["snapshots"] == 40
    assert report["prediction_errors"] > 0
    assert "champion-v1" in report["model_errors"]
    assert "trend" in report["regime_errors"]


def test_candidate_is_quarantined_until_evaluation():
    engine = ResearchLoop()
    for row in make_rows(120): engine.add_snapshot(row)
    candidate = engine.propose("champion-v1", "recalibrate probability/confidence mapping", "calibration error")
    assert candidate.status == "quarantined"
    result = engine.evaluate(candidate.candidate_id)
    assert result.samples == 36
    assert result.reproducibility_fingerprint
    assert candidate.status in {"promotion_eligible", "rejected"}
    assert result.walk_forward["valid"] is True
    assert result.out_of_sample["valid"] is True
    assert result.stress["valid"] is True


def test_experiment_is_reproducible_for_the_same_candidate():
    engine = ResearchLoop()
    for row in make_rows(120): engine.add_snapshot(row)
    candidate = engine.propose("champion-v1", "recalibrate probability/confidence mapping", "same hypothesis")
    a = engine.evaluate(candidate.candidate_id)
    b = engine.evaluate(candidate.candidate_id)
    assert a.reproducibility_fingerprint == b.reproducibility_fingerprint
    assert a.bootstrap_ci_low == b.bootstrap_ci_low
    assert a.bootstrap_ci_high == b.bootstrap_ci_high
    assert a.passed == b.passed


def test_insufficient_oos_is_fail_closed():
    engine = ResearchLoop()
    for row in make_rows(20): engine.add_snapshot(row)
    candidate = engine.propose("champion-v1", "recalibrate probability/confidence mapping", "not enough data")
    with pytest.raises(ValueError): engine.evaluate(candidate.candidate_id)


def test_candidate_never_gets_production_authority():
    engine = ResearchLoop()
    assert engine.snapshot()["production_self_modification"] is False
    assert engine.snapshot()["execution_authority"] is False
