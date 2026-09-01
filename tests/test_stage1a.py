from app.ml.historical_context import REQUIRED_CONTEXT, make_context
from app.ml.stage1a import run_stage1a


def candles(n=80):
    base = 1700000000000
    return [[base + i * 300000, 100+i*0.1, 100.2+i*0.1, 99.8+i*0.1, 100+i*0.1, 1000] for i in range(n)]


def complete_context(rows):
    return {
        str(row[0]): make_context(str(row[0]), {k: 0.1 for k in REQUIRED_CONTEXT}, {k: "verified" for k in REQUIRED_CONTEXT})
        for row in rows
    }


def test_stage1a_rejects_missing_context():
    report = run_stage1a(candles(), {})
    assert report.status == "REJECTED"
    assert report.blockers


def test_stage1a_produces_single_auditable_decision():
    report = run_stage1a(candles(), complete_context(candles()), min_train=30, test_size=10, step=10)
    assert report.status in {"REJECTED", "READY_FOR_STAGE_1B"}
    assert report.rows_accepted >= 0
    assert isinstance(report.coverage, dict)
    assert isinstance(report.validation, dict)
