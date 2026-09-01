from app.ml.historical_context import REQUIRED_CONTEXT, make_context
from app.ml.historical_enrichment import enrich_dataset


def _candles(n=40):
    rows = []
    for i in range(n):
        close = 100 + i * 0.1
        rows.append([1700000000000 + i * 300000, close-.1, close+.2, close-.2, close, 1000+i])
    return rows


def test_enrichment_reports_not_ready_when_context_is_missing():
    examples, report = enrich_dataset(_candles(), {})
    assert not examples
    assert not report.production_ready
    assert report.deferred_observations > 0


def test_enrichment_reports_ready_for_complete_context():
    candles = _candles()
    contexts = {
        str(row[0]): make_context(
            str(row[0]),
            {name: 0.1 for name in REQUIRED_CONTEXT},
            {name: "verified_source" for name in REQUIRED_CONTEXT},
        )
        for row in candles
    }
    examples, report = enrich_dataset(candles, contexts)
    assert examples
    assert report.production_ready
    assert report.deferred_observations == 0
