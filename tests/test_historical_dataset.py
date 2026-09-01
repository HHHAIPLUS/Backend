import pytest

from app.ml.historical_context import make_context, REQUIRED_CONTEXT
from app.ml.historical_dataset import assemble_point_in_time_examples


def _candles(n=40):
    rows=[]
    for i in range(n):
        close=100+i*0.1
        rows.append([1700000000000+i*300000, close-.1, close+.2, close-.2, close, 1000+i])
    return rows


def _context(ts):
    return make_context(ts, {name: 0.1 for name in REQUIRED_CONTEXT}, {name: "historical_source" for name in REQUIRED_CONTEXT})


def test_missing_context_is_deferred_not_filled():
    candles=_candles()
    accepted, deferred=assemble_point_in_time_examples(candles, {})
    assert accepted == []
    assert deferred
    assert deferred[0]["reason"] == "context_not_found"


def test_complete_context_produces_examples():
    candles=_candles()
    contexts={str(row[0]): _context(str(row[0])) for row in candles}
    accepted, deferred=assemble_point_in_time_examples(candles, contexts)
    assert accepted
    assert not deferred
    assert all(set(example.features) for example in accepted)


def test_context_timestamp_mismatch_is_rejected():
    candles=_candles()
    contexts={str(candles[24][0]): _context(str(candles[25][0]))}
    with pytest.raises(ValueError):
        assemble_point_in_time_examples(candles, contexts)
