import pytest

from app.ml.historical_context import make_context, merge_context


def test_missing_context_is_not_neutralized():
    context = make_context("2026-01-01T00:00:00+00:00", {"funding_rate": 0.01})
    assert "order_book_imbalance" in context.missing()
    assert not context.is_complete()
    with pytest.raises(ValueError):
        context.as_features()


def test_context_requires_timezone():
    with pytest.raises(ValueError):
        make_context("2026-01-01T00:00:00", {})


def test_context_can_be_merged_at_same_timestamp():
    a = make_context("2026-01-01T00:00:00Z", {"funding_rate": 0.01}, {"funding_rate": "exchange"})
    b = make_context("2026-01-01T00:00:00+00:00", {"open_interest_change": 0.2}, {"open_interest_change": "exchange"})
    merged = merge_context(a, b)
    assert merged.values["funding_rate"] == 0.01
    assert merged.values["open_interest_change"] == 0.2


def test_conflicting_context_is_rejected():
    a = make_context("2026-01-01T00:00:00Z", {"funding_rate": 0.01})
    b = make_context("2026-01-01T00:00:00Z", {"funding_rate": 0.02})
    with pytest.raises(ValueError):
        merge_context(a, b)
