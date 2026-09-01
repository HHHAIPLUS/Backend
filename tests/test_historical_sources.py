from datetime import datetime, timezone

import pytest

from app.ml.historical_sources import build_derivatives_context, nearest_prior


def test_nearest_prior_never_uses_future_data():
    points = [(1000, 1.0), (2000, 2.0), (3000, 3.0)]
    assert nearest_prior(points, 2500, 2000) == 2.0
    assert nearest_prior(points, 500, 2000) is None


def test_derivatives_context_records_only_observed_fields():
    observed = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    context = build_derivatives_context(observed, 600000, funding_points=[(500000, 0.01)], open_interest_points=[(300000, 100), (600000, 110)])
    assert context.values["funding_rate"] == 0.01
    assert context.values["open_interest_change"] == pytest.approx(0.10)
    assert context.sources["funding_rate"] == "exchange_historical_funding"


def test_insufficient_oi_history_is_not_fabricated():
    observed = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    context = build_derivatives_context(observed, 600000, open_interest_points=[(600000, 110)])
    assert "open_interest_change" not in context.values
    assert "open_interest_change" not in context.available
