from __future__ import annotations

import pytest

from app.ml.dataset_integrity import audit_dataset, require_production_ready


def _row(ts: str, complete: bool = True) -> dict:
    features = {
        "return_1": 0.01,
        "range_pct": 0.02,
        "volume_change": 0.1,
        "order_book_imbalance": 0.0,
        "funding_rate": 0.0,
        "open_interest_change": 0.0,
        "news_risk": 0.0,
        "news_sentiment": 0.0,
        "volatility_proxy": 0.1,
        "trend_strength": 0.2,
        "momentum": 0.3,
        "liquidity_stress": 0.0,
    }
    provenance = {key: complete for key in (
        "order_book_imbalance", "funding_rate", "open_interest_change",
        "news_risk", "news_sentiment", "liquidity_stress",
    )}
    return {"observed_at": ts, "features": features, "feature_provenance": provenance}


def test_incomplete_context_is_not_called_production_ready() -> None:
    audit = audit_dataset([_row("2026-01-01T00:00:00+00:00", False)])
    assert audit.incomplete_context_rows == 1
    assert not audit.production_ready
    with pytest.raises(ValueError, match="not production-ready"):
        require_production_ready([_row("2026-01-01T00:00:00+00:00", False)])


def test_duplicate_or_out_of_order_timestamps_are_rejected() -> None:
    rows = [
        _row("2026-01-01T00:01:00+00:00"),
        _row("2026-01-01T00:01:00+00:00"),
    ]
    audit = audit_dataset(rows)
    assert not audit.unique_timestamps
    assert not audit.ordered


def test_target_inside_features_is_flagged() -> None:
    row = _row("2026-01-01T00:00:00+00:00")
    row["features"]["target"] = 1
    audit = audit_dataset([row])
    assert audit.leakage_suspected
    assert not audit.production_ready


def test_complete_dataset_passes_structural_audit() -> None:
    rows = [
        _row("2026-01-01T00:00:00+00:00"),
        _row("2026-01-01T00:05:00+00:00"),
    ]
    audit = audit_dataset(rows)
    assert audit.production_ready
    assert audit.complete_context_rows == 2
