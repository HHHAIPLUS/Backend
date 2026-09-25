from __future__ import annotations

import time
import pytest

from app.ml.dataset_integrity import audit_klines, audit_dataset
from app.ml.features import FEATURES

def _candles(count=10, interval_ms=3_600_000):
    base = 1700000000000
    rows = []
    for i in range(count):
        ts = base + i * interval_ms
        close = 100.0 + i
        rows.append([ts, close - 0.5, close + 1.0, close - 1.0, close, 1000.0 + i])
    return rows

def test_closed_ohlcv_history_passes_strict_audit():
    rows = _candles()
    audit = audit_klines(rows, interval_ms=3_600_000, now_ms=rows[-1][0] + 3_600_000)
    assert audit.production_ready
    assert audit.gaps == 0
    assert audit.duplicate_timestamps == 0
    assert audit.invalid_ohlc == 0

def test_duplicate_timestamp_fails():
    rows = _candles()
    rows.insert(5, list(rows[5]))
    audit = audit_klines(rows, interval_ms=3_600_000, now_ms=rows[-1][0] + 3_600_000)
    assert not audit.production_ready
    assert audit.duplicate_timestamps == 1

def test_gap_fails():
    rows = _candles()
    rows[5][0] += 3_600_000
    audit = audit_klines(rows, interval_ms=3_600_000, now_ms=rows[-1][0] + 3_600_000)
    assert not audit.production_ready
    assert audit.gaps > 0

def test_invalid_ohlc_fails():
    rows = _candles()
    rows[3][2] = rows[3][1] - 1
    audit = audit_klines(rows, interval_ms=3_600_000, now_ms=rows[-1][0] + 3_600_000)
    assert not audit.production_ready
    assert audit.invalid_ohlc == 1

def test_open_current_candle_fails_closed():
    rows = _candles()
    now_ms = rows[-1][0] + 30 * 60_000
    audit = audit_klines(rows, interval_ms=3_600_000, now_ms=now_ms)
    assert not audit.production_ready
    assert audit.future_or_open_candles == 1

def test_prediction_features_have_no_live_only_context_dependency():
    features = {name: float(i) for i, name in enumerate(FEATURES)}
    rows = [{
        'observed_at': '2026-01-01T00:00:00+00:00',
        'features': features,
        'feature_provenance': {},
    }]
    audit = audit_dataset(rows)
    assert audit.production_ready
    assert audit.incomplete_context_rows == 0

def test_future_target_inside_features_fails():
    features = {name: float(i) for i, name in enumerate(FEATURES)}
    features['future_return'] = 0.5
    row = {'observed_at': '2026-01-01T00:00:00+00:00', 'features': features, 'feature_provenance': {}}
    audit = audit_dataset([row])
    assert audit.leakage_suspected
    assert not audit.production_ready
def test_build_dataset_rejects_duplicates_before_deduplication():
    from app.ml.bootstrap import build_dataset

    rows = _candles(count=400)
    rows.insert(100, list(rows[100]))
    with pytest.raises(ValueError, match="Historical candle audit failed"):
        build_dataset(rows, horizon=1, threshold=0.0015, interval="1h", symbol="BTCUSDT", provider="bitget")
