from __future__ import annotations

import json
import os

from app.ml.bootstrap import fetch_bitget_klines, build_dataset, audit_historical_klines
from app.ml.dataset_integrity import require_production_ready

symbol = os.getenv('PHASE1_SYMBOL', 'BTCUSDT').upper()
interval = os.getenv('PHASE1_INTERVAL', '1h')
limit = int(os.getenv('PHASE1_CANDLES', '10000'))
raw = fetch_bitget_klines(symbol, interval, limit)
candle_audit = audit_historical_klines(raw, interval)
rows = build_dataset(raw, horizon=6, threshold=0.0025, symbol=symbol, interval=interval, provider='bitget')
dataset_audit = require_production_ready(rows)
report = {
    'symbol': symbol, 'interval': interval, 'requested_candles': limit,
    'received_candles': len(raw), 'training_rows': len(rows),
    'candle_audit': candle_audit, 'dataset_audit': dataset_audit.__dict__,
    'status': 'PASS',
}
print(json.dumps(report, indent=2, sort_keys=True))