from __future__ import annotations

import os
from typing import Any

from app.ml.bootstrap import (
    audit_historical_klines,
    build_dataset,
    fetch_binance_archive_klines,
    fetch_binance_klines,
    fetch_bitget_klines,
)
from app.ml.dataset_integrity import require_production_ready

PHASE2_PROVIDER = os.getenv("HHHAI_PHASE2_PROVIDER", "bitget").strip().lower()
PHASE2_SYMBOL = os.getenv("HHHAI_BRAIN_BOOTSTRAP_SYMBOLS", "BTCUSDT").split(",")[0].strip().upper()
PHASE2_INTERVAL = os.getenv("HHHAI_PHASE2_INTERVAL", os.getenv("HHHAI_BRAIN_BOOTSTRAP_INTERVAL", "1h")).strip()
PHASE2_CANDLES = max(5000, min(30000, int(os.getenv("HHHAI_PHASE2_CANDLES", os.getenv("HHHAI_BRAIN_BOOTSTRAP_CANDLES", "10000")))))
PHASE2_HORIZON = int(os.getenv("HHHAI_PHASE2_HORIZON", os.getenv("HHHAI_PHASE2_FIXED_HORIZON", "1") or "1"))
if PHASE2_HORIZON <= 0:
    PHASE2_HORIZON = 1
PHASE2_THRESHOLD = float(
    os.getenv("HHHAI_PHASE2_FIXED_LABEL_THRESHOLD", "").strip()
    or os.getenv("HHHAI_BRAIN_LABEL_THRESHOLD", "0.0015")
)

_FETCHERS = {
    "bitget": fetch_bitget_klines,
    "binance": fetch_binance_klines,
    "binance_archive": fetch_binance_archive_klines,
}


def authoritative_config() -> dict[str, Any]:
    return {
        "provider": PHASE2_PROVIDER,
        "symbol": PHASE2_SYMBOL,
        "interval": PHASE2_INTERVAL,
        "candles": PHASE2_CANDLES,
        "horizon": PHASE2_HORIZON,
        "label_threshold": PHASE2_THRESHOLD,
    }


def fetch_authoritative_dataset() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if PHASE2_PROVIDER not in _FETCHERS:
        raise RuntimeError(f"Unsupported Phase 2 provider: {PHASE2_PROVIDER}")
    if PHASE2_PROVIDER != "bitget":
        raise RuntimeError("Phase 2 authoritative validation is locked to Bitget; change PHASE2_PROVIDER only through an explicit architecture change.")

    fetcher = _FETCHERS[PHASE2_PROVIDER]
    raw = fetcher(symbol=PHASE2_SYMBOL, interval=PHASE2_INTERVAL, limit=PHASE2_CANDLES)
    audit = audit_historical_klines(raw, PHASE2_INTERVAL)
    rows = build_dataset(
        raw,
        horizon=PHASE2_HORIZON,
        threshold=PHASE2_THRESHOLD,
        symbol=PHASE2_SYMBOL,
        interval=PHASE2_INTERVAL,
        provider=PHASE2_PROVIDER,
    )
    dataset_audit = require_production_ready(rows)
    return rows, {
        **authoritative_config(),
        "candle_audit": audit,
        "dataset_audit": dataset_audit.__dict__,
        "raw_candles": len(raw),
        "training_rows": len(rows),
    }
