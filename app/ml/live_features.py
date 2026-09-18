from __future__ import annotations

import os

from app.market_data.binance_central import CentralBinanceMarketData
from app.market_data.bitget_central import CentralBitgetMarketData


def get_live_candle_features(symbol: str) -> dict[str, float]:
    """Return live 5m candle features from the selected exchange's central cache."""
    exchange = os.getenv("HHHAI_EXECUTION_EXCHANGE", os.getenv("HHHAI_MARKET_EXCHANGE", "binance")).lower()
    if exchange == "bitget":
        return CentralBitgetMarketData.model_features(symbol.upper())
    return CentralBinanceMarketData.model_features(symbol.upper())


def enrich_missing_features(symbol: str, features: dict[str, float]) -> dict[str, float]:
    """Fill only absent candle-derived fields from the selected central cache."""
    required = (
        "return_1",
        "range_pct",
        "volume_change",
        "volatility_proxy",
        "trend_strength",
        "momentum",
    )
    missing = [name for name in required if name not in features]
    if not missing:
        return dict(features)
    live = get_live_candle_features(symbol)
    merged = dict(features)
    for name in missing:
        if name in live:
            merged[name] = live[name]
    return merged
