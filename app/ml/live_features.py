from __future__ import annotations

import os
import time
from threading import Lock
from typing import Any

import httpx

from app.ml.features import build_model_features

_BINANCE_URL = "https://fapi.binance.com/fapi/v1/klines"
_CACHE_TTL_SECONDS = 15.0
_CACHE: dict[str, tuple[float, dict[str, float]]] = {}
_LOCK = Lock()


def _fetch(symbol: str, limit: int = 30) -> dict[str, float]:
    response = httpx.get(
        _BINANCE_URL,
        params={"symbol": symbol.upper(), "interval": "5m", "limit": limit},
        timeout=httpx.Timeout(6.0, connect=3.0),
        follow_redirects=True,
        trust_env=False,
        headers={"User-Agent": "HHHAI/1.0", "Accept": "application/json"},
    )
    response.raise_for_status()
    raw: Any = response.json()
    if not isinstance(raw, list) or len(raw) < 3:
        raise RuntimeError("Binance returned insufficient 5m candles")
    candles = [row for row in raw if isinstance(row, list) and len(row) >= 6]
    features = build_model_features(candles)
    return {key: float(value) for key, value in features.items()}


def get_live_candle_features(symbol: str) -> dict[str, float]:
    """Return fresh 5m candle-derived model features with a short cache."""
    key = symbol.upper()
    now = time.monotonic()
    with _LOCK:
        cached = _CACHE.get(key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return dict(cached[1])
    features = _fetch(key)
    with _LOCK:
        _CACHE[key] = (time.monotonic(), features)
    return dict(features)


def enrich_missing_features(symbol: str, features: dict[str, float]) -> dict[str, float]:
    """Fill only absent candle-derived fields; never overwrite live/context fields."""
    required = ("return_1", "range_pct", "volume_change", "volatility_proxy", "trend_strength", "momentum")
    missing = [name for name in required if name not in features]
    if not missing:
        return features
    selected_symbol = str(symbol or os.getenv("HHHAI_LIVE_FEATURE_SYMBOL", "BTCUSDT")).upper()
    live = get_live_candle_features(selected_symbol)
    merged = dict(features)
    for name in missing:
        if name in live:
            merged[name] = live[name]
    return merged
