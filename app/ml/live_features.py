from __future__ import annotations

import os
import time
from threading import Lock
from typing import Any

import httpx

from app.ml.features import build_model_features

_BINANCE_URL = "https://fapi.binance.com/fapi/v1/klines"
_CACHE_TTL_SECONDS = 15.0
_DEFAULT_REST_COOLDOWN_SECONDS = 15 * 60
_CACHE: dict[str, tuple[float, dict[str, float]]] = {}
_LOCK = Lock()
_REST_BLOCKED_UNTIL = 0.0
_REST_BLOCK_REASON: str | None = None


def _cooldown_seconds() -> float:
    try:
        return max(60.0, float(os.getenv("HHHAI_BINANCE_REST_COOLDOWN_SECONDS", str(_DEFAULT_REST_COOLDOWN_SECONDS))))
    except ValueError:
        return float(_DEFAULT_REST_COOLDOWN_SECONDS)


def _check_rest_circuit() -> None:
    with _LOCK:
        blocked_until = _REST_BLOCKED_UNTIL
        reason = _REST_BLOCK_REASON
    now = time.monotonic()
    if blocked_until > now:
        remaining = max(1, int(blocked_until - now))
        raise RuntimeError(f"Binance REST circuit open for {remaining}s ({reason or 'rate limit'}); no retry attempted")


def _open_rest_circuit(response: httpx.Response) -> None:
    global _REST_BLOCKED_UNTIL, _REST_BLOCK_REASON
    retry_after = response.headers.get("Retry-After")
    try:
        wait_seconds = max(1.0, float(retry_after)) if retry_after is not None else _cooldown_seconds()
    except ValueError:
        wait_seconds = _cooldown_seconds()
    # Never retry immediately after a Binance 418/429. Respect Retry-After when supplied.
    with _LOCK:
        _REST_BLOCKED_UNTIL = time.monotonic() + wait_seconds
        _REST_BLOCK_REASON = f"HTTP {response.status_code}"


def _fetch(symbol: str, limit: int = 30) -> dict[str, float]:
    _check_rest_circuit()
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = httpx.get(
                _BINANCE_URL,
                params={"symbol": symbol.upper(), "interval": "5m", "limit": limit},
                timeout=httpx.Timeout(8.0, connect=4.0),
                follow_redirects=True,
                trust_env=False,
                headers={"User-Agent": "HHHAI/1.0", "Accept": "application/json"},
            )
            if response.status_code in {418, 429}:
                _open_rest_circuit(response)
                retry_after = response.headers.get("Retry-After")
                detail = f"HTTP {response.status_code}"
                if retry_after:
                    detail += f", Retry-After={retry_after}s"
                raise RuntimeError(f"Binance REST rate-limit protection triggered ({detail}); circuit opened and no further retries will be attempted")
            response.raise_for_status()
            raw: Any = response.json()
            if not isinstance(raw, list) or len(raw) < 3:
                raise RuntimeError("Binance returned insufficient 5m candles")
            candles = [row for row in raw if isinstance(row, list) and len(row) >= 6]
            features = build_model_features(candles)
            required = ("return_1", "range_pct", "volume_change", "volatility_proxy", "trend_strength", "momentum")
            missing = [name for name in required if name not in features]
            if missing:
                raise RuntimeError(f"Candle feature builder missing: {', '.join(missing)}")
            return {key: float(value) for key, value in features.items()}
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.35)
    raise RuntimeError(f"Binance 5m candle fetch failed after retries: {last_error}") from last_error


def get_live_candle_features(symbol: str) -> dict[str, float]:
    """Return fresh 5m candle-derived model features with a short cache and REST circuit breaker."""
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
