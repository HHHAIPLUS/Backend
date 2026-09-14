"""Canonical point-in-time feature construction for HHHAI."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import math

from app.ml.predictive import FEATURES

_CANDLE_INDEX = {"timestamp": 0, "open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}


def _value(source: Any, key: str, default: float = 0.0) -> float:
    if source is None:
        return default
    if isinstance(source, Mapping):
        value = source.get(key, default)
    elif isinstance(source, (list, tuple)):
        index = _CANDLE_INDEX.get(key)
        value = source[index] if index is not None and len(source) > index else default
    else:
        value = getattr(source, key, default)
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _nested(source: Any, key: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _context_values(context: Any) -> Mapping[str, Any]:
    if context is None:
        return {}
    values = getattr(context, "values", None)
    available = getattr(context, "available", None)
    if isinstance(values, Mapping) and available is not None:
        return {name: values[name] for name in available if name in values}
    if isinstance(context, Mapping):
        nested = context.get("values")
        if isinstance(nested, Mapping):
            return nested
    return {}


def _safe_return(closes: list[float], lookback: int) -> float:
    if len(closes) <= lookback or closes[-1] <= 0 or closes[-1 - lookback] <= 0:
        return 0.0
    return closes[-1] / closes[-1 - lookback] - 1.0


def build_model_features(candles: Iterable[Any] | None = None, context: Any | None = None) -> dict[str, float]:
    """Build the exact predictive feature vector from information available at T.

    The six OHLCV-derived features intentionally use multiple recent windows rather
    than only the last candle. This keeps training and live inference point-in-time
    consistent while giving the model trend, momentum, volatility and volume context.
    """
    rows = list(candles or [])
    market = _nested(context, "market") or context
    historical = _context_values(context)

    last = rows[-1] if rows else None
    previous = rows[-2] if len(rows) >= 2 else last
    last_close = _value(last, "close")
    previous_close = _value(previous, "close")
    last_volume = _value(last, "volume")
    previous_volume = _value(previous, "volume")

    return_1 = last_close / previous_close - 1.0 if last_close > 0 and previous_close > 0 else 0.0
    range_pct = (_value(last, "high") - _value(last, "low")) / last_close if last_close > 0 else 0.0
    volume_change = math.log1p(max(0.0, last_volume)) - math.log1p(max(0.0, previous_volume)) if previous_volume >= 0 else 0.0

    closes = [_value(row, "close") for row in rows]
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i] > 0 and closes[i - 1] > 0]
    recent = returns[-24:]

    if len(recent) > 1:
        mean_return = sum(recent) / len(recent)
        variance = sum((r - mean_return) ** 2 for r in recent) / (len(recent) - 1)
        volatility = math.sqrt(max(0.0, variance))
    else:
        mean_return = 0.0
        volatility = 0.0

    # Momentum is signed multi-candle price movement. Trend strength is the
    # normalized absolute slope of recent closes, avoiding the old duplicate
    # "mean return" calculations for momentum and trend.
    momentum_6 = _safe_return(closes, 6)
    momentum_12 = _safe_return(closes, 12)
    momentum = max(-1.0, min(1.0, 0.60 * momentum_6 + 0.40 * momentum_12))

    trend_window = closes[-24:]
    trend_strength = 0.0
    if len(trend_window) >= 8 and all(value > 0 for value in trend_window):
        x_mean = (len(trend_window) - 1) / 2.0
        y_mean = sum(math.log(value) for value in trend_window) / len(trend_window)
        numerator = sum((i - x_mean) * (math.log(value) - y_mean) for i, value in enumerate(trend_window))
        denominator = sum((i - x_mean) ** 2 for i in range(len(trend_window)))
        slope = numerator / denominator if denominator else 0.0
        trend_strength = min(1.0, abs(slope) / max(volatility, 1e-6) * 4.0)

    # Normalize volume change against recent volume history so one abnormal
    # candle does not dominate the feature indefinitely.
    volumes = [_value(row, "volume") for row in rows[-24:]]
    volume_change_raw = volume_change
    if len(volumes) >= 8:
        log_volumes = [math.log1p(max(0.0, value)) for value in volumes]
        v_mean = sum(log_volumes) / len(log_volumes)
        v_var = sum((value - v_mean) ** 2 for value in log_volumes) / max(1, len(log_volumes) - 1)
        v_std = math.sqrt(max(v_var, 1e-12))
        volume_change = max(-5.0, min(5.0, (log_volumes[-1] - v_mean) / v_std))
    else:
        volume_change = max(-5.0, min(5.0, volume_change_raw))

    def context_or_live(name: str, default: float = 0.0) -> float:
        if name in historical:
            return _value(historical, name, default)
        return _value(market, name, default)

    features = {
        "return_1": return_1,
        "range_pct": range_pct,
        "volume_change": volume_change,
        "order_book_imbalance": context_or_live("order_book_imbalance"),
        "funding_rate": context_or_live("funding_rate"),
        "open_interest_change": context_or_live("open_interest_change"),
        "news_risk": _value(context, "news_risk", context_or_live("news_risk")),
        "news_sentiment": _value(context, "news_sentiment", context_or_live("news_sentiment")),
        "volatility_proxy": min(1.0, max(0.0, volatility * 12.0)),
        "trend_strength": trend_strength,
        "momentum": momentum,
        "liquidity_stress": context_or_live("liquidity_stress"),
    }
    normalized: dict[str, float] = {}
    for name in FEATURES:
        try:
            value = float(features.get(name, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        normalized[name] = value if value == value and abs(value) != float("inf") else 0.0
    return normalized
