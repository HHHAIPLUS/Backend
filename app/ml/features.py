"""Canonical point-in-time feature construction for HHHAI."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

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


def build_model_features(candles: Iterable[Any] | None = None, context: Any | None = None) -> dict[str, float]:
    """Build the exact predictive feature vector from information available at T."""
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
    volume_change = last_volume / previous_volume - 1.0 if previous_volume > 0 else 0.0

    closes = [_value(row, "close") for row in rows]
    returns = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i] > 0 and closes[i - 1] > 0]
    recent = returns[-12:]
    mean_return = sum(recent) / len(recent) if recent else 0.0
    if len(recent) > 1:
        mean_sq = sum(r * r for r in recent) / len(recent)
        volatility = max(0.0, mean_sq - mean_return * mean_return) ** 0.5
    else:
        volatility = 0.0

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
        "volatility_proxy": context_or_live("volatility_proxy", min(1.0, max(0.0, volatility * 10.0))),
        "trend_strength": context_or_live("trend_strength", min(1.0, abs(mean_return) * 80.0)),
        "momentum": context_or_live("momentum", max(-1.0, min(1.0, mean_return * 40.0))),
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
