"""Canonical live feature construction for HHHAI's predictive model.

This module is intentionally small and dependency-light.  It exists as the
single compatibility boundary for callers that need model features from
recent candles plus live market/news context.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from app.ml.predictive import FEATURES


def _value(source: Any, key: str, default: float = 0.0) -> float:
    if source is None:
        return default
    if isinstance(source, Mapping):
        value = source.get(key, default)
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


def _candle_value(candle: Any, key: str, default: float = 0.0) -> float:
    return _value(candle, key, default)


def build_model_features(
    candles: Iterable[Any] | None = None,
    context: Any | None = None,
) -> dict[str, float]:
    """Build the exact feature dictionary expected by ``predictive_model``.

    ``candles`` may contain mappings or objects exposing ``open/high/low/close``
    and ``volume``.  ``context`` may be a mapping, a ``RealtimeSnapshot``, or a
    ``WorldIntelligence``-like object.  Missing live-only fields remain neutral
    rather than causing a request/import failure.
    """
    rows = list(candles or [])

    # A WorldIntelligence object stores exchange fields under ``market`` and
    # news fields at the top level.  A plain RealtimeSnapshot stores them on
    # the object itself.  Mappings are handled the same way.
    market = _nested(context, "market") or context

    last = rows[-1] if rows else None
    previous = rows[-2] if len(rows) >= 2 else last

    last_close = _candle_value(last, "close")
    previous_close = _candle_value(previous, "close")
    last_volume = _candle_value(last, "volume")
    previous_volume = _candle_value(previous, "volume")

    return_1 = (
        last_close / previous_close - 1.0
        if last_close > 0 and previous_close > 0
        else 0.0
    )
    range_pct = (
        (_candle_value(last, "high") - _candle_value(last, "low")) / last_close
        if last_close > 0
        else 0.0
    )
    volume_change = (
        last_volume / previous_volume - 1.0
        if previous_volume > 0
        else 0.0
    )

    # Prefer already-computed context values.  For candle-only callers,
    # reconstruct the short-term volatility/trend/momentum from closes.
    closes = [_candle_value(row, "close") for row in rows]
    returns = [
        closes[i] / closes[i - 1] - 1.0
        for i in range(1, len(closes))
        if closes[i] > 0 and closes[i - 1] > 0
    ]
    recent = returns[-12:]

    if recent:
        mean_return = sum(recent) / len(recent)
        if len(recent) > 1:
            mean_sq = sum(r * r for r in recent) / len(recent)
            variance = max(0.0, mean_sq - mean_return * mean_return)
            volatility = variance ** 0.5
        else:
            volatility = 0.0
    else:
        mean_return = 0.0
        volatility = 0.0

    volatility_proxy = _value(market, "volatility_proxy", min(1.0, max(0.0, volatility * 10.0)))
    trend_strength = _value(market, "trend_strength", min(1.0, abs(mean_return) * 80.0))
    momentum = _value(market, "momentum", max(-1.0, min(1.0, mean_return * 40.0)))

    # WorldIntelligence exposes these at the top level; a snapshot exposes
    # market fields only.  Read top-level first, then market for compatibility.
    news_risk = _value(context, "news_risk", _value(market, "news_risk"))
    news_sentiment = _value(context, "news_sentiment", _value(market, "news_sentiment"))
    liquidity_stress = _value(context, "liquidity_stress", _value(market, "liquidity_stress"))

    features = {
        "return_1": return_1,
        "range_pct": range_pct,
        "volume_change": volume_change,
        "order_book_imbalance": _value(market, "order_book_imbalance"),
        "funding_rate": _value(market, "funding_rate"),
        "open_interest_change": _value(market, "open_interest_change"),
        "news_risk": news_risk,
        "news_sentiment": news_sentiment,
        "volatility_proxy": volatility_proxy,
        "trend_strength": trend_strength,
        "momentum": momentum,
        "liquidity_stress": liquidity_stress,
    }

    # Keep the output contract exact and finite.
    clean: dict[str, float] = {}
    for name in FEATURES:
        value = features.get(name, 0.0)
        clean[name] = value if value == value and abs(value) != float("inf") else 0.0
    return clean
