from __future__ import annotations

from statistics import pstdev
from typing import Any


def build_model_features(
    klines: list[list[Any]],
) -> dict[str, float]:
    """Build the exact price/volume features used by the predictive model.

    This function is intentionally shared by historical bootstrap data and
    live market data so the model sees the same feature definitions in both
    environments.
    """
    candles: list[tuple[int, float, float, float, float, float]] = []

    for row in klines:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            timestamp = int(row[0])
            open_price = float(row[1])
            high = float(row[2])
            low = float(row[3])
            close = float(row[4])
            volume = max(0.0, float(row[5]))
        except (TypeError, ValueError):
            continue

        if timestamp <= 0 or min(open_price, high, low, close) <= 0:
            continue
        if high < low:
            continue
        candles.append((timestamp, open_price, high, low, close, volume))

    candles.sort(key=lambda row: row[0])

    if len(candles) < 2:
        raise ValueError("At least two valid candles are required for model features.")

    returns = [
        candles[i][4] / candles[i - 1][4] - 1.0
        for i in range(1, len(candles))
        if candles[i - 1][4] > 0
    ]

    last = candles[-1]
    previous = candles[-2]
    recent_returns = returns[-12:]

    mean_ret = float(sum(recent_returns) / len(recent_returns)) if recent_returns else 0.0
    volatility = float(pstdev(recent_returns)) if len(recent_returns) > 1 else 0.0

    volume_change = (
        last[5] / previous[5] - 1.0
        if previous[5] > 0
        else 0.0
    )

    return {
        "return_1": returns[-1] if returns else 0.0,
        "range_pct": (last[2] - last[3]) / last[4] if last[4] > 0 else 0.0,
        "volume_change": volume_change,
        "volatility_proxy": min(1.0, max(0.0, volatility * 10.0)),
        "trend_strength": min(1.0, abs(mean_ret) * 80.0),
        "momentum": max(-1.0, min(1.0, mean_ret * 40.0)),
    }
