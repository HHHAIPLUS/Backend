from __future__ import annotations

from statistics import mean, pstdev
from typing import Iterable

from app.market_data.models import Candle
from ai.models import MarketRegime


def detect_regime(candles: Iterable[Candle]) -> MarketRegime:
    rows = list(candles)
    if len(rows) < 5:
        return MarketRegime.UNKNOWN

    returns = []
    for a, b in zip(rows[-5:-1], rows[-4:]):
        if a.close:
            returns.append(b.close / a.close - 1.0)

    if not returns:
        return MarketRegime.UNKNOWN

    volatility = pstdev(returns)
    drift = mean(returns)

    if volatility > 0.02:
        return MarketRegime.HIGH_VOLATILITY
    if volatility < 0.002:
        return MarketRegime.LOW_VOLATILITY
    if drift > 0.003:
        return MarketRegime.TRENDING_UP
    if drift < -0.003:
        return MarketRegime.TRENDING_DOWN
    return MarketRegime.RANGE
