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
    """Build a point-in-time multi-horizon technical feature vector."""
    rows = list(candles or [])
    market = _nested(context, "market") or context
    historical = _context_values(context)
    closes = [_value(row, "close") for row in rows]
    highs = [_value(row, "high") for row in rows]
    lows = [_value(row, "low") for row in rows]
    volumes = [_value(row, "volume") for row in rows]
    last_close = closes[-1] if closes else 0.0

    def ret(n: int) -> float:
        return _safe_return(closes, n)

    one = ret(1)
    range_pct = ((_value(rows[-1], "high") - _value(rows[-1], "low")) / last_close) if rows and last_close > 0 else 0.0
    ranges = [((h-l)/c if c > 0 else 0.0) for h,l,c in zip(highs,lows,closes)]
    range_mean_12 = sum(ranges[-12:]) / max(1, len(ranges[-12:]))

    true_ranges = []
    for i in range(1, len(rows)):
        c = closes[i]
        prev = closes[i-1]
        if c <= 0 or prev <= 0: continue
        true_ranges.append(max(highs[i]-lows[i], abs(highs[i]-prev), abs(lows[i]-prev)) / c)
    atr_pct_14 = sum(true_ranges[-14:]) / max(1, len(true_ranges[-14:]))

    close_location = 0.0
    if rows and highs[-1] > lows[-1]:
        close_location = max(-1.0, min(1.0, 2.0 * (last_close-lows[-1]) / (highs[-1]-lows[-1]) - 1.0))

    log_vol = [math.log1p(max(0.0, v)) for v in volumes]
    volume_change = (log_vol[-1] - log_vol[-2]) if len(log_vol) >= 2 else 0.0
    recent_vol = log_vol[-24:]
    vmean = sum(recent_vol) / max(1, len(recent_vol))
    vvar = sum((v-vmean)**2 for v in recent_vol) / max(1, len(recent_vol)-1)
    volume_zscore = (log_vol[-1]-vmean) / math.sqrt(max(vvar, 1e-12)) if log_vol else 0.0

    def ema(values: list[float], span: int) -> float:
        if not values: return 0.0
        alpha = 2.0 / (span + 1.0)
        out = values[0]
        for value in values[1:]: out = alpha * value + (1.0-alpha) * out
        return out

    ema8, ema24 = ema(closes[-32:], 8), ema(closes[-32:], 24)
    ema_gap_8_24 = (ema8/ema24 - 1.0) if ema24 > 0 else 0.0

    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i-1]
        gains.append(max(delta, 0.0)); losses.append(max(-delta, 0.0))
    rg, rl = gains[-14:], losses[-14:]
    avg_gain = sum(rg)/max(1,len(rg)); avg_loss = sum(rl)/max(1,len(rl))
    rsi_14 = 0.0 if avg_gain == 0 and avg_loss == 0 else (1.0 if avg_loss == 0 else (avg_gain/(avg_gain+avg_loss))*2.0-1.0)

    prev_window = closes[-25:-1] if len(closes) >= 25 else closes[:-1]
    breakout_24 = 0.0
    if prev_window and last_close > 0:
        hi=max(prev_window); lo=min(prev_window)
        breakout_24 = max(-1.0, min(1.0, (last_close-hi)/last_close if last_close>hi else (last_close-lo)/last_close if last_close<lo else 0.0))

    recent_returns = [closes[i]/closes[i-1]-1.0 for i in range(1,len(closes)) if closes[i]>0 and closes[i-1]>0][-24:]
    mean_return = sum(recent_returns)/max(1,len(recent_returns))
    variance = sum((r-mean_return)**2 for r in recent_returns)/max(1,len(recent_returns)-1)
    volatility = math.sqrt(max(0.0, variance))
    momentum = max(-1.0, min(1.0, 0.50*ret(6) + 0.30*ret(12) + 0.20*ret(24)))

    trend_window=closes[-24:]
    trend_strength=0.0
    if len(trend_window)>=8 and all(v>0 for v in trend_window):
        xm=(len(trend_window)-1)/2.0; ym=sum(math.log(v) for v in trend_window)/len(trend_window)
        num=sum((i-xm)*(math.log(v)-ym) for i,v in enumerate(trend_window))
        den=sum((i-xm)**2 for i in range(len(trend_window)))
        slope=num/den if den else 0.0
        trend_strength=max(-1.0,min(1.0,slope/max(volatility,1e-6)*4.0))

    def context_or_live(name: str, default: float = 0.0) -> float:
        if name in historical: return _value(historical,name,default)
        return _value(market,name,default)

    features = {
        "return_1": one, "return_3": ret(3), "return_6": ret(6), "return_12": ret(12), "return_24": ret(24),
        "range_pct": range_pct, "range_mean_12": range_mean_12, "close_location": close_location,
        "atr_pct_14": atr_pct_14, "volume_change": max(-5.0,min(5.0,volume_change)),
        "volume_zscore": max(-5.0,min(5.0,volume_zscore)), "rsi_14": rsi_14,
        "ema_gap_8_24": max(-1.0,min(1.0,ema_gap_8_24)), "breakout_24": breakout_24,
        "order_book_imbalance": context_or_live("order_book_imbalance"),
        "funding_rate": context_or_live("funding_rate"), "open_interest_change": context_or_live("open_interest_change"),
        "news_risk": _value(context,"news_risk",context_or_live("news_risk")),
        "news_sentiment": _value(context,"news_sentiment",context_or_live("news_sentiment")),
        "volatility_proxy": min(1.0,max(0.0,volatility*12.0)), "trend_strength": trend_strength,
        "momentum": momentum, "liquidity_stress": context_or_live("liquidity_stress"),
    }
    # Return the full canonical market feature state. PredictiveModel/Ensemble select
    # their explicit FEATURES subset, while decision layers can still consume
    # live context features without contaminating historical supervised training.
    return {name: (float(value or 0.0) if math.isfinite(float(value or 0.0)) else 0.0) for name, value in features.items()}

