from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence

from pydantic import BaseModel, Field


CANONICAL_FEATURES = (
    "return_1", "range_pct", "volume_change", "order_book_imbalance",
    "funding_rate", "open_interest_change", "news_risk", "news_sentiment",
    "volatility_proxy", "trend_strength", "momentum", "liquidity_stress",
)


class MarketBar(BaseModel):
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    quote_volume: float | None = Field(default=None, ge=0)


class PointInTimeContext(BaseModel):
    timestamp: datetime
    funding_rate: float | None = None
    open_interest: float | None = None
    liquidation_notional: float | None = Field(default=None, ge=0)
    spread_bps: float | None = Field(default=None, ge=0)
    order_book_imbalance: float | None = Field(default=None, ge=-1, le=1)
    aggressive_buy_ratio: float | None = Field(default=None, ge=0, le=1)
    news_sentiment: float | None = Field(default=None, ge=-1, le=1)
    news_risk: float | None = Field(default=None, ge=0, le=1)


class MarketState(BaseModel):
    symbol: str
    timestamp: datetime
    timeframes: dict[str, dict[str, float | None]]
    price_structure: dict[str, float]
    volatility: dict[str, float]
    volume: dict[str, float]
    order_flow: dict[str, float | None]
    derivatives: dict[str, float | None]
    liquidity: dict[str, float | None]
    correlations: dict[str, float | None]
    regime: dict[str, float | str]
    news: dict[str, float | int]
    data_quality: float = Field(ge=0, le=1)
    missing_fields: list[str] = Field(default_factory=list)
    stale_fields: list[str] = Field(default_factory=list)
    usable: bool


@dataclass(frozen=True)
class FeatureQuality:
    score: float
    missing: tuple[str, ...]
    stale: tuple[str, ...]
    usable: bool


def _returns(closes: Sequence[float]) -> list[float]:
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]


def _atr_like(bars: Sequence[MarketBar]) -> float:
    if len(bars) < 2: return 0.0
    tr = []
    for prev, cur in zip(bars[:-1], bars[1:]):
        tr.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)) / prev.close)
    return mean(tr[-14:]) if tr else 0.0


def _timeframe_features(bars: Sequence[MarketBar]) -> dict[str, float | None]:
    if not bars: return {"close": None, "return": None, "range_pct": None, "trend": None, "atr": None}
    closes = [b.close for b in bars]
    rets = _returns(closes)
    recent = rets[-20:]
    return {"close": closes[-1], "return": (closes[-1] / closes[0] - 1.0) if closes[0] else None, "range_pct": (max(b.high for b in bars) - min(b.low for b in bars)) / closes[-1], "trend": max(-1.0, min(1.0, sum(recent) * 8.0)) if recent else 0.0, "atr": _atr_like(bars)}


def _correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b) or len(a) < 3: return None
    ra, rb = _returns(a), _returns(b)
    n = min(len(ra), len(rb))
    if n < 3: return None
    ra, rb = ra[-n:], rb[-n:]
    ma, mb = mean(ra), mean(rb)
    da, db = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    if da <= 0 or db <= 0: return None
    return max(-1.0, min(1.0, sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (da * db) ** 0.5))


def point_in_time_join(bars: Sequence[MarketBar], contexts: Sequence[PointInTimeContext], *, max_age_seconds: int = 900) -> list[tuple[MarketBar, PointInTimeContext | None]]:
    """As-of join: context can only be used at or before the bar timestamp."""
    ordered_bars = sorted(bars, key=lambda x: x.timestamp)
    ordered_context = sorted(contexts, key=lambda x: x.timestamp)
    result, index, latest = [], 0, None
    for bar in ordered_bars:
        while index < len(ordered_context) and ordered_context[index].timestamp <= bar.timestamp:
            latest = ordered_context[index]; index += 1
        if latest is None:
            result.append((bar, None)); continue
        age = (bar.timestamp - latest.timestamp).total_seconds()
        result.append((bar, latest if 0 <= age <= max_age_seconds else None))
    return result


def assess_quality(state: Mapping[str, object], *, required: Iterable[str], observed_at: datetime, max_age_seconds: int = 60, now: datetime | None = None, threshold: float = 0.80) -> FeatureQuality:
    now = now or datetime.now(timezone.utc)
    required_list = list(required)
    missing = [key for key in required_list if state.get(key) is None or (isinstance(state.get(key), float) and not isfinite(float(state[key])))]
    stale = ["market_snapshot"] if max(0.0, (now - observed_at).total_seconds()) > max_age_seconds else []
    score = max(0.0, 1.0 - len(missing) / max(1, len(required_list)) - (0.25 if stale else 0.0))
    return FeatureQuality(score=score, missing=tuple(missing), stale=tuple(stale), usable=score >= threshold)


def build_market_state(symbol: str, bars_by_timeframe: Mapping[str, Sequence[MarketBar]], context: PointInTimeContext | None, *, correlation_bars: Mapping[str, Sequence[MarketBar]] | None = None, news_count: int = 0, news_sentiment: float = 0.0, news_risk: float = 0.0, news_credibility: float = 0.0, observed_at: datetime | None = None) -> MarketState:
    clean = {tf: sorted(list(bars), key=lambda x: x.timestamp) for tf, bars in bars_by_timeframe.items() if bars}
    if not clean: raise ValueError("At least one non-empty timeframe is required")
    primary_tf = "5m" if "5m" in clean else next(iter(clean)); primary = clean[primary_tf]
    timestamp = primary[-1].timestamp; observed_at = observed_at or timestamp
    closes = [b.close for b in primary]; rets = _returns(closes); vol_window = rets[-30:]
    vol = pstdev(vol_window) if len(vol_window) > 1 else 0.0; last = primary[-1]; prev = primary[-2] if len(primary) > 1 else last
    spread = context.spread_bps if context else None; imbalance = context.order_book_imbalance if context else None; oi = context.open_interest if context else None; funding = context.funding_rate if context else None; liq = context.liquidation_notional if context else None; buy_ratio = context.aggressive_buy_ratio if context else None
    timeframe_features = {tf: _timeframe_features(bars) for tf, bars in clean.items()}
    correlations: dict[str, float | None] = {}
    if correlation_bars:
        for other, other_bars in correlation_bars.items():
            aligned = min(len(primary), len(other_bars)); correlations[other] = _correlation(closes[-aligned:], [b.close for b in other_bars][-aligned:]) if aligned >= 3 else None
    structure = {"last_return": (last.close / prev.close - 1.0) if prev.close else 0.0, "range_pct_20": (max(b.high for b in primary[-20:]) - min(b.low for b in primary[-20:])) / last.close if last.close else 0.0, "trend": timeframe_features[primary_tf]["trend"] or 0.0, "distance_from_20_high": last.close / max(b.high for b in primary[-20:]) - 1.0, "distance_from_20_low": last.close / min(b.low for b in primary[-20:]) - 1.0}
    volume_mean = mean([b.volume for b in primary[-20:]]) if primary else 0.0; volume_ratio = last.volume / volume_mean if volume_mean > 0 else 1.0
    regime_score = max(-1.0, min(1.0, structure["trend"] * 0.7 + max(-1.0, min(1.0, (volume_ratio - 1.0) / 2.0)) * 0.3))
    regime = "HIGH_VOLATILITY" if vol > 0.008 else ("TRENDING_UP" if regime_score > 0.45 else "TRENDING_DOWN" if regime_score < -0.45 else "RANGE")
    state_map = {"price": last.close, "volume": last.volume, "funding_rate": funding, "open_interest": oi, "order_book_imbalance": imbalance, "spread_bps": spread, "news_sentiment": news_sentiment, "news_risk": news_risk}
    quality = assess_quality(state_map, required=("price", "volume", "funding_rate", "open_interest", "order_book_imbalance", "spread_bps"), observed_at=observed_at, max_age_seconds=120, threshold=0.67)
    return MarketState(symbol=symbol.upper(), timestamp=timestamp, timeframes=timeframe_features, price_structure=structure, volatility={"realized": vol, "atr": _atr_like(primary)}, volume={"last": last.volume, "mean_20": volume_mean, "ratio": volume_ratio}, order_flow={"imbalance": imbalance, "aggressive_buy_ratio": buy_ratio}, derivatives={"funding_rate": funding, "open_interest": oi, "open_interest_change": None, "liquidation_notional": liq}, liquidity={"spread_bps": spread, "bid_ask_mid": last.close}, correlations=correlations, regime={"label": regime, "score": regime_score, "volatility": vol}, news={"count": news_count, "sentiment": news_sentiment, "risk": news_risk, "credibility": news_credibility}, data_quality=quality.score, missing_fields=list(quality.missing), stale_fields=list(quality.stale), usable=quality.usable)


def canonical_model_features(state: MarketState) -> dict[str, float]:
    """Single projection used by both live inference and historical training."""
    tf5 = state.timeframes.get("5m") or next(iter(state.timeframes.values()))
    return {
        "return_1": float(state.price_structure.get("last_return", 0.0)),
        "range_pct": float(state.price_structure.get("range_pct_20", 0.0)),
        "volume_change": float(state.volume.get("ratio", 1.0) - 1.0),
        "order_book_imbalance": float(state.order_flow.get("imbalance") or 0.0),
        "funding_rate": float(state.derivatives.get("funding_rate") or 0.0),
        "open_interest_change": float(state.derivatives.get("open_interest_change") or 0.0),
        "news_risk": float(state.news.get("risk", 0.0)),
        "news_sentiment": float(state.news.get("sentiment", 0.0)),
        "volatility_proxy": float(state.volatility.get("realized", 0.0)),
        "trend_strength": float(tf5.get("trend") or 0.0),
        "momentum": float(tf5.get("return") or 0.0),
        "liquidity_stress": max(0.0, min(1.0, float(state.liquidity.get("spread_bps") or 0.0) / 50.0)),
    }


def aggregate_timeframes(bars: Sequence[MarketBar], minutes: int) -> list[MarketBar]:
    if minutes <= 0: raise ValueError("minutes must be positive")
    groups: dict[int, list[MarketBar]] = defaultdict(list); bucket_ms = minutes * 60 * 1000
    for bar in sorted(bars, key=lambda x: x.timestamp): groups[int(bar.timestamp.timestamp() * 1000) // bucket_ms].append(bar)
    output = []
    for group in groups.values():
        if group: output.append(MarketBar(symbol=group[0].symbol, timeframe=f"{minutes}m", timestamp=group[0].timestamp, open=group[0].open, high=max(x.high for x in group), low=min(x.low for x in group), close=group[-1].close, volume=sum(x.volume for x in group), quote_volume=sum(x.quote_volume or 0 for x in group)))
    return output
