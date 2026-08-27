from __future__ import annotations

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class Prediction(BaseModel):
    symbol: str
    timestamp: datetime
    long_probability: float = Field(ge=0, le=1)
    short_probability: float = Field(ge=0, le=1)
    no_trade_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    regime: MarketRegime = MarketRegime.UNKNOWN
    model_version: str
    rationale: list[str] = []


class NewsSignal(BaseModel):
    source: str
    published_at: datetime
    headline: str
    relevance: float = Field(ge=0, le=1)
    sentiment: float = Field(ge=-1, le=1)
    impact: float = Field(ge=0, le=1)
    credibility: float = Field(ge=0, le=1)


class FeatureVector(BaseModel):
    symbol: str
    timestamp: datetime
    features: dict[str, float]


class TradeOutcome(BaseModel):
    trade_id: str
    model_version: str
    predicted_side: str
    realized_return: float
    was_profitable: bool
    timestamp: datetime
