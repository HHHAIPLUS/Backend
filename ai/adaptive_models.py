from __future__ import annotations
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class PositionAction(str, Enum):
    HOLD = 'hold'
    PROTECT_PROFIT = 'protect_profit'
    REDUCE = 'reduce'
    EXIT = 'exit'

class MarketCondition(str, Enum):
    FAVORABLE = 'favorable'
    NEUTRAL = 'neutral'
    DETERIORATING = 'deteriorating'
    CRITICAL = 'critical'

class PositionSnapshot(BaseModel):
    symbol: str
    side: str
    entry_price: float = Field(gt=0)
    current_price: float = Field(gt=0)
    unrealized_return: float
    peak_return: float
    opened_at: datetime
    confidence: float = Field(ge=0, le=1)

class MarketObservation(BaseModel):
    momentum: float = Field(ge=-1, le=1)
    trend_strength: float = Field(ge=0, le=1)
    selling_pressure: float = Field(ge=0, le=1)
    buying_pressure: float = Field(ge=0, le=1)
    volatility: float = Field(ge=0, le=1)
    liquidity_stress: float = Field(ge=0, le=1)
    news_risk: float = Field(ge=0, le=1)
    market_risk: float = Field(ge=0, le=1)
    thesis_integrity: float = Field(ge=0, le=1)

class AdaptiveDecision(BaseModel):
    action: PositionAction
    condition: MarketCondition
    score: float = Field(ge=0, le=1)
    reason: str
    protective_stop: float | None = None
    take_profit: float | None = None
    should_re_evaluate: bool = True
    timestamp: datetime

class PositionMemory(BaseModel):
    symbol: str
    side: str
    original_thesis: str
    highest_return: float = 0.0
    lowest_return: float = 0.0
    last_action: PositionAction = PositionAction.HOLD
    review_count: int = 0
    last_reason: str = ''
