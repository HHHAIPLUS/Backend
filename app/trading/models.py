"""Small compatibility layer for the non-executing decision engine.

Execution remains owned by the exchange adapters and safety gates.  These
models only describe a proposed trade and therefore cannot place an order.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class TradeProposal(BaseModel):
    symbol: str
    side: Side
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    reason: str = ""
