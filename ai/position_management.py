from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from ai.adaptive_models import AdaptiveDecision, PositionAction, PositionSnapshot, MarketObservation


class ManagementAction(str, Enum):
    HOLD = 'hold'
    TRAIL_PROFIT = 'trail_profit'
    REDUCE = 'reduce'
    EXIT = 'exit'
    EMERGENCY_EXIT = 'emergency_exit'
    WAIT = 'wait'


class PositionState(BaseModel):
    symbol: str
    side: str
    entry_price: float = Field(gt=0)
    current_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    unrealized_return: float
    peak_return: float = 0.0
    protected_return: float = 0.0
    protection_price: Optional[float] = None
    management_action: ManagementAction = ManagementAction.WAIT
    review_count: int = 0
    opened_at: datetime
    last_review_at: Optional[datetime] = None
    last_reason: str = ''


class ManagementDecision(BaseModel):
    symbol: str
    action: ManagementAction
    close_fraction: float = Field(ge=0, le=1)
    protection_price: Optional[float] = None
    urgency: str
    reason: str
    execution_allowed: bool = False
    review_again: bool = True
    timestamp: datetime


class PositionManager:
    """Position-management layer. It plans actions but never executes exchange orders."""
    def __init__(self):
        self.positions: dict[str, PositionState] = {}

    def register(self, position: PositionSnapshot, quantity: float, protected_return: float = 0.0) -> PositionState:
        state = PositionState(
            symbol=position.symbol, side=position.side, entry_price=position.entry_price,
            current_price=position.current_price, quantity=quantity,
            unrealized_return=position.unrealized_return, peak_return=max(position.peak_return, position.unrealized_return),
            protected_return=protected_return, opened_at=position.opened_at,
        )
        self.positions[position.symbol] = state
        return state

    def review(self, position: PositionSnapshot, market: MarketObservation, adaptive: AdaptiveDecision) -> ManagementDecision:
        state = self.positions.get(position.symbol)
        if not state:
            self.register(position, quantity=1.0)
            state = self.positions[position.symbol]

        state.current_price = position.current_price
        state.unrealized_return = position.unrealized_return
        state.peak_return = max(state.peak_return, position.unrealized_return)
        state.review_count += 1
        state.last_review_at = datetime.now(timezone.utc)

        # Emergency: never wait on extreme data/market danger.
        if market.liquidity_stress >= 0.97 or (market.news_risk >= 0.97 and market.volatility >= 0.90):
            action = ManagementAction.EMERGENCY_EXIT
            fraction = 1.0
            protection = None
            urgency = 'critical'
            reason = 'Extreme market conditions detected. The position manager recommends immediate full exit; execution remains behind the risk authority.'
        elif adaptive.action == PositionAction.EXIT:
            action = ManagementAction.EXIT
            fraction = 1.0
            protection = None
            urgency = 'high'
            reason = adaptive.reason
        elif adaptive.action == PositionAction.REDUCE:
            action = ManagementAction.REDUCE
            fraction = 0.35 if position.unrealized_return >= 0 else 0.25
            protection = adaptive.protective_stop
            urgency = 'medium'
            reason = 'Risk has increased. Reduce part of the position and keep the remaining exposure protected.'
        elif adaptive.action == PositionAction.PROTECT_PROFIT:
            action = ManagementAction.TRAIL_PROFIT
            fraction = 0.0
            protection = self._dynamic_protection(position, market, state)
            urgency = 'medium'
            reason = 'The position is profitable but conditions are deteriorating. Protect the gain dynamically instead of waiting for a fixed take-profit.'
        elif adaptive.action == PositionAction.HOLD:
            action = ManagementAction.HOLD
            fraction = 0.0
            protection = self._dynamic_protection(position, market, state) if position.unrealized_return > 0.01 else None
            urgency = 'low'
            reason = 'Conditions still support the position. Allow it to run while continuously monitoring for deterioration.'
        else:
            action = ManagementAction.WAIT
            fraction = 0.0
            protection = None
            urgency = 'low'
            reason = 'Insufficient evidence for a management change. Continue observing.'

        if protection is not None:
            state.protection_price = self._more_protective(state.protection_price, protection, position.side)
            state.protected_return = self._return_at_price(position, state.protection_price)
            protection = state.protection_price

        state.management_action = action
        state.last_reason = reason
        return ManagementDecision(
            symbol=position.symbol, action=action, close_fraction=fraction,
            protection_price=protection, urgency=urgency, reason=reason,
            execution_allowed=False, review_again=True, timestamp=datetime.now(timezone.utc),
        )

    @staticmethod
    def _dynamic_protection(position: PositionSnapshot, market: MarketObservation, state: PositionState) -> Optional[float]:
        if position.unrealized_return <= 0:
            return None
        # Wider in high volatility, tighter when momentum/thesis deteriorates.
        volatility_buffer = 0.0015 + market.volatility * 0.006
        deterioration_tightener = (1 - market.thesis_integrity) * 0.004 + market.selling_pressure * 0.003
        buffer = min(0.025, max(0.001, volatility_buffer + deterioration_tightener))
        if position.side.lower() == 'long':
            return position.current_price * (1 - buffer)
        return position.current_price * (1 + buffer)

    @staticmethod
    def _more_protective(existing: Optional[float], new: float, side: str) -> float:
        if existing is None:
            return new
        # Never loosen protection during a management cycle.
        return max(existing, new) if side.lower() == 'long' else min(existing, new)

    @staticmethod
    def _return_at_price(position: PositionSnapshot, price: float) -> float:
        if position.side.lower() == 'long':
            return (price - position.entry_price) / position.entry_price
        return (position.entry_price - price) / position.entry_price

    def snapshot(self) -> list[PositionState]:
        return list(self.positions.values())
