from __future__ import annotations
from datetime import datetime, timezone
from ai.adaptive_models import AdaptiveDecision, MarketCondition, MarketObservation, PositionAction, PositionMemory, PositionSnapshot

class AdaptivePositionEngine:
    """Decision-only engine that continuously challenges an open position."""
    def __init__(self):
        self._memory: dict[str, PositionMemory] = {}

    def register_position(self, position: PositionSnapshot, thesis: str) -> PositionMemory:
        memory = PositionMemory(symbol=position.symbol, side=position.side, original_thesis=thesis)
        self._memory[position.symbol] = memory
        return memory

    def evaluate(self, position: PositionSnapshot, market: MarketObservation) -> AdaptiveDecision:
        memory = self._memory.setdefault(position.symbol, PositionMemory(symbol=position.symbol, side=position.side, original_thesis='Unknown'))
        memory.review_count += 1
        memory.highest_return = max(memory.highest_return, position.unrealized_return)
        memory.lowest_return = min(memory.lowest_return, position.unrealized_return)

        danger = min(1.0, max(0.0,
            0.24 * market.selling_pressure + 0.18 * market.news_risk +
            0.15 * market.market_risk + 0.12 * market.liquidity_stress +
            0.11 * (1 - market.thesis_integrity) + 0.10 * (1 - market.trend_strength) +
            0.10 * max(0.0, -market.momentum)))
        strength = min(1.0, max(0.0,
            0.30 * max(0.0, market.momentum) + 0.25 * market.trend_strength +
            0.20 * market.thesis_integrity + 0.15 * market.buying_pressure +
            0.10 * (1 - market.news_risk)))

        if position.unrealized_return > 0 and danger >= 0.68:
            action, condition, score, reason = PositionAction.EXIT, MarketCondition.CRITICAL, danger, 'The profitable position is deteriorating sharply; protect remaining profit rather than waiting for a fixed target.'
        elif position.unrealized_return > 0 and danger >= 0.48:
            action, condition, score, reason = PositionAction.PROTECT_PROFIT, MarketCondition.DETERIORATING, danger, 'The position is profitable but evidence is weakening; tighten protection and keep monitoring.'
        elif danger >= 0.78:
            action, condition, score, reason = PositionAction.EXIT, MarketCondition.CRITICAL, danger, 'Market danger is critical and the original position thesis is no longer reliable enough.'
        elif danger >= 0.58:
            action, condition, score, reason = PositionAction.REDUCE, MarketCondition.DETERIORATING, danger, 'Risk has increased enough to reduce exposure while preserving recovery potential.'
        elif strength >= 0.65:
            action, condition, score, reason = PositionAction.HOLD, MarketCondition.FAVORABLE, strength, 'The thesis remains supported; allow the position to develop without a mandatory fixed take-profit.'
        else:
            action, condition, score, reason = PositionAction.HOLD, MarketCondition.NEUTRAL, max(strength, 1-danger), 'Evidence is mixed; continue observing instead of forcing an exit or fixed target.'

        memory.last_action = action
        memory.last_reason = reason
        return AdaptiveDecision(action=action, condition=condition, score=score, reason=reason, protective_stop=self._protective_stop(position, market, action), take_profit=None, should_re_evaluate=True, timestamp=datetime.now(timezone.utc))

    @staticmethod
    def _protective_stop(position: PositionSnapshot, market: MarketObservation, action: PositionAction):
        if action not in {PositionAction.PROTECT_PROFIT, PositionAction.REDUCE}:
            return None
        buffer = 0.0025 + market.volatility * 0.005
        return position.current_price * (1-buffer) if position.side.lower() == 'long' else position.current_price * (1+buffer)

    def memory(self, symbol: str):
        return self._memory.get(symbol)
