from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import exp

class ExitAction(str, Enum):
    HOLD = 'hold'
    PROTECT = 'protect'
    REDUCE = 'reduce'
    EXIT = 'exit'
    EMERGENCY_EXIT = 'emergency_exit'

@dataclass
class PositionTelemetry:
    side: str
    entry_price: float
    current_price: float
    unrealized_return: float
    peak_return: float
    minutes_open: float
    momentum: float
    trend_strength: float
    buying_pressure: float
    selling_pressure: float
    volatility: float
    liquidity_stress: float
    news_risk: float
    thesis_integrity: float
    funding_bias: float = 0.0
    open_interest_change: float = 0.0

@dataclass
class ExitDecision:
    action: ExitAction
    score: float
    profit_lock_score: float
    thesis_break_score: float
    reversal_score: float
    adverse_score: float
    protection_price: float | None
    close_fraction: float
    reason: str

class CognitiveExitEngine:
    """Adaptive position governor.

    It deliberately does not use a fixed take-profit as the primary exit rule.
    It combines current edge, reversal evidence, realized profit, drawdown from
    the trade's best unrealized return, market stress, and thesis integrity.
    """
    version = 'cognitive-exit-v1'

    @staticmethod
    def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def evaluate(self, p: PositionTelemetry) -> ExitDecision:
        profit = max(0.0, p.unrealized_return)
        retracement = 0.0
        if p.peak_return > 0:
            retracement = self._clamp((p.peak_return - p.unrealized_return) / p.peak_return)

        direction_momentum = p.momentum if p.side.lower() == 'long' else -p.momentum
        direction_pressure = p.buying_pressure if p.side.lower() == 'long' else p.selling_pressure
        opposing_pressure = p.selling_pressure if p.side.lower() == 'long' else p.buying_pressure

        reversal = self._clamp(
            0.34 * max(0.0, -direction_momentum) +
            0.24 * opposing_pressure +
            0.18 * (1 - p.trend_strength) +
            0.14 * retracement +
            0.10 * p.volatility
        )
        thesis_break = self._clamp(
            0.28 * (1 - p.thesis_integrity) +
            0.20 * max(0.0, -direction_momentum) +
            0.18 * (1 - direction_pressure) +
            0.14 * p.news_risk +
            0.12 * p.liquidity_stress +
            0.08 * abs(p.open_interest_change) * 0.5
        )
        adverse = self._clamp(
            0.34 * p.liquidity_stress +
            0.26 * p.news_risk +
            0.20 * max(0.0, -p.unrealized_return * 12) +
            0.20 * max(0.0, -direction_momentum)
        )
        profit_lock = self._clamp(
            0.38 * self._clamp(profit * 35) +
            0.30 * retracement +
            0.20 * reversal +
            0.12 * thesis_break
        )

        if p.liquidity_stress >= 0.92 or adverse >= 0.90:
            action = ExitAction.EMERGENCY_EXIT
            fraction = 1.0
            reason = 'Extreme liquidity/market stress makes waiting for a fixed target unsafe.'
        elif profit > 0 and (profit_lock >= 0.72 or reversal >= 0.78 or thesis_break >= 0.76):
            action = ExitAction.EXIT
            fraction = 1.0
            reason = 'Profit is present but the probability of giving it back has become materially higher than the value of waiting.'
        elif profit > 0 and (profit_lock >= 0.52 or reversal >= 0.55):
            action = ExitAction.REDUCE
            fraction = 0.35 if reversal < 0.70 else 0.50
            reason = 'The trade is profitable but its edge is deteriorating; bank part of the gain and let the remainder prove the thesis.'
        elif profit > 0 and (reversal >= 0.38 or thesis_break >= 0.45):
            action = ExitAction.PROTECT
            fraction = 0.0
            reason = 'The trade remains profitable, but the governor is tightening protection rather than waiting for the original target.'
        elif p.unrealized_return < 0 and (thesis_break >= 0.72 or adverse >= 0.72):
            action = ExitAction.EXIT
            fraction = 1.0
            reason = 'The original edge is sufficiently damaged that continuing to hope for recovery is inferior to exiting.'
        else:
            action = ExitAction.HOLD
            fraction = 0.0
            reason = 'The current evidence still supports keeping the position under continuous review.'

        protection = self._protection(p, action, reversal, thesis_break)
        return ExitDecision(action, max(profit_lock, reversal, thesis_break, adverse), profit_lock, thesis_break, reversal, adverse, protection, fraction, reason)

    def _protection(self, p: PositionTelemetry, action: ExitAction, reversal: float, thesis_break: float) -> float | None:
        if p.unrealized_return <= 0 or action not in {ExitAction.PROTECT, ExitAction.REDUCE, ExitAction.HOLD}:
            return None
        # The stop only moves toward profit. Higher reversal/thesis break -> tighter.
        base = 0.0015 + p.volatility * 0.006
        tighten = 0.0035 * max(reversal, thesis_break)
        buffer = self._clamp(base + tighten, 0.0010, 0.022)
        if p.side.lower() == 'long':
            return p.current_price * (1 - buffer)
        return p.current_price * (1 + buffer)
