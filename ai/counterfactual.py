from __future__ import annotations
from dataclasses import dataclass
from ai.cognitive_exit import PositionTelemetry, ExitAction, ExitDecision

@dataclass
class CounterfactualResult:
    selected: ExitAction
    utility_hold: float
    utility_reduce: float
    utility_exit: float
    explanation: str

class CounterfactualTradeTwin:
    """Small, transparent local scenario engine for open-position decisions.

    It is intentionally not presented as a market oracle. It asks: given the
    evidence we have right now, what happens to expected utility if we hold,
    reduce, or exit? This creates a second opinion against simple thresholding.
    """
    def evaluate(self, p: PositionTelemetry, decision: ExitDecision) -> CounterfactualResult:
        direction = 1 if p.side.lower() == 'long' else -1
        directional_edge = direction * p.momentum * (0.5 + 0.5 * p.trend_strength)
        stress = p.liquidity_stress * 0.5 + p.news_risk * 0.3 + (1-p.thesis_integrity) * 0.2
        profit = p.unrealized_return
        reversal = decision.reversal_score
        hold = directional_edge - stress - reversal * max(profit, 0.0) * 2.0
        reduce = 0.65 * directional_edge - 0.55 * stress + max(profit, 0.0) * 0.35
        exit_u = max(0.0, profit) * 0.9 - stress * 0.25
        utilities = {ExitAction.HOLD: hold, ExitAction.REDUCE: reduce, ExitAction.EXIT: exit_u}
        selected = max(utilities, key=utilities.get)
        # Never allow the twin to override an emergency decision.
        if decision.action == ExitAction.EMERGENCY_EXIT:
            selected = ExitAction.EMERGENCY_EXIT
        return CounterfactualResult(
            selected=selected,
            utility_hold=hold,
            utility_reduce=reduce,
            utility_exit=exit_u,
            explanation=f'Counterfactual utilities hold={hold:.4f}, reduce={reduce:.4f}, exit={exit_u:.4f}; selected={selected.value}.'
        )
