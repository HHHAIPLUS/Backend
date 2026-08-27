from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from ai.models import TradeOutcome


@dataclass
class EvaluationResult:
    trades: int
    win_rate: float
    average_return: float


def evaluate_outcomes(outcomes: Iterable[TradeOutcome]) -> EvaluationResult:
    rows = list(outcomes)
    if not rows:
        return EvaluationResult(0, 0.0, 0.0)

    return EvaluationResult(
        trades=len(rows),
        win_rate=sum(x.was_profitable for x in rows) / len(rows),
        average_return=mean(x.realized_return for x in rows),
    )
