from __future__ import annotations

from dataclasses import dataclass

from ai.evaluation import EvaluationResult


@dataclass
class CandidatePromotionPolicy:
    min_trades: int = 100
    min_win_rate: float = 0.50
    min_average_return: float = 0.0


def can_promote(result: EvaluationResult, policy: CandidatePromotionPolicy) -> bool:
    """Gate model promotion.

    This is deliberately conservative. A candidate must pass minimum sample
    size and performance gates before it can become a production candidate.
    More robust out-of-sample and risk-adjusted gates will be added later.
    """
    return (
        result.trades >= policy.min_trades
        and result.win_rate >= policy.min_win_rate
        and result.average_return > policy.min_average_return
    )
