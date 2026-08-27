from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ThesisCheck:
    integrity: float
    reasons: list[str]

def challenge_thesis(*, original_direction: str, momentum: float, trend_strength: float, buying_pressure: float, selling_pressure: float, news_risk: float, market_risk: float) -> ThesisCheck:
    bullish = original_direction.lower() == 'long'
    score = 0.50
    score += (0.18 * momentum if bullish else -0.18 * momentum)
    score += 0.18 * trend_strength
    score += (0.14 * buying_pressure if bullish else 0.14 * selling_pressure)
    score -= (0.18 * selling_pressure if bullish else 0.18 * buying_pressure)
    score -= 0.08 * news_risk + 0.08 * market_risk
    reasons=[]
    if news_risk >= .65: reasons.append('Elevated external-news risk.')
    if market_risk >= .65: reasons.append('Broader market risk is elevated.')
    if selling_pressure > buying_pressure: reasons.append('Selling pressure exceeds buying pressure.')
    if abs(momentum) < .15: reasons.append('Momentum is weak or indecisive.')
    return ThesisCheck(max(0,min(1,score)), reasons)
