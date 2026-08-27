from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from math import exp
from pydantic import BaseModel, Field


class ScenarioKind(str, Enum):
    BULLISH_CONTINUATION = "bullish_continuation"
    BEARISH_CONTINUATION = "bearish_continuation"
    RANGE_REVERSION = "range_reversion"
    VOLATILITY_EXPANSION = "volatility_expansion"
    DISORDERLY_RISK_OFF = "disorderly_risk_off"


class Scenario(BaseModel):
    kind: ScenarioKind
    probability: float = Field(ge=0, le=1)
    expected_move: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    invalidation: str
    evidence: list[str] = Field(default_factory=list)


class ScenarioRequest(BaseModel):
    symbol: str
    horizon_minutes: int = Field(default=60, ge=5, le=1440)
    momentum: float = Field(ge=-1, le=1)
    trend_strength: float = Field(ge=0, le=1)
    buying_pressure: float = Field(ge=0, le=1)
    selling_pressure: float = Field(ge=0, le=1)
    volatility: float = Field(ge=0, le=1)
    liquidity_stress: float = Field(ge=0, le=1)
    news_risk: float = Field(ge=0, le=1)
    news_sentiment: float = Field(ge=-1, le=1)
    market_risk: float = Field(ge=0, le=1)
    thesis_integrity: float = Field(ge=0, le=1, default=0.5)


class ScenarioReport(BaseModel):
    symbol: str
    horizon_minutes: int
    scenarios: list[Scenario]
    dominant_scenario: ScenarioKind
    uncertainty: float = Field(ge=0, le=1)
    expected_move: float = Field(ge=-1, le=1)
    recommendation: str
    model_version: str
    generated_at: datetime


class ScenarioEngine:
    model_version = "scenario-core-1.0"

    def generate(self, r: ScenarioRequest) -> ScenarioReport:
        bull = max(0.0, 0.5 * r.momentum + 0.35 * r.trend_strength + 0.25 * r.buying_pressure + 0.20 * max(r.news_sentiment, 0) - 0.25 * r.selling_pressure - 0.20 * r.news_risk - 0.15 * r.market_risk)
        bear = max(0.0, -0.5 * r.momentum + 0.35 * r.trend_strength + 0.25 * r.selling_pressure + 0.20 * max(-r.news_sentiment, 0) - 0.25 * r.buying_pressure + 0.20 * r.news_risk + 0.20 * r.market_risk)
        range_score = max(0.0, 0.65 * (1 - r.trend_strength) + 0.25 * (1 - abs(r.momentum)) + 0.10 * (1 - r.news_risk))
        expansion = max(0.0, 0.70 * r.volatility + 0.30 * r.liquidity_stress)
        disorder = max(0.0, 0.85 * r.liquidity_stress + 0.45 * r.market_risk + 0.25 * r.news_risk + 0.20 * max(-r.news_sentiment, 0) + 0.15 * r.volatility)

        raw = [bull, bear, range_score, expansion, disorder]
        total = sum(raw) or 1.0
        probs = [x / total for x in raw]

        labels = [
            ScenarioKind.BULLISH_CONTINUATION,
            ScenarioKind.BEARISH_CONTINUATION,
            ScenarioKind.RANGE_REVERSION,
            ScenarioKind.VOLATILITY_EXPANSION,
            ScenarioKind.DISORDERLY_RISK_OFF,
        ]
        moves = [
            0.018 + 0.035 * r.trend_strength,
            -(0.018 + 0.035 * r.trend_strength),
            0.0,
            0.0,
            -(0.015 + 0.045 * r.liquidity_stress),
        ]
        evidence = [
            ["Positive momentum/trend evidence."],
            ["Negative momentum/flow evidence."],
            ["Trend conviction is weak."],
            ["Volatility is elevated."],
            ["Liquidity or broad-market stress is elevated."],
        ]
        invalidations = [
            "Momentum loses support and selling pressure dominates.",
            "Momentum recovers and buying pressure dominates.",
            "A sustained directional breakout develops.",
            "Volatility normalizes and liquidity stabilizes.",
            "Liquidity stress and market risk normalize.",
        ]
        scenarios = []
        for i, kind in enumerate(labels):
            p = probs[i]
            # Confidence measures evidence quality, not certainty of the future.
            confidence = max(0.0, min(1.0, 0.35 + 0.45 * p + 0.20 * r.thesis_integrity))
            scenarios.append(Scenario(kind=kind, probability=p, expected_move=moves[i], confidence=confidence, invalidation=invalidations[i], evidence=evidence[i]))

        dominant_index = max(range(len(probs)), key=probs.__getitem__)
        entropy = -sum(p * (0 if p == 0 else __import__('math').log(p)) for p in probs)
        uncertainty = max(0.0, min(1.0, entropy / __import__('math').log(len(probs))))
        expected_move = sum(s.probability * s.expected_move for s in scenarios)
        dominant = scenarios[dominant_index].kind

        if uncertainty >= 0.78:
            recommendation = "Conditions are ambiguous. Reduce conviction and wait for clearer evidence."
        elif dominant == ScenarioKind.DISORDERLY_RISK_OFF:
            recommendation = "Risk-off scenario dominates. Preserve capital and avoid aggressive exposure."
        elif dominant == ScenarioKind.VOLATILITY_EXPANSION:
            recommendation = "Volatility expansion is dominant. Expect wider outcomes and tighten risk controls."
        elif dominant == ScenarioKind.BULLISH_CONTINUATION:
            recommendation = "Bullish continuation has the strongest scenario weight, but keep challenging the thesis."
        elif dominant == ScenarioKind.BEARISH_CONTINUATION:
            recommendation = "Bearish continuation has the strongest scenario weight, but confirmation is still required."
        else:
            recommendation = "Range/reversion is the strongest scenario. Avoid forcing directional conviction."

        return ScenarioReport(
            symbol=r.symbol,
            horizon_minutes=r.horizon_minutes,
            scenarios=scenarios,
            dominant_scenario=dominant,
            uncertainty=uncertainty,
            expected_move=expected_move,
            recommendation=recommendation,
            model_version=self.model_version,
            generated_at=datetime.now(timezone.utc),
        )
