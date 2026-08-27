from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field


class AgentAction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


class AgentObservation(BaseModel):
    agent_id: str
    name: str
    action: AgentAction
    score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    data_quality: float = Field(ge=0, le=1, default=1.0)
    timestamp: datetime


class AgentContext(BaseModel):
    symbol: str
    momentum: float = Field(ge=-1, le=1)
    trend_strength: float = Field(ge=0, le=1)
    buying_pressure: float = Field(ge=0, le=1)
    selling_pressure: float = Field(ge=0, le=1)
    volatility: float = Field(ge=0, le=1)
    liquidity_stress: float = Field(ge=0, le=1)
    news_risk: float = Field(ge=0, le=1)
    news_sentiment: float = Field(ge=-1, le=1)
    news_credibility: float = Field(ge=0, le=1)
    funding_bias: float = Field(ge=-1, le=1)
    open_interest_change: float = Field(ge=-1, le=1)
    correlation_risk: float = Field(ge=0, le=1)
    market_regime: str = "unknown"
    position_side: str | None = None
    unrealized_return: float = 0.0
    thesis_integrity: float = Field(ge=0, le=1, default=0.5)


class CouncilDecision(BaseModel):
    symbol: str
    action: AgentAction
    score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    disagreement: float = Field(ge=0, le=1)
    risk_level: str
    reasons: list[str] = Field(default_factory=list)
    veto_flags: list[str] = Field(default_factory=list)
    agents: list[AgentObservation]
    timestamp: datetime


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    name: str
    weight: float
    run: Callable[[AgentContext], AgentObservation]


def _obs(agent_id: str, name: str, score: float, confidence: float, reasons: list[str], quality: float = 1.0, action: AgentAction | None = None) -> AgentObservation:
    if action is None:
        if score > 0.18:
            action = AgentAction.BULLISH
        elif score < -0.18:
            action = AgentAction.BEARISH
        else:
            action = AgentAction.NEUTRAL
    return AgentObservation(agent_id=agent_id, name=name, action=action, score=max(-1, min(1, score)), confidence=max(0, min(1, confidence)), reasons=reasons, data_quality=quality, timestamp=datetime.now(timezone.utc))


def market_agent(c: AgentContext) -> AgentObservation:
    score = 0.42*c.momentum + 0.32*c.trend_strength + 0.16*c.buying_pressure - 0.28*c.selling_pressure - 0.22*c.market_regime.endswith("down")
    return _obs("market", "Market Intelligence", score, 0.72, [f"Regime: {c.market_regime}", "Combines momentum, trend and pressure."])


def technical_agent(c: AgentContext) -> AgentObservation:
    score = 0.58*c.momentum + 0.42*c.trend_strength - 0.18*c.volatility
    return _obs("technical", "Technical Intelligence", score, 0.70, ["Evaluates directional structure and momentum."])


def flow_agent(c: AgentContext) -> AgentObservation:
    score = 0.65*(c.buying_pressure-c.selling_pressure) + 0.20*c.open_interest_change - 0.20*c.liquidity_stress
    return _obs("flow", "Order-Flow Intelligence", score, 0.68, ["Evaluates buying/selling pressure, open interest and liquidity."])


def news_agent(c: AgentContext) -> AgentObservation:
    score = 0.70*c.news_sentiment*c.news_credibility - 0.65*c.news_risk
    quality = max(0.35, c.news_credibility)
    return _obs("news", "News Intelligence", score, quality, ["Weights external information by relevance/risk and source credibility."], quality)


def sentiment_agent(c: AgentContext) -> AgentObservation:
    score = 0.70*c.news_sentiment - 0.20*c.news_risk
    return _obs("sentiment", "Sentiment Intelligence", score, 0.58, ["Separates broad sentiment from raw headlines."])


def regime_agent(c: AgentContext) -> AgentObservation:
    mapping = {"trending_up": 0.75, "trending_down": -0.75, "range": 0.0, "high_volatility": -0.18, "low_volatility": 0.05, "unknown": 0.0}
    score = mapping.get(c.market_regime, 0.0)
    return _obs("regime", "Market Regime Intelligence", score, 0.74, [f"Detected regime: {c.market_regime}."])


def position_agent(c: AgentContext) -> AgentObservation:
    if not c.position_side:
        return _obs("position", "Position Intelligence", 0.0, 0.35, ["No open position supplied."], action=AgentAction.NEUTRAL)
    side = 1 if c.position_side.lower() == "long" else -1
    score = side * c.thesis_integrity - 0.55*c.liquidity_stress - 0.35*c.news_risk
    return _obs("position", "Position Intelligence", score, 0.76, ["Checks whether the current position still matches its thesis."])


def profit_agent(c: AgentContext) -> AgentObservation:
    if c.unrealized_return > 0.015:
        score = 0.35 - 0.90*c.news_risk - 0.75*c.selling_pressure - 0.55*c.liquidity_stress
        reasons = ["A profitable position is being evaluated for profit protection."]
    else:
        score = 0.05 - 0.50*c.news_risk - 0.35*c.liquidity_stress
        reasons = ["No significant profit cushion is available for protection logic."]
    return _obs("profit", "Profit Protection Intelligence", score, 0.73, reasons)


def adversarial_agent(c: AgentContext) -> AgentObservation:
    # This agent is intentionally skeptical: it scores the evidence against the
    # current directional thesis rather than trying to confirm it.
    if c.position_side:
        side = 1 if c.position_side.lower() == "long" else -1
        contradiction = -side*c.momentum + c.selling_pressure - c.buying_pressure + c.news_risk + c.liquidity_stress
    else:
        contradiction = c.volatility + c.news_risk + c.correlation_risk - abs(c.momentum)
    score = max(-1, min(1, -contradiction))
    action = AgentAction.RISK_OFF if contradiction > 0.75 else (AgentAction.BEARISH if contradiction > 0.4 else AgentAction.NEUTRAL)
    return _obs("adversarial", "Adversarial Intelligence", score, 0.82, ["Actively searches for evidence that the current decision could be wrong."], action=action)


AGENTS: tuple[AgentDefinition, ...] = (
    AgentDefinition("market", "Market Intelligence", 1.0, market_agent),
    AgentDefinition("technical", "Technical Intelligence", 0.95, technical_agent),
    AgentDefinition("flow", "Order-Flow Intelligence", 1.05, flow_agent),
    AgentDefinition("news", "News Intelligence", 1.10, news_agent),
    AgentDefinition("sentiment", "Sentiment Intelligence", 0.65, sentiment_agent),
    AgentDefinition("regime", "Market Regime Intelligence", 0.90, regime_agent),
    AgentDefinition("position", "Position Intelligence", 1.15, position_agent),
    AgentDefinition("profit", "Profit Protection Intelligence", 1.15, profit_agent),
    AgentDefinition("adversarial", "Adversarial Intelligence", 1.30, adversarial_agent),
)


class IntelligenceCouncil:
    """Runs independent specialist agents and produces one auditable council decision."""

    def __init__(self, definitions: tuple[AgentDefinition, ...] = AGENTS):
        self.definitions = definitions

    def deliberate(self, context: AgentContext) -> CouncilDecision:
        observations = [definition.run(context) for definition in self.definitions]
        weights = [d.weight * o.confidence * o.data_quality for d, o in zip(self.definitions, observations)]
        total_weight = sum(weights) or 1.0
        score = sum(o.score*w for o, w in zip(observations, weights)) / total_weight
        confidence = sum(o.confidence*w for o, w in zip(observations, weights)) / total_weight
        avg = sum(o.score for o in observations) / len(observations)
        disagreement = sum(abs(o.score-avg) for o in observations) / (2*len(observations))

        veto_flags: list[str] = []
        adversarial = next(o for o in observations if o.agent_id == "adversarial")
        if adversarial.action == AgentAction.RISK_OFF and context.position_side:
            veto_flags.append("adversarial_risk_off")
        if context.news_risk >= 0.85:
            veto_flags.append("extreme_news_risk")
        if context.liquidity_stress >= 0.90:
            veto_flags.append("extreme_liquidity_stress")

        if veto_flags:
            action = AgentAction.RISK_OFF
        elif score > 0.22:
            action = AgentAction.BULLISH
        elif score < -0.22:
            action = AgentAction.BEARISH
        else:
            action = AgentAction.NEUTRAL

        if disagreement > 0.55:
            confidence *= 0.65

        risk_level = "high" if (context.news_risk + context.liquidity_stress + context.volatility) / 3 > 0.65 else ("medium" if (context.news_risk + context.liquidity_stress + context.volatility) / 3 > 0.35 else "low")
        reasons = [f"{o.name}: {o.action.value} ({o.score:+.2f})" for o in observations]
        if disagreement > 0.45:
            reasons.append("Agent disagreement is elevated; the council is less confident.")
        if veto_flags:
            reasons.append("A safety-critical condition is preventing a normal directional conclusion.")

        return CouncilDecision(symbol=context.symbol, action=action, score=max(-1, min(1, score)), confidence=max(0, min(1, confidence)), disagreement=max(0, min(1, disagreement)), risk_level=risk_level, reasons=reasons, veto_flags=veto_flags, agents=observations, timestamp=datetime.now(timezone.utc))
