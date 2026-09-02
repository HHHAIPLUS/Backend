from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Mapping

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

    @classmethod
    def from_market_state(cls, state: Mapping[str, object], **overrides: object) -> "AgentContext":
        """Canonical MarketState adapter; specialists no longer need ad-hoc feeds."""
        features = state.get("features") or {}
        structure = state.get("price_structure") or {}
        flow = state.get("order_flow") or {}
        derivatives = state.get("derivatives") or {}
        liquidity = state.get("liquidity") or {}
        regime = state.get("regime") or {}
        news = state.get("news") or {}
        volatility = state.get("volatility") or {}
        buy = float(flow.get("aggressive_buy_ratio") or 0.5)
        realized = float(volatility.get("realized") or features.get("volatility_proxy") or 0.0)
        return cls(
            symbol=str(state.get("symbol", "UNKNOWN")),
            momentum=max(-1.0, min(1.0, float(features.get("momentum", structure.get("last_return", 0.0))))),
            trend_strength=max(0.0, min(1.0, abs(float(features.get("trend_strength", structure.get("trend", 0.0)))))),
            buying_pressure=max(0.0, min(1.0, buy)),
            selling_pressure=max(0.0, min(1.0, 1.0 - buy)),
            volatility=max(0.0, min(1.0, realized / 0.02)),
            liquidity_stress=max(0.0, min(1.0, float(features.get("liquidity_stress", (liquidity.get("spread_bps") or 0.0) / 50.0)))),
            news_risk=max(0.0, min(1.0, float(news.get("risk", 0.0) or 0.0))),
            news_sentiment=max(-1.0, min(1.0, float(news.get("sentiment", 0.0) or 0.0))),
            news_credibility=max(0.0, min(1.0, float(news.get("credibility", 0.0) or 0.0))),
            funding_bias=max(-1.0, min(1.0, float(derivatives.get("funding_rate", 0.0) or 0.0) * 100.0)),
            open_interest_change=max(-1.0, min(1.0, float(derivatives.get("open_interest_change", 0.0) or 0.0))),
            correlation_risk=max(0.0, min(1.0, float(regime.get("market_risk", 0.0) if isinstance(regime, dict) else 0.0))),
            market_regime=str(regime.get("label", "unknown")).lower() if isinstance(regime, dict) else "unknown",
            **overrides,
        )


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
        action = AgentAction.BULLISH if score > 0.18 else AgentAction.BEARISH if score < -0.18 else AgentAction.NEUTRAL
    return AgentObservation(agent_id=agent_id, name=name, action=action, score=max(-1, min(1, score)), confidence=max(0, min(1, confidence)), reasons=reasons, data_quality=quality, timestamp=datetime.now(timezone.utc))


def market_agent(c: AgentContext) -> AgentObservation:
    return _obs("market", "Market Intelligence", 0.42*c.momentum + 0.32*c.trend_strength + 0.16*c.buying_pressure - 0.28*c.selling_pressure - 0.22*c.market_regime.endswith("down"), 0.72, [f"Regime: {c.market_regime}", "Canonical market-state structure and pressure."])

def technical_agent(c: AgentContext) -> AgentObservation:
    return _obs("technical", "Technical Intelligence", 0.58*c.momentum + 0.42*c.trend_strength - 0.18*c.volatility, 0.70, ["Canonical price structure, momentum and volatility."])

def flow_agent(c: AgentContext) -> AgentObservation:
    return _obs("flow", "Order-Flow Intelligence", 0.65*(c.buying_pressure-c.selling_pressure) + 0.20*c.open_interest_change - 0.20*c.liquidity_stress, 0.68, ["Canonical flow, OI and liquidity context."])

def news_agent(c: AgentContext) -> AgentObservation:
    score = 0.70*c.news_sentiment*c.news_credibility - 0.65*c.news_risk
    return _obs("news", "News Intelligence", score, max(0.35, c.news_credibility), ["Canonical timestamped news risk, sentiment and credibility."], max(0.35, c.news_credibility))

def sentiment_agent(c: AgentContext) -> AgentObservation:
    return _obs("sentiment", "Sentiment Intelligence", 0.70*c.news_sentiment - 0.20*c.news_risk, 0.58, ["Canonical external sentiment separated from raw headlines."])

def regime_agent(c: AgentContext) -> AgentObservation:
    mapping = {"trending_up": 0.75, "trending_down": -0.75, "range": 0.0, "high_volatility": -0.18, "low_volatility": 0.05, "unknown": 0.0}
    return _obs("regime", "Market Regime Intelligence", mapping.get(c.market_regime, 0.0), 0.74, [f"Canonical regime: {c.market_regime}."])

def position_agent(c: AgentContext) -> AgentObservation:
    if not c.position_side:
        return _obs("position", "Position Intelligence", 0.0, 0.35, ["No open position supplied."], action=AgentAction.NEUTRAL)
    side = 1 if c.position_side.lower() == "long" else -1
    return _obs("position", "Position Intelligence", side*c.thesis_integrity - 0.55*c.liquidity_stress - 0.35*c.news_risk, 0.76, ["Checks thesis integrity against canonical market evidence."])

def profit_agent(c: AgentContext) -> AgentObservation:
    if c.unrealized_return > 0.015:
        return _obs("profit", "Profit Protection Intelligence", 0.35 - 0.90*c.news_risk - 0.75*c.selling_pressure - 0.55*c.liquidity_stress, 0.73, ["Evaluates profitable-position deterioration from canonical evidence."])
    return _obs("profit", "Profit Protection Intelligence", 0.05 - 0.50*c.news_risk - 0.35*c.liquidity_stress, 0.73, ["No significant profit cushion is available for protection logic."])

def adversarial_agent(c: AgentContext) -> AgentObservation:
    if c.position_side:
        side = 1 if c.position_side.lower() == "long" else -1
        contradiction = -side*c.momentum + c.selling_pressure - c.buying_pressure + c.news_risk + c.liquidity_stress
    else:
        contradiction = c.volatility + c.news_risk + c.correlation_risk - abs(c.momentum)
    score = max(-1, min(1, -contradiction))
    action = AgentAction.RISK_OFF if contradiction > 0.75 else AgentAction.BEARISH if contradiction > 0.4 else AgentAction.NEUTRAL
    return _obs("adversarial", "Adversarial Intelligence", score, 0.82, ["Searches canonical evidence for thesis failure."], action=action)


AGENTS: tuple[AgentDefinition, ...] = tuple(
    AgentDefinition(agent_id, name, 1.0, run)
    for agent_id, name, run in (
        ("market", "Market Intelligence", market_agent), ("technical", "Technical Intelligence", technical_agent),
        ("flow", "Order-Flow Intelligence", flow_agent), ("news", "News Intelligence", news_agent),
        ("sentiment", "Sentiment Intelligence", sentiment_agent), ("regime", "Market Regime Intelligence", regime_agent),
        ("position", "Position Intelligence", position_agent), ("profit", "Profit Protection Intelligence", profit_agent),
        ("adversarial", "Adversarial Intelligence", adversarial_agent),
    )
)


class IntelligenceCouncil:
    """Specialists consume one canonical state; weights may come only from validated evidence."""

    def __init__(self, definitions: tuple[AgentDefinition, ...] = AGENTS):
        self.definitions = definitions

    def deliberate(self, context: AgentContext, learned_weights: Mapping[str, float] | None = None) -> CouncilDecision:
        observations = [definition.run(context) for definition in self.definitions]
        learned = learned_weights or {}
        weights = [max(0.05, float(learned.get(d.agent_id, d.weight))) * o.confidence * o.data_quality for d, o in zip(self.definitions, observations)]
        total_weight = sum(weights) or 1.0
        score = sum(o.score*w for o, w in zip(observations, weights)) / total_weight
        confidence = sum(o.confidence*w for o, w in zip(observations, weights)) / total_weight
        avg = sum(o.score for o in observations) / len(observations)
        disagreement = sum(abs(o.score-avg) for o in observations) / (2*len(observations))
        veto_flags: list[str] = []
        adversarial = next(o for o in observations if o.agent_id == "adversarial")
        if adversarial.action == AgentAction.RISK_OFF and context.position_side: veto_flags.append("adversarial_risk_off")
        if context.news_risk >= 0.85: veto_flags.append("extreme_news_risk")
        if context.liquidity_stress >= 0.90: veto_flags.append("extreme_liquidity_stress")
        action = AgentAction.RISK_OFF if veto_flags else AgentAction.BULLISH if score > 0.22 else AgentAction.BEARISH if score < -0.22 else AgentAction.NEUTRAL
        if disagreement > 0.55: confidence *= 0.65
        risk_avg = (context.news_risk + context.liquidity_stress + context.volatility) / 3
        risk_level = "high" if risk_avg > 0.65 else "medium" if risk_avg > 0.35 else "low"
        reasons = [f"{o.name}: {o.action.value} ({o.score:+.2f})" for o in observations]
        if learned: reasons.append("Specialist weights are evidence-learned where sufficient agent-level outcomes exist; otherwise neutral weight is used.")
        if disagreement > 0.45: reasons.append("Agent disagreement is elevated; confidence is reduced.")
        if veto_flags: reasons.append("A safety-critical condition prevents a normal directional conclusion.")
        return CouncilDecision(symbol=context.symbol, action=action, score=max(-1, min(1, score)), confidence=max(0, min(1, confidence)), disagreement=max(0, min(1, disagreement)), risk_level=risk_level, reasons=reasons, veto_flags=veto_flags, agents=observations, timestamp=datetime.now(timezone.utc))
