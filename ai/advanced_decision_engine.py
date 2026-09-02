from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import log
from statistics import mean
from typing import Any, Iterable

from ai.adversarial import AdversarialEngine
from ai.agents import AgentContext, IntelligenceCouncil
from ai.counterfactual import CounterfactualTradeTwin
from app.ml.adaptive_intelligence import AdaptiveIntelligence, AdaptiveObservation


@dataclass(frozen=True)
class DecisionQuality:
    samples: int
    directional_accuracy: float
    mean_return: float
    abstention_rate: float
    confidence_error: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class AdvancedDecision:
    symbol: str
    action: str
    confidence: float
    calibrated_confidence: float
    expected_value: float
    scenario_uncertainty: float
    disagreement: float
    execution_candidate: bool
    vetoes: list[str]
    contradictions: list[str]
    reasons: list[str]
    evidence: dict[str, Any]
    what_would_change: list[str]
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdvancedDecisionEngine:
    """Stage-5 decision layer: evidence -> prediction -> uncertainty -> action.

    It has no order/execution authority. Risk vetoes are absolute and the
    engine never mutates predictive artifacts or promotes candidates.
    """

    MIN_SCENARIO_SAMPLES = 30
    MAX_ABSTENTION = 0.85

    def __init__(self, adaptive: AdaptiveIntelligence | None = None):
        self.adaptive = adaptive or AdaptiveIntelligence()
        self.council = IntelligenceCouncil()
        self.adversarial = AdversarialEngine()
        self.counterfactual = CounterfactualTradeTwin()
        self._quality: list[tuple[str, str, float, float, bool]] = []

    @staticmethod
    def context_from_market_state(state: dict[str, Any], *, position_side: str | None = None, unrealized_return: float = 0.0, thesis_integrity: float = 0.5) -> AgentContext:
        features = state.get("features") or {}
        structure = state.get("price_structure") or {}
        flow = state.get("order_flow") or {}
        derivatives = state.get("derivatives") or {}
        liquidity = state.get("liquidity") or {}
        regime = state.get("regime") or {}
        news = state.get("news") or {}
        realized = float((state.get("volatility") or {}).get("realized") or features.get("volatility_proxy") or 0.0)
        return AgentContext(
            symbol=str(state.get("symbol", "UNKNOWN")),
            momentum=max(-1.0, min(1.0, float(features.get("momentum", (state.get("timeframes") or {}).get("5m", {}).get("return") or 0.0)))),
            trend_strength=max(0.0, min(1.0, abs(float(features.get("trend_strength", structure.get("trend", 0.0)))))),
            buying_pressure=max(0.0, min(1.0, float(flow.get("aggressive_buy_ratio", 0.5) or 0.5))),
            selling_pressure=max(0.0, min(1.0, 1.0 - float(flow.get("aggressive_buy_ratio", 0.5) or 0.5))),
            volatility=max(0.0, min(1.0, realized / 0.02)),
            liquidity_stress=max(0.0, min(1.0, float(features.get("liquidity_stress", (liquidity.get("spread_bps") or 0.0) / 50.0)))),
            news_risk=max(0.0, min(1.0, float(news.get("risk", features.get("news_risk", 0.0)) or 0.0))),
            news_sentiment=max(-1.0, min(1.0, float(news.get("sentiment", features.get("news_sentiment", 0.0)) or 0.0))),
            news_credibility=max(0.0, min(1.0, float(news.get("credibility", 0.0) or 0.0))),
            funding_bias=max(-1.0, min(1.0, float(derivatives.get("funding_rate", features.get("funding_rate", 0.0)) or 0.0) * 100.0)),
            open_interest_change=max(-1.0, min(1.0, float(derivatives.get("open_interest_change", features.get("open_interest_change", 0.0)) or 0.0))),
            correlation_risk=max(0.0, min(1.0, float((regime.get("market_risk", 0.0) if isinstance(regime, dict) else 0.0)))),
            market_regime=str(regime.get("label", "unknown")).lower() if isinstance(regime, dict) else "unknown",
            position_side=position_side,
            unrealized_return=float(unrealized_return),
            thesis_integrity=max(0.0, min(1.0, thesis_integrity)),
        )

    @staticmethod
    def _entropy(probabilities: list[float]) -> float:
        return -sum(p * log(p) for p in probabilities if p > 0) / log(max(2, len(probabilities)))

    def learned_scenarios(self, *, regime: str, horizon: int, observations: Iterable[AdaptiveObservation]) -> dict[str, Any]:
        rows = [o for o in observations if o.regime == regime and int(o.horizon) == int(horizon)]
        if len(rows) < self.MIN_SCENARIO_SAMPLES:
            rows = [o for o in observations if int(o.horizon) == int(horizon)]
        labels = ["bullish_continuation", "bearish_continuation", "range_reversion", "volatility_expansion", "disorderly_risk_off"]
        counts = {k: 1.0 for k in labels}
        moves = {k: [] for k in labels}
        for row in rows:
            action = row.action.lower()
            ret = float(row.realized_return)
            if "long" in action or "bull" in action:
                label = "bullish_continuation" if ret > 0 else "range_reversion"
            elif "short" in action or "bear" in action:
                label = "bearish_continuation" if ret < 0 else "range_reversion"
            elif "risk" in action or "exit" in action:
                label = "disorderly_risk_off" if ret < -0.01 else "volatility_expansion"
            else:
                label = "range_reversion"
            counts[label] += 1.0
            moves[label].append(ret)
        total = sum(counts.values())
        probabilities = {k: v / total for k, v in counts.items()}
        expected_moves = {k: max(-0.20, min(0.20, mean(v))) if v else 0.0 for k, v in moves.items()}
        expected_value = sum(probabilities[k] * expected_moves[k] for k in labels)
        return {"samples": len(rows), "probabilities": probabilities, "expected_moves": expected_moves, "expected_value": expected_value, "uncertainty": self._entropy(list(probabilities.values()))}

    @staticmethod
    def _calibrate_confidence(raw: float, rows: list[AdaptiveObservation]) -> float:
        if not rows:
            return max(0.0, min(1.0, raw))
        bins = []
        for lo in [i / 10 for i in range(10)]:
            group = [r for r in rows if lo <= r.confidence < lo + 0.1]
            if group:
                observed = mean(r.realized_return > 0 for r in group)
                predicted = mean(r.confidence for r in group)
                bins.append((predicted, observed, len(group)))
        if not bins:
            return raw
        nearest = min(bins, key=lambda x: abs(x[0] - raw))
        shrink = min(1.0, nearest[2] / 50.0)
        return max(0.0, min(1.0, raw * (1 - shrink) + nearest[1] * shrink))

    def decision_quality(self) -> DecisionQuality:
        if not self._quality:
            return DecisionQuality(0, 0.0, 0.0, 0.0, 0.0)
        rows = self._quality
        actionable = [r for r in rows if not r[4]]
        correct = sum((r[1] == "LONG" and r[2] > 0) or (r[1] == "SHORT" and r[2] < 0) for r in actionable)
        accuracy = correct / len(actionable) if actionable else 0.0
        error = mean(abs(r[3] - (1.0 if ((r[1] == "LONG" and r[2] > 0) or (r[1] == "SHORT" and r[2] < 0)) else 0.0)) for r in actionable) if actionable else 0.0
        return DecisionQuality(len(rows), accuracy, mean(r[2] for r in rows), sum(r[4] for r in rows) / len(rows), error)

    def record_outcome(self, *, decision: AdvancedDecision, realized_return: float) -> None:
        self._quality.append((decision.symbol, decision.action, float(realized_return), decision.calibrated_confidence, not decision.execution_candidate))
        if len(self._quality) > 10000:
            del self._quality[:-10000]

    def evaluate(self, *, market_state: dict[str, Any], predictive: dict[str, Any], risk_vetoes: list[str] | None = None, position_side: str | None = None, unrealized_return: float = 0.0, thesis_integrity: float = 0.5) -> AdvancedDecision:
        context = self.context_from_market_state(market_state, position_side=position_side, unrealized_return=unrealized_return, thesis_integrity=thesis_integrity)
        council = self.council.deliberate(context)
        obs = list(self.adaptive.observations)
        scenarios = self.learned_scenarios(regime=context.market_regime, horizon=int(predictive.get("horizon", 6)), observations=obs)
        p = predictive.get("probabilities") or {}
        long_p, short_p = float(p.get("long", 0.0)), float(p.get("short", 0.0))
        predictive_action = "LONG" if long_p > short_p else "SHORT" if short_p > long_p else "WAIT"
        raw_conf = max(long_p, short_p)
        model_version = str(predictive.get("model_version", "unknown"))
        cal_rows = [o for o in obs if o.model_version == model_version and o.regime == context.market_regime]
        calibrated = self._calibrate_confidence(raw_conf, cal_rows)
        adapted = self.adaptive.adapt_confidence(calibrated, model_version=model_version, regime=context.market_regime, features=market_state.get("features") or {})
        calibrated = float(adapted["adjusted_confidence"])
        uncertainty = max(float(scenarios["uncertainty"]), float(predictive.get("uncertainty", 0.0)), float(council.disagreement))
        proposed = predictive_action.lower()
        adversarial = self.adversarial.evaluate(symbol=context.symbol, proposed_action=proposed, context=context.model_dump())
        contradictions: list[str] = []
        if predictive_action == "LONG" and council.action.value == "bearish": contradictions.append("predictive_model_vs_council")
        if predictive_action == "SHORT" and council.action.value == "bullish": contradictions.append("predictive_model_vs_council")
        dominant = max(scenarios["probabilities"], key=scenarios["probabilities"].get)
        if predictive_action == "LONG" and dominant == "bearish_continuation": contradictions.append("predictive_model_vs_learned_scenario")
        if predictive_action == "SHORT" and dominant == "bullish_continuation": contradictions.append("predictive_model_vs_learned_scenario")
        if context.news_risk >= 0.75: contradictions.append("elevated_news_risk")
        vetoes = list(risk_vetoes or []) + list(council.veto_flags)
        if not market_state.get("usable", True) or float(market_state.get("data_quality", 1.0)) < 0.80: vetoes.append("insufficient_data_quality")
        if adversarial.should_block: vetoes.append("adversarial_block")
        if uncertainty >= 0.78: vetoes.append("high_decision_uncertainty")
        abstention = min(self.MAX_ABSTENTION, max(0.50, self.adaptive.learned_abstention_threshold()))
        if calibrated < abstention: vetoes.append("adaptive_abstention")
        if contradictions and len(contradictions) >= 2: vetoes.append("material_contradiction")
        vetoes = list(dict.fromkeys(vetoes))
        if vetoes:
            action = "WAIT"
            execution = False
            confidence = calibrated
            reason = "The final gate abstained because independent evidence, uncertainty, or a safety veto was not acceptable."
        else:
            action = predictive_action
            execution = action in {"LONG", "SHORT"}
            confidence = calibrated
            reason = "Validated predictive direction survived calibrated fusion, scenario analysis, adversarial challenge, and the final abstention gate."
        reasons = [reason, f"Council={council.action.value} score={council.score:+.3f} disagreement={council.disagreement:.3f}", f"Predictive={predictive_action} probabilities long={long_p:.3f} short={short_p:.3f}", f"Learned scenario={dominant} EV={scenarios['expected_value']:+.5f}", f"Adversarial robustness={adversarial.robustness:.3f}"]
        reasons.extend(f"Contradiction: {x}" for x in contradictions)
        return AdvancedDecision(
            symbol=context.symbol,
            action=action,
            confidence=confidence,
            calibrated_confidence=calibrated,
            expected_value=float(scenarios["expected_value"]),
            scenario_uncertainty=uncertainty,
            disagreement=council.disagreement,
            execution_candidate=execution,
            vetoes=vetoes,
            contradictions=contradictions,
            reasons=reasons,
            evidence={"council": council.model_dump(mode="json"), "scenarios": scenarios, "adversarial": adversarial.model_dump(mode="json"), "adaptive": adapted, "predictive": predictive, "decision_quality": self.decision_quality().as_dict()},
            what_would_change=["A material change in calibrated predictive probability", "A regime change supported by new market-state evidence", "Scenario probabilities shifting materially", "Adversarial contradictions resolving or strengthening", "Data quality returning above the safety threshold"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
