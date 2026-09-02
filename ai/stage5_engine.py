from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from statistics import mean

from ai.agents import AgentContext, CouncilDecision, IntelligenceCouncil
from ai.advanced_decision_engine import AdvancedDecision, AdvancedDecisionEngine
from app.ml.adaptive_intelligence import AdaptiveIntelligence


class EvidenceWeightedCouncil(IntelligenceCouncil):
    """Council whose specialist weights are learned from agent-level outcomes."""

    def __init__(self, adaptive: AdaptiveIntelligence):
        super().__init__()
        self.adaptive = adaptive

    def deliberate(self, context: AgentContext, learned_weights: Mapping[str, float] | None = None) -> CouncilDecision:
        weights = self.adaptive.learned_model_weights()
        agent_weights = {k.removeprefix("agent:"): v for k, v in weights.items() if k.startswith("agent:")}
        return super().deliberate(context, agent_weights or None)


class Stage5DecisionEngine(AdvancedDecisionEngine):
    """Production Stage-5 orchestrator with evidence-weighted specialists."""

    VERSION = "stage5-advanced-decision-v1"
    MIN_COUNTERFACTUAL_SAMPLES = 30
    FUSION_MARGIN = 0.10

    def __init__(self, adaptive: AdaptiveIntelligence | None = None):
        super().__init__(adaptive=adaptive)
        self.council = EvidenceWeightedCouncil(self.adaptive)

    def _empirical_counterfactual(self, action: str, regime: str, horizon: int) -> dict[str, object]:
        rows = [o for o in self.adaptive.observations if int(o.horizon) == int(horizon) and o.regime == regime]
        if len(rows) < self.MIN_COUNTERFACTUAL_SAMPLES:
            rows = [o for o in self.adaptive.observations if int(o.horizon) == int(horizon)]
        candidates = {"LONG": [], "SHORT": [], "WAIT": []}
        for row in rows:
            a = row.action.upper()
            key = "LONG" if "LONG" in a or "BULL" in a else "SHORT" if "SHORT" in a or "BEAR" in a else "WAIT"
            candidates[key].append(float(row.realized_return))
        stats = {k: {"samples": len(v), "mean_return": mean(v) if v else 0.0, "loss_rate": sum(x < 0 for x in v) / len(v) if v else 0.0} for k, v in candidates.items()}
        selected = stats.get(action, stats["WAIT"])
        return {"regime": regime, "horizon": horizon, "selected_action": action, "selected": selected, "actions": stats, "sufficient_evidence": selected["samples"] >= self.MIN_COUNTERFACTUAL_SAMPLES}

    @staticmethod
    def _fuse(result: AdvancedDecision, predictive: dict, scenario: dict) -> dict[str, float | str]:
        probabilities = predictive.get("probabilities") or {}
        lp, sp = float(probabilities.get("long", 0.0)), float(probabilities.get("short", 0.0))
        council = result.evidence.get("council") or {}
        cp = (float(council.get("score", 0.0)) + 1.0) / 2.0
        scenarios = scenario.get("probabilities") or {}
        bp, bearp = float(scenarios.get("bullish_continuation", 0.0)), float(scenarios.get("bearish_continuation", 0.0))
        fused_long = 0.55 * lp + 0.25 * cp + 0.20 * bp
        fused_short = 0.55 * sp + 0.25 * (1.0 - cp) + 0.20 * bearp
        action = "LONG" if fused_long > fused_short else "SHORT" if fused_short > fused_long else "WAIT"
        confidence = max(fused_long, fused_short)
        return {"long": fused_long, "short": fused_short, "margin": abs(fused_long - fused_short), "action": action, "confidence": confidence, "components": {"predictive": {"long": lp, "short": sp}, "council": {"long": cp, "short": 1.0 - cp}, "scenario": {"long": bp, "short": bearp}}}

    def evaluate(self, **kwargs) -> AdvancedDecision:
        result = super().evaluate(**kwargs)
        market_state = kwargs["market_state"]
        predictive = kwargs["predictive"]
        scenarios = result.evidence.get("scenarios") or {}
        fusion = self._fuse(result, predictive, scenarios)
        vetoes = list(result.vetoes)
        if result.execution_candidate and (fusion["action"] == "WAIT" or fusion["margin"] < self.FUSION_MARGIN):
            vetoes.append("low_fusion_margin")
        elif result.execution_candidate and fusion["action"] != result.action:
            vetoes.append("fusion_direction_conflict")
        regime = str((market_state.get("regime") or {}).get("label", "unknown")).lower()
        horizon = int(predictive.get("horizon", 6))
        counterfactual = self._empirical_counterfactual(result.action, regime, horizon)
        selected = counterfactual["selected"]
        if result.execution_candidate and counterfactual["sufficient_evidence"] and float(selected["mean_return"]) <= 0:
            vetoes.append("empirical_counterfactual_negative")
        vetoes = list(dict.fromkeys(vetoes))
        action = "WAIT" if vetoes else result.action
        reasons = list(result.reasons)
        reasons.append(f"Calibrated fusion: {fusion['action']} long={fusion['long']:.3f} short={fusion['short']:.3f} margin={fusion['margin']:.3f}.")
        reasons.append(f"Empirical counterfactual: selected={result.action}, samples={selected['samples']}, mean_return={selected['mean_return']:+.5f}, loss_rate={selected['loss_rate']:.3f}.")
        if "low_fusion_margin" in vetoes: reasons.append("Independent predictive, council and learned-scenario evidence is not separated enough to justify directional exposure.")
        if "fusion_direction_conflict" in vetoes: reasons.append("The calibrated fusion layer disagrees with the proposed direction.")
        if "empirical_counterfactual_negative" in vetoes: reasons.append("Historical matched-action outcomes do not justify the proposed exposure at this regime/horizon.")
        evidence = dict(result.evidence)
        evidence["calibrated_fusion"] = fusion
        evidence["empirical_counterfactual"] = counterfactual
        return replace(result, action=action, confidence=float(fusion["confidence"]), calibrated_confidence=float(fusion["confidence"]), execution_candidate=action in {"LONG", "SHORT"}, vetoes=vetoes, reasons=reasons, evidence=evidence)

    def status(self) -> dict[str, object]:
        return {
            "engine": self.VERSION,
            "decision_only": True,
            "execution_authority": False,
            "risk_vetoes_absolute": True,
            "canonical_market_state": True,
            "learned_specialist_weights": True,
            "calibrated_fusion": True,
            "learned_scenarios": True,
            "expected_value_scenarios": True,
            "adversarial_thesis_challenge": True,
            "contradiction_detection": True,
            "empirical_counterfactual_support": True,
            "explicit_abstention": True,
            "auditable_reasons": True,
            "decision_quality_separate_from_pnl": True,
            "production_model_mutation": False,
        }
