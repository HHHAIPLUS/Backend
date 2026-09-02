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

    def evaluate(self, **kwargs) -> AdvancedDecision:
        result = super().evaluate(**kwargs)
        market_state = kwargs["market_state"]
        predictive = kwargs["predictive"]
        regime = str((market_state.get("regime") or {}).get("label", "unknown")).lower()
        horizon = int(predictive.get("horizon", 6))
        counterfactual = self._empirical_counterfactual(result.action, regime, horizon)
        vetoes = list(result.vetoes)
        selected = counterfactual["selected"]
        if result.execution_candidate and counterfactual["sufficient_evidence"] and float(selected["mean_return"]) <= 0:
            vetoes.append("empirical_counterfactual_negative")
        vetoes = list(dict.fromkeys(vetoes))
        action = "WAIT" if vetoes else result.action
        reasons = list(result.reasons)
        reasons.append(f"Empirical counterfactual: selected={result.action}, samples={selected['samples']}, mean_return={selected['mean_return']:+.5f}, loss_rate={selected['loss_rate']:.3f}.")
        if "empirical_counterfactual_negative" in vetoes:
            reasons.append("Historical matched-action outcomes do not justify the proposed exposure at this regime/horizon.")
        evidence = dict(result.evidence)
        evidence["empirical_counterfactual"] = counterfactual
        return replace(result, action=action, execution_candidate=action in {"LONG", "SHORT"}, vetoes=vetoes, reasons=reasons, evidence=evidence)

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
