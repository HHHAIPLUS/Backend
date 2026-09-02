from __future__ import annotations

from collections.abc import Mapping

from ai.agents import AgentContext, CouncilDecision, IntelligenceCouncil
from ai.advanced_decision_engine import AdvancedDecisionEngine
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

    def __init__(self, adaptive: AdaptiveIntelligence | None = None):
        super().__init__(adaptive=adaptive)
        self.council = EvidenceWeightedCouncil(self.adaptive)

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
