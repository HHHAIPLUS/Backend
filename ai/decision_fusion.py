from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class FinalDecision:
    action:str
    confidence:float
    reason:str
    execution_candidate:bool
    vetoes:list[str]

class DecisionFusion:
    """Connects existing intelligence layers into one fail-closed decision.

    This is orchestration, not a new prediction source. No layer can bypass risk.
    """
    def decide(self, *, council_action:str, council_confidence:float, disagreement:float,
               predictive:dict, adversarial_block:bool, scenario_uncertainty:float,
               data_quality:float, risk_vetoes:list[str]|None=None) -> FinalDecision:
        vetoes=list(risk_vetoes or [])
        if data_quality < .8: vetoes.append('insufficient_data_quality')
        if adversarial_block: vetoes.append('adversarial_block')
        if scenario_uncertainty >= .78: vetoes.append('high_scenario_uncertainty')
        if predictive.get('abstain',True): vetoes.append('no_validated_predictive_model')
        if disagreement >= .60: vetoes.append('high_agent_disagreement')
        if vetoes:
            return FinalDecision('WAIT',max(0.0,min(1.0,1-scenario_uncertainty)), 'One or more independent gates vetoed a directional decision.',False,vetoes)
        p=predictive['probabilities']; long_p=p.get('long',0); short_p=p.get('short',0)
        if council_action=='bullish' and long_p>=.55:
            return FinalDecision('LONG',min(council_confidence,long_p), 'Council and validated predictive model agree on the bullish direction.',True,[])
        if council_action=='bearish' and short_p>=.55:
            return FinalDecision('SHORT',min(council_confidence,short_p), 'Council and validated predictive model agree on the bearish direction.',True,[])
        return FinalDecision('WAIT',min(council_confidence,max(long_p,short_p)), 'The independent intelligence layers do not agree strongly enough to justify exposure.',False,['cross_layer_disagreement'])
