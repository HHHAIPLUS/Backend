from ai.decision_fusion import DecisionFusion

def test_fusion_abstains_without_validated_model():
    r=DecisionFusion().decide(council_action='bullish',council_confidence=.9,disagreement=.1,predictive={'abstain':True,'probabilities':{}},adversarial_block=False,scenario_uncertainty=.2,data_quality=1)
    assert r.action=='WAIT' and 'no_validated_predictive_model' in r.vetoes

def test_fusion_requires_cross_layer_agreement():
    r=DecisionFusion().decide(council_action='bullish',council_confidence=.9,disagreement=.1,predictive={'abstain':False,'probabilities':{'long':.8,'short':.1,'flat':.1}},adversarial_block=False,scenario_uncertainty=.2,data_quality=1)
    assert r.action=='LONG' and r.execution_candidate
