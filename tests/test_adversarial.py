from ai.adversarial import AdversarialEngine, ChallengeSeverity

def test_adversary_blocks_when_multiple_extreme_risks_align():
    r=AdversarialEngine().evaluate(symbol='BTCUSDT',proposed_action='long',context={'position_side':'long','momentum':.7,'trend_strength':.7,'buying_pressure':.1,'selling_pressure':.95,'news_risk':.95,'liquidity_stress':.95,'volatility':.9,'news_credibility':.9})
    assert r.should_block
    assert r.challenges

def test_adversary_does_not_invent_a_block_without_contradiction():
    r=AdversarialEngine().evaluate(symbol='BTCUSDT',proposed_action='long',context={'position_side':'long','momentum':.8,'trend_strength':.9,'buying_pressure':.9,'selling_pressure':.1,'news_risk':.05,'liquidity_stress':.05,'volatility':.2,'news_credibility':.9})
    assert not r.should_block
    assert r.robustness > .8
