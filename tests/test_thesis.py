from ai.thesis import challenge_thesis

def test_long_thesis_deteriorates_under_bearish_evidence():
    r=challenge_thesis(original_direction='long', momentum=-.8, trend_strength=.2, buying_pressure=.1, selling_pressure=.9, news_risk=.9, market_risk=.8)
    assert r.integrity < .5
    assert r.reasons
