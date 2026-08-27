from ai.cognitive_exit import CognitiveExitEngine, ExitAction, PositionTelemetry
from ai.counterfactual import CounterfactualTradeTwin

def base(**kwargs):
    data=dict(side="long",entry_price=100,current_price=103,unrealized_return=.03,peak_return=.04,minutes_open=20,momentum=.15,trend_strength=.8,buying_pressure=.72,selling_pressure=.28,volatility=.15,liquidity_stress=.1,news_risk=.1,thesis_integrity=.85)
    data.update(kwargs)
    return PositionTelemetry(**data)

def test_profitable_reversal_exits_or_reduces():
    d=CognitiveExitEngine().evaluate(base(momentum=-.8,selling_pressure=.85,trend_strength=.25,news_risk=.35))
    assert d.action in {ExitAction.REDUCE, ExitAction.EXIT, ExitAction.PROTECT}
    assert d.profit_lock_score > .4

def test_profitable_trade_can_hold():
    d=CognitiveExitEngine().evaluate(base(current_price=103.5,unrealized_return=.035,peak_return=.036,momentum=.65,trend_strength=.9,buying_pressure=.9,selling_pressure=.1,news_risk=.05,thesis_integrity=.95))
    assert d.action in {ExitAction.HOLD, ExitAction.PROTECT}

def test_extreme_stress_is_emergency():
    d=CognitiveExitEngine().evaluate(base(momentum=-.9,liquidity_stress=.98,news_risk=.95))
    assert d.action == ExitAction.EMERGENCY_EXIT

def test_counterfactual_is_transparent():
    p=base(momentum=-.7,selling_pressure=.85,trend_strength=.2)
    d=CognitiveExitEngine().evaluate(p)
    twin=CounterfactualTradeTwin().evaluate(p,d)
    assert twin.selected is not None
    assert isinstance(twin.explanation,str) and 'utilities' in twin.explanation
