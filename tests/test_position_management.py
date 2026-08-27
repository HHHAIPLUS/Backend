from datetime import datetime, timezone
from ai.adaptive_engine import AdaptivePositionEngine
from ai.adaptive_models import MarketObservation, PositionSnapshot
from ai.position_management import ManagementAction, PositionManager


def pos(pnl=0.04, price=104):
    return PositionSnapshot(symbol='BTCUSDT', side='long', entry_price=100, current_price=price, unrealized_return=pnl, peak_return=pnl, opened_at=datetime.now(timezone.utc), confidence=.8)


def market(**overrides):
    data=dict(momentum=.7, trend_strength=.8, selling_pressure=.1, buying_pressure=.8, volatility=.2, liquidity_stress=.1, news_risk=.1, market_risk=.1, thesis_integrity=.9)
    data.update(overrides)
    return MarketObservation(**data)


def test_profitable_trade_gets_dynamic_protection_without_fixed_tp():
    ae=AdaptivePositionEngine(); ae.register_position(pos(), 'bullish momentum')
    d=ae.evaluate(pos(), market(selling_pressure=.9, news_risk=.85, thesis_integrity=.25))
    pm=PositionManager(); pm.register(pos(), 1)
    md=pm.review(pos(), market(selling_pressure=.9, news_risk=.85, thesis_integrity=.25), d)
    assert md.action == ManagementAction.TRAIL_PROFIT
    assert md.protection_price is not None
    assert md.close_fraction == 0


def test_strong_winner_can_run():
    ae=AdaptivePositionEngine(); ae.register_position(pos(), 'bullish momentum')
    d=ae.evaluate(pos(), market())
    pm=PositionManager(); pm.register(pos(), 1)
    md=pm.review(pos(), market(), d)
    assert md.action == ManagementAction.HOLD
    assert md.execution_allowed is False


def test_extreme_conditions_trigger_emergency_exit():
    ae=AdaptivePositionEngine(); ae.register_position(pos(), 'bullish momentum')
    m=market(liquidity_stress=.99, volatility=.95)
    d=ae.evaluate(pos(), m)
    pm=PositionManager(); pm.register(pos(), 1)
    md=pm.review(pos(), m, d)
    assert md.action == ManagementAction.EMERGENCY_EXIT
    assert md.close_fraction == 1
    assert md.execution_allowed is False


def test_protection_never_moves_backward_for_long():
    pm=PositionManager(); pm.register(pos(), 1)
    ae=AdaptivePositionEngine(); ae.register_position(pos(), 'bullish')
    m=market(selling_pressure=.9, news_risk=.85, thesis_integrity=.25)
    d=ae.evaluate(pos(),m); first=pm.review(pos(),m,d).protection_price
    m2=market(selling_pressure=.5, news_risk=.5, thesis_integrity=.55)
    d2=ae.evaluate(pos(price=106,pnl=.06),m2); second=pm.review(pos(price=106,pnl=.06),m2,d2).protection_price
    assert second is None or second >= first
