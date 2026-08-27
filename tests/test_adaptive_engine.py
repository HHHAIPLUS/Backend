from datetime import datetime, timezone
from ai.adaptive_engine import AdaptivePositionEngine
from ai.adaptive_models import MarketObservation, PositionAction, PositionSnapshot

def make_position(pnl=.04):
    return PositionSnapshot(symbol='BTCUSDT', side='long', entry_price=100, current_price=104, unrealized_return=pnl, peak_return=pnl, opened_at=datetime.now(timezone.utc), confidence=.8)

def test_profitable_trade_can_exit_before_fixed_tp():
    e=AdaptivePositionEngine(); p=make_position(); e.register_position(p,'Bullish momentum')
    m=MarketObservation(momentum=-.8, trend_strength=.25, selling_pressure=.95, buying_pressure=.2, volatility=.75, liquidity_stress=.6, news_risk=.9, market_risk=.85, thesis_integrity=.15)
    d=e.evaluate(p,m)
    assert d.action == PositionAction.EXIT and d.take_profit is None

def test_strong_winner_can_run_without_fixed_tp():
    e=AdaptivePositionEngine(); p=make_position(); e.register_position(p,'Bullish momentum')
    m=MarketObservation(momentum=.85, trend_strength=.9, selling_pressure=.1, buying_pressure=.9, volatility=.2, liquidity_stress=.05, news_risk=.05, market_risk=.1, thesis_integrity=.9)
    d=e.evaluate(p,m)
    assert d.action == PositionAction.HOLD and d.take_profit is None
