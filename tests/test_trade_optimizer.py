import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ai.trade_optimizer import TradeCandidate,MarketRegime,TradeOptimizer
def c(**o):
 d=dict(symbol='BTCUSDT',side='long',entry=100,target=110,invalidation=95,probability_of_success=.65,regime_fit=.8,confirmation=.8,timing_quality=.8,liquidity_score=.9,news_risk=.1); d.update(o); return TradeCandidate(**d)
def test_ev(): assert TradeOptimizer().evaluate(c(),MarketRegime('trend',liquidity=.9))['expected_value']>0
def test_rr(): assert TradeOptimizer().evaluate(c(target=101),MarketRegime('range',liquidity=.9))['decision']=='wait'
def test_timing(): assert TradeOptimizer().evaluate(c(timing_quality=.2),MarketRegime('trend',liquidity=.9))['decision']=='wait'
def test_news(): assert TradeOptimizer().evaluate(c(news_risk=.95),MarketRegime('trend',liquidity=.9))['decision']=='wait'
def test_authority(): assert TradeOptimizer().evaluate(c(),MarketRegime('trend',liquidity=.9))['execution_authority'] is False
