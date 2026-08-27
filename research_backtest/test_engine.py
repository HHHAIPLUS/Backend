from research_backtest.engine import Trade, run

def test_costs_and_drawdown():
    r=run([Trade(100,5,2), Trade(-40,2,1)])
    assert r.trades==2 and r.net_pnl==50 and r.wins==1 and r.losses==1
    assert r.win_rate==0.5 and r.max_drawdown==43
