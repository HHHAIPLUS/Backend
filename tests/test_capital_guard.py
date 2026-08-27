from ai.capital_guard import CapitalGuard, RiskSnapshot, KillSwitch

def s(**x):
 d=dict(equity=10000,free_margin=8000,daily_pnl_pct=0,drawdown_pct=1,proposed_risk_pct=.25,leverage=2,open_positions=1,expected_slippage_bps=2,data_fresh=True,exchange_healthy=True);d.update(x);return RiskSnapshot(**d)
def test_allow(): assert CapitalGuard().evaluate(s())['decision']=='allow'
def test_daily_loss(): assert CapitalGuard().evaluate(s(daily_pnl_pct=-3.1))['decision']=='emergency_stop'
def test_stale(): assert CapitalGuard().evaluate(s(data_fresh=False))['decision']=='block'
def test_risk_cap(): assert CapitalGuard().evaluate(s(proposed_risk_pct=1))['decision']=='block'
def test_kill_switch():
 k=KillSwitch();k.engage('test');assert k.gate()['enabled'] and not k.gate()['execution_authority']
