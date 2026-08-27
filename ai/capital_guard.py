from dataclasses import dataclass
from enum import Enum

class GuardDecision(str, Enum):
    ALLOW='allow'; BLOCK='block'; EMERGENCY_STOP='emergency_stop'

@dataclass
class CapitalPolicy:
    max_position_risk_pct: float=0.5
    max_daily_loss_pct: float=3.0
    max_drawdown_pct: float=8.0
    max_leverage: float=5.0
    max_open_positions: int=3
    min_free_margin_pct: float=30.0
    max_slippage_bps: float=20.0

@dataclass
class RiskSnapshot:
    equity: float; free_margin: float; daily_pnl_pct: float; drawdown_pct: float
    proposed_risk_pct: float; leverage: float; open_positions: int
    expected_slippage_bps: float; data_fresh: bool=True; exchange_healthy: bool=True

class CapitalGuard:
    def __init__(self, policy=None): self.policy=policy or CapitalPolicy()
    def evaluate(self,s):
        if s.equity<=0: return {'decision':'emergency_stop','reasons':['Non-positive equity'],'execution_authority':False}
        reasons=[]
        if not s.data_fresh: reasons.append('Market data is stale.')
        if not s.exchange_healthy: reasons.append('Exchange health check failed.')
        if s.daily_pnl_pct <= -self.policy.max_daily_loss_pct: reasons.append('Daily loss limit reached.')
        if s.drawdown_pct >= self.policy.max_drawdown_pct: reasons.append('Maximum drawdown limit reached.')
        if s.proposed_risk_pct > self.policy.max_position_risk_pct: reasons.append('Proposed position risk is too large.')
        if s.leverage > self.policy.max_leverage: reasons.append('Leverage exceeds policy.')
        if s.open_positions >= self.policy.max_open_positions: reasons.append('Maximum open-position count reached.')
        free_pct=s.free_margin/s.equity*100
        if free_pct < self.policy.min_free_margin_pct: reasons.append('Free margin buffer is too small.')
        if s.expected_slippage_bps > self.policy.max_slippage_bps: reasons.append('Expected slippage is too high.')
        decision='emergency_stop' if s.daily_pnl_pct <= -self.policy.max_daily_loss_pct or s.drawdown_pct >= self.policy.max_drawdown_pct else ('block' if reasons else 'allow')
        return {'decision':decision,'reasons':reasons,'free_margin_pct':free_pct,'execution_authority':False}
    def size_for_risk(self,equity,stop_distance_pct,risk_pct):
        if equity<=0 or stop_distance_pct<=0 or risk_pct<=0:return 0.0
        return equity*(risk_pct/100)/(stop_distance_pct/100)

class KillSwitch:
    def __init__(self): self.enabled=False; self.reason=''
    def engage(self,reason): self.enabled=True; self.reason=reason
    def reset(self): self.enabled=False; self.reason=''
    def gate(self): return {'enabled':self.enabled,'reason':self.reason,'execution_authority':False}
