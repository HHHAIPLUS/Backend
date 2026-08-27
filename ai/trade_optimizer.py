from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterable
class TradeDecision(str, Enum):
    TRADE='trade'; WAIT='wait'; BLOCK='block'
@dataclass
class MarketRegime:
    name:str; trend_strength:float=0.0; volatility:float=0.0; liquidity:float=1.0
@dataclass
class TradeCandidate:
    symbol:str; side:str; entry:float; target:float; invalidation:float; probability_of_success:float; regime_fit:float; confirmation:float; timing_quality:float; liquidity_score:float; news_risk:float=0.0
@dataclass
class OptimizationPolicy:
    min_expected_value:float=0.0; min_risk_reward:float=1.25; min_trade_quality:float=0.60; min_entry_timing:float=0.55; max_news_risk:float=0.70
class TradeOptimizer:
    def __init__(self, policy:OptimizationPolicy|None=None): self.policy=policy or OptimizationPolicy()
    @staticmethod
    def risk_reward(c:TradeCandidate)->float:
        risk=abs(c.entry-c.invalidation); reward=abs(c.target-c.entry); return reward/risk if risk>0 else 0.0
    @staticmethod
    def expected_value(c:TradeCandidate)->float:
        risk=abs(c.entry-c.invalidation); reward=abs(c.target-c.entry); p=max(0,min(1,c.probability_of_success)); return p*reward-(1-p)*risk
    @staticmethod
    def trade_quality(c:TradeCandidate,r:MarketRegime)->float:
        vals=[c.probability_of_success,c.regime_fit,c.confirmation,c.liquidity_score,r.liquidity]; return sum(max(0,min(1,v)) for v in vals)/len(vals)
    def evaluate(self,c:TradeCandidate,r:MarketRegime)->dict:
        rr=self.risk_reward(c); ev=self.expected_value(c); q=self.trade_quality(c,r); timing=max(0,min(1,c.timing_quality)); reasons=[]
        if rr<self.policy.min_risk_reward: reasons.append('Risk/reward is below the minimum threshold.')
        if ev<=self.policy.min_expected_value: reasons.append('Expected value is not positive enough.')
        if q<self.policy.min_trade_quality: reasons.append('Overall trade quality is below the threshold.')
        if timing<self.policy.min_entry_timing: reasons.append('Entry timing is not sufficiently confirmed.')
        if c.news_risk>self.policy.max_news_risk: reasons.append('Current news/event risk is too high.')
        return {'decision':TradeDecision.TRADE.value if not reasons else TradeDecision.WAIT.value,'symbol':c.symbol,'side':c.side,'risk_reward':rr,'expected_value':ev,'trade_quality':q,'entry_timing':timing,'regime':r.name,'reasons':reasons,'execution_authority':False}
    def rank(self,candidates:Iterable[TradeCandidate],r:MarketRegime)->list[dict]:
        results=[self.evaluate(c,r) for c in candidates]; return sorted(results,key=lambda x:(x['decision']=='trade',x['expected_value'],x['trade_quality']),reverse=True)
