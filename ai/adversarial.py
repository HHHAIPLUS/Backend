from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field

class ChallengeSeverity(str, Enum):
    LOW='low'; MEDIUM='medium'; HIGH='high'; CRITICAL='critical'

class Challenge(BaseModel):
    challenge_id: str
    title: str
    severity: ChallengeSeverity
    score: float = Field(ge=0, le=1)
    evidence: list[str] = []
    counterfactual: str
    action: str
    timestamp: datetime

class AdversarialReport(BaseModel):
    symbol: str
    challenged_action: str
    robustness: float = Field(ge=0, le=1)
    challenge_pressure: float = Field(ge=0, le=1)
    should_block: bool
    challenges: list[Challenge]
    summary: str
    timestamp: datetime

class AdversarialEngine:
    """Tries to falsify a proposed decision. It has no execution authority."""
    def evaluate(self, *, symbol: str, proposed_action: str, context: dict) -> AdversarialReport:
        c=[]
        side = context.get('position_side')
        momentum=float(context.get('momentum',0)); trend=float(context.get('trend_strength',0))
        buy=float(context.get('buying_pressure',0)); sell=float(context.get('selling_pressure',0))
        news=float(context.get('news_risk',0)); liq=float(context.get('liquidity_stress',0))
        vol=float(context.get('volatility',0)); cred=float(context.get('news_credibility',1))
        def add(i,title,sev,score,evidence,counter,action):
            c.append(Challenge(challenge_id=i,title=title,severity=sev,score=max(0,min(1,score)),evidence=evidence,counterfactual=counter,action=action,timestamp=datetime.now(timezone.utc)))
        directional = proposed_action.lower() in {'bullish','long','hold','trail_profit'}
        if directional and sell > buy + .25:
            add('flow_contradiction','Selling pressure contradicts the thesis',ChallengeSeverity.HIGH,min(1,(sell-buy+.25)/1.25),['Selling pressure materially exceeds buying pressure.'],'If selling accelerates, the bullish thesis can fail quickly.','reduce_or_exit')
        if (not directional) and buy > sell + .25:
            add('flow_contradiction_bear','Buying pressure contradicts the bearish thesis',ChallengeSeverity.HIGH,min(1,(buy-sell+.25)/1.25),['Buying pressure materially exceeds selling pressure.'],'If buyers absorb supply, a bearish position may reverse.','reduce_or_exit')
        if news >= .7:
            add('news_shock','External-news risk is elevated',ChallengeSeverity.HIGH,news,['News risk is elevated.'], 'A new headline can invalidate a technical setup before indicators react.','wait_or_reduce')
        if liq >= .7:
            add('liquidity','Liquidity stress could amplify slippage',ChallengeSeverity.HIGH,liq,['Liquidity stress is elevated.'],'A fast move may make planned exits more expensive or slower.','reduce_exposure')
        if vol >= .8:
            add('volatility','Extreme volatility makes the forecast fragile',ChallengeSeverity.MEDIUM,vol,['Volatility is elevated.'],'The expected path can be overwhelmed by a larger-than-normal move.','lower_size')
        if trend < .35 and abs(momentum) < .2:
            add('weak_structure','Directional evidence is weak',ChallengeSeverity.MEDIUM,.65,['Trend and momentum are both weak.'],'The market may remain range-bound and punish directional conviction.','wait')
        if cred < .5 and news > .45:
            add('source_quality','News evidence is weakly verified',ChallengeSeverity.MEDIUM,1-cred,['News credibility is low relative to news risk.'],'Do not overreact to an unverified headline.','verify')
        pressure=max((x.score for x in c),default=0)
        # Critical block only when multiple independent severe risks align or one extreme condition exists.
        critical=sum(1 for x in c if x.severity in {ChallengeSeverity.HIGH,ChallengeSeverity.CRITICAL} and x.score>=.75)
        block = pressure >= .92 or critical >= 2
        robustness=max(0,min(1,1-pressure*.72-len(c)*.025))
        if block: summary='Adversarial review found strong evidence against the proposed action. Normal execution should be blocked pending risk review.'
        elif c: summary='The proposal survived, but material counter-evidence exists. Re-evaluate before increasing exposure.'
        else: summary='No material contradiction was detected in the supplied evidence.'
        return AdversarialReport(symbol=symbol,challenged_action=proposed_action,robustness=robustness,challenge_pressure=pressure,should_block=block,challenges=c,summary=summary,timestamp=datetime.now(timezone.utc))
