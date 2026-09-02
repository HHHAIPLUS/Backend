from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from types import MethodType
from typing import Any

from ai.stage5_engine import Stage5DecisionEngine
from app.market_data.realtime import build_world_intelligence
from app.ml.predictive import predictive_model


@dataclass
class PositionObservation:
    exchange: str; symbol: str; side: str; quantity: float; entry_price: float; current_price: float
    unrealized_return: float; peak_return: float; thesis_integrity: float; momentum: float; trend_strength: float
    buying_pressure: float; selling_pressure: float; volatility: float; liquidity_stress: float; news_risk: float
    market_risk: float; funding_bias: float; open_interest_change: float; expected_continuation_value: float
    downside_risk: float; timestamp: str


@dataclass
class PositionDecision:
    action: str; close_fraction: float; thesis_integrity: float; expected_continuation_value: float
    downside_risk: float; shock_score: float; protection_price: float | None; reason: str; evidence: dict[str, Any]
    def as_dict(self): return asdict(self)


class PositionIntelligenceEngine:
    VERSION = "stage6-position-intelligence-v1"

    @staticmethod
    def clamp(x, lo=0.0, hi=1.0): return max(lo, min(hi, float(x)))

    def thesis_integrity(self, p: PositionObservation, thesis: dict[str, Any]) -> float:
        d = 1.0 if p.side.lower() == "long" else -1.0
        entry_m = float(thesis.get("momentum", 0.0) or 0.0)
        pressure = p.buying_pressure if d > 0 else p.selling_pressure
        alignment = self.clamp(0.5 + d * p.momentum * 0.5)
        original = self.clamp(0.5 + d * entry_m * 0.5)
        deterioration = max(0.0, -d * (p.momentum - entry_m))
        return self.clamp(0.22*alignment + 0.16*original + 0.22*self.clamp(p.trend_strength) + 0.20*self.clamp(pressure) + 0.10*(1-self.clamp(p.news_risk)) + 0.10*(1-self.clamp(p.liquidity_stress)) - 0.20*deterioration)

    def evaluate(self, p: PositionObservation, thesis: dict[str, Any], *, predictive=None, stage5=None, previous_protection=None) -> PositionDecision:
        if not all(isfinite(float(x)) and float(x) > 0 for x in (p.entry_price, p.current_price, p.quantity)):
            return PositionDecision("hold", 0, 0, 0, 1, 1, previous_protection, "Invalid position telemetry; fail closed.", {"invalid_telemetry": True})
        predictive = predictive or {}; stage5 = stage5 or {}
        integrity = self.thesis_integrity(p, thesis)
        probs = predictive.get("probabilities") or {}
        dp = float(probs.get("long", 0) if p.side.lower()=="long" else probs.get("short", 0))
        expected = float(stage5.get("expected_value", p.expected_continuation_value) or 0) + (dp-0.5)*0.04
        expected = max(-0.20, min(0.20, expected))
        adverse_m = max(0, -(p.momentum if p.side.lower()=="long" else -p.momentum))
        opposing = p.selling_pressure if p.side.lower()=="long" else p.buying_pressure
        retrace = self.clamp((p.peak_return-p.unrealized_return)/p.peak_return) if p.peak_return>0 else 0
        downside = self.clamp(0.24*p.volatility+0.18*p.liquidity_stress+0.18*p.news_risk+0.14*p.market_risk+0.14*adverse_m+0.12*opposing+0.10*max(0,-p.unrealized_return*12))
        shock = self.clamp(0.30*p.news_risk+0.25*p.market_risk+0.25*p.liquidity_stress+0.20*adverse_m)
        edge = expected-downside*0.025
        if shock>=0.86 or p.liquidity_stress>=0.94: action, frac, reason = "emergency_exit", 1.0, "Market-wide or liquidity shock makes continued exposure unsafe."
        elif integrity<0.38 and p.unrealized_return<=0: action, frac, reason = "exit", 1.0, "The original trade thesis has materially failed while the position is losing."
        elif integrity<0.38 or downside>=0.78: action, frac, reason = (("exit",1.0) if p.unrealized_return>0.01 else ("reduce",0.50)), "Thesis integrity or downside risk has deteriorated beyond continuation value."
        elif p.unrealized_return>0 and (retrace>=0.35 or edge<-0.005 or shock>=0.55): action, frac, reason = "reduce", 0.35, "Profit is available but continuation value has deteriorated; bank part of the gain."
        elif p.unrealized_return>0 and (retrace>=0.18 or downside>=0.45): action, frac, reason = "protect", 0.0, "The thesis remains viable, but protection should tighten as risk rises."
        else: action, frac, reason = "hold", 0.0, "Continuation evidence remains stronger than current downside and shock risk."
        protection = None
        if action in {"hold","protect","reduce"} and p.unrealized_return>0:
            buf=self.clamp(0.001+p.volatility*0.006+max(downside,shock)*0.003,0.001,0.018)
            proposed=p.current_price*(1-buf) if p.side.lower()=="long" else p.current_price*(1+buf)
            protection = proposed if previous_protection is None else (max(float(previous_protection),proposed) if p.side.lower()=="long" else min(float(previous_protection),proposed))
        return PositionDecision(action,frac,integrity,expected,downside,shock,protection,reason,{"predictive_probability":dp,"retracement_from_peak":retrace,"edge_after_risk":edge,"stage5_action":stage5.get("action"),"stage5_vetoes":stage5.get("vetoes",[]),"thesis":thesis})

POSITION_ENGINE=PositionIntelligenceEngine()


def install_stage6_position_intelligence(trader: Any) -> None:
    if getattr(trader,"_stage6_installed",False): return
    trader._stage6_installed=True; trader.stage6_version=POSITION_ENGINE.VERSION; trader.stage6_reviews=0; trader.stage6_last_decision=None; trader.stage6_last_error=None; trader._stage6_state={}

    async def manage(self,symbol):
        mode=self.execution_mode
        if mode in {"testnet","live"} and not self._execution_gate()[0]: return []
        exchange="paper" if mode=="paper" else self._exchange_for_market(); adapter=None
        if mode!="paper":
            adapter=self._adapters()[exchange]
            try: raw=await adapter.get_positions(symbol if exchange=="bitget" else None)
            except TypeError: raw=await adapter.get_positions()
            raw_rows=self._position_rows(raw)
        else:
            raw_rows=[{"symbol":p.symbol,"holdSide":p.side,"positionAmt":p.quantity,"entryPrice":p.entry_price,"markPrice":p.mark_price} for p in self.paper.positions.values() if p.symbol.upper()==symbol.upper()]
        results=[]; now=datetime.now(timezone.utc)
        for row in raw_rows:
            ps,side,qty,entry,current=self._position_fields(row)
            if ps!=symbol.upper() or qty<=0 or entry<=0 or current<=0: continue
            key=f"{exchange}:{ps}:{side}"
            world=(await asyncio.to_thread(build_world_intelligence,ps,exchange)).model_dump(mode="json")
            market=world.get("market",world); f=market.get("features") or {}; flow=market.get("order_flow") or {}; der=market.get("derivatives") or {}; reg=market.get("regime") or {}; liq=market.get("liquidity") or {}; news=market.get("news") or {}
            if mode=="paper": current=float(f.get("last_price",current) or current); self.paper.mark(ps,current)
            ret=(current-entry)/entry if side=="long" else (entry-current)/entry; peak=max(ret,self.position_peaks.get(key,ret)); self.position_peaks[key]=peak
            saved=self._stage6_state.get(key,{})
            thesis=saved.get("thesis") or {"side":side,"entry_price":entry,"momentum":float(f.get("momentum",0) or 0),"trend_strength":abs(float(f.get("trend_strength",0) or 0)),"opened_at":now.isoformat(),"entry_evidence":market}
            p=PositionObservation(exchange,ps,side,qty,entry,current,ret,peak,0.5,float(f.get("momentum",0) or 0),abs(float(f.get("trend_strength",0) or 0)),float(flow.get("aggressive_buy_ratio",0.5) or 0.5),1-float(flow.get("aggressive_buy_ratio",0.5) or 0.5),float(f.get("volatility_proxy",0) or 0),float(f.get("liquidity_stress",0) or liq.get("spread_bps",0)/50),float(news.get("risk",f.get("news_risk",0)) or 0),float(reg.get("market_risk",0) or 0),float(der.get("funding_rate",0) or 0)*100,float(der.get("open_interest_change",0) or 0),0,0,now.isoformat())
            pred=predictive_model.predict(f)
            try: s5=Stage5DecisionEngine().evaluate(market_state=market,predictive=pred,risk_vetoes=[],position_side=side,unrealized_return=ret,thesis_integrity=0.5).as_dict()
            except Exception: s5={"action":"WAIT","vetoes":["stage5_unavailable"],"expected_value":0}
            d=POSITION_ENGINE.evaluate(p,thesis,predictive=pred,stage5=s5,previous_protection=saved.get("protection_price")); action={"status":"observed","mode":mode}; cooldown=float(saved.get("cooldown_until",0) or 0)
            if now.timestamp()>=cooldown and mode in {"testnet","live"} and d.action in {"exit","emergency_exit"}:
                pm=await adapter.get_position_mode() if hasattr(adapter,"get_position_mode") else "ONE_WAY"
                try: action=await adapter.close_position(ps,side,qty,pm)
                except Exception as e: action={"status":"error","error":f"{type(e).__name__}: {e}"}
            elif now.timestamp()>=cooldown and mode in {"testnet","live"} and d.action=="reduce":
                pm=await adapter.get_position_mode() if hasattr(adapter,"get_position_mode") else "ONE_WAY"; cq=min(qty,max(0.001,self._safe_quantity(qty*d.close_fraction)))
                try: action=await adapter.close_position(ps,side,cq,pm)
                except Exception as e: action={"status":"error","error":f"{type(e).__name__}: {e}"}
            elif now.timestamp()>=cooldown and mode in {"testnet","live"} and d.protection_price is not None:
                old=saved.get("protection_price"); changed=old is None or abs(d.protection_price-float(old))/current>=0.0005
                if changed:
                    pm=await adapter.get_position_mode() if hasattr(adapter,"get_position_mode") else "ONE_WAY"
                    try: action=await adapter.update_dynamic_protection(ps,side,qty,d.protection_price,pm)
                    except Exception as e: action={"status":"error","error":f"{type(e).__name__}: {e}"}
            elif mode=="paper":
                if d.action in {"exit","emergency_exit"}: o=self.paper.close(ps,current); action={"status":"closed","order_id":getattr(o,"order_id",None),"mode":"paper"}
                elif d.action=="reduce" and d.close_fraction>0:
                    pp=self.paper.positions.get(ps)
                    if pp:
                        rq=min(pp.quantity,pp.quantity*d.close_fraction); pp.realized_pnl+=(current-pp.entry_price)*rq*(1 if pp.side=="long" else -1); pp.quantity-=rq; action={"status":"reduced","reduced_quantity":rq,"remaining_quantity":pp.quantity,"mode":"paper"}
            remaining=0.0
            if mode=="paper": remaining=float(self.paper.positions.get(ps).quantity) if self.paper.positions.get(ps) else 0.0
            elif d.action in {"exit","emergency_exit","reduce"} and action.get("status")!="error":
                try:
                    rr=await adapter.get_positions(ps if exchange=="bitget" else None)
                    for r in self._position_rows(rr):
                        ss,sd,qq,_,_=self._position_fields(r)
                        if ss==ps and sd==side: remaining=qq; break
                except Exception as e: action={**action,"status":"reconciliation_failed","reconciliation_error":f"{type(e).__name__}: {e}"}
            state={"thesis":thesis,"protection_price":d.protection_price or saved.get("protection_price"),"peak_return":peak,"opened_at":thesis.get("opened_at"),"last_action":d.action,"last_review":now.isoformat(),"cooldown_until":now.timestamp()+ (10 if d.action in {"exit","emergency_exit","reduce"} else 8),"remaining_quantity":remaining,"entry_evidence":thesis.get("entry_evidence"),"last_decision":d.as_dict()}; self._stage6_state[key]=state; self.stage6_reviews+=1; self.stage6_last_decision={"symbol":ps,"side":side,**d.as_dict(),"remaining_quantity":remaining,"execution":action}; results.append(self.stage6_last_decision)
            try:
                from app.persistence.repository import upsert_position_state,record_event
                await upsert_position_state(exchange,ps,side,state); await record_event("stage6_position_review",{"exchange":exchange,"symbol":ps,"side":side,"decision":d.as_dict(),"stage5":s5,"execution":action,"remaining_quantity":remaining})
            except Exception as e: self.stage6_last_error=f"persistence: {type(e).__name__}: {e}"
        self.last_position_management=results[-20:]; return results

    async def run(self):
        import time
        nd=nm=0.0
        while self.running:
            now=time.monotonic()
            if now>=nm:
                for s in self.config.symbols:
                    try: await manage(self,s)
                    except Exception as e: self.stage6_last_error=f"{type(e).__name__}: {e}"
                nm=now+max(2,self.position_review_interval)
            if now>=nd:
                for s in self.config.symbols:
                    try: self.last_cycle=await self.run_cycle(s); self.last_cycle_at=datetime.now(timezone.utc); self.last_error=None
                    except Exception as e: self.last_error=f"{type(e).__name__}: {e}"
                nd=now+self.config.interval_seconds
            await asyncio.sleep(1)
    trader._manage_open_positions=MethodType(manage,trader); trader._run=MethodType(run,trader)
    original_status=trader.status
    def status(self):
        r=original_status(); r.update({"position_intelligence":self.stage6_version,"position_reviews":self.stage6_reviews,"stage6_last_decision":self.stage6_last_decision,"stage6_last_error":self.stage6_last_error,"position_management_execution_authority":self.execution_mode in {"testnet","live"} and self._execution_gate()[0]}); return r
    trader.status=MethodType(status,trader)
