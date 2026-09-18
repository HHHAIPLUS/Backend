from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import uuid4

from ai.agents import AgentContext, IntelligenceCouncil
from ai.capital_guard import CapitalGuard, RiskSnapshot
from ai.decision_fusion import DecisionFusion
from ai.paper_trading import PaperExecutionEngine, TradingMode
from ai.scenario_engine import ScenarioEngine, ScenarioRequest
from ai.trade_optimizer import MarketRegime, TradeCandidate, TradeOptimizer
from ai.cognitive_exit import CognitiveExitEngine, ExitAction, PositionTelemetry
from ai.counterfactual import CounterfactualTradeTwin
from app.persistence.repository import upsert_position_state, load_position_states
from app.core.config import settings
from app.exchanges.factory import adapters
from app.exchanges.safe_router import SafeExchangeRouter
from app.market_data.realtime import build_world_intelligence
from app.market_data.binance_user_stream import binance_user_stream
from app.ml.predictive import predictive_model
from app.ml.predictive_brain import predictive_brain
from app.persistence.repository import record_decision, record_outcome, record_event
from app.persistence.supabase import store

log = logging.getLogger("hhhai.autonomous_trader")
brain = predictive_brain


@dataclass
class BotConfig:
    mode: str = "paper"
    symbols: tuple[str, ...] = ("BTCUSDT",)
    interval_seconds: int = 30
    paper_equity: float = 10_000.0
    min_confidence: float = 0.60
    max_spread_bps: float = 12.0
    risk_pct: float = 0.5
    reward_risk: float = 1.8


class AutonomousTrader:
    """End-to-end decision/execution coordinator."""

    def __init__(self) -> None:
        symbols = tuple(x.strip().upper() for x in os.getenv("HHHAI_TRADE_SYMBOLS", "BTCUSDT").split(",") if x.strip())
        self.position_review_interval = max(2, int(os.getenv("HHHAI_POSITION_REVIEW_INTERVAL_SECONDS", "5")))
        self.max_position_reviews_per_cycle = max(1, int(os.getenv("HHHAI_MAX_POSITION_REVIEWS_PER_CYCLE", "20")))
        self.config = BotConfig(
            mode=os.getenv("HHHAI_TRADING_MODE", "paper").lower(), symbols=symbols or ("BTCUSDT",),
            interval_seconds=max(10, int(os.getenv("HHHAI_TRADING_INTERVAL_SECONDS", "30"))),
            paper_equity=max(100.0, float(os.getenv("HHHAI_PAPER_EQUITY", "10000"))),
            min_confidence=float(os.getenv("HHHAI_MIN_TRADE_CONFIDENCE", "0.60")),
            max_spread_bps=float(os.getenv("HHHAI_MAX_SPREAD_BPS", "12")),
            risk_pct=float(os.getenv("HHHAI_RISK_PER_TRADE_PCT", "0.5")), reward_risk=float(os.getenv("HHHAI_MIN_REWARD_RR", "1.8")),
        )
        self.running = False; self.started_at: datetime | None = None; self.last_cycle_at: datetime | None = None
        self.last_cycle: dict[str, Any] | None = None; self.last_error: str | None = None; self._task: asyncio.Task | None = None
        self.paper = PaperExecutionEngine(TradingMode.PAPER); self.council = IntelligenceCouncil(); self.scenarios = ScenarioEngine()
        self.optimizer = TradeOptimizer(); self.fusion = DecisionFusion(); self.guard = CapitalGuard(); self.router = SafeExchangeRouter(self._adapters())
        self.decision_ids: list[str] = []; self._day_key: str | None = None; self._day_start_equity: float | None = None; self._peak_equity: float | None = None
        self.exit_engine = CognitiveExitEngine(); self.counterfactual = CounterfactualTradeTwin(); self.position_peaks: dict[str, float] = {}
        self.position_opened: dict[str, datetime] = {}; self.position_last_action: dict[str, str] = {}; self.position_cooldown_until: dict[str, datetime] = {}
        self.last_position_management: list[dict[str, Any]] = []

    @property
    def execution_mode(self) -> str: return self.config.mode
    def _adapters(self): return adapters(testnet=self.execution_mode == "testnet")

    def _execution_gate(self) -> tuple[bool, str]:
        mode = self.execution_mode
        if mode == "paper": return True, "paper execution"
        autotrading_enabled = os.getenv("HHHAI_AUTOTRADING_ENABLED", "false").lower() == "true"
        if mode == "testnet":
            if not autotrading_enabled: return False, "autonomous trading is disabled by HHHAI_AUTOTRADING_ENABLED"
            if not settings.testnet_trading_enabled: return False, "testnet execution is disabled by TESTNET_TRADING_ENABLED"
            return True, "testnet execution enabled"
        if mode == "live":
            if not autotrading_enabled: return False, "autonomous trading is disabled by HHHAI_AUTOTRADING_ENABLED"
            if not settings.live_trading_enabled: return False, "live execution is disabled by LIVE_TRADING_ENABLED"
            return True, "live execution enabled"
        return False, f"unsupported trading mode: {mode}"

    def status(self) -> dict[str, Any]:
        allowed, reason = self._execution_gate()
        return {"engine":"HHHAI Autonomous Trading Engine","running":self.running,"mode":self.execution_mode,"symbols":list(self.config.symbols),"interval_seconds":self.config.interval_seconds,"execution_gate":allowed,"execution_gate_reason":reason,"live_trading_enabled":settings.live_trading_enabled,"testnet_trading_enabled":settings.testnet_trading_enabled,"validated_model":brain.bundle is not None,"paper":self.paper.snapshot(),"last_cycle_at":self.last_cycle_at.isoformat() if self.last_cycle_at else None,"last_error":self.last_error,"execution_authority":allowed and self.execution_mode in {"testnet","live"},"real_money":self.execution_mode=="live" and settings.live_trading_enabled,"cognitive_exit_engine":self.exit_engine.version,"last_position_management":self.last_position_management[-20:]}

    async def start(self) -> dict[str, Any]:
        if self.running: return self.status()
        self.running=True; self.started_at=datetime.now(timezone.utc); self.last_error=None; await self._hydrate_position_state(); self._task=asyncio.create_task(self._run(),name="hhhai-autonomous-trader"); return self.status()

    async def stop(self) -> dict[str, Any]:
        self.running=False; task=self._task; self._task=None
        if task and task is not asyncio.current_task():
            try: await asyncio.wait_for(task,timeout=5)
            except (asyncio.TimeoutError,asyncio.CancelledError): task.cancel()
        return self.status()

    async def _run(self) -> None:
        log.warning("AUTOTRADER_LOOP_STARTED mode=%s symbols=%s",self.execution_mode,self.config.symbols); next_decision=0.0; next_management=0.0
        while self.running:
            now=asyncio.get_running_loop().time()
            if self.execution_mode in {"testnet","live"} and now>=next_management:
                reviews=0
                for symbol in self.config.symbols:
                    if reviews>=self.max_position_reviews_per_cycle: break
                    try: managed=await self._manage_open_positions(symbol); reviews+=len(managed); self.last_error=None
                    except Exception as exc: self.last_error=f"{type(exc).__name__}: {exc}"; log.exception("Position-management cycle failed for %s",symbol)
                next_management=now+self.position_review_interval
            if now>=next_decision:
                for symbol in self.config.symbols:
                    try:
                        self.last_cycle=await self.run_cycle(symbol)
                        self.last_cycle_at=datetime.now(timezone.utc)
                        self.last_error=None
                        log.warning("AUTOTRADER_CYCLE symbol=%s action=%s execution=%s risk=%s",symbol,self.last_cycle.get("decision",{}).get("action"),self.last_cycle.get("execution",{}).get("status"),self.last_cycle.get("risk",{}).get("decision"))
                    except Exception as exc: self.last_error=f"{type(exc).__name__}: {exc}"; log.exception("Autonomous decision cycle failed for %s",symbol)
                next_decision=now+self.config.interval_seconds
            try: await asyncio.sleep(min(1.0,self.position_review_interval))
            except asyncio.CancelledError: break
        log.info("Autonomous trader stopped")

    async def _hydrate_position_state(self) -> None:
        if not store.configured: return
        try:
            rows=await load_position_states()
            for row in rows or []:
                payload=row.get('payload') or {}; key=f"{row.get('exchange')}:{row.get('symbol')}:{row.get('side')}"
                if payload.get('peak_return') is not None: self.position_peaks[key]=float(payload['peak_return'])
                if payload.get('opened_at'): self.position_opened[key]=datetime.fromisoformat(str(payload['opened_at']).replace('Z','+00:00'))
                if payload.get('last_action'): self.position_last_action[key]=str(payload['last_action'])
        except Exception: log.exception('Unable to hydrate persistent position state; starting from live exchange state')

    @staticmethod
    def _position_rows(raw):
        if isinstance(raw,dict): raw=raw.get('data',raw.get('positions',raw.get('list',[])))
        return raw if isinstance(raw,list) else []
    @staticmethod
    def _position_fields(row: dict[str,Any]):
        symbol=str(row.get('symbol') or row.get('instId') or '').upper(); hold_side=str(row.get('holdSide') or '').lower(); raw_amt=row.get('positionAmt',row.get('total',row.get('available','0')))
        try: amount=float(raw_amt or 0)
        except Exception: amount=0.0
        side=hold_side if hold_side in {'long','short'} else ('long' if amount>0 else 'short'); qty=abs(amount); entry=float(row.get('entryPrice') or row.get('openPriceAvg') or row.get('avgOpenPrice') or 0); current=float(row.get('markPrice') or row.get('currentPrice') or row.get('mark_price') or 0)
        return symbol,side,qty,entry,current

    async def _manage_open_positions(self,symbol: str, rows: list[dict[str,Any]] | None = None) -> list[dict[str,Any]]:
        exchange=self._exchange_for_market(); adapter=self._adapters()[exchange]
        if exchange=='binance':
            from app.market_data.binance_user_stream import binance_user_stream
            if not binance_user_stream.is_healthy(): return []
            rows=binance_user_stream.positions(symbol)
        elif rows is None:
            try: raw=await adapter.get_positions(symbol if exchange=='bitget' else None)
            except TypeError: raw=await adapter.get_positions()
            except Exception as exc: log.warning('Position sync failed for %s: %s',symbol,exc); return []
            rows=self._position_rows(raw)
        else:
            rows = self._position_rows(rows)
        managed=[]
        for row in rows:
            psymbol,side,qty,entry,current=self._position_fields(row)
            if psymbol!=symbol or qty<=0 or entry<=0: continue
            if current<=0:
                try: current=float((await asyncio.to_thread(build_world_intelligence,psymbol,exchange)).market.price)
                except Exception: continue
            key=f"{exchange}:{psymbol}:{side}"; ret=(current-entry)/entry if side=='long' else (entry-current)/entry; peak=max(ret,self.position_peaks.get(key,ret)); self.position_peaks[key]=peak; opened=self.position_opened.setdefault(key,datetime.now(timezone.utc))
            world=(await asyncio.to_thread(build_world_intelligence,psymbol,exchange)).model_dump(mode='json'); m=world['market']; imbalance=float(m.get('order_book_imbalance') or 0); buying,selling=self._pressures(imbalance); direction=1 if side=='long' else -1
            telemetry=PositionTelemetry(side=side,entry_price=entry,current_price=current,unrealized_return=ret,peak_return=peak,minutes_open=max(0.0,(datetime.now(timezone.utc)-opened).total_seconds()/60),momentum=float(world.get('momentum_proxy') or 0),trend_strength=float(world.get('trend_strength') or 0),buying_pressure=buying,selling_pressure=selling,volatility=float(m.get('volatility_proxy') or 0),liquidity_stress=float(world.get('liquidity_stress') or 0),news_risk=float(world.get('news_risk') or 0),thesis_integrity=max(0.0,min(1.0,0.5+direction*float(world.get('momentum_proxy') or 0)*0.25+float(world.get('trend_strength') or 0)*0.25)),funding_bias=float(m.get('funding_rate') or 0)*100,open_interest_change=float(m.get('open_interest_change') or 0))
            decision=self.exit_engine.evaluate(telemetry); twin=self.counterfactual.evaluate(telemetry,decision)
            if decision.action==ExitAction.HOLD and twin.selected in {ExitAction.REDUCE,ExitAction.EXIT}: decision.action=twin.selected; decision.reason+=' Counterfactual trade twin independently found that the alternative had better current utility.'; decision.close_fraction=0.35 if twin.selected==ExitAction.REDUCE else 1.0
            now=datetime.now(timezone.utc); cooldown=self.position_cooldown_until.get(key)
            if cooldown and now<cooldown and decision.action in {ExitAction.REDUCE,ExitAction.PROTECT,ExitAction.HOLD}: decision.action=ExitAction.HOLD; decision.close_fraction=0.0
            action_result={'status':'observed'}
            if decision.action in {ExitAction.EXIT,ExitAction.EMERGENCY_EXIT}:
                mode=await adapter.get_position_mode() if hasattr(adapter,'get_position_mode') else 'ONE_WAY'; action_result=await adapter.close_position(psymbol,side,qty,mode); self.position_last_action[key]=decision.action.value; self.position_cooldown_until[key]=now+timedelta(seconds=15)
            elif decision.action==ExitAction.REDUCE and decision.close_fraction>0:
                mode=await adapter.get_position_mode() if hasattr(adapter,'get_position_mode') else 'ONE_WAY'; close_qty=max(0.001,self._safe_quantity(qty*decision.close_fraction)); close_qty=min(qty,close_qty); action_result=await adapter.close_position(psymbol,side,close_qty,mode); self.position_last_action[key]=decision.action.value; self.position_cooldown_until[key]=now+timedelta(seconds=15)
            elif decision.protection_price is not None:
                mode=await adapter.get_position_mode() if hasattr(adapter,'get_position_mode') else 'ONE_WAY'; action_result=await adapter.update_dynamic_protection(psymbol,side,qty,decision.protection_price,mode); self.position_last_action[key]=decision.action.value; self.position_cooldown_until[key]=now+timedelta(seconds=8)
            state_payload={'peak_return':peak,'opened_at':opened.isoformat(),'last_action':self.position_last_action.get(key,'hold'),'last_review':now.isoformat(),'entry_price':entry,'current_price':current}
            if store.configured:
                try: await upsert_position_state(exchange,psymbol,side,state_payload); await record_event('cognitive_position_management',{'exchange':exchange,'symbol':psymbol,'side':side,'telemetry':telemetry.__dict__,'decision':decision.__dict__,'counterfactual':twin.__dict__,'execution':action_result})
                except Exception: log.exception('Position management persistence failed for %s',psymbol)
            managed.append({'exchange':exchange,'symbol':psymbol,'side':side,'decision':decision.__dict__,'counterfactual':twin.__dict__,'execution':action_result})
        self.last_position_management=managed[-20:]; return managed

    @staticmethod
    def _pressures(imbalance:float)->tuple[float,float]: return max(0.0,min(1.0,0.5+imbalance/2)),max(0.0,min(1.0,0.5-imbalance/2))
    def _context(self,world:dict[str,Any])->AgentContext:
        m=world['market']; buying,selling=self._pressures(float(m.get('order_book_imbalance') or 0)); return AgentContext(symbol=world['symbol'],momentum=max(-1,min(1,float(world.get('momentum_proxy') or 0))),trend_strength=max(0,min(1,float(world.get('trend_strength') or 0))),buying_pressure=buying,selling_pressure=selling,volatility=max(0,min(1,float(m.get('volatility_proxy') or 0))),liquidity_stress=max(0,min(1,float(world.get('liquidity_stress') or 0))),news_risk=max(0,min(1,float(world.get('news_risk') or 0))),news_sentiment=max(-1,min(1,float(world.get('news_sentiment') or 0))),news_credibility=max(0,min(1,float(world.get('news_credibility') or 0))),funding_bias=max(-1,min(1,float(m.get('funding_rate') or 0)*100)),open_interest_change=max(-1,min(1,float(m.get('open_interest_change') or 0))),correlation_risk=max(0,min(1,float(world.get('market_risk') or 0))),market_regime=str(world.get('regime') or 'unknown'))

    def _features(self,world:dict[str,Any])->dict[str,Any]:
        m=world['market']; model_features=dict(m.get('model_features') or {})
        return {**model_features,'symbol':world['symbol'],'order_book_imbalance':float(m.get('order_book_imbalance') or 0),'funding_rate':float(m.get('funding_rate') or 0),'open_interest_change':float(m.get('open_interest_change') or 0),'news_risk':float(world.get('news_risk') or 0),'news_sentiment':float(world.get('news_sentiment') or 0),'liquidity_stress':float(world.get('liquidity_stress') or 0)}

    def _candidate(self,world:dict[str,Any],action:str)->TradeCandidate:
        m=world['market']; price=float(m['price']); vol=max(0.002,float(m.get('volatility_proxy') or 0.005)); stop_distance=min(0.03,max(0.003,vol*1.5)); side='long' if action=='LONG' else 'short'; invalidation=price*(1-stop_distance) if side=='long' else price*(1+stop_distance); target=price*(1+stop_distance*self.config.reward_risk) if side=='long' else price*(1-stop_distance*self.config.reward_risk); p=brain.predict(self._features(world)); probability=float(p['probabilities'].get('long' if side=='long' else 'short',0)); regime=str(world.get('regime') or 'unknown')
        return TradeCandidate(symbol=world['symbol'],side=side,entry=price,target=target,invalidation=invalidation,probability_of_success=probability,regime_fit=0.75 if regime in {'trending_up','trending_down'} else 0.55,confirmation=0.75,timing_quality=max(0.0,1-float(m.get('volatility_proxy') or 0)),liquidity_score=max(0.0,1-float(world.get('liquidity_stress') or 0)),news_risk=float(world.get('news_risk') or 0))

    async def run_test10_canary(self, symbol: str = "DOGEUSDT") -> dict[str, Any]:
        """Controlled live execution test: one tiny, exchange-valid position, then observe and close it."""
        exchange = self._exchange_for_market()
        if exchange != "bitget":
            raise RuntimeError("Test 10 canary is configured for Bitget")
        world_model = await asyncio.to_thread(build_world_intelligence, symbol, exchange)
        world = world_model.model_dump(mode="json")
        predictive = brain.predict(self._features(world))
        probs = predictive.get("probabilities") or {}
        long_p = float(probs.get("long") or 0)
        short_p = float(probs.get("short") or 0)
        action = "LONG" if long_p >= short_p else "SHORT"
        candidate = self._candidate(world, action)
        confidence = max(long_p, short_p)
        decision = {
            "action": action,
            "execution_candidate": True,
            "confidence": max(confidence, 0.60),
            "reason": "Test 10 controlled live execution canary using HHHAI live market/predictive data.",
            "vetoes": [],
        }
        risk = await self._risk_check(world, decision, candidate, test10=True)
        if not risk.get("allowed"):
            raise RuntimeError(f"Test 10 risk gate blocked execution: {risk.get('reasons')}")
        execution = await self._execute(symbol, world, decision, candidate, risk, test10=True)
        log.warning("TEST10_EXECUTION %s", execution)
        result = {"symbol": symbol, "action": action, "predictive": predictive, "risk": risk, "execution": execution}
        if execution.get("status") not in {"filled", "submitted"}:
            raise RuntimeError(f"Test 10 order was not confirmed: {execution}")
        adapter = self._adapters()["bitget"]
        management_observed = []
        for attempt in range(6):
            await asyncio.sleep(2)
            try:
                raw_positions = await adapter.get_positions(symbol)
                rows = self._position_rows(raw_positions)
                log.warning("TEST10_POSITION_OBSERVATION attempt=%s rows=%s", attempt + 1, rows)
            except Exception as exc:
                rows = []
                log.warning("TEST10_POSITION_OBSERVATION_FAILED attempt=%s error=%s", attempt + 1, exc)
            managed_now = await self._manage_open_positions(symbol, rows=rows)
            if managed_now:
                management_observed.extend(managed_now)
                log.warning("TEST10_MANAGEMENT_OBSERVED attempt=%s count=%s", attempt + 1, len(managed_now))
                break
        result["management_observed"] = management_observed[-5:]
        rows = self._position_rows(await adapter.get_positions(symbol))
        open_rows = [r for r in rows if self._position_fields(r)[0] == symbol and self._position_fields(r)[2] > 0]
        if open_rows:
            psymbol, side, qty, _, _ = self._position_fields(open_rows[0])
            mode = await adapter.get_position_mode(psymbol)
            result["cleanup"] = await adapter.close_position(psymbol, side, qty, mode)
        await asyncio.sleep(3)
        remaining = self._position_rows(await adapter.get_positions(symbol))
        result["remaining_open"] = [r for r in remaining if self._position_fields(r)[0] == symbol and self._position_fields(r)[2] > 0]
        if not management_observed:
            raise RuntimeError("Test 10 order filled but position management was not observed")
        if result["remaining_open"]:
            raise RuntimeError("Test 10 cleanup did not fully close the position")
        return result

    async def _settle_paper_positions(self, world: dict[str, Any]) -> list[dict[str, Any]]:
        if self.execution_mode != "paper":
            return []
        hold_seconds=max(60,int(os.getenv("HHHAI_PAPER_HOLD_SECONDS","300")))
        symbol=str(world["symbol"]).upper()
        position=self.paper.positions.get(symbol)
        if not position or not position.decision_id:
            return []
        age=(datetime.now(timezone.utc)-position.opened_at).total_seconds()
        if age < hold_seconds:
            return []
        price=float(world["market"]["price"])
        gross=(price-position.entry_price)/position.entry_price if position.side=="long" else (position.entry_price-price)/position.entry_price
        net=float(gross-(2*0.0008))
        record_id=position.decision_id
        action=position.side.upper()
        close_order=self.paper.close(symbol,price)
        payload={"record_id":record_id,"symbol":symbol,"action":action,"realized_return":net,"outcome_profitable":net>0,"entry_price":position.entry_price,"exit_price":price,"holding_seconds":age,"mode":"paper","closed_at":datetime.now(timezone.utc).isoformat(),"close_order_id":getattr(close_order,"order_id",None)}
        if store.configured:
            try: await record_outcome(record_id,payload)
            except Exception: log.exception("Failed to persist paper outcome %s",record_id)
        try:
            from app.ml.adaptive_intelligence import AdaptiveObservation, adaptive_intelligence
            obs=AdaptiveObservation(symbol=symbol,model_version="paper",action=action,confidence=0.0,realized_return=net,observed_at=payload["closed_at"],regime=str(world.get("regime") or "unknown"),horizon=6,expected_probability=None,features={})
            adaptive_intelligence.add_observation(obs)
        except Exception:
            pass
        await record_event("paper_trade_completed",payload) if store.configured else None
        return [payload]

    async def run_cycle(self,symbol:str)->dict[str,Any]:
        world_model=await asyncio.to_thread(build_world_intelligence,symbol,self._exchange_for_market()); world=world_model.model_dump(mode='json')
        paper_outcomes=await self._settle_paper_positions(world)
        context=self._context(world); council=self.council.deliberate(context).model_dump(mode='json'); scenario_req=ScenarioRequest(symbol=symbol,horizon_minutes=60,momentum=context.momentum,trend_strength=context.trend_strength,buying_pressure=context.buying_pressure,selling_pressure=context.selling_pressure,volatility=context.volatility,liquidity_stress=context.liquidity_stress,news_risk=context.news_risk,news_sentiment=context.news_sentiment,market_risk=context.correlation_risk,thesis_integrity=context.thesis_integrity); scenario=self.scenarios.generate(scenario_req).model_dump(mode='json'); proposed='long' if council['action']=='bullish' else 'short' if council['action']=='bearish' else 'wait'
        from ai.adversarial import AdversarialEngine
        adversarial=AdversarialEngine().evaluate(symbol=symbol,proposed_action=proposed,context={'position_side':None,'momentum':context.momentum,'trend_strength':context.trend_strength,'buying_pressure':context.buying_pressure,'selling_pressure':context.selling_pressure,'volatility':context.volatility,'liquidity_stress':context.liquidity_stress,'news_risk':context.news_risk,'news_credibility':context.news_credibility}).model_dump(mode='json'); predictive=brain.predict(self._features(world)); fusion=self.fusion.decide(council_action=council['action'],council_confidence=float(council['confidence']),disagreement=float(council['disagreement']),predictive=predictive,adversarial_block=bool(adversarial['should_block']),scenario_uncertainty=float(scenario['uncertainty']),data_quality=float(world.get('data_quality') or 0),risk_vetoes=council['veto_flags']); decision=fusion.__dict__.copy(); candidate=None; optimizer_result=None
        if decision['action'] in {'LONG','SHORT'}:
            candidate=self._candidate(world,decision['action']); regime_name=str(world.get('regime') or 'unknown'); optimizer_result=self.optimizer.evaluate(candidate,MarketRegime(name=regime_name,trend_strength=context.trend_strength,volatility=context.volatility,liquidity=context.liquidity_stress)).copy()
            if optimizer_result['decision']!='trade': decision['action']='WAIT'; decision['execution_candidate']=False; decision['vetoes']=list(decision.get('vetoes',[]))+['trade_quality_optimizer']
        risk=await self._risk_check(world,decision,candidate)
        if not risk['allowed']: decision['action']='WAIT'; decision['execution_candidate']=False; decision['vetoes']=list(decision.get('vetoes',[]))+risk['reasons']
        execution=await self._execute(symbol,world,decision,candidate,risk); record_id=str(uuid4()); payload={'record_id':record_id,'symbol':symbol,'action':decision['action'],'thesis':decision['reason'],'features':self._features(world),'confidence':float(decision['confidence']),'model_version':predictive.get('version','untrained'),'created_at':datetime.now(timezone.utc).isoformat(),'pipeline':{'council':council,'scenario':scenario,'adversarial':adversarial,'predictive':predictive,'fusion':decision,'risk':risk,'execution':execution}}; self.decision_ids.append(record_id); self.decision_ids=self.decision_ids[-200:]
        if execution.get("status") == "filled_simulated":
            paper_position=self.paper.positions.get(symbol)
            if paper_position is not None:
                paper_position.decision_id=record_id
        if store.configured:
            try: await record_decision(symbol,payload); await record_event('autonomous_cycle',payload)
            except Exception: log.exception('Failed to persist autonomous cycle for %s',symbol)
        return {'symbol':symbol,'observed_at':world.get('observed_at'),'decision':decision,'risk':risk,'execution':execution,'predictive':predictive,'council':council,'scenario':scenario,'adversarial':adversarial,'optimizer':optimizer_result,'record_id':record_id,'paper_outcomes':paper_outcomes}

    def _exchange_for_market(self)->str: return os.getenv('HHHAI_EXECUTION_EXCHANGE','binance').lower() if self.execution_mode in {'testnet','live'} else os.getenv('HHHAI_MARKET_EXCHANGE','binance').lower()

    async def _risk_check(self,world:dict[str,Any],decision:dict[str,Any],candidate:TradeCandidate|None,test10:bool=False)->dict[str,Any]:
        if not candidate or not decision.get('execution_candidate'): return {'allowed':False,'reasons':['no executable candidate'],'decision':'block'}
        equity=self.config.paper_equity; open_positions=len(self.paper.positions)
        if self.execution_mode in {'testnet','live'}:
            try:
                exchange=self._exchange_for_market()
                if exchange=='binance':
                    equity=float(binance_user_stream.cross_wallet_balance() or 0); open_positions=len(binance_user_stream.positions())
                    if equity<=0: raise RuntimeError('Binance user-data balance is not ready')
                else:
                    exchange_adapter=self._adapters()[exchange]; account=await exchange_adapter.get_account_status(); equity=float(account.get('available_balance') or account.get('total_wallet_balance') or 0); positions=await exchange_adapter.get_positions(); open_positions=sum(1 for p in self._position_rows(positions) if abs(float(p.get('positionAmt',p.get('total',0)) or 0))>0)
            except Exception as exc: return {'allowed':False,'reasons':[f'account health unavailable: {exc}'],'decision':'block'}
        m=world['market']; spread_bps=max(0.0,(float(m['ask'])-float(m['bid']))/float(m['price'])*10000); stop_pct=abs(candidate.entry-candidate.invalidation)/candidate.entry*100; size=self.guard.size_for_risk(equity,stop_pct,self.config.risk_pct); self._update_equity_state(equity); daily_pnl_pct=0.0 if not self._day_start_equity else (equity-self._day_start_equity)/self._day_start_equity*100.0; drawdown_pct=0.0 if not self._peak_equity else max(0.0,(self._peak_equity-equity)/self._peak_equity*100.0); snapshot=RiskSnapshot(equity=equity,free_margin=equity,daily_pnl_pct=daily_pnl_pct,drawdown_pct=drawdown_pct,proposed_risk_pct=self.config.risk_pct,leverage=min(5,size*candidate.entry/max(equity,1e-9)),open_positions=open_positions,expected_slippage_bps=spread_bps,data_fresh=float(world.get('data_quality') or 0)>=0.9,exchange_healthy=True); result=self.guard.evaluate(snapshot); reasons=list(result.get('reasons',[]))
        if spread_bps>self.config.max_spread_bps: reasons.append('spread exceeds execution limit')
        if not test10 and decision.get('confidence',0)<self.config.min_confidence: reasons.append('confidence below execution threshold')
        return {'allowed':not reasons and result.get('decision')=='allow','reasons':reasons,'decision':result.get('decision'),'equity':equity,'quantity':size,'spread_bps':spread_bps,'stop_distance_pct':stop_pct}

    def _update_equity_state(self,equity:float)->None:
        now_key=datetime.now(timezone.utc).date().isoformat()
        if self._day_key!=now_key or self._day_start_equity is None: self._day_key=now_key; self._day_start_equity=equity; self._peak_equity=equity
        else: self._peak_equity=max(self._peak_equity or equity,equity)

    async def _execute(self,symbol:str,world:dict[str,Any],decision:dict[str,Any],candidate:TradeCandidate|None,risk:dict[str,Any],test10:bool=False)->dict[str,Any]:
        if not candidate or not decision.get('execution_candidate') or not risk.get('allowed'): return {'status':'not_executed','reason':'execution gates did not pass','live_exchange_order':False}
        quantity=float(risk.get('quantity') or 0)
        if quantity<=0: return {'status':'not_executed','reason':'calculated quantity is zero','live_exchange_order':False}
        price=float(world['market']['price']); side='buy' if candidate.side=='long' else 'sell'
        if self.execution_mode=='paper': order=self.paper.submit(symbol,side,quantity,price); return {'status':'filled_simulated','order_id':order.order_id,'mode':'paper','live_exchange_order':False,'quantity':quantity,'price':price}
        exchange=self._exchange_for_market(); safe_qty=self._safe_quantity(quantity); stop_price=float(candidate.invalidation); take_profit=float(candidate.target)
        if exchange=='bitget':
            adapter=self._adapters()[exchange]
            position_mode=await adapter.get_position_mode(symbol)
            contracts=await adapter.get_contract_config(symbol)
            contract=contracts[0] if isinstance(contracts,list) and contracts else contracts
            if isinstance(contract,dict):
                price_place=int(contract.get('pricePlace') or 8)
                price_step=float(contract.get('priceEndStep') or 1) * (10 ** (-price_place))
                if price_step>0:
                    stop_price=round(round(stop_price/price_step)*price_step,price_place)
                    take_profit=round(round(take_profit/price_step)*price_step,price_place)
            order={'symbol':symbol,'marginCoin':'USDT','side':side,'orderType':'market','size':safe_qty,'tradeSide':'open','marginMode':'crossed'}
            if not test10:
                order['presetStopLossPrice']=str(stop_price)
                order['presetStopSurplusPrice']=str(take_profit)
            result=await self.router.place_order(exchange,order,testnet=self.execution_mode=='testnet'); order_id=result.get('orderId') if isinstance(result,dict) else None
            if order_id and hasattr(adapter,'wait_for_fill'):
                detail=await adapter.wait_for_fill(symbol,order_id,float(os.getenv('HHHAI_ORDER_FILL_TIMEOUT_SECONDS','5'))); state=str(detail.get('state') or '').lower()
                if state not in {'filled','partially_filled'}:
                    if state in {'live','new'}:
                        try: await adapter.cancel_order(symbol,order_id)
                        except Exception: pass
                    return {'status':'not_executed','mode':self.execution_mode,'live_exchange_order':False,'exchange':exchange,'response':result,'fill_state':state or 'unknown','reason':'Exchange did not confirm a fill within the execution window.'}
                filled_qty=float(detail.get('baseVolume') or detail.get('size') or safe_qty); return {'status':'filled','mode':self.execution_mode,'live_exchange_order':self.execution_mode=='live','exchange':exchange,'response':result,'fill':detail,'protection':({'stop_loss':stop_price,'take_profit':take_profit,'attached':True,'quantity':filled_qty} if not test10 else {'attached':False,'reason':'Test 10 monitoring window'})}
            return {'status':'submitted','mode':self.execution_mode,'live_exchange_order':self.execution_mode=='live','exchange':exchange,'response':result,'protection':{'stop_loss':stop_price,'take_profit':take_profit,'attached':True}}
        adapter=self._adapters()[exchange]; position_mode=await adapter.get_position_mode(); position_side='LONG' if candidate.side=='long' else 'SHORT'; order={'symbol':symbol,'side':side.upper(),'type':'MARKET','quantity':safe_qty}
        if position_mode=='HEDGE': order['positionSide']=position_side
        result=await self.router.place_order(exchange,order,testnet=self.execution_mode=='testnet'); filled_qty=float(result.get('executedQty') or safe_qty) if isinstance(result,dict) else safe_qty; protection=await adapter.place_protection(symbol=symbol,side=candidate.side,quantity=filled_qty,stop_price=stop_price,take_profit=take_profit,position_mode=position_mode); return {'status':'submitted','mode':self.execution_mode,'live_exchange_order':self.execution_mode=='live','exchange':exchange,'response':result,'protection':protection}

    @staticmethod
    def _safe_quantity(value:float)->float: return max(0.001,math.floor(value*1000)/1000)

trader=AutonomousTrader()
