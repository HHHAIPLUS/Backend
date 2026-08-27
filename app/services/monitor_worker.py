from __future__ import annotations
import asyncio,logging,os
from datetime import datetime,timezone
from app.market_data.realtime import build_world_intelligence
from app.exchanges.factory import adapters
from ai.adaptive_engine import AdaptivePositionEngine
from ai.adaptive_models import PositionSnapshot, MarketObservation
from ai.position_management import PositionManager
from app.persistence.repository import record_event
from app.persistence.supabase import store

log=logging.getLogger('hhhai.monitor')

class MarketMonitor:
    def __init__(self):
        self.symbols=[x.strip().upper() for x in os.getenv('HHHAI_WATCH_SYMBOLS','BTCUSDT').split(',') if x.strip()]
        self.interval=max(5,int(os.getenv('HHHAI_MONITOR_INTERVAL_SECONDS','15')))
        self.latest={}; self.position_reviews=[]; self._stop=asyncio.Event(); self.adaptive=AdaptivePositionEngine(); self.manager=PositionManager()
    async def _review_positions(self, world):
        configured=[]
        for name,adapter in adapters().items():
            try:
                positions=await adapter.get_positions()
            except Exception:
                continue
            if not isinstance(positions,list): positions=positions.get('data',positions.get('positions',[])) if isinstance(positions,dict) else []
            for p in positions:
                symbol=str(p.get('symbol') or p.get('instId') or '').upper()
                raw_amt=p.get('positionAmt',p.get('total',p.get('available','0')))
                try: amount=float(raw_amt)
                except: continue
                if not symbol or abs(amount)<=0: continue
                side='long' if amount>0 else 'short'
                entry=float(p.get('entryPrice') or p.get('openPriceAvg') or p.get('avgOpenPrice') or 0)
                current=float(p.get('markPrice') or p.get('markPrice') or p.get('currentPrice') or 0)
                if entry<=0 or current<=0: continue
                ret=(current-entry)/entry if side=='long' else (entry-current)/entry
                snap=PositionSnapshot(symbol=symbol,side=side,entry_price=entry,current_price=current,unrealized_return=ret,peak_return=ret,opened_at=datetime.now(timezone.utc),confidence=0.5)
                m=world.market
                imbalance=m.order_book_imbalance
                obs=MarketObservation(momentum=world.momentum_proxy,trend_strength=world.trend_strength,buying_pressure=max(0,min(1,.5+imbalance/2)),selling_pressure=max(0,min(1,.5-imbalance/2)),volatility=min(1,m.volatility_proxy*5),liquidity_stress=world.liquidity_stress,news_risk=world.news_risk,market_risk=world.market_risk,thesis_integrity=.5)
                self.adaptive.register_position(snap,'Existing exchange position; thesis reconstructed from current evidence.')
                decision=self.adaptive.evaluate(snap,obs)
                management=self.manager.review(snap,obs,decision)
                row={'exchange':name,'symbol':symbol,'position':snap.model_dump(mode='json'),'adaptive_decision':decision.model_dump(mode='json'),'management_decision':management.model_dump(mode='json'),'reviewed_at':datetime.now(timezone.utc).isoformat()}
                self.position_reviews.append(row); self.position_reviews=self.position_reviews[-100:]
                if store.configured:
                    try: await record_event('position_review',row)
                    except Exception: pass
    async def run(self):
        log.info('HHHAI observation worker started for %s',self.symbols)
        while not self._stop.is_set():
            for symbol in self.symbols:
                try:
                    world=await asyncio.to_thread(build_world_intelligence,symbol)
                    self.latest[symbol]=world.model_dump(mode='json')
                    if store.configured:
                        try: await record_event('market_observation',self.latest[symbol])
                        except Exception: pass
                    await self._review_positions(world)
                except Exception as exc: log.warning('Observation failed for %s: %s',symbol,exc)
            try: await asyncio.wait_for(self._stop.wait(),timeout=self.interval)
            except asyncio.TimeoutError: pass
        log.info('HHHAI observation worker stopped')
    def stop(self): self._stop.set()
monitor=MarketMonitor()
