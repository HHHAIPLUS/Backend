from __future__ import annotations

import json, logging, math, threading, time
from collections import deque
from datetime import datetime, timezone
from typing import Any
try:
    from websockets.sync.client import connect as websocket_connect
except ImportError:
    websocket_connect = None
from app.market_data.realtime import FeedHealth, RealtimeSnapshot
from app.ml.features import build_model_features

log=logging.getLogger("hhhai.binance_central")
MARKET_BASE="wss://fstream.binance.com/market"
PUBLIC_BASE="wss://fstream.binance.com/public"
MAX_CANDLES=30
STALE_SECONDS=15

class _SymbolState:
    def __init__(self,symbol:str)->None:
        self.symbol=symbol; self.lock=threading.Lock(); self.market_thread=None; self.book_thread=None; self.market_stop=threading.Event(); self.book_stop=threading.Event(); self.ready=threading.Event(); self.last_update=0.0; self.last_error=None; self.price=0.0; self.mark_price=0.0; self.bid=0.0; self.ask=0.0; self.bid_qty=0.0; self.ask_qty=0.0; self.volume_24h=0.0; self.price_change_24h=0.0; self.funding_rate=None; self.open_interest=None; self.previous_open_interest=None; self.candles=deque(maxlen=MAX_CANDLES); self.current_candle=None

class CentralBinanceMarketData:
    """Single in-process source of truth for live Binance market features."""
    _states={}; _states_lock=threading.Lock(); _universe={}; _universe_updated_at=0.0; _universe_thread=None; _universe_stop=threading.Event()
    @classmethod
    def _state(cls,symbol:str):
        symbol=symbol.upper()
        with cls._states_lock:
            state=cls._states.get(symbol)
            if state is None: state=_SymbolState(symbol); cls._states[symbol]=state
        cls._start_symbol(state); return state
    @classmethod
    def _start_symbol(cls,state):
        if websocket_connect is None: state.last_error="The 'websockets' package is not installed."; return
        if not state.market_thread or not state.market_thread.is_alive():
            state.market_stop.clear(); state.market_thread=threading.Thread(target=cls._run_market,args=(state,),name=f"binance-market-{state.symbol}",daemon=True); state.market_thread.start()
        if not state.book_thread or not state.book_thread.is_alive():
            state.book_stop.clear(); state.book_thread=threading.Thread(target=cls._run_book,args=(state,),name=f"binance-book-{state.symbol}",daemon=True); state.book_thread.start()
    @classmethod
    def _run_market(cls,state):
        streams="/".join([f"{state.symbol.lower()}@ticker",f"{state.symbol.lower()}@markPrice@1s",f"{state.symbol.lower()}@kline_5m"]); url=f"{MARKET_BASE}/stream?streams={streams}"; delay=2.0
        while not state.market_stop.is_set():
            try:
                with websocket_connect(url,proxy=None,open_timeout=10,close_timeout=5,ping_interval=20,ping_timeout=60,max_size=2**20) as websocket:
                    delay=2.0
                    while not state.market_stop.is_set():
                        try: raw=websocket.recv(timeout=30)
                        except TimeoutError: continue
                        if raw is None: raise RuntimeError("Binance market WebSocket returned no message")
                        cls._process_market(state,raw)
            except Exception as exc:
                with state.lock: state.last_error=f"{type(exc).__name__}: {exc}"
                log.warning("Central Binance market WebSocket error for %s: %s",state.symbol,exc)
            if state.market_stop.is_set(): break
            state.market_stop.wait(delay); delay=min(delay*2.0,30.0)
    @classmethod
    def _run_book(cls,state):
        url=f"{PUBLIC_BASE}/ws/{state.symbol.lower()}@bookTicker"; delay=2.0
        while not state.book_stop.is_set():
            try:
                with websocket_connect(url,proxy=None,open_timeout=10,close_timeout=5,ping_interval=20,ping_timeout=60,max_size=512*1024) as websocket:
                    delay=2.0
                    while not state.book_stop.is_set():
                        try: raw=websocket.recv(timeout=30)
                        except TimeoutError: continue
                        if raw is None: raise RuntimeError("Binance book WebSocket returned no message")
                        cls._process_book(state,raw)
            except Exception as exc:
                with state.lock: state.last_error=f"{type(exc).__name__}: {exc}"
                log.warning("Central Binance book WebSocket error for %s: %s",state.symbol,exc)
            if state.book_stop.is_set(): break
            state.book_stop.wait(delay); delay=min(delay*2.0,30.0)
    @classmethod
    def _process_market(cls,state,raw):
        payload=json.loads(raw); data=payload.get("data",payload)
        if not isinstance(data,dict): return
        event=data.get("e")
        with state.lock:
            if event=="24hrTicker":
                state.price=float(data.get("c") or state.price or 0); state.volume_24h=float(data.get("q") or 0); state.price_change_24h=float(data.get("P") or 0)/100.0
            elif event=="markPriceUpdate":
                state.mark_price=float(data.get("p") or state.mark_price or 0); state.funding_rate=float(data.get("r") or 0)
            elif event=="kline":
                k=data.get("k") or {}; candle=[float(k.get("t") or 0),float(k.get("o") or 0),float(k.get("h") or 0),float(k.get("l") or 0),float(k.get("c") or 0),float(k.get("v") or 0)]; state.current_candle=candle
                if bool(k.get("x")) and candle[4]>0:
                    if not state.candles or state.candles[-1][0]!=candle[0]: state.candles.append(candle)
                    else: state.candles[-1]=candle
            state.last_update=time.time(); state.ready.set()
    @classmethod
    def _process_book(cls,state,raw):
        data=json.loads(raw); data=data.get("data",data)
        if not isinstance(data,dict): return
        with state.lock:
            state.bid=float(data.get("b") or state.bid or 0); state.ask=float(data.get("a") or state.ask or 0); state.bid_qty=float(data.get("B") or state.bid_qty or 0); state.ask_qty=float(data.get("A") or state.ask_qty or 0); state.last_update=time.time(); state.ready.set()
    @classmethod
    def snapshot(cls,symbol:str)->RealtimeSnapshot:
        state=cls._state(symbol)
        if not state.ready.wait(timeout=3): raise RuntimeError(f"Binance WebSocket data not ready for {symbol.upper()}")
        with state.lock:
            age=time.time()-state.last_update
            if age>STALE_SECONDS or state.price<=0: raise RuntimeError(f"Binance WebSocket market data is stale for {state.symbol}")
            bid=state.bid or state.price; ask=state.ask or state.price; imbalance=(state.bid_qty-state.ask_qty)/max(state.bid_qty+state.ask_qty,1e-12); now=datetime.now(timezone.utc)
            health=FeedHealth(source="binance_futures_websocket_central",status="healthy",latency_ms=round(age*1000,2),observed_at=now,stale_after_seconds=STALE_SECONDS,error=state.last_error)
            return RealtimeSnapshot(symbol=state.symbol,source="binance_futures_websocket_central",price=state.mark_price or state.price,bid=bid,ask=ask,volume_24h=max(0.0,state.volume_24h),funding_rate=state.funding_rate,open_interest=state.open_interest,open_interest_change=(state.open_interest/state.previous_open_interest-1.0 if state.open_interest and state.previous_open_interest else None),order_book_imbalance=max(-1.0,min(1.0,imbalance)),volatility_proxy=abs(state.price_change_24h),price_change_24h=state.price_change_24h,observed_at=now,feed_health=health)
    @classmethod
    def model_features(cls,symbol:str)->dict[str,float]:
        state=cls._state(symbol)
        with state.lock: candles=list(state.candles); current=list(state.current_candle) if state.current_candle else None; price_change=state.price_change_24h
        if len(candles)<3: raise RuntimeError(f"Waiting for Binance WebSocket 5m candle history for {symbol.upper()}: {len(candles)}/{MAX_CANDLES} closed candles cached")
        rows=candles[-MAX_CANDLES:]
        if current and current[0]==rows[-1][0]: rows[-1]=current
        features=build_model_features(rows); required=("return_1","range_pct","volume_change","volatility_proxy","trend_strength","momentum"); missing=[name for name in required if name not in features]
        if missing: raise RuntimeError("Central Binance feature builder missing: "+", ".join(missing))
        result={key:float(value) for key,value in features.items()}; result.setdefault("return_1",price_change); return result
    @classmethod
    def _start_universe(cls):
        if websocket_connect is None: return
        with cls._states_lock:
            if cls._universe_thread and cls._universe_thread.is_alive(): return
            cls._universe_stop.clear(); cls._universe_thread=threading.Thread(target=cls._run_universe,name="binance-central-universe",daemon=True); cls._universe_thread.start()
    @classmethod
    def _run_universe(cls):
        url=f"{MARKET_BASE}/stream?streams=!ticker@arr"; delay=2.0
        while not cls._universe_stop.is_set():
            try:
                with websocket_connect(url,proxy=None,open_timeout=10,close_timeout=5,ping_interval=20,ping_timeout=60,max_size=8*2**20) as websocket:
                    delay=2.0
                    while not cls._universe_stop.is_set():
                        try: raw=websocket.recv(timeout=30)
                        except TimeoutError: continue
                        if raw is None: raise RuntimeError("Binance universe WebSocket returned no message")
                        payload=json.loads(raw); data=payload.get("data",payload); events=data if isinstance(data,list) else [data]
                        with cls._states_lock:
                            for item in events:
                                if not isinstance(item,dict): continue
                                symbol=str(item.get("s") or "").upper(); price=float(item.get("c") or 0); volume=float(item.get("q") or 0); trades=float(item.get("n") or 0); high=float(item.get("h") or 0); low=float(item.get("l") or 0)
                                if not symbol.endswith("USDT") or "_" in symbol or price<=0: continue
                                cls._universe[symbol]={"price":price,"quote_volume":volume,"trade_count":trades,"range_pct":max(0.0,(high-low)/price),"updated_at":time.time()}
                            cls._universe_updated_at=time.time()
            except Exception as exc: log.warning("Central Binance universe WebSocket error: %s",exc)
            if cls._universe_stop.is_set(): break
            cls._universe_stop.wait(delay); delay=min(delay*2.0,30.0)
    @classmethod
    def rank_symbols(cls,limit:int=5)->list[str]:
        cls._start_universe(); now=time.time()
        with cls._states_lock: rows=list(cls._universe.items())
        ranked=[]
        for symbol,row in rows:
            if now-float(row.get("updated_at",0))>30: continue
            score=math.log1p(max(0.0,row["quote_volume"]))*0.70+math.log1p(max(0.0,row["trade_count"]))*0.20+min(row["range_pct"],1.0)*10.0*0.10; ranked.append((score,symbol))
        ranked.sort(reverse=True); return [symbol for _,symbol in ranked[:max(1,int(limit))]]
    @classmethod
    def install(cls):
        from app.market_data import realtime
        if getattr(realtime.BinancePublicFeed,"_hhhai_central_installed",False): return
        realtime.BinancePublicFeed.snapshot=lambda self,symbol: cls.snapshot(symbol); realtime.BinancePublicFeed._hhhai_central_installed=True; log.info("Installed centralized Binance WebSocket market-data source")
