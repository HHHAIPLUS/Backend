from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from statistics import pstdev
from typing import Any

try:
    from websockets.sync.client import connect as websocket_connect
except ImportError:
    websocket_connect = None

from app.market_data.realtime import FeedHealth, RealtimeSnapshot
from app.ml.features import build_model_features

log = logging.getLogger("hhhai.binance_central")

WS_BASE = "wss://fstream.binance.com/stream"
MAX_CANDLES = 30
STALE_SECONDS = 15


class _SymbolState:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.last_update = 0.0
        self.last_error: str | None = None
        self.price = 0.0
        self.bid = 0.0
        self.ask = 0.0
        self.bid_qty = 0.0
        self.ask_qty = 0.0
        self.depth_bid_qty = 0.0
        self.depth_ask_qty = 0.0
        self.volume_24h = 0.0
        self.price_change_24h = 0.0
        self.funding_rate: float | None = None
        self.open_interest: float | None = None
        self.previous_open_interest: float | None = None
        self.candles: deque[list[float]] = deque(maxlen=MAX_CANDLES)
        self.current_candle: list[float] | None = None


class CentralBinanceMarketData:
    """Single in-process source of truth for live Binance market features.

    Live market data comes from Binance Futures WebSocket streams.  No Binance
    public REST request is made by this component.  The same cached state feeds
    world intelligence, predictive features and universe selection.
    """

    _states: dict[str, _SymbolState] = {}
    _states_lock = threading.Lock()
    _universe: dict[str, dict[str, float]] = {}
    _universe_updated_at = 0.0
    _universe_thread: threading.Thread | None = None
    _universe_stop = threading.Event()

    @classmethod
    def _state(cls, symbol: str) -> _SymbolState:
        symbol = symbol.upper()
        with cls._states_lock:
            state = cls._states.get(symbol)
            if state is None:
                state = _SymbolState(symbol)
                cls._states[symbol] = state
        cls._start_symbol(state)
        return state

    @classmethod
    def _start_symbol(cls, state: _SymbolState) -> None:
        if websocket_connect is None:
            state.last_error = "The 'websockets' package is not installed."
            return
        if state.thread and state.thread.is_alive():
            return
        state.stop.clear()
        state.thread = threading.Thread(
            target=cls._run_symbol,
            args=(state,),
            name=f"binance-central-{state.symbol}",
            daemon=True,
        )
        state.thread.start()

    @classmethod
    def _run_symbol(cls, state: _SymbolState) -> None:
        delay = 2.0
        streams = "/".join(
            [
                f"{state.symbol.lower()}@ticker",
                f"{state.symbol.lower()}@markPrice@1s",
                f"{state.symbol.lower()}@depth5@100ms",
                f"{state.symbol.lower()}@kline_5m",
            ]
        )
        url = f"{WS_BASE}?streams={streams}"
        while not state.stop.is_set():
            try:
                with websocket_connect(
                    url,
                    proxy=None,
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=60,
                    max_size=2**20,
                ) as websocket:
                    delay = 2.0
                    with state.lock:
                        state.last_error = None
                    while not state.stop.is_set():
                        try:
                            raw = websocket.recv(timeout=30)
                            if raw is None:
                                raise RuntimeError("Binance WebSocket returned no message")
                            cls._process(state, raw)
                        except TimeoutError:
                            continue
            except Exception as exc:
                with state.lock:
                    state.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("Central Binance WebSocket error for %s: %s", state.symbol, exc)
            if state.stop.is_set():
                break
            state.stop.wait(delay)
            delay = min(delay * 2.0, 30.0)

    @classmethod
    def _process(cls, state: _SymbolState, raw: str) -> None:
        payload = json.loads(raw)
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            return
        event = data.get("e")
        with state.lock:
            if event == "24hrTicker":
                state.price = float(data.get("c") or state.price or 0)
                state.bid = float(data.get("b") or state.bid or state.price or 0)
                state.ask = float(data.get("a") or state.ask or state.price or 0)
                state.bid_qty = float(data.get("B") or 0)
                state.ask_qty = float(data.get("A") or 0)
                state.volume_24h = float(data.get("q") or 0)
                state.price_change_24h = float(data.get("P") or 0) / 100.0
                state.last_update = time.time()
                state.ready.set()
            elif event == "depthUpdate":
                bids = data.get("b") or []
                asks = data.get("a") or []
                state.depth_bid_qty = sum(float(row[1]) for row in bids if len(row) >= 2)
                state.depth_ask_qty = sum(float(row[1]) for row in asks if len(row) >= 2)
                state.last_update = time.time()
            elif event == "markPriceUpdate":
                state.funding_rate = float(data.get("r") or 0)
                state.last_update = time.time()
            elif event == "kline":
                k = data.get("k") or {}
                candle = [
                    float(k.get("t") or 0),
                    float(k.get("o") or 0),
                    float(k.get("h") or 0),
                    float(k.get("l") or 0),
                    float(k.get("c") or 0),
                    float(k.get("v") or 0),
                ]
                state.current_candle = candle
                if bool(k.get("x")) and candle[4] > 0:
                    if not state.candles or state.candles[-1][0] != candle[0]:
                        state.candles.append(candle)
                    else:
                        state.candles[-1] = candle
                state.last_update = time.time()
                state.ready.set()

    @classmethod
    def snapshot(cls, symbol: str) -> RealtimeSnapshot:
        state = cls._state(symbol)
        if not state.ready.wait(timeout=3):
            raise RuntimeError(f"Binance WebSocket data not ready for {symbol.upper()}")
        with state.lock:
            age = time.time() - state.last_update
            if age > STALE_SECONDS or state.price <= 0:
                raise RuntimeError(f"Binance WebSocket market data is stale for {state.symbol}")
            bid = state.bid or state.price
            ask = state.ask or state.price
            bid_qty = state.depth_bid_qty or state.bid_qty
            ask_qty = state.depth_ask_qty or state.ask_qty
            imbalance = (bid_qty - ask_qty) / max(bid_qty + ask_qty, 1e-12)
            now = datetime.now(timezone.utc)
            health = FeedHealth(
                source="binance_futures_websocket_central",
                status="healthy",
                latency_ms=round(age * 1000, 2),
                observed_at=now,
                stale_after_seconds=STALE_SECONDS,
                error=state.last_error,
            )
            return RealtimeSnapshot(
                symbol=state.symbol,
                source="binance_futures_websocket_central",
                price=state.price,
                bid=bid,
                ask=ask,
                volume_24h=max(0.0, state.volume_24h),
                funding_rate=state.funding_rate,
                open_interest=state.open_interest,
                open_interest_change=(
                    state.open_interest / state.previous_open_interest - 1.0
                    if state.open_interest and state.previous_open_interest
                    else None
                ),
                order_book_imbalance=max(-1.0, min(1.0, imbalance)),
                volatility_proxy=abs(state.price_change_24h),
                price_change_24h=state.price_change_24h,
                observed_at=now,
                feed_health=health,
            )

    @classmethod
    def model_features(cls, symbol: str) -> dict[str, float]:
        state = cls._state(symbol)
        with state.lock:
            candles = list(state.candles)
            current = list(state.current_candle) if state.current_candle else None
            price_change = state.price_change_24h
        if len(candles) < 3:
            raise RuntimeError(
                f"Waiting for Binance WebSocket 5m candle history for {symbol.upper()}: "
                f"{len(candles)}/{MAX_CANDLES} closed candles cached"
            )
        rows = candles[-MAX_CANDLES:]
        if current and current[0] == rows[-1][0]:
            rows[-1] = current
        features = build_model_features(rows)
        required = ("return_1", "range_pct", "volume_change", "volatility_proxy", "trend_strength", "momentum")
        missing = [name for name in required if name not in features]
        if missing:
            raise RuntimeError("Central Binance feature builder missing: " + ", ".join(missing))
        result = {key: float(value) for key, value in features.items()}
        result.setdefault("return_1", price_change)
        return result

    @classmethod
    def _start_universe(cls) -> None:
        if websocket_connect is None:
            return
        with cls._states_lock:
            if cls._universe_thread and cls._universe_thread.is_alive():
                return
            cls._universe_stop.clear()
            cls._universe_thread = threading.Thread(
                target=cls._run_universe,
                name="binance-central-universe",
                daemon=True,
            )
            cls._universe_thread.start()

    @classmethod
    def _run_universe(cls) -> None:
        url = f"{WS_BASE}?streams=!ticker@arr"
        delay = 2.0
        while not cls._universe_stop.is_set():
            try:
                with websocket_connect(
                    url,
                    proxy=None,
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=60,
                    max_size=8 * 2**20,
                ) as websocket:
                    delay = 2.0
                    while not cls._universe_stop.is_set():
                        try:
                            raw = websocket.recv(timeout=30)
                            if raw is None:
                                raise RuntimeError("Binance universe WebSocket returned no message")
                            payload = json.loads(raw)
                            data = payload.get("data", payload)
                            events = data if isinstance(data, list) else [data]
                            updated = 0
                            with cls._states_lock:
                                for item in events:
                                    if not isinstance(item, dict):
                                        continue
                                    symbol = str(item.get("s") or "").upper()
                                    if not symbol.endswith("USDT") or "_" in symbol:
                                        continue
                                    price = float(item.get("c") or 0)
                                    volume = float(item.get("q") or 0)
                                    trades = float(item.get("n") or 0)
                                    high = float(item.get("h") or 0)
                                    low = float(item.get("l") or 0)
                                    if price <= 0:
                                        continue
                                    cls._universe[symbol] = {
                                        "price": price,
                                        "quote_volume": volume,
                                        "trade_count": trades,
                                        "range_pct": max(0.0, (high - low) / price),
                                        "updated_at": time.time(),
                                    }
                                    updated += 1
                                if updated:
                                    cls._universe_updated_at = time.time()
                        except TimeoutError:
                            continue
            except Exception as exc:
                log.warning("Central Binance universe WebSocket error: %s", exc)
            if cls._universe_stop.is_set():
                break
            cls._universe_stop.wait(delay)
            delay = min(delay * 2.0, 30.0)

    @classmethod
    def rank_symbols(cls, limit: int = 5) -> list[str]:
        cls._start_universe()
        now = time.time()
        with cls._states_lock:
            rows = list(cls._universe.items())
        fresh = [(symbol, row) for symbol, row in rows if now - float(row.get("updated_at", 0)) <= 30]
        ranked: list[tuple[float, str]] = []
        for symbol, row in fresh:
            score = (
                __import__("math").log1p(max(0.0, row["quote_volume"])) * 0.70
                + __import__("math").log1p(max(0.0, row["trade_count"])) * 0.20
                + min(row["range_pct"], 1.0) * 10.0 * 0.10
            )
            ranked.append((score, symbol))
        ranked.sort(reverse=True)
        return [symbol for _, symbol in ranked[: max(1, int(limit))]]

    @classmethod
    def install(cls) -> None:
        """Redirect HHHAI's Binance public market snapshot to the central cache."""
        from app.market_data import realtime
        if getattr(realtime.BinancePublicFeed, "_hhhai_central_installed", False):
            return
        realtime.BinancePublicFeed.snapshot = lambda self, symbol: cls.snapshot(symbol)
        realtime.BinancePublicFeed._hhhai_central_installed = True
        log.info("Installed centralized Binance WebSocket market-data source")
