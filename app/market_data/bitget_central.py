from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone

import httpx

try:
    from websockets.sync.client import connect as websocket_connect
except ImportError:
    websocket_connect = None

from app.market_data.realtime import FeedHealth, RealtimeSnapshot
from app.ml.features import build_model_features

log = logging.getLogger("hhhai.bitget_central")

WS_URL = "wss://ws.bitget.com/v2/ws/public"
REST_URL = "https://api.bitget.com"
MAX_CANDLES = 400
STALE_SECONDS = 15
BOOTSTRAP_INTERVAL = 300.0
REST_MIN_INTERVAL = 1.0
REST_COOLDOWN_SECONDS = 60.0
REST_MARKET_FALLBACK_INTERVAL = 5.0

class _State:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.lock = threading.Lock()
        self.thread = None
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.last_update = 0.0
        self.last_error = None
        self.price = 0.0
        self.mark_price = 0.0
        self.bid = 0.0
        self.ask = 0.0
        self.bid_qty = 0.0
        self.ask_qty = 0.0
        self.volume_24h = 0.0
        self.price_change_24h = 0.0
        self.funding_rate = None
        self.open_interest = None
        self.candles = deque(maxlen=MAX_CANDLES)
        self.current_candle = None
        self.last_bootstrap = 0.0
        self.rest_blocked_until = 0.0
        self.rest_last_request = 0.0
        self.rest_market_last_request = 0.0

class CentralBitgetMarketData:
    """Single in-process Bitget USDT-futures market-data source.

    WebSocket carries ticker, level-5 book and 1h candles. REST is used only
    for one-time/periodic candle bootstrap, not for continuous observation.
    """

    _states = {}
    _lock = threading.Lock()

    @classmethod
    def _state(cls, symbol: str) -> _State:
        symbol = symbol.upper()
        with cls._lock:
            state = cls._states.get(symbol)
            if state is None:
                state = _State(symbol)
                cls._states[symbol] = state
        cls._start(state)
        return state

    @classmethod
    def _start(cls, state: _State) -> None:
        if websocket_connect is None:
            state.last_error = "The 'websockets' package is not installed."
            return
        if not state.thread or not state.thread.is_alive():
            state.stop.clear()
            state.thread = threading.Thread(
                target=cls._run,
                args=(state,),
                name=f"bitget-market-{state.symbol}",
                daemon=True,
            )
            state.thread.start()

    @classmethod
    def _run(cls, state: _State) -> None:
        delay = 2.0
        while not state.stop.is_set():
            try:
                with websocket_connect(
                    WS_URL,
                    proxy=None,
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=None,
                    max_size=2**20,
                ) as ws:
                    delay = 2.0
                    subscribe = {
                        "op": "subscribe",
                        "args": [
                            {"instType": "USDT-FUTURES", "channel": "ticker", "instId": state.symbol},
                            {"instType": "USDT-FUTURES", "channel": "books5", "instId": state.symbol},
                            {"instType": "USDT-FUTURES", "channel": "candle1H", "instId": state.symbol},
                        ],
                    }
                    ws.send(json.dumps(subscribe, separators=(",", ":")))
                    next_ping = time.monotonic() + 25.0
                    while not state.stop.is_set():
                        timeout = max(1.0, min(5.0, next_ping - time.monotonic()))
                        try:
                            raw = ws.recv(timeout=timeout)
                            if raw is not None:
                                if raw == "pong":
                                    next_ping = time.monotonic() + 25.0
                                else:
                                    cls._process(state, raw)
                        except TimeoutError:
                            pass
                        if time.monotonic() >= next_ping:
                            ws.send("ping")
                            next_ping = time.monotonic() + 25.0
            except Exception as exc:
                with state.lock:
                    state.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("Bitget WebSocket error for %s: %s", state.symbol, exc)
            if state.stop.is_set():
                break
            state.stop.wait(delay)
            delay = min(delay * 2.0, 30.0)

    @classmethod
    def _process(cls, state: _State, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return
        if payload.get("event") in {"subscribe", "unsubscribe"}:
            return
        data = payload.get("data")
        arg = payload.get("arg") or {}
        channel = arg.get("channel")
        rows = data if isinstance(data, list) else ([data] if data else [])
        with state.lock:
            if channel == "ticker":
                if not rows:
                    return
                row = rows[0]
                state.price = float(row.get("lastPr") or state.price or 0)
                state.mark_price = float(row.get("markPrice") or state.mark_price or 0)
                state.bid = float(row.get("bidPr") or state.bid or 0)
                state.ask = float(row.get("askPr") or state.ask or 0)
                state.bid_qty = float(row.get("bidSz") or state.bid_qty or 0)
                state.ask_qty = float(row.get("askSz") or state.ask_qty or 0)
                state.volume_24h = float(row.get("quoteVolume") or 0)
                state.price_change_24h = float(row.get("change24h") or 0)
                state.funding_rate = float(row.get("fundingRate") or 0)
                state.open_interest = float(row.get("holdingAmount") or 0) or None
            elif channel == "books5":
                if not rows:
                    return
                row = rows[0]
                bids = row.get("b") or []
                asks = row.get("a") or []
                if bids:
                    state.bid = float(bids[0][0])
                    state.bid_qty = sum(float(x[1]) for x in bids if len(x) >= 2)
                if asks:
                    state.ask = float(asks[0][0])
                    state.ask_qty = sum(float(x[1]) for x in asks if len(x) >= 2)
            elif channel == "candle5m":
                for row in rows:
                    if not isinstance(row, list) or len(row) < 6:
                        continue
                    candle = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                    state.current_candle = candle
                    if not state.candles or state.candles[-1][0] != candle[0]:
                        state.candles.append(candle)
                    else:
                        state.candles[-1] = candle
            else:
                return
            state.last_update = time.time()
            state.ready.set()

    @classmethod
    def _bootstrap_candles(cls, state: _State) -> None:
        now = time.time()
        with state.lock:
            if now < state.rest_blocked_until:
                return
            if now - state.last_bootstrap < BOOTSTRAP_INTERVAL:
                return
            wait_for = REST_MIN_INTERVAL - (now - state.rest_last_request)
            if wait_for > 0:
                return
            state.rest_last_request = now
            state.last_bootstrap = now
        try:
            response = httpx.get(
                f"{REST_URL}/api/v2/mix/market/candles",
                params={"productType": "USDT-FUTURES", "symbol": state.symbol, "granularity": "1H", "limit": MAX_CANDLES},
                timeout=8.0,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    cooldown = max(REST_COOLDOWN_SECONDS, min(float(retry_after or REST_COOLDOWN_SECONDS), 900.0))
                except (TypeError, ValueError):
                    cooldown = REST_COOLDOWN_SECONDS
                with state.lock:
                    state.rest_blocked_until = time.time() + cooldown
                raise RuntimeError(f"Bitget candle bootstrap rate limited (429); cooldown {cooldown:.0f}s")
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("data") or []
            candles = []
            for row in reversed(rows):
                if isinstance(row, list) and len(row) >= 6:
                    candles.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
            with state.lock:
                state.candles.clear()
                state.candles.extend(candles[-MAX_CANDLES:])
        except Exception as exc:
            log.warning("Bitget candle bootstrap failed for %s: %s", state.symbol, exc)

    @classmethod
    def _rest_market_fallback(cls, state: _State) -> bool:
        now = time.time()
        with state.lock:
            if now - state.rest_market_last_request < REST_MARKET_FALLBACK_INTERVAL:
                return False
            state.rest_market_last_request = now
        try:
            response = httpx.get(
                f"{REST_URL}/api/v2/mix/market/ticker",
                params={"productType": "USDT-FUTURES", "symbol": state.symbol},
                timeout=5.0,
            )
            if response.status_code == 429:
                raise RuntimeError("Bitget ticker fallback rate limited (429)")
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") or {}
            row = data[0] if isinstance(data, list) and data else data
            if not row:
                return False
            with state.lock:
                state.price = float(row.get("lastPr") or state.price or 0)
                state.mark_price = float(row.get("markPrice") or state.mark_price or 0)
                state.bid = float(row.get("bidPr") or state.bid or 0)
                state.ask = float(row.get("askPr") or state.ask or 0)
                state.bid_qty = float(row.get("bidSz") or state.bid_qty or 0)
                state.ask_qty = float(row.get("askSz") or state.ask_qty or 0)
                state.volume_24h = float(row.get("quoteVolume") or state.volume_24h or 0)
                state.price_change_24h = float(row.get("change24h") or state.price_change_24h or 0)
                state.funding_rate = float(row.get("fundingRate") or state.funding_rate or 0)
                state.open_interest = float(row.get("holdingAmount") or state.open_interest or 0) or None
                state.last_update = time.time()
                state.ready.set()
            log.warning("Bitget WebSocket stale for %s; using rate-limited REST ticker fallback", state.symbol)
            return True
        except Exception as exc:
            log.warning("Bitget REST market fallback failed for %s: %s", state.symbol, exc)
            return False

    @classmethod
    def snapshot(cls, symbol: str) -> RealtimeSnapshot:
        state = cls._state(symbol)
        cls._bootstrap_candles(state)
        with state.lock:
            stale = (time.time() - state.last_update) > STALE_SECONDS or state.price <= 0
        if stale:
            cls._rest_market_fallback(state)
        if not state.ready.wait(timeout=5):
            raise RuntimeError(f"Bitget WebSocket data not ready for {symbol.upper()}")
        with state.lock:
            age = time.time() - state.last_update
            if age > STALE_SECONDS or state.price <= 0:
                raise RuntimeError(f"Bitget WebSocket market data is stale for {state.symbol}")
            bid = state.bid or state.price
            ask = state.ask or state.price
            imbalance = (state.bid_qty - state.ask_qty) / max(state.bid_qty + state.ask_qty, 1e-12)
            now = datetime.now(timezone.utc)
            return RealtimeSnapshot(
                symbol=state.symbol,
                source="bitget_futures_websocket_central",
                price=state.mark_price or state.price,
                bid=bid,
                ask=ask,
                volume_24h=max(0.0, state.volume_24h),
                funding_rate=state.funding_rate,
                open_interest=state.open_interest,
                open_interest_change=None,
                order_book_imbalance=max(-1.0, min(1.0, imbalance)),
                volatility_proxy=abs(state.price_change_24h),
                price_change_24h=state.price_change_24h,
                observed_at=now,
                feed_health=FeedHealth(
                    source="bitget_futures_websocket_central",
                    status="healthy",
                    latency_ms=round(age * 1000, 2),
                    observed_at=now,
                    stale_after_seconds=STALE_SECONDS,
                    error=state.last_error,
                ),
            )

    @classmethod
    def model_features(cls, symbol: str) -> dict[str, float]:
        state = cls._state(symbol)
        cls._bootstrap_candles(state)
        with state.lock:
            candles = list(state.candles)
            current = list(state.current_candle) if state.current_candle else None
        if len(candles) < 3:
            raise RuntimeError(f"Waiting for Bitget 1h candle history for {symbol.upper()}: {len(candles)}/{MAX_CANDLES}")
        rows = candles[-MAX_CANDLES:]
        # The exchange candle stream can contain the still-forming candle. Do not
        # feed an open candle into the predictive model; training uses closed bars.
        if current and rows and current[0] == rows[-1][0]:
            rows = rows[:-1]
        if len(rows) < 25:
            raise RuntimeError(f"Insufficient closed Bitget candles for {symbol.upper()}: {len(rows)}")
        features = build_model_features(rows)
        required = ("return_1", "range_pct", "volume_change", "volatility_proxy", "trend_strength", "momentum")
        missing = [name for name in required if name not in features]
        if missing:
            raise RuntimeError("Central Bitget feature builder missing: " + ", ".join(missing))
        return {key: float(value) for key, value in features.items()}

    @classmethod
    def rank_symbols(cls, limit: int = 5) -> list[str]:
        # Keep the live selector deterministic and rate-limit safe. Symbols are
        # configured explicitly for live execution; no repeated exchange-wide REST scan.
        explicit = [x.strip().upper() for x in __import__("os").getenv("HHHAI_TRADE_SYMBOLS", "BTCUSDT").split(",") if x.strip()]
        return explicit[:max(1, int(limit))]
