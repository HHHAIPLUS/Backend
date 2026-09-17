from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any

try:
    from websockets.sync.client import connect as websocket_connect
except ImportError:
    websocket_connect = None

from app.core.config import settings

log = logging.getLogger("hhhai.binance_user_stream")


class BinanceUserDataStream:
    """Central authenticated Binance Futures account/order/position state.

    Position management reads this cache instead of repeatedly polling
    positionRisk. REST remains an execution/reconciliation mechanism, not the
    live position feed.
    """

    def __init__(self, testnet: bool = False) -> None:
        self.testnet = bool(testnet)
        self.api_key = settings.binance_api_key
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._last_event_at = 0.0
        self._last_error: str | None = None
        self._last_event_type: str | None = None
        self._listen_key: str | None = None
        self._positions: dict[str, dict[str, Any]] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._balances: dict[str, dict[str, Any]] = {}
        self._account_config: dict[str, Any] = {}

    @property
    def stream_base(self) -> str:
        return "wss://testnet.binancefuture.com" if self.testnet else "wss://fstream.binance.com"

    @property
    def api_ws(self) -> str:
        return "wss://testnet.binancefuture.com/ws-fapi/v1" if self.testnet else "wss://ws-fapi.binance.com/ws-fapi/v1"

    def start(self) -> None:
        if not self.api_key:
            self._last_error = "Binance API key is not configured"
            return
        if websocket_connect is None:
            self._last_error = "The 'websockets' package is not installed"
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="binance-user-data", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _api_call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if websocket_connect is None:
            raise RuntimeError("websockets package unavailable")
        request_id = str(uuid.uuid4())
        payload = {
            "id": request_id,
            "method": method,
            "params": {"apiKey": self.api_key, **(params or {})},
        }
        with websocket_connect(
            self.api_ws,
            proxy=None,
            open_timeout=10,
            close_timeout=5,
            ping_interval=20,
            ping_timeout=60,
            max_size=2**20,
        ) as ws:
            ws.send(json.dumps(payload))
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                raw = ws.recv(timeout=5)
                if raw is None:
                    continue
                response = json.loads(raw)
                if response.get("id") != request_id:
                    continue
                if int(response.get("status") or 0) != 200:
                    error = response.get("error") or {}
                    raise RuntimeError(f"Binance user-data API {error.get('code')}: {error.get('msg')}")
                return response.get("result") or {}
        raise TimeoutError(f"Timed out waiting for Binance {method}")

    def _start_listen_key(self) -> str:
        result = self._api_call("userDataStream.start")
        key = str(result.get("listenKey") or "")
        if not key:
            raise RuntimeError("Binance did not return a listenKey")
        return key

    def _keepalive(self) -> None:
        try:
            result = self._api_call("userDataStream.ping")
            key = result.get("listenKey")
            if key:
                with self._lock:
                    self._listen_key = str(key)
        except Exception as exc:
            log.warning("Binance user-data keepalive failed: %s", exc)
            with self._lock:
                self._last_error = f"keepalive: {type(exc).__name__}: {exc}"

    def _run(self) -> None:
        reconnect_delay = 5.0
        next_keepalive = 0.0
        connected_at = 0.0
        while not self._stop.is_set():
            try:
                listen_key = self._start_listen_key()
                with self._lock:
                    self._listen_key = listen_key
                    self._last_error = None
                url = f"{self.stream_base}/ws/{listen_key}"
                connected_at = time.monotonic()
                next_keepalive = connected_at + 45 * 60
                log.info("Binance Futures user-data WebSocket connected")
                with websocket_connect(
                    url,
                    proxy=None,
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=60,
                    max_size=4 * 2**20,
                ) as ws:
                    reconnect_delay = 5.0
                    while not self._stop.is_set():
                        if time.monotonic() >= next_keepalive:
                            self._keepalive()
                            next_keepalive = time.monotonic() + 45 * 60
                        if time.monotonic() - connected_at >= 23 * 60 * 60:
                            log.info("Rotating Binance user-data WebSocket before 24-hour connection limit")
                            break
                        try:
                            raw = ws.recv(timeout=5)
                        except TimeoutError:
                            continue
                        if raw is None:
                            raise RuntimeError("Binance user-data WebSocket returned no message")
                        self._process(raw)
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                log.warning("Binance user-data WebSocket error: %s", exc)
            if self._stop.is_set():
                break
            self._stop.wait(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2.0, 60.0)

    def _process(self, raw: str) -> None:
        payload = json.loads(raw)
        event = payload.get("e")
        event_time = float(payload.get("E") or payload.get("T") or 0) / 1000.0
        with self._lock:
            self._last_event_at = max(self._last_event_at, event_time or time.time())
            self._last_event_type = str(event or "unknown")
            self._last_error = None
            if event == "ACCOUNT_UPDATE":
                self._apply_account_update(payload)
            elif event == "ORDER_TRADE_UPDATE":
                self._apply_order_update(payload)
            elif event == "ACCOUNT_CONFIG_UPDATE":
                self._account_config.update(payload.get("ac") or {})
            elif event == "listenKeyExpired":
                self._ready.clear()
                raise RuntimeError("Binance user-data listenKey expired")
            elif event == "MARGIN_CALL":
                log.warning("Binance user-data margin-call event received")
            self._ready.set()

    def _apply_account_update(self, payload: dict[str, Any]) -> None:
        account = payload.get("a") or {}
        for balance in account.get("B") or []:
            asset = str(balance.get("a") or "").upper()
            if asset:
                self._balances[asset] = dict(balance)
        # ACCOUNT_UPDATE position payloads are the authoritative event state.
        # A funding-only event can contain only the affected position, so replace
        # only symbols included in the event and preserve unrelated positions.
        for position in account.get("P") or []:
            symbol = str(position.get("s") or "").upper()
            if not symbol:
                continue
            raw_qty = float(position.get("pa") or 0)
            position_side = str(position.get("ps") or "BOTH").upper()
            if position_side == "BOTH":
                side = "long" if raw_qty > 0 else "short"
            else:
                side = "long" if position_side == "LONG" else "short"
            qty = abs(raw_qty)
            key = f"{symbol}:{side}"
            if qty <= 0:
                self._positions.pop(key, None)
                continue
            self._positions[key] = {
                "symbol": symbol,
                "side": side,
                "holdSide": side,
                "positionSide": position_side,
                "positionAmt": raw_qty,
                "quantity": qty,
                "entryPrice": float(position.get("ep") or 0),
                "breakEvenPrice": float(position.get("bep") or 0),
                "unrealizedProfit": float(position.get("up") or 0),
                "marginType": position.get("mt"),
                "isolatedWallet": float(position.get("iw") or 0),
                "eventTime": payload.get("E"),
            }

    def _apply_order_update(self, payload: dict[str, Any]) -> None:
        order = dict(payload.get("o") or {})
        symbol = str(order.get("s") or "").upper()
        order_id = str(order.get("i") or "")
        if not symbol or not order_id:
            return
        order["symbol"] = symbol
        order["orderId"] = order_id
        order["stopPrice"] = float(order.get("sp") or 0)
        order["status"] = order.get("X")
        order["executionType"] = order.get("x")
        key = f"{symbol}:{order_id}"
        status = str(order.get("X") or "")
        if status in {"CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "FILLED"}:
            self._orders.pop(key, None)
        else:
            self._orders[key] = order

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = [dict(row) for row in self._positions.values()]
        if symbol:
            symbol = symbol.upper()
            rows = [row for row in rows if row.get("symbol") == symbol]
        return rows

    def protection_orders(self, symbol: str) -> list[dict[str, Any]]:
        symbol = symbol.upper()
        with self._lock:
            return [dict(row) for row in self._orders.values() if row.get("symbol") == symbol and str(row.get("type") or row.get("o") or "").upper() in {"STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT", "TRAILING_STOP_MARKET"}]

    def is_healthy(self, max_age_seconds: float = 30.0) -> bool:
        with self._lock:
            return self._ready.is_set() and self._last_event_at > 0 and time.time() - self._last_event_at <= max_age_seconds and not self._last_error

    def health(self) -> dict[str, Any]:
        with self._lock:
            age = time.time() - self._last_event_at if self._last_event_at else None
            return {
                "connected_state": self._ready.is_set(),
                "healthy": bool(self._ready.is_set() and age is not None and age <= 30 and not self._last_error),
                "last_event_type": self._last_event_type,
                "last_event_age_seconds": age,
                "last_error": self._last_error,
                "positions": len(self._positions),
                "open_orders": len(self._orders),
            }


binance_user_stream = BinanceUserDataStream(testnet=settings.binance_testnet)
