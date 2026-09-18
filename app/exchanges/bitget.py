from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import threading
import time
import uuid
from collections import deque
from urllib.parse import urlencode

import httpx

from app.exchanges.base import ExchangeAdapter
from app.core.config import settings


class BitgetAdapter(ExchangeAdapter):
    name = "bitget"

    # Deliberately conservative global limiter. Bitget documents 6000 requests/IP/min
    # overall, plus endpoint-specific limits. HHHAI stays far below those ceilings.
    _rate_lock = threading.Lock()
    _request_times = deque()
    _max_requests_per_second = 2
    _max_requests_per_minute = 120
    _blocked_until = 0.0

    def __init__(self, testnet: bool | None = None):
        self.key = settings.bitget_api_key
        self.secret = settings.bitget_api_secret
        self.passphrase = settings.bitget_passphrase
        self.testnet = settings.bitget_testnet if testnet is None else testnet
        self.base = settings.bitget_testnet_url if self.testnet else settings.bitget_url

    @classmethod
    def _wait_for_rate_slot(cls) -> None:
        while True:
            now = time.monotonic()
            with cls._rate_lock:
                if now < cls._blocked_until:
                    sleep_for = cls._blocked_until - now
                else:
                    while cls._request_times and now - cls._request_times[0] >= 60:
                        cls._request_times.popleft()
                    recent_second = sum(1 for t in cls._request_times if now - t < 1)
                    if len(cls._request_times) < cls._max_requests_per_minute and recent_second < cls._max_requests_per_second:
                        cls._request_times.append(now)
                        return
                    next_times = []
                    if recent_second >= cls._max_requests_per_second:
                        next_times.append(min(t for t in cls._request_times if now - t < 1) + 1 - now)
                    if len(cls._request_times) >= cls._max_requests_per_minute:
                        next_times.append(cls._request_times[0] + 60 - now)
                    sleep_for = max(0.05, min(next_times or [0.5]))
            time.sleep(max(0.05, sleep_for))

    @classmethod
    def _block_after_429(cls, retry_after: str | None) -> None:
        try:
            delay = float(retry_after or 300)
        except (TypeError, ValueError):
            delay = 300.0
        with cls._rate_lock:
            cls._blocked_until = max(cls._blocked_until, time.monotonic() + min(max(delay, 60.0), 900.0))

    def _headers(self, method, path, query="", body=""):
        if not self.key or not self.secret or not self.passphrase:
            raise RuntimeError("Bitget credentials are not configured")
        ts = str(int(time.time() * 1000))
        qs = f"?{query}" if query else ""
        pre = ts + method.upper() + path + qs + body
        sign = base64.b64encode(hmac.new(self.secret.encode(), pre.encode(), hashlib.sha256).digest()).decode()
        return {
            "ACCESS-KEY": self.key,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US",
        }

    async def _request(self, method, path, params=None, body=None, private=False):
        params = params or {}
        body_text = json.dumps(body, separators=(",", ":")) if body else ""
        query = urlencode(params)
        headers = self._headers(method, path, query, body_text) if private else {"Content-Type": "application/json", "locale": "en-US"}
        if self.testnet:
            headers["paptrading"] = "1"

        await asyncio.to_thread(self._wait_for_rate_slot)

        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method,
                self.base + path,
                params=params,
                json=body,
                headers=headers,
            )
            if response.status_code == 429:
                self._block_after_429(response.headers.get("Retry-After"))
                raise RuntimeError("Bitget REST rate limit reached; HHHAI execution is blocked until cooldown expires")
            response.raise_for_status()
            data = response.json()
            if data.get("code") not in (None, "00000", 0):
                raise RuntimeError(f"Bitget API error: {data.get('code')} {data.get('msg')}")
            return data.get("data", data)

    async def get_account_status(self):
        accounts = await self._request("GET", "/api/v2/mix/account/accounts", {"productType": "USDT-FUTURES"}, private=True)
        return {"exchange": self.name, "testnet": self.testnet, "accounts": accounts}

    async def get_positions(self, symbol=None):
        p = {"productType": "USDT-FUTURES"}
        if symbol:
            p["symbol"] = symbol.upper()
        return await self._request("GET", "/api/v2/mix/position/all-position", p, private=True)

    async def get_ticker(self, symbol):
        return await self._request("GET", "/api/v2/mix/market/ticker", {"productType": "USDT-FUTURES", "symbol": symbol.upper()})

    async def place_order(self, order):
        payload = {"productType": "USDT-FUTURES", **order}
        payload.setdefault("clientOid", f"HHHAI-{uuid.uuid4().hex[:20]}")
        return await self._request("POST", "/api/v2/mix/order/place-order", body=payload, private=True)

    async def get_order_detail(self, symbol, order_id=None, client_oid=None):
        params = {"productType": "USDT-FUTURES", "symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = str(order_id)
        elif client_oid:
            params["clientOid"] = str(client_oid)
        else:
            raise ValueError("order_id or client_oid is required")
        return await self._request("GET", "/api/v2/mix/order/detail", params=params, private=True)

    async def wait_for_fill(self, symbol, order_id, timeout_seconds=5.0):
        deadline = time.monotonic() + max(0.5, timeout_seconds)
        latest = None
        while time.monotonic() < deadline:
            latest = await self.get_order_detail(symbol, order_id=order_id)
            state = str((latest or {}).get("state", "")).lower()
            if state in {"filled", "canceled", "cancelled"}:
                break
            await asyncio.sleep(0.35)
        return latest or {}

    async def cancel_order(self, symbol, order_id):
        return await self._request(
            "POST",
            "/api/v2/mix/order/cancel-order",
            body={"productType": "USDT-FUTURES", "symbol": symbol.upper(), "orderId": str(order_id)},
            private=True,
        )

    async def close_position(self, symbol, side, quantity, position_mode="ONE_WAY"):
        close_side = "buy" if side.lower() == "short" else "sell"
        payload = {
            "symbol": symbol.upper(),
            "marginCoin": "USDT",
            "size": str(quantity),
            "side": close_side,
            "orderType": "market",
            "marginMode": "crossed",
            "reduceOnly": "YES",
        }
        if position_mode == "HEDGE":
            payload["tradeSide"] = "close"
        return await self.place_order(payload)

    async def get_position_mode(self, symbol="BTCUSDT"):
        account = await self._request(
            "GET",
            "/api/v2/mix/account/account",
            {"symbol": symbol.upper(), "productType": "USDT-FUTURES", "marginCoin": "USDT"},
            private=True,
        )
        mode = str((account or {}).get("posMode") or "").lower()
        if mode == "one_way_mode":
            return "ONE_WAY"
        if mode == "hedge_mode":
            return "HEDGE"
        raise RuntimeError("Bitget position mode could not be determined safely")

    async def get_plan_orders(self, symbol):
        return await self._request(
            "GET",
            "/api/v2/mix/order/orders-plan-pending",
            params={"productType": "USDT-FUTURES", "symbol": symbol.upper()},
            private=True,
        )

    async def cancel_plan_order(self, symbol, order_id=None, client_oid=None, plan_type=None):
        item = {}
        if order_id:
            item["orderId"] = str(order_id)
        if client_oid:
            item["clientOid"] = str(client_oid)
        payload = {"productType": "USDT-FUTURES", "symbol": symbol.upper(), "marginCoin": "USDT", "orderIdList": [item]}
        if plan_type:
            payload["planType"] = plan_type
        return await self._request("POST", "/api/v2/mix/order/cancel-plan-order", body=payload, private=True)

    async def _cancel_hhhai_protection(self, symbol):
        rows = await self.get_plan_orders(symbol)
        if isinstance(rows, dict):
            rows = rows.get("data", rows.get("list", rows.get("orders", [])))
        for row in rows or []:
            client_oid = str(row.get("clientOid") or "")
            if client_oid.startswith("HHHAI-"):
                try:
                    await self.cancel_plan_order(symbol, order_id=row.get("orderId"), client_oid=client_oid, plan_type=row.get("planType"))
                except Exception:
                    pass

    async def update_dynamic_protection(self, symbol, side, quantity, stop_price, position_mode="HEDGE"):
        await self._cancel_hhhai_protection(symbol)
        return await self.place_protection(symbol, side, quantity, stop_price, None, position_mode=position_mode)

    async def place_protection(self, symbol, side, quantity, stop_price, take_profit, position_mode="HEDGE"):
        hold_side = "long" if side.lower() == "long" else "short"
        client_oid = f"HHHAI-{uuid.uuid4().hex[:20]}"
        payload = {
            "marginCoin": "USDT",
            "productType": "USDT-FUTURES",
            "symbol": symbol.upper(),
            "planType": "loss_plan",
            "triggerPrice": str(stop_price),
            "triggerType": "mark_price",
            "executePrice": "0",
            "holdSide": hold_side,
            "size": str(quantity),
            "clientOid": client_oid,
        }
        stop = await self._request("POST", "/api/v2/mix/order/place-tpsl-order", body=payload, private=True)
        if take_profit is None:
            return {"stop_loss": stop, "take_profit": None, "attached": True, "client_oid": client_oid}
        tp_oid = f"HHHAI-{uuid.uuid4().hex[:20]}"
        tp = dict(payload, planType="profit_plan", triggerPrice=str(take_profit), clientOid=tp_oid)
        take = await self._request("POST", "/api/v2/mix/order/place-tpsl-order", body=tp, private=True)
        return {"stop_loss": stop, "take_profit": take, "attached": True, "client_oids": [client_oid, tp_oid]}
