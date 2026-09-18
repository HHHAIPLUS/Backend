from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import threading
import time
import uuid
import math
from collections import deque
from urllib.parse import urlencode

import httpx

from app.exchanges.base import ExchangeAdapter
from app.core.config import settings


class BitgetAdapter(ExchangeAdapter):
    name = "bitget"
    _live_canary_trade_count = 0

    # Deliberately conservative global limiter. Bitget documents 6000 requests/IP/min
    # overall, plus endpoint-specific limits. HHHAI stays far below those ceilings.
    _rate_lock = threading.Lock()
    _request_times = deque()
    _max_requests_per_second = 1
    _max_requests_per_minute = 60
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
            if response.status_code >= 400:
                raise RuntimeError(f"Bitget HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            if data.get("code") not in (None, "00000", 0):
                raise RuntimeError(f"Bitget API error: {data.get('code')} {data.get('msg')}")
            return data.get("data", data)

    async def get_account_status(self):
        accounts = await self._request("GET", "/api/v2/mix/account/accounts", {"productType": "USDT-FUTURES"}, private=True)
        rows = accounts if isinstance(accounts, list) else [accounts]
        usdt = next((row for row in rows if str(row.get("marginCoin") or "").upper() == "USDT"), None)
        if not usdt:
            raise RuntimeError("Bitget USDT-FUTURES account balance is unavailable")
        available = float(usdt.get("available") or 0)
        total = float(usdt.get("accountEquity") or usdt.get("usdtEquity") or usdt.get("equity") or available or 0)
        return {
            "exchange": self.name,
            "testnet": self.testnet,
            "available_balance": available,
            "total_wallet_balance": total,
            "accounts": accounts,
        }

    async def get_positions(self, symbol=None):
        p = {"productType": "USDT-FUTURES"}
        if symbol:
            p["symbol"] = symbol.upper()
        return await self._request("GET", "/api/v2/mix/position/all-position", p, private=True)

    async def get_ticker(self, symbol):
        return await self._request("GET", "/api/v2/mix/market/ticker", {"productType": "USDT-FUTURES", "symbol": symbol.upper()})

    async def get_contract_config(self, symbol=None):
        params = {"productType": "USDT-FUTURES"}
        if symbol:
            params["symbol"] = symbol.upper()
        return await self._request("GET", "/api/v2/mix/market/contracts", params=params)

    async def set_leverage(self, symbol, leverage):
        return await self._request(
            "POST",
            "/api/v2/mix/account/set-leverage",
            body={
                "symbol": symbol.upper(),
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT",
                "leverage": str(leverage),
            },
            private=True,
        )

    @staticmethod
    def _minimum_order_size(contract, price):
        min_num = float(contract.get("minTradeNum") or 0)
        min_usdt = float(contract.get("minTradeUSDT") or 0)
        multiplier = float(contract.get("sizeMultiplier") or 0)
        required = max(min_num, min_usdt / max(price, 1e-12))
        if multiplier > 0:
            required = math.ceil(required / multiplier - 1e-12) * multiplier
        return required

    async def place_order(self, order):
        payload = {"productType": "USDT-FUTURES", **order}
        payload.setdefault("clientOid", f"HHHAI-{uuid.uuid4().hex[:20]}")
        canary = (
            not self.testnet
            and settings.live_trading_enabled
            and __import__("os").getenv("HHHAI_LIVE_CANARY_ENABLED", "false").lower() == "true"
            and str(payload.get("orderType") or "").lower() == "market"
        )
        if canary:
            max_trades = max(1, int(__import__("os").getenv("HHHAI_LIVE_CANARY_MAX_TRADES", "1")))
            max_notional = float(__import__("os").getenv("HHHAI_LIVE_CANARY_MAX_NOTIONAL_USD", "2.00"))
            if self.__class__._live_canary_trade_count >= max_trades:
                raise RuntimeError(f"Live canary trade limit reached ({max_trades})")
            if not max_notional > 0:
                raise RuntimeError("Invalid live canary maximum notional")
            symbol = str(payload.get("symbol") or "").upper()
            size = float(payload.get("size") or 0)
            ticker = await self.get_ticker(symbol)
            ticker_row = ticker[0] if isinstance(ticker, list) and ticker else ticker
            price = float((ticker_row or {}).get("lastPr") or 0)
            contracts = await self.get_contract_config(symbol)
            contract = contracts[0] if isinstance(contracts, list) and contracts else contracts
            if size <= 0 or price <= 0 or not isinstance(contract, dict):
                raise RuntimeError("Live canary blocked order: Bitget order size, price, or contract rules are unavailable")
            minimum_size = self._minimum_order_size(contract, price)
            if minimum_size > 0 and size < minimum_size:
                size = minimum_size
                payload["size"] = str(size)
            notional = size * price
            if notional > max_notional + 1e-9:
                raise RuntimeError(
                    f"Live canary blocked order: minimum valid notional {notional:.8f} USDT exceeds maximum {max_notional:.8f} USDT"
                )
            await self.set_leverage(symbol, 2)
        result = await self._request("POST", "/api/v2/mix/order/place-order", body=payload, private=True)
        if canary:
            self.__class__._live_canary_trade_count += 1
        return result

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
        # Bitget hedge mode requires an explicit close trade side. Sending it
        # consistently also avoids ambiguity during the controlled canary cleanup.
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
