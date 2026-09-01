from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.market_data.intelligence import MarketBar, PointInTimeContext


class MarketDataProviderError(RuntimeError):
    pass


def _dt(ms: int | str) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


def _client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(12.0, connect=5.0), follow_redirects=True, trust_env=False, headers={"User-Agent": "HHHAI/2.0"})


def _raise(response: httpx.Response, source: str) -> Any:
    try:
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise MarketDataProviderError(f"{source} HTTP error: {response.status_code}") from exc
    try:
        return response.json()
    except ValueError as exc:
        raise MarketDataProviderError(f"{source} returned non-JSON data") from exc


def _imbalance(bids: list, asks: list) -> tuple[float, float | None]:
    bid_qty = sum(float(x[1]) for x in bids if len(x) >= 2)
    ask_qty = sum(float(x[1]) for x in asks if len(x) >= 2)
    total = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total if total > 0 else 0.0
    bid = float(bids[0][0]) if bids else 0.0
    ask = float(asks[0][0]) if asks else 0.0
    spread_bps = ((ask - bid) / ((ask + bid) / 2.0)) * 10000 if bid > 0 and ask > 0 else None
    return max(-1.0, min(1.0, imbalance)), spread_bps


class BinanceMarketData:
    base = "https://fapi.binance.com"

    def candles(self, symbol: str, interval: str = "5m", limit: int = 500, start_ms: int | None = None, end_ms: int | None = None) -> list[MarketBar]:
        params: dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": min(max(limit, 1), 1500)}
        if start_ms is not None: params["startTime"] = start_ms
        if end_ms is not None: params["endTime"] = end_ms
        with _client() as client:
            data = _raise(client.get(f"{self.base}/fapi/v1/klines", params=params), "binance klines")
        return [MarketBar(symbol=symbol.upper(), timeframe=interval, timestamp=_dt(row[0]), open=float(row[1]), high=float(row[2]), low=float(row[3]), close=float(row[4]), volume=float(row[5]), quote_volume=float(row[7])) for row in data if len(row) >= 8]

    def funding(self, symbol: str, limit: int = 1000, start_ms: int | None = None, end_ms: int | None = None) -> list[PointInTimeContext]:
        params: dict[str, Any] = {"symbol": symbol.upper(), "limit": min(max(limit, 1), 1000)}
        if start_ms is not None: params["startTime"] = start_ms
        if end_ms is not None: params["endTime"] = end_ms
        with _client() as client:
            data = _raise(client.get(f"{self.base}/fapi/v1/fundingRate", params=params), "binance funding")
        return [PointInTimeContext(timestamp=_dt(row[0]), funding_rate=float(row[2])) for row in data]

    def open_interest_history(self, symbol: str, period: str = "5m", limit: int = 500, start_ms: int | None = None, end_ms: int | None = None) -> list[PointInTimeContext]:
        params: dict[str, Any] = {"symbol": symbol.upper(), "period": period, "limit": min(max(limit, 1), 500)}
        if start_ms is not None: params["startTime"] = start_ms
        if end_ms is not None: params["endTime"] = end_ms
        with _client() as client:
            data = _raise(client.get(f"{self.base}/futures/data/openInterestHist", params=params), "binance open interest")
        return [PointInTimeContext(timestamp=_dt(row["timestamp"]), open_interest=float(row["sumOpenInterest"])) for row in data]

    def trade_flow(self, symbol: str, limit: int = 500) -> list[PointInTimeContext]:
        with _client() as client:
            data = _raise(client.get(f"{self.base}/fapi/v1/aggTrades", params={"symbol": symbol.upper(), "limit": min(max(limit, 1), 1000)}), "binance aggregate trades")
        return [PointInTimeContext(timestamp=_dt(row["T"]), aggressive_buy_ratio=(0.0 if row.get("m") else 1.0)) for row in data]

    def liquidations(self, symbol: str, limit: int = 100) -> list[PointInTimeContext]:
        # Binance's public REST force-order endpoint is limited historical data;
        # the feed is treated as recent context rather than an unlimited archive.
        with _client() as client:
            data = _raise(client.get(f"{self.base}/fapi/v1/forceOrders", params={"symbol": symbol.upper(), "limit": min(max(limit, 1), 1000)}), "binance liquidations")
        return [PointInTimeContext(timestamp=_dt(row["time"]), liquidation_notional=float(row.get("price", 0)) * float(row.get("origQty", 0))) for row in data]

    def order_book(self, symbol: str, limit: int = 20) -> PointInTimeContext:
        with _client() as client:
            data = _raise(client.get(f"{self.base}/fapi/v1/depth", params={"symbol": symbol.upper(), "limit": min(max(limit, 5), 1000)}), "binance orderbook")
        imbalance, spread = _imbalance(data.get("bids", []), data.get("asks", []))
        return PointInTimeContext(timestamp=_dt(data.get("E") or int(datetime.now(timezone.utc).timestamp() * 1000)), order_book_imbalance=imbalance, spread_bps=spread)


class BitgetMarketData:
    base = "https://api.bitget.com"

    def candles(self, symbol: str, interval: str = "5m", limit: int = 200, start_ms: int | None = None, end_ms: int | None = None) -> list[MarketBar]:
        params: dict[str, Any] = {"category": "USDT-FUTURES", "symbol": symbol.upper(), "interval": interval, "limit": min(max(limit, 1), 200)}
        if start_ms is not None: params["startTime"] = start_ms
        if end_ms is not None: params["endTime"] = end_ms
        with _client() as client:
            payload = _raise(client.get(f"{self.base}/api/v3/market/history-candles", params=params), "bitget candles")
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        return [MarketBar(symbol=symbol.upper(), timeframe=interval, timestamp=_dt(row[0]), open=float(row[1]), high=float(row[2]), low=float(row[3]), close=float(row[4]), volume=float(row[5]), quote_volume=float(row[6])) for row in rows if len(row) >= 7]

    def open_interest(self, symbol: str) -> PointInTimeContext:
        params = {"category": "USDT-FUTURES", "symbol": symbol.upper()}
        with _client() as client:
            payload = _raise(client.get(f"{self.base}/api/v3/market/open-interest", params=params), "bitget open interest")
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        rows = data.get("list", []) if isinstance(data, dict) else []
        value = float(rows[0]["openInterest"]) if rows else None
        ts = int(data.get("ts") or payload.get("requestTime") or 0)
        return PointInTimeContext(timestamp=_dt(ts), open_interest=value)

    def order_book(self, symbol: str, limit: int = 20) -> PointInTimeContext:
        params = {"category": "USDT-FUTURES", "symbol": symbol.upper(), "limit": min(max(limit, 1), 1000)}
        with _client() as client:
            payload = _raise(client.get(f"{self.base}/api/v3/market/orderbook", params=params), "bitget orderbook")
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        imbalance, spread = _imbalance(data.get("b", []), data.get("a", []))
        ts = int(data.get("ts") or payload.get("requestTime") or 0)
        return PointInTimeContext(timestamp=_dt(ts), order_book_imbalance=imbalance, spread_bps=spread)

    def trade_flow(self, symbol: str, limit: int = 100) -> list[PointInTimeContext]:
        with _client() as client:
            payload = _raise(client.get(f"{self.base}/api/v2/mix/market/fills", params={"productType": "USDT-FUTURES", "symbol": symbol.upper(), "limit": min(max(limit, 1), 100)}), "bitget recent fills")
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        return [PointInTimeContext(timestamp=_dt(row["ts"]), aggressive_buy_ratio=(1.0 if str(row.get("side", "")).lower() == "buy" else 0.0)) for row in rows]

    def liquidations(self, symbol: str, limit: int = 100) -> list[PointInTimeContext]:
        params = {"category": "USDT-FUTURES", "symbol": symbol.upper(), "limit": min(max(limit, 1), 100)}
        with _client() as client:
            payload = _raise(client.get(f"{self.base}/api/v3/market/liquidations", params=params), "bitget liquidations")
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        rows = data.get("list", []) if isinstance(data, dict) else []
        return [PointInTimeContext(timestamp=_dt(row["ts"]), liquidation_notional=float(row.get("price", 0)) * float(row.get("amount", 0))) for row in rows]


def provider(name: str) -> BinanceMarketData | BitgetMarketData:
    normalized = name.strip().lower()
    if normalized == "binance": return BinanceMarketData()
    if normalized == "bitget": return BitgetMarketData()
    raise ValueError(f"Unsupported market-data provider: {name}")
