from __future__ import annotations

import json
import logging
import threading
import time
import re
from statistics import pstdev
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

from app.ml.features import build_model_features

import httpx
from pydantic import BaseModel, Field

try:
    from websockets.sync.client import connect as websocket_connect
except ImportError:
    websocket_connect = None


log = logging.getLogger("hhhai.market_data")


class FeedHealth(BaseModel):
    source: str
    status: str
    latency_ms: float | None = None
    observed_at: datetime
    stale_after_seconds: int
    error: str | None = None


class RealtimeSnapshot(BaseModel):
    symbol: str
    source: str
    price: float = Field(gt=0)
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    volume_24h: float = Field(ge=0)
    funding_rate: float | None = None
    open_interest: float | None = None
    open_interest_change: float | None = None
    order_book_imbalance: float = 0.0
    volatility_proxy: float = 0.0
    price_change_24h: float = 0.0
    observed_at: datetime
    feed_health: FeedHealth
    model_features: dict[str, float] = Field(default_factory=dict)


class NewsEvent(BaseModel):
    source: str
    title: str
    url: str | None = None
    published_at: datetime
    sentiment: float = Field(ge=-1, le=1)
    relevance: float = Field(ge=0, le=1)
    impact: float = Field(ge=0, le=1)
    credibility: float = Field(ge=0, le=1)
    keywords: list[str] = Field(default_factory=list)


class WorldIntelligence(BaseModel):
    symbol: str
    market: RealtimeSnapshot
    news: list[NewsEvent]
    news_risk: float = Field(ge=0, le=1)
    news_sentiment: float = Field(ge=-1, le=1)
    data_quality: float = Field(ge=0, le=1)
    danger_flags: list[str] = Field(default_factory=list)
    momentum_proxy: float = 0.0
    trend_strength: float = 0.0
    liquidity_stress: float = 0.0
    market_risk: float = 0.0
    news_credibility: float = 0.0
    observed_at: datetime


class BinanceWebSocketFeed:
    """
    Persistent Binance USDⓈ-M Futures market-data feed.

    WebSocket is used when available.  A public REST snapshot is used as
    a safety fallback because HHHAI must not fail its entire observation
    cycle just because an outbound WebSocket connection is unavailable.

    Binance's current Futures routing is:
        /market/stream -> combined regular market streams
        /public/stream -> high-frequency public streams such as depth
    """

    websocket_base = "wss://fstream.binance.com/market/stream"
    rest_base = "https://fapi.binance.com"

    _instances: dict[str, "BinanceWebSocketFeed"] = {}
    _instances_lock = threading.Lock()

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.stream_symbol = self.symbol.lower()

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._data_ready = threading.Event()

        self._last_update = 0.0
        self._last_error: str | None = None

        self._price = 0.0
        self._bid = 0.0
        self._ask = 0.0
        self._bid_qty = 0.0
        self._ask_qty = 0.0
        self._depth_bid_qty = 0.0
        self._depth_ask_qty = 0.0
        self._volume_24h = 0.0
        self._price_change_24h = 0.0
        self._funding_rate: float | None = None
        self._open_interest: float | None = None
        self._previous_open_interest: float | None = None
        self._model_features: dict[str, float] = {}
        self._model_features_at = 0.0

    @classmethod
    def get(cls, symbol: str) -> "BinanceWebSocketFeed":
        symbol = symbol.upper()

        with cls._instances_lock:
            feed = cls._instances.get(symbol)
            if feed is None:
                feed = cls(symbol)
                cls._instances[symbol] = feed

            feed.start()
            return feed

    def start(self) -> None:
        if websocket_connect is None:
            self._last_error = "The 'websockets' package is not installed."
            return

        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"binance-ws-{self.symbol}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _stream_url(self) -> str:
        streams = "/".join(
            [
                f"{self.stream_symbol}@ticker",
                f"{self.stream_symbol}@markPrice@1s",
                f"{self.stream_symbol}@depth5@100ms",
            ]
        )
        return f"{self.websocket_base}?streams={streams}"

    def _process_message(self, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning(
                "Invalid Binance WebSocket JSON for %s: %s",
                self.symbol,
                exc,
            )
            return

        data = payload.get("data", payload)
        if not isinstance(data, dict):
            return

        event_type = data.get("e")

        try:
            with self._lock:
                if event_type == "24hrTicker":
                    price = float(data.get("c", 0) or 0)
                    bid = float(data.get("b", 0) or 0)
                    ask = float(data.get("a", 0) or 0)

                    if price > 0:
                        self._price = price
                    if bid > 0:
                        self._bid = bid
                    if ask > 0:
                        self._ask = ask

                    self._bid_qty = float(data.get("B", 0) or 0)
                    self._ask_qty = float(data.get("A", 0) or 0)
                    self._volume_24h = float(data.get("q", 0) or 0)
                    self._price_change_24h = (
                        float(data.get("P", 0) or 0) / 100.0
                    )
                    self._last_update = time.time()
                    self._data_ready.set()

                    log.info(
                        "Binance WebSocket ticker received for %s: "
                        "price=%s bid=%s ask=%s",
                        self.symbol,
                        self._price,
                        self._bid,
                        self._ask,
                    )

                elif event_type == "depthUpdate":
                    bids = data.get("b") or []
                    asks = data.get("a") or []
                    self._depth_bid_qty = sum(float(row[1]) for row in bids if len(row) >= 2)
                    self._depth_ask_qty = sum(float(row[1]) for row in asks if len(row) >= 2)
                    self._last_update = time.time()

                elif event_type == "markPriceUpdate":
                    self._funding_rate = float(data.get("r", 0) or 0)
                    self._last_update = time.time()

                    if self._price > 0:
                        self._data_ready.set()

                    log.debug(
                        "Binance mark-price update received for %s",
                        self.symbol,
                    )

        except (TypeError, ValueError) as exc:
            log.warning(
                "Invalid Binance WebSocket values for %s: %s",
                self.symbol,
                exc,
            )

    def _run(self) -> None:
        reconnect_delay = 2.0

        while not self._stop.is_set():
            try:
                url = self._stream_url()

                log.info(
                    "Connecting Binance WebSocket for %s: %s",
                    self.symbol,
                    url,
                )

                with websocket_connect(
                    url,
                    proxy=None,
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=60,
                    max_size=2**20,
                    logger=log,
                ) as websocket:
                    log.info(
                        "Binance WebSocket CONNECTED for %s",
                        self.symbol,
                    )

                    with self._lock:
                        self._last_error = None

                    reconnect_delay = 2.0

                    while not self._stop.is_set():
                        try:
                            message = websocket.recv(timeout=30)

                            if message is None:
                                raise RuntimeError(
                                    "Binance WebSocket returned no message"
                                )

                            self._process_message(message)

                        except TimeoutError:
                            continue

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

                with self._lock:
                    self._last_error = error

                log.warning(
                    "Binance WebSocket error for %s: %s",
                    self.symbol,
                    error,
                )

            if self._stop.is_set():
                break

            log.info(
                "Reconnecting Binance WebSocket for %s in %.1f seconds",
                self.symbol,
                reconnect_delay,
            )

            self._stop.wait(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30.0)

    def _rest_model_features(self) -> dict[str, float]:
        """Fetch recent candles used exclusively for predictive features."""
        headers = {"User-Agent": "HHHAI/1.0", "Accept": "application/json"}
        timeout = httpx.Timeout(8.0, connect=5.0)
        with httpx.Client(
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            trust_env=False,
        ) as client:
            response = client.get(
                f"{self.rest_base}/fapi/v1/klines",
                params={"symbol": self.symbol, "interval": "5m", "limit": 30},
            )
            response.raise_for_status()
            candles = response.json()
        return build_model_features(candles)

    def _rest_snapshot(self) -> dict[str, float | None]:
        """
        Public Binance REST fallback.

        This does not require API keys and prevents the monitoring worker
        from becoming blind when Render cannot establish a WebSocket.
        """
        headers = {
            "User-Agent": "HHHAI/1.0",
            "Accept": "application/json",
        }

        timeout = httpx.Timeout(8.0, connect=5.0)

        with httpx.Client(
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            trust_env=False,
        ) as client:
            ticker_response = client.get(
                f"{self.rest_base}/fapi/v1/ticker/24hr",
                params={"symbol": self.symbol},
            )
            ticker_response.raise_for_status()
            ticker = ticker_response.json()

            depth_response = client.get(
                f"{self.rest_base}/fapi/v1/depth",
                params={"symbol": self.symbol, "limit": 5},
            )
            depth_response.raise_for_status()
            depth = depth_response.json()

            funding_response = client.get(
                f"{self.rest_base}/fapi/v1/premiumIndex",
                params={"symbol": self.symbol},
            )
            funding_response.raise_for_status()
            funding = funding_response.json()

            oi_response = client.get(
                f"{self.rest_base}/fapi/v1/openInterest",
                params={"symbol": self.symbol},
            )
            oi_response.raise_for_status()
            oi = oi_response.json()

            candles_response = client.get(
                f"{self.rest_base}/fapi/v1/klines",
                params={"symbol": self.symbol, "interval": "5m", "limit": 30},
            )
            candles_response.raise_for_status()
            candles = candles_response.json()

        bids = depth.get("bids") or []
        asks = depth.get("asks") or []

        bid_price = float(bids[0][0]) if bids else float(ticker["lastPrice"])
        ask_price = float(asks[0][0]) if asks else float(ticker["lastPrice"])

        bid_qty = sum(float(row[1]) for row in bids)
        ask_qty = sum(float(row[1]) for row in asks)
        closes = [float(row[4]) for row in candles if len(row) > 4 and float(row[4]) > 0]
        rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))] if len(closes) > 1 else []
        short_rets = rets[-12:]
        volatility_proxy = min(1.0, pstdev(short_rets) * 10) if len(short_rets) > 1 else 0.0
        trend_strength = min(1.0, abs(sum(short_rets)) * 12) if short_rets else 0.0
        momentum = max(-1.0, min(1.0, sum(rets[-3:]) * 25)) if rets else 0.0

        return {
            "price": float(ticker["lastPrice"]),
            "bid": bid_price,
            "ask": ask_price,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "volume_24h": float(ticker.get("quoteVolume", 0) or 0),
            "price_change_24h": (
                float(ticker.get("priceChangePercent", 0) or 0) / 100.0
            ),
            "funding_rate": float(funding.get("lastFundingRate", 0) or 0),
            "open_interest": float(oi.get("openInterest", 0) or 0),
            "volatility_proxy": volatility_proxy,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "model_features": build_model_features(candles),
        }

    def _snapshot_from_rest(self, started: float) -> RealtimeSnapshot:
        try:
            data = self._rest_snapshot()
        except Exception as exc:
            raise RuntimeError(
                "Binance market data unavailable from both WebSocket "
                f"and REST fallback: {type(exc).__name__}: {exc}"
            ) from exc

        previous_oi = None
        price = float(data["price"] or 0)
        bid = float(data["bid"] or price)
        ask = float(data["ask"] or price)
        bid_qty = float(data["bid_qty"] or 0)
        ask_qty = float(data["ask_qty"] or 0)
        if data.get("open_interest") is not None:
            with self._lock:
                previous_oi = self._open_interest
                self._previous_open_interest = previous_oi
                self._open_interest = float(data["open_interest"])

        if price <= 0:
            raise RuntimeError(
                f"Binance REST returned an invalid price for {self.symbol}"
            )

        imbalance = (bid_qty - ask_qty) / max(
            bid_qty + ask_qty,
            1e-12,
        )

        now = datetime.now(timezone.utc)

        health = FeedHealth(
            source="binance_futures_rest_fallback",
            status="healthy",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            observed_at=now,
            stale_after_seconds=30,
            error=(
                "WebSocket unavailable; using Binance public REST fallback."
            ),
        )

        log.warning(
            "Using Binance REST fallback for %s because WebSocket "
            "market data is unavailable.",
            self.symbol,
        )

        return RealtimeSnapshot(
            symbol=self.symbol,
            source="binance_futures_rest_fallback",
            price=price,
            bid=bid,
            ask=ask,
            volume_24h=max(0.0, float(data["volume_24h"] or 0)),
            funding_rate=data["funding_rate"],
            open_interest=data.get("open_interest"),
            open_interest_change=((float(data["open_interest"]) / previous_oi) - 1.0 if data.get("open_interest") and previous_oi else None),
            order_book_imbalance=max(-1.0, min(1.0, imbalance)),
            volatility_proxy=float(data.get("volatility_proxy") or abs(float(data["price_change_24h"] or 0))),
            price_change_24h=float(data["price_change_24h"] or 0),
            observed_at=now,
            feed_health=health,
            model_features=dict(data.get("model_features") or {}),
        )

    def snapshot(self) -> RealtimeSnapshot:
        started = time.perf_counter()
        self.start()

        # If the WebSocket has not produced data yet, don't make the
        # whole observation worker wait 12 seconds before trying REST.
        if not self._data_ready.wait(timeout=3):
            return self._snapshot_from_rest(started)

        with self._lock:
            price = self._price
            bid = self._bid
            ask = self._ask
            bid_qty = self._depth_bid_qty or self._bid_qty
            ask_qty = self._depth_ask_qty or self._ask_qty
            open_interest = self._open_interest
            volume = self._volume_24h
            funding = self._funding_rate
            price_change = self._price_change_24h
            last_update = self._last_update
            last_error = self._last_error

        age = time.time() - last_update if last_update else float("inf")

        # A connected but stale WebSocket must not be trusted.
        if price <= 0 or age > 15:
            return self._snapshot_from_rest(started)

        if bid <= 0:
            bid = price
        if ask <= 0:
            ask = price

        imbalance = (bid_qty - ask_qty) / max(
            bid_qty + ask_qty,
            1e-12,
        )

        # Keep the predictive feature definitions identical to bootstrap.
        # The websocket supplies live ticker/order-book data, while this
        # lightweight REST call refreshes the candle-derived model features
        # at most once every 30 seconds. If it fails, fail closed rather than
        # feeding the model mismatched 24h proxy values.
        if time.time() - self._model_features_at > 30.0:
            try:
                model_features = self._rest_model_features()
                with self._lock:
                    self._model_features = model_features
                    self._model_features_at = time.time()
            except Exception as exc:
                log.warning(
                    "Unable to refresh Binance model features for %s: %s",
                    self.symbol,
                    exc,
                )

        with self._lock:
            model_features = dict(self._model_features)

        if not model_features:
            # No model-compatible candle history means predictive inference
            # must not be treated as valid. The caller will see missing
            # features and the validated-model gate remains fail-closed.
            model_features = {}

        now = datetime.now(timezone.utc)

        health = FeedHealth(
            source="binance_futures_websocket",
            status="healthy",
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            observed_at=now,
            stale_after_seconds=15,
            error=last_error,
        )

        return RealtimeSnapshot(
            symbol=self.symbol,
            source="binance_futures_websocket",
            price=price,
            bid=bid,
            ask=ask,
            volume_24h=max(0.0, volume),
            funding_rate=funding,
            open_interest=open_interest,
            open_interest_change=None,
            order_book_imbalance=max(-1.0, min(1.0, imbalance)),
            volatility_proxy=abs(price_change),
            price_change_24h=price_change,
            observed_at=now,
            feed_health=health,
            model_features=model_features,
        )


class BinancePublicFeed:
    """
    Compatibility wrapper used by the rest of HHHAI.

    WebSocket is preferred; public REST is automatically used if the
    WebSocket cannot provide fresh data.
    """

    def snapshot(self, symbol: str) -> RealtimeSnapshot:
        return BinanceWebSocketFeed.get(symbol).snapshot()


class BitgetPublicFeed:
    """Public Bitget USDT-M Futures snapshot using REST market data."""

    rest_base = "https://api.bitget.com"

    def snapshot(self, symbol: str) -> RealtimeSnapshot:
        symbol = symbol.upper()
        started = time.perf_counter()
        headers = {"User-Agent": "HHHAI/1.0", "Accept": "application/json"}
        timeout = httpx.Timeout(8.0, connect=5.0)

        try:
            with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True, trust_env=True) as client:
                ticker_response = client.get(
                    f"{self.rest_base}/api/v2/mix/market/ticker",
                    params={"productType": "USDT-FUTURES", "symbol": symbol},
                )
                ticker_response.raise_for_status()
                ticker_payload = ticker_response.json()
                tickers = ticker_payload.get("data") or []
                ticker = tickers[0] if isinstance(tickers, list) and tickers else ticker_payload.get("data", {})

                depth_response = client.get(
                    f"{self.rest_base}/api/v2/mix/market/orderbook",
                    params={"productType": "USDT-FUTURES", "symbol": symbol, "limit": 5},
                )
                depth_response.raise_for_status()
                depth_payload = depth_response.json()
                depth = depth_payload.get("data") or {}

                oi_response = client.get(
                    f"{self.rest_base}/api/v2/mix/market/open-interest",
                    params={"productType": "USDT-FUTURES", "symbol": symbol},
                )
                oi_response.raise_for_status()
                oi_payload = oi_response.json()

                candles_response = client.get(
                    f"{self.rest_base}/api/v2/mix/market/candles",
                    params={"productType": "USDT-FUTURES", "symbol": symbol, "granularity": "5m", "limit": 30},
                )
                candles_response.raise_for_status()
                candles_payload = candles_response.json()

            price = float(ticker.get("lastPr", 0) or 0)
            bid = float(ticker.get("bidPr", 0) or price)
            ask = float(ticker.get("askPr", 0) or price)
            bids = depth.get("bids") or depth.get("b") or []
            asks = depth.get("asks") or depth.get("a") or []
            bid_qty = sum(float(row[1]) for row in bids if len(row) >= 2)
            ask_qty = sum(float(row[1]) for row in asks if len(row) >= 2)
            if bid <= 0 and bids:
                bid = float(bids[0][0])
            if ask <= 0 and asks:
                ask = float(asks[0][0])
            if price <= 0:
                raise RuntimeError(f"Bitget returned an invalid price for {symbol}")

            imbalance = (bid_qty - ask_qty) / max(bid_qty + ask_qty, 1e-12)
            change = float(ticker.get("change24h", ticker.get("price24hPcnt", 0)) or 0)
            # Bitget v2 classic reports change24h as a decimal fraction.
            if abs(change) > 1.0:
                change /= 100.0
            raw_candles = candles_payload.get("data") or []
            closes = [float(r[4]) for r in raw_candles if isinstance(r, list) and len(r) >= 5 and float(r[4]) > 0]
            rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))] if len(closes) > 1 else []
            volatility_proxy = min(1.0, pstdev(rets[-12:]) * 10) if len(rets[-12:]) > 1 else abs(change)
            trend_strength = min(1.0, abs(sum(rets[-12:])) * 12) if rets else min(1.0, abs(change) * 4)
            momentum = max(-1.0, min(1.0, sum(rets[-3:]) * 25)) if rets else max(-1.0, min(1.0, change * 5))
            oi_data = oi_payload.get("data") or {}
            if isinstance(oi_data, list): oi_data = oi_data[0] if oi_data else {}
            oi = float(oi_data.get("openInterest", oi_data.get("openInterestValue", 0)) or 0)
            now = datetime.now(timezone.utc)
            health = FeedHealth(
                source="bitget_futures_rest",
                status="healthy",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                observed_at=now,
                stale_after_seconds=30,
                error=None,
            )
            return RealtimeSnapshot(
                symbol=symbol,
                source="bitget_futures_rest",
                price=price,
                bid=bid if bid > 0 else price,
                ask=ask if ask > 0 else price,
                volume_24h=max(0.0, float(ticker.get("usdtVolume", ticker.get("quoteVolume", ticker.get("turnover24h", 0))) or 0)),
                funding_rate=float(ticker.get("fundingRate", 0) or 0),
                open_interest=oi or None,
                open_interest_change=None,
                order_book_imbalance=max(-1.0, min(1.0, imbalance)),
                volatility_proxy=volatility_proxy,
                price_change_24h=change,
                observed_at=now,
                feed_health=health,
                model_features=build_model_features(raw_candles),
            )
        except Exception as exc:
            raise RuntimeError(f"Bitget market data unavailable for {symbol}: {type(exc).__name__}: {exc}") from exc


class CoinDeskNewsFeed:

    POSITIVE = {"surge", "rally", "bullish", "approval", "approved", "adoption", "inflow", "growth", "record", "breakout", "launch", "partnership", "gain"}
    NEGATIVE = {"crash", "drop", "fall", "bearish", "hack", "hacked", "lawsuit", "ban", "banned", "outflow", "liquidation", "liquidations", "exploit", "fraud", "warning", "loss", "decline"}

    @classmethod
    def score_headline(cls, title: str, symbol: str | None = None) -> tuple[float, float, float, list[str]]:
        words = set(re.findall(r"[a-z0-9$-]+", title.lower()))
        positive = len(words & cls.POSITIVE)
        negative = len(words & cls.NEGATIVE)
        sentiment = max(-1.0, min(1.0, (positive - negative) / max(1, positive + negative)))
        keywords = []
        if positive: keywords.extend(sorted(words & cls.POSITIVE))
        if negative: keywords.extend(sorted(words & cls.NEGATIVE))
        target = (symbol or "").replace("USDT", "").replace("USD", "").upper()
        relevant = 0.35 if target and target.lower() in title.lower() else 0.10
        if any(k in words for k in {"bitcoin", "btc", "crypto", "cryptocurrency", "market", "futures"}): relevant = max(relevant, 0.45)
        impact = min(1.0, 0.25 + 0.15 * len(keywords) + (0.25 if negative >= 2 or positive >= 2 else 0.0))
        return sentiment, relevant, impact, keywords

    url = (
        "https://www.coindesk.com/"
        "arc/outboundfeeds/rss/"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(X11; Linux x86_64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0.0.0 "
            "Safari/537.36"
        ),
        "Accept": (
            "application/rss+xml, "
            "application/xml, "
            "text/xml, "
            "*/*"
        ),
    }

    def fetch(
        self,
        limit: int = 12,
        symbol: str | None = None,
    ) -> list[NewsEvent]:

        try:

            with httpx.Client(
                timeout=10.0,
                headers=self.headers,
                follow_redirects=True,
                trust_env=True,
            ) as client:

                response = client.get(
                    self.url
                )

                if response.status_code != 200:

                    raise RuntimeError(
                        f"CoinDesk HTTP "
                        f"{response.status_code}"
                    )

                text = response.text

            root = ET.fromstring(
                text
            )

            events: list[NewsEvent] = []

            for item in root.findall(
                ".//item"
            )[:limit]:

                title = (
                    item.findtext(
                        "title"
                    )
                    or ""
                ).strip()

                pub = (
                    item.findtext(
                        "pubDate"
                    )
                    or ""
                ).strip()

                if pub:

                    published = (
                        parsedate_to_datetime(
                            pub
                        ).astimezone(
                            timezone.utc
                        )
                    )

                else:

                    published = (
                        datetime.now(
                            timezone.utc
                        )
                    )

                sentiment, relevance, impact, keywords = self.score_headline(title, symbol)
                events.append(
                    NewsEvent(
                        source="CoinDesk RSS",
                        title=title,
                        url=item.findtext("link") or None,
                        published_at=published,
                        sentiment=sentiment,
                        relevance=relevance,
                        impact=impact,
                        credibility=0.65,
                        keywords=keywords,
                    )
                )

            return events

        except Exception as exc:

            raise RuntimeError(
                f"News feed unavailable: {exc}"
            ) from exc


def assess_news(
    events: list[NewsEvent],
) -> tuple[
    float,
    float,
    list[str],
]:

    if not events:

        return (
            0.0,
            0.0,
            [],
        )

    now = datetime.now(
        timezone.utc
    )

    fresh = [
        event
        for event in events
        if (
            now - event.published_at
        ).total_seconds()
        <= 3600
    ]

    weighted = [event for event in fresh if event.relevance > 0.0]
    risk = max(min(1.0, len(fresh) / 8), min(1.0, sum(event.impact * max(event.relevance, 0.25) for event in fresh) / 3.0))
    sentiment = (
        sum(event.sentiment * max(event.relevance, 0.25) * max(event.impact, 0.25) for event in weighted)
        / sum(max(event.relevance, 0.25) * max(event.impact, 0.25) for event in weighted)
        if weighted else 0.0
    )
    flags = []
    if len(fresh) >= 4: flags.append("fresh_news_cluster")
    if risk >= 0.70: flags.append("high_news_impact")
    if sentiment <= -0.45: flags.append("negative_news_sentiment")
    if sentiment >= 0.45: flags.append("positive_news_sentiment")

    return (
        risk,
        max(
            -1.0,
            min(
                1.0,
                sentiment,
            ),
        ),
        flags,
    )


def build_world_intelligence(
    symbol: str,
    exchange: str = "binance",
) -> WorldIntelligence:

    exchange = exchange.lower().strip()
    if exchange == "binance":
        market = BinancePublicFeed().snapshot(symbol)
    elif exchange == "bitget":
        market = BitgetPublicFeed().snapshot(symbol)
    else:
        raise RuntimeError(f"Unsupported exchange: {exchange}")

    events = CoinDeskNewsFeed().fetch(symbol=symbol)

    (
        news_risk,
        sentiment,
        flags,
    ) = assess_news(
        events
    )

    if (
        market.order_book_imbalance
        < -0.35
    ):

        flags.append(
            "sell_side_orderbook_pressure"
        )

    if (
        market.volatility_proxy
        > 0.08
    ):

        flags.append(
            "abnormal_price_volatility"
        )

    if (
        abs(
            market.funding_rate
            or 0
        )
        > 0.0005
    ):

        flags.append(
            "elevated_funding"
        )

    quality = (
        1.0
        if (
            market.feed_health.status
            == "healthy"
        )
        else 0.0
    )

    momentum = max(
        -1.0,
        min(
            1.0,
            market.price_change_24h
            * 5.0,
        ),
    )

    trend_strength = max(
        0.0,
        min(
            1.0,
            abs(
                market.volatility_proxy
            )
            * 4.0,
        ),
    )

    liquidity_stress = max(
        0.0,
        min(
            1.0,
            abs(
                market.order_book_imbalance
            )
            * 0.7,
        ),
    )

    market_risk = max(
        0.0,
        min(
            1.0,
            (
                0.45
                * news_risk
                + 0.35
                * liquidity_stress
                + 0.20
                * min(
                    1.0,
                    market.volatility_proxy
                    * 5,
                )
            ),
        ),
    )

    credibility = (
        sum(
            event.credibility
            for event in events
        )
        / len(events)
        if events
        else 0.0
    )

    return WorldIntelligence(

        symbol=symbol.upper(),

        market=market,

        news=events,

        news_risk=news_risk,

        news_sentiment=sentiment,

        data_quality=quality,

        danger_flags=flags,

        momentum_proxy=momentum,

        trend_strength=trend_strength,

        liquidity_stress=liquidity_stress,

        market_risk=market_risk,

        news_credibility=credibility,

        observed_at=datetime.now(
            timezone.utc
        ),
    )
