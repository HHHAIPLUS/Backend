"""Historical derivatives context sources used by Stage 1A.

The source layer is intentionally capability-aware: it only records a field when
an exchange provides a timestamped historical observation. Unsupported historical
features remain unavailable instead of being fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import httpx

from app.ml.historical_context import HistoricalContext, make_context

BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_OI_HIST = "https://futures.binance.com/futures/data/openInterestHist"
BITGET_FUNDING = "https://api.bitget.com/api/v2/mix/market/history-fund-rate"


@dataclass(frozen=True)
class SourceCapability:
    provider: str
    feature: str
    historical: bool
    timestamped: bool
    notes: str


CAPABILITIES = (
    SourceCapability("binance", "funding_rate", True, True, "Funding-rate history endpoint"),
    SourceCapability("binance", "open_interest_change", True, True, "Historical open-interest statistics endpoint"),
    SourceCapability("bitget", "funding_rate", True, True, "Historical funding-rate endpoint"),
    SourceCapability("bitget", "open_interest_change", False, True, "Public endpoint exposes current OI; no historical series is assumed here"),
    SourceCapability("binance", "order_book_imbalance", False, False, "Do not reconstruct historical snapshots from current order book"),
    SourceCapability("binance", "news_sentiment", False, False, "Exchange market-data API does not provide historical news sentiment"),
    SourceCapability("binance", "news_risk", False, False, "Requires a timestamped external news archive"),
    SourceCapability("binance", "liquidity_stress", False, False, "Requires historical depth/trade microstructure data"),
)


def _ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware")
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def _client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True, trust_env=False, headers={"User-Agent": "HHHAI/1.0", "Accept": "application/json"})


def _between(ts: int, start_ms: int, end_ms: int) -> bool:
    return start_ms <= ts <= end_ms


def fetch_binance_funding(symbol: str, start: datetime, end: datetime, limit: int = 1000) -> list[tuple[int, float]]:
    params = {"symbol": symbol.upper(), "startTime": _ms(start), "endTime": _ms(end), "limit": min(1000, max(1, limit))}
    with _client() as client:
        response = client.get(BINANCE_FUNDING, params=params)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, list):
        raise RuntimeError("Binance funding response was not a list")
    out: list[tuple[int, float]] = []
    for item in data:
        try:
            ts = int(item["fundingTime"])
            rate = float(item["fundingRate"])
        except (KeyError, TypeError, ValueError):
            continue
        if _between(ts, _ms(start), _ms(end)):
            out.append((ts, rate))
    return sorted(set(out))


def fetch_binance_open_interest(symbol: str, start: datetime, end: datetime, period: str = "5m", limit: int = 500) -> list[tuple[int, float]]:
    params = {"symbol": symbol.upper(), "period": period, "limit": min(500, max(1, limit)), "startTime": _ms(start), "endTime": _ms(end)}
    with _client() as client:
        response = client.get(BINANCE_OI_HIST, params=params)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, list):
        raise RuntimeError("Binance open-interest response was not a list")
    out: list[tuple[int, float]] = []
    for item in data:
        try:
            ts = int(item["timestamp"])
            oi = float(item["sumOpenInterest"])
        except (KeyError, TypeError, ValueError):
            continue
        if _between(ts, _ms(start), _ms(end)):
            out.append((ts, oi))
    return sorted(set(out))


def fetch_bitget_funding(symbol: str, start: datetime, end: datetime, page_size: int = 100) -> list[tuple[int, float]]:
    params = {"symbol": symbol.upper(), "productType": "USDT-FUTURES", "pageSize": min(100, max(1, page_size)), "pageNo": 1}
    with _client() as client:
        response = client.get(BITGET_FUNDING, params=params)
        response.raise_for_status()
        payload = response.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    out: list[tuple[int, float]] = []
    for item in rows:
        try:
            ts = int(item["fundingTime"])
            rate = float(item["fundingRate"])
        except (KeyError, TypeError, ValueError):
            continue
        if _between(ts, _ms(start), _ms(end)):
            out.append((ts, rate))
    return sorted(set(out))


def nearest_prior(points: Iterable[tuple[int, float]], timestamp_ms: int, max_age_ms: int) -> float | None:
    best: tuple[int, float] | None = None
    for ts, value in points:
        if ts <= timestamp_ms and (best is None or ts > best[0]):
            best = (ts, value)
    if best is None or timestamp_ms - best[0] > max_age_ms:
        return None
    return best[1]


def build_derivatives_context(
    observed_at: str,
    timestamp_ms: int,
    funding_points: Iterable[tuple[int, float]] = (),
    open_interest_points: Iterable[tuple[int, float]] = (),
    max_age_ms: int = 8 * 60 * 60 * 1000,
) -> HistoricalContext:
    funding = nearest_prior(funding_points, timestamp_ms, max_age_ms)
    oi_now = nearest_prior(open_interest_points, timestamp_ms, max_age_ms)
    oi_prev = nearest_prior(open_interest_points, timestamp_ms - 300_000, max_age_ms)
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    if funding is not None:
        values["funding_rate"] = funding
        sources["funding_rate"] = "exchange_historical_funding"
    if oi_now is not None and oi_prev is not None and oi_prev != 0:
        values["open_interest_change"] = oi_now / oi_prev - 1.0
        sources["open_interest_change"] = "binance_historical_open_interest"
    return make_context(observed_at, values, sources)
