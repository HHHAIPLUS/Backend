from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.market_data.historical import provider
from app.market_data.intelligence import MarketBar, PointInTimeContext, build_market_state
from app.market_data.news import aggregate_news
from app.market_data.realtime import BinancePublicFeed, BitgetPublicFeed

router = APIRouter(prefix="/api/market-intelligence", tags=["market-intelligence"])


def _realtime(exchange: str, symbol: str):
    if exchange == "binance":
        return BinancePublicFeed().snapshot(symbol)
    if exchange == "bitget":
        return BitgetPublicFeed().snapshot(symbol)
    raise HTTPException(status_code=400, detail="exchange must be binance or bitget")


@router.get("/snapshot")
def snapshot(
    exchange: str = Query("binance"),
    symbol: str = Query("BTCUSDT", min_length=3),
):
    exchange = exchange.lower()
    snap = _realtime(exchange, symbol.upper())
    bars = provider(exchange).candles(symbol.upper(), interval="5m", limit=100)
    context = PointInTimeContext(
        timestamp=snap.observed_at,
        funding_rate=snap.funding_rate,
        open_interest=snap.open_interest,
        order_book_imbalance=snap.order_book_imbalance,
        spread_bps=((snap.ask - snap.bid) / ((snap.ask + snap.bid) / 2) * 10000 if snap.bid > 0 and snap.ask > 0 else None),
    )
    state = build_market_state(symbol.upper(), {"5m": bars}, context, observed_at=snap.observed_at)
    return state.model_dump(mode="json")


@router.get("/health")
def intelligence_health():
    return {"status": "ok", "schema": "market-state-v1", "point_in_time": True, "live_schema_shared": True}
