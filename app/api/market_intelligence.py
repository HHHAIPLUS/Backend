from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.market_data.cache import MarketDataCache
from app.market_data.historical import provider
from app.market_data.intelligence import MarketBar, PointInTimeContext, build_market_state
from app.market_data.news import aggregate_news, fetch_rss
from app.market_data.realtime import BinancePublicFeed, BitgetPublicFeed

router = APIRouter(prefix="/api/market-intelligence", tags=["market-intelligence"])
_cache = MarketDataCache(ttl_seconds=30)


def _realtime(exchange: str, symbol: str):
    if exchange == "binance": return BinancePublicFeed().snapshot(symbol)
    if exchange == "bitget": return BitgetPublicFeed().snapshot(symbol)
    raise HTTPException(status_code=400, detail="exchange must be binance or bitget")


def _bars(exchange: str, symbol: str) -> dict[str, list[MarketBar]]:
    source = provider(exchange)
    result: dict[str, list[MarketBar]] = {}
    for frame in ("1m", "5m", "15m", "1H"):
        try: result[frame] = source.candles(symbol, interval=frame, limit=200)
        except Exception: result[frame] = []
    return {k: v for k, v in result.items() if v}


def _context(exchange: str, symbol: str, snap) -> PointInTimeContext:
    source = provider(exchange)
    orderbook = source.order_book(symbol)
    context = PointInTimeContext(timestamp=min(snap.observed_at, orderbook.timestamp), funding_rate=snap.funding_rate, open_interest=snap.open_interest, order_book_imbalance=orderbook.order_book_imbalance, spread_bps=orderbook.spread_bps)
    try:
        flow = source.trade_flow(symbol, limit=100); eligible = [x for x in flow if x.timestamp <= context.timestamp]
        if eligible: context.aggressive_buy_ratio = sum(x.aggressive_buy_ratio or 0 for x in eligible) / len(eligible)
    except Exception: pass
    try:
        liquidations = source.liquidations(symbol, limit=100); eligible = [x for x in liquidations if x.timestamp <= context.timestamp]
        if eligible: context.liquidation_notional = sum(x.liquidation_notional or 0 for x in eligible)
    except Exception: pass
    return context


def _cross_asset_bars(exchange: str, symbol: str) -> dict[str, list[MarketBar]]:
    source = provider(exchange); assets = {"BTCUSDT", "ETHUSDT"}
    assets.discard(symbol.upper())
    result = {}
    for asset in assets:
        try: result[asset] = source.candles(asset, interval="5m", limit=200)
        except Exception: pass
    return result


@router.get("/snapshot")
def snapshot(exchange: str = Query("binance"), symbol: str = Query("BTCUSDT", min_length=3)):
    exchange = exchange.lower(); symbol = symbol.upper()
    cache_key = f"{exchange}:{symbol}:market-state-v1"
    cached = _cache.get(cache_key)
    if cached is not None: return cached
    snap = _realtime(exchange, symbol); frames = _bars(exchange, symbol)
    if not frames: raise HTTPException(status_code=503, detail="No candle timeframe is currently available")
    context = _context(exchange, symbol, snap)
    try:
        news = fetch_rss("https://www.coindesk.com/arc/outboundfeeds/rss/", "coindesk")
        news_summary = aggregate_news(news, now=snap.observed_at)
    except Exception:
        news_summary = {"count": 0, "sentiment": 0.0, "risk": 0.0, "credibility": 0.0}
    state = build_market_state(symbol, frames, context, correlation_bars=_cross_asset_bars(exchange, symbol), news_count=int(news_summary["count"]), news_sentiment=float(news_summary["sentiment"]), news_risk=float(news_summary["risk"]), news_credibility=float(news_summary["credibility"]), observed_at=snap.observed_at)
    payload = state.model_dump(mode="json")
    _cache.set(cache_key, payload)
    return payload


@router.get("/health")
def intelligence_health():
    return {"status": "ok", "schema": "market-state-v1", "point_in_time": True, "live_schema_shared": True, "supported_exchanges": ["binance", "bitget"], "timeframes": ["1m", "5m", "15m", "1H"], "context": ["order_book", "trade_flow", "funding", "open_interest", "liquidations", "news", "cross_asset"], "degradation": "fail-closed-quality-gate"}
