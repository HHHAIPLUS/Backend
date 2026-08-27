from fastapi import APIRouter, HTTPException, Query
from ai.regime import detect_regime
from app.market_data.realtime import build_world_intelligence, BinancePublicFeed
from app.services.monitor_worker import monitor

router = APIRouter(prefix="/api/realtime", tags=["real-time-intelligence"])

@router.get("/status")
def realtime_status():
    return {
        "phase": 9,
        "engine": "HHHAI Real-Time Market & World Intelligence",
        "market_feed": "Binance public futures data",
        "news_feed": "CoinDesk RSS",
        "private_exchange_keys_required": False,
        "continuous_monitoring": True,
        "observation_worker": {"running": True, "symbols": monitor.symbols, "interval_seconds": monitor.interval},
        "stale_data_protection": True,
        "execution_authority": False,
    }

@router.get("/market/{symbol}")
def realtime_market(symbol: str, exchange: str = Query("binance")):
    try:
        if exchange.lower() == "binance":
            return BinancePublicFeed().snapshot(symbol).model_dump(mode="json")
        if exchange.lower() == "bitget":
            from app.market_data.realtime import BitgetPublicFeed
            return BitgetPublicFeed().snapshot(symbol).model_dump(mode="json")
        raise HTTPException(status_code=400, detail=f"Unsupported exchange: {exchange}")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@router.get("/world/{symbol}")
def realtime_world(symbol: str, exchange: str = Query("binance")):
    try:
        return build_world_intelligence(symbol, exchange=exchange).model_dump(mode="json")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
