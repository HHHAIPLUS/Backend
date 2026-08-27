from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(prefix="/api/status", tags=["status"])

@router.get("")
def status():
    return {
        "service": "hhhai-backend",
        "environment": settings.app_env,
        "live_trading_enabled": settings.live_trading_enabled,
        "components": {
            "api": "ready",
            "market_data": "phase_9_realtime",
            "exchange_adapters": "foundation",
            "trading_engine": "foundation",
            "risk_engine": "foundation",
            "ai": "phase_9_realtime_intelligence",
        },
    }
