from fastapi import APIRouter, Header, HTTPException
from app.api.admin import require_admin
from ai.autonomous_trader import trader

router = APIRouter(prefix="/api/trading", tags=["autonomous-trading"])

@router.get("/status")
def status():
    return trader.status()

@router.post("/cycle")
async def cycle(symbol: str = "BTCUSDT", x_hhhai_admin_token: str | None = Header(default=None)):
    # Manual cycles are harmless in paper mode, but every non-paper cycle is
    # authenticated because it can reach a configured exchange execution path.
    if trader.execution_mode != "paper":
        require_admin(x_hhhai_admin_token)
    try:
        return await trader.run_cycle(symbol.upper())
    except Exception as exc:
        raise HTTPException(503, f"Autonomous cycle failed: {type(exc).__name__}: {exc}")

@router.post("/start")
async def start(x_hhhai_admin_token: str | None = Header(default=None)):
    require_admin(x_hhhai_admin_token)
    return await trader.start()

@router.post("/stop")
async def stop(x_hhhai_admin_token: str | None = Header(default=None)):
    require_admin(x_hhhai_admin_token)
    return await trader.stop()


@router.get('/management')
def management():
    return {
        'engine': trader.exit_engine.version,
        'continuous': True,
        'last_reviews': trader.last_position_management[-50:],
    }
