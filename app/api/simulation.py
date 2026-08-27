from fastapi import APIRouter
from pydantic import BaseModel, Field
from ai.simulation_service import SimulationService
from ai.simulation_engine import Candle, SimulationConfig

router = APIRouter(prefix="/api/simulation", tags=["simulation"])
service = SimulationService()

class CandleIn(BaseModel):
    timestamp: int
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

class ReplayRequest(BaseModel):
    candles: list[CandleIn]
    signals: list[int]
    fee_rate: float = 0.0004
    slippage_bps: float = 2.0
    funding_rate: float = 0.0001
    starting_equity: float = 10_000
    leverage: float = 1.0

@router.get("/status")
def status():
    return {
        "engine": "HHHAI Advanced Backtesting & Simulation",
        "mode": "research_only",
        "execution_authority": False,
        "supports": ["replay", "walk_forward", "monte_carlo", "fees", "slippage", "funding"],
    }

@router.post("/replay")
def replay_endpoint(request: ReplayRequest):
    candles = [Candle(**c.model_dump()) for c in request.candles]
    cfg = SimulationConfig(
        fee_rate=request.fee_rate,
        slippage_bps=request.slippage_bps,
        funding_rate=request.funding_rate,
        starting_equity=request.starting_equity,
        leverage=request.leverage,
    )
    return service.run_replay(candles, request.signals, cfg)
