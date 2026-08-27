from fastapi import APIRouter
from pydantic import BaseModel, Field

from ai.performance_optimizer import PerformanceOptimizer, TradeRecord

router = APIRouter(prefix="/api/performance", tags=["performance-optimization"])
optimizer = PerformanceOptimizer()


class TradeIn(BaseModel):
    trade_id: str
    symbol: str
    side: str
    entry: float = Field(gt=0)
    exit: float = Field(gt=0)
    highest_favorable: float
    lowest_adverse: float
    expected_direction: str
    actual_direction: str
    pnl: float
    fees: float = Field(default=0, ge=0)
    reason: str = ""


class AnalysisRequest(BaseModel):
    trades: list[TradeIn]


@router.get("/status")
def status():
    return {
        "engine": "HHHAI Trading Intelligence Optimization II",
        "capabilities": [
            "exit-quality measurement",
            "missed-opportunity analysis",
            "false-signal analysis",
            "performance attribution",
            "profit-factor optimization",
        ],
        "execution_authority": False,
        "live_trading": False,
    }


@router.post("/analyze")
def analyze(request: AnalysisRequest):
    trades = [TradeRecord(**t.model_dump()) for t in request.trades]
    return optimizer.analyze(trades)
