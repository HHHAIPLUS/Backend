from fastapi import APIRouter
from pydantic import BaseModel, Field

from ai.portfolio_risk import Exposure, PortfolioRiskEngine

router = APIRouter(prefix="/api/portfolio", tags=["portfolio-risk"])
engine = PortfolioRiskEngine()


class ExposureIn(BaseModel):
    symbol: str
    side: str
    notional: float = Field(gt=0)
    beta_to_btc: float = 0.0
    beta_to_eth: float = 0.0
    volatility: float = 0.0


class PortfolioRequest(BaseModel):
    equity: float = Field(gt=0)
    exposures: list[ExposureIn]


@router.get("/status")
def status():
    return {
        "engine": "HHHAI Portfolio & Market-Wide Risk",
        "execution_authority": False,
        "live_trading": False,
        "capabilities": [
            "gross exposure",
            "net exposure",
            "single-asset concentration",
            "correlation clusters",
            "BTC/ETH market linkage",
        ],
    }


@router.post("/evaluate")
def evaluate(request: PortfolioRequest):
    exposures = [Exposure(**x.model_dump()) for x in request.exposures]
    return engine.evaluate(request.equity, exposures)
