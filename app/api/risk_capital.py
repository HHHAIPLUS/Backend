from __future__ import annotations

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from ai.portfolio_risk import Exposure
from ai.risk_capital_engine import AccountState, MarketSafety, RiskCapitalEngine
from app.api.admin import require_admin
from app.persistence.repository import load_risk_controls, record_event, upsert_risk_control

router = APIRouter(prefix="/api/risk", tags=["risk-capital"])
engine = RiskCapitalEngine()


class ExposureIn(BaseModel):
    symbol: str
    side: str
    notional: float = Field(ge=0)
    beta_to_btc: float = 0.0
    beta_to_eth: float = 0.0
    volatility: float = 0.0


class EvaluateRequest(BaseModel):
    account: AccountState
    market: MarketSafety
    proposed_risk_pct: float = 0.0
    proposed_notional: float = 0.0
    exposures: list[ExposureIn] = Field(default_factory=list)
    exchange: str = "unknown"
    model_confidence: float | None = Field(default=None, ge=0, le=1)


async def hydrate_risk_controls() -> None:
    try:
        for row in await load_risk_controls():
            scope = str(row.get("scope", ""))
            if not row.get("enabled"):
                continue
            reason = str(row.get("reason") or "Persisted risk control")
            if scope == "global":
                engine.engage_global_kill(reason)
            elif scope.startswith("exchange:"):
                engine.engage_exchange_kill(scope.split(":", 1)[1], reason)
    except Exception:
        # Fail safe: a persistence outage never clears an in-memory kill switch.
        return


@router.get("/status")
def status():
    return {"engine": "HHHAI Independent Risk & Capital Intelligence", "policy": engine.policy.__dict__, "kills": engine.kill_status(), "execution_authority": False}


@router.post("/evaluate")
def evaluate(request: EvaluateRequest):
    exposures = [Exposure(**x.model_dump()) for x in request.exposures]
    result = engine.evaluate(request.account, request.market, request.proposed_risk_pct, request.proposed_notional, exposures, request.exchange, request.model_confidence)
    return {**result.__dict__, "execution_authority": False}


@router.post("/kill/global/engage")
async def engage_global(reason: str = "Manual global emergency stop", x_hhhai_admin_token: str | None = Header(default=None)):
    require_admin(x_hhhai_admin_token)
    engine.engage_global_kill(reason)
    await upsert_risk_control("global", True, reason)
    await record_event("risk_global_kill_engaged", {"reason": reason})
    return engine.kill_status()


@router.post("/kill/global/reset")
async def reset_global(x_hhhai_admin_token: str | None = Header(default=None)):
    require_admin(x_hhhai_admin_token)
    engine.reset_global_kill()
    await upsert_risk_control("global", False, "")
    await record_event("risk_global_kill_reset", {})
    return engine.kill_status()


@router.post("/kill/exchange/{exchange}/engage")
async def engage_exchange(exchange: str, reason: str = "Manual exchange emergency stop", x_hhhai_admin_token: str | None = Header(default=None)):
    require_admin(x_hhhai_admin_token)
    scope = f"exchange:{exchange.lower()}"
    engine.engage_exchange_kill(exchange, reason)
    await upsert_risk_control(scope, True, reason)
    await record_event("risk_exchange_kill_engaged", {"exchange": exchange.lower(), "reason": reason})
    return engine.kill_status()


@router.post("/kill/exchange/{exchange}/reset")
async def reset_exchange(exchange: str, x_hhhai_admin_token: str | None = Header(default=None)):
    require_admin(x_hhhai_admin_token)
    engine.reset_exchange_kill(exchange)
    await upsert_risk_control(f"exchange:{exchange.lower()}", False, "")
    await record_event("risk_exchange_kill_reset", {"exchange": exchange.lower()})
    return engine.kill_status()
