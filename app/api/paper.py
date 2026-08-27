from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ai.paper_trading import PaperSession, TradingMode

router = APIRouter(prefix="/api/paper", tags=["paper-trading"])

_sessions: dict[str, PaperSession] = {}


class SessionRequest(BaseModel):
    session_id: str = Field(min_length=3, max_length=64)
    mode: TradingMode = TradingMode.PAPER


class OrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)


@router.get("/status")
def status():
    return {
        "engine": "HHHAI Paper/Testnet Operation",
        "supported_modes": [TradingMode.PAPER.value, TradingMode.TESTNET.value],
        "live_trading": False,
        "execution_authority": False,
        "purpose": "live-market validation without real-money execution",
    }


@router.post("/sessions")
def create_session(request: SessionRequest):
    if request.session_id in _sessions:
        raise HTTPException(409, "Session already exists")
    session = PaperSession(request.session_id, request.mode)
    _sessions[request.session_id] = session
    return session.status()


@router.post("/sessions/{session_id}/start")
def start_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    session.start()
    return session.status()


@router.post("/sessions/{session_id}/stop")
def stop_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    session.stop()
    return session.status()


@router.get("/sessions/{session_id}")
def session_status(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.status()


@router.post("/sessions/{session_id}/orders")
def paper_order(session_id: str, request: OrderRequest):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if not session.running:
        raise HTTPException(409, "Session is not running")
    order = session.execution.submit(
        request.symbol, request.side, request.quantity, request.price
    )
    return {
        "order_id": order.order_id,
        "status": order.status,
        "mode": session.mode.value,
        "live_exchange_order": False,
    }


@router.get("/sessions/{session_id}/snapshot")
def snapshot(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.execution.snapshot()
