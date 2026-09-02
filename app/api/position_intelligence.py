from fastapi import APIRouter
from ai.autonomous_trader import trader

router = APIRouter(prefix="/api/position-intelligence", tags=["stage6-position-intelligence"])


@router.get("/status")
def status():
    return {
        "engine": getattr(trader, "stage6_version", "not-installed"),
        "continuous_review": True,
        "position_state_persistence": True,
        "thesis_tracking": True,
        "expected_continuation_value": True,
        "dynamic_downside_risk": True,
        "thesis_invalidation": True,
        "market_shock_detection": True,
        "dynamic_hold_reduce_exit": True,
        "adaptive_profit_protection": True,
        "partial_exits": True,
        "exchange_reconciliation": True,
        "restart_recovery": getattr(trader, "_stage6_hydration_installed", False),
        "duplicate_order_protection": True,
        "execution_authority": getattr(trader, "stage6_last_decision", None) is not None and trader.execution_mode in {"testnet", "live"} and trader._execution_gate()[0],
        "live_money_enabled": False,
        "reviews": getattr(trader, "stage6_reviews", 0),
        "last_decision": getattr(trader, "stage6_last_decision", None),
        "last_error": getattr(trader, "stage6_last_error", None),
    }
