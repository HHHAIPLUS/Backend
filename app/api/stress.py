from fastapi import APIRouter
from ai.stress_harness import StressHarness, standard_scenarios

router = APIRouter(prefix="/api/stress", tags=["stress"])
harness = StressHarness()


@router.get("/status")
def status():
    return {
        "engine": "HHHAI Extreme Stress & Failure Testing",
        "mode": "research_only",
        "execution_authority": False,
        "live_trading": False,
        "scenarios": [x.name for x in standard_scenarios()],
    }


@router.get("/scenarios")
def scenarios():
    return [
        {
            "name": x.name,
            "description": x.description,
            "severity": x.severity,
        }
        for x in standard_scenarios()
    ]


@router.get("/guards")
def guards():
    return {
        "stale_data": "block",
        "duplicate_orders": "reject",
        "restart_recovery": "fail_closed",
        "live_execution": "disabled",
    }
