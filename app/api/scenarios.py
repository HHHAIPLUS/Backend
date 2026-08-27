from fastapi import APIRouter
from ai.scenario_engine import ScenarioEngine, ScenarioRequest

router = APIRouter(prefix="/api/scenarios", tags=["prediction-scenarios"])
_engine = ScenarioEngine()


@router.get("/status")
def scenario_status():
    return {
        "phase": 11,
        "engine": "HHHAI Advanced Prediction & Scenario Engine",
        "scenario_count": 5,
        "multiple_futures": True,
        "uncertainty_tracking": True,
        "execution_authority": False,
        "fixed_target_required": False,
        "model_version": _engine.model_version,
    }


@router.post("/analyze")
def analyze(request: ScenarioRequest):
    return _engine.generate(request).model_dump(mode="json")
