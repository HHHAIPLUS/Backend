from fastapi import APIRouter, HTTPException
from app.core.config import settings
from ai.agents import AgentContext, IntelligenceCouncil
from ai.adaptive_engine import AdaptivePositionEngine
from app.api.realtime import build_world_intelligence
from app.api.positions import manager
from ai.decision_fusion import DecisionFusion
from app.ml.predictive import predictive_model

router = APIRouter(prefix="/api/control-center", tags=["control-center"])
_council = IntelligenceCouncil()
_adaptive = AdaptivePositionEngine()

@router.get("/status")
def control_center_status():
    return {
        "engine": "HHHAI Autonomous Trading Control Center",
        "phase": 18,
        "state": "WATCHING",
        "live_trading_enabled": settings.live_trading_enabled,
        "execution_authority": False,
        "real_money": False,
        "validated_model": predictive_model.model is not None,
        "backend_driven": True,
        "continuous_monitoring": True,
        "fixed_take_profit_required": False,
        "decision_layers": [
            "real_time_market_intelligence", "multi_agent_council",
            "scenario_engine", "adversarial_intelligence",
            "adaptive_position_management", "controlled_learning",
            "portfolio_risk", "capital_guard", "trade_quality_optimizer", "performance_feedback",
        ],
        "safety": {"stale_data_block": True, "kill_switch": True, "production_execution": False},
    }

@router.get("/overview/{symbol}")
def control_center_overview(symbol: str = "BTCUSDT"):
    try:
        world = build_world_intelligence(symbol).model_dump(mode="json")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    m = world["market"]
    imbalance = float(m.get("order_book_imbalance") or 0)
    buying = max(0.0, min(1.0, 0.5 + imbalance / 2))
    selling = max(0.0, min(1.0, 0.5 - imbalance / 2))
    regime = world.get("regime") or "unknown"
    ctx = AgentContext(
        symbol=symbol,
        momentum=max(-1.0, min(1.0, float(m.get("momentum_proxy") or 0))),
        trend_strength=max(0.0, min(1.0, float(m.get("trend_strength") or .5))),
        buying_pressure=buying,
        selling_pressure=selling,
        volatility=max(0.0, min(1.0, float(m.get("volatility_proxy") or 0))),
        liquidity_stress=max(0.0, min(1.0, float(m.get("liquidity_stress") or 0))),
        news_risk=max(0.0, min(1.0, float(world.get("news_risk") or 0))),
        news_sentiment=max(-1.0, min(1.0, float(world.get("news_sentiment") or 0))),
        news_credibility=max(0.0, min(1.0, float(world.get("news_credibility") or .7))),
        funding_bias=max(-1.0, min(1.0, float(m.get("funding_rate") or 0) * 100)),
        open_interest_change=max(-1.0, min(1.0, float(m.get("open_interest_change") or 0))),
        correlation_risk=max(0.0, min(1.0, float(world.get("market_risk") or 0))),
        market_regime=regime,
    )
    council = _council.deliberate(ctx).model_dump(mode="json")
    flags = world.get("danger_flags") or []
    if flags or council["veto_flags"]:
        action = "WAIT"
        reason = "Safety or uncertainty conditions require observation before any new action."
    elif council["action"] == "bullish":
        action = "BULLISH BIAS"
        reason = "The intelligence council currently has a positive directional bias."
    elif council["action"] == "bearish":
        action = "BEARISH BIAS"
        reason = "The intelligence council currently has a negative directional bias."
    else:
        action = "HOLD / WAIT"
        reason = "The evidence is mixed and HHHAI is not forcing a directional conclusion."

    predictive = predictive_model.predict({
        "return_1": float(m.get("price_change_24h") or 0),
        "range_pct": float(m.get("volatility_proxy") or 0),
        "order_book_imbalance": imbalance,
        "funding_rate": float(m.get("funding_rate") or 0),
        "open_interest_change": float(m.get("open_interest_change") or 0),
        "news_risk": float(world.get("news_risk") or 0),
        "news_sentiment": float(world.get("news_sentiment") or 0),
        "volatility_proxy": float(m.get("volatility_proxy") or 0),
        "trend_strength": float(world.get("trend_strength") or 0),
        "momentum": float(world.get("momentum_proxy") or 0),
        "liquidity_stress": float(world.get("liquidity_stress") or 0),
    })
    fusion = DecisionFusion().decide(council_action=council["action"], council_confidence=council["confidence"], disagreement=council["disagreement"], predictive=predictive, adversarial_block=bool("adversarial_risk_off" in council["veto_flags"]), scenario_uncertainty=0.5, data_quality=float(world.get("data_quality") or 0), risk_vetoes=council["veto_flags"])

    return {
        "state": "WATCHING",
        "symbol": symbol,
        "price": m.get("price"),
        "action": action,
        "final_decision": fusion.__dict__,
        "predictive_model": predictive,
        "risk": council["risk_level"],
        "confidence": council["confidence"],
        "reason": reason,
        "market": world,
        "council": council,
        "position_count": len(manager.positions),
        "execution_authority": False,
        "real_money": False,
        "validated_model": predictive_model.model is not None,
    }
