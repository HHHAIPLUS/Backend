from app.api.adversarial import router as adversarial_router
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from app.services.monitor_worker import monitor
from fastapi.middleware.cors import CORSMiddleware
import os
from app.api.health import router as health_router
from app.api.status import router as status_router
from app.api.integration import router as integration_router
from app.api.admin import router as admin_router
from app.api.adaptive import router as adaptive_router, hydrate_adaptive
from app.api.council import router as council_router
from app.api.realtime import router as realtime_router
from app.api.positions import router as positions_router
from app.api.scenarios import router as scenarios_router
from app.api.learning import router as learning_router, hydrate_learning
from app.api.simulation import router as simulation_router
from app.api.paper import router as paper_router
from app.api.stress import router as stress_router
from app.api.capital import router as capital_router
from app.api.portfolio import router as portfolio_router
from app.api.control_center import router as control_center_router
from app.api.trade_optimizer import router as trade_optimizer_router
from app.api.performance import router as performance_router
from app.api.model import router as model_router
from app.api.markets import router as markets_router
from app.api.trading import router as trading_router
from app.api.market_intelligence import router as market_intelligence_router
from app.api.stage5 import router as stage5_router
from app.ml.model_persistence import hydrate_model
from ai.autonomous_trader import trader
from ai.position_intelligence import install_stage6_position_intelligence
from app.core.config import settings

install_stage6_position_intelligence(trader)

@asynccontextmanager
async def lifespan(app):
    await hydrate_learning()
    await hydrate_adaptive()
    await hydrate_model()
    task = asyncio.create_task(monitor.run())
    if os.getenv("HHHAI_AUTOTRADING_ENABLED", "false").lower() == "true":
        await trader.start()
    try:
        yield
    finally:
        monitor.stop()
        await task
        if trader.running:
            await trader.stop()

app = FastAPI(title=settings.app_name, version='1.0.0', description='HHHAI backend — cumulative Stage 6', lifespan=lifespan)
allowed_origins = [x.strip() for x in (os.getenv('HHHAI_CORS_ORIGINS') or settings.cors_origins).split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.include_router(adversarial_router); app.include_router(model_router); app.include_router(markets_router); app.include_router(trading_router); app.include_router(market_intelligence_router)
app.include_router(simulation_router); app.include_router(paper_router); app.include_router(stress_router); app.include_router(capital_router); app.include_router(portfolio_router); app.include_router(control_center_router); app.include_router(trade_optimizer_router); app.include_router(performance_router)
app.include_router(health_router); app.include_router(status_router); app.include_router(integration_router); app.include_router(admin_router); app.include_router(adaptive_router); app.include_router(council_router); app.include_router(realtime_router); app.include_router(positions_router); app.include_router(scenarios_router); app.include_router(learning_router); app.include_router(stage5_router)

@app.get('/')
def root():
    return {'name': settings.app_name, 'version': settings.app_version if hasattr(settings, 'app_version') else '1.0.0', 'phase': 'Stage 6 - Autonomous Position Intelligence', 'live_trading_enabled': settings.live_trading_enabled, 'execution_authority': False, 'mode': settings.app_env}
