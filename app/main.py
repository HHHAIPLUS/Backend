from app.api.adversarial import router as adversarial_router
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from app.services.monitor_worker import monitor
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
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
from app.api.research import router as research_router, hydrate_research
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
from app.api.position_intelligence import router as position_intelligence_router
from app.api.risk_capital import router as risk_capital_router
from app.ml.model_persistence import hydrate_model
from app.ml.predictive_brain import predictive_brain
from app.ml.binance_historical import fetch_binance_klines
from app.ml.bootstrap import build_dataset
from ai.autonomous_trader import trader
from ai.position_intelligence import install_stage6_position_intelligence
from ai.stage6_hydration import install_stage6_hydration
from ai.stage8_integration import install_stage8_risk, hydrate_stage8_risk
from ai.multi_coin_selection import install_multi_coin_selection
from app.market_data.binance_central import CentralBinanceMarketData
from app.market_data.bitget_central import CentralBitgetMarketData
from app.market_data import realtime as realtime_market_data
from app.market_data.binance_user_stream import binance_user_stream
from app.services.binance_execution_guard import install_binance_execution_guard
from app.exchanges.factory import adapters
from app.core.config import settings

log = logging.getLogger(__name__)

realtime_market_data.BitgetPublicFeed.snapshot = lambda self, symbol: CentralBitgetMarketData.snapshot(symbol)

CentralBinanceMarketData.install()
install_binance_execution_guard(trader)
trader.position_review_interval = max(2, trader.position_review_interval)
install_stage6_position_intelligence(trader)
install_stage6_hydration(trader)
stage8_risk = install_stage8_risk(trader)
install_multi_coin_selection(trader)

@asynccontextmanager
async def lifespan(app):
    if os.getenv("HHHAI_AUTO_BOOTSTRAP_BRAIN", "false").lower() == "true" and predictive_brain.bundle is None:
        try:
            symbol = os.getenv("HHHAI_BRAIN_BOOTSTRAP_SYMBOL", "BTCUSDT").upper()
            limit = max(5000, min(10000, int(os.getenv("HHHAI_BRAIN_BOOTSTRAP_CANDLES", "8000"))))
            raw = await asyncio.to_thread(fetch_binance_klines, symbol, "5m", limit)
            rows = build_dataset(raw, horizon=6, threshold=float(os.getenv("HHHAI_BRAIN_LABEL_THRESHOLD", "0.0025")))
            report = await asyncio.to_thread(predictive_brain.train, rows, f"brain-{symbol}-5m")
            log.warning("PREDICTIVE_BRAIN_BOOTSTRAP status=%s version=%s reason=%s", report.status, report.version, report.reason)
        except Exception as exc:
            log.error("PREDICTIVE_BRAIN_BOOTSTRAP_FAILED %s", exc)
    await hydrate_learning()
    await hydrate_adaptive()
    await hydrate_research()
    await hydrate_model()
    await hydrate_stage8_risk(stage8_risk)
    exchange = os.getenv("HHHAI_EXECUTION_EXCHANGE", os.getenv("HHHAI_MARKET_EXCHANGE", "binance")).lower()
    log.warning("HHHAI_RUNTIME exchange=%s bitget_testnet=%s trading_mode=%s live_enabled=%s autotrading=%s", exchange, settings.bitget_testnet, settings.hhhai_trading_mode, settings.live_trading_enabled, settings.hhhai_autotrading_enabled)
    if exchange == "bitget":
        try:
            account = await adapters()["bitget"].get_account_status()
            log.warning("BITGET_AUTH_CHECK_OK available_balance=%s", account.get("available_balance"))
            try:
                positions = await adapters()["bitget"].get_positions("DOGEUSDT")
                log.warning("TEST10_POSITION_CHECK %s", positions)
                if os.getenv("HHHAI_TEST10_CLEANUP_EXISTING", "true").lower() == "true":
                    for row in positions or []:
                        if float(row.get("total") or 0) > 0:
                            mode = await adapters()["bitget"].get_position_mode("DOGEUSDT")
                            cleanup = await adapters()["bitget"].close_position("DOGEUSDT", str(row.get("holdSide") or "long"), float(row.get("total")), mode)
                            log.warning("TEST10_EXISTING_POSITION_CLOSED %s", cleanup)
                    positions = await adapters()["bitget"].get_positions("DOGEUSDT")
                    log.warning("TEST10_POSITION_AFTER_CLEANUP %s", positions)
            except Exception as exc:
                log.error("TEST10_POSITION_CHECK_FAILED %s", exc)
        except Exception as exc:
            log.error("BITGET_AUTH_CHECK_FAILED %s", exc)
    if exchange == "binance":
        CentralBinanceMarketData._start_universe()
    task = asyncio.create_task(monitor.run())
    if exchange == "binance" and trader.execution_mode in {"live", "testnet"}:
        binance_user_stream.start()
    if exchange == "bitget" and trader.execution_mode == "live" and os.getenv("HHHAI_TEST10_RUN_ON_START", "false").lower() == "true":
        try:
            canary_cycle = await trader.run_test10_canary("DOGEUSDT")
            log.warning("TEST10_CANARY_COMPLETE action=%s execution=%s management=%s remaining_open=%s", canary_cycle.get("action"), canary_cycle.get("execution", {}).get("status"), bool(canary_cycle.get("management_observed")), len(canary_cycle.get("remaining_open", [])))
        except Exception as exc:
            log.error("TEST10_CANARY_CYCLE_FAILED %s", exc)
    if os.getenv("HHHAI_AUTOTRADING_ENABLED", "false").lower() == "true":
        await trader.start()
        log.warning("AUTOTRADER_START_OK mode=%s", trader.execution_mode)
    try:
        yield
    finally:
        monitor.stop()
        await task
        if trader.running:
            await trader.stop()
        binance_user_stream.stop()

app = FastAPI(title=settings.app_name, version='1.0.0', description='HHHAI backend — cumulative Stage 8', lifespan=lifespan)
allowed_origins = [x.strip() for x in (os.getenv('HHHAI_CORS_ORIGINS') or settings.cors_origins).split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.include_router(adversarial_router); app.include_router(model_router); app.include_router(markets_router); app.include_router(trading_router); app.include_router(market_intelligence_router)
app.include_router(simulation_router); app.include_router(paper_router); app.include_router(stress_router); app.include_router(capital_router); app.include_router(portfolio_router); app.include_router(control_center_router); app.include_router(trade_optimizer_router); app.include_router(performance_router)
app.include_router(health_router); app.include_router(status_router); app.include_router(integration_router); app.include_router(admin_router); app.include_router(adaptive_router); app.include_router(council_router); app.include_router(realtime_router); app.include_router(positions_router); app.include_router(scenarios_router); app.include_router(learning_router); app.include_router(research_router); app.include_router(stage5_router); app.include_router(position_intelligence_router); app.include_router(risk_capital_router)

@app.get('/')
def root():
    return {'name': settings.app_name, 'version': settings.app_version if hasattr(settings, 'app_version') else '1.0.0', 'phase': 'Stage 8 - Risk & Capital Intelligence', 'live_trading_enabled': settings.live_trading_enabled, 'execution_authority': False, 'mode': settings.app_env, 'risk_engine': 'independent'}
