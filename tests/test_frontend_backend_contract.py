from pathlib import Path


def test_frontend_required_read_contracts_are_registered():
    """Verify the frontend contract exists in the router modules and is wired by app.main."""
    from app.api import control_center, health, integration, learning, markets, model, realtime, status, trading

    expected = {
        "/api/status": status.router,
        "/api/trading/status": trading.router,
        "/api/realtime/world/{symbol}": realtime.router,
        "/api/trading/management": trading.router,
        "/api/learning/status": learning.router,
        "/api/control-center/status": control_center.router,
        "/api/integration/status": integration.router,
        "/api/model/status": model.router,
        "/api/markets": markets.router,
        "/api/health": health.router,
    }

    for path, router in expected.items():
        paths = {route.path for route in router.routes}
        assert path in paths, f"Contract route missing from its router: {path}"

    main_source = Path("app/main.py").read_text(encoding="utf-8")
    required_router_names = {
        "health_router", "status_router", "integration_router", "learning_router",
        "control_center_router", "model_router", "markets_router", "trading_router",
        "realtime_router",
    }
    for name in required_router_names:
        assert f"app.include_router({name})" in main_source, f"Router not wired in app.main: {name}"


def test_health_contract_is_explicit():
    from app.api.health import router
    assert any(route.path == "/api/health" and "GET" in route.methods for route in router.routes)
