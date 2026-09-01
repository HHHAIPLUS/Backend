from app.main import app


def _routes():
    return {(route.path, method) for route in app.routes for method in getattr(route, "methods", set())}


def test_frontend_required_read_contracts_are_registered():
    routes = _routes()
    required = {
        ("/api/status", "GET"),
        ("/api/trading/status", "GET"),
        ("/api/realtime/world/{symbol}", "GET"),
        ("/api/trading/management", "GET"),
        ("/api/learning/status", "GET"),
        ("/api/control-center/status", "GET"),
        ("/api/integration/status", "GET"),
        ("/api/model/status", "GET"),
        ("/api/markets", "GET"),
        ("/api/health", "GET"),
    }
    missing = sorted(required - routes)
    assert not missing, f"Frontend/backend contract routes missing: {missing}"


def test_health_contract_is_explicit():
    routes = _routes()
    assert ("/api/health", "GET") in routes
