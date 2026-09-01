from fastapi.testclient import TestClient

from app.main import app


def test_market_intelligence_health_contract():
    client = TestClient(app)
    response = client.get("/api/market-intelligence/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["point_in_time"] is True
    assert set(payload["supported_exchanges"]) == {"binance", "bitget"}
    assert "trade_flow" in payload["context"]
