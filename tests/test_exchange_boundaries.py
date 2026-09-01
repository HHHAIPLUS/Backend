import pytest

from app.exchanges.binance import BinanceAdapter
from app.exchanges.bitget import BitgetAdapter


@pytest.mark.asyncio
async def test_binance_protection_payload_is_fail_closed(monkeypatch):
    adapter = BinanceAdapter(testnet=True)
    calls = []

    async def fake_request(method, path, params=None, signed=False):
        calls.append((method, path, params, signed))
        return {"orderId": len(calls)}

    monkeypatch.setattr(adapter, "_request", fake_request)
    result = await adapter.place_protection("BTCUSDT", "long", 0.01, 100.0, None)

    assert result["attached"] is True
    assert len(calls) == 1
    method, path, params, signed = calls[0]
    assert method == "POST"
    assert path == "/fapi/v1/order"
    assert signed is True
    assert params["type"] == "STOP_MARKET"
    assert params["side"] == "SELL"
    assert params["closePosition"] == "true"
    assert params["workingType"] == "MARK_PRICE"


@pytest.mark.asyncio
async def test_binance_close_long_is_reduce_only(monkeypatch):
    adapter = BinanceAdapter(testnet=True)
    calls = []

    async def fake_request(method, path, params=None, signed=False):
        calls.append((method, path, params, signed))
        return {"orderId": 1}

    monkeypatch.setattr(adapter, "_request", fake_request)
    await adapter.close_position("BTCUSDT", "long", 0.01)

    params = calls[0][2]
    assert params["side"] == "SELL"
    assert params["reduceOnly"] == "true"
    assert params["type"] == "MARKET"


@pytest.mark.asyncio
async def test_bitget_close_short_uses_reduce_only_close_side(monkeypatch):
    adapter = BitgetAdapter(testnet=True)
    captured = {}

    async def fake_request(method, path, params=None, body=None, private=False):
        captured.update(method=method, path=path, params=params, body=body, private=private)
        return {"orderId": "1"}

    monkeypatch.setattr(adapter, "_request", fake_request)
    await adapter.close_position("BTCUSDT", "short", 0.01)

    assert captured["method"] == "POST"
    assert captured["path"] == "/api/v2/mix/order/place-order"
    assert captured["private"] is True
    assert captured["body"]["side"] == "buy"
    assert captured["body"]["reduceOnly"] == "YES"
    assert captured["body"]["orderType"] == "market"


@pytest.mark.asyncio
async def test_exchange_adapters_fail_without_credentials(monkeypatch):
    monkeypatch.setattr("app.exchanges.binance.settings.binance_api_key", "")
    monkeypatch.setattr("app.exchanges.binance.settings.binance_api_secret", "")
    with pytest.raises(RuntimeError, match="credentials"):
        BinanceAdapter(testnet=True)._signed("GET", "/private")

    monkeypatch.setattr("app.exchanges.bitget.settings.bitget_api_key", "")
    monkeypatch.setattr("app.exchanges.bitget.settings.bitget_api_secret", "")
    monkeypatch.setattr("app.exchanges.bitget.settings.bitget_passphrase", "")
    with pytest.raises(RuntimeError, match="credentials"):
        BitgetAdapter(testnet=True)._headers("GET", "/private")
