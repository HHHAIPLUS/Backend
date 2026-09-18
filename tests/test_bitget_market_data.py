import json
import time

from app.market_data.bitget_central import CentralBitgetMarketData, _State
from app.exchanges.bitget import BitgetAdapter


def test_bitget_websocket_ticker_and_book_parsing():
    state = _State("BTCUSDT")
    CentralBitgetMarketData._process(
        state,
        json.dumps({
            "arg": {"channel": "ticker"},
            "data": [{
                "lastPr": "77000",
                "markPrice": "77001",
                "bidPr": "76999",
                "askPr": "77001",
                "bidSz": "2",
                "askSz": "3",
                "quoteVolume": "1000000",
                "change24h": "0.01",
                "fundingRate": "0.0001",
                "holdingAmount": "123",
            }],
        }),
    )
    CentralBitgetMarketData._process(
        state,
        json.dumps({
            "arg": {"channel": "books5"},
            "data": [{"b": [["76999", "2"], ["76998", "1"]], "a": [["77001", "3"], ["77002", "2"]]}],
        }),
    )
    assert state.price == 77000
    assert state.bid == 76999
    assert state.ask == 77001
    assert state.bid_qty == 3
    assert state.ask_qty == 5
    assert state.open_interest == 123


def test_bitget_rate_limiter_is_conservative():
    assert BitgetAdapter._max_requests_per_second <= 2
    assert BitgetAdapter._max_requests_per_minute <= 120
    assert BitgetAdapter._max_requests_per_minute < 6000


def test_bitget_ws_state_does_not_depend_on_rest_for_live_updates():
    state = _State("BTCUSDT")
    CentralBitgetMarketData._process(
        state,
        json.dumps({
            "arg": {"channel": "ticker"},
            "data": [{"lastPr": "77000", "bidPr": "76999", "askPr": "77001"}],
        }),
    )
    assert state.ready.is_set()
    assert time.time() - state.last_update < 2


def test_bitget_account_status_normalizes_usdt_balance(monkeypatch):
    async def fake_request(self, method, path, params=None, body=None, private=False):
        return [{"marginCoin": "USDT", "available": "1.75", "accountEquity": "2.00"}]
    monkeypatch.setattr(BitgetAdapter, "_request", fake_request)
    import asyncio
    result = asyncio.run(BitgetAdapter(testnet=False).get_account_status())
    assert result["available_balance"] == 1.75
    assert result["total_wallet_balance"] == 2.0
