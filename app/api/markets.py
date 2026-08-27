from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/markets", tags=["markets"] )

BINANCE_URL = "https://fapi.binance.com"
BITGET_URL = "https://api.bitget.com"


def _binance_symbols() -> list[dict]:
    with httpx.Client(timeout=10.0, follow_redirects=True, trust_env=True) as client:
        r = client.get(f"{BINANCE_URL}/fapi/v1/exchangeInfo")
        r.raise_for_status()
        data = r.json()
    rows = []
    for s in data.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        rows.append({
            "symbol": str(s.get("symbol", "")).upper(),
            "base_asset": s.get("baseAsset"),
            "quote_asset": s.get("quoteAsset"),
            "contract_type": s.get("contractType"),
        })
    return sorted(rows, key=lambda x: x["symbol"])


def _bitget_symbols() -> list[dict]:
    with httpx.Client(timeout=10.0, follow_redirects=True, trust_env=True) as client:
        r = client.get(
            f"{BITGET_URL}/api/v2/mix/market/contracts",
            params={"productType": "USDT-FUTURES"},
        )
        r.raise_for_status()
        payload = r.json()
    rows = []
    for s in payload.get("data", []):
        if s.get("symbolStatus") not in (None, "normal", "online"):
            continue
        symbol = str(s.get("symbol", "")).upper()
        if not symbol:
            continue
        rows.append({
            "symbol": symbol,
            "base_asset": s.get("baseCoin"),
            "quote_asset": s.get("quoteCoin", "USDT"),
            "contract_type": s.get("symbolType") or "PERPETUAL",
        })
    return sorted(rows, key=lambda x: x["symbol"])


@router.get("")
def market_symbols():
    result = {"binance": [], "bitget": [], "errors": {}}
    try:
        result["binance"] = _binance_symbols()
    except Exception as exc:
        result["errors"]["binance"] = f"{type(exc).__name__}: {exc}"
    try:
        result["bitget"] = _bitget_symbols()
    except Exception as exc:
        result["errors"]["bitget"] = f"{type(exc).__name__}: {exc}"

    if not result["binance"] and not result["bitget"]:
        raise HTTPException(status_code=503, detail="No exchange symbols are currently available")

    return result
