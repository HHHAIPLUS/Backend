from __future__ import annotations

import time
from typing import Any

import httpx
import numpy as np

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BATCH_SIZE = 500
RETRIES = 3
TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def _valid(row: Any) -> bool:
    if not isinstance(row, list) or len(row) < 6:
        return False
    try:
        ts, op, hi, lo, cl, vol = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
    except (TypeError, ValueError):
        return False
    return ts > 0 and min(op, hi, lo, cl) > 0 and hi >= lo and np.isfinite([ts, op, hi, lo, cl, vol]).all()


def fetch_binance_klines(symbol: str, interval: str = "5m", limit: int = 5000) -> list[list[Any]]:
    requested = min(10000, max(500, int(limit)))
    symbol = symbol.upper().strip()
    if not symbol or not interval:
        raise ValueError("Symbol and interval are required")

    rows: list[list[Any]] = []
    end_time: int | None = None
    headers = {"User-Agent": "HHHAI/1.0", "Accept": "application/json"}
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True, trust_env=False, headers=headers) as client:
        while len(rows) < requested:
            batch_limit = min(BATCH_SIZE, requested - len(rows))
            params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": batch_limit}
            if end_time is not None:
                params["endTime"] = end_time
            last_error: Exception | None = None
            for attempt in range(RETRIES):
                try:
                    response = client.get(BINANCE_KLINES_URL, params=params)
                    body = response.text.strip()
                    if response.status_code >= 400:
                        raise RuntimeError(f"Binance HTTP {response.status_code}: {body[:200]}")
                    data = response.json()
                    if not isinstance(data, list):
                        raise RuntimeError(f"Binance returned non-list response: {body[:200]}")
                    valid = [row for row in data if _valid(row)]
                    if not valid:
                        raise RuntimeError(f"Binance returned no valid candles: {body[:200]}")
                    rows = valid + rows
                    end_time = int(valid[0][0]) - 1
                    last_error = None
                    break
                except (httpx.TimeoutException, httpx.NetworkError, ValueError, RuntimeError) as exc:
                    last_error = exc
                    if attempt + 1 < RETRIES:
                        time.sleep(1.0 + attempt)
            if last_error is not None:
                raise RuntimeError(f"Binance historical kline fetch failed: {last_error}") from last_error
            time.sleep(0.25)

    unique = {int(row[0]): row for row in rows if _valid(row)}
    result = sorted(unique.values(), key=lambda row: int(row[0]))[-requested:]
    if len(result) < requested:
        raise RuntimeError(f"Binance returned only {len(result)} usable candles out of {requested} requested")
    return result
