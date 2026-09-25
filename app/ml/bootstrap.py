from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
import numpy as np
import io
import zipfile
from app.ml.predictive_brain import HORIZONS
from app.ml.features import build_model_features
from app.ml.dataset_integrity import audit_klines, require_production_ready, DatasetAudit

BINANCE_KLINES_HOSTS = ["https://fapi.binance.com", "https://fapi1.binance.com", "https://fapi2.binance.com", "https://fapi3.binance.com", "https://fapi4.binance.com"]
BINANCE_KLINES_PATH = "/fapi/v1/klines"
BITGET_KLINES_URL = "https://api.bitget.com/api/v3/market/history-candles"
BINANCE_BATCH_SIZE = 500
BITGET_BATCH_SIZE = 100
BINANCE_RETRIES_PER_HOST = 2
BITGET_RETRIES_PER_REQUEST = 2
BITGET_GRANULARITY = {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m","1h":"1H","4h":"4H","6h":"6H","12h":"12H","1d":"1D"}
HISTORICAL_REQUEST_DELAY = 0.25
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
CONTEXT_FEATURES = ("order_book_imbalance", "funding_rate", "open_interest_change", "news_risk", "news_sentiment", "liquidity_stress")

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
    "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
}

def _interval_ms(interval: str) -> int:
    key = str(interval).strip().lower()
    if key not in INTERVAL_MS:
        raise ValueError(f"Unsupported historical interval: {interval}")
    return INTERVAL_MS[key]

def audit_historical_klines(klines: list[list[Any]], interval: str) -> dict[str, Any]:
    audit = audit_klines(klines, interval_ms=_interval_ms(interval))
    if not audit.production_ready:
        raise RuntimeError("Historical candle integrity gate failed: " + audit.reason)
    return audit.__dict__


def _is_json_response(response: httpx.Response) -> bool:
    body = response.text.strip()
    content_type = response.headers.get("content-type", "").lower()
    return "application/json" in content_type or body.startswith("[") or body.startswith("{")


def _validate_candle_row(row: Any) -> bool:
    if not isinstance(row, list) or len(row) < 6:
        return False
    try:
        timestamp, op, hi, lo, cl, vol = int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
    except (TypeError, ValueError):
        return False
    return timestamp > 0 and min(op, hi, lo, cl) > 0 and hi >= lo and np.isfinite([timestamp, op, hi, lo, cl, vol]).all()


def _deduplicate_klines(klines: list[list[Any]]) -> list[list[Any]]:
    unique: dict[int, list[Any]] = {}
    for row in klines:
        if _validate_candle_row(row):
            unique[int(row[0])] = row
    return sorted(unique.values(), key=lambda row: int(row[0]))


def _request_binance_batch(client: httpx.Client, symbol: str, interval: str, limit: int, end_time: int | None) -> list[list[Any]]:
    params: dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": min(BINANCE_BATCH_SIZE, max(1, int(limit)))}
    if end_time is not None:
        params["endTime"] = end_time
    last_error: Exception | None = None
    for host in BINANCE_KLINES_HOSTS:
        for attempt in range(BINANCE_RETRIES_PER_HOST):
            try:
                response = client.get(f"{host}{BINANCE_KLINES_PATH}", params=params)
                body = response.text.strip()
                if response.status_code == 418:
                    last_error = RuntimeError(f"Binance HTTP 418 from {host}")
                    break
                if response.status_code >= 400:
                    last_error = RuntimeError(f"Binance HTTP {response.status_code} from {host}")
                    if attempt + 1 < BINANCE_RETRIES_PER_HOST:
                        time.sleep(1.0 + attempt)
                        continue
                    break
                if not body or not _is_json_response(response):
                    last_error = RuntimeError(f"Binance returned unusable response from {host}")
                    break
                data = response.json()
                valid = [row for row in data if _validate_candle_row(row)] if isinstance(data, list) else []
                if valid:
                    return valid
                last_error = RuntimeError(f"Binance returned no valid candles from {host}")
                break
            except (httpx.TimeoutException, httpx.NetworkError, ValueError) as exc:
                last_error = exc
                if attempt + 1 < BINANCE_RETRIES_PER_HOST:
                    time.sleep(1.0 + attempt)
    raise last_error or RuntimeError("Unable to retrieve Binance market data")


def fetch_binance_klines(symbol: str, interval: str = "5m", limit: int = 1500) -> list[list[Any]]:
    requested = min(10000, max(500, int(limit)))
    symbol = symbol.upper().strip()
    if not symbol or not interval:
        raise ValueError("Symbol and interval are required")
    all_klines: list[list[Any]] = []
    end_time: int | None = None
    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True, trust_env=False, headers={"User-Agent": "HHHAI/1.0", "Accept": "application/json"}) as client:
        while len(all_klines) < requested:
            batch_limit = min(BINANCE_BATCH_SIZE, requested - len(all_klines))
            batch = _request_binance_batch(client, symbol, interval, batch_limit, end_time)
            all_klines = batch + all_klines
            if len(batch) < batch_limit:
                break
            end_time = int(batch[0][0]) - 1
            time.sleep(HISTORICAL_REQUEST_DELAY)
    result = _deduplicate_klines(all_klines)[-requested:]
    if len(result) < requested:
        raise RuntimeError(f"Binance returned only {len(result)} usable candles out of {requested} requested")
    return result


def _normalize_bitget_candle(row: Any) -> list[Any] | None:
    if not isinstance(row, list) or len(row) < 6:
        return None
    try:
        values = [int(float(row[0])), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
    except (TypeError, ValueError):
        return None
    if values[0] <= 0 or min(values[1:5]) <= 0 or values[2] < values[3] or not np.isfinite(values).all():
        return None
    values[5] = max(0.0, values[5])
    return values


def _request_bitget_batch(client: httpx.Client, symbol: str, granularity: str, limit: int, end_time: int | None) -> list[list[Any]]:
    normalized_granularity = BITGET_GRANULARITY.get(str(granularity).lower(), str(granularity))
    params: dict[str, Any] = {"category": "USDT-FUTURES", "symbol": symbol.upper(), "interval": normalized_granularity, "limit": min(BITGET_BATCH_SIZE, max(1, int(limit)))}
    if end_time is not None:
        params["endTime"] = str(end_time)
    last_error: Exception | None = None
    for attempt in range(BITGET_RETRIES_PER_REQUEST):
        try:
            response = client.get(BITGET_KLINES_URL, params=params, headers={"User-Agent": "HHHAI/1.0", "Accept": "application/json"})
            if response.status_code >= 400:
                raise RuntimeError(f"Bitget HTTP {response.status_code}")
            data = response.json()
            if not isinstance(data, dict) or data.get("code") not in ("00000", 0, None):
                raise RuntimeError(f"Bitget returned unexpected response: {str(data)[:300]}")
            rows = data.get("data", [])
            valid = [normalized for row in rows if (normalized := _normalize_bitget_candle(row)) is not None]
            if valid:
                return valid
            raise RuntimeError("Bitget returned no valid candles")
        except (httpx.TimeoutException, httpx.NetworkError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < BITGET_RETRIES_PER_REQUEST:
                time.sleep(1.0 + attempt)
    raise last_error or RuntimeError("Unable to retrieve Bitget market data")


def fetch_bitget_klines(symbol: str, interval: str = "5m", limit: int = 1500) -> list[list[Any]]:
    requested = min(30000, max(500, int(limit)))
    symbol = symbol.upper().strip()
    if not symbol or not interval:
        raise ValueError("Symbol and interval are required")
    all_klines: list[list[Any]] = []
    end_time: int | None = int(time.time() * 1000)
    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True, trust_env=True, headers={"User-Agent": "HHHAI/1.0", "Accept": "application/json"}) as client:
        while len(_deduplicate_klines(all_klines)) < requested:
            unique_before = len(_deduplicate_klines(all_klines))
            batch_limit = BITGET_BATCH_SIZE
            batch = _request_bitget_batch(client, symbol, interval, batch_limit, end_time)
            all_klines = batch + all_klines
            unique_after = len(_deduplicate_klines(all_klines))
            if unique_after <= unique_before:
                raise RuntimeError("Bitget historical pagination made no progress; refusing to reuse the same candle page.")
            if len(batch) < batch_limit:
                break
            # Bitget rounds endTime to the candle boundary. Passing the exact
            # oldest timestamp and deduplicating the overlap is safer than subtracting
            # 1ms, which can skip a boundary candle on some intervals.
            end_time = int(sorted(batch, key=lambda row: int(row[0]))[0][0])
            time.sleep(HISTORICAL_REQUEST_DELAY)
    result = _deduplicate_klines(all_klines)
    interval_ms = _interval_ms(interval)
    now_ms = int(time.time() * 1000)
    result = [row for row in result if int(row[0]) + interval_ms <= now_ms][-requested:]
    if len(result) < requested:
        raise RuntimeError(f"Bitget returned only {len(result)} closed usable candles out of {requested} requested")
    return result



def fetch_binance_archive_klines(symbol: str, interval: str = "5m", limit: int = 8000) -> list[list[Any]]:
    """Fallback to Binance public USD-M futures historical archives."""
    if interval not in {"1m","3m","5m","15m","30m","1h","4h","1d"}:
        raise ValueError(f"Unsupported Binance archive interval: {interval}")
    target = max(500, min(10000, int(limit)))
    now = datetime.now(timezone.utc)
    month_cursor = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
    collected: list[list[Any]] = []
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=True, trust_env=False,
                      headers={"User-Agent": "HHHAI/1.0", "Accept": "application/zip"}) as client:
        for _ in range(3):
            month = month_cursor.strftime("%Y-%m")
            url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol.upper()}/{interval}/{symbol.upper()}-{interval}-{month}.zip"
            response = client.get(url)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    for name in archive.namelist():
                        if name.lower().endswith(".csv"):
                            with archive.open(name) as fh:
                                for line in io.TextIOWrapper(fh, encoding="utf-8", newline=""):
                                    parts = line.strip().split(",")
                                    if len(parts) >= 6:
                                        try:
                                            row = [int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
                                            if _validate_candle_row(row):
                                                collected.append(row)
                                        except (TypeError, ValueError):
                                            pass
            month_cursor = (month_cursor - timedelta(days=1)).replace(day=1)
            if len(collected) >= target:
                break
    result = _deduplicate_klines(collected)[-target:]
    if len(result) < target:
        raise RuntimeError(f"Binance historical archive returned only {len(result)} usable candles out of {target} requested")
    return result

def fetch_historical_klines(symbol: str, interval: str = "5m", limit: int = 1500) -> tuple[list[list[Any]], str]:
    errors: list[str] = []
    for provider, fetcher in (("bitget", fetch_bitget_klines), ("binance", fetch_binance_klines), ("binance_archive", fetch_binance_archive_klines)):
        try:
            return fetcher(symbol=symbol, interval=interval, limit=limit), provider
        except Exception as exc:
            errors.append(f"{provider}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Historical market data unavailable from all configured providers. " + " | ".join(errors))


def _candle_to_dict(row: list[Any]) -> dict[str, Any]:
    return {"observed_at": datetime.fromtimestamp(int(row[0]) / 1000, timezone.utc).isoformat(), "open": float(row[1]), "high": float(row[2]), "low": float(row[3]), "close": float(row[4]), "volume": max(0.0, float(row[5]))}


def build_dataset(klines: list[list[Any]], horizon: int = 6, threshold: float = 0.0025, take_profit: float = 0.004, stop_loss: float = 0.004, *, symbol: str = "", interval: str = "", provider: str = "") -> list[dict[str, Any]]:
    if horizon <= 0 or threshold <= 0 or take_profit <= 0 or stop_loss <= 0:
        raise ValueError("Horizon, threshold, take_profit and stop_loss must be greater than zero")
    raw = list(klines)
    # Duplicate/gap/open-candle defects must be rejected before any normalization
    # can hide them. Deduplication is only for internal callers that have already
    # passed the authoritative raw-candle audit.
    if interval:
        audit_historical_klines(raw, interval)
    raw = _deduplicate_klines(raw)
    candles = [_candle_to_dict(row) for row in raw]
    if len(candles) < 50:
        raise ValueError(f"Not enough valid OHLCV candles: {len(candles)}")
    lookback = 336
    if len(candles) <= lookback + horizon:
        raise ValueError("Not enough candles for the requested lookback and horizon")
    rows: list[dict[str, Any]] = []
    max_horizon = max(HORIZONS)
    for i in range(lookback, len(candles) - max_horizon):
        window = candles[i - lookback:i + 1]
        last = window[-1]
        future_return = candles[i + horizon]["close"] / last["close"] - 1.0
        horizon_returns = {str(h): candles[i + h]["close"] / last["close"] - 1.0 for h in HORIZONS if i + h < len(candles)}
        barrier_returns = {}
        for h in HORIZONS:
            if i + h >= len(candles):
                continue
            long_barrier = last["close"] * (1.0 + take_profit)
            short_barrier = last["close"] * (1.0 - stop_loss)
            barrier = None
            for j in range(i + 1, i + h + 1):
                hi, lo = candles[j]["high"], candles[j]["low"]
                hit_long, hit_short = hi >= long_barrier, lo <= short_barrier
                if hit_long and hit_short:
                    barrier = 0.0
                    break
                if hit_long:
                    barrier = take_profit
                    break
                if hit_short:
                    barrier = -stop_loss
                    break
            if barrier is None:
                barrier = candles[i + h]["close"] / last["close"] - 1.0
            barrier_returns[str(h)] = float(barrier)
        # Phase 2 supervised targets use the future close return at the selected horizon.
        # Barrier outcomes remain stored separately for later execution research.
        future_trade_return = future_return
        label = 1 if future_trade_return > threshold else -1 if future_trade_return < -threshold else 0
        candle_rows = [[int(datetime.fromisoformat(c["observed_at"]).timestamp() * 1000), c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in window]
        model_features = build_model_features(candle_rows)
        rows.append({"observed_at": last["observed_at"], "features": model_features, "label": label, "outcome_return": future_trade_return, "outcome_horizon": horizon, "outcome_return_by_horizon": horizon_returns, "barrier_return_by_horizon": barrier_returns, "close_return_by_horizon": horizon_returns, "context_available": {name: False for name in CONTEXT_FEATURES}, "feature_provenance": {}, "data_source": provider or "ohlcv_only", "symbol": symbol.upper(), "interval": interval, "candle": {"timestamp": int(datetime.fromisoformat(last["observed_at"]).timestamp() * 1000), "open": last["open"], "high": last["high"], "low": last["low"], "close": last["close"], "volume": last["volume"]}})
    return rows


