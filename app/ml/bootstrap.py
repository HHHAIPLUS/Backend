from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.predictive import FEATURES, predictive_model
from app.ml.validation import walk_forward, evaluate_predictions
from app.ml.features import build_model_features
from app.ml.dataset_integrity import audit_training_rows, DatasetIntegrityError

# Historical market-data configuration and exchange helpers remain unchanged.
# The important Stage 1A contract is that bootstrap must reject incomplete
# context rather than encode missing information as if it were neutral.

BINANCE_KLINES_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]
BINANCE_KLINES_PATH = "/fapi/v1/klines"
BITGET_KLINES_URL = "https://api.bitget.com/api/v2/mix/market/candles"
BINANCE_BATCH_SIZE = 500
BITGET_BATCH_SIZE = 200
BINANCE_RETRIES_PER_HOST = 2
BITGET_RETRIES_PER_REQUEST = 2
HISTORICAL_REQUEST_DELAY = 0.25
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


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
    requested = max(BINANCE_BATCH_SIZE, min(1500, int(limit)))
    symbol = symbol.upper().strip()
    if not symbol or not interval:
        raise ValueError("Symbol and interval are required")
    all_klines: list[list[Any]] = []
    end_time: int | None = None
    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True, trust_env=False, headers={"User-Agent": "HHHAI/1.0", "Accept": "application/json"}) as client:
        while len(all_klines) < requested:
            remaining = requested - len(all_klines)
            batch_limit = min(BINANCE_BATCH_SIZE, remaining)
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
    params: dict[str, Any] = {"productType": "USDT-FUTURES", "symbol": symbol.upper(), "granularity": granularity, "limit": min(BITGET_BATCH_SIZE, max(1, int(limit)))}
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
    requested = max(BITGET_BATCH_SIZE, min(1500, int(limit)))
    symbol = symbol.upper().strip()
    if not symbol or not interval:
        raise ValueError("Symbol and interval are required")
    all_klines: list[list[Any]] = []
    end_time: int | None = None
    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True, trust_env=True, headers={"User-Agent": "HHHAI/1.0", "Accept": "application/json"}) as client:
        while len(all_klines) < requested:
            batch_limit = min(BITGET_BATCH_SIZE, requested - len(all_klines))
            batch = _request_bitget_batch(client, symbol, interval.lower(), batch_limit, end_time)
            all_klines = batch + all_klines
            if len(batch) < batch_limit:
                break
            end_time = int(sorted(batch, key=lambda row: int(row[0]))[0][0]) - 1
            time.sleep(HISTORICAL_REQUEST_DELAY)
    result = _deduplicate_klines(all_klines)[-requested:]
    if len(result) < requested:
        raise RuntimeError(f"Bitget returned only {len(result)} usable candles out of {requested} requested")
    return result


def fetch_historical_klines(symbol: str, interval: str = "5m", limit: int = 1500) -> tuple[list[list[Any]], str]:
    errors: list[str] = []
    for provider, fetcher in (("binance", fetch_binance_klines), ("bitget", fetch_bitget_klines)):
        try:
            return fetcher(symbol=symbol, interval=interval, limit=limit), provider
        except Exception as exc:
            errors.append(f"{provider}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Historical market data unavailable from all configured providers. " + " | ".join(errors))


def _candle_to_dict(row: list[Any]) -> dict[str, Any]:
    return {
        "observed_at": datetime.fromtimestamp(int(row[0]) / 1000, timezone.utc).isoformat(),
        "open": float(row[1]), "high": float(row[2]), "low": float(row[3]), "close": float(row[4]), "volume": max(0.0, float(row[5])),
    }


def build_dataset(klines: list[list[Any]], horizon: int = 6, threshold: float = 0.0025) -> list[dict[str, Any]]:
    if horizon <= 0 or threshold <= 0:
        raise ValueError("Horizon and threshold must be greater than zero")
    candles = [_candle_to_dict(row) for row in _deduplicate_klines(klines)]
    if len(candles) < 50:
        raise ValueError(f"Not enough valid OHLCV candles: {len(candles)}")
    lookback = 24
    if len(candles) <= lookback + horizon:
        raise ValueError("Not enough candles for the requested lookback and horizon")
    rows: list[dict[str, Any]] = []
    for i in range(lookback, len(candles) - horizon):
        window = candles[i - lookback:i + 1]
        last = window[-1]
        future = candles[i + horizon]["close"] / last["close"] - 1.0
        label = 1 if future > threshold else -1 if future < -threshold else 0
        candle_rows = [[int(datetime.fromisoformat(c["observed_at"]).timestamp() * 1000), c["open"], c["high"], c["low"], c["close"], c["volume"]] for c in window]
        model_features = build_model_features(candle_rows)
        # Do not silently claim missing context is neutral. These values are
        # explicitly marked unavailable so the training gate can reject them.
        context_availability = {name: False for name in ("order_book_imbalance", "funding_rate", "open_interest_change", "news_risk", "news_sentiment", "liquidity_stress")}
        rows.append({"observed_at": last["observed_at"], "features": model_features, "label": label, "outcome_return": future, "context_available": context_availability, "data_source": "ohlcv_only"})
    if not rows:
        raise ValueError("Dataset construction produced zero training rows")
    return rows


def validate_and_promote(rows: list[dict[str, Any]], version: str = "bootstrap") -> dict[str, Any]:
    if not rows:
        return {"status": "REJECTED", "reason": "No training rows were supplied.", "rows": 0}
    try:
        audit = audit_training_rows(rows, required_context_features=("order_book_imbalance", "funding_rate", "open_interest_change", "news_risk", "news_sentiment", "liquidity_stress"))
    except DatasetIntegrityError as exc:
        return {"status": "REJECTED", "reason": f"Dataset integrity gate failed: {exc}", "rows": len(rows)}
    min_train = max(300, min(700, len(rows) // 2))
    folds = walk_forward(rows, min_train=min_train, test_size=100, step=100)
    if not folds:
        return {"status": "REJECTED", "reason": "Not enough historical rows for walk-forward validation.", "rows": len(rows), "audit": audit}
    predictions: list[tuple[int, int, float]] = []
    for fold in folds:
        X = [[r["features"].get(k, 0.0) for k in FEATURES] for r in fold.train]
        y = [int(r["label"]) for r in fold.train]
        if len(set(y)) < 3:
            continue
        model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
        model.fit(X, y)
        Xtest = [[r["features"].get(k, 0.0) for k in FEATURES] for r in fold.test]
        pred = model.predict(Xtest)
        probs = model.predict_proba(Xtest)
        classes = list(model.classes_)
        for r, p, prob in zip(fold.test, pred, probs):
            confidence = float(max(prob)) if len(prob) else 0.0
            predictions.append((int(r["label"]), int(p), float(r["outcome_return"]) if p != 0 else 0.0))
    metrics = evaluate_predictions(predictions)
    status = "PROMOTED" if metrics.get("accuracy", 0.0) >= 0.52 and metrics.get("balanced_accuracy", 0.0) >= 0.50 and metrics.get("average_return", 0.0) > 0 else "REJECTED"
    return {"status": status, "version": version, "metrics": metrics, "rows": len(rows), "audit": audit}
