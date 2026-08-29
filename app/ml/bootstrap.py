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


# ============================================================================
# Historical market-data providers
# ============================================================================

BINANCE_KLINES_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

BINANCE_KLINES_PATH = "/fapi/v1/klines"
BINANCE_BATCH_SIZE = 500
BINANCE_RETRIES_PER_HOST = 2
BINANCE_REQUEST_DELAY = 0.25

BITGET_REST_BASE = "https://api.bitget.com"
BITGET_CANDLES_PATH = "/api/v2/mix/market/candles"
BITGET_BATCH_SIZE = 500
BITGET_RETRIES_PER_HOST = 2
BITGET_REQUEST_DELAY = 0.25
BITGET_PRODUCT_TYPE = "USDT-FUTURES"


# ============================================================================
# Common helpers
# ============================================================================


def _validate_candle_row(row: Any) -> bool:
    if not isinstance(row, (list, tuple)):
        return False

    if len(row) < 6:
        return False

    try:
        int(row[0])
        float(row[1])
        float(row[2])
        float(row[3])
        float(row[4])
        float(row[5])
    except (TypeError, ValueError, IndexError):
        return False

    return True


def _normalize_ohlcv_row(row: Any) -> list[Any] | None:
    if not _validate_candle_row(row):
        return None

    try:
        timestamp = int(row[0])
        open_price = float(row[1])
        high_price = float(row[2])
        low_price = float(row[3])
        close_price = float(row[4])
        volume = float(row[5])
    except (TypeError, ValueError, IndexError):
        return None

    if timestamp <= 0:
        return None

    if min(
        open_price,
        high_price,
        low_price,
        close_price,
    ) <= 0:
        return None

    if volume < 0:
        return None

    return [
        timestamp,
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
    ]


def _interval_to_milliseconds(interval: str) -> int:
    units = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }

    interval = interval.strip().lower()

    if len(interval) < 2:
        raise ValueError(f"Unsupported interval: {interval!r}")

    try:
        amount = int(interval[:-1])
    except ValueError as exc:
        raise ValueError(
            f"Unsupported interval: {interval!r}"
        ) from exc

    unit = interval[-1]

    if amount <= 0 or unit not in units:
        raise ValueError(f"Unsupported interval: {interval!r}")

    return amount * units[unit]


# ============================================================================
# Binance request
# ============================================================================


def _request_binance_batch(
    client: httpx.Client,
    symbol: str,
    interval: str,
    limit: int,
    end_time: int | None,
) -> list[list[Any]]:
    params: dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(BINANCE_BATCH_SIZE, max(1, int(limit))),
    }

    if end_time is not None:
        params["endTime"] = end_time

    last_error: Exception | None = None

    for host in BINANCE_KLINES_HOSTS:
        url = f"{host}{BINANCE_KLINES_PATH}"

        for attempt in range(BINANCE_RETRIES_PER_HOST):
            try:
                response = client.get(url, params=params)
                status = response.status_code
                body = response.text.strip()

                if status == 418:
                    last_error = RuntimeError(
                        f"Binance returned HTTP 418 from {url}."
                    )
                    break

                if status >= 400:
                    last_error = httpx.HTTPStatusError(
                        f"Binance HTTP error {status} from {url}",
                        request=response.request,
                        response=response,
                    )

                    if attempt + 1 < BINANCE_RETRIES_PER_HOST:
                        time.sleep(1.0 + attempt)
                        continue

                    break

                if not body:
                    last_error = RuntimeError(
                        f"Binance returned an empty response from {url}."
                    )
                    break

                try:
                    data = response.json()
                except ValueError as exc:
                    last_error = RuntimeError(
                        f"Binance returned invalid JSON from {url}: "
                        f"{body[:300]!r}"
                    )

                    if attempt + 1 < BINANCE_RETRIES_PER_HOST:
                        time.sleep(1.0 + attempt)
                        continue

                    raise last_error from exc

                if not isinstance(data, list) or not data:
                    last_error = RuntimeError(
                        f"Binance returned an invalid candle list from {url}."
                    )
                    break

                normalized = [
                    candle
                    for row in data
                    if (candle := _normalize_ohlcv_row(row)) is not None
                ]

                if not normalized:
                    last_error = RuntimeError(
                        f"Binance returned no usable candles from {url}."
                    )
                    break

                return normalized

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ) as exc:
                last_error = exc

                if attempt + 1 < BINANCE_RETRIES_PER_HOST:
                    time.sleep(1.0 + attempt)
                    continue

                break

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Unable to retrieve Binance market data "
        "from any configured endpoint."
    )


def fetch_binance_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 1500,
) -> list[list[Any]]:
    requested = max(
        BINANCE_BATCH_SIZE,
        min(1500, int(limit)),
    )

    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    _interval_to_milliseconds(interval)

    all_klines: list[list[Any]] = []
    end_time: int | None = None

    headers = {
        "User-Agent": "HHHAI/1.0",
        "Accept": "application/json",
    }

    with httpx.Client(
        timeout=30,
        follow_redirects=True,
        trust_env=False,
        headers=headers,
    ) as client:
        while len(all_klines) < requested:
            remaining = requested - len(all_klines)
            batch_limit = min(BINANCE_BATCH_SIZE, remaining)

            batch = _request_binance_batch(
                client=client,
                symbol=symbol,
                interval=interval,
                limit=batch_limit,
                end_time=end_time,
            )

            if not batch:
                break

            batch = sorted(batch, key=lambda row: int(row[0]))
            all_klines = batch + all_klines

            if len(batch) < batch_limit:
                break

            end_time = int(batch[0][0]) - 1
            time.sleep(BINANCE_REQUEST_DELAY)

    unique = {
        int(row[0]): row
        for row in all_klines
        if _normalize_ohlcv_row(row) is not None
    }

    result = sorted(unique.values(), key=lambda row: int(row[0]))
    result = result[-requested:]

    if len(result) < requested:
        raise RuntimeError(
            f"Binance returned only {len(result)} usable candles; "
            f"{requested} were requested."
        )

    return result


# ============================================================================
# Bitget request
# ============================================================================


def _request_bitget_batch(
    client: httpx.Client,
    symbol: str,
    interval: str,
    limit: int,
    end_time: int | None,
) -> list[list[Any]]:
    params: dict[str, Any] = {
        "productType": BITGET_PRODUCT_TYPE,
        "symbol": symbol.upper(),
        "granularity": interval,
        "limit": min(BITGET_BATCH_SIZE, max(1, int(limit))),
    }

    if end_time is not None:
        params["endTime"] = str(end_time)

    url = f"{BITGET_REST_BASE}{BITGET_CANDLES_PATH}"
    last_error: Exception | None = None

    for attempt in range(BITGET_RETRIES_PER_HOST):
        try:
            response = client.get(url, params=params)
            status = response.status_code
            body = response.text.strip()

            if status >= 400:
                last_error = httpx.HTTPStatusError(
                    f"Bitget HTTP error {status}",
                    request=response.request,
                    response=response,
                )

                if attempt + 1 < BITGET_RETRIES_PER_HOST:
                    time.sleep(1.0 + attempt)
                    continue

                break

            if not body:
                last_error = RuntimeError(
                    "Bitget returned an empty response."
                )

                if attempt + 1 < BITGET_RETRIES_PER_HOST:
                    time.sleep(1.0 + attempt)
                    continue

                break

            try:
                payload = response.json()
            except ValueError as exc:
                last_error = RuntimeError(
                    f"Bitget returned invalid JSON: {body[:300]!r}"
                )

                if attempt + 1 < BITGET_RETRIES_PER_HOST:
                    time.sleep(1.0 + attempt)
                    continue

                raise last_error from exc

            if not isinstance(payload, dict):
                last_error = RuntimeError(
                    "Bitget returned an unexpected response structure."
                )
                break

            code = str(payload.get("code", ""))

            if code not in {"", "00000"}:
                last_error = RuntimeError(
                    f"Bitget API error {code}: "
                    f"{payload.get('msg', 'unknown error')}"
                )
                break

            raw_data = payload.get("data")

            if not isinstance(raw_data, list) or not raw_data:
                last_error = RuntimeError(
                    "Bitget returned no candle data."
                )
                break

            normalized = [
                candle
                for row in raw_data
                if (candle := _normalize_ohlcv_row(row)) is not None
            ]

            if not normalized:
                last_error = RuntimeError(
                    "Bitget returned no usable candles."
                )
                break

            return normalized

        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
        ) as exc:
            last_error = exc

            if attempt + 1 < BITGET_RETRIES_PER_HOST:
                time.sleep(1.0 + attempt)
                continue

            break

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Unable to retrieve Bitget historical market data."
    )


def fetch_bitget_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 1500,
) -> list[list[Any]]:
    requested = max(
        BITGET_BATCH_SIZE,
        min(1500, int(limit)),
    )

    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    _interval_to_milliseconds(interval)

    all_klines: list[list[Any]] = []
    end_time: int | None = None

    headers = {
        "User-Agent": "HHHAI/1.0",
        "Accept": "application/json",
    }

    with httpx.Client(
        timeout=30,
        follow_redirects=True,
        trust_env=False,
        headers=headers,
    ) as client:
        while len(all_klines) < requested:
            remaining = requested - len(all_klines)
            batch_limit = min(BITGET_BATCH_SIZE, remaining)

            batch = _request_bitget_batch(
                client=client,
                symbol=symbol,
                interval=interval,
                limit=batch_limit,
                end_time=end_time,
            )

            if not batch:
                break

            batch = sorted(batch, key=lambda row: int(row[0]))
            all_klines = batch + all_klines

            if len(batch) < batch_limit:
                break

            end_time = int(batch[0][0]) - 1
            time.sleep(BITGET_REQUEST_DELAY)

    unique = {
        int(row[0]): row
        for row in all_klines
        if _normalize_ohlcv_row(row) is not None
    }

    result = sorted(unique.values(), key=lambda row: int(row[0]))
    result = result[-requested:]

    if len(result) < requested:
        raise RuntimeError(
            f"Bitget returned only {len(result)} usable candles; "
            f"{requested} were requested."
        )

    return result


# ============================================================================
# Provider fallback
# ============================================================================


def fetch_historical_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 1500,
) -> tuple[list[list[Any]], str]:
    errors: list[str] = []

    for provider, fetcher in (
        ("binance", fetch_binance_klines),
        ("bitget", fetch_bitget_klines),
    ):
        try:
            return (
                fetcher(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                ),
                provider,
            )
        except Exception as exc:
            errors.append(
                f"{provider}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "All historical market-data providers failed. "
        + " | ".join(errors)
    )


# ============================================================================
# Dataset construction
# ============================================================================


def build_dataset(
    klines: list[list[Any]],
    horizon: int = 6,
    threshold: float = 0.0025,
) -> list[dict[str, Any]]:
    if len(klines) < 50:
        raise ValueError(
            f"Not enough candles to build the dataset: {len(klines)}."
        )

    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    if threshold <= 0:
        raise ValueError("Threshold must be greater than zero.")

    candles: list[dict[str, Any]] = []

    for row in klines:
        normalized = _normalize_ohlcv_row(row)

        if normalized is None:
            continue

        timestamp, open_price, high_price, low_price, close_price, volume = (
            normalized
        )

        candles.append({
            "observed_at": datetime.fromtimestamp(
                timestamp / 1000,
                timezone.utc,
            ).isoformat(),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        })

    if len(candles) < 50:
        raise ValueError(
            "Not enough valid OHLCV candles after validation."
        )

    lookback = 24

    if len(candles) <= lookback + horizon:
        raise ValueError(
            "Not enough candles for the requested lookback and horizon."
        )

    rows: list[dict[str, Any]] = []

    for i in range(lookback, len(candles) - horizon):
        window = candles[i - lookback:i + 1]
        last = window[-1]
        prev = window[-2]

        returns = [
            window[j]["close"] / window[j - 1]["close"] - 1
            for j in range(1, len(window))
            if window[j - 1]["close"] > 0
        ]

        recent_returns = returns[-12:]

        mean_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
        vol = float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0

        vol_change = (
            last["volume"] / prev["volume"] - 1
            if prev["volume"] > 0
            else 0.0
        )

        future = (
            candles[i + horizon]["close"] / last["close"] - 1
            if last["close"] > 0
            else 0.0
        )

        label = (
            1
            if future > threshold
            else -1
            if future < -threshold
            else 0
        )

        features = {
            "return_1": returns[-1] if returns else 0.0,
            "range_pct": (
                (last["high"] - last["low"]) / last["close"]
                if last["close"] > 0
                else 0.0
            ),
            "volume_change": vol_change,
            "order_book_imbalance": 0.0,
            "funding_rate": 0.0,
            "open_interest_change": 0.0,
            "news_risk": 0.0,
            "news_sentiment": 0.0,
            "volatility_proxy": min(1.0, vol * 10),
            "trend_strength": min(1.0, abs(mean_ret) * 80),
            "momentum": max(-1.0, min(1.0, mean_ret * 40)),
            "liquidity_stress": 0.0,
        }

        rows.append({
            "observed_at": last["observed_at"],
            "features": features,
            "label": label,
            "outcome_return": future,
        })

    if not rows:
        raise ValueError(
            "Dataset construction produced zero training rows."
        )

    return rows


# ============================================================================
# Walk-forward validation and model promotion
# ============================================================================


def validate_and_promote(
    rows: list[dict[str, Any]],
    version: str = "bootstrap",
) -> dict[str, Any]:
    if not rows:
        return {
            "status": "REJECTED",
            "reason": "No training rows were supplied.",
            "rows": 0,
        }

    folds = walk_forward(
        rows,
        min_train=max(300, min(700, len(rows) // 2)),
        test_size=100,
        step=100,
    )

    if not folds:
        return {
            "status": "REJECTED",
            "reason": (
                "Not enough historical rows "
                "for walk-forward validation."
            ),
            "rows": len(rows),
        }

    fold_reports: list[dict[str, Any]] = []
    predictions: list[tuple[int, int, float]] = []

    for fold in folds:
        X = [
            [r["features"].get(k, 0.0) for k in FEATURES]
            for r in fold.train
        ]

        y = [r["label"] for r in fold.train]

        if len(set(y)) < 2:
            continue

        model = Pipeline([
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                ),
            ),
        ])

        model.fit(X, y)

        Xtest = [
            [r["features"].get(k, 0.0) for k in FEATURES]
            for r in fold.test
        ]

        pred = model.predict(Xtest)
        probs = model.predict_proba(Xtest)

        actual_labels = [r["label"] for r in fold.test]

        for r, p, prob in zip(fold.test, pred, probs):
            confidence = float(np.max(prob))
            realized = float(r["outcome_return"])

            if confidence >= 0.55 and int(p) != 0:
                trade_return = (
                    realized
                    if int(p) == int(r["label"])
                    else -abs(realized)
                )
            else:
                trade_return = 0.0

            predictions.append(
                (
                    int(r["label"]),
                    int(p),
                    trade_return,
                )
            )

        fold_reports.append({
            "train": len(fold.train),
            "test": len(fold.test),
            "accuracy": float(
                np.mean(
                    np.asarray(pred) == np.asarray(actual_labels)
                )
            ),
        })

    if not predictions:
        return {
            "status": "REJECTED",
            "version": predictive_model.version,
            "reason": (
                "No valid walk-forward predictions "
                "were produced."
            ),
            "rows": len(rows),
            "folds": len(fold_reports),
            "folds_detail": fold_reports,
        }

    metrics = evaluate_predictions(predictions)
    metrics["folds"] = len(fold_reports)
    metrics["folds_detail"] = fold_reports

    if (
        metrics["accuracy"] < 0.52
        or metrics["balanced_accuracy"] < 0.50
        or metrics["avg_return"] <= 0
    ):
        return {
            "status": "REJECTED",
            "version": predictive_model.version,
            "metrics": metrics,
        }

    predictive_model.train(
        rows,
        version=version,
        min_rows=max(1, len(rows)),
    )

    return {
        "status": "PROMOTED",
        "version": version,
        "metrics": metrics,
        "rows": len(rows),
    }
