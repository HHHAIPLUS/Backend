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


# Binance Futures API endpoints.
# If one endpoint returns 418, HHHAI can try another endpoint.
BINANCE_KLINES_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]

BINANCE_KLINES_PATH = "/fapi/v1/klines"

# Keep individual requests comfortably below Binance's maximum.
BINANCE_BATCH_SIZE = 500

# Maximum number of attempts against a single endpoint.
BINANCE_RETRIES_PER_HOST = 2


def _request_binance_batch(
    client: httpx.Client,
    symbol: str,
    interval: str,
    limit: int,
    end_time: int | None,
) -> list[list[Any]]:
    """
    Request one Binance Futures candle batch.

    A 418 response means the current Binance route/IP has been rejected.
    In that situation the caller can move to another Binance endpoint.
    """

    params: dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(BINANCE_BATCH_SIZE, limit),
    }

    if end_time is not None:
        params["endTime"] = end_time

    last_error: Exception | None = None

    for host in BINANCE_KLINES_HOSTS:
        url = f"{host}{BINANCE_KLINES_PATH}"

        for attempt in range(BINANCE_RETRIES_PER_HOST):
            try:
                response = client.get(url, params=params)

                # Binance 418 = temporary IP/rate-limit rejection.
                # Do not keep hammering the same endpoint.
                if response.status_code == 418:
                    last_error = httpx.HTTPStatusError(
                        "Binance returned HTTP 418 (rate-limit/IP restriction)",
                        request=response.request,
                        response=response,
                    )

                    # Brief pause before trying another route.
                    time.sleep(1.0 + attempt)

                    # Move immediately to the next Binance endpoint.
                    break

                response.raise_for_status()

                data = response.json()

                if not isinstance(data, list):
                    raise RuntimeError(
                        "Binance returned an unexpected candle response."
                    )

                return data

            except httpx.HTTPStatusError as exc:
                last_error = exc

                # For ordinary HTTP errors, retry this endpoint once.
                if response.status_code != 418 and attempt + 1 < BINANCE_RETRIES_PER_HOST:
                    time.sleep(1.0 + attempt)
                    continue

                break

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc

                if attempt + 1 < BINANCE_RETRIES_PER_HOST:
                    time.sleep(1.0 + attempt)
                    continue

                break

    if last_error is not None:
        raise last_error

    raise RuntimeError("Unable to retrieve Binance market data.")


def fetch_binance_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 1500,
) -> list[list[Any]]:
    """
    Fetch historical Binance Futures candles in multiple small batches.

    The result is:
        - ordered oldest -> newest
        - duplicate-free
        - limited to the requested number of candles
    """

    requested = max(500, min(1500, int(limit)))
    symbol = symbol.upper()

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

            # We are walking backwards through history.
            # Prepend the older batch to the existing candles.
            all_klines = batch + all_klines

            # If Binance returned fewer candles than requested,
            # there are no more historical candles needed.
            if len(batch) < batch_limit:
                break

            # The first candle is now the oldest candle in this batch.
            earliest_open_time = int(batch[0][0])

            # Next request must end immediately before this candle.
            end_time = earliest_open_time - 1

            # Avoid unnecessarily aggressive requests.
            time.sleep(0.25)

    # Remove duplicates by candle open timestamp.
    unique: dict[int, list[Any]] = {}

    for row in all_klines:
        if row:
            unique[int(row[0])] = row

    result = sorted(
        unique.values(),
        key=lambda row: int(row[0]),
    )

    return result[-requested:]


def build_dataset(
    klines: list[list[Any]],
    horizon: int = 6,
    threshold: float = 0.0025,
) -> list[dict[str, Any]]:

    candles = []

    for row in klines:
        candles.append({
            "observed_at": datetime.fromtimestamp(
                int(row[0]) / 1000,
                timezone.utc,
            ).isoformat(),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })

    rows: list[dict[str, Any]] = []

    lookback = 24

    for i in range(lookback, len(candles) - horizon):
        window = candles[i - lookback:i + 1]

        last = window[-1]
        prev = window[-2]

        returns = [
            window[j]["close"] / window[j - 1]["close"] - 1
            for j in range(1, len(window))
            if window[j - 1]["close"]
        ]

        mean_ret = (
            float(np.mean(returns[-12:]))
            if returns
            else 0.0
        )

        vol = (
            float(np.std(returns[-12:]))
            if len(returns) > 1
            else 0.0
        )

        vol_change = (
            last["volume"] / prev["volume"] - 1
            if prev["volume"]
            else 0.0
        )

        future = (
            candles[i + horizon]["close"] / last["close"] - 1
            if last["close"]
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
                if last["close"]
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

    return rows


def validate_and_promote(
    rows: list[dict[str, Any]],
    version: str = "bootstrap",
) -> dict[str, Any]:

    folds = walk_forward(
        rows,
        min_train=max(300, min(700, len(rows) // 2)),
        test_size=100,
        step=100,
    )

    if not folds:
        return {
            "status": "REJECTED",
            "reason": "Not enough historical rows for walk-forward validation.",
            "rows": len(rows),
        }

    fold_reports = []
    predictions: list[tuple[int, int, float]] = []

    for fold in folds:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ])

        X = [
            [r["features"].get(k, 0.0) for k in FEATURES]
            for r in fold.train
        ]

        y = [r["label"] for r in fold.train]

        if len(set(y)) < 3:
            continue

        model.fit(X, y)

        Xtest = [
            [r["features"].get(k, 0.0) for k in FEATURES]
            for r in fold.test
        ]

        pred = model.predict(Xtest)
        probs = model.predict_proba(Xtest)

        for r, p, prob in zip(fold.test, pred, probs):
            confidence = float(max(prob))
            realized = float(r["outcome_return"])

            trade_return = (
                realized
                if confidence >= 0.55 and int(p) == r["label"]
                else (
                    -abs(realized)
                    if confidence >= 0.55
                    else 0.0
                )
            )

            predictions.append(
                (r["label"], int(p), trade_return)
            )

        fold_reports.append({
            "train": len(fold.train),
            "test": len(fold.test),
            "accuracy": float(
                np.mean(
                    np.asarray(pred)
                    == np.asarray(
                        [r["label"] for r in fold.test]
                    )
                )
            ),
        })

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
