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


# Binance Futures REST endpoints.
#
# HHHAI tries these independently because a deployment IP may be
# temporarily rejected by one Binance route while another route works.
BINANCE_KLINES_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

BINANCE_KLINES_PATH = "/fapi/v1/klines"

# Keep individual requests well below Binance's maximum.
BINANCE_BATCH_SIZE = 500

# Maximum retries for temporary network/HTTP failures on one endpoint.
BINANCE_RETRIES_PER_HOST = 2

# Small delay between historical-data requests.
BINANCE_REQUEST_DELAY = 0.25


def _request_binance_batch(
    client: httpx.Client,
    symbol: str,
    interval: str,
    limit: int,
    end_time: int | None,
) -> list[list[Any]]:
    """
    Request one Binance Futures candle batch.

    A Binance route is considered unusable when it returns:
        - HTTP 418
        - HTTP 202 with an empty body
        - any other empty response
        - a non-JSON response
        - invalid JSON
        - JSON that is not a list

    In those situations HHHAI moves to the next Binance endpoint.
    """

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
            response: httpx.Response | None = None

            try:
                response = client.get(
                    url,
                    params=params,
                )

                status = response.status_code
                body = response.text.strip()
                content_type = response.headers.get(
                    "content-type",
                    "",
                ).lower()

                # ---------------------------------------------------------
                # HTTP 418
                # ---------------------------------------------------------
                # Binance may temporarily reject the deployment IP.
                # Do not continue hammering the same endpoint.
                if status == 418:
                    last_error = RuntimeError(
                        f"Binance returned HTTP 418 from {url} "
                        f"(rate-limit/IP restriction)."
                    )

                    time.sleep(1.0 + attempt)

                    # Immediately move to the next endpoint.
                    break

                # ---------------------------------------------------------
                # HTTP 202
                # ---------------------------------------------------------
                # A 202 with no usable body is not valid candle data.
                # Treat the endpoint as unusable and move on.
                if status == 202:
                    if not body:
                        last_error = RuntimeError(
                            f"Binance returned an empty response from "
                            f"{url} (HTTP 202)."
                        )
                    else:
                        last_error = RuntimeError(
                            f"Binance returned HTTP 202 from {url} "
                            f"with an unexpected response: "
                            f"{body[:500]!r}"
                        )

                    # Do not repeatedly retry the same route.
                    break

                # ---------------------------------------------------------
                # Other HTTP errors
                # ---------------------------------------------------------
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

                # ---------------------------------------------------------
                # Empty response
                # ---------------------------------------------------------
                if not body:
                    last_error = RuntimeError(
                        f"Binance returned an empty response from "
                        f"{url} (HTTP {status})."
                    )

                    # An empty successful response is not usable.
                    # Move to the next endpoint.
                    break

                # ---------------------------------------------------------
                # Content type
                # ---------------------------------------------------------
                # Binance normally returns application/json.
                #
                # Some upstream proxies may omit the exact content type,
                # so application/json is preferred but we still allow
                # a JSON-looking body.
                looks_like_json = (
                    body.startswith("[")
                    or body.startswith("{")
                )

                if (
                    "application/json" not in content_type
                    and not looks_like_json
                ):
                    last_error = RuntimeError(
                        f"Binance returned a non-JSON response from "
                        f"{url} (HTTP {status}, "
                        f"content-type={content_type!r}, "
                        f"body={body[:500]!r})."
                    )

                    # Move to another endpoint.
                    break

                # ---------------------------------------------------------
                # JSON decoding
                # ---------------------------------------------------------
                try:
                    data = response.json()
                except ValueError as exc:
                    last_error = RuntimeError(
                        f"Binance returned invalid JSON from {url}: "
                        f"{body[:500]!r}"
                    )

                    if attempt + 1 < BINANCE_RETRIES_PER_HOST:
                        time.sleep(1.0 + attempt)
                        continue

                    break

                # ---------------------------------------------------------
                # Validate candle payload
                # ---------------------------------------------------------
                if not isinstance(data, list):
                    last_error = RuntimeError(
                        f"Binance returned an unexpected candle response "
                        f"from {url}: {str(data)[:500]!r}"
                    )

                    break

                if not data:
                    last_error = RuntimeError(
                        f"Binance returned an empty candle list from "
                        f"{url}."
                    )

                    break

                # Basic validation of the first candle.
                if not isinstance(data[0], list) or len(data[0]) < 6:
                    last_error = RuntimeError(
                        f"Binance returned malformed candle data from "
                        f"{url}: {str(data[0])[:500]!r}"
                    )

                    break

                return data

            except httpx.TimeoutException as exc:
                last_error = exc

                if attempt + 1 < BINANCE_RETRIES_PER_HOST:
                    time.sleep(1.0 + attempt)
                    continue

                break

            except httpx.NetworkError as exc:
                last_error = exc

                if attempt + 1 < BINANCE_RETRIES_PER_HOST:
                    time.sleep(1.0 + attempt)
                    continue

                break

            except httpx.HTTPStatusError as exc:
                last_error = exc

                if attempt + 1 < BINANCE_RETRIES_PER_HOST:
                    time.sleep(1.0 + attempt)
                    continue

                break

            except RuntimeError as exc:
                last_error = exc

                # Data/route problems should move to another endpoint
                # instead of repeatedly requesting the same bad response.
                break

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Unable to retrieve Binance market data from any endpoint."
    )


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

    requested = max(
        BINANCE_BATCH_SIZE,
        min(1500, int(limit)),
    )

    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError("Symbol cannot be empty.")

    if not interval:
        raise ValueError("Interval cannot be empty.")

    all_klines: list[list[Any]] = []

    # We walk backwards from the newest available candle.
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

            batch_limit = min(
                BINANCE_BATCH_SIZE,
                remaining,
            )

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
            # Prepend the older batch.
            all_klines = batch + all_klines

            # If Binance returned fewer candles than requested,
            # there is no need to request another historical batch.
            if len(batch) < batch_limit:
                break

            # Find the earliest candle in this batch.
            try:
                earliest_open_time = int(batch[0][0])
            except (TypeError, ValueError, IndexError) as exc:
                raise RuntimeError(
                    "Binance returned malformed candle timestamps."
                ) from exc

            # The next request must end immediately before this candle.
            end_time = earliest_open_time - 1

            time.sleep(BINANCE_REQUEST_DELAY)

    # -------------------------------------------------------------
    # Remove duplicates by candle open timestamp.
    # -------------------------------------------------------------
    unique: dict[int, list[Any]] = {}

    for row in all_klines:
        if not row:
            continue

        try:
            timestamp = int(row[0])
        except (TypeError, ValueError, IndexError):
            continue

        unique[timestamp] = row

    result = sorted(
        unique.values(),
        key=lambda row: int(row[0]),
    )

    result = result[-requested:]

    if not result:
        raise RuntimeError(
            "Binance returned no usable historical candles."
        )

    return result


def build_dataset(
    klines: list[list[Any]],
    horizon: int = 6,
    threshold: float = 0.0025,
) -> list[dict[str, Any]]:
    """
    Convert OHLCV candles into supervised-learning examples.

    Each example contains:
        - timestamp
        - engineered features
        - future-direction label
        - realized future return
    """

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
        if len(row) < 6:
            continue

        try:
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
        except (TypeError, ValueError, IndexError):
            continue

    if len(candles) < 50:
        raise ValueError(
            "Not enough valid OHLCV candles after validation."
        )

    rows: list[dict[str, Any]] = []

    lookback = 24

    if len(candles) <= lookback + horizon:
        raise ValueError(
            "Not enough candles for the requested lookback and horizon."
        )

    for i in range(
        lookback,
        len(candles) - horizon,
    ):
        window = candles[
            i - lookback:i + 1
        ]

        last = window[-1]
        prev = window[-2]

        returns = [
            window[j]["close"] / window[j - 1]["close"] - 1
            for j in range(
                1,
                len(window),
            )
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

        # Three-class target:
        #
        #  1  = price rises beyond threshold
        #  0  = neutral
        # -1  = price falls beyond threshold
        label = (
            1
            if future > threshold
            else -1
            if future < -threshold
            else 0
        )

        features = {
            "return_1": (
                returns[-1]
                if returns
                else 0.0
            ),

            "range_pct": (
                (last["high"] - last["low"])
                / last["close"]
                if last["close"]
                else 0.0
            ),

            "volume_change": vol_change,

            # These are intentionally neutral during historical
            # bootstrap because these values are not contained in
            # the OHLCV candle payload.
            #
            # The live intelligence layer can populate them later.
            "order_book_imbalance": 0.0,
            "funding_rate": 0.0,
            "open_interest_change": 0.0,
            "news_risk": 0.0,
            "news_sentiment": 0.0,

            "volatility_proxy": min(
                1.0,
                vol * 10,
            ),

            "trend_strength": min(
                1.0,
                abs(mean_ret) * 80,
            ),

            "momentum": max(
                -1.0,
                min(
                    1.0,
                    mean_ret * 40,
                ),
            ),

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


def validate_and_promote(
    rows: list[dict[str, Any]],
    version: str = "bootstrap",
) -> dict[str, Any]:
    """
    Validate a candidate model using walk-forward testing.

    The model is promoted only when:
        - accuracy >= 52%
        - balanced accuracy >= 50%
        - average simulated return > 0
    """

    if not rows:
        return {
            "status": "REJECTED",
            "reason": "No training rows were supplied.",
            "rows": 0,
        }

    folds = walk_forward(
        rows,
        min_train=max(
            300,
            min(
                700,
                len(rows) // 2,
            ),
        ),
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
        model = Pipeline([
            (
                "scale",
                StandardScaler(),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                ),
            ),
        ])

        X = [
            [
                r["features"].get(
                    k,
                    0.0,
                )
                for k in FEATURES
            ]
            for r in fold.train
        ]

        y = [
            r["label"]
            for r in fold.train
        ]

        # A three-class classifier needs all three
        # classes in the training data.
        if len(set(y)) < 3:
            continue

        model.fit(X, y)

        Xtest = [
            [
                r["features"].get(
                    k,
                    0.0,
                )
                for k in FEATURES
            ]
            for r in fold.test
        ]

        pred = model.predict(Xtest)
        probs = model.predict_proba(Xtest)

        actual_labels = [
            r["label"]
            for r in fold.test
        ]

        for r, p, prob in zip(
            fold.test,
            pred,
            probs,
        ):
            confidence = float(
                max(prob)
            )

            realized = float(
                r["outcome_return"]
            )

            # Only simulate a trade when confidence
            # reaches the configured threshold.
            if confidence >= 0.55:
                if int(p) == r["label"]:
                    trade_return = realized
                else:
                    trade_return = -abs(realized)
            else:
                trade_return = 0.0

            predictions.append(
                (
                    r["label"],
                    int(p),
                    trade_return,
                )
            )

        fold_reports.append({
            "train": len(fold.train),
            "test": len(fold.test),
            "accuracy": float(
                np.mean(
                    np.asarray(pred)
                    == np.asarray(actual_labels)
                )
            ),
        })

    # Every fold may be skipped if its training data
    # does not contain all three target classes.
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

    metrics = evaluate_predictions(
        predictions
    )

    metrics["folds"] = len(
        fold_reports
    )

    metrics["folds_detail"] = (
        fold_reports
    )

    # Promotion gate.
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

    # Promote only after passing walk-forward validation.
    predictive_model.train(
        rows,
        version=version,
        min_rows=max(
            1,
            len(rows),
        ),
    )

    return {
        "status": "PROMOTED",
        "version": version,
        "metrics": metrics,
        "rows": len(rows),
    }
