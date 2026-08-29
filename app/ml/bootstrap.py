from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.predictive import FEATURES, predictive_model
from app.ml.validation import walk_forward, evaluate_predictions


BINANCE_KLINES = "https://fapi.binance.com/fapi/v1/klines"

# Binance allows large requests, but smaller batches are more reliable,
# especially when the deployment IP has recently received a rate-limit response.
BINANCE_BATCH_SIZE = 500


def fetch_binance_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 1500,
) -> list[list[Any]]:
    """
    Fetch historical Binance futures candles in smaller batches.

    The returned candles are ordered from oldest to newest and limited
    to the requested number of candles.
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
            batch_limit = min(BINANCE_BATCH_SIZE, requested - len(all_klines))

            params: dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
                "limit": batch_limit,
            }

            if end_time is not None:
                params["endTime"] = end_time

            last_error: Exception | None = None
            response: httpx.Response | None = None

            # Retry temporary Binance/Cloudflare responses.
            for attempt in range(4):
                try:
                    response = client.get(BINANCE_KLINES, params=params)

                    if response.status_code == 418:
                        # Binance has temporarily rejected the IP.
                        # Wait progressively longer before retrying.
                        wait_seconds = 2 ** attempt
                        time.sleep(wait_seconds)
                        continue

                    response.raise_for_status()
                    break

                except httpx.HTTPError as exc:
                    last_error = exc

                    if attempt < 3:
                        time.sleep(2 ** attempt)
                    else:
                        raise

            if response is None:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("Binance returned no response.")

            response.raise_for_status()

            batch = response.json()

            if not isinstance(batch, list) or not batch:
                break

            all_klines = batch + all_klines

            # If fewer candles than requested were returned, there is
            # nothing more to fetch.
            if len(batch) < batch_limit:
                break

            # Fetch the previous batch before the earliest candle.
            earliest_open_time = int(batch[0][0])
            end_time = earliest_open_time - 1

            # Small delay between requests to reduce rate-limit pressure.
            time.sleep(0.25)

    # Remove duplicate candles by open timestamp.
    unique: dict[int, list[Any]] = {}

    for row in all_klines:
        if row:
            unique[int(row[0])] = row

    result = sorted(unique.values(), key=lambda row: int(row[0]))

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
