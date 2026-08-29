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
# CONFIGURATION
# ============================================================================

# Binance Futures REST endpoints.
#
# Binance may temporarily reject a Render deployment IP on one route.
# We therefore try several public Binance routes before giving up.
BINANCE_KLINES_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

BINANCE_KLINES_PATH = "/fapi/v1/klines"

# Bitget USDT-M Futures REST endpoint.
BITGET_KLINES_URL = (
    "https://api.bitget.com/api/v2/mix/market/candles"
)

# Maximum candles requested in one Binance request.
BINANCE_BATCH_SIZE = 500

# Conservative Bitget batch size.
#
# Keeping this below the exchange's maximum makes the bootstrap
# less sensitive to upstream/proxy limitations.
BITGET_BATCH_SIZE = 200

# Number of attempts for temporary Binance failures.
BINANCE_RETRIES_PER_HOST = 2

# Number of attempts for temporary Bitget failures.
BITGET_RETRIES_PER_REQUEST = 2

# Delay between historical requests.
HISTORICAL_REQUEST_DELAY = 0.25

# HTTP timeout.
HTTP_TIMEOUT = httpx.Timeout(
    30.0,
    connect=10.0,
)


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def _is_json_response(response: httpx.Response) -> bool:
    """
    Determine whether a response is JSON or at least looks like JSON.

    Some proxies/CDNs do not always provide the expected content-type.
    """
    body = response.text.strip()

    content_type = response.headers.get(
        "content-type",
        "",
    ).lower()

    if "application/json" in content_type:
        return True

    return (
        body.startswith("[")
        or body.startswith("{")
    )


def _validate_candle_row(
    row: Any,
) -> bool:
    """
    Basic validation for a normalized or exchange-native candle row.
    """
    if not isinstance(row, list):
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
    except (TypeError, ValueError):
        return False

    return True


def _deduplicate_klines(
    klines: list[list[Any]],
) -> list[list[Any]]:
    """
    Remove duplicate candles using candle open timestamp.

    Returns candles ordered oldest -> newest.
    """

    unique: dict[int, list[Any]] = {}

    for row in klines:
        if not _validate_candle_row(row):
            continue

        try:
            timestamp = int(row[0])
        except (TypeError, ValueError):
            continue

        unique[timestamp] = row

    return sorted(
        unique.values(),
        key=lambda row: int(row[0]),
    )


# ============================================================================
# BINANCE
# ============================================================================


def _request_binance_batch(
    client: httpx.Client,
    symbol: str,
    interval: str,
    limit: int,
    end_time: int | None,
) -> list[list[Any]]:
    """
    Request one Binance Futures candle batch.

    A Binance endpoint is considered unusable when it returns:

        - HTTP 418
        - HTTP 202 with an empty body
        - another empty response
        - non-JSON content
        - invalid JSON
        - malformed candle data

    HHHAI then moves to another Binance endpoint.
    """

    params: dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(
            BINANCE_BATCH_SIZE,
            max(1, int(limit)),
        ),
    }

    if end_time is not None:
        params["endTime"] = end_time

    last_error: Exception | None = None

    for host in BINANCE_KLINES_HOSTS:
        url = f"{host}{BINANCE_KLINES_PATH}"

        for attempt in range(
            BINANCE_RETRIES_PER_HOST
        ):
            response: httpx.Response | None = None

            try:
                response = client.get(
                    url,
                    params=params,
                )

                status = response.status_code
                body = response.text.strip()

                # ---------------------------------------------------------
                # HTTP 418
                # ---------------------------------------------------------

                if status == 418:
                    last_error = RuntimeError(
                        f"Binance returned HTTP 418 from {url} "
                        "(rate-limit/IP restriction)."
                    )

                    # Do not hammer this route.
                    time.sleep(
                        1.0 + attempt
                    )

                    # Move to the next Binance route.
                    break

                # ---------------------------------------------------------
                # HTTP 202
                # ---------------------------------------------------------

                if status == 202:
                    if body:
                        last_error = RuntimeError(
                            f"Binance returned HTTP 202 from {url} "
                            f"with unexpected body: {body[:500]!r}"
                        )
                    else:
                        last_error = RuntimeError(
                            f"Binance returned an empty response "
                            f"from {url} (HTTP 202)."
                        )

                    # Treat this endpoint as unusable.
                    break

                # ---------------------------------------------------------
                # Other HTTP errors
                # ---------------------------------------------------------

                if status >= 400:
                    last_error = httpx.HTTPStatusError(
                        f"Binance HTTP {status} from {url}",
                        request=response.request,
                        response=response,
                    )

                    if (
                        attempt + 1
                        < BINANCE_RETRIES_PER_HOST
                    ):
                        time.sleep(
                            1.0 + attempt
                        )
                        continue

                    break

                # ---------------------------------------------------------
                # Empty response
                # ---------------------------------------------------------

                if not body:
                    last_error = RuntimeError(
                        f"Binance returned an empty response "
                        f"from {url} (HTTP {status})."
                    )

                    break

                # ---------------------------------------------------------
                # JSON validation
                # ---------------------------------------------------------

                if not _is_json_response(
                    response
                ):
                    content_type = response.headers.get(
                        "content-type",
                        "",
                    )

                    last_error = RuntimeError(
                        f"Binance returned a non-JSON response "
                        f"from {url} "
                        f"(HTTP {status}, "
                        f"content-type={content_type!r}, "
                        f"body={body[:500]!r})."
                    )

                    break

                try:
                    data = response.json()

                except ValueError as exc:
                    last_error = RuntimeError(
                        f"Binance returned invalid JSON "
                        f"from {url}: {body[:500]!r}"
                    )

                    if (
                        attempt + 1
                        < BINANCE_RETRIES_PER_HOST
                    ):
                        time.sleep(
                            1.0 + attempt
                        )
                        continue

                    break

                # ---------------------------------------------------------
                # Payload validation
                # ---------------------------------------------------------

                if not isinstance(
                    data,
                    list,
                ):
                    last_error = RuntimeError(
                        f"Binance returned an unexpected "
                        f"candle response from {url}: "
                        f"{str(data)[:500]!r}"
                    )

                    break

                if not data:
                    last_error = RuntimeError(
                        f"Binance returned an empty candle "
                        f"list from {url}."
                    )

                    break

                valid_rows = [
                    row
                    for row in data
                    if _validate_candle_row(row)
                ]

                if not valid_rows:
                    last_error = RuntimeError(
                        f"Binance returned no valid candle "
                        f"rows from {url}."
                    )

                    break

                return valid_rows

            except httpx.TimeoutException as exc:
                last_error = exc

                if (
                    attempt + 1
                    < BINANCE_RETRIES_PER_HOST
                ):
                    time.sleep(
                        1.0 + attempt
                    )
                    continue

                break

            except httpx.NetworkError as exc:
                last_error = exc

                if (
                    attempt + 1
                    < BINANCE_RETRIES_PER_HOST
                ):
                    time.sleep(
                        1.0 + attempt
                    )
                    continue

                break

            except httpx.HTTPStatusError as exc:
                last_error = exc

                if (
                    attempt + 1
                    < BINANCE_RETRIES_PER_HOST
                ):
                    time.sleep(
                        1.0 + attempt
                    )
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
    """
    Fetch historical Binance Futures candles.

    Returns:
        oldest -> newest
        duplicate-free
        limited to requested number of candles
    """

    requested = max(
        BINANCE_BATCH_SIZE,
        min(
            1500,
            int(limit),
        ),
    )

    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError(
            "Symbol cannot be empty."
        )

    if not interval:
        raise ValueError(
            "Interval cannot be empty."
        )

    all_klines: list[list[Any]] = []

    # Walk backwards through history.
    end_time: int | None = None

    headers = {
        "User-Agent": "HHHAI/1.0",
        "Accept": "application/json",
    }

    with httpx.Client(
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
        trust_env=False,
        headers=headers,
    ) as client:

        while len(all_klines) < requested:

            remaining = (
                requested
                - len(all_klines)
            )

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

            # Binance gives us the newer batch.
            # We are walking backwards, so prepend it.
            all_klines = (
                batch
                + all_klines
            )

            if len(batch) < batch_limit:
                break

            try:
                earliest_open_time = int(
                    batch[0][0]
                )

            except (
                TypeError,
                ValueError,
                IndexError,
            ) as exc:
                raise RuntimeError(
                    "Binance returned malformed "
                    "candle timestamps."
                ) from exc

            end_time = (
                earliest_open_time
                - 1
            )

            time.sleep(
                HISTORICAL_REQUEST_DELAY
            )

    result = _deduplicate_klines(
        all_klines
    )

    result = result[-requested:]

    if not result:
        raise RuntimeError(
            "Binance returned no usable "
            "historical candles."
        )

    if len(result) < requested:
        raise RuntimeError(
            "Binance returned only "
            f"{len(result)} usable candles "
            f"out of {requested} requested."
        )

    return result


# ============================================================================
# BITGET
# ============================================================================


def _normalize_bitget_candle(
    row: Any,
) -> list[Any] | None:
    """
    Convert a Bitget v2 candle row into the OHLCV structure
    expected by build_dataset().

    Bitget candle format:

        [timestamp, open, high, low, close, volume, ...]
    """

    if not isinstance(row, list):
        return None

    if len(row) < 6:
        return None

    try:
        timestamp = int(
            float(row[0])
        )

        open_price = float(
            row[1]
        )

        high_price = float(
            row[2]
        )

        low_price = float(
            row[3]
        )

        close_price = float(
            row[4]
        )

        volume = float(
            row[5]
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if timestamp <= 0:
        return None

    if (
        open_price <= 0
        or high_price <= 0
        or low_price <= 0
        or close_price <= 0
    ):
        return None

    if high_price < low_price:
        return None

    return [
        timestamp,
        open_price,
        high_price,
        low_price,
        close_price,
        max(0.0, volume),
    ]


def _request_bitget_batch(
    client: httpx.Client,
    symbol: str,
    granularity: str,
    limit: int,
    end_time: int | None,
) -> list[list[Any]]:
    """
    Request one Bitget USDT-M Futures candle batch.

    Bitget's response is normalized into the same six-column
    OHLCV structure used by Binance.
    """

    params: dict[str, Any] = {
        "productType": "USDT-FUTURES",
        "symbol": symbol.upper(),
        "granularity": granularity,
        "limit": min(
            BITGET_BATCH_SIZE,
            max(1, int(limit)),
        ),
    }

    if end_time is not None:
        params["endTime"] = str(
            end_time
        )

    headers = {
        "User-Agent": "HHHAI/1.0",
        "Accept": "application/json",
    }

    last_error: Exception | None = None

    for attempt in range(
        BITGET_RETRIES_PER_REQUEST
    ):
        response: httpx.Response | None = None

        try:
            response = client.get(
                BITGET_KLINES_URL,
                params=params,
                headers=headers,
            )

            status = response.status_code
            body = response.text.strip()

            if status >= 400:
                last_error = httpx.HTTPStatusError(
                    f"Bitget HTTP {status}",
                    request=response.request,
                    response=response,
                )

                if (
                    attempt + 1
                    < BITGET_RETRIES_PER_REQUEST
                ):
                    time.sleep(
                        1.0 + attempt
                    )
                    continue

                break

            if not body:
                last_error = RuntimeError(
                    "Bitget returned an empty response."
                )

                if (
                    attempt + 1
                    < BITGET_RETRIES_PER_REQUEST
                ):
                    time.sleep(
                        1.0 + attempt
                    )
                    continue

                break

            if not _is_json_response(
                response
            ):
                last_error = RuntimeError(
                    "Bitget returned a non-JSON response: "
                    f"{body[:500]!r}"
                )

                if (
                    attempt + 1
                    < BITGET_RETRIES_PER_REQUEST
                ):
                    time.sleep(
                        1.0 + attempt
                    )
                    continue

                break

            try:
                payload = response.json()

            except ValueError as exc:
                last_error = RuntimeError(
                    "Bitget returned invalid JSON: "
                    f"{body[:500]!r}"
                )

                if (
                    attempt + 1
                    < BITGET_RETRIES_PER_REQUEST
                ):
                    time.sleep(
                        1.0 + attempt
                    )
                    continue

                raise last_error from exc

            if not isinstance(
                payload,
                dict,
            ):
                last_error = RuntimeError(
                    "Bitget returned an unexpected "
                    "response structure."
                )

                break

            code = str(
                payload.get(
                    "code",
                    "",
                )
            )

            # Bitget normally uses code "00000" for success.
            if code and code != "00000":
                message = payload.get(
                    "msg",
                    payload.get(
                        "message",
                        "Unknown Bitget error.",
                    ),
                )

                last_error = RuntimeError(
                    f"Bitget API error {code}: "
                    f"{message}"
                )

                if (
                    attempt + 1
                    < BITGET_RETRIES_PER_REQUEST
                ):
                    time.sleep(
                        1.0 + attempt
                    )
                    continue

                break

            raw_data = payload.get(
                "data"
            )

            if not isinstance(
                raw_data,
                list,
            ):
                last_error = RuntimeError(
                    "Bitget candle response "
                    "contains no data list."
                )

                break

            normalized: list[list[Any]] = []

            for row in raw_data:
                candle = _normalize_bitget_candle(
                    row
                )

                if candle is not None:
                    normalized.append(
                        candle
                    )

            if not normalized:
                last_error = RuntimeError(
                    "Bitget returned no usable "
                    "candle rows."
                )

                break

            return normalized

        except httpx.TimeoutException as exc:
            last_error = exc

            if (
                attempt + 1
                < BITGET_RETRIES_PER_REQUEST
            ):
                time.sleep(
                    1.0 + attempt
                )
                continue

            break

        except httpx.NetworkError as exc:
            last_error = exc

            if (
                attempt + 1
                < BITGET_RETRIES_PER_REQUEST
            ):
                time.sleep(
                    1.0 + attempt
                )
                continue

            break

        except httpx.HTTPStatusError as exc:
            last_error = exc

            if (
                attempt + 1
                < BITGET_RETRIES_PER_REQUEST
            ):
                time.sleep(
                    1.0 + attempt
                )
                continue

            break

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Unable to retrieve Bitget market data."
    )


def fetch_bitget_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 1500,
) -> list[list[Any]]:
    """
    Fetch historical Bitget USDT-M Futures candles.

    Returns the same normalized six-column OHLCV structure
    as fetch_binance_klines().
    """

    requested = max(
        BITGET_BATCH_SIZE,
        min(
            1500,
            int(limit),
        ),
    )

    symbol = symbol.upper().strip()

    if not symbol:
        raise ValueError(
            "Symbol cannot be empty."
        )

    if not interval:
        raise ValueError(
            "Interval cannot be empty."
        )

    # Bitget uses granularity names such as 5m.
    granularity = interval.lower()

    all_klines: list[list[Any]] = []

    # Walk backwards from the newest available candle.
    end_time: int | None = None

    headers = {
        "User-Agent": "HHHAI/1.0",
        "Accept": "application/json",
    }

    with httpx.Client(
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
        trust_env=True,
        headers=headers,
    ) as client:

        while len(all_klines) < requested:

            remaining = (
                requested
                - len(all_klines)
            )

            batch_limit = min(
                BITGET_BATCH_SIZE,
                remaining,
            )

            batch = _request_bitget_batch(
                client=client,
                symbol=symbol,
                granularity=granularity,
                limit=batch_limit,
                end_time=end_time,
            )

            if not batch:
                break

            all_klines = (
                batch
                + all_klines
            )

            if len(batch) < batch_limit:
                break

            # Bitget may return candles newest-first.
            batch_sorted = sorted(
                batch,
                key=lambda row: int(
                    row[0]
                ),
            )

            earliest_open_time = int(
                batch_sorted[0][0]
            )

            # Ask for candles before the earliest
            # candle we already received.
            end_time = (
                earliest_open_time
                - 1
            )

            time.sleep(
                HISTORICAL_REQUEST_DELAY
            )

    result = _deduplicate_klines(
        all_klines
    )

    result = result[-requested:]

    if not result:
        raise RuntimeError(
            "Bitget returned no usable "
            "historical candles."
        )

    if len(result) < requested:
        raise RuntimeError(
            "Bitget returned only "
            f"{len(result)} usable candles "
            f"out of {requested} requested."
        )

    return result


# ============================================================================
# MULTI-EXCHANGE HISTORICAL DATA
# ============================================================================


def fetch_historical_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 1500,
) -> tuple[list[list[Any]], str]:
    """
    Fetch historical candles using provider fallback.

    Provider order:

        1. Binance
        2. Bitget

    This keeps both exchanges in HHHAI while preventing one exchange's
    temporary API problem from blocking model bootstrap.

    Returns:

        (candles, provider_name)
    """

    errors: list[str] = []

    # ------------------------------------------------------------------
    # Provider 1: Binance
    # ------------------------------------------------------------------

    try:
        klines = fetch_binance_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

        return (
            klines,
            "binance",
        )

    except Exception as exc:
        errors.append(
            "binance: "
            f"{type(exc).__name__}: {exc}"
        )

    # ------------------------------------------------------------------
    # Provider 2: Bitget
    # ------------------------------------------------------------------

    try:
        klines = fetch_bitget_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )

        return (
            klines,
            "bitget",
        )

    except Exception as exc:
        errors.append(
            "bitget: "
            f"{type(exc).__name__}: {exc}"
        )

    raise RuntimeError(
        "Historical market data unavailable "
        "from all configured providers. "
        + " | ".join(errors)
    )


# ============================================================================
# DATASET BUILDING
# ============================================================================


def build_dataset(
    klines: list[list[Any]],
    horizon: int = 6,
    threshold: float = 0.0025,
) -> list[dict[str, Any]]:
    """
    Convert OHLCV candles into supervised-learning examples.

    Each example contains:

        - observation timestamp
        - engineered features
        - future-direction label
        - realized future return
    """

    if horizon <= 0:
        raise ValueError(
            "Horizon must be greater than zero."
        )

    if threshold <= 0:
        raise ValueError(
            "Threshold must be greater than zero."
        )

    if len(klines) < 50:
        raise ValueError(
            "Not enough candles to build "
            f"the dataset: {len(klines)}."
        )

    candles: list[dict[str, Any]] = []

    for row in klines:

        if not isinstance(
            row,
            list,
        ):
            continue

        if len(row) < 6:
            continue

        try:
            timestamp = int(
                row[0]
            )

            open_price = float(
                row[1]
            )

            high_price = float(
                row[2]
            )

            low_price = float(
                row[3]
            )

            close_price = float(
                row[4]
            )

            volume = float(
                row[5]
            )

        except (
            TypeError,
            ValueError,
            IndexError,
        ):
            continue

        if (
            timestamp <= 0
            or close_price <= 0
            or high_price <= 0
            or low_price <= 0
        ):
            continue

        candles.append({
            "observed_at": datetime.fromtimestamp(
                timestamp / 1000,
                timezone.utc,
            ).isoformat(),

            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": max(
                0.0,
                volume,
            ),
        })

    # Always sort the normalized dataset chronologically.
    candles.sort(
        key=lambda row: row["observed_at"]
    )

    if len(candles) < 50:
        raise ValueError(
            "Not enough valid OHLCV candles "
            "after validation."
        )

    lookback = 24

    if (
        len(candles)
        <= lookback + horizon
    ):
        raise ValueError(
            "Not enough candles for the "
            "requested lookback and horizon."
        )

    rows: list[dict[str, Any]] = []

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
            (
                window[j]["close"]
                / window[j - 1]["close"]
            ) - 1.0

            for j in range(
                1,
                len(window),
            )

            if window[j - 1]["close"]
        ]

        recent_returns = returns[-12:]

        mean_ret = (
            float(
                np.mean(
                    recent_returns
                )
            )
            if recent_returns
            else 0.0
        )

        vol = (
            float(
                np.std(
                    recent_returns
                )
            )
            if len(recent_returns) > 1
            else 0.0
        )

        vol_change = (
            (
                last["volume"]
                / prev["volume"]
            ) - 1.0

            if prev["volume"]
            else 0.0
        )

        future = (
            (
                candles[i + horizon]["close"]
                / last["close"]
            ) - 1.0

            if last["close"]
            else 0.0
        )

        # --------------------------------------------------------------
        # Three-class target
        #
        #  1  = price rises beyond threshold
        #  0  = neutral
        # -1  = price falls beyond threshold
        # --------------------------------------------------------------

        if future > threshold:
            label = 1

        elif future < -threshold:
            label = -1

        else:
            label = 0

        # --------------------------------------------------------------
        # Features
        #
        # During historical OHLCV bootstrap, features that cannot be
        # reconstructed safely from the candle feed remain neutral.
        #
        # The live intelligence layer can populate these later.
        # --------------------------------------------------------------

        features = {
            "return_1": (
                returns[-1]
                if returns
                else 0.0
            ),

            "range_pct": (
                (
                    last["high"]
                    - last["low"]
                )
                / last["close"]

                if last["close"]
                else 0.0
            ),

            "volume_change": vol_change,

            "order_book_imbalance": 0.0,

            "funding_rate": 0.0,

            "open_interest_change": 0.0,

            "news_risk": 0.0,

            "news_sentiment": 0.0,

            "volatility_proxy": min(
                1.0,
                max(
                    0.0,
                    vol * 10.0,
                ),
            ),

            "trend_strength": min(
                1.0,
                abs(mean_ret) * 80.0,
            ),

            "momentum": max(
                -1.0,
                min(
                    1.0,
                    mean_ret * 40.0,
                ),
            ),

            "liquidity_stress": 0.0,
        }

        rows.append({
            "observed_at": last[
                "observed_at"
            ],

            "features": features,

            "label": label,

            "outcome_return": future,
        })

    if not rows:
        raise ValueError(
            "Dataset construction produced "
            "zero training rows."
        )

    return rows


# ============================================================================
# WALK-FORWARD VALIDATION + PROMOTION
# ============================================================================


def validate_and_promote(
    rows: list[dict[str, Any]],
    version: str = "bootstrap",
) -> dict[str, Any]:
    """
    Validate a candidate model using walk-forward testing.

    Promotion requirements:

        accuracy >= 52%
        balanced_accuracy >= 50%
        average simulated return > 0

    The candidate is only promoted after passing these gates.
    """

    if not rows:
        return {
            "status": "REJECTED",
            "reason": (
                "No training rows were supplied."
            ),
            "rows": 0,
        }

    min_train = max(
        300,
        min(
            700,
            len(rows) // 2,
        ),
    )

    folds = walk_forward(
        rows,
        min_train=min_train,
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

    fold_reports: list[
        dict[str, Any]
    ] = []

    predictions: list[
        tuple[int, int, float]
    ] = []

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
            int(
                r["label"]
            )

            for r in fold.train
        ]

        # A three-class classifier needs:
        #
        # -1
        #  0
        #  1
        #
        # in the training set.

        if len(set(y)) < 3:
            continue

        model.fit(
            X,
            y,
        )

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

        pred = model.predict(
            Xtest
        )

        probs = model.predict_proba(
            Xtest
        )

        actual_labels = [
            int(
                r["label"]
            )

            for r in fold.test
        ]

        for (
            r,
            p,
            prob,
        ) in zip(
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

                if int(p) == int(
                    r["label"]
                ):
                    trade_return = realized

                else:
                    trade_return = -abs(
                        realized
                    )

            else:
                trade_return = 0.0

            predictions.append(
                (
                    int(
                        r["label"]
                    ),

                    int(p),

                    trade_return,
                )
            )

        fold_accuracy = float(
            np.mean(
                np.asarray(pred)
                == np.asarray(
                    actual_labels
                )
            )
        )

        fold_reports.append({
            "train": len(
                fold.train
            ),

            "test": len(
                fold.test
            ),

            "accuracy": fold_accuracy,
        })

    # --------------------------------------------------------------
    # No valid predictions
    # --------------------------------------------------------------

    if not predictions:
        return {
            "status": "REJECTED",

            "version": (
                predictive_model.version
            ),

            "reason": (
                "No valid walk-forward "
                "predictions were produced."
            ),

            "rows": len(rows),

            "folds": len(
                fold_reports
            ),

            "folds_detail": (
                fold_reports
            ),
        }

    # --------------------------------------------------------------
    # Evaluate predictions
    # --------------------------------------------------------------

    metrics = evaluate_predictions(
        predictions
    )

    metrics["folds"] = len(
        fold_reports
    )

    metrics["folds_detail"] = (
        fold_reports
    )

    # --------------------------------------------------------------
    # Promotion gate
    # --------------------------------------------------------------

    if (
        metrics["accuracy"] < 0.52

        or metrics[
            "balanced_accuracy"
        ] < 0.50

        or metrics[
            "avg_return"
        ] <= 0
    ):

        return {
            "status": "REJECTED",

            "version": (
                predictive_model.version
            ),

            "metrics": metrics,

            "rows": len(rows),
        }

    # --------------------------------------------------------------
    # Promote the validated model
    # --------------------------------------------------------------

    report = predictive_model.train(
        rows,
        version=version,
        min_rows=max(
            1,
            len(rows),
        ),
    )

    if not report.trained:
        return {
            "status": "REJECTED",

            "version": (
                predictive_model.version
            ),

            "reason": report.reason,

            "metrics": metrics,

            "rows": len(rows),
        }

    return {
        "status": "PROMOTED",

        "version": version,

        "metrics": metrics,

        "rows": len(rows),
    }
