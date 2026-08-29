from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.api.admin import require_admin
from app.ml.predictive import predictive_model
from app.ml.bootstrap import (
    fetch_historical_klines,
    build_dataset,
    validate_and_promote,
)
from app.ml.model_persistence import persist_model


router = APIRouter(
    prefix="/api/model",
    tags=["predictive-model"],
)


class BootstrapRequest(BaseModel):
    symbol: str = Field(
        default="BTCUSDT",
        min_length=5,
        max_length=30,
    )

    interval: str = "5m"

    # Allow larger bootstrap datasets.
    # The endpoint will use at least 5,000 candles.
    limit: int = Field(
        default=5000,
        ge=500,
        le=10000,
    )

    horizon: int = Field(
        default=6,
        ge=1,
        le=24,
    )

    threshold: float = Field(
        default=0.0025,
        gt=0,
        lt=0.1,
    )


@router.get("/status")
def status():
    return {
        "version": predictive_model.version,
        "trained": predictive_model.model is not None,
        "execution_gate": predictive_model.model is not None,
        "artifact": str(predictive_model.model_path),
    }


@router.post("/bootstrap")
def bootstrap(
    request: BootstrapRequest,
    x_hhhai_admin_token: str | None = Header(
        default=None,
    ),
):
    require_admin(x_hhhai_admin_token)

    try:
        # ---------------------------------------------------------
        # Determine the actual bootstrap history size.
        # ---------------------------------------------------------
        #
        # We require at least 5,000 candles for bootstrap.
        # The request may ask for more, up to the API maximum
        # of 10,000 candles.
        #
        # This gives walk-forward validation substantially more
        # historical examples than the old 1,500-candle limit.
        effective_limit = max(
            5000,
            request.limit,
        )

        # ---------------------------------------------------------
        # Historical market data
        # ---------------------------------------------------------
        #
        # fetch_historical_klines() is responsible for exchange
        # fallback:
        #
        #   Binance -> Bitget
        #
        # The API layer does not directly choose an exchange.
        # It asks the market-data layer for historical candles and
        # records which provider actually supplied them.
        raw, provider = fetch_historical_klines(
            symbol=request.symbol,
            interval=request.interval,
            limit=effective_limit,
        )

        # ---------------------------------------------------------
        # Build supervised-learning dataset
        # ---------------------------------------------------------
        rows = build_dataset(
            raw,
            request.horizon,
            request.threshold,
        )

        # ---------------------------------------------------------
        # Minimum usable training dataset
        # ---------------------------------------------------------
        #
        # With 5,000 candles and a 24-candle lookback plus the
        # requested horizon, we should normally have thousands of
        # usable examples. Keep this guard so a broken/incomplete
        # provider response cannot silently reach model training.
        if len(rows) < 500:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Historical dataset contains only "
                    f"{len(rows)} usable rows; "
                    "at least 500 are required."
                ),
            )

        # ---------------------------------------------------------
        # Walk-forward validation + promotion
        # ---------------------------------------------------------
        result = validate_and_promote(
            rows,
            version=(
                f"bootstrap-"
                f"{request.symbol.upper()}-"
                f"{request.interval}"
            ),
        )

        # ---------------------------------------------------------
        # Bootstrap metadata
        # ---------------------------------------------------------
        result["data_provider"] = provider
        result["symbol"] = request.symbol.upper()
        result["interval"] = request.interval
        result["requested_candles"] = request.limit
        result["effective_candles"] = effective_limit
        result["training_rows"] = len(rows)

        # ---------------------------------------------------------
        # Persist only a successfully promoted model
        # ---------------------------------------------------------
        #
        # A rejected candidate must never overwrite the existing
        # model artifact.
        if result.get("status") == "PROMOTED":
            import asyncio

            asyncio.run(
                persist_model(
                    result.get("metrics")
                )
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model bootstrap failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
