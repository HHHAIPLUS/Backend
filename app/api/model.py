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

    limit: int = Field(
        default=1500,
        ge=500,
        le=1500,
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
        # Historical market data
        # ---------------------------------------------------------
        #
        # IMPORTANT:
        #
        # This intentionally uses fetch_historical_klines()
        # instead of fetch_binance_klines().
        #
        # fetch_historical_klines() already implements:
        #
        #   1. Binance
        #   2. Bitget fallback
        #
        # Therefore, if Binance is unavailable, blocked, or
        # returns an unusable response, HHHAI can continue with
        # Bitget instead of immediately failing the bootstrap.
        #
        raw, provider = fetch_historical_klines(
            symbol=request.symbol,
            interval=request.interval,
            limit=request.limit,
        )

        # ---------------------------------------------------------
        # Build supervised-learning dataset
        # ---------------------------------------------------------
        rows = build_dataset(
            raw,
            request.horizon,
            request.threshold,
        )

        # The bootstrap requires enough historical examples
        # to perform meaningful walk-forward validation.
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

        # Include the actual historical-data provider in the
        # bootstrap response so we know whether Binance or
        # Bitget supplied the training data.
        result["data_provider"] = provider

        result["symbol"] = request.symbol.upper()
        result["interval"] = request.interval
        result["requested_candles"] = request.limit
        result["training_rows"] = len(rows)

        # ---------------------------------------------------------
        # Persist only a successfully promoted model
        # ---------------------------------------------------------
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
