# HHHAI Backend

Backend-driven autonomous trading intelligence for personal use.

## Runtime

- FastAPI API
- Continuous market/position observation worker
- Multi-agent intelligence council
- Scenario and adversarial layers
- Validated predictive-model gate
- Dynamic position-management engine
- Capital guard and kill switch
- Binance USDⓈ-M Futures adapter
- Bitget Futures adapter with demo-trading support
- Supabase persistence
- Controlled learning and model-promotion workflow

## Safety defaults

`LIVE_TRADING_ENABLED=false` and `TESTNET_TRADING_ENABLED=false` are the defaults. Do not enable real-money execution until independent walk-forward, paper/demo, stress and operational checks have passed.

## Supabase

Run `migrations/001_hhhai.sql` in the Supabase SQL editor. The backend uses the server-only service-role key; never expose it to the frontend.

Supabase REST exposes database tables through `/rest/v1`, which is what the persistence layer uses.

## Exchanges

### Binance

Set `BINANCE_API_KEY`, `BINANCE_API_SECRET`, and keep `BINANCE_TESTNET=true` during validation. The adapter uses signed USDⓈ-M Futures REST requests.

### Bitget

Set `BITGET_API_KEY`, `BITGET_API_SECRET`, and `BITGET_PASSPHRASE`. With `BITGET_TESTNET=true`, the adapter uses Bitget demo trading and sends the required `paptrading: 1` header.

## Predictive model

HHHAI does not pretend the deterministic baseline is a proven AI predictor. Production directional decisions require a validated model artifact.

Train only through:

`python tools/train_model.py <historical.csv> --version <version>`

The trainer performs chronological walk-forward validation and refuses promotion when the configured accuracy, balanced-accuracy or average-return gates fail.

No model artifact is bundled because an unvalidated/stale model must never silently become production intelligence.

## Deployment

Render uses `render.yaml` /:

`uvicorn app.main:app --host 0.0.0.0 --port $PORT`

The frontend is deployed separately to Vercel.

## Tests

From this directory:

`pytest -q`

All repository tests must pass before deployment.

## Autonomous trading path

The backend now contains an end-to-end coordinator at `/api/trading` that connects:

market/world data → specialist council → scenario analysis → adversarial challenge → validated predictive model → decision fusion → trade-quality optimization → capital guard → paper/testnet/live execution gate → decision journal.

Execution is fail-closed by default. Set `HHHAI_TRADING_MODE=paper` for continuous simulated execution. For exchange testnet/demo execution, use `HHHAI_TRADING_MODE=testnet` and explicitly enable `TESTNET_TRADING_ENABLED=true` after credentials are configured. Real-money execution remains separately gated by `LIVE_TRADING_ENABLED=true` and should only be enabled after independent validation.

Use the authenticated admin control room to bootstrap a predictive model from historical Binance candles and to start/stop the autonomous worker. A model is never accepted unless its walk-forward gates pass.

Run `migrations/002_autonomous_trading.sql` after the original migration so validated model artifacts survive backend restarts when Supabase is configured.


## Deployment wiring

- Frontend: set `VITE_HHHAI_API_URL` in Vercel to the Render backend URL.
- Backend: set `CORS_ORIGINS` in Render to the exact Vercel origin(s).
- Keep exchange credentials, Supabase service-role key, and `HHHAI_ADMIN_TOKEN` only on Render.
- `HHHAI_AUTOTRADING_ENABLED` defaults to `false`; `HHHAI_TRADING_MODE=paper` is the safe default.
- Testnet and live execution are separate gates. Live execution does not require the testnet flag.
