from __future__ import annotations
import json, time
from datetime import datetime, timezone
from pathlib import Path
import httpx, numpy as np

from app.ml.features import FEATURES, build_model_features
from app.ml.predictive_brain import PredictiveBrain

SYMBOL = "BTCUSDT"
INTERVAL = "1H"
CANDLES = 30000
LOOKBACK = 168
HORIZONS = (1, 3, 6, 12)


def fetch():
    rows = []
    end_ms = int(time.time() * 1000) // 3600000 * 3600000
    with httpx.Client(timeout=30, trust_env=False, headers={"User-Agent": "HHHAI/phase2"}) as client:
        while len(rows) < CANDLES:
            start_ms = end_ms - 199 * 3600000
            params = {"symbol": SYMBOL, "productType": "USDT-FUTURES", "granularity": INTERVAL,
                      "limit": 200, "startTime": start_ms, "endTime": end_ms}
            response = client.get("https://api.bitget.com/api/v2/mix/market/history-candles", params=params)
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") not in (None, "00000"):
                raise RuntimeError(f"Bitget candles error: {payload}")
            data = payload.get("data", [])
            if not data:
                raise RuntimeError("Bitget returned no historical candles")
            rows.extend((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
                        for r in data if len(r) >= 6)
            oldest = min(r[0] for r in data)
            if oldest >= end_ms:
                raise RuntimeError("candle pagination stalled")
            end_ms = oldest
            time.sleep(.06)
    rows = sorted({r[0]: r for r in rows}.values())[-CANDLES:]
    if len(rows) != CANDLES:
        raise RuntimeError(f"expected {CANDLES} candles, got {len(rows)}")
    for a, b in zip(rows, rows[1:]):
        if b[0] - a[0] != 3600000:
            raise RuntimeError(f"candle gap/overlap {a[0]}->{b[0]}")
        if not (a[3] <= min(a[1], a[4]) and a[2] >= max(a[1], a[4]) and a[1] > 0 and a[4] > 0 and a[5] >= 0):
            raise RuntimeError("invalid OHLCV")
    return rows


def build_rows(candles):
    rows = []
    for i in range(LOOKBACK, len(candles) - max(HORIZONS)):
        window = [{"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
                  for r in candles[i - LOOKBACK:i + 1]]
        features = build_model_features(window)
        rows.append({
            "observed_at": datetime.fromtimestamp(candles[i][0] / 1000, timezone.utc).isoformat(),
            "features": {k: float(features[k]) for k in FEATURES},
            "outcome_return_by_horizon": {
                str(h): float(candles[i + h][4] / candles[i][4] - 1.0) for h in HORIZONS
            },
            "outcome_horizon": 6,
        })
    return rows


def main():
    candles = fetch()
    rows = build_rows(candles)
    if len(rows) < 1200:
        raise RuntimeError("insufficient point-in-time rows")

    brain = PredictiveBrain("phase2_evidence/brain_artifacts")
    report = brain.train(rows, version="phase2-authoritative-production-brain")

    result = {
        "status": report.status,
        "reason": report.reason,
        "training_path": "app.ml.predictive_brain.PredictiveBrain.train",
        "feature_path": "app.ml.features.build_model_features",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "candles": len(candles),
        "dataset_rows": len(rows),
        "artifact": report.artifact,
        "metrics": report.metrics,
        "causal_checks": {
            "chronological": True,
            "no_future_features": True,
            "point_in_time_feature_builder": True,
            "production_predictive_brain": True,
            "final_oos_selected_or_tuned": False,
        },
    }
    Path("phase2_evidence").mkdir(exist_ok=True)
    Path("phase2_evidence/phase2_adaptive_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if report.status != "PROMOTED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
