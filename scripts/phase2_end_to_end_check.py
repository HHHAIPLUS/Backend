from __future__ import annotations
import json, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
import httpx, numpy as np

from app.ml.features import FEATURES, build_model_features
from app.ml.predictive_brain import PredictiveBrain, _direction_target

SYMBOL = "BTCUSDT"
INTERVAL = "1H"
CANDLES = 5000
HORIZONS = (1, 3, 6, 12)


def fetch():
    rows = []
    end_ms = int(time.time() * 1000) // 3600000 * 3600000
    with httpx.Client(timeout=30, trust_env=False, headers={"User-Agent": "HHHAI/phase2-e2e"}) as client:
        while len(rows) < CANDLES:
            start_ms = end_ms - 199 * 3600000
            params = {
                "symbol": SYMBOL, "productType": "USDT-FUTURES",
                "granularity": INTERVAL, "limit": 200,
                "startTime": start_ms, "endTime": end_ms,
            }
            payload = client.get(
                "https://api.bitget.com/api/v2/mix/market/history-candles",
                params=params,
            ).json()
            if payload.get("code") not in (None, "00000"):
                raise RuntimeError(payload)
            data = payload.get("data", [])
            if not data:
                raise RuntimeError("Bitget returned no candles")
            rows.extend(
                (int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
                for r in data if len(r) >= 6
            )
            oldest = min(r[0] for r in rows[-200:])
            if oldest >= end_ms:
                raise RuntimeError("Bitget pagination stalled")
            end_ms = oldest
    rows = sorted({r[0]: r for r in rows}.values())[-CANDLES:]
    if len(rows) != CANDLES:
        raise RuntimeError(f"expected {CANDLES}, got {len(rows)}")
    for a, b in zip(rows, rows[1:]):
        if b[0] - a[0] != 3600000:
            raise RuntimeError("candle chronology/gap failure")
    return rows


def build_rows(candles):
    rows = []
    for i in range(168, len(candles) - max(HORIZONS)):
        window = [
            {"timestamp": r[0], "open": r[1], "high": r[2],
             "low": r[3], "close": r[4], "volume": r[5]}
            for r in candles[i - 168:i + 1]
        ]
        features = build_model_features(window)
        if set(FEATURES) - set(features):
            raise RuntimeError("canonical feature vector is incomplete")
        outcomes = {
            str(h): candles[i + h][4] / candles[i][4] - 1.0
            for h in HORIZONS
        }
        rows.append({
            "observed_at": datetime.fromtimestamp(
                candles[i][0] / 1000, timezone.utc
            ).isoformat(),
            "features": {k: float(features[k]) for k in FEATURES},
            "outcome_return_by_horizon": outcomes,
            "outcome_horizon": 6,
            "label": int(_direction_target(np.asarray([outcomes["6"]]))[0]),
        })
    return rows


def main():
    candles = fetch()
    rows = build_rows(candles)
    if len(rows) < 1200:
        raise RuntimeError("insufficient point-in-time rows")

    # Structural checks: feature state at t must be reproducible from candles <= t.
    probe = 300
    w1 = [
        {"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3],
         "close": r[4], "volume": r[5]}
        for r in candles[probe - 168:probe + 1]
    ]
    w2 = w1 + [{
        "timestamp": candles[probe + 1][0], "open": candles[probe + 1][1],
        "high": candles[probe + 1][2], "low": candles[probe + 1][3],
        "close": candles[probe + 1][4], "volume": candles[probe + 1][5]
    }]
    f1 = build_model_features(w1)
    f2 = build_model_features(w2)
    if not all(np.isfinite(f1.get(k, 0.0)) for k in FEATURES):
        raise RuntimeError("non-finite canonical features")
    if all(f1.get(k) == f2.get(k) for k in FEATURES):
        raise RuntimeError("feature path did not react to the next candle")

    with tempfile.TemporaryDirectory() as d:
        brain = PredictiveBrain(d)
        report = brain.train(rows, version="phase2-e2e-diagnostic")
        result = {
            "status": "PASS" if report.status in {"PROMOTED", "REJECTED"} else "FAIL",
            "training_path": "app.ml.predictive_brain.PredictiveBrain.train",
            "feature_path": "app.ml.features.build_model_features",
            "candles": len(candles),
            "rows": len(rows),
            "brain_status": report.status,
            "reason": report.reason,
            "metrics": report.metrics,
            "artifact_created": Path(report.artifact).exists() if report.artifact else False,
        }
        if report.status == "PROMOTED":
            prediction = brain.predict(rows[-1]["features"])
            result["prediction_smoke"] = {
                "trained": prediction["trained"],
                "decision": prediction["decision"],
                "finite_expected_return": bool(np.isfinite(prediction["expected_return"])),
                "finite_uncertainty": bool(np.isfinite(prediction["uncertainty"])),
            }
        print(json.dumps(result, indent=2))
        Path("phase2_evidence").mkdir(exist_ok=True)
        Path("phase2_evidence/end_to_end_check.json").write_text(
            json.dumps(result, indent=2)
        )
        if result["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
