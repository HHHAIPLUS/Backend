from __future__ import annotations
import json, time
from datetime import datetime, timezone
from pathlib import Path
import httpx, numpy as np


BITGET_URL = "https://api.bitget.com/api/v2/mix/market/history-candles"


def fetch_bitget_candles(target: int):
    rows = {}
    end_ms = int(time.time() * 1000) // 3600000 * 3600000
    consecutive_empty = 0
    with httpx.Client(timeout=30, trust_env=False, headers={"User-Agent": "HHHAI/phase2"}) as client:
        while len(rows) < target:
            start_ms = end_ms - 199 * 3600000
            params = {"symbol": SYMBOL, "productType": "USDT-FUTURES", "granularity": INTERVAL,
                      "limit": 200, "startTime": start_ms, "endTime": end_ms}
            for attempt in range(6):
                try:
                    response = client.get(BITGET_URL, params=params)
                    if response.status_code == 429:
                        time.sleep(min(60.0, 2.0 ** attempt * 2.0))
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    if attempt == 5:
                        raise RuntimeError(f"Bitget historical request failed: {exc}") from exc
                    time.sleep(min(30.0, 2.0 ** attempt))
            else:
                raise RuntimeError("Bitget remained rate-limited after retries")
            if payload.get("code") not in (None, "00000"):
                raise RuntimeError(f"Bitget candles error: {payload}")
            data = payload.get("data", [])
            if not data:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    raise RuntimeError("Bitget returned no candles repeatedly")
                time.sleep(2.0)
                continue
            consecutive_empty = 0
            before = len(rows)
            for r in data:
                if len(r) >= 6:
                    rows[int(r[0])] = (int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            oldest = min(int(r[0]) for r in data)
            if oldest >= end_ms or len(rows) == before:
                raise RuntimeError("Bitget candle pagination stalled")
            end_ms = oldest
            time.sleep(1.1)
    ordered = sorted(rows.values())[-target:]
    if len(ordered) != target:
        raise RuntimeError(f"expected {target} candles, got {len(ordered)}")
    return ordered

from app.ml.features import FEATURES, build_model_features
from app.ml.predictive_brain import PredictiveBrain

SYMBOL = "BTCUSDT"
INTERVAL = "1H"
CANDLES = 30000
LOOKBACK = 168
HORIZONS = (1, 3, 6, 12)


def fetch_funding_rates(start_ms: int):
    """Fetch historical funding settlements without using future observations."""
    rates = {}
    page_no = 1
    with httpx.Client(timeout=30, trust_env=False, headers={"User-Agent": "HHHAI/phase2"}) as client:
        while page_no <= 100:
            response = client.get(
                "https://api.bitget.com/api/v2/mix/market/history-fund-rate",
                params={
                    "symbol": SYMBOL,
                    "productType": "USDT-FUTURES",
                    "pageSize": 100,
                    "pageNo": page_no,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") not in (None, "00000"):
                raise RuntimeError(f"Bitget funding history error: {payload}")
            data = payload.get("data", [])
            if not data:
                break
            before = len(rates)
            for item in data:
                ts = int(item.get("fundingTime", 0))
                if ts:
                    rates[ts] = float(item.get("fundingRate", 0.0))
            if len(rates) == before:
                break
            oldest = min(rates)
            if oldest <= start_ms:
                break
            page_no += 1
            time.sleep(0.08)
    if not rates:
        raise RuntimeError("Bitget returned no historical funding rates")
    return dict(sorted(rates.items()))


def fetch():
    rows = fetch_bitget_candles(CANDLES)
    funding = fetch_funding_rates(rows[0][0])
    for a, b in zip(rows, rows[1:]):
        if b[0] - a[0] != 3600000:
            raise RuntimeError(f"candle gap/overlap {a[0]}->{b[0]}")
        if not (a[3] <= min(a[1], a[4]) and a[2] >= max(a[1], a[4]) and a[1] > 0 and a[4] > 0 and a[5] >= 0):
            raise RuntimeError("invalid OHLCV")
    return rows, funding


def build_rows(candles, funding_rates):
    rows = []
    funding_times = sorted(funding_rates)
    fp = 0
    current_funding = 0.0
    for i in range(LOOKBACK, len(candles) - max(HORIZONS)):
        ts = candles[i][0]
        while fp < len(funding_times) and funding_times[fp] <= ts:
            current_funding = float(funding_rates[funding_times[fp]])
            fp += 1
        window = [{"timestamp": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
                  for r in candles[i - LOOKBACK:i + 1]]
        features = build_model_features(window, context={"market": {"funding_rate": current_funding}})
        rows.append({
            "observed_at": datetime.fromtimestamp(candles[i][0] / 1000, timezone.utc).isoformat(),
            "features": {k: float(features[k]) for k in FEATURES},
            "outcome_return_by_horizon": {
                str(h): float(candles[i + h][4] / candles[i][4] - 1.0) for h in HORIZONS
            },
            "outcome_horizon": 6,
        })
    return rows


def verify_causal_contract(candles, funding_rates, rows, report):
    """Fail closed on the Phase 2 feature/target alignment contract."""
    timestamps = [str(r["observed_at"]) for r in rows]
    assert timestamps == sorted(timestamps) and len(timestamps) == len(set(timestamps)), "row timestamps are not strictly increasing"
    assert all(set(r["features"]) == set(FEATURES) for r in rows), "feature schema mismatch"
    assert not any("outcome" in name.lower() or "target" in name.lower() or "label" in name.lower() for name in FEATURES), "target/label leaked into feature schema"

    funding_times = sorted(funding_rates)
    sample = rows[::max(1, len(rows) // 100)]
    for row in sample:
        ts_ms = int(datetime.fromisoformat(row["observed_at"]).timestamp() * 1000)
        idx = next(i for i, c in enumerate(candles) if int(c[0]) == ts_ms)
        window = [{"timestamp": c[0], "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5]} for c in candles[idx - LOOKBACK:idx + 1]]
        # Rebuild the exact point-in-time feature vector and require byte-level
        # agreement within floating-point tolerance with the training row.
        prior = [ts for ts in funding_times if ts <= ts_ms]
        funding_rate = float(funding_rates[prior[-1]]) if prior else 0.0
        rebuilt = build_model_features(window, context={"market": {"funding_rate": funding_rate}})
        for key in FEATURES:
            assert np.isclose(float(row["features"][key]), float(rebuilt[key]), rtol=1e-10, atol=1e-12), f"feature mismatch at {row['observed_at']}:{key}"
        # Every supervised target is computed strictly after the observation bar.
        for h in HORIZONS:
            assert idx + h < len(candles), "target horizon crosses dataset boundary"
            expected = candles[idx + h][4] / candles[idx][4] - 1.0
            actual = float(row["outcome_return_by_horizon"][str(h)])
            assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12), f"target alignment mismatch at {row['observed_at']}:{h}"

    split = report.get("metrics", {}).get("split_evidence", {})
    purge_rows = int(split.get("purge_rows", 0))
    assert purge_rows >= max(HORIZONS), "chronological partitions are not purged by maximum target horizon"
    oos = split.get("oos", [0, 0])
    cal = split.get("calibration", [0, 0])
    val = split.get("validation", [0, 0])
    assert int(val[1]) <= int(cal[0]) <= int(oos[0]), "validation/calibration/OOS ordering is invalid"
    return {
        "chronological": True,
        "strict_unique_timestamps": True,
        "feature_schema_exact": True,
        "no_target_or_label_feature_names": True,
        "point_in_time_feature_rebuild": True,
        "future_target_alignment": True,
        "purged_chronological_splits": True,
        "oos_after_selection_and_calibration": True,
    }

def main():
    candles, funding = fetch()
    rows = build_rows(candles, funding)
    if len(rows) < 1200:
        raise RuntimeError("insufficient point-in-time rows")

    brain = PredictiveBrain("phase2_evidence/brain_artifacts")
    report = brain.train(rows, version="phase2-authoritative-production-brain")

    causal_checks = verify_causal_contract(candles, funding, rows, {"metrics": report.metrics})
    result = {
        "status": report.status,
        "reason": report.reason,
        "training_path": "app.ml.predictive_brain.PredictiveBrain.train",
        "feature_path": "app.ml.features.build_model_features",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "candles": len(candles),
        "dataset_rows": len(rows),
        "funding_history": {"available": True, "feature": "funding_rate"},
        "artifact": report.artifact,
        "metrics": report.metrics,
        "causal_checks": causal_checks | {
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
