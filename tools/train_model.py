from __future__ import annotations

import argparse
import csv
import json

from app.ml.predictive import FEATURES
from app.ml.predictive_brain import predictive_brain


def load(path: str, horizon: int) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            features = {k: float(raw.get(k, 0) or 0) for k in FEATURES}
            outcome_return = float(raw.get("outcome_return", 0) or 0)
            label = int(raw.get("label", 0) or 0)
            observed_at = raw.get("observed_at", "")
            rows.append({
                "observed_at": observed_at,
                "features": features,
                "label": label,
                "outcome_return": outcome_return,
                "outcome_horizon": horizon,
                "outcome_return_by_horizon": {str(horizon): outcome_return},
                "barrier_return_by_horizon": {str(horizon): outcome_return},
                "symbol": raw.get("symbol", ""),
                "interval": raw.get("interval", ""),
                "data_source": raw.get("data_source", "csv"),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Train HHHAI's single predictive-brain pipeline.")
    ap.add_argument("csv")
    ap.add_argument("--version", required=True)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--artifact-dir", default="artifacts")
    args = ap.parse_args()

    # The old script independently trained Logistic Regression and applied an
    # ad-hoc confidence/return rule. That produced a second training system
    # whose result could disagree with the production PredictiveBrain. Keep
    # this CLI as a data-import convenience, but use exactly the same
    # chronological selection/calibration/OOS promotion path as production.
    rows = load(args.csv, args.horizon)
    report = predictive_brain.train(rows, version=args.version)

    output = {
        "status": report.status,
        "version": report.version,
        "reason": report.reason,
        "metrics": report.metrics,
        "artifact": report.artifact,
        "rows": len(rows),
    }
    print(json.dumps(output, indent=2, default=str))
    if report.status != "PROMOTED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
