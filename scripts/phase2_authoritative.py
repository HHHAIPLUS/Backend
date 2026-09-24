from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.phase2_authority import authoritative_config, fetch_authoritative_dataset
from app.ml.predictive_brain import predictive_brain

CONFIG = authoritative_config()
def main():
    rows, meta = fetch_authoritative_dataset()
    report = predictive_brain.train(rows, CONFIG.get("version", "phase2-authoritative"))
    payload = {
        "phase": 2,
        "status": report.status,
        "version": report.version,
        "reason": report.reason,
        "artifact": report.artifact,
        "metrics": report.metrics,
        "data": meta,
        "authoritative_config": CONFIG,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if report.status != "PROMOTED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
