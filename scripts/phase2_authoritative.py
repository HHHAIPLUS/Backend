from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.model_persistence import hydrate_model, persist_brain
from app.ml.phase2_authority import authoritative_config, fetch_authoritative_dataset
from app.ml.predictive_brain import predictive_brain

CONFIG = authoritative_config()
VERSION = os.getenv("PHASE2_VERSION", "phase2-authoritative")


def main() -> int:
    rows, meta = fetch_authoritative_dataset()
    report = predictive_brain.train(rows, VERSION)
    persistence = {"persisted": False, "hydrated": False}

    if report.status == "PROMOTED":
        persisted = asyncio.run(persist_brain(report.metrics))
        persistence["persisted"] = bool(persisted)
        if not persisted:
            report.status = "REJECTED"
            report.reason = "OOS gates passed, but the promoted artifact could not be persisted to Supabase."
        else:
            hydrated = asyncio.run(hydrate_model(report.version))
            persistence["hydrated"] = bool(hydrated)
            if not hydrated:
                report.status = "REJECTED"
                report.reason = "Artifact was persisted, but production hydration from Supabase failed."
            else:
                manifest = predictive_brain.manifest() or {}
                bundle = predictive_brain.bundle or {}
                if (
                    manifest.get("version") != report.version
                    or manifest.get("feature_hash") != bundle.get("feature_hash")
                    or manifest.get("schema_version") != bundle.get("schema_version")
                ):
                    report.status = "REJECTED"
                    report.reason = "Persisted artifact hydration did not match the trained promoted artifact."

    payload = {
        "status": report.status,
        "version": report.version,
        "reason": report.reason,
        "metrics": report.metrics,
        "artifact": report.artifact,
        "persistence": persistence,
        "provider": meta["provider"],
        "symbol": meta["symbol"],
        "interval": meta["interval"],
        "horizon": meta["horizon"],
        "label_threshold": meta["label_threshold"],
        "candle_count": meta["raw_candles"],
        "dataset_rows": meta["training_rows"],
        "oos_start": os.getenv("PHASE2_OOS_START"),
        "oos_end": os.getenv("PHASE2_OOS_END"),
        "authoritative_config": CONFIG,
    }
    Path("phase2_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str)
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if report.status == "PROMOTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
