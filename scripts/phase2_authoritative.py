from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# When executing a script directly, Python puts scripts/ on sys.path rather
# than the repository root. Add the root explicitly so package imports work
# identically on GitHub Actions, Render and local execution.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.bootstrap import fetch_historical_klines, build_dataset, audit_historical_klines
from app.ml.predictive_brain import predictive_brain

SYMBOL = os.getenv("HHHAI_BRAIN_BOOTSTRAP_SYMBOLS", "BTCUSDT").split(",")[0].strip().upper()
INTERVAL = os.getenv("HHHAI_BRAIN_BOOTSTRAP_INTERVAL", "1h").strip()
LIMIT = max(5000, min(30000, int(os.getenv("HHHAI_BRAIN_BOOTSTRAP_CANDLES", "30000"))))
THRESHOLD = float(os.getenv("HHHAI_BRAIN_LABEL_THRESHOLD", "0.0015"))

def main() -> int:
    raw, provider = fetch_historical_klines(SYMBOL, INTERVAL, LIMIT)
    if provider != "bitget":
        raise RuntimeError(f"Authoritative Phase 2 requires Bitget history; got {provider}")
    audit = audit_historical_klines(raw, INTERVAL)
    rows = build_dataset(raw, horizon=6, threshold=THRESHOLD, symbol=SYMBOL, interval=INTERVAL, provider=provider)
    report = predictive_brain.train(rows, "phase2-authoritative-2025-oos")
    payload = {
        "status": report.status,
        "version": report.version,
        "reason": report.reason,
        "metrics": report.metrics,
        "artifact": report.artifact,
        "provider": provider,
        "candle_count": len(raw),
        "dataset_rows": len(rows),
        "candle_audit": audit,
    }
    Path("phase2_report.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if report.status == "PROMOTED" else 2

if __name__ == "__main__":
    raise SystemExit(main())
