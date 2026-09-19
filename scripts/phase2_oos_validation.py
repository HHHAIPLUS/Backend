from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from app.ml.bootstrap import audit_historical_klines, build_dataset, fetch_historical_klines
from app.ml.predictive_brain import predictive_brain


def main() -> int:
    symbol = os.getenv("PHASE2_SYMBOL", "BTCUSDT").upper()
    interval = os.getenv("PHASE2_INTERVAL", "1h")
    candles = int(os.getenv("PHASE2_CANDLES", "30000"))
    threshold = float(os.getenv("HHHAI_BRAIN_LABEL_THRESHOLD", "0.0015"))

    raw, provider = fetch_historical_klines(symbol, interval, candles)
    if provider != "bitget":
        raise RuntimeError(f"Phase 2 requires Bitget futures history; received {provider}")

    audit = audit_historical_klines(raw, interval)
    rows = build_dataset(
        raw,
        horizon=6,
        threshold=threshold,
        symbol=symbol,
        interval=interval,
        provider=provider,
    )
    labels = Counter(int(row["label"]) for row in rows)
    report = predictive_brain.train(rows, version=f"phase2-{symbol}-{interval}")

    payload = {
        "phase": 2,
        "status": report.status,
        "version": report.version,
        "symbol": symbol,
        "interval": interval,
        "provider": provider,
        "requested_candles": candles,
        "candles": len(raw),
        "rows": len(rows),
        "threshold": threshold,
        "candle_audit": audit,
        "label_distribution": dict(sorted(labels.items())),
        "reason": report.reason,
        "metrics": report.metrics,
    }

    out = Path(os.getenv("PHASE2_REPORT_PATH", "artifacts/phase2_oos_report.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))

    # A Phase 2 validation command is successful only when the model itself
    # reports PROMOTED after every fixed untouched-OOS gate has passed.
    return 0 if report.status == "PROMOTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
