from __future__ import annotations

import json
import os
from collections import Counter

from app.ml.bootstrap import audit_historical_klines, build_dataset, fetch_historical_klines
from app.ml.predictive_brain import predictive_brain


def main() -> int:
    symbol = os.getenv("PHASE2_SYMBOL", "BTCUSDT").upper()
    interval = os.getenv("PHASE2_INTERVAL", "1h")
    candles = int(os.getenv("PHASE2_CANDLES", "10000"))
    raw, provider = fetch_historical_klines(symbol, interval, candles)
    if provider != "bitget":
        raise RuntimeError(f"Phase 2 requires Bitget futures history; received {provider}")
    audit = audit_historical_klines(raw, interval)
    rows = build_dataset(
        raw,
        horizon=6,
        threshold=float(os.getenv("HHHAI_BRAIN_LABEL_THRESHOLD", "0.0015")),
        symbol=symbol,
        interval=interval,
        provider=provider,
    )
    labels = Counter(int(row["label"]) for row in rows)
    report = predictive_brain.train(rows, version=f"phase2-{symbol}-{interval}")
    payload = {
        "symbol": symbol,
        "interval": interval,
        "provider": provider,
        "candles": len(raw),
        "rows": len(rows),
        "candle_audit": audit,
        "label_distribution": dict(sorted(labels.items())),
        "status": report.status,
        "version": report.version,
        "reason": report.reason,
        "metrics": report.metrics,
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if report.status == "PROMOTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
