from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any


class MarketDataCache:
    """Small disk-backed TTL cache for reproducible historical enrichment.

    Keys are deterministic and payloads are JSON-only so cached observations can
    be inspected, replayed and invalidated without coupling the feature layer
    to a database implementation.
    """

    def __init__(self, root: str | Path = ".cache/market_data", ttl_seconds: int = 300):
        self.root = Path(root)
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._lock = Lock()

    def _path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.root / f"{safe}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        with self._lock:
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text())
                if time.time() - float(payload["stored_at"]) > self.ttl_seconds:
                    return None
                return payload["value"]
            except (OSError, ValueError, KeyError, TypeError):
                return None

    def set(self, key: str, value: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        payload = {"stored_at": time.time(), "value": value}
        with self._lock:
            temporary.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True))
            temporary.replace(path)

    def invalidate(self, key: str) -> None:
        with self._lock:
            try:
                self._path(key).unlink()
            except FileNotFoundError:
                pass

    def clear(self) -> None:
        with self._lock:
            if not self.root.exists():
                return
            for path in self.root.glob("*.json"):
                path.unlink(missing_ok=True)
