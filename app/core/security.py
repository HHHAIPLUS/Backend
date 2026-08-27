from __future__ import annotations
import os, secrets

def constant_time_equal(provided: str, expected: str) -> bool:
    return bool(provided and expected and secrets.compare_digest(provided, expected))

def live_trading_enabled() -> bool:
    return os.getenv('LIVE_TRADING_ENABLED', 'false').lower() == 'true'
