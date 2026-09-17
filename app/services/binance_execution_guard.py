from __future__ import annotations

from types import MethodType

from app.exchanges.binance import BinanceAdapter


def install_binance_execution_guard(trader) -> None:
    """Make Binance private API health a hard live/testnet execution gate."""
    cls = trader.__class__
    if getattr(cls, "_hhhai_binance_execution_guard_installed", False):
        return
    original = cls._execution_gate

    def guarded(self):
        allowed, reason = original(self)
        if not allowed:
            return allowed, reason
        exchange = self._exchange_for_market()
        if exchange == "binance" and not BinanceAdapter.private_execution_healthy():
            return False, BinanceAdapter._private_health_error() or "Binance private API is unavailable"
        return allowed, reason

    cls._execution_gate = guarded
    cls._hhhai_binance_execution_guard_installed = True
