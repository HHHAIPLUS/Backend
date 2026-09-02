from __future__ import annotations

from types import MethodType

from app.persistence.repository import load_position_states


def install_stage6_hydration(trader):
    if getattr(trader, "_stage6_hydration_installed", False):
        return
    original = trader._hydrate_position_state

    async def hydrate(self):
        await original()
        try:
            rows = await load_position_states()
            for row in rows or []:
                key = f"{row.get('exchange')}:{row.get('symbol')}:{row.get('side')}"
                payload = row.get("payload") or {}
                if key and payload:
                    self._stage6_state[key] = payload
                    if payload.get("peak_return") is not None:
                        self.position_peaks[key] = float(payload["peak_return"])
        except Exception as exc:
            self.stage6_last_error = f"hydration: {type(exc).__name__}: {exc}"

    trader._hydrate_position_state = MethodType(hydrate, trader)
    trader._stage6_hydration_installed = True
