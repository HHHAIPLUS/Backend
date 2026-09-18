from __future__ import annotations

import asyncio
import logging
import os

from app.market_data.realtime import build_world_intelligence
from app.persistence.repository import record_event
from app.persistence.supabase import store

log = logging.getLogger("hhhai.monitor")


class MarketMonitor:
    def __init__(self):
        self.symbols = [x.strip().upper() for x in os.getenv("HHHAI_WATCH_SYMBOLS", "BTCUSDT").split(",") if x.strip()]
        self.interval = max(5, int(os.getenv("HHHAI_MONITOR_INTERVAL_SECONDS", "15")))
        self.latest = {}
        self.position_reviews = []
        self._stop = asyncio.Event()

    async def run(self):
        exchange = os.getenv("HHHAI_EXECUTION_EXCHANGE", os.getenv("HHHAI_MARKET_EXCHANGE", "binance")).lower()
        log.info("HHHAI observation worker started for %s on %s", self.symbols, exchange)
        while not self._stop.is_set():
            for symbol in self.symbols:
                try:
                    world = await asyncio.to_thread(build_world_intelligence, symbol, exchange)
                    self.latest[symbol] = world.model_dump(mode="json")
                    if store.configured:
                        try:
                            await record_event("market_observation", self.latest[symbol])
                        except Exception:
                            pass
                except Exception as exc:
                    log.warning("Observation failed for %s on %s: %s", symbol, exchange, exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass
        log.info("HHHAI observation worker stopped")

    def stop(self):
        self._stop.set()


monitor = MarketMonitor()
