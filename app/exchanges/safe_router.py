from app.services.execution_guard import ExecutionGuard
from app.core.config import settings
class SafeExchangeRouter:
    def __init__(self, adapters, guard=None): self.adapters=adapters; self.guard=guard or ExecutionGuard()
    async def place_order(self, exchange, order, testnet=False):
        self.guard.assert_allowed(testnet=testnet)
        if not testnet and not settings.live_trading_enabled: raise RuntimeError('Live execution is disabled')
        adapter=self.adapters.get(exchange)
        if adapter is None: raise ValueError(f'Unsupported exchange: {exchange}')
        return await adapter.place_order(order)
