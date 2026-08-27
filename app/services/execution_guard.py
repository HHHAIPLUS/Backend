from app.core.config import settings
from app.services.safety import kill_switch

class ExecutionGuard:
    def assert_allowed(self, *, testnet: bool=False):
        if kill_switch.enabled:
            raise RuntimeError(f'Execution blocked by kill switch: {kill_switch.reason}')
        if testnet:
            if not settings.testnet_trading_enabled:
                raise RuntimeError('Testnet execution is disabled.')
            return
        if not settings.live_trading_enabled:
            raise RuntimeError('Live trading is disabled until controlled validation is complete.')
