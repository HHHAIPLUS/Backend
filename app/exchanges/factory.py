from app.exchanges.binance import BinanceAdapter
from app.exchanges.bitget import BitgetAdapter

def adapters(testnet: bool | None = None):
    return {
        "binance": BinanceAdapter(testnet=testnet),
        "bitget": BitgetAdapter(testnet=testnet),
    }
