"""Runtime compatibility patches applied by Python's site initialization."""

# Binance USDⓈ-M Futures combined market streams use /stream, not
# /market/stream.  Keep this tiny compatibility patch isolated so the
# existing market-data implementation can be corrected without changing
# trading logic.
try:
    from app.market_data.realtime import BinanceWebSocketFeed

    BinanceWebSocketFeed.websocket_base = "wss://fstream.binance.com/stream"
except Exception:
    # Do not prevent the application from starting if an optional import
    # changes in a future release. The normal application code will report
    # its own market-data health state.
    pass
