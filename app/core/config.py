from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    app_name:str='HHHAI'; app_env:str='development'; log_level:str='INFO'
    database_url:str=''; supabase_url:str=''; supabase_service_role_key:str=''
    live_trading_enabled:bool=False; testnet_trading_enabled:bool=False
    cors_origins:str='http://localhost:5173'
    binance_api_key:str=''; binance_api_secret:str=''; binance_testnet:bool=True
    binance_url:str='https://fapi.binance.com'; binance_testnet_url:str='https://testnet.binancefuture.com'
    bitget_api_key:str=''; bitget_api_secret:str=''; bitget_passphrase:str=''; bitget_testnet:bool=True
    bitget_url:str='https://api.bitget.com'; bitget_testnet_url:str='https://api.bitget.com'
    model_artifact_dir:str='artifacts'; min_model_trades:int=500
    model_min_accuracy:float=0.52; model_min_balanced_accuracy:float=0.50; model_min_average_return:float=0.0
    hhhai_autotrading_enabled:bool=False; hhhai_trading_mode:str='paper'; hhhai_trade_symbols:str='BTCUSDT'
    hhhai_execution_exchange:str='binance'; hhhai_market_exchange:str='binance'; hhhai_trading_interval_seconds:int=30
    hhhai_paper_equity:float=10000.0; hhhai_min_trade_confidence:float=0.60; hhhai_max_spread_bps:float=12.0
    hhhai_risk_per_trade_pct:float=0.5; hhhai_min_reward_rr:float=1.8
    model_config=SettingsConfigDict(env_file='.env',env_file_encoding='utf-8',extra='ignore')
settings=Settings()
