from __future__ import annotations
from typing import Iterable
from app.market_data.models import Candle

def build_basic_features(candles: Iterable[Candle], context: dict[str,float]|None=None) -> dict[str,float]:
    rows=list(candles); context=context or {}
    if not rows:return {}
    last=rows[-1]; previous=rows[-2] if len(rows)>=2 else last
    returns=[(r.close/rows[i-1].close-1) if i and rows[i-1].close else 0 for i,r in enumerate(rows)]
    vol_change=(last.volume/previous.volume-1) if previous.volume else 0
    return {
      'close':last.close,'volume':last.volume,'return_1':(last.close/previous.close-1) if previous.close else 0,
      'range_pct':((last.high-last.low)/last.close) if last.close else 0,'volume_change':vol_change,
      'volatility_proxy':float(context.get('volatility_proxy',0)),'order_book_imbalance':float(context.get('order_book_imbalance',0)),
      'funding_rate':float(context.get('funding_rate',0)),'open_interest_change':float(context.get('open_interest_change',0)),
      'news_risk':float(context.get('news_risk',0)),'news_sentiment':float(context.get('news_sentiment',0)),
      'trend_strength':float(context.get('trend_strength',0)),'momentum':float(context.get('momentum',returns[-1] if returns else 0)),
      'liquidity_stress':float(context.get('liquidity_stress',0))
    }
