from __future__ import annotations
import hashlib,hmac,time,uuid,math,os
from urllib.parse import urlencode
import httpx
from app.exchanges.base import ExchangeAdapter
from app.core.config import settings

class BinanceAdapter(ExchangeAdapter):
    name='binance'; _live_canary_trade_count=0; _private_blocked_until=0.0; _private_block_reason=''; _private_block_retry_after=0.0
    _account_cache={}; _account_cache_at={}; _positions_cache={}; _positions_cache_at={}; _position_mode_cache={}; _position_mode_cache_at={}; _symbol_rules_cache={}; _symbol_rules_cache_at={}; _ticker_cache={}; _ticker_cache_at={}
    def __init__(self,testnet:bool|None=None): self.api_key=settings.binance_api_key; self.secret=settings.binance_api_secret; self.testnet=settings.binance_testnet if testnet is None else testnet; self.base=settings.binance_testnet_url if self.testnet else settings.binance_url
    def _signed(self,method,path,params=None):
        if not self.api_key or not self.secret: raise RuntimeError('Binance credentials are not configured')
        params=dict(params or {}); params['timestamp']=int(time.time()*1000); params.setdefault('recvWindow',5000); query=urlencode(params,doseq=True); params['signature']=hmac.new(self.secret.encode(),query.encode(),hashlib.sha256).hexdigest(); return method,path,params,{'X-MBX-APIKEY':self.api_key}
    @classmethod
    def _private_health_error(cls):
        remaining=max(0.0,cls._private_blocked_until-time.time()); return f'Binance private API temporarily blocked for {int(math.ceil(remaining))}s: {cls._private_block_reason}' if remaining>0 else None
    @classmethod
    def private_execution_healthy(cls): return cls._private_health_error() is None
    @classmethod
    def _block_private(cls,response):
        retry=response.headers.get('Retry-After')
        try: wait=float(retry) if retry is not None else 900.0
        except Exception: wait=900.0
        wait=max(60.0,min(wait,259200.0)); cls._private_blocked_until=max(cls._private_blocked_until,time.time()+wait); cls._private_block_retry_after=wait; cls._private_block_reason=f'HTTP {response.status_code}; Binance requires backoff before private requests'
    @classmethod
    def _clear_shared_cache(cls): cls._account_cache.clear(); cls._account_cache_at.clear(); cls._positions_cache.clear(); cls._positions_cache_at.clear(); cls._position_mode_cache.clear(); cls._position_mode_cache_at.clear()
    async def _request(self,method,path,params=None,signed=False):
        if signed:
            health=self._private_health_error()
            if health: raise RuntimeError(health)
        headers={}
        if signed: method,path,params,headers=self._signed(method,path,params)
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.request(method,self.base+path,params=params,headers=headers)
            if r.status_code in (418,429) and signed: self._block_private(r); raise RuntimeError(f'Binance private API rate-limit/banned response HTTP {r.status_code}; execution blocked until backoff expires')
            r.raise_for_status(); return r.json()
    async def get_account_status(self):
        key=self.testnet; now=time.time()
        if key in self.__class__._account_cache and now-self.__class__._account_cache_at.get(key,0)<10: return self.__class__._account_cache[key]
        data=await self._request('GET','/fapi/v2/account',signed=True); result={'exchange':self.name,'testnet':self.testnet,'available_balance':float(data.get('availableBalance',0)),'total_wallet_balance':float(data.get('totalWalletBalance',0)),'raw':data}; self.__class__._account_cache[key]=result; self.__class__._account_cache_at[key]=now; return result
    async def get_positions(self):
        key=self.testnet; now=time.time()
        if key in self.__class__._positions_cache and now-self.__class__._positions_cache_at.get(key,0)<10: return self.__class__._positions_cache[key]
        data=await self._request('GET','/fapi/v2/positionRisk',signed=True); self.__class__._positions_cache[key]=data; self.__class__._positions_cache_at[key]=now; return data
    async def get_position_mode(self):
        key=self.testnet; now=time.time()
        if key in self.__class__._position_mode_cache and now-self.__class__._position_mode_cache_at.get(key,0)<60: return self.__class__._position_mode_cache[key]
        data=await self._request('GET','/fapi/v1/positionSide/dual',signed=True); result='HEDGE' if bool(data.get('dualSidePosition')) else 'ONE_WAY'; self.__class__._position_mode_cache[key]=result; self.__class__._position_mode_cache_at[key]=now; return result
    async def get_symbol_rules(self,symbol):
        key=(self.testnet,symbol.upper()); now=time.time()
        if key in self.__class__._symbol_rules_cache and now-self.__class__._symbol_rules_cache_at.get(key,0)<3600: return self.__class__._symbol_rules_cache[key]
        data=await self._request('GET','/fapi/v1/exchangeInfo'); wanted=symbol.upper()
        for item in data.get('symbols',[]):
            if item.get('symbol')!=wanted: continue
            filters={f.get('filterType'):f for f in item.get('filters',[])}; lot=filters.get('MARKET_LOT_SIZE') or filters.get('LOT_SIZE') or {}; notional=filters.get('NOTIONAL') or filters.get('MIN_NOTIONAL') or {}; result={'symbol':wanted,'status':item.get('status'),'quantity_min':float(lot.get('minQty',0) or 0),'quantity_max':float(lot.get('maxQty',0) or 0),'quantity_step':float(lot.get('stepSize',0) or 0),'min_notional':float(notional.get('minNotional',0) or 0)}; self.__class__._symbol_rules_cache[key]=result; self.__class__._symbol_rules_cache_at[key]=now; return result
        raise RuntimeError(f'Binance Futures symbol not found: {wanted}')
    @staticmethod
    def normalize_quantity(quantity,rules):
        step=float(rules.get('quantity_step') or 0); minimum=float(rules.get('quantity_min') or 0); maximum=float(rules.get('quantity_max') or 0)
        if step<=0 or minimum<=0: raise RuntimeError('Binance returned invalid quantity filters')
        value=math.floor(float(quantity)/step)*step
        if maximum>0: value=min(value,maximum)
        if value<minimum: return 0.0
        return float(f'{value:.8f}')
    async def set_leverage(self,symbol,leverage): return await self._request('POST','/fapi/v1/leverage',{'symbol':symbol.upper(),'leverage':max(1,min(5,int(leverage)))},signed=True)
    async def place_algo_protection(self,symbol,side,quantity,trigger_price,order_type,position_mode='ONE_WAY'):
        close_side='SELL' if side.lower()=='long' else 'BUY'; params={'algoType':'CONDITIONAL','symbol':symbol.upper(),'side':close_side,'type':order_type,'triggerPrice':self._fmt_price(trigger_price),'workingType':'MARK_PRICE','priceProtect':'TRUE','newOrderRespType':'RESULT','clientAlgoId':f'HHHAI-{uuid.uuid4().hex[:20]}' }
        if position_mode=='HEDGE': params['positionSide']='LONG' if side.lower()=='long' else 'SHORT'; params['quantity']=self._fmt_qty(quantity)
        else: params['positionSide']='BOTH'; params['closePosition']='true'
        return await self._request('POST','/fapi/v1/algoOrder',params,signed=True)
    async def place_protection(self,symbol,side,quantity,stop_price,take_profit,position_mode='ONE_WAY'):
        stop_result=await self.place_algo_protection(symbol,side,quantity,stop_price,'STOP_MARKET',position_mode)
        if take_profit is None: return {'stop_loss':stop_result,'take_profit':None,'attached':True}
        try: take_result=await self.place_algo_protection(symbol,side,quantity,take_profit,'TAKE_PROFIT_MARKET',position_mode)
        except Exception:
            try: await self.cancel_algo_order(symbol,stop_result.get('algoId') if isinstance(stop_result,dict) else None)
            finally:
                try: await self.close_position(symbol,side,quantity,position_mode)
                except Exception: pass
            raise
        return {'stop_loss':stop_result,'take_profit':take_result,'attached':True}
    async def close_position(self,symbol,side,quantity,position_mode='ONE_WAY'):
        close_side='SELL' if side.lower()=='long' else 'BUY'; order={'symbol':symbol.upper(),'side':close_side,'type':'MARKET','quantity':self._fmt_qty(quantity)}
        if position_mode=='HEDGE': order['positionSide']='LONG' if side.lower()=='long' else 'SHORT'
        else: order['reduceOnly']='true'
        result=await self._request('POST','/fapi/v1/order',order,True); self.__class__._clear_shared_cache(); return result
    async def get_open_algo_orders(self,symbol=None): return await self._request('GET','/fapi/v1/openAlgoOrders',({'symbol':symbol.upper()} if symbol else {}),True)
    async def cancel_algo_order(self,symbol,algo_id=None,client_algo_id=None):
        params={'symbol':symbol.upper()};
        if algo_id is not None: params['algoId']=algo_id
        elif client_algo_id: params['clientAlgoId']=client_algo_id
        else: raise ValueError('algo_id or client_algo_id is required')
        return await self._request('DELETE','/fapi/v1/algoOrder',params,True)
    async def cancel_protection_orders(self,symbol):
        orders=await self.get_open_algo_orders(symbol)
        for order in orders or []:
            if str(order.get('clientAlgoId') or '').startswith('HHHAI-') and str(order.get('orderType') or order.get('type') or '').upper() in {'STOP_MARKET','TAKE_PROFIT_MARKET','STOP','TAKE_PROFIT','TRAILING_STOP_MARKET'}:
                try: await self.cancel_algo_order(symbol,order.get('algoId'),order.get('clientAlgoId'))
                except Exception: pass
    async def update_dynamic_protection(self,symbol,side,quantity,stop_price,position_mode='ONE_WAY',take_profit=None):
        await self.cancel_protection_orders(symbol)
        return await self.place_protection(symbol,side,quantity,stop_price,take_profit,position_mode=position_mode)
    @staticmethod
    def _fmt_price(value): return f'{float(value):.8f}'.rstrip('0').rstrip('.')
    @staticmethod
    def _fmt_qty(value): return f'{float(value):.8f}'.rstrip('0').rstrip('.')
    async def get_open_orders(self,symbol=None): return await self._request('GET','/fapi/v1/openOrders',({'symbol':symbol} if symbol else {}),True)
    async def get_ticker(self,symbol):
        key=(self.testnet,symbol.upper()); now=time.time()
        if key in self.__class__._ticker_cache and now-self.__class__._ticker_cache_at.get(key,0)<1: return self.__class__._ticker_cache[key]
        result=await self._request('GET','/fapi/v1/ticker/price',{'symbol':symbol.upper()}); self.__class__._ticker_cache[key]=result; self.__class__._ticker_cache_at[key]=now; return result
    async def place_order(self,order):
        if str(order.get('type','')).upper()=='MARKET' and order.get('quantity') is not None:
            symbol=str(order.get('symbol','')).upper(); rules=await self.get_symbol_rules(symbol)
            if rules.get('status')!='TRADING': raise RuntimeError(f'Binance Futures symbol is not trading: {symbol}')
            account=await self.get_account_status(); available=max(0.0,float(account.get('available_balance') or 0)); ticker=await self.get_ticker(symbol); price=float(ticker.get('price') or 0)
            if available<=0 or price<=0: raise RuntimeError('Binance available margin or market price is unavailable')
            await self.set_leverage(symbol,5); max_quantity_by_margin=(available*5.0)/price; quantity=self.normalize_quantity(min(float(order['quantity']),max_quantity_by_margin),rules)
            if quantity<=0: raise RuntimeError(f'Order quantity is below Binance minimum for {symbol} with current available margin')
            notional=quantity*price; minimum_notional=float(rules.get('min_notional') or 0)
            if minimum_notional>0 and notional<minimum_notional: raise RuntimeError(f'Order notional {notional:.8f} USDT is below Binance minimum {minimum_notional:.8f} USDT for {symbol}')
            live_canary=(not self.testnet and settings.live_trading_enabled and os.getenv('HHHAI_LIVE_CANARY_ENABLED','false').lower()=='true')
            if live_canary:
                max_notional=float(os.getenv('HHHAI_LIVE_CANARY_MAX_NOTIONAL_USD','2.00')); max_trades=max(1,int(os.getenv('HHHAI_LIVE_CANARY_MAX_TRADES','1')))
                if self.__class__._live_canary_trade_count>=max_trades: raise RuntimeError(f'Live canary trade limit reached ({max_trades})')
                if not math.isfinite(max_notional) or max_notional<=0: raise RuntimeError('Invalid live canary maximum notional')
                if notional>max_notional+1e-9: raise RuntimeError(f'Live canary blocked order: notional {notional:.8f} USDT exceeds maximum {max_notional:.8f} USDT')
            order=dict(order); order['quantity']=self._fmt_qty(quantity)
        result=await self._request('POST','/fapi/v1/order',order,True); self.__class__._clear_shared_cache()
        if str(order.get('type','')).upper()=='MARKET' and not self.testnet and settings.live_trading_enabled and os.getenv('HHHAI_LIVE_CANARY_ENABLED','false').lower()=='true': self.__class__._live_canary_trade_count+=1
        return result
    async def cancel_order(self,symbol,order_id): return await self._request('DELETE','/fapi/v1/order',{'symbol':symbol,'orderId':order_id},True)
