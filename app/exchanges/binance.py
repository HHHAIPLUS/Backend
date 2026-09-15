from __future__ import annotations
import hashlib,hmac,time,uuid,math,os
from urllib.parse import urlencode
import httpx
from app.exchanges.base import ExchangeAdapter
from app.core.config import settings

class BinanceAdapter(ExchangeAdapter):
    name='binance'
    _live_canary_trade_count=0
    def __init__(self, testnet: bool|None=None):
        self.api_key=settings.binance_api_key
        self.secret=settings.binance_api_secret
        self.testnet=settings.binance_testnet if testnet is None else testnet
        self.base=settings.binance_testnet_url if self.testnet else settings.binance_url
    def _signed(self, method, path, params=None):
        if not self.api_key or not self.secret: raise RuntimeError('Binance credentials are not configured')
        params=dict(params or {}); params['timestamp']=int(time.time()*1000); params.setdefault('recvWindow',5000)
        query=urlencode(params,doseq=True); sig=hmac.new(self.secret.encode(),query.encode(),hashlib.sha256).hexdigest(); params['signature']=sig
        return method,path,params,{'X-MBX-APIKEY':self.api_key}
    async def _request(self, method,path,params=None,signed=False):
        headers={}
        if signed: method,path,params,headers=self._signed(method,path,params)
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.request(method,self.base+path,params=params,headers=headers)
            r.raise_for_status(); return r.json()
    async def get_account_status(self):
        data=await self._request('GET','/fapi/v2/account',signed=True)
        return {'exchange':self.name,'testnet':self.testnet,'available_balance':float(data.get('availableBalance',0)),'total_wallet_balance':float(data.get('totalWalletBalance',0)),'raw':data}
    async def get_positions(self): return await self._request('GET','/fapi/v2/positionRisk',signed=True)
    async def get_position_mode(self):
        data = await self._request('GET','/fapi/v1/positionSide/dual',signed=True)
        return 'HEDGE' if bool(data.get('dualSidePosition')) else 'ONE_WAY'
    async def get_symbol_rules(self, symbol):
        data = await self._request('GET','/fapi/v1/exchangeInfo')
        wanted = symbol.upper()
        for item in data.get('symbols', []):
            if item.get('symbol') != wanted: continue
            filters = {f.get('filterType'): f for f in item.get('filters', [])}
            lot = filters.get('MARKET_LOT_SIZE') or filters.get('LOT_SIZE') or {}
            notional = filters.get('NOTIONAL') or filters.get('MIN_NOTIONAL') or {}
            return {'symbol':wanted,'status':item.get('status'),'quantity_min':float(lot.get('minQty',0) or 0),'quantity_max':float(lot.get('maxQty',0) or 0),'quantity_step':float(lot.get('stepSize',0) or 0),'min_notional':float(notional.get('minNotional',0) or 0)}
        raise RuntimeError(f'Binance Futures symbol not found: {wanted}')
    @staticmethod
    def normalize_quantity(quantity, rules):
        step=float(rules.get('quantity_step') or 0); minimum=float(rules.get('quantity_min') or 0); maximum=float(rules.get('quantity_max') or 0)
        if step<=0 or minimum<=0: raise RuntimeError('Binance returned invalid quantity filters')
        value=math.floor(float(quantity)/step)*step
        if maximum>0: value=min(value,maximum)
        if value<minimum: return 0.0
        return float(f'{value:.8f}')
    async def set_leverage(self, symbol, leverage):
        leverage=max(1,min(5,int(leverage)))
        return await self._request('POST','/fapi/v1/leverage',{'symbol':symbol.upper(),'leverage':leverage},signed=True)
    async def place_protection(self, symbol, side, quantity, stop_price, take_profit, position_mode='ONE_WAY'):
        close_side='SELL' if side=='long' else 'BUY'; common={'symbol':symbol.upper(),'side':close_side,'workingType':'MARK_PRICE','priceProtect':'TRUE'}
        if position_mode=='HEDGE': common['positionSide']='LONG' if side=='long' else 'SHORT'; common['quantity']=self._fmt_qty(quantity)
        else: common['closePosition']='true'
        stop=dict(common,type='STOP_MARKET',stopPrice=self._fmt_price(stop_price),newClientOrderId=f'HHHAI-{uuid.uuid4().hex[:20]}')
        stop_result=await self._request('POST','/fapi/v1/order',stop,signed=True)
        if take_profit is None: return {'stop_loss':stop_result,'take_profit':None,'attached':True}
        take=dict(common,type='TAKE_PROFIT_MARKET',stopPrice=self._fmt_price(take_profit),newClientOrderId=f'HHHAI-{uuid.uuid4().hex[:20]}')
        try: take_result=await self._request('POST','/fapi/v1/order',take,signed=True)
        except Exception:
            try:
                if isinstance(stop_result,dict) and stop_result.get('orderId') is not None: await self.cancel_order(symbol,stop_result['orderId'])
            finally:
                close={'symbol':symbol.upper(),'side':close_side,'type':'MARKET','quantity':self._fmt_qty(quantity)}
                if position_mode=='HEDGE': close['positionSide']='LONG' if side=='long' else 'SHORT'
                else: close['reduceOnly']='true'
                try: await self._request('POST','/fapi/v1/order',close,signed=True)
                except Exception: pass
            raise
        return {'stop_loss':stop_result,'take_profit':take_result,'attached':True}
    async def close_position(self,symbol,side,quantity,position_mode='ONE_WAY'):
        close_side='SELL' if side.lower()=='long' else 'BUY'; order={'symbol':symbol.upper(),'side':close_side,'type':'MARKET','quantity':self._fmt_qty(quantity)}
        if position_mode=='HEDGE': order['positionSide']='LONG' if side.lower()=='long' else 'SHORT'
        else: order['reduceOnly']='true'
        return await self._request('POST','/fapi/v1/order',order,signed=True)
    async def cancel_protection_orders(self,symbol):
        orders=await self.get_open_orders(symbol)
        for order in orders or []:
            if order.get('type') in {'STOP_MARKET','TAKE_PROFIT_MARKET','TRAILING_STOP_MARKET','STOP','TAKE_PROFIT'} and str(order.get('clientOrderId') or '').startswith('HHHAI-'):
                try: await self.cancel_order(symbol,order.get('orderId'))
                except Exception: pass
    async def update_dynamic_protection(self,symbol,side,quantity,stop_price,position_mode='ONE_WAY'):
        try: await self.cancel_protection_orders(symbol)
        except Exception: pass
        return await self.place_protection(symbol,side,quantity,stop_price,None,position_mode=position_mode)
    @staticmethod
    def _fmt_price(value): return f'{float(value):.8f}'.rstrip('0').rstrip('.')
    @staticmethod
    def _fmt_qty(value): return f'{float(value):.8f}'.rstrip('0').rstrip('.')
    async def get_open_orders(self,symbol=None): return await self._request('GET','/fapi/v1/openOrders',({'symbol':symbol} if symbol else {}),True)
    async def get_ticker(self,symbol): return await self._request('GET','/fapi/v1/ticker/price',{'symbol':symbol.upper()})
    async def place_order(self,order):
        if str(order.get('type','')).upper()=='MARKET' and order.get('quantity') is not None:
            symbol=str(order.get('symbol','')).upper()
            rules=await self.get_symbol_rules(symbol)
            if rules.get('status')!='TRADING': raise RuntimeError(f'Binance Futures symbol is not trading: {symbol}')
            account=await self.get_account_status()
            available=max(0.0,float(account.get('available_balance') or 0))
            ticker=await self.get_ticker(symbol)
            price=float(ticker.get('price') or 0)
            if available<=0 or price<=0: raise RuntimeError('Binance available margin or market price is unavailable')
            await self.set_leverage(symbol,5)
            max_quantity_by_margin=(available*5.0)/price
            requested=float(order['quantity'])
            quantity=min(requested,max_quantity_by_margin)
            quantity=self.normalize_quantity(quantity,rules)
            if quantity<=0: raise RuntimeError(f'Order quantity is below Binance minimum for {symbol} with current available margin')
            notional=quantity*price
            minimum_notional=float(rules.get('min_notional') or 0)
            if minimum_notional>0 and notional<minimum_notional: raise RuntimeError(f'Order notional {notional:.8f} USDT is below Binance minimum {minimum_notional:.8f} USDT for {symbol}')

            # Temporary Test 10 safety gate. It exists only when explicitly
            # enabled for the live canary and does not affect normal trading.
            live_canary = (not self.testnet and settings.live_trading_enabled and os.getenv('HHHAI_LIVE_CANARY_ENABLED','false').lower()=='true')
            if live_canary:
                max_notional=float(os.getenv('HHHAI_LIVE_CANARY_MAX_NOTIONAL_USD','2.00'))
                max_trades=max(1,int(os.getenv('HHHAI_LIVE_CANARY_MAX_TRADES','1')))
                if self.__class__._live_canary_trade_count >= max_trades:
                    raise RuntimeError(f'Live canary trade limit reached ({max_trades})')
                if not math.isfinite(max_notional) or max_notional<=0:
                    raise RuntimeError('Invalid live canary maximum notional')
                if notional > max_notional + 1e-9:
                    raise RuntimeError(f'Live canary blocked order: notional {notional:.8f} USDT exceeds maximum {max_notional:.8f} USDT')

            order=dict(order); order['quantity']=self._fmt_qty(quantity)
        result = await self._request('POST','/fapi/v1/order',order,True)
        if str(order.get('type','')).upper()=='MARKET' and not self.testnet and settings.live_trading_enabled and os.getenv('HHHAI_LIVE_CANARY_ENABLED','false').lower()=='true':
            self.__class__._live_canary_trade_count += 1
        return result
    async def cancel_order(self,symbol,order_id): return await self._request('DELETE','/fapi/v1/order',{'symbol':symbol,'orderId':order_id},True)
