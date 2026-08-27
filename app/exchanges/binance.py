from __future__ import annotations
import hashlib,hmac,time,uuid
from urllib.parse import urlencode
import httpx
from app.exchanges.base import ExchangeAdapter
from app.core.config import settings

class BinanceAdapter(ExchangeAdapter):
    name='binance'
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
    async def place_protection(self, symbol, side, quantity, stop_price, take_profit, position_mode='ONE_WAY'):
        close_side = 'SELL' if side == 'long' else 'BUY'
        common = {'symbol': symbol.upper(), 'side': close_side, 'workingType': 'MARK_PRICE', 'priceProtect': 'TRUE'}
        if position_mode == 'HEDGE':
            common['positionSide'] = 'LONG' if side == 'long' else 'SHORT'
            common['quantity'] = self._fmt_qty(quantity)
        else:
            common['closePosition'] = 'true'
        stop = dict(common, type='STOP_MARKET', stopPrice=self._fmt_price(stop_price), newClientOrderId=f'HHHAI-{uuid.uuid4().hex[:20]}')
        stop_result = await self._request('POST','/fapi/v1/order',stop,signed=True)
        if take_profit is None:
            return {'stop_loss': stop_result, 'take_profit': None, 'attached': True}
        take = dict(common, type='TAKE_PROFIT_MARKET', stopPrice=self._fmt_price(take_profit), newClientOrderId=f'HHHAI-{uuid.uuid4().hex[:20]}')
        try:
            take_result = await self._request('POST','/fapi/v1/order',take,signed=True)
        except Exception:
            try:
                if isinstance(stop_result, dict) and stop_result.get('orderId') is not None:
                    await self.cancel_order(symbol, stop_result['orderId'])
            finally:
                close = {'symbol': symbol.upper(), 'side': close_side, 'type': 'MARKET'}
                if position_mode == 'HEDGE':
                    close['positionSide'] = 'LONG' if side == 'long' else 'SHORT'
                    close['quantity'] = self._fmt_qty(quantity)
                else:
                    close['quantity'] = self._fmt_qty(quantity)
                    close['reduceOnly'] = 'true'
                try:
                    await self._request('POST','/fapi/v1/order',close,signed=True)
                except Exception:
                    pass
            raise
        return {'stop_loss': stop_result, 'take_profit': take_result, 'attached': True}

    async def close_position(self, symbol, side, quantity, position_mode="ONE_WAY"):
        close_side = "SELL" if side.lower() == "long" else "BUY"
        order = {"symbol": symbol.upper(), "side": close_side, "type": "MARKET", "quantity": self._fmt_qty(quantity)}
        if position_mode == "HEDGE":
            order["positionSide"] = "LONG" if side.lower() == "long" else "SHORT"
        else:
            order["reduceOnly"] = "true"
        return await self._request("POST", "/fapi/v1/order", order, signed=True)

    async def cancel_protection_orders(self, symbol):
        orders = await self.get_open_orders(symbol)
        for order in orders or []:
            if order.get('type') in {'STOP_MARKET','TAKE_PROFIT_MARKET','TRAILING_STOP_MARKET','STOP','TAKE_PROFIT'} and str(order.get('clientOrderId') or '').startswith('HHHAI-'):
                try:
                    await self.cancel_order(symbol, order.get('orderId'))
                except Exception:
                    pass

    async def update_dynamic_protection(self, symbol, side, quantity, stop_price, position_mode="ONE_WAY"):
        # Replace only HHHAI-style protective orders; unrelated/manual orders are left alone.
        try:
            await self.cancel_protection_orders(symbol)
        except Exception:
            pass
        return await self.place_protection(symbol, side, quantity, stop_price, None, position_mode=position_mode)
    @staticmethod
    def _fmt_price(value): return f'{float(value):.8f}'.rstrip('0').rstrip('.')
    @staticmethod
    def _fmt_qty(value): return f'{float(value):.8f}'.rstrip('0').rstrip('.')
    async def get_open_orders(self,symbol=None): return await self._request('GET','/fapi/v1/openOrders',({'symbol':symbol} if symbol else {}),True)
    async def get_ticker(self,symbol): return await self._request('GET','/fapi/v1/ticker/price',{'symbol':symbol.upper()})
    async def place_order(self,order):
        return await self._request('POST','/fapi/v1/order',order,True)
    async def cancel_order(self,symbol,order_id): return await self._request('DELETE','/fapi/v1/order',{'symbol':symbol,'orderId':order_id},True)
