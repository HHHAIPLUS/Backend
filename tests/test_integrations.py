import hashlib,hmac
from app.exchanges.binance import BinanceAdapter
from app.exchanges.bitget import BitgetAdapter

def test_binance_hmac_shape(monkeypatch):
    monkeypatch.setenv('BINANCE_API_KEY','k'); monkeypatch.setenv('BINANCE_API_SECRET','s')
    a=BinanceAdapter(); a.api_key='k'; a.secret='s'
    _,_,params,headers=a._signed('GET','/fapi/v2/account',{'foo':'bar'})
    payload='foo=bar&timestamp='+str(params['timestamp'])+'&recvWindow=5000'
    assert params['signature']==hmac.new(b's',payload.encode(),hashlib.sha256).hexdigest()
    assert headers['X-MBX-APIKEY']=='k'

def test_bitget_signature_headers(monkeypatch):
    a=BitgetAdapter(); a.key='k'; a.secret='s'; a.passphrase='p'
    h=a._headers('GET','/api/v2/mix/account/accounts','productType=USDT-FUTURES','')
    assert {'ACCESS-KEY','ACCESS-SIGN','ACCESS-TIMESTAMP','ACCESS-PASSPHRASE'}<=set(h)


def test_exchange_factory_respects_execution_mode(monkeypatch):
    from app.exchanges.factory import adapters
    monkeypatch.setattr('app.exchanges.binance.settings.binance_testnet', True)
    assert adapters(testnet=False)['binance'].testnet is False
    assert adapters(testnet=True)['binance'].testnet is True

import asyncio


def test_bitget_position_mode_reads_account_configuration(monkeypatch):
    a=BitgetAdapter(); a.key='k'; a.secret='s'; a.passphrase='p'
    async def fake_request(method,path,params=None,body=None,private=False):
        assert path=='/api/v2/mix/account/account'
        assert params['symbol']=='BTCUSDT'
        return {'posMode':'one_way_mode'}
    monkeypatch.setattr(a,'_request',fake_request)
    assert asyncio.run(a.get_position_mode('BTCUSDT'))=='ONE_WAY'


def test_bitget_wait_for_fill_stops_on_filled(monkeypatch):
    a=BitgetAdapter(); a.key='k'; a.secret='s'; a.passphrase='p'
    async def fake_detail(symbol, order_id=None, client_oid=None):
        return {'state':'filled','baseVolume':'0.002'}
    monkeypatch.setattr(a,'get_order_detail',fake_detail)
    detail=asyncio.run(a.wait_for_fill('BTCUSDT','123',0.5))
    assert detail['state']=='filled'
    assert detail['baseVolume']=='0.002'
