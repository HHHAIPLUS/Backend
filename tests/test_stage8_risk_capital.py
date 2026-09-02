from datetime import datetime, timezone, timedelta

from ai.portfolio_risk import Exposure
from ai.risk_capital_engine import AccountState, MarketSafety, RiskCapitalEngine


def fresh_market():
    return MarketSafety(data_quality=1.0, observed_at=datetime.now(timezone.utc).isoformat(), spread_bps=2, expected_slippage_bps=2)


def account(**kwargs):
    base = dict(equity=10_000, free_margin=8_000)
    base.update(kwargs)
    return AccountState(**base)


def test_risk_per_trade_and_leverage_are_hard_vetoes():
    e = RiskCapitalEngine()
    r = e.evaluate(account(leverage=6), fresh_market(), 0.5, 1000)
    assert not r.allowed
    assert any("leverage" in x.lower() for x in r.reasons)
    r = e.evaluate(account(), fresh_market(), 0.51, 1000)
    assert not r.allowed
    assert any("risk per trade" in x.lower() for x in r.reasons)


def test_portfolio_and_correlated_exposure_limits_are_enforced():
    e = RiskCapitalEngine()
    exposures = [Exposure("BTCUSDT", "long", 6000, beta_to_btc=1.0), Exposure("ETHUSDT", "long", 5000, beta_to_eth=1.0)]
    r = e.evaluate(account(), fresh_market(), 0.5, 100, exposures)
    assert not r.allowed
    assert any("exposure" in x.lower() or "concentrated" in x.lower() for x in r.reasons)


def test_daily_rolling_loss_and_drawdown_trigger_emergency_stop():
    e = RiskCapitalEngine()
    for kwargs, text in [
        ({"daily_pnl_pct": -3.0}, "daily loss"),
        ({"rolling_pnl_pct": -5.0}, "rolling loss"),
        ({"drawdown_pct": 8.0}, "drawdown"),
    ]:
        r = e.evaluate(account(**kwargs), fresh_market(), 0.5, 100)
        assert r.decision == "emergency_stop"
        assert r.emergency_stop
        assert any(text in x.lower() for x in r.reasons)


def test_stale_contradictory_and_shock_data_block_entries():
    e = RiskCapitalEngine()
    stale = MarketSafety(data_quality=1.0, observed_at=(datetime.now(timezone.utc)-timedelta(seconds=60)).isoformat())
    assert not e.evaluate(account(), stale, 0.5, 100).allowed
    contradictory = fresh_market(); contradictory.contradictory = True
    assert not e.evaluate(account(), contradictory, 0.5, 100).allowed
    shock = fresh_market(); shock.shock = True
    assert not e.evaluate(account(), shock, 0.5, 100).allowed


def test_kill_switches_are_fail_closed_and_persistable_state_is_representable():
    e = RiskCapitalEngine()
    e.engage_global_kill("catastrophic test")
    r = e.evaluate(account(), fresh_market(), 0.5, 100)
    assert r.decision == "emergency_stop"
    e.reset_global_kill(); e.engage_exchange_kill("binance", "exchange instability")
    r = e.evaluate(account(), fresh_market(), 0.5, 100, exchange="binance")
    assert r.decision == "emergency_stop"
    assert e.kill_status()["exchanges"]["binance"]["enabled"]


def test_model_confidence_cannot_bypass_capital_controls():
    e = RiskCapitalEngine()
    r = e.evaluate(account(drawdown_pct=8.0), fresh_market(), 0.01, 10, model_confidence=1.0)
    assert not r.allowed
    assert r.emergency_stop


def test_size_is_capped_by_independent_risk_policy():
    e = RiskCapitalEngine()
    assert e.size_for_risk(10_000, 1.0, 10.0) == e.size_for_risk(10_000, 1.0, 0.5)
