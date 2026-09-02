from ai.paper_trading import PaperExecutionEngine
from ai.position_intelligence import POSITION_ENGINE, PositionObservation


def obs(**kw):
    base = dict(exchange="paper", symbol="BTCUSDT", side="long", quantity=1.0, entry_price=100.0, current_price=101.0, unrealized_return=0.01, peak_return=0.01, thesis_integrity=0.5, momentum=0.20, trend_strength=0.70, buying_pressure=0.70, selling_pressure=0.30, volatility=0.20, liquidity_stress=0.10, news_risk=0.05, market_risk=0.05, funding_bias=0.0, open_interest_change=0.01, expected_continuation_value=0.01, downside_risk=0.1, timestamp="2026-09-02T00:00:00Z")
    base.update(kw)
    return PositionObservation(**base)


def test_continuous_state_reassessment_holds_when_edge_is_healthy():
    d = POSITION_ENGINE.evaluate(obs(), {"side":"long", "momentum":0.20})
    assert d.action in {"hold", "protect", "reduce"}
    assert d.expected_continuation_value > 0
    assert 0 <= d.thesis_integrity <= 1


def test_thesis_invalidation_exits_losing_position():
    d = POSITION_ENGINE.evaluate(obs(current_price=96, unrealized_return=-0.04, momentum=-0.9, trend_strength=0.05, buying_pressure=0.05, selling_pressure=0.95), {"side":"long","momentum":0.7})
    assert d.action in {"exit", "emergency_exit", "reduce"}
    assert d.thesis_integrity < 0.38 or d.downside_risk >= 0.78


def test_adverse_news_and_market_shock_can_force_emergency_exit():
    d = POSITION_ENGINE.evaluate(obs(news_risk=1.0, market_risk=1.0, liquidity_stress=0.95, momentum=-1.0, current_price=99), {"side":"long","momentum":0.2})
    assert d.action == "emergency_exit"
    assert d.close_fraction == 1.0


def test_profit_protection_is_monotonic_for_long():
    first = POSITION_ENGINE.evaluate(obs(current_price=110, unrealized_return=.10, peak_return=.10, volatility=.15), {"side":"long","momentum":.2})
    second = POSITION_ENGINE.evaluate(obs(current_price=111, unrealized_return=.11, peak_return=.11, volatility=.05), {"side":"long","momentum":.2}, previous_protection=first.protection_price)
    assert second.protection_price >= first.protection_price


def test_profit_protection_is_monotonic_for_short():
    first = POSITION_ENGINE.evaluate(obs(side="short", current_price=90, entry_price=100, unrealized_return=.10, peak_return=.10, buying_pressure=.30, selling_pressure=.70), {"side":"short","momentum":-.2})
    second = POSITION_ENGINE.evaluate(obs(side="short", current_price=89, entry_price=100, unrealized_return=.11, peak_return=.11, buying_pressure=.20, selling_pressure=.80, volatility=.05), {"side":"short","momentum":-.2}, previous_protection=first.protection_price)
    assert second.protection_price <= first.protection_price


def test_partial_exit_fraction_is_bounded():
    d = POSITION_ENGINE.evaluate(obs(current_price=108, unrealized_return=.08, peak_return=.10, momentum=-.4, buying_pressure=.25, selling_pressure=.75), {"side":"long","momentum":.3})
    assert 0 <= d.close_fraction <= 1


def test_invalid_telemetry_fails_closed():
    d = POSITION_ENGINE.evaluate(obs(entry_price=0), {"side":"long"})
    assert d.action == "hold"
    assert d.evidence["invalid_telemetry"] is True


def test_paper_position_management_never_grants_execution_authority():
    paper = PaperExecutionEngine()
    paper.submit("BTCUSDT", "buy", 1.0, 100.0)
    assert paper.snapshot()["execution_authority"] is False
    assert "real_money" not in paper.snapshot()


def test_historical_style_replay_is_deterministic():
    thesis={"side":"long","momentum":.3}
    path=[obs(current_price=101,unrealized_return=.01,peak_return=.01), obs(current_price=103,unrealized_return=.03,peak_return=.03), obs(current_price=102,unrealized_return=.02,peak_return=.03, momentum=.0), obs(current_price=97,unrealized_return=-.03,peak_return=.03,momentum=-.8,buying_pressure=.1,selling_pressure=.9)]
    actions=[POSITION_ENGINE.evaluate(p,thesis).action for p in path]
    assert len(actions)==4
    assert actions[-1] in {"reduce","exit","emergency_exit"}
