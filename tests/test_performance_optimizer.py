from ai.performance_optimizer import PerformanceOptimizer, TradeRecord


def trade(**overrides):
    data = dict(
        trade_id="T1",
        symbol="BTCUSDT",
        side="long",
        entry=100,
        exit=108,
        highest_favorable=110,
        lowest_adverse=97,
        expected_direction="long",
        actual_direction="long",
        pnl=8,
        fees=0.5,
        reason="order_flow",
    )
    data.update(overrides)
    return TradeRecord(**data)


def test_exit_quality_is_measured():
    score = PerformanceOptimizer.exit_quality(trade())
    assert 0 < score < 1


def test_missed_opportunity_is_detected():
    missed = PerformanceOptimizer.missed_opportunity(trade())
    assert missed > 0


def test_false_signal_is_detected():
    assert PerformanceOptimizer.classify_signal(
        trade(expected_direction="long", actual_direction="short")
    ) == "false_signal"


def test_profit_factor():
    trades = [trade(pnl=10), trade(trade_id="T2", pnl=-5, exit=95)]
    assert PerformanceOptimizer.profit_factor(trades) == 9.5 / 5.5


def test_attribution_and_authority():
    result = PerformanceOptimizer().analyze([trade()])
    assert "order_flow" in result["performance_attribution"]
    assert result["execution_authority"] is False
