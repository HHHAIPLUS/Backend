from app.ml.phase3_robustness import Phase3Robustness, RobustnessConfig


def _observations(n=100):
    regimes = ("bull", "bear", "sideways")
    vol = ("low", "high")
    liq = ("low", "high")
    return [
        {
            "timestamp": f"2026-01-{(i % 28) + 1:02d}T{i % 24:02d}:00:00",
            "gross_return": 0.002 if i % 3 else -0.001,
            "regime": regimes[i % len(regimes)],
            "volatility_bucket": vol[i % len(vol)],
            "liquidity_bucket": liq[i % len(liq)],
        }
        for i in range(n)
    ]


def test_walk_forward_is_chronological_and_reports_worst_period():
    result = Phase3Robustness(RobustnessConfig(folds=5, min_trades_per_fold=10)).walk_forward(_observations())
    assert len(result["folds"]) == 5
    assert [x["fold"] for x in result["folds"]] == [1, 2, 3, 4, 5]
    assert "worst_fold_net_return" in result


def test_regime_and_market_condition_breakdowns_are_not_hidden():
    engine = Phase3Robustness(RobustnessConfig(folds=5, min_trades_per_fold=10))
    result = engine.analyze(_observations())
    assert set(result["regimes"]) == {"bear", "bull", "sideways"}
    assert result["volatility_liquidity"]
    assert result["execution_authority"] is False
    assert result["live_trading"] is False


def test_cost_stress_is_more_conservative_than_base():
    result = Phase3Robustness(RobustnessConfig(folds=5, min_trades_per_fold=10)).execution_cost_sensitivity(_observations())
    assert result["adverse_execution"]["total_net_return"] <= result["base"]["total_net_return"]
    assert result["stress_3x_cost"]["total_net_return"] <= result["base"]["total_net_return"]


def test_monte_carlo_is_reproducible():
    cfg = RobustnessConfig(folds=5, min_trades_per_fold=10, monte_carlo_runs=300, seed=123)
    observations = _observations()
    a = Phase3Robustness(cfg).monte_carlo(observations)
    b = Phase3Robustness(cfg).monte_carlo(observations)
    assert a == b
    assert a["runs"] == 300


def test_empty_or_invalid_observations_fail_closed():
    engine = Phase3Robustness(RobustnessConfig(folds=5, min_trades_per_fold=10))
    try:
        engine.analyze([])
        assert False, "empty robustness input must fail"
    except ValueError:
        pass

    try:
        engine.analyze([{"gross_return": float("nan")}])
        assert False, "non-finite robustness input must fail"
    except ValueError:
        pass
