from ai.simulation_engine import Candle, SimulationConfig, replay, walk_forward_slices, monte_carlo_returns

def candles(n=8):
    return [Candle(i, 100+i, 101+i, 99+i, 100+i, 1000) for i in range(n)]

def test_replay_accounts_for_costs_and_returns():
    result = replay(candles(), [1, 1, 1, -1, 0, 0, 0, 0], SimulationConfig())
    assert result.trades == 1
    assert result.fees > 0
    assert result.funding_cost > 0

def test_walk_forward_produces_non_overlapping_test_windows():
    data = candles(12)
    windows = list(walk_forward_slices(data, 4, 2))
    assert len(windows) == 4
    assert windows[0][0][-1].timestamp < windows[0][1][0].timestamp

def test_monte_carlo_is_deterministic_with_seed():
    a = monte_carlo_returns([0.01, -0.005, 0.02], paths=20, seed=7)
    b = monte_carlo_returns([0.01, -0.005, 0.02], paths=20, seed=7)
    assert a == b
