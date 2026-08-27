from __future__ import annotations
from dataclasses import asdict
from .simulation_engine import Candle, SimulationConfig, replay, walk_forward_slices, monte_carlo_returns

class SimulationService:
    """Research-only simulation boundary. It never places exchange orders."""

    def run_replay(self, candles, signals, config=None):
        result = replay(candles, signals, config)
        return asdict(result)

    def run_walk_forward(self, candles, signal_factory, train_size, test_size, config=None):
        results = []
        for train, test in walk_forward_slices(candles, train_size, test_size):
            # The factory receives training data and returns test signals.
            signals = signal_factory(train, test)
            results.append(asdict(replay(test, signals, config)))
        return results

    def run_monte_carlo(self, returns, paths=1000, seed=13):
        values = monte_carlo_returns(returns, paths, seed)
        if not values:
            return {"paths": 0, "worst": 0, "median": 0, "best": 0}
        values.sort()
        return {
            "paths": len(values),
            "worst": values[0],
            "median": values[len(values)//2],
            "best": values[-1],
        }
