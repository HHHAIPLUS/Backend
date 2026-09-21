from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import math
import random

import numpy as np


@dataclass(frozen=True)
class RobustnessConfig:
    folds: int = 5
    min_trades_per_fold: int = 20
    fee_rate: float = 0.0008
    spread_rate: float = 0.0002
    slippage_rate: float = 0.0003
    adverse_slippage_rate: float = 0.0010
    monte_carlo_runs: int = 2000
    seed: int = 42


class Phase3Robustness:
    """Research-only robustness analysis.

    This module never places orders and never changes Phase 2 acceptance gates.
    It consumes frozen trade/market observations and reports degradation rather
    than selecting parameters from the final robustness periods.
    """

    def __init__(self, config: RobustnessConfig | None = None):
        self.config = config or RobustnessConfig()

    @staticmethod
    def _returns(observations: list[dict[str, Any]]) -> np.ndarray:
        values = np.asarray(
            [float(x["gross_return"]) for x in observations], dtype=float
        )
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Robustness observations must contain finite gross_return values.")
        return values

    @staticmethod
    def _drawdown(values: Iterable[float]) -> float:
        equity = np.cumsum(np.asarray(list(values), dtype=float))
        if len(equity) == 0:
            return 0.0
        peak = np.maximum.accumulate(np.r_[0.0, equity])
        return float(np.max(peak[1:] - equity))

    def _net(self, gross: np.ndarray, extra_cost: float = 0.0) -> np.ndarray:
        return gross - (
            self.config.fee_rate
            + self.config.spread_rate
            + self.config.slippage_rate
            + extra_cost
        )

    def walk_forward(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        if len(observations) < self.config.folds * self.config.min_trades_per_fold:
            raise ValueError("Not enough observations for the requested walk-forward folds.")

        # Chronological contiguous evaluation only. The engine deliberately does
        # not optimize parameters on these periods.
        n = len(observations)
        edges = np.linspace(0, n, self.config.folds + 1, dtype=int)
        folds = []
        for i in range(self.config.folds):
            chunk = observations[edges[i]:edges[i + 1]]
            if len(chunk) < self.config.min_trades_per_fold:
                raise ValueError("A walk-forward fold is below the minimum trade count.")
            gross = self._returns(chunk)
            net = self._net(gross)
            folds.append({
                "fold": i + 1,
                "start": str(chunk[0].get("timestamp", i)),
                "end": str(chunk[-1].get("timestamp", i)),
                "trades": len(chunk),
                "gross_total_return": float(gross.sum()),
                "net_total_return": float(net.sum()),
                "avg_net_return": float(net.mean()),
                "max_drawdown": self._drawdown(net),
            })

        return {
            "folds": folds,
            "positive_net_folds": int(sum(x["net_total_return"] > 0 for x in folds)),
            "worst_fold_net_return": float(min(x["net_total_return"] for x in folds)),
            "worst_fold_drawdown": float(max(x["max_drawdown"] for x in folds)),
        }

    def regime_analysis(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in observations:
            regime = str(row.get("regime", "unknown")).lower()
            groups.setdefault(regime, []).append(row)

        result = {}
        for regime, rows in sorted(groups.items()):
            gross = self._returns(rows)
            net = self._net(gross)
            result[regime] = {
                "trades": len(rows),
                "net_total_return": float(net.sum()),
                "avg_net_return": float(net.mean()),
                "max_drawdown": self._drawdown(net),
            }
        return result

    def volatility_liquidity_analysis(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in observations:
            volatility = str(row.get("volatility_bucket", "unknown")).lower()
            liquidity = str(row.get("liquidity_bucket", "unknown")).lower()
            key = f"{volatility}:{liquidity}"
            buckets.setdefault(key, []).append(row)

        result = {}
        for key, rows in sorted(buckets.items()):
            gross = self._returns(rows)
            net = self._net(gross)
            result[key] = {
                "trades": len(rows),
                "net_total_return": float(net.sum()),
                "avg_net_return": float(net.mean()),
                "max_drawdown": self._drawdown(net),
            }
        return result

    def execution_cost_sensitivity(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        gross = self._returns(observations)
        scenarios = {
            "base": 0.0,
            "spread_2x": self.config.spread_rate,
            "slippage_2x": self.config.slippage_rate,
            "adverse_execution": self.config.adverse_slippage_rate,
            "stress_3x_cost": (
                2 * (self.config.spread_rate + self.config.slippage_rate)
            ),
        }
        result = {}
        for name, extra in scenarios.items():
            net = self._net(gross, extra)
            result[name] = {
                "total_net_return": float(net.sum()),
                "avg_net_return": float(net.mean()),
                "max_drawdown": self._drawdown(net),
                "profitable": bool(net.sum() > 0),
            }
        return result

    def monte_carlo(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        net = self._net(self._returns(observations))
        rng = random.Random(self.config.seed)
        totals = np.empty(self.config.monte_carlo_runs, dtype=float)
        drawdowns = np.empty(self.config.monte_carlo_runs, dtype=float)
        for i in range(self.config.monte_carlo_runs):
            sample = [net[rng.randrange(len(net))] for _ in range(len(net))]
            totals[i] = float(np.sum(sample))
            drawdowns[i] = self._drawdown(sample)

        return {
            "runs": self.config.monte_carlo_runs,
            "seed": self.config.seed,
            "probability_positive_total_return": float(np.mean(totals > 0)),
            "total_return_p05": float(np.quantile(totals, 0.05)),
            "total_return_p50": float(np.quantile(totals, 0.50)),
            "total_return_p95": float(np.quantile(totals, 0.95)),
            "drawdown_p95": float(np.quantile(drawdowns, 0.95)),
        }

    def analyze(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        if any("gross_return" not in row for row in observations):
            raise ValueError("Every robustness observation requires gross_return.")
        return {
            "research_only": True,
            "execution_authority": False,
            "live_trading": False,
            "walk_forward": self.walk_forward(observations),
            "regimes": self.regime_analysis(observations),
            "volatility_liquidity": self.volatility_liquidity_analysis(observations),
            "execution_cost_sensitivity": self.execution_cost_sensitivity(observations),
            "monte_carlo": self.monte_carlo(observations),
        }
