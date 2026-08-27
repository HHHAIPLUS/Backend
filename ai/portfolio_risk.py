from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math


@dataclass
class Exposure:
    symbol: str
    side: str
    notional: float
    beta_to_btc: float = 0.0
    beta_to_eth: float = 0.0
    volatility: float = 0.0


@dataclass
class PortfolioPolicy:
    max_gross_exposure_pct: float = 300.0
    max_net_exposure_pct: float = 150.0
    max_correlated_cluster_pct: float = 100.0
    max_single_asset_pct: float = 50.0
    correlation_threshold: float = 0.75


class PortfolioRiskEngine:
    """Portfolio-level veto layer. It cannot execute trades."""

    def __init__(self, policy: PortfolioPolicy | None = None):
        self.policy = policy or PortfolioPolicy()

    def exposure_report(self, equity: float, exposures: Iterable[Exposure]) -> dict:
        positions = list(exposures)
        if equity <= 0:
            raise ValueError("equity must be positive")

        gross = sum(abs(x.notional) for x in positions)
        net = sum(x.notional if x.side.lower() == "long" else -x.notional for x in positions)

        asset_pct = {
            x.symbol: abs(x.notional) / equity * 100 for x in positions
        }

        btc_linked = sum(abs(x.notional) for x in positions if abs(x.beta_to_btc) >= self.policy.correlation_threshold)
        eth_linked = sum(abs(x.notional) for x in positions if abs(x.beta_to_eth) >= self.policy.correlation_threshold)

        return {
            "gross_exposure_pct": gross / equity * 100,
            "net_exposure_pct": abs(net) / equity * 100,
            "single_asset_pct": max(asset_pct.values(), default=0),
            "btc_correlated_cluster_pct": btc_linked / equity * 100,
            "eth_correlated_cluster_pct": eth_linked / equity * 100,
            "position_count": len(positions),
        }

    def evaluate(self, equity: float, exposures: Iterable[Exposure]) -> dict:
        report = self.exposure_report(equity, exposures)
        reasons = []

        if report["gross_exposure_pct"] > self.policy.max_gross_exposure_pct:
            reasons.append("Gross portfolio exposure is above the policy limit.")
        if report["net_exposure_pct"] > self.policy.max_net_exposure_pct:
            reasons.append("Net directional exposure is above the policy limit.")
        if report["single_asset_pct"] > self.policy.max_single_asset_pct:
            reasons.append("A single asset represents too much account exposure.")
        if report["btc_correlated_cluster_pct"] > self.policy.max_correlated_cluster_pct:
            reasons.append("BTC-correlated exposure is too concentrated.")
        if report["eth_correlated_cluster_pct"] > self.policy.max_correlated_cluster_pct:
            reasons.append("ETH-correlated exposure is too concentrated.")

        return {
            "decision": "block" if reasons else "allow",
            "reasons": reasons,
            "report": report,
            "execution_authority": False,
        }

    def correlation_adjusted_risk(self, exposures: list[Exposure]) -> float:
        """Simple conservative concentration proxy for correlated positions."""
        if not exposures:
            return 0.0
        weighted = 0.0
        for x in exposures:
            correlation_factor = max(abs(x.beta_to_btc), abs(x.beta_to_eth), 0.0)
            weighted += abs(x.notional) * (1.0 + correlation_factor)
        return weighted
