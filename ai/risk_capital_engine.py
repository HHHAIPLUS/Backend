from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Iterable

from ai.portfolio_risk import Exposure, PortfolioRiskEngine, PortfolioPolicy


@dataclass(frozen=True)
class RiskPolicy:
    max_risk_per_trade_pct: float = 0.50
    max_gross_exposure_pct: float = 300.0
    max_net_exposure_pct: float = 150.0
    max_correlated_exposure_pct: float = 100.0
    max_single_asset_pct: float = 50.0
    max_leverage: float = 5.0
    max_open_positions: int = 3
    max_daily_loss_pct: float = 3.0
    max_rolling_loss_pct: float = 5.0
    max_drawdown_pct: float = 8.0
    min_free_margin_pct: float = 30.0
    max_slippage_bps: float = 20.0
    max_spread_bps: float = 12.0
    stale_after_seconds: int = 30
    max_execution_failures: int = 3


@dataclass
class AccountState:
    equity: float
    free_margin: float
    daily_pnl_pct: float = 0.0
    rolling_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    leverage: float = 0.0
    open_positions: int = 0
    execution_failures: int = 0
    observed_at: str | None = None


@dataclass
class MarketSafety:
    data_quality: float = 0.0
    observed_at: str | None = None
    exchange_healthy: bool = True
    contradictory: bool = False
    spread_bps: float = 0.0
    expected_slippage_bps: float = 0.0
    shock: bool = False


@dataclass
class RiskDecision:
    allowed: bool
    decision: str
    reasons: list[str] = field(default_factory=list)
    emergency_stop: bool = False
    risk_pct: float = 0.0
    max_notional: float = 0.0
    free_margin_pct: float = 0.0
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RiskCapitalEngine:
    """Independent survival layer. It only vetoes/sizes; it cannot create an order."""

    def __init__(self, policy: RiskPolicy | None = None) -> None:
        self.policy = policy or RiskPolicy()
        self.portfolio = PortfolioRiskEngine(PortfolioPolicy(
            max_gross_exposure_pct=self.policy.max_gross_exposure_pct,
            max_net_exposure_pct=self.policy.max_net_exposure_pct,
            max_correlated_cluster_pct=self.policy.max_correlated_exposure_pct,
            max_single_asset_pct=self.policy.max_single_asset_pct,
        ))
        self._global_kill = False
        self._global_reason = ""
        self._exchange_kills: dict[str, str] = {}
        self._last_account: AccountState | None = None

    @property
    def global_kill(self) -> bool:
        return self._global_kill

    def engage_global_kill(self, reason: str) -> None:
        self._global_kill = True
        self._global_reason = reason.strip() or "Global risk kill switch engaged."

    def reset_global_kill(self) -> None:
        self._global_kill = False
        self._global_reason = ""

    def engage_exchange_kill(self, exchange: str, reason: str) -> None:
        self._exchange_kills[exchange.lower()] = reason.strip() or "Exchange risk kill switch engaged."

    def reset_exchange_kill(self, exchange: str) -> None:
        self._exchange_kills.pop(exchange.lower(), None)

    def kill_status(self) -> dict:
        return {"global": {"enabled": self._global_kill, "reason": self._global_reason},
                "exchanges": {k: {"enabled": True, "reason": v} for k, v in self._exchange_kills.items()},
                "execution_authority": False}

    def evaluate(self, account: AccountState, market: MarketSafety,
                 proposed_risk_pct: float, proposed_notional: float,
                 exposures: Iterable[Exposure] = (), exchange: str = "unknown",
                 model_confidence: float | None = None) -> RiskDecision:
        self._last_account = account
        reasons: list[str] = []
        emergency = False
        equity = float(account.equity)
        free = float(account.free_margin)
        free_pct = (free / equity * 100.0) if equity > 0 else 0.0
        if not isfinite(equity) or equity <= 0:
            reasons.append("Account equity is non-positive or invalid.")
            emergency = True
        if self._global_kill:
            reasons.append(self._global_reason)
            emergency = True
        if exchange.lower() in self._exchange_kills:
            reasons.append(self._exchange_kills[exchange.lower()])
            emergency = True
        if not market.exchange_healthy:
            reasons.append("Exchange health check failed.")
            emergency = True
        if market.data_quality < 0.90 or self._stale(market.observed_at):
            reasons.append("Market data is stale or below the minimum quality threshold.")
        if market.contradictory:
            reasons.append("Market data is contradictory.")
        if market.shock:
            reasons.append("Market shock detected.")
        if account.daily_pnl_pct <= -self.policy.max_daily_loss_pct:
            reasons.append("Daily loss limit reached.")
            emergency = True
        if account.rolling_pnl_pct <= -self.policy.max_rolling_loss_pct:
            reasons.append("Rolling loss limit reached.")
            emergency = True
        if account.drawdown_pct >= self.policy.max_drawdown_pct:
            reasons.append("Maximum drawdown limit reached.")
            emergency = True
        if proposed_risk_pct > self.policy.max_risk_per_trade_pct:
            reasons.append("Proposed risk per trade exceeds the independent risk limit.")
        if account.leverage > self.policy.max_leverage:
            reasons.append("Account leverage exceeds the independent leverage limit.")
        if account.open_positions >= self.policy.max_open_positions:
            reasons.append("Maximum open-position count reached.")
        if free_pct < self.policy.min_free_margin_pct:
            reasons.append("Free-margin buffer is below the independent minimum.")
        if market.spread_bps > self.policy.max_spread_bps:
            reasons.append("Spread exceeds the independent execution limit.")
        if market.expected_slippage_bps > self.policy.max_slippage_bps:
            reasons.append("Expected slippage exceeds the independent execution limit.")
        if account.execution_failures >= self.policy.max_execution_failures:
            reasons.append("Repeated execution failures require an execution circuit breaker.")
            emergency = True
        if proposed_notional < 0 or not isfinite(proposed_notional):
            reasons.append("Proposed notional is invalid.")
        portfolio = self.portfolio.evaluate(equity, exposures)
        reasons.extend(portfolio["reasons"])
        # Confidence is informational here. It can never remove a risk veto.
        _ = model_confidence
        decision = "emergency_stop" if emergency else ("block" if reasons else "allow")
        return RiskDecision(allowed=decision == "allow", decision=decision, reasons=reasons,
                            emergency_stop=emergency, risk_pct=max(0.0, min(proposed_risk_pct, self.policy.max_risk_per_trade_pct)),
                            max_notional=max(0.0, proposed_notional), free_margin_pct=free_pct)

    def size_for_risk(self, equity: float, stop_distance_pct: float,
                      risk_pct: float | None = None, price: float | None = None) -> float:
        risk = self.policy.max_risk_per_trade_pct if risk_pct is None else min(risk_pct, self.policy.max_risk_per_trade_pct)
        if equity <= 0 or stop_distance_pct <= 0 or risk <= 0:
            return 0.0
        notional = equity * (risk / 100.0) / (stop_distance_pct / 100.0)
        return notional if price is None else notional / price

    @staticmethod
    def _stale(observed_at: str | None, max_age: int = 30) -> bool:
        if not observed_at:
            return True
        try:
            dt = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - dt).total_seconds() > max_age
        except (TypeError, ValueError):
            return True
