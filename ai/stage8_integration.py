from __future__ import annotations

from types import MethodType

from ai.risk_capital_engine import AccountState, MarketSafety, RiskCapitalEngine
from app.persistence.repository import load_risk_controls


async def hydrate_stage8_risk(engine: RiskCapitalEngine) -> None:
    try:
        for row in await load_risk_controls():
            if not row.get("enabled"):
                continue
            scope = str(row.get("scope", ""))
            reason = str(row.get("reason") or "Persisted risk control")
            if scope == "global":
                engine.engage_global_kill(reason)
            elif scope.startswith("exchange:"):
                engine.engage_exchange_kill(scope.split(":", 1)[1], reason)
    except Exception:
        return


def install_stage8_risk(trader) -> RiskCapitalEngine:
    """Install an independent veto around the existing entry-risk calculation.

    The wrapper deliberately leaves position-closing paths alone: reducing risk
    must remain possible even while an entry circuit breaker is active.
    """
    engine = RiskCapitalEngine()
    original = trader._risk_check

    async def guarded_risk_check(self, world, decision, candidate):
        base = await original(world, decision, candidate)
        if not base.get("allowed") or not candidate or not decision.get("execution_candidate"):
            return base

        exchange = self._exchange_for_market()
        equity = float(base.get("equity") or self.config.paper_equity)
        quantity = float(base.get("quantity") or 0.0)
        price = float(world.get("market", {}).get("price") or 0.0)
        spread = float(base.get("spread_bps") or 0.0)
        observed_at = world.get("observed_at")
        data_quality = float(world.get("data_quality") or 0.0)
        account = AccountState(
            equity=equity,
            free_margin=equity,
            daily_pnl_pct=0.0 if not self._day_start_equity else (equity - self._day_start_equity) / self._day_start_equity * 100.0,
            rolling_pnl_pct=0.0 if not self._day_start_equity else (equity - self._day_start_equity) / self._day_start_equity * 100.0,
            drawdown_pct=0.0 if not self._peak_equity else max(0.0, (self._peak_equity - equity) / self._peak_equity * 100.0),
            leverage=(quantity * price / equity) if equity > 0 else 0.0,
            open_positions=len(self.paper.positions) if self.execution_mode == "paper" else 0,
        )
        market = MarketSafety(
            data_quality=data_quality,
            observed_at=observed_at,
            exchange_healthy=True,
            contradictory=bool(world.get("market", {}).get("contradictory", False)),
            spread_bps=spread,
            expected_slippage_bps=spread,
            shock=float(world.get("news_risk") or 0.0) >= 0.95 or float(world.get("liquidity_stress") or 0.0) >= 0.95,
        )
        result = engine.evaluate(
            account=account,
            market=market,
            proposed_risk_pct=self.config.risk_pct,
            proposed_notional=quantity * price,
            exchange=exchange,
            model_confidence=decision.get("confidence"),
        )
        if not result.allowed:
            base = dict(base)
            base["allowed"] = False
            base["decision"] = result.decision
            base["reasons"] = list(base.get("reasons", [])) + result.reasons
            base["stage8_risk"] = result.__dict__
            return base
        base = dict(base)
        base["stage8_risk"] = result.__dict__
        return base

    trader._risk_check = MethodType(guarded_risk_check, trader)
    trader.stage8_risk_engine = engine
    return engine
