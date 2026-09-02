from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from types import MethodType
from typing import Any

from ai.stage5_engine import Stage5DecisionEngine
from app.ml.predictive import predictive_model


@dataclass
class PositionObservation:
    exchange: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_return: float
    peak_return: float
    thesis_integrity: float
    momentum: float
    trend_strength: float
    buying_pressure: float
    selling_pressure: float
    volatility: float
    liquidity_stress: float
    news_risk: float
    market_risk: float
    funding_bias: float
    open_interest_change: float
    expected_continuation_value: float
    downside_risk: float
    timestamp: str


@dataclass
class PositionDecision:
    action: str
    close_fraction: float
    thesis_integrity: float
    expected_continuation_value: float
    downside_risk: float
    shock_score: float
    protection_price: float | None
    reason: str
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PositionIntelligenceEngine:
    """Stage-6 position governor.

    This layer owns position-management decisions only. It cannot place orders.
    It combines the Stage-5 decision layer with live position telemetry and the
    original trade thesis. Protective prices are monotonic: a review may tighten
    protection but never loosen an existing exchange safety level.
    """

    VERSION = "stage6-position-intelligence-v1"

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def _finite_positive(value: float) -> bool:
        return isfinite(value) and value > 0

    def _thesis_integrity(self, side: str, thesis: dict[str, Any], obs: dict[str, Any]) -> float:
        direction = 1.0 if side.lower() == "long" else -1.0
        entry_momentum = float(thesis.get("momentum", 0.0) or 0.0)
        current_momentum = float(obs.get("momentum", 0.0) or 0.0)
        momentum_alignment = self._clamp(0.5 + direction * current_momentum * 0.5)
        original_alignment = self._clamp(0.5 + direction * entry_momentum * 0.5)
        trend = self._clamp(float(obs.get("trend_strength", 0.0) or 0.0))
        pressure = float(obs.get("buying_pressure", 0.5) if side.lower() == "long" else obs.get("selling_pressure", 0.5))
        news = self._clamp(float(obs.get("news_risk", 0.0) or 0.0))
        liquidity = self._clamp(float(obs.get("liquidity_stress", 0.0) or 0.0))
        deterioration = max(0.0, -direction * (current_momentum - entry_momentum))
        return self._clamp(0.22 * momentum_alignment + 0.16 * original_alignment + 0.22 * trend + 0.20 * pressure + 0.10 * (1 - news) + 0.10 * (1 - liquidity) - 0.20 * deterioration)

    def evaluate(self, observation: PositionObservation, thesis: dict[str, Any], *, predictive: dict[str, Any] | None = None, stage5: dict[str, Any] | None = None, previous_protection: float | None = None) -> PositionDecision:
        if not self._finite_positive(observation.entry_price) or not self._finite_positive(observation.current_price) or not self._finite_positive(observation.quantity):
            return PositionDecision("hold", 0.0, 0.0, 0.0, 1.0, 1.0, previous_protection, "Invalid position telemetry; fail closed to observation-only mode.", {"invalid_telemetry": True})

        obs = asdict(observation)
        integrity = self._thesis_integrity(observation.side, thesis, obs)
        predictive = predictive or {}
        stage5 = stage5 or {}
        expected = float(stage5.get("expected_value", observation.expected_continuation_value) or observation.expected_continuation_value)
        probs = predictive.get("probabilities") or {}
        directional_probability = float(probs.get("long", 0.0) if observation.side.lower() == "long" else probs.get("short", 0.0))
        expected += (directional_probability - 0.5) * 0.04
        expected = max(-0.20, min(0.20, expected))

        adverse_momentum = max(0.0, -(observation.momentum if observation.side.lower() == "long" else -observation.momentum))
        opposing_pressure = observation.selling_pressure if observation.side.lower() == "long" else observation.buying_pressure
        retracement = 0.0
        if observation.peak_return > 0:
            retracement = self._clamp((observation.peak_return - observation.unrealized_return) / observation.peak_return)
        downside = self._clamp(
            0.24 * observation.volatility
            + 0.18 * observation.liquidity_stress
            + 0.18 * observation.news_risk
            + 0.14 * observation.market_risk
            + 0.14 * adverse_momentum
            + 0.12 * opposing_pressure
            + 0.10 * max(0.0, -observation.unrealized_return * 12)
        )
        shock = self._clamp(0.30 * observation.news_risk + 0.25 * observation.market_risk + 0.25 * observation.liquidity_stress + 0.20 * adverse_momentum)
        edge = expected - downside * 0.025
        thesis_break = integrity < 0.38
        extreme = shock >= 0.86 or observation.liquidity_stress >= 0.94

        if extreme:
            action, fraction = "emergency_exit", 1.0
            reason = "Market-wide or liquidity shock makes continued exposure unsafe."
        elif thesis_break and observation.unrealized_return <= 0:
            action, fraction = "exit", 1.0
            reason = "The original trade thesis has materially failed while the position is losing."
        elif thesis_break or downside >= 0.78:
            action, fraction = ("exit", 1.0) if observation.unrealized_return > 0.01 else ("reduce", 0.50)
            reason = "Thesis integrity or downside risk has deteriorated beyond the position's continuation edge."
        elif observation.unrealized_return > 0 and (retracement >= 0.35 or edge < -0.005 or shock >= 0.55):
            action, fraction = "reduce", 0.35
            reason = "Profit is available but continuation value no longer dominates deterioration; bank part of the gain."
        elif observation.unrealized_return > 0 and (retracement >= 0.18 or downside >= 0.45):
            action, fraction = "protect", 0.0
            reason = "The thesis remains viable, but protection should tighten as risk rises."
        else:
            action, fraction = "hold", 0.0
            reason = "Continuation evidence remains stronger than current downside and shock risk."

        protection = None
        if action in {"hold", "protect", "reduce"} and observation.unrealized_return > 0:
            buffer = self._clamp(0.001 + observation.volatility * 0.006 + max(downside, shock) * 0.003, 0.001, 0.018)
            proposed = observation.current_price * (1 - buffer) if observation.side.lower() == "long" else observation.current_price * (1 + buffer)
            if previous_protection is not None:
                protection = max(previous_protection, proposed) if observation.side.lower() == "long" else min(previous_protection, proposed)
            else:
                protection = proposed

        return PositionDecision(
            action, fraction, integrity, expected, downside, shock, protection, reason,
            {"predictive_probability": directional_probability, "retracement_from_peak": retracement, "edge_after_risk": edge, "stage5_action": stage5.get("action"), "stage5_vetoes": stage5.get("vetoes", []), "thesis": thesis},
        )


POSITION_ENGINE = PositionIntelligenceEngine()


def install_stage6_position_intelligence(trader: Any) -> None:
    """Install the Stage-6 manager into the existing autonomous loop.

    The patch is intentionally isolated so Stage-5 decision/execution code is
    not duplicated or given additional authority. Paper, testnet and live all
    use the same decision path; only the execution adapter differs.
    """
    if getattr(trader, "_stage6_installed", False):
        return

    trader._stage6_installed = True
    trader.stage6_version = POSITION_ENGINE.VERSION
    trader.stage6_reviews = 0
    trader.stage6_last_decision = None
    trader.stage6_last_error = None

    async def manage(self, symbol: str) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        rows: list[dict[str, Any]] = []
        mode = self.execution_mode
        if mode in {"testnet", "live"}:
            if not self._execution_gate()[0]:
                return []
            exchange = self._exchange_for_market()
            adapter = self._adapters()[exchange]
            try:
                raw = await adapter.get_positions(symbol if exchange == "bitget" else None)
            except TypeError:
                raw = await adapter.get_positions()
            raw_rows = self._position_rows(raw)
        else:
            exchange = "paper"
            raw_rows = [
                {"symbol": p.symbol, "holdSide": p.side, "positionAmt": p.quantity, "entryPrice": p.entry_price, "markPrice": p.mark_price}
                for p in self.paper.positions.values() if p.symbol.upper() == symbol.upper()
            ]

        for row in raw_rows:
            psymbol, side, qty, entry, current = self._position_fields(row)
            if psymbol != symbol.upper() or qty <= 0 or entry <= 0 or current <= 0:
                continue
            key = f"{exchange}:{psymbol}:{side}"
            if mode == "paper":
                current = float((await __import__("asyncio").to_thread(build_world_intelligence, psymbol, exchange)).market.features.get("last_price", current) or current)
                self.paper.mark(psymbol, current)
            world = (await __import__("asyncio").to_thread(build_world_intelligence, psymbol, exchange)).model_dump(mode="json")
            market = world.get("market", world)
            features = market.get("features") or {}
            flow = market.get("order_flow") or {}
            derivatives = market.get("derivatives") or {}
            regime = market.get("regime") or {}
            liquidity = market.get("liquidity") or {}
            news = market.get("news") or {}
            momentum = float(features.get("momentum", 0.0) or 0.0)
            trend = abs(float(features.get("trend_strength", 0.0) or 0.0))
            buy = float(flow.get("aggressive_buy_ratio", 0.5) or 0.5)
            sell = 1 - buy
            ret = (current - entry) / entry if side == "long" else (entry - current) / entry
            peak = max(ret, self.position_peaks.get(key, ret))
            self.position_peaks[key] = peak
            state = getattr(self, "_stage6_state", {})
            self._stage6_state = state
            saved = state.get(key, {})
            thesis = saved.get("thesis") or {"side": side, "entry_price": entry, "momentum": momentum, "trend_strength": trend, "opened_at": now.isoformat(), "entry_evidence": market}
            previous_protection = saved.get("protection_price")
            obs = PositionObservation(exchange, psymbol, side, qty, entry, current, ret, peak, 0.5, momentum, trend, buy, sell, float(features.get("volatility_proxy", 0.0) or 0.0), float(features.get("liquidity_stress", 0.0) or liquidity.get("spread_bps", 0.0) / 50.0), float(news.get("risk", features.get("news_risk", 0.0)) or 0.0), float(regime.get("market_risk", 0.0) or 0.0), float(derivatives.get("funding_rate", 0.0) or 0.0) * 100, float(derivatives.get("open_interest_change", 0.0) or 0.0), 0.0, 0.0, now.isoformat())
            # Stage-5 is decision-only; position intelligence remains the final position governor.
            predictive = predictive_model.predict(features)
            try:
                stage5 = Stage5DecisionEngine().evaluate(market_state=market, predictive=predictive, risk_vetoes=[], position_side=side, unrealized_return=ret, thesis_integrity=0.5).as_dict()
            except Exception:
                stage5 = {"action": "WAIT", "vetoes": ["stage5_unavailable"], "expected_value": 0.0}
            decision = POSITION_ENGINE.evaluate(obs, thesis, predictive=predictive, stage5=stage5, previous_protection=previous_protection)

            # Never churn protection. Replace only when the new level tightens materially.
            protection_changed = previous_protection is None or (abs(float(decision.protection_price or 0) - float(previous_protection or 0)) / current >= 0.0005)
            action_result: dict[str, Any] = {"status": "observed", "mode": mode}
            action_key = f"{decision.action}:{round(decision.close_fraction, 3)}"
            cooldown_until = saved.get("cooldown_until")
            in_cooldown = bool(cooldown_until and now.timestamp() < float(cooldown_until))
            if not in_cooldown and mode in {"testnet", "live"} and decision.action in {"exit", "emergency_exit"}:
                pmode = await adapter.get_position_mode() if hasattr(adapter, "get_position_mode") else "ONE_WAY"
                try:
                    action_result = await adapter.close_position(psymbol, side, qty, pmode)
                except Exception as exc:
                    action_result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                self.position_cooldown_until[key] = now.timestamp() + 10
            elif not in_cooldown and mode in {"testnet", "live"} and decision.action == "reduce" and decision.close_fraction > 0:
                pmode = await adapter.get_position_mode() if hasattr(adapter, "get_position_mode") else "ONE_WAY"
                close_qty = min(qty, max(0.001, self._safe_quantity(qty * decision.close_fraction)))
                try:
                    action_result = await adapter.close_position(psymbol, side, close_qty, pmode)
                except Exception as exc:
                    action_result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                self.position_cooldown_until[key] = now.timestamp() + 10
            elif not in_cooldown and mode in {"testnet", "live"} and decision.protection_price is not None and protection_changed:
                pmode = await adapter.get_position_mode() if hasattr(adapter, "get_position_mode") else "ONE_WAY"
                try:
                    action_result = await adapter.update_dynamic_protection(psymbol, side, qty, decision.protection_price, pmode)
                except Exception as exc:
                    action_result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                self.position_cooldown_until[key] = now.timestamp() + 8
            elif mode == "paper":
                # Paper mode executes the same reduce/exit decisions locally and never touches an exchange.
                if decision.action in {"exit", "emergency_exit"}:
                    order = self.paper.close(psymbol, current)
                    action_result = {"status": "closed", "order_id": getattr(order, "order_id", None), "mode": "paper"}
                elif decision.action == "reduce" and decision.close_fraction > 0:
                    p = self.paper.positions.get(psymbol)
                    if p:
                        reduced = min(p.quantity, p.quantity * decision.close_fraction)
                        p.realized_pnl += (current - p.entry_price) * reduced * (1 if p.side == "long" else -1)
                        p.quantity -= reduced
                        action_result = {"status": "reduced", "reduced_quantity": reduced, "remaining_quantity": p.quantity, "mode": "paper"}
            remaining = 0.0
            # Reconcile after every critical action; for paper use the simulator state.
            if mode == "paper":
                p = self.paper.positions.get(psymbol)
                remaining = float(p.quantity) if p else 0.0
            elif decision.action in {"exit", "emergency_exit", "reduce"} and action_result.get("status") != "error":
                try:
                    rr = await adapter.get_positions(psymbol if exchange == "bitget" else None)
                    for rrrow in self._position_rows(rr):
                        ss, sd, qq, _, _ = self._position_fields(rrrow)
                        if ss == psymbol and sd == side:
                            remaining = qq
                            break
                except Exception as exc:
                    action_result["reconciliation_error"] = f"{type(exc).__name__}: {exc}"
                    action_result["status"] = "reconciliation_failed"
            state[key] = {"thesis": thesis, "protection_price": decision.protection_price or previous_protection, "peak_return": peak, "opened_at": thesis.get("opened_at"), "last_action": decision.action, "last_action_key": action_key, "last_review": now.isoformat(), "cooldown_until": now.timestamp() + 8 if decision.action in {"protect", "hold"} else now.timestamp() + 10, "remaining_quantity": remaining, "entry_evidence": thesis.get("entry_evidence"), "last_decision": decision.as_dict()}
            self.stage6_reviews += 1
            self.stage6_last_decision = {"symbol": psymbol, "side": side, **decision.as_dict(), "remaining_quantity": remaining, "execution": action_result}
            rows.append(self.stage6_last_decision)
            try:
                from app.persistence.repository import upsert_position_state, record_event
                await upsert_position_state(exchange, psymbol, side, state[key])
                await record_event("stage6_position_review", {"exchange": exchange, "symbol": psymbol, "side": side, "decision": decision.as_dict(), "stage5": stage5, "execution": action_result, "remaining_quantity": remaining})
            except Exception as exc:
                self.stage6_last_error = f"persistence: {type(exc).__name__}: {exc}"
        self.last_position_management = rows[-20:]
        return rows

    async def stage6_run(self) -> None:
        import asyncio
        while self.running:
            for symbol in self.config.symbols:
                try:
                    await manage(self, symbol)
                except Exception as exc:
                    self.stage6_last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(max(2, self.position_review_interval))

    trader._manage_open_positions = MethodType(manage, trader)
    trader._stage6_original_run = trader._run

    async def run(self):
        import asyncio
        import time
        next_decision = 0.0
        next_management = 0.0
        while self.running:
            now_loop = time.monotonic()
            if now_loop >= next_management:
                for symbol in self.config.symbols:
                    try:
                        await manage(self, symbol)
                    except Exception as exc:
                        self.stage6_last_error = f"{type(exc).__name__}: {exc}"
                next_management = now_loop + max(2, self.position_review_interval)
            if now_loop >= next_decision:
                for symbol in self.config.symbols:
                    try:
                        self.last_cycle = await self.run_cycle(symbol)
                        self.last_cycle_at = datetime.now(timezone.utc)
                        self.last_error = None
                    except Exception as exc:
                        self.last_error = f"{type(exc).__name__}: {exc}"
                next_decision = now_loop + self.config.interval_seconds
            await asyncio.sleep(1)

    trader._run = MethodType(run, trader)

    original_status = trader.status
    def status(self):
        result = original_status()
        result.update({"position_intelligence": self.stage6_version, "position_reviews": self.stage6_reviews, "stage6_last_decision": self.stage6_last_decision, "stage6_last_error": self.stage6_last_error, "position_management_execution_authority": self.execution_mode in {"testnet", "live"} and self._execution_gate()[0]})
        return result
    trader.status = MethodType(status, trader)
