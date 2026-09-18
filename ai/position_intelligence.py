from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from types import MethodType
from typing import Any

from ai.cognitive_exit import CognitiveExitEngine, ExitAction, PositionTelemetry
from ai.counterfactual import CounterfactualTradeTwin
from ai.stage5_engine import Stage5DecisionEngine
from app.market_data.binance_central import CentralBinanceMarketData
from app.market_data.binance_user_stream import binance_user_stream
from app.market_data.realtime import build_world_intelligence
from app.ml.adaptive_intelligence import adaptive_intelligence
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
    take_profit_price: float | None
    reason: str
    evidence: dict[str, Any]

    def as_dict(self):
        return asdict(self)


class PositionIntelligenceEngine:
    VERSION = "stage6-position-intelligence-v2"

    @staticmethod
    def clamp(x, lo=0.0, hi=1.0):
        return max(lo, min(hi, float(x)))

    def thesis_integrity(self, p: PositionObservation, thesis: dict[str, Any]) -> float:
        d = 1.0 if p.side.lower() == "long" else -1.0
        entry_m = float(thesis.get("momentum", 0) or 0)
        pressure = p.buying_pressure if d > 0 else p.selling_pressure
        alignment = self.clamp(0.5 + d * p.momentum * 0.5)
        original = self.clamp(0.5 + d * entry_m * 0.5)
        deterioration = max(0, -d * (p.momentum - entry_m))
        return self.clamp(
            0.22 * alignment + 0.16 * original + 0.22 * self.clamp(p.trend_strength)
            + 0.20 * self.clamp(pressure) + 0.10 * (1 - self.clamp(p.news_risk))
            + 0.10 * (1 - self.clamp(p.liquidity_stress)) - 0.20 * deterioration
        )

    def evaluate(
        self,
        p: PositionObservation,
        thesis: dict[str, Any],
        *,
        predictive=None,
        stage5=None,
        previous_protection=None,
        current_take_profit=None,
    ) -> PositionDecision:
        if not all(isfinite(float(x)) and float(x) > 0 for x in (p.entry_price, p.current_price, p.quantity)):
            return PositionDecision("hold", 0, 0, 0, 1, 1, previous_protection, current_take_profit, "Invalid position telemetry; fail closed.", {"invalid_telemetry": True})

        predictive = predictive or {}
        stage5 = stage5 or {}
        integrity = self.thesis_integrity(p, thesis)
        probs = predictive.get("probabilities") or {}
        dp = float(probs.get("long", 0) if p.side.lower() == "long" else probs.get("short", 0))
        expected = float(stage5.get("expected_value", p.expected_continuation_value) or 0) + (dp - 0.5) * 0.04
        expected = max(-0.20, min(0.20, expected))
        adverse_m = max(0, -(p.momentum if p.side.lower() == "long" else -p.momentum))
        opposing = p.selling_pressure if p.side.lower() == "long" else p.buying_pressure
        retrace = self.clamp((p.peak_return - p.unrealized_return) / p.peak_return) if p.peak_return > 0 else 0
        downside = self.clamp(
            0.24 * p.volatility + 0.18 * p.liquidity_stress + 0.18 * p.news_risk
            + 0.14 * p.market_risk + 0.14 * adverse_m + 0.12 * opposing
            + 0.10 * max(0, -p.unrealized_return * 12)
        )
        shock = self.clamp(0.30 * p.news_risk + 0.25 * p.market_risk + 0.25 * p.liquidity_stress + 0.20 * adverse_m)
        edge = expected - downside * 0.025
        thesis_break = integrity < 0.38

        if shock >= 0.86 or p.liquidity_stress >= 0.94:
            action, frac, reason = "emergency_exit", 1.0, "Market-wide or liquidity shock makes continued exposure unsafe."
        elif thesis_break and p.unrealized_return <= 0:
            action, frac, reason = "exit", 1.0, "The original trade thesis has materially failed while the position is losing."
        elif thesis_break or downside >= 0.78:
            action, frac, reason = (("exit", 1.0) if p.unrealized_return > 0.01 else ("reduce", 0.50)), "Thesis integrity or downside risk has deteriorated beyond continuation value."
        elif p.unrealized_return > 0 and (retrace >= 0.35 or edge < -0.005 or shock >= 0.55):
            action, frac, reason = "reduce", 0.35, "Profit is available but continuation value has deteriorated; bank part of the gain."
        elif p.unrealized_return > 0 and (retrace >= 0.18 or downside >= 0.45):
            action, frac, reason = "protect", 0.0, "The thesis remains viable, but protection should tighten as risk rises."
        else:
            action, frac, reason = "hold", 0.0, "Continuation evidence remains stronger than current downside and shock risk."

        protection = None
        if action in {"hold", "protect", "reduce"} and p.unrealized_return > 0:
            buf = self.clamp(0.001 + p.volatility * 0.006 + max(downside, shock) * 0.003, 0.001, 0.018)
            proposed = p.current_price * (1 - buf) if p.side.lower() == "long" else p.current_price * (1 + buf)
            if previous_protection is None:
                protection = proposed
            elif p.side.lower() == "long":
                protection = max(float(previous_protection), proposed)
            else:
                protection = min(float(previous_protection), proposed)

        # A target is extended only when the predictive direction, trend and
        # continuation evidence agree. This never weakens the protective stop.
        target = current_take_profit
        direction_ok = (p.side.lower() == "long" and dp >= 0.62) or (p.side.lower() == "short" and dp >= 0.62)
        if action in {"hold", "protect", "reduce"} and p.unrealized_return > 0 and direction_ok and p.trend_strength >= 0.35 and expected > 0:
            extension = max(0.004, p.volatility * 0.02 + p.trend_strength * 0.01)
            proposed_target = p.current_price * (1 + extension) if p.side.lower() == "long" else p.current_price * (1 - extension)
            if target is None:
                target = proposed_target
            elif p.side.lower() == "long":
                target = max(float(target), proposed_target)
            else:
                target = min(float(target), proposed_target)

        return PositionDecision(
            action, frac, integrity, expected, downside, shock, protection, target, reason,
            {
                "predictive_probability": dp,
                "retracement_from_peak": retrace,
                "edge_after_risk": edge,
                "stage5_action": stage5.get("action"),
                "stage5_vetoes": stage5.get("vetoes", []),
                "thesis": thesis,
            },
        )


POSITION_ENGINE = PositionIntelligenceEngine()
COGNITIVE_EXIT = CognitiveExitEngine()
COUNTERFACTUAL = CounterfactualTradeTwin()


def _current_protection(symbol: str) -> tuple[float | None, float | None]:
    stop = None
    target = None
    for order in binance_user_stream.protection_orders(symbol):
        order_type = str(order.get("o") or order.get("type") or "").upper()
        price = float(order.get("sp") or order.get("stopPrice") or 0)
        if price <= 0:
            continue
        if "TAKE_PROFIT" in order_type:
            target = price
        elif "STOP" in order_type:
            stop = price
    return stop, target


def install_stage6_position_intelligence(trader: Any) -> None:
    if getattr(trader, "_stage6_installed", False):
        return

    trader._stage6_installed = True
    trader.stage6_version = POSITION_ENGINE.VERSION
    trader.stage6_reviews = 0
    trader.stage6_last_decision = None
    trader.stage6_last_error = None
    trader._stage6_state = {}

    async def manage(self, symbol, rows=None):
        mode = self.execution_mode
        if mode in {"testnet", "live"} and not self._execution_gate()[0]:
            return []

        exchange = "paper" if mode == "paper" else self._exchange_for_market()
        adapter = None
        if mode != "paper":
            if exchange == "binance":
                if not binance_user_stream.is_healthy():
                    self.stage6_last_error = "Binance user-data stream is not healthy; position management is fail-closed."
                    return []
                adapter = self._adapters()[exchange]
                raw_rows = binance_user_stream.positions(symbol)
            else:
                adapter = self._adapters()[exchange]
                if rows is not None:
                    raw_rows = rows
                else:
                    try:
                        raw_rows = await adapter.get_positions(symbol if exchange == "bitget" else None)
                    except TypeError:
                        raw_rows = await adapter.get_positions()
                    except Exception as exc:
                        self.stage6_last_error = f"Position sync failed: {type(exc).__name__}: {exc}"
                        return []
        else:
            raw_rows = [
                {"symbol": p.symbol, "holdSide": p.side, "positionAmt": p.quantity, "entryPrice": p.entry_price, "markPrice": p.mark_price}
                for p in self.paper.positions.values() if p.symbol.upper() == symbol.upper()
            ]

        results = []
        now = datetime.now(timezone.utc)
        for row in raw_rows:
            ps, side, qty, entry, _ = self._position_fields(row)
            if ps != symbol.upper() or qty <= 0 or entry <= 0:
                continue

            key = f"{exchange}:{ps}:{side}"
            try:
                world = await __import__("asyncio").to_thread(build_world_intelligence, ps, exchange)
                world_dict = world.model_dump(mode="json")
                market = world_dict.get("market") or {}
                current = float(market.get("price") or 0)
                if mode == "paper":
                    current = float(row.get("markPrice") or current)
                    self.paper.mark(ps, current)
                if current <= 0:
                    continue

                features = CentralBinanceMarketData.model_features(ps) if exchange == "binance" else dict(market.get("features") or {})
                market["features"] = features
                world_dict["market"] = market
                imbalance = float(market.get("order_book_imbalance") or 0)
                buying = max(0.0, min(1.0, 0.5 + imbalance / 2))
                selling = max(0.0, min(1.0, 0.5 - imbalance / 2))
                direction = 1 if side == "long" else -1
                ret = (current - entry) / entry if side == "long" else (entry - current) / entry
                peak = max(ret, self.position_peaks.get(key, ret))
                self.position_peaks[key] = peak
                saved = self._stage6_state.get(key, {})
                thesis = saved.get("thesis") or {
                    "side": side,
                    "entry_price": entry,
                    "momentum": float(features.get("momentum", 0) or 0),
                    "trend_strength": abs(float(features.get("trend_strength", 0) or 0)),
                    "opened_at": now.isoformat(),
                    "entry_evidence": market,
                }

                predictive_input = {
                    **features,
                    "symbol": ps,
                    "order_book_imbalance": imbalance,
                    "funding_rate": float(market.get("funding_rate") or 0),
                    "news_risk": float(world_dict.get("news_risk") or 0),
                    "news_sentiment": float(world_dict.get("news_sentiment") or 0),
                    "liquidity_stress": float(world_dict.get("liquidity_stress") or 0),
                }
                pred = predictive_model.predict(predictive_input)
                try:
                    stage5 = Stage5DecisionEngine(adaptive_intelligence).evaluate(
                        market_state=world_dict,
                        predictive=pred,
                        risk_vetoes=[],
                        position_side=side,
                        unrealized_return=ret,
                        thesis_integrity=saved.get("thesis_integrity", 0.5),
                    ).__dict__
                except Exception:
                    stage5 = {"action": "WAIT", "vetoes": ["stage5_unavailable"], "expected_value": 0}

                p = PositionObservation(
                    exchange, ps, side, qty, entry, current, ret, peak, 0.5,
                    float(features.get("momentum", 0) or 0),
                    abs(float(features.get("trend_strength", 0) or 0)),
                    buying, selling,
                    float(features.get("volatility_proxy", 0) or 0),
                    float(world_dict.get("liquidity_stress") or 0),
                    float(world_dict.get("news_risk") or 0),
                    float(world_dict.get("market_risk") or 0),
                    float(market.get("funding_rate") or 0) * 100,
                    float(market.get("open_interest_change") or 0),
                    0, 0, now.isoformat(),
                )
                d = POSITION_ENGINE.evaluate(
                    p, thesis, predictive=pred, stage5=stage5,
                    previous_protection=saved.get("protection_price"),
                    current_take_profit=saved.get("take_profit_price"),
                )

                telemetry = PositionTelemetry(
                    side=side, entry_price=entry, current_price=current,
                    unrealized_return=ret, peak_return=peak,
                    minutes_open=max(0.0, (now - datetime.fromisoformat(thesis["opened_at"])).total_seconds() / 60),
                    momentum=p.momentum, trend_strength=p.trend_strength,
                    buying_pressure=buying, selling_pressure=selling,
                    volatility=p.volatility, liquidity_stress=p.liquidity_stress,
                    news_risk=p.news_risk, thesis_integrity=d.thesis_integrity,
                    funding_bias=p.funding_bias, open_interest_change=p.open_interest_change,
                )
                cognitive = COGNITIVE_EXIT.evaluate(telemetry)
                twin = COUNTERFACTUAL.evaluate(telemetry, cognitive)

                # Combine the existing Stage-6 engine, cognitive exit governor and
                # counterfactual twin. Emergency exits always win; otherwise a
                # stronger independent exit can override a HOLD.
                if cognitive.action in {ExitAction.EMERGENCY_EXIT, ExitAction.EXIT} and d.action in {"hold", "protect"}:
                    d.action = "emergency_exit" if cognitive.action == ExitAction.EMERGENCY_EXIT else "exit"
                    d.close_fraction = 1.0
                    d.reason = cognitive.reason
                elif d.action == "hold" and twin.selected in {ExitAction.REDUCE, ExitAction.EXIT}:
                    d.action = "reduce" if twin.selected == ExitAction.REDUCE else "exit"
                    d.close_fraction = 0.35 if d.action == "reduce" else 1.0
                    d.reason += " Counterfactual review independently found weaker continuation value."
                if cognitive.protection_price is not None and d.protection_price is not None:
                    d.protection_price = max(d.protection_price, cognitive.protection_price) if side == "long" else min(d.protection_price, cognitive.protection_price)

                old_stop, old_target = _current_protection(ps) if exchange == "binance" else (None, None)
                if d.protection_price is None:
                    d.protection_price = old_stop
                if d.take_profit_price is None:
                    d.take_profit_price = old_target

                cooldown = float(saved.get("cooldown_until", 0) or 0)
                action: dict[str, Any] = {"status": "observed", "mode": mode}
                if now.timestamp() >= cooldown and mode in {"testnet", "live"} and d.action in {"exit", "emergency_exit"}:
                    pm = await adapter.get_position_mode()
                    try:
                        action = await adapter.close_position(ps, side, qty, pm)
                    except Exception as exc:
                        action = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                elif now.timestamp() >= cooldown and mode in {"testnet", "live"} and d.action == "reduce":
                    pm = await adapter.get_position_mode()
                    close_qty = min(qty, max(0.001, self._safe_quantity(qty * d.close_fraction)))
                    try:
                        action = await adapter.close_position(ps, side, close_qty, pm)
                    except Exception as exc:
                        action = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                elif now.timestamp() >= cooldown and mode in {"testnet", "live"} and d.protection_price is not None:
                    stop_changed = old_stop is None or abs(d.protection_price - old_stop) / current >= 0.0005
                    target_changed = d.take_profit_price is not None and (old_target is None or abs(d.take_profit_price - old_target) / current >= 0.001)
                    if stop_changed or target_changed:
                        pm = await adapter.get_position_mode()
                        try:
                            action = await adapter.update_dynamic_protection(
                                ps, side, qty, d.protection_price, pm, take_profit=d.take_profit_price
                            )
                        except Exception as exc:
                            action = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                elif mode == "paper":
                    if d.action in {"exit", "emergency_exit"}:
                        o = self.paper.close(ps, current)
                        action = {"status": "closed", "order_id": getattr(o, "order_id", None), "mode": "paper"}

                state = {
                    "thesis": thesis,
                    "thesis_integrity": d.thesis_integrity,
                    "protection_price": d.protection_price,
                    "take_profit_price": d.take_profit_price,
                    "peak_return": peak,
                    "opened_at": thesis.get("opened_at"),
                    "last_action": d.action,
                    "last_review": now.isoformat(),
                    "cooldown_until": now.timestamp() + (10 if d.action in {"exit", "emergency_exit", "reduce"} else 2),
                    "entry_evidence": thesis.get("entry_evidence"),
                    "last_decision": d.as_dict(),
                    "cognitive_exit": cognitive.__dict__,
                    "counterfactual": twin.__dict__,
                }
                self._stage6_state[key] = state
                self.stage6_reviews += 1
                self.stage6_last_decision = {"symbol": ps, "side": side, **d.as_dict(), "execution": action, "cognitive_exit": cognitive.__dict__, "counterfactual": twin.__dict__}
                results.append(self.stage6_last_decision)

                try:
                    from app.persistence.repository import upsert_position_state, record_event
                    await upsert_position_state(exchange, ps, side, state)
                    await record_event("stage6_position_review", {"exchange": exchange, "symbol": ps, "side": side, "decision": d.as_dict(), "cognitive_exit": cognitive.__dict__, "counterfactual": twin.__dict__, "execution": action})
                except Exception as exc:
                    self.stage6_last_error = f"persistence: {type(exc).__name__}: {exc}"
            except Exception as exc:
                self.stage6_last_error = f"{type(exc).__name__}: {exc}"

        self.last_position_management = results[-20:]
        return results

    async def run(self):
        import asyncio
        import time
        nd = nm = 0.0
        while self.running:
            now = time.monotonic()
            if now >= nm:
                for s in self.config.symbols:
                    try:
                        await manage(self, s)
                    except Exception as exc:
                        self.stage6_last_error = f"{type(exc).__name__}: {exc}"
                nm = now + max(2, self.position_review_interval)
            if now >= nd:
                # Entry scanning remains on the normal slower cadence. Position
                # management is deliberately independent and faster.
                for s in self.config.symbols:
                    try:
                        self.last_cycle = await self.run_cycle(s)
                        self.last_cycle_at = datetime.now(timezone.utc)
                        self.last_error = None
                    except Exception as exc:
                        self.last_error = f"{type(exc).__name__}: {exc}"
                nd = now + self.config.interval_seconds
            await asyncio.sleep(1)

    async def _manage_non_binance(self, symbol: str, exchange: str):
        # Preserve the existing adapter path for Bitget; the Binance redesign
        # must not break a non-Binance exchange integration.
        adapter = self._adapters()[exchange]
        try:
            raw = await adapter.get_positions(symbol if exchange == "bitget" else None)
        except TypeError:
            raw = await adapter.get_positions()
        except Exception:
            return []
        return []

    trader._manage_open_positions = MethodType(manage, trader)
    trader._run = MethodType(run, trader)

    original_status = trader.status

    def status(self):
        result = original_status()
        result.update({
            "position_intelligence": self.stage6_version,
            "position_reviews": self.stage6_reviews,
            "stage6_last_decision": self.stage6_last_decision,
            "stage6_last_error": self.stage6_last_error,
            "position_management_execution_authority": self.execution_mode in {"testnet", "live"} and self._execution_gate()[0],
            "binance_user_stream": binance_user_stream.health(),
        })
        return result

    trader.status = MethodType(status, trader)
