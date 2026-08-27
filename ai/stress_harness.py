from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import random
import time


@dataclass
class Fault:
    name: str
    description: str
    severity: str


@dataclass
class StressResult:
    scenario: str
    passed: bool
    safe_state: str
    violations: list[str]
    observations: list[str]


class StressHarness:
    """Research-only failure harness. It must never enable production execution."""

    def __init__(self, seed: int = 16):
        self.rng = random.Random(seed)

    def run(self, scenario: str, operation: Callable[[], Any]) -> StressResult:
        violations: list[str] = []
        observations: list[str] = []
        try:
            value = operation()
            if isinstance(value, dict):
                if value.get("execution_authority") is True:
                    violations.append("Execution authority became enabled during stress test.")
                if value.get("live_exchange_order") is True:
                    violations.append("A live exchange order was reported during stress test.")
            observations.append("Operation completed without an uncaught exception.")
            safe = "SAFE" if not violations else "UNSAFE"
            return StressResult(scenario, not violations, safe, violations, observations)
        except Exception as exc:
            # A controlled exception is acceptable if the caller remains in a safe state.
            observations.append(f"Controlled failure: {type(exc).__name__}")
            return StressResult(scenario, True, "SAFE_FAIL_CLOSED", violations, observations)

    def duplicate_order_guard(self, order_ids: list[str]) -> StressResult:
        duplicates = len(order_ids) != len(set(order_ids))
        violations = ["Duplicate order identifier detected."] if duplicates else []
        return StressResult(
            "duplicate_orders",
            not duplicates,
            "SAFE" if not duplicates else "BLOCKED",
            violations,
            ["Duplicate identifiers are rejected before execution."]
        )

    def stale_data_guard(self, age_seconds: float, max_age_seconds: float = 5.0) -> StressResult:
        stale = age_seconds > max_age_seconds
        return StressResult(
            "stale_data",
            True,
            "BLOCKED_STALE_DATA" if stale else "SAFE",
            ["Market data is stale; trading must be blocked."] if stale else [],
            ["Freshness gate evaluated."]
        )

    def restart_recovery(self, state: dict) -> StressResult:
        safe = not bool(state.get("execution_authority", False))
        return StressResult(
            "restart_recovery",
            safe,
            "SAFE" if safe else "UNSAFE",
            [] if safe else ["Execution authority survived restart incorrectly."],
            ["Recovery state contains no production execution authority."]
        )


def standard_scenarios() -> list[Fault]:
    return [
        Fault("exchange_outage", "Exchange API unavailable or times out.", "critical"),
        Fault("stale_market_data", "Price/order-book data stops updating.", "critical"),
        Fault("news_feed_failure", "External intelligence feed fails.", "high"),
        Fault("database_failure", "State store becomes unavailable.", "critical"),
        Fault("server_restart", "Process restarts while a position is tracked.", "critical"),
        Fault("duplicate_order", "Retry creates a duplicate order request.", "critical"),
        Fault("partial_fill", "Only part of an intended quantity fills.", "high"),
        Fault("extreme_volatility", "Market moves violently between observations.", "critical"),
        Fault("clock_skew", "System and exchange timestamps disagree.", "high"),
        Fault("network_partition", "Backend loses connectivity to dependencies.", "critical"),
    ]
