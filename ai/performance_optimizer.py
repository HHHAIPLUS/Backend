from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    side: str
    entry: float
    exit: float
    highest_favorable: float
    lowest_adverse: float
    expected_direction: str
    actual_direction: str
    pnl: float
    fees: float = 0.0
    reason: str = ""


class PerformanceOptimizer:
    """Post-trade intelligence. It evaluates decisions; it never executes trades."""

    @staticmethod
    def exit_quality(t: TradeRecord) -> float:
        if t.side.lower() == "long":
            best = t.highest_favorable - t.entry
            captured = t.exit - t.entry
        else:
            best = t.entry - t.lowest_adverse
            captured = t.entry - t.exit

        if best <= 0:
            return 1.0 if captured <= 0 else 0.0
        return max(0.0, min(1.0, captured / best))

    @staticmethod
    def missed_opportunity(t: TradeRecord) -> float:
        if t.side.lower() == "long":
            potential = t.highest_favorable - t.entry
        else:
            potential = t.entry - t.lowest_adverse
        if potential <= 0:
            return 0.0
        captured = t.pnl + t.fees
        return max(0.0, potential - max(captured, 0.0))

    @staticmethod
    def classify_signal(t: TradeRecord) -> str:
        expected = t.expected_direction.lower()
        actual = t.actual_direction.lower()
        if expected != actual:
            return "false_signal"
        if t.pnl > 0:
            return "correct"
        return "correct_direction_lost_trade"

    @staticmethod
    def profit_factor(trades: Iterable[TradeRecord]) -> float:
        wins = sum(max(0.0, t.pnl - t.fees) for t in trades)
        losses = sum(max(0.0, -(t.pnl - t.fees)) for t in trades)
        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return wins / losses

    def analyze(self, trades: list[TradeRecord]) -> dict:
        if not trades:
            return {
                "trade_count": 0,
                "profit_factor": 0.0,
                "net_pnl": 0.0,
                "average_exit_quality": 0.0,
                "false_signal_rate": 0.0,
                "missed_opportunity_total": 0.0,
                "performance_attribution": {},
            }

        exit_scores = [self.exit_quality(t) for t in trades]
        false_signals = [self.classify_signal(t) == "false_signal" for t in trades]
        net = sum(t.pnl - t.fees for t in trades)

        attribution: dict[str, dict[str, float]] = {}
        for t in trades:
            bucket = t.reason or "unspecified"
            item = attribution.setdefault(bucket, {"trades": 0.0, "net_pnl": 0.0, "wins": 0.0})
            item["trades"] += 1
            item["net_pnl"] += t.pnl - t.fees
            item["wins"] += 1 if t.pnl - t.fees > 0 else 0

        return {
            "trade_count": len(trades),
            "profit_factor": self.profit_factor(trades),
            "net_pnl": net,
            "average_exit_quality": sum(exit_scores) / len(exit_scores),
            "false_signal_rate": sum(false_signals) / len(trades),
            "missed_opportunity_total": sum(self.missed_opportunity(t) for t in trades),
            "performance_attribution": attribution,
            "execution_authority": False,
        }
