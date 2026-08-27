from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class TradingMode(str, Enum):
    PAPER = "paper"
    TESTNET = "testnet"


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: str = "simulated"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PaperPosition:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    realized_pnl: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        direction = 1 if self.side.lower() == "long" else -1
        return (self.mark_price - self.entry_price) * self.quantity * direction


class PaperExecutionEngine:
    """Execution simulator. Never sends orders to a production exchange."""

    def __init__(self, mode: TradingMode = TradingMode.PAPER):
        self.mode = mode
        self.orders: list[PaperOrder] = []
        self.positions: dict[str, PaperPosition] = {}
        self._counter = 0

    def submit(self, symbol: str, side: str, quantity: float, price: float) -> PaperOrder:
        if quantity <= 0 or price <= 0:
            raise ValueError("quantity and price must be positive")

        self._counter += 1
        order = PaperOrder(
            order_id=f"PAPER-{self._counter:06d}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
        )
        self.orders.append(order)

        position_side = "long" if side.lower() == "buy" else "short"
        existing = self.positions.get(symbol)
        if existing is None:
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                side=position_side,
                quantity=quantity,
                entry_price=price,
                mark_price=price,
            )
        else:
            existing.mark_price = price

        return order

    def mark(self, symbol: str, price: float) -> Optional[PaperPosition]:
        position = self.positions.get(symbol)
        if position:
            position.mark_price = price
        return position

    def close(self, symbol: str, price: float) -> Optional[PaperOrder]:
        position = self.positions.get(symbol)
        if not position:
            return None
        side = "sell" if position.side == "long" else "buy"
        pnl = position.unrealized_pnl if price == position.mark_price else (
            (price - position.entry_price)
            * position.quantity
            * (1 if position.side == "long" else -1)
        )
        position.realized_pnl += pnl
        order = self.submit(symbol, side, position.quantity, price)
        del self.positions[symbol]
        return order

    def snapshot(self) -> dict:
        return {
            "mode": self.mode.value,
            "execution_authority": False,
            "orders": len(self.orders),
            "open_positions": len(self.positions),
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "mark_price": p.mark_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "realized_pnl": p.realized_pnl,
                }
                for p in self.positions.values()
            ],
        }


class PaperSession:
    def __init__(self, session_id: str, mode: TradingMode = TradingMode.PAPER):
        self.session_id = session_id
        self.mode = mode
        self.started_at = datetime.now(timezone.utc)
        self.running = False
        self.execution = PaperExecutionEngine(mode)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def status(self) -> dict:
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "running": self.running,
            "started_at": self.started_at.isoformat(),
            "execution_authority": False,
            "real_money": False,
        }
