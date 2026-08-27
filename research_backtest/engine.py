from dataclasses import dataclass
from typing import Iterable

@dataclass
class Trade:
    pnl: float
    fees: float = 0.0
    funding: float = 0.0

@dataclass
class BacktestResult:
    trades: int
    net_pnl: float
    wins: int
    losses: int
    win_rate: float
    max_drawdown: float

def run(trades: Iterable[Trade]) -> BacktestResult:
    equity = peak = 0.0
    max_drawdown = wins = losses = 0
    items = list(trades)
    for t in items:
        net = t.pnl - t.fees - t.funding
        equity += net
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak-equity)
        wins += net > 0
        losses += net < 0
    n=len(items)
    return BacktestResult(n, equity, wins, losses, wins/n if n else 0.0, max_drawdown)
