from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math, random

@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class SimulationConfig:
    fee_rate: float = 0.0004
    slippage_bps: float = 2.0
    funding_rate: float = 0.0001
    starting_equity: float = 10_000.0
    leverage: float = 1.0
    seed: int = 13

@dataclass
class SimulationResult:
    final_equity: float
    total_return: float
    max_drawdown: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    fees: float
    funding_cost: float
    liquidations: int

def _slip(price: float, side: str, bps: float) -> float:
    m = bps / 10_000
    return price * (1 + m if side == "buy" else 1 - m)

def replay(
    candles: Iterable[Candle],
    signals: Iterable[int],
    config: SimulationConfig | None = None,
) -> SimulationResult:
    cfg = config or SimulationConfig()
    equity = cfg.starting_equity
    peak = equity
    max_dd = 0.0
    fees = funding = 0.0
    trades = wins = losses = liquidations = 0
    position = 0
    entry = 0.0
    rng = random.Random(cfg.seed)

    for c, signal in zip(candles, signals):
        if position == 0 and signal != 0:
            side = "buy" if signal > 0 else "sell"
            entry = _slip(c.close, side, cfg.slippage_bps)
            fees += equity * cfg.fee_rate
            equity -= equity * cfg.fee_rate
            position = 1 if signal > 0 else -1
            trades += 1
        elif position != 0:
            # Funding is charged while a leveraged position is open.
            funding_charge = abs(equity * cfg.leverage) * cfg.funding_rate
            equity -= funding_charge
            funding += funding_charge

            if signal == -position:
                exit_side = "sell" if position > 0 else "buy"
                exit_price = _slip(c.close, exit_side, cfg.slippage_bps)
                pnl_pct = (exit_price - entry) / entry * position * cfg.leverage
                equity *= max(0.0, 1.0 + pnl_pct)
                fees += equity * cfg.fee_rate
                equity -= equity * cfg.fee_rate
                if equity <= 0:
                    liquidations += 1
                    equity = 0.0
                elif pnl_pct > 0:
                    wins += 1
                else:
                    losses += 1
                position = 0

        peak = max(peak, equity)
        dd = 0 if peak == 0 else (peak - equity) / peak
        max_dd = max(max_dd, dd)

    total_return = 0 if cfg.starting_equity == 0 else equity / cfg.starting_equity - 1
    return SimulationResult(
        final_equity=equity,
        total_return=total_return,
        max_drawdown=max_dd,
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=(wins / (wins + losses)) if wins + losses else 0.0,
        fees=fees,
        funding_cost=funding,
        liquidations=liquidations,
    )

def walk_forward_slices(candles: list[Candle], train_size: int, test_size: int):
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    start = 0
    while start + train_size + test_size <= len(candles):
        yield (
            candles[start:start + train_size],
            candles[start + train_size:start + train_size + test_size],
        )
        start += test_size

def monte_carlo_returns(returns: list[float], paths: int = 1000, seed: int = 13):
    if not returns:
        return []
    rng = random.Random(seed)
    result = []
    for _ in range(paths):
        equity = 1.0
        for r in (rng.choice(returns) for _ in returns):
            equity *= max(0.0, 1.0 + r)
        result.append(equity - 1.0)
    return result
