from __future__ import annotations
from typing import Any

async def close_position(adapter, symbol: str, side: str, quantity: float, position_mode: str = 'ONE_WAY') -> dict[str, Any]:
    if quantity <= 0:
        raise ValueError('quantity must be positive')
    return await adapter.close_position(symbol, side, quantity, position_mode)
