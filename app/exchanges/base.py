from abc import ABC, abstractmethod
from typing import Any

class ExchangeAdapter(ABC):
    name: str

    @abstractmethod
    async def get_account_status(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def place_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
