from __future__ import annotations
from collections import deque
from ai.agents import CouncilDecision

class AgentMemory:
    """Small in-process decision journal for Phase 8. Persistent storage comes later."""
    def __init__(self, max_items: int = 200):
        self._items = deque(maxlen=max_items)

    def record(self, decision: CouncilDecision) -> None:
        self._items.append(decision)

    def recent(self, limit: int = 20) -> list[CouncilDecision]:
        return list(self._items)[-limit:][::-1]
