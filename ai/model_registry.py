from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: str
    created_at: datetime
    status: str


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ModelVersion] = {}

    def register(self, name: str, version: str, status: str = "candidate") -> ModelVersion:
        model = ModelVersion(
            name=name,
            version=version,
            created_at=datetime.now(timezone.utc),
            status=status,
        )
        self._models[f"{name}:{version}"] = model
        return model

    def get(self, name: str, version: str) -> ModelVersion | None:
        return self._models.get(f"{name}:{version}")
