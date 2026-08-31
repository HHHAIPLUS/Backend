from __future__ import annotations

import importlib
from pathlib import Path


def _project_modules() -> list[str]:
    modules: list[str] = []
    for root in (Path("app"), Path("ai"), Path("research_backtest")):
        for path in root.rglob("*.py"):
            if path.name.startswith("test_"):
                continue
            module = ".".join(path.with_suffix("").parts)
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            modules.append(module)
    return sorted(set(modules))


def test_all_project_modules_import():
    failures: list[str] = []
    for module in _project_modules():
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - failure message is the assertion
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    assert not failures, "Project import smoke test failed:\n" + "\n".join(failures)


def test_model_feature_compatibility_contract():
    from app.ml.features import build_model_features
    from app.ml.predictive import FEATURES

    features = build_model_features(
        [
            {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 10.0},
            {"open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 12.0},
        ],
        {"order_book_imbalance": 0.2, "news_risk": 0.1},
    )
    assert list(features) == FEATURES
    assert all(isinstance(value, float) for value in features.values())


def test_legacy_decision_proposal_imports():
    from ai.decision import DecisionEngine
    from app.trading.models import Side, TradeProposal

    assert Side.LONG.value == "long"
    assert TradeProposal is not None
    assert DecisionEngine is not None
