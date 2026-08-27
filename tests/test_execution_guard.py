import pytest
from app.services.execution_guard import ExecutionGuard

def test_guard_blocks_by_default(monkeypatch):
    monkeypatch.setenv('LIVE_TRADING_ENABLED','false')
    with pytest.raises(RuntimeError): ExecutionGuard().assert_allowed()
