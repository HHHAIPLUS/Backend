import os

import pytest

from ai.autonomous_trader import AutonomousTrader


def test_default_execution_is_fail_closed(monkeypatch):
    monkeypatch.delenv("HHHAI_TRADING_MODE", raising=False)
    monkeypatch.setenv("HHHAI_AUTOTRADING_ENABLED", "false")
    monkeypatch.setattr("ai.autonomous_trader.settings.live_trading_enabled", False)
    monkeypatch.setattr("ai.autonomous_trader.settings.testnet_trading_enabled", False)
    t = AutonomousTrader()
    assert t.execution_mode == "paper"
    assert t._execution_gate()[0] is True
    assert t.status()["execution_authority"] is False


def test_testnet_requires_explicit_gate(monkeypatch):
    monkeypatch.setenv("HHHAI_TRADING_MODE", "testnet")
    monkeypatch.setenv("HHHAI_AUTOTRADING_ENABLED", "true")
    monkeypatch.setattr("ai.autonomous_trader.settings.testnet_trading_enabled", False)
    t = AutonomousTrader()
    allowed, reason = t._execution_gate()
    assert not allowed
    assert "TESTNET_TRADING_ENABLED" in reason


def test_live_requires_explicit_live_gate(monkeypatch):
    monkeypatch.setenv("HHHAI_TRADING_MODE", "live")
    monkeypatch.setenv("HHHAI_AUTOTRADING_ENABLED", "true")
    monkeypatch.setattr("ai.autonomous_trader.settings.live_trading_enabled", False)
    t = AutonomousTrader()
    assert t._execution_gate()[0] is False


def test_live_mode_does_not_require_testnet_flag(monkeypatch):
    monkeypatch.setenv("HHHAI_TRADING_MODE", "live")
    monkeypatch.setenv("HHHAI_AUTOTRADING_ENABLED", "true")
    monkeypatch.setattr("ai.autonomous_trader.settings.live_trading_enabled", True)
    monkeypatch.setattr("ai.autonomous_trader.settings.testnet_trading_enabled", False)
    t = AutonomousTrader()
    assert t._execution_gate() == (True, "live execution enabled")


def test_equity_state_tracks_daily_loss_and_drawdown():
    t = AutonomousTrader()
    t._update_equity_state(10000)
    t._update_equity_state(9000)
    assert t._day_start_equity == 10000
    assert t._peak_equity == 10000
