from ai.paper_trading import PaperExecutionEngine, PaperSession, TradingMode


def test_paper_engine_never_has_live_execution_authority():
    engine = PaperExecutionEngine()
    order = engine.submit("BTCUSDT", "buy", 0.01, 100000)
    assert order.status == "simulated"
    assert engine.snapshot()["execution_authority"] is False


def test_session_must_be_started_before_order():
    session = PaperSession("test-session", TradingMode.PAPER)
    assert session.status()["real_money"] is False
    session.start()
    assert session.status()["running"] is True


def test_mark_updates_unrealized_pnl():
    engine = PaperExecutionEngine()
    engine.submit("BTCUSDT", "buy", 1, 100)
    position = engine.mark("BTCUSDT", 110)
    assert position is not None
    assert position.unrealized_pnl == 10
