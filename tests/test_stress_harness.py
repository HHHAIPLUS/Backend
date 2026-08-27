from ai.stress_harness import StressHarness, standard_scenarios


def test_standard_failure_matrix_exists():
    names = {x.name for x in standard_scenarios()}
    assert {"exchange_outage", "stale_market_data", "duplicate_order", "server_restart"} <= names


def test_duplicate_order_guard_blocks_duplicates():
    result = StressHarness().duplicate_order_guard(["A", "A"])
    assert result.passed is False
    assert result.safe_state == "BLOCKED"


def test_stale_data_blocks_trading():
    result = StressHarness().stale_data_guard(10, 5)
    assert result.passed is True
    assert result.safe_state == "BLOCKED_STALE_DATA"


def test_restart_recovery_is_fail_closed():
    result = StressHarness().restart_recovery({"execution_authority": False})
    assert result.passed is True
    assert result.safe_state == "SAFE"


def test_stress_run_never_accepts_live_execution():
    result = StressHarness().run(
        "live_authority_probe",
        lambda: {"execution_authority": False, "live_exchange_order": False},
    )
    assert result.passed is True
