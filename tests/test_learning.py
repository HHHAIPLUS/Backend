from ai.self_learning import ControlledLearningEngine


def test_learning_records_outcome_without_self_modifying_production():
    e = ControlledLearningEngine()
    row = e.record_decision(symbol="BTCUSDT", action="hold", thesis="trend intact", features={"momentum": .7}, confidence=.8)
    e.record_outcome(row.record_id, .012)
    s = e.status()
    assert s["completed_outcomes"] == 1
    assert s["production_self_modification"] is False
    assert s["execution_authority"] is False


def test_candidate_stays_quarantined_until_evaluation():
    e = ControlledLearningEngine()
    c = e.propose_candidate(base_model="production", proposed_change="raise confirmation threshold", reason="false positives")
    assert c.status == "quarantined"
