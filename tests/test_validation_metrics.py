from app.ml.validation import evaluate_predictions


def test_validation_metrics_distinguish_predictions_and_trades():
    result = evaluate_predictions([
        (1, 1, 0.02),
        (-1, 1, -0.01),
        (0, 0, 0.0),
    ])

    assert result["predictions"] == 3
    assert result["trades"] == 2
    assert result["total_return"] == 0.01
    assert result["avg_return"] == 0.005
    assert result["class_support"] == {"-1": 1, "0": 1, "1": 1}


def test_empty_validation_is_safe():
    result = evaluate_predictions([])
    assert result["predictions"] == 0
    assert result["trades"] == 0
    assert result["avg_return"] == 0.0
    assert result["balanced_accuracy"] == 0.0
