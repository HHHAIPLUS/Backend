# HHHAI model training

`train_model.py` is the only supported path for producing a production predictive-model artifact.

Input CSV must contain `observed_at`, `label` (-1/0/1), `outcome_return`, plus the feature columns listed in `app/ml/predictive.py`.

The script performs chronological walk-forward validation. It refuses promotion when accuracy, balanced accuracy, or average outcome return fails the configured minimum gates.

No model artifact is shipped with the repository. This prevents HHHAI from treating an unvalidated or stale model as production intelligence.
