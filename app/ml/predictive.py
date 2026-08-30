from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "return_1",
    "range_pct",
    "volume_change",
    "order_book_imbalance",
    "funding_rate",
    "open_interest_change",
    "news_risk",
    "news_sentiment",
    "volatility_proxy",
    "trend_strength",
    "momentum",
    "liquidity_stress",
]


@dataclass
class ModelReport:
    trained: bool
    version: str
    metrics: dict
    reason: str


class PredictiveModel:
    def __init__(
        self,
        artifact_dir: str = "artifacts",
    ):
        self.path = Path(artifact_dir)

        self.path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model_path = (
            self.path / "direction_model.json"
        )

        self.model = None
        self.version = "untrained"

        self._load()

    def _load(self) -> None:
        """
        Load a previously promoted compact model artifact.

        Older artifacts containing X/y are also supported so an
        existing deployment does not crash during startup.
        """

        if not self.model_path.exists():
            return

        try:
            data = json.loads(
                self.model_path.read_text()
            )

            # -----------------------------------------------------
            # New compact artifact format.
            # -----------------------------------------------------
            if all(
                key in data
                for key in (
                    "version",
                    "coef",
                    "intercept",
                    "classes",
                    "mean",
                    "scale",
                )
            ):
                self.load_compact_artifact(data)
                return

            # -----------------------------------------------------
            # Backwards compatibility with the old artifact format.
            # -----------------------------------------------------
            if "X" in data and "y" in data:
                X = np.asarray(
                    data["X"],
                    dtype=float,
                )

                y = np.asarray(
                    data["y"],
                    dtype=int,
                )

                if (
                    X.ndim != 2
                    or X.shape[1] != len(FEATURES)
                    or len(X) != len(y)
                    or len(set(y.tolist())) < 3
                ):
                    raise ValueError(
                        "Stored model artifact contains invalid "
                        "training data."
                    )

                model = Pipeline(
                    [
                        (
                            "scale",
                            StandardScaler(),
                        ),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=500,
                            ),
                        ),
                    ]
                )

                model.fit(X, y)

                self.model = model
                self.version = data["version"]

                return

            raise ValueError(
                "Unrecognized predictive model artifact format."
            )

        except Exception:
            # Never allow a corrupted/unusable artifact to make
            # the entire API fail during startup.
            self.model = None
            self.version = "untrained"

    def vector(
        self,
        features: dict,
    ) -> list[float]:
        """
        Convert a feature dictionary into the exact feature
        ordering expected by the model.
        """

        return [
            float(
                features.get(
                    key,
                    0.0,
                )
                or 0.0
            )
            for key in FEATURES
        ]

    def predict(
        self,
        features: dict,
    ) -> dict:
        """
        Generate directional probabilities.

        The model must already have been validated and promoted.
        """

        if self.model is None:
            return {
                "trained": False,
                "abstain": True,
                "version": self.version,
                "probabilities": {
                    "short": 0.0,
                    "flat": 1.0,
                    "long": 0.0,
                },
                "reason": (
                    "No validated model artifact "
                    "is available."
                ),
            }

        x = np.asarray(
            [self.vector(features)],
            dtype=float,
        )

        probabilities = (
            self.model.predict_proba(x)[0]
        )

        classes = list(
            self.model.classes_
        )

        probability_map = {
            str(class_id): float(probability)
            for class_id, probability
            in zip(
                classes,
                probabilities,
            )
        }

        return {
            "trained": True,
            "abstain": False,
            "version": self.version,
            "probabilities": {
                "short": probability_map.get(
                    "-1",
                    0.0,
                ),
                "flat": probability_map.get(
                    "0",
                    0.0,
                ),
                "long": probability_map.get(
                    "1",
                    0.0,
                ),
            },
        }

    def artifact(self) -> dict | None:
        """
        Return the compact model artifact.

        Only model parameters are persisted. The full historical
        training dataset is not stored inside direction_model.json.
        """

        if self.model is None:
            return None

        scaler = self.model.named_steps[
            "scale"
        ]

        classifier = self.model.named_steps[
            "clf"
        ]

        return {
            "version": self.version,
            "features": FEATURES,
            "coef": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "classes": classifier.classes_.tolist(),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }

    def load_compact_artifact(
        self,
        data: dict,
    ) -> None:
        """
        Reconstruct a trained sklearn pipeline from the compact
        persisted artifact.
        """

        required = (
            "version",
            "coef",
            "intercept",
            "classes",
            "mean",
            "scale",
        )

        missing = [
            key
            for key in required
            if key not in data
        ]

        if missing:
            raise ValueError(
                "Compact model artifact is missing: "
                + ", ".join(missing)
            )

        mean = np.asarray(
            data["mean"],
            dtype=float,
        )

        scale_values = np.asarray(
            data["scale"],
            dtype=float,
        )

        coef = np.asarray(
            data["coef"],
            dtype=float,
        )

        intercept = np.asarray(
            data["intercept"],
            dtype=float,
        )

        classes = np.asarray(
            data["classes"],
        )

        if len(mean) != len(FEATURES):
            raise ValueError(
                "Model scaler has an invalid feature count."
            )

        if len(scale_values) != len(FEATURES):
            raise ValueError(
                "Model scale has an invalid feature count."
            )

        if coef.shape[-1] != len(FEATURES):
            raise ValueError(
                "Model coefficients have an invalid feature count."
            )

        if len(classes) != 3:
            raise ValueError(
                "Predictive model must contain exactly "
                "three classes: -1, 0 and 1."
            )

        scaler = StandardScaler()

        scaler.mean_ = mean
        scaler.scale_ = scale_values
        scaler.var_ = scale_values ** 2
        scaler.n_features_in_ = len(
            FEATURES
        )

        classifier = LogisticRegression(
            max_iter=500,
        )

        classifier.classes_ = classes
        classifier.coef_ = coef
        classifier.intercept_ = intercept
        classifier.n_features_in_ = len(
            FEATURES
        )

        classifier.n_iter_ = np.asarray(
            [1]
        )

        self.model = Pipeline(
            [
                (
                    "scale",
                    scaler,
                ),
                (
                    "clf",
                    classifier,
                ),
            ]
        )

        self.version = str(
            data["version"]
        )

    def train(
        self,
        rows: list[dict],
        version: str,
        min_rows: int = 500,
    ) -> ModelReport:
        """
        Train the final candidate model.

        This method should only be called after independent
        walk-forward validation has passed.
        """

        if len(rows) < min_rows:
            return ModelReport(
                False,
                self.version,
                {},
                (
                    f"Need at least {min_rows} "
                    "labeled examples; "
                    f"received {len(rows)}."
                ),
            )

        X = np.asarray(
            [
                self.vector(
                    row["features"]
                )
                for row in rows
            ],
            dtype=float,
        )

        y = np.asarray(
            [
                int(row["label"])
                for row in rows
            ],
            dtype=int,
        )

        classes = set(
            y.tolist()
        )

        if classes != {-1, 0, 1}:
            return ModelReport(
                False,
                self.version,
                {},
                (
                    "Training data must contain "
                    "all three labels: -1, 0 and 1."
                ),
            )

        model = Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                    ),
                ),
            ]
        )

        model.fit(
            X,
            y,
        )

        self.model = model
        self.version = version

        # Persist only the compact trained model.
        artifact = self.artifact()

        if artifact is None:
            return ModelReport(
                False,
                self.version,
                {},
                "Failed to create model artifact.",
            )

        self.model_path.write_text(
            json.dumps(
                artifact,
                indent=2,
            )
        )

        return ModelReport(
            True,
            version,
            {},
            (
                "Candidate model trained and "
                "compact artifact persisted. "
                "Promotion requires independent "
                "walk-forward validation."
            ),
        )


predictive_model = PredictiveModel()