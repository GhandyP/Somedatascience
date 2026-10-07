"""A persisted scikit-learn pipeline for first-name classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_feature_union

LABELS = ("boy", "girl")

# A moderate penalty keeps character signals interpretable without overfitting the fixture.
LOGISTIC_C = 1.0
# Extra iterations avoid convergence warnings as the real character vocabulary grows.
LOGISTIC_MAX_ITER = 1000
# Liblinear is a deterministic linear solver that handles this binary sparse signal well.
LOGISTIC_SOLVER = "liblinear"
# A fixed seed makes any solver tie-breaking reproducible across equivalent fits.
RANDOM_STATE = 17


def build_pipeline(random_state: int | None = RANDOM_STATE) -> Pipeline:
    """Return a fresh, unfitted pipeline that accepts raw name strings."""
    return Pipeline(
        steps=[
            ("features", build_feature_union()),
            (
                "classifier",
                LogisticRegression(
                    C=LOGISTIC_C,
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=random_state,
                    solver=LOGISTIC_SOLVER,
                ),
            ),
        ]
    )


def save_pipeline(pipeline: Pipeline, path: str | Path) -> Path:
    """Persist a fitted pipeline, including its fitted feature transformers."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    return destination


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a pipeline persisted by :func:`save_pipeline`.

    Only load a pipeline you produced or trust: joblib deserialization reconstructs
    arbitrary Python objects and can execute code contained in the file.
    """
    pipeline: Any = joblib.load(Path(path))
    if not isinstance(pipeline, Pipeline):
        raise TypeError("saved model is not a scikit-learn Pipeline")
    return pipeline
