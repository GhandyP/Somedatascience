"""Build the per-year name-level supervised-learning table from raw observations.

Each label is the dominant gender for that specific year. The model receives only
the name and never the year; that deliberate mismatch makes gender-flipping names
hard to predict.
"""

from __future__ import annotations

import pandas as pd

LEGACY_EXAMPLE_NAMES = (
    "Juan", "Maria", "Carlos", "Laura", "Pedro", "Sofia", "Jorge", "Valentina", "Luis", "Isabela"
)
LEGACY_EXAMPLE_LABELS = (
    "boy", "girl", "boy", "girl", "boy", "girl", "boy", "girl", "boy", "girl"
)


def build_task_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw rows into one label and ambiguity score per name and year."""
    required = {"name", "year", "percent", "sex"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw data is missing columns: {sorted(missing)}")
    totals = (
        raw.assign(percent=pd.to_numeric(raw["percent"], errors="raise"))
        .groupby(["name", "year", "sex"], as_index=False, sort=True)["percent"]
        .sum()
    )
    wide = totals.pivot(index=["name", "year"], columns="sex", values="percent").fillna(0.0)
    boy = wide.get("boy", pd.Series(0.0, index=wide.index))
    girl = wide.get("girl", pd.Series(0.0, index=wide.index))
    denominator = boy + girl
    if (denominator <= 0).any():
        raise ValueError("each name must have positive total percent")
    result = pd.DataFrame(
        {"name": wide.index.get_level_values("name").astype(str), "year": wide.index.get_level_values("year"), "girl_share": girl / denominator}
    )
    result["label"] = result["girl_share"].ge(0.5).map({True: "girl", False: "boy"})
    result["ambiguity"] = 2 * result["girl_share"].where(result["girl_share"] <= 0.5, 1 - result["girl_share"])
    # The task table is self-contained: every evaluation uses these per-year
    # name-level rows, and split strategy is the only varying factor in leakage comparison.
    return result.reset_index(drop=True)
