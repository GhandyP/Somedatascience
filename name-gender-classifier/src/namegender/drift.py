"""Offline analysis of names whose gender association changes by decade."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .evaluate import split_by_name
from .model import build_pipeline


def decade_of(year):
    """Return the decade containing *year*, such as 1980 for 1987."""
    return (pd.to_numeric(year) // 10 * 10).astype(int) if hasattr(year, "__len__") else int(year) // 10 * 10


def _shares_from_task_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {"name", "year", "girl_share"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"table is missing columns: {sorted(missing)}")
    rows = table[["name", "year", "girl_share"]].copy()
    rows["decade"] = decade_of(rows["year"])
    # A task table has already discarded the per-sex percentages.  This fallback
    # is useful for drift analysis, but raw observations are required when the
    # weighting of names within a decade matters.
    return (
        rows.groupby(["name", "decade"], as_index=False, sort=True)["girl_share"]
        .mean()
        .rename(columns={"girl_share": "female_share"})
    )


def decade_share(table: pd.DataFrame) -> pd.DataFrame:
    """Return one female share per ``(name, decade)``.

    For raw observations, percentages are aggregated per sex within each
    decade before division.  This is preferable to averaging yearly shares:
    averaging yearly shares gives every year equal weight regardless of how
    common the name was that year.  A task table produced by
    :func:`namegender.labels.build_task_table` has no raw percentages, so its
    available yearly shares are averaged as a documented fallback.
    """
    if {"name", "year", "percent", "sex"} <= set(table.columns):
        rows = table[["name", "year", "percent", "sex"]].copy()
        rows["decade"] = decade_of(rows["year"])
        totals = rows.groupby(["name", "decade", "sex"], as_index=False, sort=True)["percent"].sum()
        totals["sex"] = totals["sex"].replace({"girl": "female", "boy": "male"})
        wide = totals.pivot_table(index=["name", "decade"], columns="sex", values="percent", aggfunc="sum", fill_value=0.0)
        female = wide.get("female", pd.Series(0.0, index=wide.index))
        denominator = wide.sum(axis=1)
        if (denominator <= 0).any():
            raise ValueError("each name and decade must have positive total percent")
        return pd.DataFrame(
            {"name": wide.index.get_level_values("name"), "decade": wide.index.get_level_values("decade"), "female_share": female / denominator}
        ).reset_index(drop=True)
    return _shares_from_task_table(table)


def dominant_series(shares: pd.DataFrame) -> pd.DataFrame:
    """Add the dominant gender for every name and decade in *shares*."""
    required = {"name", "decade", "female_share"}
    missing = required - set(shares.columns)
    if missing:
        raise ValueError(f"shares is missing columns: {sorted(missing)}")
    result = shares.copy()
    result["dominant_gender"] = result["female_share"].ge(0.5).map({True: "female", False: "male"})
    return result.sort_values(["name", "decade"], ignore_index=True)


def detect_flips(table_or_shares: pd.DataFrame) -> set[str]:
    """Return names whose dominant gender changes across their observed decades."""
    shares = table_or_shares if {"female_share", "decade"} <= set(table_or_shares.columns) else decade_share(table_or_shares)
    dominant = dominant_series(shares)
    changes = dominant.groupby("name")["dominant_gender"].nunique()
    return set(changes[changes > 1].index.astype(str))


def drift_table(table_or_shares: pd.DataFrame) -> pd.DataFrame:
    """Summarize each flipped name, including its range and last transition decade."""
    shares = table_or_shares if {"female_share", "decade"} <= set(table_or_shares.columns) else decade_share(table_or_shares)
    dominant = dominant_series(shares)
    flipped = detect_flips(dominant)
    rows = []
    for name, group in dominant[dominant["name"].isin(flipped)].groupby("name", sort=True):
        group = group.sort_values("decade")
        changes = group.loc[group["dominant_gender"].ne(group["dominant_gender"].shift()), "decade"]
        rows.append(
            {
                "name": name,
                "first_decade": int(group["decade"].iloc[0]),
                "last_decade": int(group["decade"].iloc[-1]),
                "first_dominant_gender": group["dominant_gender"].iloc[0],
                "last_dominant_gender": group["dominant_gender"].iloc[-1],
                "min_female_share": float(group["female_share"].min()),
                "max_female_share": float(group["female_share"].max()),
                "swing": float(group["female_share"].max() - group["female_share"].min()),
                "last_change_decade": int(changes.iloc[-1]),
            }
        )
    return pd.DataFrame(rows, columns=["name", "first_decade", "last_decade", "first_dominant_gender", "last_dominant_gender", "min_female_share", "max_female_share", "swing", "last_change_decade"])


def write_drift_csv(table: pd.DataFrame, path: str | Path) -> Path:
    """Write a drift table to CSV, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination


def drift_vs_error(table: pd.DataFrame, test_size=0.2, random_state=1) -> dict[str, float | int]:
    """Compare model errors for flipped names against stable names.

    A flipped name has different per-year labels across its decades, while the
    model sees only the name and never the year.  Therefore no single
    name-only prediction can be right for all rows of that name.  The returned
    counts accompany every share and accuracy so the comparison cannot hide
    its sample sizes.
    """
    train, test = split_by_name(table["name"], test_size=test_size, random_state=random_state)
    model = build_pipeline(random_state=random_state).fit(table["name"].iloc[train], table["label"].iloc[train])
    predictions = model.predict(table["name"].iloc[test])
    labels = table["label"].iloc[test].to_numpy()
    names = table["name"].iloc[test].astype(str).to_numpy()
    flipped = detect_flips(table)
    is_flipped = pd.Series(names).isin(flipped).to_numpy()
    wrong = predictions != labels
    flipped_rows = int(is_flipped.sum())
    stable_rows = int((~is_flipped).sum())
    flipped_wrong = int((wrong & is_flipped).sum())
    stable_wrong = int((wrong & ~is_flipped).sum())
    return {
        "test_rows": len(test),
        "flipped_test_rows": flipped_rows,
        "stable_test_rows": stable_rows,
        "wrong_test_rows": int(wrong.sum()),
        "flipped_wrong_rows": flipped_wrong,
        "stable_wrong_rows": stable_wrong,
        "flipped_wrong_share": flipped_wrong / int(wrong.sum()) if wrong.sum() else 0.0,
        "flipped_test_share": flipped_rows / len(test),
        "flipped_accuracy": float((~wrong[is_flipped]).mean()) if flipped_rows else float("nan"),
        "stable_accuracy": float((~wrong[~is_flipped]).mean()) if stable_rows else float("nan"),
        "flipped_test_names": int(pd.unique(names[is_flipped]).size),
        "stable_test_names": int(pd.unique(names[~is_flipped]).size),
    }
