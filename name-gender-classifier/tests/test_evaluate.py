"""Behavioral tests for leakage-free evaluation and its report."""

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from namegender.evaluate import (
    ambiguity_breakdown,
    evaluate_methods,
    format_report,
    leakage_comparison,
    legacy_illusion,
    split_by_name,
    BUCKETS,
)
from namegender.data import load_raw
from namegender.labels import LEGACY_EXAMPLE_LABELS, LEGACY_EXAMPLE_NAMES, build_task_table
from namegender.model import build_pipeline


@pytest.fixture(scope="session")
def task_table():
    return build_task_table(load_raw())


@pytest.fixture(scope="session")
def sample_table(task_table):
    # Select four names per ambiguity bucket with seed 2024, then retain two
    # yearly rows per name: deterministic, repeated-name, and bounded at 40 rows.
    parts = []
    for bucket, lower, upper in BUCKETS[:-1]:
        mask = (task_table["ambiguity"] >= lower) & (task_table["ambiguity"] < upper)
        candidates = task_table.loc[mask]
        candidates = candidates[candidates.groupby("name")["name"].transform("size") >= 2]
        names = candidates["name"].drop_duplicates().sample(n=4, random_state=2024)
        parts.append(candidates[candidates["name"].isin(names)].groupby("name", sort=True).head(2))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="session")
def fitted_model(sample_table):
    table = sample_table

    @lru_cache(maxsize=None)
    def get(random_state, test_size):
        train, _ = split_by_name(table["name"], test_size=test_size, random_state=random_state)
        return build_pipeline(random_state=random_state).fit(table["name"].iloc[train], table["label"].iloc[train])

    return get


@pytest.fixture(scope="session")
def cached_leakage_result(task_table):
    return leakage_comparison(task_table.copy(), random_state=17)


def test_split_by_name_has_disjoint_groups():
    names = np.array(["Alex", "Alex", "Mia", "Mia", "Sam", "Jo"])
    train, test = split_by_name(names, test_size=.5, random_state=7)
    assert set(names[train]).isdisjoint(set(names[test]))


def test_ambiguity_breakdown_always_carries_counts_and_omits_empty_buckets(sample_table):
    table = sample_table
    result = ambiguity_breakdown(table, test_indices=np.arange(len(table)), predictions=table["label"])
    assert all("count" in bucket for bucket in result.values())
    assert all(bucket["count"] > 0 for bucket in result.values())


def test_legacy_illusion_reproduces_one_for_the_wrong_reason():
    """One is expected because the model is fitted and scored on identical names, not because it generalizes."""
    assert legacy_illusion() == 1.0
    assert len(LEGACY_EXAMPLE_NAMES) == len(LEGACY_EXAMPLE_LABELS) == 10


def test_leakage_is_higher_under_row_split(cached_leakage_result):
    assert cached_leakage_result["row_split"] > cached_leakage_result["name_split"]


def test_evaluate_methods_has_majority_and_model_is_at_least_majority(sample_table):
    result = evaluate_methods(sample_table, random_state=11)
    assert "majority" in result
    assert result["model"]["accuracy"] >= result["majority"]["accuracy"]
    assert all(item["test_count"] > 0 and item["test_name_count"] > 0 for item in result.values())


def test_format_report_is_nonempty_and_shows_ambiguity_count(sample_table):
    report = format_report(sample_table, random_state=11)
    assert report.strip()
    assert "test rows" in report.lower()
    assert "test names" in report.lower()


def test_report_model_matches_evaluate_methods_model(fitted_model, sample_table):
    table = sample_table
    result = evaluate_methods(table, random_state=11)
    report = format_report(table, random_state=11)
    assert len(fitted_model(11, 0.2).predict(table["name"])) == len(table)
    model_line = next(line for line in report.splitlines() if line.startswith("model:"))
    assert float(model_line.split("accuracy=")[1].split()[0]) == round(result["model"]["accuracy"], 4)


def test_full_report_model_matches_evaluate_methods_model(task_table):
    result = evaluate_methods(task_table, random_state=1)
    report = format_report(task_table, random_state=1)
    model_line = next(line for line in report.splitlines() if line.startswith("model:"))
    printed_accuracy = float(model_line.split("accuracy=")[1].split()[0])
    assert printed_accuracy == round(result["model"]["accuracy"], 4)


def test_leakage_comparison_accepts_empty_attrs(sample_table):
    table = sample_table
    assert table.attrs == {}
    result = leakage_comparison(table, random_state=11)
    assert {"name_split", "row_split", "inflation", "name_test_count", "row_test_count", "name_test_name_count", "row_test_name_count"} <= set(result)


def test_small_ambiguity_bucket_is_uninterpretable_in_report():
    table = pd.DataFrame(
        {
            "name": ["Alex", "Anna", "Emma", "Mark", "Luke"] * 2,
            "year": [2000] * 5 + [2001] * 5,
            "girl_share": [0.0, 1.0, 0.5, 0.0, 0.0] * 2,
            "label": ["boy", "girl", "girl", "boy", "boy"] * 2,
            "ambiguity": [0.0, 0.0, 1.0, 0.0, 0.0] * 2,
        }
    )
    report = format_report(table, random_state=11)
    assert "genuinely 50/50: too small to interpret" in report
    assert "genuinely 50/50: accuracy=" not in report


def test_exact_half_share_is_reported_in_highest_ambiguity_bucket(task_table):
    table = task_table[task_table["ambiguity"] == 1.0].head(1)
    result = ambiguity_breakdown(table, test_indices=np.array([0]), predictions=np.array(["girl"]))
    assert result["genuinely 50/50"] == {"accuracy": 1.0, "count": 1, "name_count": 1}
