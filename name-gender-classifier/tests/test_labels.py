"""Tests for the per-year name-level task table."""

import numpy as np

from namegender.labels import build_task_table


def test_shares_and_ambiguity_are_bounded_and_consistent():
    raw = __import__("pandas").DataFrame(
        [
            {"year": 2000, "name": "Alex", "percent": 0.03, "sex": "boy"},
            {"year": 2001, "name": "Alex", "percent": 0.01, "sex": "girl"},
            {"year": 2000, "name": "Mia", "percent": 0.04, "sex": "girl"},
        ]
    )
    table = build_task_table(raw)
    assert table["girl_share"].between(0, 1).all()
    assert table["ambiguity"].between(0, 1).all()
    assert np.allclose(table["ambiguity"], 2 * np.minimum(table["girl_share"], 1 - table["girl_share"]))
    assert set(table.columns) == {"name", "year", "girl_share", "label", "ambiguity"}
    assert table.set_index(["name", "year"]).loc[("Alex", 2000), "label"] == "boy"
    assert table.attrs == {}


def test_build_task_table_retains_repeated_year_rows_for_a_name():
    raw = __import__("pandas").DataFrame(
        [
            {"year": 2000, "name": "Alex", "percent": 0.03, "sex": "boy"},
            {"year": 2000, "name": "Alex", "percent": 0.01, "sex": "girl"},
            {"year": 2001, "name": "Alex", "percent": 0.02, "sex": "boy"},
            {"year": 2001, "name": "Alex", "percent": 0.02, "sex": "girl"},
        ]
    )
    table = build_task_table(raw)
    assert table["name"].value_counts().loc["Alex"] == 2


def test_exact_half_share_is_in_highest_ambiguity_bucket():
    raw = __import__("pandas").DataFrame(
        [
            {"year": 2000, "name": "Alex", "percent": 0.01, "sex": "boy"},
            {"year": 2000, "name": "Alex", "percent": 0.01, "sex": "girl"},
        ]
    )
    table = build_task_table(raw)
    assert table.set_index(["name", "year"]).loc[("Alex", 2000), "ambiguity"] == 1.0
