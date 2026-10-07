"""Behavioral tests for temporal gender-association drift."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from namegender.data import load_raw
from namegender.drift import (
    decade_share,
    detect_flips,
    drift_table,
    drift_vs_error,
    write_drift_csv,
)
from namegender.labels import build_task_table


@pytest.fixture(scope="module")
def task_table():
    return build_task_table(load_raw())


def test_known_flips_are_detected(task_table):
    flips = detect_flips(task_table)
    assert {"Ashley", "Alexis", "Avery", "Beverly"} <= set(flips)


def test_known_stable_names_are_not_detected(task_table):
    flips = detect_flips(task_table)
    assert "John" not in flips
    assert "Mary" not in flips


def test_decade_share_weights_percentages_not_yearly_means():
    raw = pd.DataFrame(
        {
            "name": ["Casey"] * 4,
            "year": [2000, 2001, 2000, 2001],
            "sex": ["girl", "girl", "boy", "boy"],
            "percent": [90.0, 1.0, 10.0, 1.0],
        }
    )
    result = decade_share(raw)
    assert result.loc[0, "female_share"] == 91.0 / 102.0
    assert result.loc[0, "female_share"] != (0.9 + 0.01) / 2


def test_flip_count_has_sane_non_target_bound(task_table):
    """The bound is a sanity check, not a measured target."""
    flips = detect_flips(task_table)
    assert 0 < len(flips) < task_table["name"].nunique()


def test_drift_table_contains_swing_and_last_transition(task_table):
    result = drift_table(task_table)
    assert not result.empty
    assert {"swing", "last_change_decade", "min_female_share", "max_female_share"} <= set(result)
    assert (result["swing"] == result["max_female_share"] - result["min_female_share"]).all()
    assert result["last_change_decade"].notna().all()


def test_write_drift_csv_matches_header_and_rows(task_table, tmp_path):
    table = drift_table(task_table)
    path = write_drift_csv(table, tmp_path / "reports" / "drift.csv")
    written = pd.read_csv(path)
    assert list(written.columns) == list(table.columns)
    assert len(written) == len(table)


def test_drift_vs_error_reports_counts_and_lower_flipped_accuracy(task_table):
    """A flipped name has labels from different decades, but the model sees no year."""
    result = drift_vs_error(task_table)
    assert result["test_rows"] == result["flipped_test_rows"] + result["stable_test_rows"]
    assert result["wrong_test_rows"] == result["flipped_wrong_rows"] + result["stable_wrong_rows"]
    assert result["flipped_test_share"] == result["flipped_test_rows"] / result["test_rows"]
    assert result["flipped_accuracy"] < result["stable_accuracy"]


def test_importing_drift_reads_no_dataset_and_opens_no_socket():
    """The module must import with no dataset access and no network.

    Sockets are blocked before the import and the module's public callables are
    asserted afterwards, so a silently skipped import cannot pass this test.
    """
    src = str(Path(__file__).resolve().parents[1] / "src")
    script = "\n".join(
        [
            "import socket",
            "socket.socket = lambda *a, **k: (_ for _ in ()).throw(",
            "    AssertionError('socket created'))",
            "import sys",
            f"sys.path.insert(0, {src!r})",
            "import namegender.drift as d",
            "assert callable(d.decade_share)",
            "assert callable(d.detect_flips)",
            "assert callable(d.drift_vs_error)",
            "",
        ]
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
