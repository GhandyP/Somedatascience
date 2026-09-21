from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from namegender import cli


FIXTURE = Path(__file__).parent / "fixtures" / "baby_names_sample.csv"


def run_cli(*args):
    return cli.main(list(args))


@pytest.mark.parametrize("args", [("--help",), ("fetch", "--help"), ("evaluate", "--help"), ("drift", "--help"), ("reproduce", "--help")])
def test_help_exits_zero(args, capsys):
    assert run_cli(*args) == 0
    assert "usage:" in capsys.readouterr().out.lower()


def test_make_help_includes_target_descriptions(capsys):
    import subprocess

    project_dir = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["make", "-C", str(project_dir), "help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Install the project" in result.stdout
    assert "Remove generated caches" in result.stdout


def test_evaluate_fixture_prints_sample_count(capsys):
    assert run_cli("evaluate", "--raw", str(FIXTURE)) == 0
    output = capsys.readouterr().out
    assert "test rows=" in output


def test_reproduce_writes_report_files(tmp_path, capsys):
    assert run_cli("reproduce", "--raw", str(FIXTURE), "--reports-dir", str(tmp_path)) == 0
    capsys.readouterr()
    evaluation = tmp_path / "evaluation.txt"
    drift = tmp_path / "drift.csv"
    assert evaluation.exists()
    assert drift.exists()
    assert "model: accuracy=" in evaluation.read_text()


def test_missing_raw_is_readable_error(capsys):
    assert run_cli("evaluate", "--raw", "/does/not/exist.csv") != 0
    error = capsys.readouterr().err
    assert "does not exist" in error
    assert "Traceback" not in error


def test_default_reports_dir_points_to_project_reports():
    project_dir = Path(__file__).resolve().parents[1]
    assert cli.DEFAULT_REPORTS_DIR == project_dir / "reports"


def test_reproduce_default_reports_dir(monkeypatch, tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_bytes(FIXTURE.read_bytes())
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(cli, "DEFAULT_REPORTS_DIR", reports_dir)
    assert run_cli("reproduce", "--raw", str(raw)) == 0
    assert (reports_dir / "evaluation.txt").exists()
    assert (reports_dir / "drift.csv").exists()


def test_importing_cli_does_not_fetch(monkeypatch):
    import namegender.data as data

    def fail_network(*args, **kwargs):
        raise AssertionError("importing the CLI must not fetch")

    monkeypatch.setattr(data, "urlopen", fail_network)
    sys.modules.pop("namegender.cli", None)
    importlib.import_module("namegender.cli")
