"""Command-line entry points for the reproducible name-gender analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import default_raw_path, download_raw, load_raw
from .drift import drift_table, drift_vs_error, write_drift_csv
from .errors import DataError
from .evaluate import format_report
from .labels import build_task_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the name-gender classifier analysis.")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download and verify the raw dataset")
    fetch.add_argument("--raw", type=Path, help="destination path for the downloaded CSV")

    for name, help_text in (
        ("evaluate", "print the honest evaluation report"),
        ("drift", "print drift results and write the drift CSV"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--raw", type=Path, help="path to a raw CSV")

    reproduce = commands.add_parser("reproduce", help="run the complete analysis and write reports")
    reproduce.add_argument("--raw", type=Path, help="path to a raw CSV")
    reproduce.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    return parser


def _load_task_table(raw: Path | None, *, allow_download: bool = False):
    path = raw or default_raw_path()
    if allow_download and raw is None and not path.exists():
        path = download_raw(path)
    return build_task_table(load_raw(path))


def _run_fetch(raw: Path | None) -> int:
    destination = download_raw(raw)
    print(f"Downloaded and verified raw dataset to {destination}")
    return 0


def _run_evaluate(raw: Path | None) -> int:
    print(format_report(_load_task_table(raw)))
    return 0


def _drift_output(table, destination: Path) -> None:
    drift = drift_table(table)
    write_drift_csv(drift, destination)
    summary = drift_vs_error(table)
    print(f"Drift summary: {len(drift)} flipped names")
    print(
        f"Flipped accuracy={summary['flipped_accuracy']:.4f} "
        f"(test rows={summary['flipped_test_rows']}, test names={summary['flipped_test_names']}); "
        f"stable accuracy={summary['stable_accuracy']:.4f} "
        f"(test rows={summary['stable_test_rows']}, test names={summary['stable_test_names']})"
    )
    print(f"Drift CSV: {destination}")


def _run_drift(raw: Path | None) -> int:
    table = _load_task_table(raw)
    _drift_output(table, DEFAULT_REPORTS_DIR / "drift.csv")
    return 0


def _run_reproduce(raw: Path | None, reports_dir: Path) -> int:
    table = _load_task_table(raw, allow_download=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = reports_dir / "evaluation.txt"
    report = format_report(table)
    evaluation_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    _drift_output(table, reports_dir / "drift.csv")
    print(f"Evaluation report: {evaluation_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "fetch":
            return _run_fetch(args.raw)
        if args.command == "evaluate":
            return _run_evaluate(args.raw)
        if args.command == "drift":
            return _run_drift(args.raw)
        return _run_reproduce(args.raw, args.reports_dir)
    except SystemExit as exc:
        return int(exc.code)
    except (DataError, OSError, ValueError) as exc:
        print(f"namegender: error: {exc}", file=sys.stderr)
        return 1
