"""Command-line interface for reproducible Hong Kong rainfall evaluation."""

import argparse
import json
import sys
from pathlib import Path

from .evaluation import build_samples, chronological_split, evaluate_models
from .ingest import fetch_archive, load_archive

DEFAULT_ARCHIVE = Path("data/hongkongrainfall-v2.zip")
DEFAULT_OUTPUT = Path("reports/evaluation.json")


def _range(samples):
    if not len(samples):
        return None
    return {
        "count": len(samples),
        "source_date": [samples.source_date.min().strftime("%Y-%m-%d"), samples.source_date.max().strftime("%Y-%m-%d")],
        "target_date": [samples.target_date.min().strftime("%Y-%m-%d"), samples.target_date.max().strftime("%Y-%m-%d")],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hk-rainfall")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch", help="fetch and verify the pinned local archive")
    fetch_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    reproduce_parser = subparsers.add_parser("reproduce", help="evaluate an already-fetched local archive")
    reproduce_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    reproduce_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if args.command == "fetch":
        try:
            print(fetch_archive(args.archive))
            return 0
        except Exception as exc:
            print(f"Fetch failed: {exc}", file=sys.stderr)
            return 1

    try:
        frame, provenance = load_archive(args.archive)
        samples = build_samples(frame)
        train, validation, test = chronological_split(samples)
        metrics = evaluate_models(train, validation, test)
        report = {
            "dataset": {key: provenance[key] for key in ("dataset", "version", "archive_sha256")},
            "splits": {name: _range(partition) for name, partition in (
                ("train", train), ("validation", validation), ("test", test)
            )},
            "metrics": metrics,
        }
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0
    except Exception as exc:
        print(f"Reproduction failed: {exc}. Run 'make data' first to fetch the local archive.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
