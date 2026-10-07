#!/usr/bin/env python3
"""Regenerate the committed sample fixture from the local raw CSV cache."""

import csv
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "baby-names.csv"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "baby_names_sample.csv"
YEARS = {"1880", "1881"}
LIMIT_PER_GROUP = 30


def generate_bytes(source: Path, *, per_group: int = LIMIT_PER_GROUP) -> bytes:
    """Select earliest eligible rows and serialize as quoted UTF-8 CSV with LF."""
    counts = {}
    output = io.StringIO(newline="")
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\n")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Source CSV has no header: {source}")
        writer.writerow(reader.fieldnames)
        for row in reader:
            year, sex = row["year"], row["sex"]
            key = (year, sex)
            if year in YEARS and counts.get(key, 0) < per_group:
                values = [row[field] for field in reader.fieldnames]
                raw_indices = {reader.fieldnames.index("year"), reader.fieldnames.index("percent")}
                fields = [
                    value if index in raw_indices else '"' + value.replace('"', '""') + '"'
                    for index, value in enumerate(values)
                ]
                output.write(",".join(fields) + "\n")
                counts[key] = counts.get(key, 0) + 1
    return output.getvalue().encode("utf-8")


def main() -> int:
    if not RAW_PATH.is_file():
        raise SystemExit(
            f"Raw dataset cache is absent: {RAW_PATH}. Populate it locally first; "
            "fixture regeneration does not download data."
        )
    FIXTURE_PATH.write_bytes(generate_bytes(RAW_PATH))
    print(f"Regenerated {FIXTURE_PATH} from local cache {RAW_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
