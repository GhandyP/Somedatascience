"""Offline-first access to the SSA-derived baby names dataset."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from .errors import DataError, IntegrityError, SchemaError

# The URL is the pinned hadley/data-baby-names GitHub mirror selected in the feature decision.
SOURCE_URL = "https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv"
# The source is SSA-derived baby names mirrored by Hadley Wickham's data-baby-names repository.
SOURCE_NAME = "SSA Popular Baby Names, mirrored by hadley/data-baby-names"
# The source is public-domain US Government work under 17 U.S.C. 105, derived from SSA Popular Baby Names.
SOURCE_LICENSE = "Public domain, US Government work, 17 U.S.C. 105, derived from SSA Popular Baby Names"
# The source contains the top 1,000 names per sex per year from 1880-2008, not the complete SSA roster.
SOURCE_COVERAGE = "Top 1,000 names per sex per year, 1880-2008; not the complete SSA roster"
# This byte count comes from the cached source fetched for the N2 data decision.
EXPECTED_BYTES = 7447879
# This SHA-256 comes from the cached source fetched for the N2 data decision.
EXPECTED_SHA256 = "1259523fa76e5c18151a4c7612b854b22605d07127c596126940e2941ba15d3c"
# This row count comes from the cached source inspection recorded in the N2 data decision.
EXPECTED_ROW_COUNT = 258000
# This year range comes from the cached source inspection recorded in the N2 data decision.
EXPECTED_YEAR_RANGE = (1880, 2008)
# These are the exact sex values observed in the cached source.
EXPECTED_SEX_VALUES = frozenset({"boy", "girl"})
# These are the required source columns, in their source and public API order.
REQUIRED_COLUMNS = ("year", "name", "percent", "sex")


def default_raw_path() -> Path:
    """Return the repository cache path, independent of the current working directory."""
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "baby-names.csv"


def verify_integrity(
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_SHA256,
    expected_bytes: int = EXPECTED_BYTES,
) -> None:
    """Stream *path* and require both its byte count and SHA-256 to match."""
    observed_bytes = 0
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                observed_bytes += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise IntegrityError(
            f"Unable to read file for integrity check: {path}; "
            f"expected bytes={expected_bytes}, observed bytes={observed_bytes}; "
            f"expected SHA-256={expected_sha256}, observed SHA-256={digest.hexdigest()}"
        ) from exc

    observed_sha256 = digest.hexdigest()
    if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
        raise IntegrityError(
            f"Integrity mismatch for {path}: expected bytes={expected_bytes}, "
            f"observed bytes={observed_bytes}; expected SHA-256={expected_sha256}, "
            f"observed SHA-256={observed_sha256}"
        )


def download_raw(dest: str | Path | None = None, *, url: str = SOURCE_URL) -> Path:
    """Download and atomically install the verified raw dataset."""
    destination = Path(dest) if dest is not None else default_raw_path()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as target:
            temporary = Path(target.name)
            with urlopen(url) as source:  # noqa: S310 - explicit network boundary by API design
                shutil.copyfileobj(source, target)
        verify_integrity(
            temporary,
            expected_sha256=EXPECTED_SHA256,
            expected_bytes=EXPECTED_BYTES,
        )
        os.replace(temporary, destination)
        temporary = None
        return destination
    except (OSError, IntegrityError) as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if isinstance(exc, IntegrityError):
            raise
        raise DataError(f"Could not download raw dataset from {url}: {exc}") from exc


def load_raw(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the raw dataset without performing network access.

    The default cache path is integrity-checked against the pinned dataset before
    parsing, so a truncated or replaced cache cannot be trained on silently.
    Explicit paths are not integrity-checked because they are intended for test
    fixtures and other caller-provided data rather than the pinned dataset.
    """
    path = Path(csv_path) if csv_path is not None else default_raw_path()
    if not path.exists():
        raise DataError(f"Raw dataset does not exist: {path}")
    if csv_path is None:
        verify_integrity(path)
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise SchemaError(f"Could not parse raw dataset {path}: {exc}") from exc

    actual_columns = tuple(frame.columns)
    if set(actual_columns) != set(REQUIRED_COLUMNS):
        missing = sorted(set(REQUIRED_COLUMNS) - set(actual_columns))
        unexpected = sorted(set(actual_columns) - set(REQUIRED_COLUMNS))
        raise SchemaError(f"Invalid columns: missing={missing}, unexpected={unexpected}")

    frame = frame.loc[:, REQUIRED_COLUMNS]
    if frame.loc[:, REQUIRED_COLUMNS].isnull().any().any():
        raise SchemaError("Required columns contain null values")

    years = pd.to_numeric(frame["year"], errors="coerce")
    percents = pd.to_numeric(frame["percent"], errors="coerce")
    if years.isnull().any() or ((years % 1) != 0).any():
        raise SchemaError("year must contain integral numeric values")
    if ((years < EXPECTED_YEAR_RANGE[0]) | (years > EXPECTED_YEAR_RANGE[1])).any():
        raise SchemaError(f"year must be within {EXPECTED_YEAR_RANGE}")
    if percents.isnull().any() or ((percents <= 0.0) | (percents > 1.0)).any():
        raise SchemaError("percent must be in the interval (0.0, 1.0]")

    names = frame["name"].astype(str)
    sexes = frame["sex"].astype(str)
    if names.str.strip().eq("").any():
        raise SchemaError("name must not be empty or whitespace-only")
    if not set(sexes).issubset(EXPECTED_SEX_VALUES):
        raise SchemaError(f"sex values must be within {sorted(EXPECTED_SEX_VALUES)}")

    result = frame.copy()
    result["year"] = years.astype("int64")
    result["percent"] = percents.astype("float64")
    result["name"] = names.astype("string")
    result["sex"] = sexes.astype("string")
    return result.loc[:, REQUIRED_COLUMNS]
