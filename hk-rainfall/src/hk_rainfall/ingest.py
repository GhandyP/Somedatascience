"""Pinned and integrity-checked download and parsing for a Kaggle-hosted dataset."""

from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

DATASET = "act18l/hongkongrainfall"
DATASET_VERSION = 2
ARCHIVE_SHA256 = "73bba9f13dd13ba2c8b5e92f72b1da5b03496ceca326d157f3a0147274abea02"
ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/act18l/hongkongrainfall?datasetVersionNumber=2"
EXPECTED_FIRST_DATE = "2010-01-01"
EXPECTED_LAST_DATE = "2019-10-31"
EXPECTED_SOURCE_COLUMNS = [
    "year", "month", "day", "pressure", "maxtemp", "temparature", "mintemp", "dewpoint",
    "humidity", "cloud", "rainfall", "low visibility hour", "sunshine", "radiation",
    "evaporation", "winddirection", "windspeed",
]


class IntegrityError(ValueError):
    """Downloaded bytes did not match the pinned archive."""


class DownloadError(RuntimeError):
    """The remote archive could not be downloaded."""


def _download(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def fetch_archive(destination: str | Path) -> Path:
    """Download only v2, verify its pinned digest, and atomically publish it."""
    destination = Path(destination)
    try:
        payload = _download(ARCHIVE_URL)
    except (OSError, URLError) as exc:
        raise DownloadError(f"could not download Kaggle dataset v{DATASET_VERSION}") from exc
    if sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise IntegrityError("Kaggle archive SHA-256 does not match the pinned v2 archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_csv(csv_bytes: bytes, *, expected_span: tuple[str, str] | None = None):
    """Parse source CSV while retaining source names and explicit rainfall states.

    `-` in rainfall means not detected (no event); `微量` means a trace event.
    Empty and dash values in other measurements are ordinary missing values.
    """
    try:
        frame = pd.read_csv(
            __import__("io").BytesIO(csv_bytes), encoding="gb18030", dtype=str,
            keep_default_na=False,
        )
    except (UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"invalid GB18030 CSV: {exc}") from exc
    source_columns = list(frame.columns)
    if not frame.empty and source_columns != EXPECTED_SOURCE_COLUMNS:
        raise ValueError("schema must match the 17 observed source columns and their order")
    if frame.empty:
        # pandas preserves headers for header-only CSVs; validate that shape before reporting emptiness.
        if source_columns != EXPECTED_SOURCE_COLUMNS:
            raise ValueError("schema must match the 17 observed source columns and their order")
        raise ValueError("CSV input must be non-empty")
    if source_columns != EXPECTED_SOURCE_COLUMNS:
        raise ValueError("schema must match the 17 observed source columns and their order")
    rain_column = "rainfall"
    date_values = pd.to_datetime(
        dict(year=pd.to_numeric(frame["year"], errors="coerce"),
             month=pd.to_numeric(frame["month"], errors="coerce"),
             day=pd.to_numeric(frame["day"], errors="coerce")), errors="coerce",
    )
    dates = pd.Series(date_values, index=frame.index)
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("year/month/day must form unique valid dates")
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be increasing and provide continuous daily coverage")
    expected_dates = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D")
    if len(dates) != len(expected_dates) or not dates.reset_index(drop=True).equals(pd.Series(expected_dates)):
        raise ValueError("dates must provide continuous daily coverage")
    span = expected_span or (EXPECTED_FIRST_DATE, EXPECTED_LAST_DATE)
    if (dates.iloc[0].strftime("%Y-%m-%d"), dates.iloc[-1].strftime("%Y-%m-%d")) != span:
        raise ValueError(f"date span must be {span[0]} through {span[1]}")
    rainfall = frame[rain_column]
    allowed = {"-", "微量"}
    invalid = [value for value in rainfall if value not in allowed and not _is_number(value)]
    if invalid:
        raise ValueError(f"schema contains unrecognized rainfall values: {invalid[:3]}")
    frame["date"] = dates
    frame["rainfall_event"] = rainfall.map(
        lambda value: False if value == "-" else True if value == "微量" else float(value) > 0
    )
    frame["rainfall_status"] = rainfall.map(
        lambda value: "not_detected" if value == "-" else "trace" if value == "微量" else "measured"
    )
    # Non-rainfall source measurements: blank and '-' are missing, never rainfall semantics.
    for column in source_columns:
        if column != rain_column:
            frame[column] = frame[column].replace({"": pd.NA, "-": pd.NA})
    return frame, {"source_columns": source_columns, "encoding": "gb18030", "dataset": DATASET,
                   "version": DATASET_VERSION, "archive_sha256": ARCHIVE_SHA256}


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def load_archive(path: str | Path):
    """Verify an archive on disk before extracting and parsing its sole CSV member."""
    payload = Path(path).read_bytes()
    if sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise IntegrityError("Kaggle archive SHA-256 does not match the pinned v2 archive")
    try:
        with ZipFile(Path(path)) as archive:
            if archive.namelist() != ["hongkong.csv"]:
                raise ValueError("archive schema must contain only hongkong.csv")
            return load_csv(archive.read("hongkong.csv"))
    except BadZipFile as exc:
        raise IntegrityError("verified archive is not a readable ZIP file") from exc
