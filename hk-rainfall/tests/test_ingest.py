from hashlib import sha256
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from hk_rainfall.ingest import (
    ARCHIVE_URL, DATASET_VERSION, DownloadError, IntegrityError, fetch_archive, load_csv,
)


def archive_bytes(csv_bytes):
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("hongkong.csv", csv_bytes)
    return output.getvalue()


@pytest.fixture
def csv_bytes():
    columns = ["year", "month", "day", "pressure", "maxtemp", "temparature", "mintemp",
               "dewpoint", "humidity", "cloud", "rainfall", "low visibility hour", "sunshine",
               "radiation", "evaporation", "winddirection", "windspeed"]
    return (
        ",".join(columns) + "\n"
        + "2010,1,1,,,,,,,,-,,,,,,\n"
        + "2010,1,2,,,,,,,,微量,0,,,,,\n"
        + "2010,1,3,,,,,,,,0,-,,,,,\n"
    ).encode("gb18030")


def test_load_preserves_rainfall_semantics_and_other_missingness(csv_bytes):
    frame, provenance = load_csv(csv_bytes, expected_span=("2010-01-01", "2010-01-03"))
    assert frame["rainfall"].tolist() == ["-", "微量", "0"]
    assert frame["rainfall_event"].tolist() == [False, True, False]
    assert frame["rainfall_status"].tolist() == ["not_detected", "trace", "measured"]
    assert frame["low visibility hour"].isna().tolist() == [True, False, True]
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2010-01-01", "2010-01-02", "2010-01-03"]
    assert provenance["source_columns"][:4] == ["year", "month", "day", "pressure"]
    assert "date" not in provenance["source_columns"]


def test_load_rejects_schema_and_date_drift(csv_bytes):
    with pytest.raises(ValueError, match="schema"):
        load_csv(b"year,month,day,rainfall\n2010,1,1,0\n")
    with pytest.raises(ValueError, match="date"):
        load_csv(csv_bytes.replace(b"2010,1,1", b"2011,1,1"), expected_span=("2010-01-01", "2010-01-03"))


def test_load_rejects_empty_and_non_contiguous_input(csv_bytes):
    with pytest.raises(ValueError, match="non-empty"):
        load_csv(b"year,month,day,pressure,maxtemp,temparature,mintemp,dewpoint,humidity,cloud,rainfall,low visibility hour,sunshine,radiation,evaporation,winddirection,windspeed\n")
    gapped = csv_bytes.replace(b"2010,1,2", b"2010,1,4")
    with pytest.raises(ValueError, match="continuous"):
        load_csv(gapped, expected_span=("2010-01-01", "2010-01-03"))


def test_fetch_verifies_archive_before_publishing(tmp_path, monkeypatch):
    payload = archive_bytes(b"synthetic archive")
    requested_urls = []

    def download(url):
        requested_urls.append(url)
        return payload

    monkeypatch.setattr("hk_rainfall.ingest._download", download)
    monkeypatch.setattr("hk_rainfall.ingest.ARCHIVE_SHA256", sha256(payload).hexdigest())
    destination = tmp_path / "archive.zip"
    assert fetch_archive(destination) == destination
    assert destination.read_bytes() == payload
    assert requested_urls == [ARCHIVE_URL]
    assert ARCHIVE_URL == (
        "https://www.kaggle.com/api/v1/datasets/download/act18l/hongkongrainfall"
        "?datasetVersionNumber=2"
    )
    assert DATASET_VERSION == 2


def test_fetch_failure_leaves_no_destination_or_temporary_file(tmp_path, monkeypatch):
    monkeypatch.setattr("hk_rainfall.ingest._download", lambda url: b"bad")
    with pytest.raises(IntegrityError):
        fetch_archive(tmp_path / "archive.zip")
    assert list(tmp_path.iterdir()) == []


def test_download_errors_are_reported_without_files(tmp_path, monkeypatch):
    monkeypatch.setattr("hk_rainfall.ingest._download", lambda url: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(DownloadError):
        fetch_archive(tmp_path / "archive.zip")
    assert list(tmp_path.iterdir()) == []
