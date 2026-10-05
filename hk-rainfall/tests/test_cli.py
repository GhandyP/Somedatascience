import json

import pandas as pd

from hk_rainfall import cli
from hk_rainfall.evaluation import EvaluationSamples


def sample(dates, labels):
    return EvaluationSamples(
        features=pd.DataFrame({"signal": range(len(labels))}),
        target=pd.Series(labels, dtype=bool),
        source_date=pd.Series(pd.to_datetime(dates) - pd.Timedelta(1, unit="D")),
        target_date=pd.Series(pd.to_datetime(dates)),
    )


def dataset():
    train = sample(["2010-01-02", "2011-01-02", "2012-01-02", "2013-01-02"], [0, 1, 0, 1])
    validation = sample(["2017-01-02", "2018-01-02"], [0, 1])
    test = sample(["2019-01-02", "2019-02-02"], [0, 1])
    return train, validation, test


def test_fetch_dispatch_calls_fetch_once(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "fetch_archive", lambda path: calls.append(path))
    assert cli.main(["fetch", "--archive", str(tmp_path / "cache.zip")]) == 0
    assert calls == [tmp_path / "cache.zip"]


def test_reproduce_missing_archive_fails_without_fetch(monkeypatch, tmp_path, capsys):
    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(cli, "load_archive", missing)
    monkeypatch.setattr(cli, "fetch_archive", lambda path: (_ for _ in ()).throw(AssertionError("network")))
    assert cli.main(["reproduce", "--archive", str(tmp_path / "missing.zip")]) != 0
    assert "make data" in capsys.readouterr().err


def test_reproduce_emits_aggregate_only_deterministic_json(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_archive", lambda path: (pd.DataFrame(), {"dataset": "fixture", "version": 2, "archive_sha256": "abc"}))
    monkeypatch.setattr(cli, "build_samples", lambda frame: object())
    monkeypatch.setattr(cli, "chronological_split", lambda samples: dataset())
    monkeypatch.setattr(cli, "evaluate_models", lambda *splits: {"logistic": {"test": {"roc_auc": None, "accuracy": 0.5}}})
    monkeypatch.setattr(cli, "fetch_archive", lambda path: (_ for _ in ()).throw(AssertionError("network")))

    output = tmp_path / "evaluation.json"
    args = ["reproduce", "--archive", str(tmp_path / "archive.zip"), "--output", str(output)]
    assert cli.main(args) == 0
    first = output.read_bytes()
    assert cli.main(args) == 0
    assert output.read_bytes() == first
    report = json.loads(first)
    assert set(report) == {"dataset", "splits", "metrics"}
    assert set(report["splits"]) == {"train", "validation", "test"}
    assert all(set(split) == {"count", "source_date", "target_date"} for split in report["splits"].values())
    assert "timestamp" not in report
    assert "rows" not in report and "source" not in report
    assert report["metrics"]["logistic"]["test"]["roc_auc"] is None
