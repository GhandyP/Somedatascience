import warnings

import numpy as np
import pandas as pd

from hk_rainfall.evaluation import _metrics, build_samples, chronological_split, evaluate_models


def synthetic_frame(start="2010-01-01", end="2010-01-05"):
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({
        "date": dates,
        "pressure": [str(1000 + i) for i in range(len(dates))],
        "maxtemp": [str(20 + i) for i in range(len(dates))],
        "temparature": [str(15 + i) for i in range(len(dates))],
        "mintemp": [str(10 + i) for i in range(len(dates))],
        "dewpoint": [str(8 + i) for i in range(len(dates))],
        "humidity": [str(70 + i) for i in range(len(dates))],
        "cloud": [str(i) for i in range(len(dates))],
        "low visibility hour": ["-" if i == 2 else str(i) for i in range(len(dates))],
        "sunshine": [str(i) for i in range(len(dates))],
        "radiation": [str(i) for i in range(len(dates))],
        "evaporation": [str(i) for i in range(len(dates))],
        "winddirection": [str(i) for i in range(len(dates))],
        "windspeed": [str(i) for i in range(len(dates))],
        "rainfall_event": [bool(i % 2) for i in range(len(dates))],
        "rainfall_status": ["not_detected"] * len(dates),
    })


def test_build_samples_does_not_emit_numpy_generic_timedelta_deprecation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_samples(synthetic_frame())

    matching = [
        warning for warning in caught
        if issubclass(warning.category, DeprecationWarning)
        and "generic" in str(warning.message)
        and "timedelta" in str(warning.message).lower()
    ]
    assert not matching, [str(warning.message) for warning in matching]


def test_build_samples_rejects_subday_and_multiday_gaps():
    import pytest

    for dates in (
        pd.to_datetime(["2010-01-01 00:00", "2010-01-01 12:00"]),
        pd.to_datetime(["2010-01-01", "2010-01-03"]),
    ):
        frame = synthetic_frame().iloc[:2].copy()
        frame["date"] = dates
        with pytest.raises(ValueError, match="continuous daily"):
            build_samples(frame)


def test_samples_lag_features_and_label_by_exactly_one_day():
    samples = build_samples(synthetic_frame())
    assert samples.target_date.tolist() == list(pd.date_range("2010-01-02", "2010-01-05"))
    assert samples.source_date.tolist() == list(pd.date_range("2010-01-01", "2010-01-04"))
    assert samples.target.tolist() == [True, False, True, False]
    assert samples.features["prior_day_rainfall_event"].tolist() == [False, True, False, True]
    assert "rainfall_status" not in samples.features
    assert not any(name in {"rainfall", "rainfall_status", "target", "target_date"} for name in samples.features)
    assert "target_month" not in samples.features
    assert "target_day_of_year" not in samples.features
    assert samples.features["pressure"].tolist() == [1000, 1001, 1002, 1003]


def test_chronological_partitions_use_target_date_and_are_disjoint():
    frame = synthetic_frame("2010-01-01", "2019-01-02")
    samples = build_samples(frame)
    train, validation, test = chronological_split(samples)
    assert train.target_date.max() == pd.Timestamp("2016-12-31")
    assert validation.target_date.min() == pd.Timestamp("2017-01-01")
    assert validation.target_date.max() == pd.Timestamp("2018-12-31")
    assert test.target_date.min() == pd.Timestamp("2019-01-01")
    assert train.target_date.max() < validation.target_date.min() < test.target_date.min()
    for left, right in ((train, validation), (validation, test), (train, test)):
        assert set(left.target_date).isdisjoint(right.target_date)
        assert set(left.source_date).isdisjoint(right.source_date)


def test_metrics_reject_empty_slices():
    import pytest
    with pytest.raises(ValueError, match="empty slice"):
        _metrics([], [])


def test_target_dates_near_year_boundary_have_nearby_cyclical_features():
    samples = build_samples(synthetic_frame("2019-12-30", "2020-01-03"))
    before = samples.features.iloc[0][["target_date_sin", "target_date_cos"]].to_numpy(dtype=float)
    after = samples.features.iloc[1][["target_date_sin", "target_date_cos"]].to_numpy(dtype=float)
    assert np.linalg.norm(before - after) < 0.03


def test_evaluation_reports_model_and_baseline_metrics_and_handles_single_class():
    frame = synthetic_frame("2010-01-01", "2010-01-20")
    samples = build_samples(frame)
    metrics = evaluate_models(samples, samples, samples)
    assert set(metrics) == {"logistic", "train_prevalence", "persistence"}
    assert all(set(result) == {"validation", "test"} for result in metrics.values())
    for result in metrics.values():
        for slice_metrics in result.values():
            assert {"average_precision", "brier_score", "roc_auc", "precision", "recall", "f1", "accuracy"} <= set(slice_metrics)
    assert metrics["logistic"]["test"]["roc_auc"] is not None
    assert evaluate_models(samples, samples, samples) == metrics
    from hk_rainfall.evaluation import EvaluationSamples
    one_class = EvaluationSamples(
        samples.features.iloc[:3], pd.Series([False] * 3),
        samples.source_date.iloc[:3], samples.target_date.iloc[:3],
    )
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        single_metrics = evaluate_models(samples, one_class, one_class)
    for result in single_metrics.values():
        for split in result.values():
            assert split["roc_auc"] is None
            assert split["average_precision"] is None
    assert not any("No positive class found" in str(item.message) for item in caught)
    assert np.isfinite(metrics["train_prevalence"]["test"]["brier_score"])
