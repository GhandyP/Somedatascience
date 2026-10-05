"""Leakage-safe chronological evaluation for next-day rainfall occurrence."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WEATHER_COLUMNS = (
    "pressure", "maxtemp", "temparature", "mintemp", "dewpoint", "humidity",
    "cloud", "low visibility hour", "sunshine", "radiation", "evaporation",
    "winddirection", "windspeed",
)


@dataclass(frozen=True)
class EvaluationSamples:
    """Feature rows observed on source_date with next-day target labels."""

    features: pd.DataFrame
    target: pd.Series
    source_date: pd.Series
    target_date: pd.Series

    def __len__(self):
        return len(self.target)


def build_samples(frame: pd.DataFrame) -> EvaluationSamples:
    """Pair each day with tomorrow's occurrence label without target leakage."""
    required = {"date", "rainfall_event", *WEATHER_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["date"], errors="raise").reset_index(drop=True)
    events = frame["rainfall_event"].reset_index(drop=True).astype(bool)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("dates must be unique and increasing")
    if len(dates) > 1 and not (dates.diff().dropna() == pd.Timedelta(1, unit="D")).all():
        raise ValueError("dates must be continuous daily observations")

    source = frame.loc[:, WEATHER_COLUMNS].reset_index(drop=True).apply(pd.to_numeric, errors="coerce")
    source["prior_day_rainfall_event"] = events
    target_dates = dates.iloc[1:].reset_index(drop=True)
    source_dates = dates.iloc[:-1].reset_index(drop=True)
    year_start = target_dates.dt.to_period("Y").dt.start_time
    next_year = year_start + pd.offsets.YearBegin(1)
    year_length = (next_year - year_start).dt.days
    phase = 2 * 3.141592653589793 * (target_dates.dt.dayofyear - 1) / year_length
    source["target_date_sin"] = np.sin(phase)
    source["target_date_cos"] = np.cos(phase)
    return EvaluationSamples(
        features=source.iloc[:-1].reset_index(drop=True),
        target=events.iloc[1:].reset_index(drop=True),
        source_date=source_dates,
        target_date=target_dates,
    )


def chronological_split(samples: EvaluationSamples):
    """Return train/validation/test in fixed, target-date chronological windows."""
    dates = samples.target_date
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError("target dates must be strictly increasing")
    train_mask = dates.between("2010-01-02", "2016-12-31")
    validation_mask = dates.between("2017-01-01", "2018-12-31")
    test_mask = dates.between("2019-01-01", "2019-10-31")
    partitions = tuple(_take(samples, mask) for mask in (train_mask, validation_mask, test_mask))
    train, validation, test = partitions
    for left, right in ((train, validation), (validation, test), (train, test)):
        if set(left.target_date).intersection(right.target_date):
            raise AssertionError("target-date partitions overlap")
        if set(left.source_date).intersection(right.source_date):
            raise AssertionError("source-date partitions overlap")
    if len(train) and len(validation) and not train.target_date.max() < validation.target_date.min():
        raise AssertionError("training targets must precede validation targets")
    if len(validation) and len(test) and not validation.target_date.max() < test.target_date.min():
        raise AssertionError("validation targets must precede test targets")
    return partitions


def _take(samples, mask):
    indices = mask[mask].index
    return EvaluationSamples(
        samples.features.loc[indices].reset_index(drop=True),
        samples.target.loc[indices].reset_index(drop=True),
        samples.source_date.loc[indices].reset_index(drop=True),
        samples.target_date.loc[indices].reset_index(drop=True),
    )


def _metrics(target, probability):
    truth = pd.Series(target).astype(bool).to_numpy()
    probability = pd.Series(probability).to_numpy(dtype=float)
    if not len(truth):
        raise ValueError("cannot compute metrics for an empty slice")
    predicted = probability >= 0.5
    has_both_classes = len(set(truth)) == 2
    return {
        "average_precision": float(average_precision_score(truth, probability)) if has_both_classes else None,
        "brier_score": float(brier_score_loss(truth, probability)),
        "roc_auc": float(roc_auc_score(truth, probability)) if has_both_classes else None,
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(truth, predicted)),
    }


def evaluate_models(train: EvaluationSamples, validation: EvaluationSamples, test: EvaluationSamples):
    """Fit only on training rows; report validation and test metrics for three predictors."""
    if not len(train) or len(set(train.target.astype(bool))) < 2:
        raise ValueError("training targets must contain both classes")
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(random_state=0, max_iter=1000),
    )
    model.fit(train.features, train.target.astype(int))
    prevalence = float(train.target.astype(bool).mean())
    predictions = {}
    for split_name, samples in (("validation", validation), ("test", test)):
        predictions[split_name] = {
            "logistic": model.predict_proba(samples.features)[:, 1],
            "train_prevalence": [prevalence] * len(samples),
            "persistence": samples.features["prior_day_rainfall_event"].astype(float).to_numpy(),
        }
    return {
        model_name: {
            split_name: _metrics(samples.target, predictions[split_name][model_name])
            for split_name, samples in (("validation", validation), ("test", test))
        }
        for model_name in ("logistic", "train_prevalence", "persistence")
    }
