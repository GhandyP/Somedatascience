"""Behavioral tests for the raw-name classification pipeline."""

import importlib

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from namegender.model import LABELS, build_pipeline, load_pipeline, save_pipeline

NAMES = ["Juan", "Maria", "Carlos", "Laura", "Pedro", "Sofia", "Jorge", "Valentina"]
TARGETS = ["boy", "girl", "boy", "girl", "boy", "girl", "boy", "girl"]


def test_build_pipeline_returns_fresh_unfitted_pipelines():
    first = build_pipeline()
    second = build_pipeline()

    assert isinstance(first, Pipeline)
    assert first is not second
    assert first.named_steps["features"] is not second.named_steps["features"]
    assert first.named_steps["classifier"] is not second.named_steps["classifier"]
    assert not hasattr(first.named_steps["features"].transformer_list[1], "vocabulary_")


def test_line_52_regression_predicts_raw_name_after_fit():
    pipeline = build_pipeline().fit(NAMES, TARGETS)

    prediction = pipeline.predict(["Ricardo"])

    assert prediction.shape == (1,)
    assert prediction[0] in LABELS


def test_learns_gender_pattern_beyond_balanced_majority_baseline():
    """Assert the learner fits at all; other tests pass with constant ``boy`` predictions.

    Generalization to unseen names is a separate concern evaluated in the evaluation
    task rather than here.
    """
    feminine_names = [
        "Olivia", "Emma", "Ava", "Sophia", "Isabella", "Mia", "Amelia", "Harper",
        "Evelyn", "Abigail", "Emily", "Ella", "Elizabeth", "Camila", "Luna", "Sofia",
        "Avery", "Mila", "Aria", "Scarlett", "Victoria", "Madison", "Layla", "Grace",
        "Chloe", "Penelope", "Nora", "Riley", "Hannah", "Lily",
    ]
    masculine_names = [
        "Liam", "Noah", "Oliver", "James", "Elijah", "William", "Henry", "Lucas",
        "Benjamin", "Christopher", "Anthony", "Matthew", "Sebastian", "Daniel", "Jack",
        "Michael", "Alexander", "Owen", "Asher", "Ethan", "Jackson", "Mason", "John",
        "Hudson", "Aiden", "Joseph", "David", "Robert", "Thomas", "Charles",
    ]
    names = feminine_names + masculine_names
    targets = np.array(["girl"] * len(feminine_names) + ["boy"] * len(masculine_names))
    majority = np.max(np.bincount(targets == "boy")) / len(targets)

    assert majority == 0.5

    pipeline = build_pipeline().fit(names, targets)
    accuracy = np.mean(pipeline.predict(names) == targets)

    assert accuracy >= majority + 0.25


def test_unseen_characters_do_not_crash_prediction():
    pipeline = build_pipeline().fit(NAMES, TARGETS)

    predictions = pipeline.predict(["Xzqvb", "Qwertyuiop"])

    assert len(predictions) == 2
    assert set(predictions) <= set(LABELS)


def test_vectorizer_lives_inside_pipeline_and_starts_without_vocabulary():
    pipeline = build_pipeline()
    vectorizer = dict(pipeline.named_steps["features"].transformer_list)["character"]

    assert not hasattr(vectorizer, "vocabulary_")
    pipeline.fit(NAMES, TARGETS)
    assert hasattr(pipeline.named_steps["features"], "transformer_list")
    assert hasattr(vectorizer, "vocabulary_")


def test_pipeline_is_cloneable():
    cloned = clone(build_pipeline())

    assert isinstance(cloned, Pipeline)
    assert cloned is not build_pipeline()


def test_persistence_round_trip_predicts_raw_names_identically(tmp_path):
    pipeline = build_pipeline().fit(NAMES, TARGETS)
    before = pipeline.predict(["Ricardo", "Daniela"])
    path = save_pipeline(pipeline, tmp_path / "model.joblib")

    loaded = load_pipeline(path)

    assert path.exists()
    assert np.array_equal(before, loaded.predict(["Ricardo", "Daniela"]))


def test_predict_proba_is_normalized_and_agrees_with_predict():
    pipeline = build_pipeline().fit(NAMES, TARGETS)
    probabilities = pipeline.predict_proba(["Ricardo", "Daniela"])
    predictions = pipeline.predict(["Ricardo", "Daniela"])

    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(pipeline.classes_[probabilities.argmax(axis=1)], predictions)


def test_same_training_data_and_random_state_are_deterministic():
    first = build_pipeline(random_state=23).fit(NAMES, TARGETS)
    second = build_pipeline(random_state=23).fit(NAMES, TARGETS)

    assert np.array_equal(first.predict(NAMES), second.predict(NAMES))
    assert np.allclose(first.predict_proba(NAMES), second.predict_proba(NAMES))


def test_importing_model_does_not_fit_or_read_dataset(monkeypatch):
    import namegender.data as data_module

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dataset read during model import")

    monkeypatch.setattr(data_module, "load_raw", fail_if_called)
    import namegender.model as model_module
    importlib.reload(model_module)
