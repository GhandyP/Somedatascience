"""Behavioral tests for honest name-level baselines."""

import numpy as np

from namegender.baselines import LastLetterBaseline, MajorityBaseline, NameLookupBaseline


def test_baselines_predict_known_labels_and_beat_coin_flip_on_separable_input():
    names = np.array(["Anna", "Emma", "Lena", "Mia", "Mark", "Luke", "John"])
    labels = np.array(["girl", "girl", "girl", "girl", "boy", "boy", "boy"])
    new_names = ["Clara", "Nora", "Mia", "Frank"]
    for baseline in (MajorityBaseline(), LastLetterBaseline(), NameLookupBaseline()):
        predictions = baseline.fit(names, labels).predict(new_names)
        assert len(predictions) == len(new_names)
        assert set(predictions) <= {"boy", "girl"}
        assert np.mean(predictions == ["girl", "girl", "girl", "boy"]) > 0.5


def test_name_lookup_returns_seen_training_label():
    baseline = NameLookupBaseline().fit(["Alex", "Maria"], ["boy", "girl"])
    assert baseline.predict(["Maria"])[0] == "girl"
