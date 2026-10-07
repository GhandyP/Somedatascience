"""Behavioral tests for raw-name feature extraction."""

import numpy as np
from sklearn.base import clone

from namegender.features import NameNumericFeatures, build_feature_union


def test_numeric_features_are_two_dimensional_and_named():
    transformer = NameNumericFeatures()
    values = transformer.fit_transform(["A", "O'Neil", "Jean-Luc", "mIxEd"])

    assert values.ndim == 2
    assert values.shape[0] == 4
    assert values.shape[1] == len(transformer.get_feature_names_out())
    assert np.issubdtype(values.dtype, np.floating)


def test_numeric_features_handle_edge_case_names():
    transformer = NameNumericFeatures()
    values = transformer.fit_transform(["A", "O'Neil", "Jean-Luc", "mIxEd"])

    assert values.shape == (4, 4)
    assert np.isfinite(values).all()
    assert clone(transformer).fit(["A"]).transform(["A"]).shape == (1, 4)


def test_feature_union_is_fresh_and_contains_character_ngrams():
    first = build_feature_union()
    second = build_feature_union()

    assert first is not second
    assert len(first.transformer_list) == 2
    names = dict(first.transformer_list)
    assert isinstance(names["numeric"], NameNumericFeatures)
    assert names["character"].analyzer == "char"
    assert names["character"].ngram_range == (2, 3)
    assert not hasattr(names["character"], "vocabulary_")

    fitted = first.fit(["Anna", "Bob"])
    output = fitted.transform(["Anna"])
    assert output.shape[0] == 1
    assert output.shape[1] > len(names["numeric"].get_feature_names_out())


def test_numeric_feature_values_are_defensible():
    values = NameNumericFeatures().fit_transform(["A", "Bob"])

    # length, vowels, consonants, and vowel-ending indicator
    np.testing.assert_allclose(values, [[1.0, 1.0, 0.0, 1.0], [3.0, 1.0, 2.0, 0.0]])
