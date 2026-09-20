"""Leakage-free feature transformers for first-name classification."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


class NameNumericFeatures(BaseEstimator, TransformerMixin):
    """Extract stable, numeric properties from raw name strings.

    The transformer is intentionally stateless: these properties are defined by
    each input name and do not depend on the training rows.
    """

    _FEATURE_NAMES = np.array(
        ["name_length", "vowel_count", "consonant_count", "ends_in_vowel"],
        dtype=object,
    )
    _VOWELS = frozenset("aeiou")

    def fit(self, X: Sequence[str], y=None):
        """Validate the input shape while learning no row-dependent state."""
        self._validate_names(X)
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        """Return one floating-point feature row for each raw name."""
        names = self._validate_names(X)
        features = []
        for name in names:
            letters = [character.lower() for character in name if character.isalpha()]
            vowels = sum(character in self._VOWELS for character in letters)
            consonants = len(letters) - vowels
            ends_in_vowel = bool(letters and letters[-1] in self._VOWELS)
            features.append(
                [float(len(name)), float(vowels), float(consonants), float(ends_in_vowel)]
            )
        return np.asarray(features, dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return stable names for the numeric output columns."""
        return self._FEATURE_NAMES.copy()

    @staticmethod
    def _validate_names(X: Sequence[str]) -> list[str]:
        names = list(X)
        if any(not isinstance(name, str) for name in names):
            raise TypeError("name features require a sequence of strings")
        return names


def build_feature_union() -> FeatureUnion:
    """Build fresh numeric and lowercased character n-gram features."""
    return FeatureUnion(
        transformer_list=[
            ("numeric", NameNumericFeatures()),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char",
                    lowercase=True,
                    ngram_range=(2, 3),
                ),
            ),
        ]
    )
