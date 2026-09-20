"""Simple, leakage-visible baselines for name classification."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

LABELS = frozenset({"boy", "girl"})


class MajorityBaseline:
    """Predict the most common label observed during fitting."""

    def fit(self, names, labels):
        labels = list(labels)
        if not labels or not set(labels) <= LABELS:
            raise ValueError("labels must be non-empty and drawn from boy/girl")
        counts = Counter(labels)
        self.label_ = max(counts, key=lambda label: (counts[label], label == "boy"))
        return self

    def predict(self, names):
        self._check_fitted()
        return np.full(len(names), self.label_, dtype=object)

    def _check_fitted(self):
        if not hasattr(self, "label_"):
            raise RuntimeError("baseline is not fitted")


class LastLetterBaseline(MajorityBaseline):
    """Predict each final-letter majority, falling back to the overall majority."""

    def fit(self, names, labels):
        super().fit(names, labels)
        votes = defaultdict(list)
        for name, label in zip(names, labels):
            votes[str(name)[-1].lower()].append(label)
        self.by_letter_ = {
            letter: Counter(values).most_common(1)[0][0] for letter, values in votes.items()
        }
        return self

    def predict(self, names):
        self._check_fitted()
        return np.array([self.by_letter_.get(str(name)[-1].lower(), self.label_) for name in names], dtype=object)


class NameLookupBaseline(MajorityBaseline):
    """Look up seen names, with a majority fallback for unseen names.

    Under a split by name this necessarily degenerates to the majority baseline:
    every test name is unseen in training. That is the point, not a flaw—it
    demonstrates the leakage that a name lookup exploits under a row split.
    """

    def fit(self, names, labels):
        super().fit(names, labels)
        self.labels_by_name_ = dict(zip(map(str, names), labels))
        return self

    def predict(self, names):
        self._check_fitted()
        return np.array([self.labels_by_name_.get(str(name), self.label_) for name in names], dtype=object)
