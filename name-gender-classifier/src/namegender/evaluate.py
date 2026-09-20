"""Leakage-aware evaluation utilities and the N4 plain-text report."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .baselines import LastLetterBaseline, MajorityBaseline, NameLookupBaseline
from .labels import LEGACY_EXAMPLE_LABELS, LEGACY_EXAMPLE_NAMES
from .model import build_pipeline

# Ambiguity is 0 for a pure name and 1 for an exact 50/50 name. The final
# bucket is inclusive so the reachable value 1.0 is not dropped.
# Fewer than ten rows cannot support a meaningful bucket accuracy claim.
MIN_INTERPRETABLE_BUCKET_ROWS = 10

BUCKETS = (
    ("nearly pure", 0.0, 0.1),
    ("fairly skewed", 0.1, 0.25),
    ("ambiguous", 0.25, 0.5),
    ("very ambiguous", 0.5, 1.0),
    ("genuinely 50/50", 1.0, 1.0),
)


def split_by_name(names: Sequence[str], test_size=0.2, random_state=1):
    """Return train/test row indices while keeping each name on one side only."""
    names = np.asarray(names).astype(str)
    unique = np.unique(names)
    train_names, test_names = train_test_split(unique, test_size=test_size, random_state=random_state)
    test_set = set(test_names)
    test = np.flatnonzero(np.isin(names, list(test_set)))
    train = np.flatnonzero(~np.isin(names, list(test_set)))
    return train, test


def _accuracy(predictions, labels):
    return float(np.mean(np.asarray(predictions) == np.asarray(labels)))


def _fit_evaluation(table: pd.DataFrame, test_size=0.2, random_state=1):
    """Fit the name-level task once and return its split, model, and results."""
    train, test = split_by_name(table["name"], test_size=test_size, random_state=random_state)
    names, labels = table["name"].to_numpy(), table["label"].to_numpy()
    fit_names, fit_labels = names[train], labels[train]
    methods = {
        "majority": MajorityBaseline(),
        "last_letter": LastLetterBaseline(),
        "name_lookup": NameLookupBaseline(),
    }
    test_name_count = int(np.unique(names[test]).size)
    results = {}
    for name, method in methods.items():
        predictions = method.fit(fit_names, fit_labels).predict(names[test])
        results[name] = {
            "method": name,
            "accuracy": _accuracy(predictions, labels[test]),
            "test_count": len(test),
            "test_name_count": test_name_count,
        }
    model = build_pipeline(random_state=random_state).fit(fit_names, fit_labels)
    predictions = model.predict(names[test])
    results["model"] = {
        "method": "model",
        "accuracy": _accuracy(predictions, labels[test]),
        "test_count": len(test),
        "test_name_count": test_name_count,
    }
    return train, test, model, results


def evaluate_methods(table: pd.DataFrame, test_size=0.2, random_state=1):
    """Compare baselines and the N3 model on a group-by-name split."""
    return _fit_evaluation(table, test_size=test_size, random_state=random_state)[3]


def ambiguity_breakdown(table: pd.DataFrame, test_indices, predictions):
    """Return accuracy and count for every non-empty ambiguity bucket."""
    indices = np.asarray(test_indices)
    predictions = np.asarray(predictions)
    labels = table["label"].to_numpy()[indices]
    ambiguity = table["ambiguity"].to_numpy()[indices]
    result = {}
    for name, lower, upper in BUCKETS:
        if name == BUCKETS[-1][0]:
            selected = (ambiguity >= lower) & (ambiguity <= upper)
        else:
            selected = (ambiguity >= lower) & (ambiguity < upper)
        count = int(selected.sum())
        if count:
            result[name] = {
                "accuracy": _accuracy(predictions[selected], labels[selected]),
                "count": count,
                "name_count": int(table["name"].iloc[indices[selected]].nunique()),
            }
    return result


def leakage_comparison(table: pd.DataFrame, test_size=0.2, random_state=1):
    """Compare the same name-level task and model under two splits.

    The ONLY difference between the two numbers is the split: the table, target,
    and model construction are identical. This isolates leakage from the split.
    """
    names, labels = table["name"].astype(str).to_numpy(), table["label"].astype(str).to_numpy()
    train, test = split_by_name(names, test_size=test_size, random_state=random_state)
    grouped = build_pipeline(random_state=random_state).fit(names[train], labels[train]).score(names[test], labels[test])
    row_train, row_test = train_test_split(
        np.arange(len(table)), test_size=test_size, random_state=random_state, stratify=labels
    )
    leaky = build_pipeline(random_state=random_state).fit(names[row_train], labels[row_train]).score(names[row_test], labels[row_test])
    return {
        "name_split": float(grouped),
        "row_split": float(leaky),
        "inflation": float(leaky - grouped),
        "name_test_count": len(test),
        "row_test_count": len(row_test),
        "name_test_name_count": int(np.unique(names[test]).size),
        "row_test_name_count": int(np.unique(names[row_test]).size),
    }


def legacy_illusion():
    """Return 1.0 because the model is fitted and scored on the same ten names."""
    model = build_pipeline().fit(LEGACY_EXAMPLE_NAMES, LEGACY_EXAMPLE_LABELS)
    return _accuracy(model.predict(LEGACY_EXAMPLE_NAMES), LEGACY_EXAMPLE_LABELS)


def format_report(table: pd.DataFrame, test_size=0.2, random_state=1):
    """Format evaluation results with sample counts beside every accuracy."""
    train, test, model, results = _fit_evaluation(table, test_size=test_size, random_state=random_state)
    breakdown = ambiguity_breakdown(table, test, model.predict(table["name"].iloc[test]))
    leakage = leakage_comparison(table, test_size=test_size, random_state=random_state)
    lines = ["Name-gender classifier evaluation (split by name)", ""]
    for result in results.values():
        lines.append(
            f"{result['method']}: accuracy={result['accuracy']:.4f} "
            f"(test rows={result['test_count']}, test names={result['test_name_count']})"
        )
    lines.append("\nAmbiguity breakdown:")
    for bucket, result in breakdown.items():
        counts = f"(test rows={result['count']}, test names={result['name_count']})"
        if result["count"] < MIN_INTERPRETABLE_BUCKET_ROWS:
            lines.append(f"{bucket}: too small to interpret {counts}")
        else:
            lines.append(f"{bucket}: accuracy={result['accuracy']:.4f} {counts}")
    lines.append(
        f"\nLeakage comparison (same table, target, and model; only the split differs): "
        f"name split accuracy={leakage['name_split']:.4f} "
        f"(test rows={leakage['name_test_count']}, test names={leakage['name_test_name_count']}), "
        f"row split accuracy={leakage['row_split']:.4f} "
        f"(test rows={leakage['row_test_count']}, test names={leakage['row_test_name_count']}), "
        f"inflation={leakage['inflation']:.4f}"
    )
    lines.append(
        f"Legacy illusion: accuracy={legacy_illusion():.4f} "
        f"(test rows={len(LEGACY_EXAMPLE_NAMES)}, test names={len(set(LEGACY_EXAMPLE_NAMES))}, same data fit and score)"
    )
    return "\n".join(lines)
