"""ClaimGuard verifier: evaluation metrics.

Computes accuracy, macro/weighted F1, per-class precision/recall/F1, and a
confusion matrix - not accuracy alone (Step 9 instruction).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(labels: np.ndarray, preds: np.ndarray, id2label: dict[int, str]) -> dict[str, Any]:
    label_ids = sorted(id2label.keys())
    label_names = [id2label[i] for i in label_ids]

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", labels=label_ids, zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", labels=label_ids, zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=label_ids, zero_division=0
    )
    per_class = {
        label_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(label_ids))
    }

    cm = confusion_matrix(labels, preds, labels=label_ids)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": label_names,
        "num_examples": int(len(labels)),
    }


def compute_metrics_by_group(
    labels: np.ndarray, preds: np.ndarray, id2label: dict[int, str], group_keys: list[str]
) -> dict[str, Any]:
    """Same metrics as compute_metrics(), computed separately for each
    distinct value in `group_keys` (parallel array to labels/preds) - so
    aggregate dev performance can never silently hide a group with much
    worse behavior (Step 12: used to break dev metrics down by
    source_dataset/source_subset, e.g. 'fever', 'halueval.qa', etc.).
    Groups with 0 examples are omitted, never reported with fabricated
    zero-support metrics.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    group_keys = np.asarray(group_keys)
    result: dict[str, Any] = {}
    for group in sorted(set(group_keys.tolist())):
        mask = group_keys == group
        if not mask.any():
            continue
        result[group] = compute_metrics(labels[mask], preds[mask], id2label)
    return result
