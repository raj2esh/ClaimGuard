"""Calibration diagnostics for the Step 12 binary verifier's probabilities
(Step 16). No temperature scaling, no retraining - purely descriptive
measurement of how well the verifier's own confidence matches its
accuracy.

## Scope: gold-evidence candidates only (documented, not silently assumed)

A calibration/"correctness" analysis requires a ground-truth label for
each individual (premise, hypothesis) pair the verifier scored. FEVER
only supplies that ground truth for GOLD evidence: a gold-SUPPORTS
candidate should be predicted entailment, a gold-REFUTES candidate should
be predicted contradiction. A non-gold candidate paired with a claim has
no canonical entailment/contradiction ground truth in FEVER's own
annotation - the annotators never judged whether that specific, likely
irrelevant, sentence entails or contradicts the claim. Fabricating a
"correct" label for non-gold pairs would misrepresent measurement as
ground truth, so `expected_calibration_error` and `brier_score` below are
computed ONLY over gold-evidence candidates. Non-gold evidence confidence
is still reported elsewhere (score distributions) as a purely descriptive
statistic, never as an accuracy-labeled calibration point.
"""

from __future__ import annotations

import math
from typing import Any

N_BINS = 10  # fixed bins: [0.0,0.1), [0.1,0.2), ..., [0.9,1.0]


def _bin_index(confidence: float, n_bins: int = N_BINS) -> int:
    idx = int(confidence * n_bins)
    return min(idx, n_bins - 1)  # confidence == 1.0 falls in the last bin


def gold_calibration_points(gold_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build (confidence, correct) points from gold-evidence candidate
    records. Each record must have `entailment_probability`,
    `contradiction_probability`, and `true_label` ("SUPPORTS"/"REFUTES").
    `confidence` = the probability of whichever class the verifier
    actually predicted for that pair (its own confidence in its own
    argmax decision, not just the entailment probability); `correct` =
    whether that argmax matches the FEVER-implied expected class."""
    points = []
    for r in gold_records:
        p_ent = r["entailment_probability"]
        p_con = r["contradiction_probability"]
        if not (math.isfinite(p_ent) and math.isfinite(p_con)):
            raise ValueError(f"Non-finite verifier probability in gold record {r.get('corpus_id')!r}")
        predicted_entailment = p_ent >= p_con
        confidence = p_ent if predicted_entailment else p_con
        expected_entailment = r["true_label"] == "SUPPORTS"
        correct = predicted_entailment == expected_entailment
        points.append({"confidence": confidence, "correct": correct})
    return points


def reliability_bins(points: list[dict[str, Any]], n_bins: int = N_BINS) -> list[dict[str, Any]]:
    """Fixed-width reliability-diagram bins over [0,1]. Each bin reports
    count, mean predicted confidence, and observed accuracy (empty bins
    are still reported with count=0, not omitted, so gaps in verifier
    confidence are visible)."""
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n_bins)]
    for p in points:
        buckets[_bin_index(p["confidence"], n_bins)].append(p)

    bins = []
    for i, bucket in enumerate(buckets):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if bucket:
            mean_conf = sum(p["confidence"] for p in bucket) / len(bucket)
            accuracy = sum(1 for p in bucket if p["correct"]) / len(bucket)
        else:
            mean_conf, accuracy = None, None
        bins.append({
            "bin_range": [lo, hi], "count": len(bucket),
            "mean_confidence": mean_conf, "accuracy": accuracy,
        })
    return bins


def expected_calibration_error(points: list[dict[str, Any]], n_bins: int = N_BINS) -> float | None:
    """ECE = sum_b (|bucket_b| / N) * |accuracy_b - mean_confidence_b|,
    over the fixed bins. Standard definition; no smoothing/temperature
    scaling. Returns None if there are no points to compute over."""
    if not points:
        return None
    bins = reliability_bins(points, n_bins)
    n = len(points)
    return sum(
        (b["count"] / n) * abs(b["accuracy"] - b["mean_confidence"])
        for b in bins if b["count"] > 0
    )


def brier_score(gold_records: list[dict[str, Any]]) -> float | None:
    """Brier score for the entailment-probability forecast against the
    binary FEVER-implied outcome (1 if true_label==SUPPORTS else 0),
    computed over gold-evidence candidates only (see module docstring).
    Lower is better; 0.0 is a perfect forecaster, 0.25 is what an
    always-predict-0.5 forecaster achieves."""
    if not gold_records:
        return None
    total = 0.0
    for r in gold_records:
        p_ent = r["entailment_probability"]
        outcome = 1.0 if r["true_label"] == "SUPPORTS" else 0.0
        total += (p_ent - outcome) ** 2
    return total / len(gold_records)


def confidence_distribution(points: list[dict[str, Any]], n_bins: int = N_BINS) -> dict[str, Any]:
    """Descriptive confidence histogram (fraction of points per bin) -
    separate from `reliability_bins`, which also carries accuracy."""
    bins = reliability_bins(points, n_bins)
    n = len(points)
    return {
        "n_points": n,
        "histogram": [{"bin_range": b["bin_range"], "fraction": (b["count"] / n if n else 0.0)} for b in bins],
    }


def is_systematically_overconfident(points: list[dict[str, Any]], n_bins: int = N_BINS) -> dict[str, Any]:
    """For each non-empty bin, whether mean_confidence > accuracy
    (overconfident) or < accuracy (underconfident). Returns per-bin
    verdicts plus an overall verdict (overconfident if a majority of
    non-empty bins, weighted by count, are overconfident)."""
    bins = reliability_bins(points, n_bins)
    non_empty = [b for b in bins if b["count"] > 0]
    if not non_empty:
        return {"bins": bins, "overall": None}
    weighted_overconf = sum(b["count"] for b in non_empty if b["mean_confidence"] > b["accuracy"])
    total = sum(b["count"] for b in non_empty)
    for b in non_empty:
        b["overconfident"] = b["mean_confidence"] > b["accuracy"]
    overall = "overconfident" if weighted_overconf / total > 0.5 else "not_systematically_overconfident"
    return {"bins": bins, "fraction_of_points_in_overconfident_bins": weighted_overconf / total, "overall": overall}
