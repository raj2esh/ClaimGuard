"""Step 21: ablation-specific pure diagnostic functions. Reuses
`claimguard.evaluation.ragtruth_eval` for all standard detection/decision-
rate/correction-outcome arithmetic (not duplicated here) - this module
adds ONLY the confidence-by-outcome breakdown needed to check whether the
Step 15/16 verifier-overconfidence pattern persists on RAGTruth data.

Pure arithmetic only - no model calls, no inference-path imports.
"""

from __future__ import annotations

import statistics
from typing import Any


def confidence_by_outcome(
    mapped_decisions: list[str], confidences: list[float], gold_hallucinated: list[bool],
) -> dict[str, Any]:
    """Splits verifier confidence into 4 groups by (predicted decision x
    actual gold label) - the diagnostic Step 16 used for FEVER
    (gold-SUPPORTS/REFUTES/non-gold entailment means), adapted to
    RAGTruth's available ground truth (RAGTruth has no per-evidence gold-
    support annotation like FEVER, so this groups by the RESPONSE-level
    gold_has_hallucination label instead):

        ACCEPT & gold=False   -> correct accept (true negative)
        ACCEPT & gold=True    -> WRONG accept (false negative - missed hallucination)
        CORRECT & gold=False  -> WRONG flag (false positive - unnecessary correction)
        CORRECT & gold=True   -> correct flag (true positive)

    If confidence is high across ALL FOUR groups regardless of
    correctness, that is the same overconfidence signature Step 16 found
    on FEVER, now checked on RAGTruth."""
    groups: dict[str, list[float]] = {
        "accept_correct_negative": [], "accept_wrong_missed_hallucination": [],
        "correct_wrong_false_positive": [], "correct_correct_positive": [],
    }
    for decision, conf, gold in zip(mapped_decisions, confidences, gold_hallucinated):
        if decision == "ACCEPT" and not gold:
            groups["accept_correct_negative"].append(conf)
        elif decision == "ACCEPT" and gold:
            groups["accept_wrong_missed_hallucination"].append(conf)
        elif decision == "CORRECT" and not gold:
            groups["correct_wrong_false_positive"].append(conf)
        elif decision == "CORRECT" and gold:
            groups["correct_correct_positive"].append(conf)
    return {
        name: {
            "n": len(vals), "mean_confidence": statistics.mean(vals) if vals else None,
            "median_confidence": statistics.median(vals) if vals else None,
        }
        for name, vals in groups.items()
    }


def evidence_movement_rate(baseline_evidence_ids: list[list[str]], comparison_evidence_ids: list[list[str]]) -> dict[str, Any]:
    """Fraction of responses whose SELECTED-evidence set changed at all
    between two evidence-selection conditions (e.g. reranked vs
    no-reranker) - a simple, non-fabricated measure of how much the
    reranker actually moves the final candidate set, independent of
    whether that movement helped or hurt."""
    if len(baseline_evidence_ids) != len(comparison_evidence_ids):
        raise ValueError("baseline and comparison lists must be the same length (paired responses).")
    n = len(baseline_evidence_ids)
    if n == 0:
        return {"n": 0, "changed_rate": None}
    changed = sum(1 for a, b in zip(baseline_evidence_ids, comparison_evidence_ids) if set(a) != set(b))
    return {"n": n, "changed_rate": changed / n, "n_changed": changed}
