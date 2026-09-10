"""RAGTruth end-to-end evaluation: label semantics, response-level
detection-prediction derivation, correction-outcome categorization, and
metric computation (Step 19).

## RAGTruth gold label semantics (verified against
`claimguard.datasets.ragtruth`, not assumed)

`has_hallucination` (bool, from `claimguard.datasets.ragtruth.join_records`)
is True iff the response has at least one hallucination span in `labels`.
This is RAGTruth's OWN independent binary label - it is NEVER reinterpreted
as FEVER entailment/contradiction anywhere in this module or in
`scripts/evaluate_ragtruth.py`. RAGTruth is treated as an independent
hallucination-evaluation dataset, per Step 19 Section 6.

## Scope discipline (critical)

This module contains ONLY evaluation-layer logic: label mapping, outcome
categorization, and metric arithmetic. It NEVER imports
`claimguard.correction`/`claimguard.decision`/`claimguard.generation`/
`claimguard.retrieval`/`claimguard.reranking`/`claimguard.verification`,
and none of its functions accept a model object or make an inference
call. Gold RAGTruth values (`has_hallucination`, `labels`) are consumed
ONLY by the functions in this module - never threaded into any pipeline
call. See `tests/test_ragtruth_evaluation.py` for the structural guard
confirming `claimguard.evaluation` source files never import an
inference-path module.
"""

from __future__ import annotations

from typing import Any, Literal

DetectionPrediction = Literal["hallucinated", "not_hallucinated"]

# Step 18's mapped per-attempt decision vocabulary (ACCEPT/CORRECT/ABSTAIN).
NOT_HALLUCINATED_DECISION = "ACCEPT"


def attempt0_detection_prediction(mapped_decision_attempt0: str) -> DetectionPrediction:
    """The PRIMARY, correction-uncontaminated detection signal: does the
    verifier/decision-policy layer flag this response on its FIRST pass,
    before any correction happens? `ACCEPT` -> not_hallucinated;
    `CORRECT` or `ABSTAIN` -> hallucinated (both mean "the policy did not
    simply accept this candidate as-is"). This is deliberately NOT the
    same signal as `final_outcome_prediction` below, which can differ
    after correction is applied (relevant only for conditions where
    correction runs)."""
    return "not_hallucinated" if mapped_decision_attempt0 == NOT_HALLUCINATED_DECISION else "hallucinated"


def final_outcome_prediction(final_decision: str) -> DetectionPrediction:
    """END-STATE signal (post-correction, where correction is enabled for
    the evaluated condition): `ACCEPT` -> not_hallucinated, `ABSTAIN` ->
    hallucinated. `CORRECT` is never a terminal `final_decision` (Step 18
    guarantee), so this function never needs to handle it."""
    return "not_hallucinated" if final_decision == NOT_HALLUCINATED_DECISION else "hallucinated"


def confusion_counts(gold_hallucinated: list[bool], predicted_hallucinated: list[bool]) -> dict[str, int]:
    """TP/FP/FN/TN with 'hallucinated' as the positive class. A small,
    directly-testable pure function (no sklearn dependency needed for
    this 2x2 case), so the exact convention is unambiguous and auditable
    without cross-referencing a third-party library's label-ordering
    behavior."""
    if len(gold_hallucinated) != len(predicted_hallucinated):
        raise ValueError("gold_hallucinated and predicted_hallucinated must have the same length.")
    tp = fp = fn = tn = 0
    for g, p in zip(gold_hallucinated, predicted_hallucinated):
        if g and p:
            tp += 1
        elif not g and p:
            fp += 1
        elif g and not p:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def detection_metrics(gold_hallucinated: list[bool], predicted_hallucinated: list[bool]) -> dict[str, Any]:
    """accuracy, precision/recall/F1 for BOTH the 'hallucinated' positive
    class and the symmetric 'not_hallucinated' class, plus macro F1 -
    never accuracy alone (Step 19 Section 7 explicit requirement)."""
    n = len(gold_hallucinated)
    counts = confusion_counts(gold_hallucinated, predicted_hallucinated)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    accuracy = (tp + tn) / n if n else None

    p_h, r_h, f1_h = _precision_recall_f1(tp, fp, fn)
    p_n, r_n, f1_n = _precision_recall_f1(tn, fn, fp)
    macro_f1 = (f1_h + f1_n) / 2 if f1_h is not None and f1_n is not None else None

    return {
        "n": n, "accuracy": accuracy,
        "hallucinated_precision": p_h, "hallucinated_recall": r_h, "hallucinated_f1": f1_h,
        "not_hallucinated_precision": p_n, "not_hallucinated_recall": r_n, "not_hallucinated_f1": f1_n,
        "macro_f1": macro_f1,
        "confusion_matrix": counts,
    }


def decision_rate_breakdown(mapped_decisions_attempt0: list[str], final_decisions: list[str],
                             correction_occurred_flags: list[bool], max_attempts_flags: list[bool]) -> dict[str, Any]:
    """ACCEPT/CORRECT/ABSTAIN rate reporting (Step 19 Section 7): the
    attempt-0 decision distribution (what the policy wanted to do before
    any correction), the correction rate, the max-attempts rate, and the
    final-outcome distribution (ACCEPT/ABSTAIN only, since CORRECT is
    never terminal)."""
    n = len(mapped_decisions_attempt0)
    if n == 0:
        return {"n": 0}
    accept0 = sum(1 for d in mapped_decisions_attempt0 if d == "ACCEPT")
    correct0 = sum(1 for d in mapped_decisions_attempt0 if d == "CORRECT")
    abstain0 = sum(1 for d in mapped_decisions_attempt0 if d == "ABSTAIN")
    final_accept = sum(1 for d in final_decisions if d == "ACCEPT")
    final_abstain = sum(1 for d in final_decisions if d == "ABSTAIN")
    n_corrected = sum(1 for c in correction_occurred_flags if c)
    n_correction_success = sum(
        1 for c, f in zip(correction_occurred_flags, final_decisions) if c and f == "ACCEPT"
    )
    n_max_attempts = sum(1 for m in max_attempts_flags if m)
    return {
        "n": n,
        "attempt0_accept_rate": accept0 / n, "attempt0_correct_rate": correct0 / n,
        "attempt0_abstain_rate": abstain0 / n,
        "final_accept_rate": final_accept / n, "final_abstain_rate": final_abstain / n,
        "correction_attempted_rate": n_corrected / n,
        "correction_success_rate_among_attempted": (n_correction_success / n_corrected) if n_corrected else None,
        "max_attempts_reached_rate": n_max_attempts / n,
    }


CorrectionOutcomeCategory = Literal[
    "hallucinated_corrected_successfully",
    "hallucinated_missed",
    "hallucinated_still_flagged",
    "not_hallucinated_preserved",
    "not_hallucinated_degraded",
]


def categorize_correction_outcome(gold_hallucinated: bool, correction_occurred: bool, final_decision: str) -> str:
    """Deterministic per-response categorization for the correction-
    quality analysis (Step 19 Section 9). Uses the ORIGINAL RAGTruth gold
    label ONLY to choose a bucket for an already-computed outcome - the
    label is never fed back into inference. `final_decision` is always
    `ACCEPT` or `ABSTAIN` (Step 18 guarantee: `CORRECT` is never
    terminal).

        gold=True,  final=ACCEPT,  correction_occurred=True  -> hallucinated_corrected_successfully
        gold=True,  final=ACCEPT,  correction_occurred=False -> hallucinated_missed   (never even flagged)
        gold=True,  final=ABSTAIN                             -> hallucinated_still_flagged
        gold=False, final=ACCEPT                              -> not_hallucinated_preserved
        gold=False, final=ABSTAIN                             -> not_hallucinated_degraded

    A response that was NOT hallucinated but still underwent an
    unnecessary correction attempt before ending in ACCEPT is still
    counted as `not_hallucinated_preserved` (the FINAL state is correct);
    track that separately via `unnecessary_correction` if needed - this
    function reports terminal-outcome category only, not effort spent.
    """
    if final_decision == "ABSTAIN":
        return "hallucinated_still_flagged" if gold_hallucinated else "not_hallucinated_degraded"
    if gold_hallucinated:
        return "hallucinated_corrected_successfully" if correction_occurred else "hallucinated_missed"
    return "not_hallucinated_preserved"


def is_unnecessary_correction(gold_hallucinated: bool, correction_occurred: bool) -> bool:
    """A diagnostic flag (not an outcome category): correction was
    attempted on a response RAGTruth's annotators did NOT flag as
    hallucinated."""
    return (not gold_hallucinated) and correction_occurred
