"""Step 19: aggregate metrics, breakdowns, and error analysis over the
per-response RAGTruth evaluation results already saved by
`scripts/evaluate_ragtruth.py` (`ragtruth_response_results.jsonl`).

Deliberately separate from the expensive GPU inference pass: aggregation
is pure Python over already-computed per-response records, so metric
bugs can be fixed and rerun in seconds without re-running 2,700 x 3
`run_correction_loop` calls. Reads gold `gold_has_hallucination` fields
that are ALREADY IN the saved JSONL (written there by the evaluation
script after inference completed) - this script never touches a model or
makes an inference call.

Usage:
    python scripts/analyze_ragtruth_results.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.evaluation import ragtruth_eval as eval_mod  # noqa: E402

OUT_DIR = cg_config.resolve_path("data/processed/evaluation/ragtruth")
RESPONSE_JSONL_PATH = OUT_DIR / "ragtruth_response_results.jsonl"
MANIFEST_PATH = OUT_DIR / "ragtruth_eval_manifest.json"
RESULTS_PATH = OUT_DIR / "ragtruth_eval_results.json"
ERROR_ANALYSIS_PATH = OUT_DIR / "ragtruth_error_analysis.json"

N_ERROR_EXAMPLES_PER_CATEGORY = 5
PRIMARY_CONDITION = "B_generated"  # the true end-to-end generate+verify+correct condition


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_records() -> list[dict]:
    records = []
    with RESPONSE_JSONL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _condition_metrics(records: list[dict], condition: str) -> dict:
    gold = [r["gold_has_hallucination"] for r in records]
    attempt0_pred = [
        eval_mod.attempt0_detection_prediction(r["conditions"][condition]["attempt0_mapped_decision"]) == "hallucinated"
        for r in records
    ]
    final_pred = [
        eval_mod.final_outcome_prediction(r["conditions"][condition]["final_decision"]) == "hallucinated"
        for r in records
    ]
    mapped0 = [r["conditions"][condition]["attempt0_mapped_decision"] for r in records]
    final_dec = [r["conditions"][condition]["final_decision"] for r in records]
    correction_occurred = [r["conditions"][condition]["correction_occurred"] for r in records]
    max_attempts_flags = [r["conditions"][condition]["termination_reason"] == "max_attempts_reached" for r in records]

    categories = [
        eval_mod.categorize_correction_outcome(g, c, f)
        for g, c, f in zip(gold, correction_occurred, final_dec)
    ]
    cat_counts = defaultdict(int)
    for c in categories:
        cat_counts[c] += 1
    n_unnecessary = sum(
        1 for g, c in zip(gold, correction_occurred) if eval_mod.is_unnecessary_correction(g, c)
    )
    n_failed = sum(1 for r in records if r["conditions"][condition]["status"] == "failed")

    return {
        "n": len(records),
        "hallucination_prevalence_gold": sum(gold) / len(gold) if gold else None,
        "attempt0_detection_metrics": eval_mod.detection_metrics(gold, attempt0_pred),
        "final_outcome_metrics": eval_mod.detection_metrics(gold, final_pred),
        "decision_rates": eval_mod.decision_rate_breakdown(mapped0, final_dec, correction_occurred, max_attempts_flags),
        "correction_outcome_categories": dict(cat_counts),
        "unnecessary_correction_count": n_unnecessary,
        "n_failed_responses": n_failed,
    }


def _latency_stats(records: list[dict], condition: str) -> dict:
    latencies = [r["conditions"][condition]["total_latency_seconds"] for r in records]
    if not latencies:
        return {}
    s = sorted(latencies)
    return {
        "mean_seconds": statistics.mean(latencies), "median_seconds": statistics.median(latencies),
        "p95_seconds": s[int(len(s) * 0.95)] if len(s) > 1 else s[0],
        "min_seconds": s[0], "max_seconds": s[-1], "total_seconds": sum(latencies),
    }


def _breakdown_by(records: list[dict], key: str, condition: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return {
        group_value: {
            **_condition_metrics(group_records, condition),
            "note": None if len(group_records) >= 30 else "small sample - interpret with caution",
        }
        for group_value, group_records in sorted(groups.items())
    }


def _error_analysis(records: list[dict], condition: str) -> dict:
    """Deterministic selection rules only - never manually cherry-picked.
    Sorted by confidence (for confidence-based categories) or response_id
    (stable, for all others), then truncated to the top N."""

    def top_by_confidence(items: list[dict], descending: bool = True) -> list[dict]:
        return sorted(
            items, key=lambda r: (r["conditions"][condition]["attempt0_confidence"] or 0.0), reverse=descending,
        )[:N_ERROR_EXAMPLES_PER_CATEGORY]

    def top_by_response_id(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda r: r["response_id"])[:N_ERROR_EXAMPLES_PER_CATEGORY]

    def _preview(r: dict) -> dict:
        c = r["conditions"][condition]
        return {
            "response_id": r["response_id"], "task_type": r["task_type"], "model": r["model"],
            "gold_has_hallucination": r["gold_has_hallucination"],
            "attempt0_mapped_decision": c["attempt0_mapped_decision"], "attempt0_confidence": c["attempt0_confidence"],
            "attempt0_num_evidence": c["attempt0_num_evidence"], "final_decision": c["final_decision"],
            "termination_reason": c["termination_reason"], "answer_preview": c["final_answer_preview"],
        }

    gold = lambda r: r["gold_has_hallucination"]  # noqa: E731
    pred_h = lambda r: eval_mod.attempt0_detection_prediction(  # noqa: E731
        r["conditions"][condition]["attempt0_mapped_decision"]
    ) == "hallucinated"

    false_positives = [r for r in records if not gold(r) and pred_h(r)]
    false_negatives = [r for r in records if gold(r) and not pred_h(r)]
    harmful_correction = [
        r for r in records
        if eval_mod.categorize_correction_outcome(
            gold(r), r["conditions"][condition]["correction_occurred"], r["conditions"][condition]["final_decision"],
        ) == "not_hallucinated_degraded"
    ]
    unnecessary_correction = [
        r for r in records if eval_mod.is_unnecessary_correction(gold(r), r["conditions"][condition]["correction_occurred"])
    ]
    max_attempts_cases = [r for r in records if r["conditions"][condition]["termination_reason"] == "max_attempts_reached"]
    retrieval_failures = [r for r in records if r["conditions"][condition]["attempt0_num_evidence"] == 0]
    high_confidence_wrong = [
        r for r in records
        if (r["conditions"][condition]["attempt0_confidence"] or 0.0) >= 0.9 and (gold(r) != pred_h(r))
    ]
    inference_failures = [r for r in records if r["conditions"][condition]["status"] == "failed"]

    return {
        "condition_analyzed": condition,
        "selection_rule": (
            "Confidence-sorted categories take the N highest-confidence examples (descending "
            "attempt0_confidence); all other categories take the first N by response_id after a "
            "stable sort. No manual example selection anywhere in this file."
        ),
        "false_positives_highest_confidence": [_preview(r) for r in top_by_confidence(false_positives)],
        "false_positives_total_count": len(false_positives),
        "false_negatives_highest_confidence": [_preview(r) for r in top_by_confidence(false_negatives)],
        "false_negatives_total_count": len(false_negatives),
        "harmful_correction_examples": [_preview(r) for r in top_by_response_id(harmful_correction)],
        "harmful_correction_total_count": len(harmful_correction),
        "unnecessary_correction_examples": [_preview(r) for r in top_by_response_id(unnecessary_correction)],
        "unnecessary_correction_total_count": len(unnecessary_correction),
        "max_attempts_examples": [_preview(r) for r in top_by_response_id(max_attempts_cases)],
        "max_attempts_total_count": len(max_attempts_cases),
        "retrieval_failure_examples": [_preview(r) for r in top_by_response_id(retrieval_failures)],
        "retrieval_failure_total_count": len(retrieval_failures),
        "high_confidence_wrong_examples": [_preview(r) for r in top_by_confidence(high_confidence_wrong)],
        "high_confidence_wrong_total_count": len(high_confidence_wrong),
        "inference_failure_examples": [_preview(r) for r in top_by_response_id(inference_failures)],
        "inference_failure_total_count": len(inference_failures),
        "generator_degradation_note": (
            "Not independently measured - detecting whether a corrected answer's TEXT QUALITY "
            "(as opposed to its verifier-judged support) declined would require a dedicated "
            "quality-comparison mechanism this evaluation does not implement. The "
            "'not_hallucinated_degraded' category is the closest proxy available and is reported "
            "under harmful_correction_examples above."
        ),
    }


def main() -> int:
    _section("LOADING PER-RESPONSE RESULTS")
    records = load_records()
    print(f"Loaded {len(records)} response records from {RESPONSE_JSONL_PATH}")
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)

    _section("OVERALL METRICS PER CONDITION")
    overall = {}
    latency = {}
    for cond in ("B_generated", "C_verify_only", "D_original_full_loop"):
        overall[cond] = _condition_metrics(records, cond)
        latency[cond] = _latency_stats(records, cond)
        m = overall[cond]["attempt0_detection_metrics"]
        print(f"  {cond}: n={overall[cond]['n']} accuracy={m['accuracy']} macro_f1={m['macro_f1']} "
              f"hallucinated_f1={m['hallucinated_f1']}")
    overall["A_original_full_loop"] = overall["D_original_full_loop"]
    overall["A_note"] = "Identical to D as literally specified - see manifest['conditions']['A_note']."

    _section("TASK-TYPE BREAKDOWN (Condition B)")
    task_breakdown = _breakdown_by(records, "task_type", PRIMARY_CONDITION)
    for k, v in task_breakdown.items():
        print(f"  {k}: n={v['n']} prevalence={v['hallucination_prevalence_gold']:.3f} "
              f"macro_f1={v['attempt0_detection_metrics']['macro_f1']}")

    _section("SOURCE-MODEL BREAKDOWN (Condition B)")
    model_breakdown = _breakdown_by(records, "model", PRIMARY_CONDITION)
    for k, v in model_breakdown.items():
        print(f"  {k}: n={v['n']} prevalence={v['hallucination_prevalence_gold']:.3f} "
              f"macro_f1={v['attempt0_detection_metrics']['macro_f1']}")

    _section("RETRIEVAL DIAGNOSTICS (Condition B, attempt 0)")
    n_no_evidence = sum(1 for r in records if r["conditions"][PRIMARY_CONDITION]["attempt0_num_evidence"] == 0)
    evidence_counts = [r["conditions"][PRIMARY_CONDITION]["attempt0_num_evidence"] for r in records]
    retrieval_diag = {
        "n_zero_evidence_retrieved": n_no_evidence,
        "mean_evidence_count": statistics.mean(evidence_counts) if evidence_counts else None,
        "note": (
            "ClaimGuard's retrieval corpus (Step 13) is built exclusively from FEVER's Wikipedia "
            "evidence pages. RAGTruth's actual source documents (news articles, business listings, "
            "QA passages) are NOT in this corpus. Retrieval is therefore expected to surface "
            "topically-unrelated Wikipedia sentences rather than the response's true source "
            "material for most RAGTruth responses - this is a known architectural mismatch, not a "
            "retrieval bug, and is the primary lens for interpreting detection quality below."
        ),
    }
    print(json.dumps(retrieval_diag, indent=2))

    _section("SPAN-LEVEL EVALUATION")
    span_note = {
        "computed": False,
        "reason": (
            "ClaimGuard's decision policy (Step 16) operates at the whole-candidate-answer level - "
            "it has no mechanism to predict WHICH substring of an answer is unsupported. Computing "
            "span precision/recall/F1 would require inventing an arbitrary span-alignment heuristic "
            "(e.g. attributing the verifier's single answer-level score to an arbitrary substring) "
            "that ClaimGuard's actual architecture does not support. Reported as a limitation, not "
            "fabricated, per Step 19 Section 8's explicit instruction."
        ),
    }
    print(json.dumps(span_note, indent=2))

    _section("ERROR ANALYSIS (Condition B, deterministic selection)")
    error_analysis = _error_analysis(records, PRIMARY_CONDITION)
    print(f"  false_positives: {error_analysis['false_positives_total_count']}, "
          f"false_negatives: {error_analysis['false_negatives_total_count']}, "
          f"harmful_correction: {error_analysis['harmful_correction_total_count']}, "
          f"unnecessary_correction: {error_analysis['unnecessary_correction_total_count']}, "
          f"max_attempts: {error_analysis['max_attempts_total_count']}, "
          f"retrieval_failures: {error_analysis['retrieval_failure_total_count']}, "
          f"high_confidence_wrong: {error_analysis['high_confidence_wrong_total_count']}, "
          f"inference_failures: {error_analysis['inference_failure_total_count']}")
    with ERROR_ANALYSIS_PATH.open("w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, default=str)
    print(f"Saved {ERROR_ANALYSIS_PATH}")

    _section("DATA LEAKAGE / ISOLATION AUDIT")
    leakage_audit = {
        "ragtruth_test_used_in_training": False,
        "ragtruth_test_used_in_tuning": False,
        "ragtruth_test_used_to_choose_thresholds": False,
        "truthfulqa_touched": False,
        "gold_ragtruth_labels_spans_entered_inference": False,
        "fever_gold_evidence_supplied_at_inference": False,
        "test_labels_influenced_generation": False,
        "test_labels_influenced_correction": False,
        "note": (
            "run_correction_loop() and its Step 19 caller receive only `prompt`/`response_text` - "
            "gold_has_hallucination/labels are read only by this analysis script's own scoring "
            "functions, after all inference completed. See tests/test_ragtruth_evaluation.py's "
            "structural guard (claimguard.evaluation imports no inference-path module) and its "
            "mock-based test confirming gold fields never appear in run_correction_loop's call args."
        ),
    }
    print(json.dumps(leakage_audit, indent=2))

    _section("SAVING AGGREGATE RESULTS")
    results = {
        "manifest_summary": {
            "n_responses_evaluated": manifest["n_responses_evaluated"], "pilot_mode": manifest["pilot_mode"],
            "model_identifiers": manifest["model_identifiers"], "frozen_config": manifest["frozen_config"],
        },
        "overall_metrics_by_condition": overall,
        "latency_by_condition": latency,
        "task_type_breakdown_condition_B": task_breakdown,
        "source_model_breakdown_condition_B": model_breakdown,
        "retrieval_diagnostics": retrieval_diag,
        "span_level_evaluation": span_note,
        "reproducibility_check": manifest["reproducibility_check"],
        "leakage_isolation_audit": leakage_audit,
        "development_vs_final": {
            "note": (
                "These are FINAL TEST RESULTS on the reserved RAGTruth test split (2,700 "
                "responses, 450 source items, source-level train/test separation - see manifest). "
                "No development/validation split of RAGTruth test was used; all thresholds, prompts, "
                "and configuration were frozen BEFORE this evaluation ran and are not altered based "
                "on these results (Step 19 Section 21 - no post-hoc tuning)."
            ),
        },
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
