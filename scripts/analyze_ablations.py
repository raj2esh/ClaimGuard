"""Step 21: aggregate the controlled ablation results. Combines NEWLY
measured data (Conditions A/B/C, on a deterministic RAGTruth subset -
`ablation_response_results.jsonl`) with REUSED, already-validated Step 19
artifacts (Conditions D/E, and full-2700-response context numbers) -
never recomputing what Step 19 already measured.

Usage:
    python scripts/analyze_ablations.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.evaluation import ragtruth_eval as rt_eval  # noqa: E402
from claimguard.evaluation import ablation_eval as ab_eval  # noqa: E402

ABLATION_DIR = cg_config.resolve_path("data/processed/evaluation/ablations")
RAGTRUTH_DIR = cg_config.resolve_path("data/processed/evaluation/ragtruth")
TRUTHFULQA_DIR = cg_config.resolve_path("data/processed/evaluation/truthfulqa")

RESPONSE_JSONL_PATH = ABLATION_DIR / "ablation_response_results.jsonl"
RUN_MANIFEST_PATH = ABLATION_DIR / "ablation_run_manifest.json"
RESULTS_PATH = ABLATION_DIR / "ablation_results.json"
COMPARISON_CSV_PATH = ABLATION_DIR / "ablation_comparison.csv"
ERROR_ANALYSIS_PATH = ABLATION_DIR / "ablation_error_analysis.json"
MANIFEST_PATH = ABLATION_DIR / "ablation_manifest.json"

CONDITIONS = ("A_reranked", "B_no_reranker", "C_faiss_top1")
N_ERROR_EXAMPLES = 5


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _condition_metrics(records: list[dict], condition: str) -> dict:
    gold = [r["gold_has_hallucination"] for r in records]
    mapped0 = [r["conditions"][condition]["attempt0_mapped_decision"] for r in records]
    final_dec = [r["conditions"][condition]["final_decision"] for r in records]
    correction_occurred = [r["conditions"][condition]["correction_occurred"] for r in records]
    max_attempts_flags = [r["conditions"][condition]["termination_reason"] == "max_attempts_reached" for r in records]
    confidences = [r["conditions"][condition]["attempt0_confidence"] for r in records]

    attempt0_pred = [rt_eval.attempt0_detection_prediction(d) == "hallucinated" for d in mapped0]
    final_pred = [rt_eval.final_outcome_prediction(d) == "hallucinated" for d in final_dec]

    categories = [rt_eval.categorize_correction_outcome(g, c, f) for g, c, f in zip(gold, correction_occurred, final_dec)]
    from collections import Counter
    cat_counts = Counter(categories)
    n_unnecessary = sum(1 for g, c in zip(gold, correction_occurred) if rt_eval.is_unnecessary_correction(g, c))
    n_failed = sum(1 for r in records if r["conditions"][condition]["status"] == "failed")

    return {
        "n": len(records),
        "attempt0_detection_metrics": rt_eval.detection_metrics(gold, attempt0_pred),
        "final_outcome_metrics": rt_eval.detection_metrics(gold, final_pred),
        "decision_rates": rt_eval.decision_rate_breakdown(mapped0, final_dec, correction_occurred, max_attempts_flags),
        "correction_outcome_categories": dict(cat_counts),
        "unnecessary_correction_count": n_unnecessary,
        "n_failed_responses": n_failed,
        "confidence_by_outcome": ab_eval.confidence_by_outcome(mapped0, confidences, gold),
    }


def main() -> int:
    _section("LOADING NEW ABLATION DATA (Conditions A/B/C, subset)")
    records = load_jsonl(RESPONSE_JSONL_PATH)
    with RUN_MANIFEST_PATH.open() as f:
        run_manifest = json.load(f)
    n = len(records)
    print(f"Loaded {n} response records (subset of RAGTruth test), from {RESPONSE_JSONL_PATH}")

    _section("LOADING REUSED STEP 19 ARTIFACTS (Conditions D/E, full 2700-response context)")
    with (RAGTRUTH_DIR / "ragtruth_eval_results.json").open() as f:
        step19_results = json.load(f)
    step19_full_records = load_jsonl(RAGTRUTH_DIR / "ragtruth_response_results.jsonl")
    step19_by_id = {r["response_id"]: r for r in step19_full_records}
    print(f"Step 19 full results loaded: {step19_results['manifest_summary']['n_responses_evaluated']} responses")

    _section("CROSS-VALIDATION: subset Condition A vs Step 19's full-run stored results (same response_ids)")
    matched, mismatched = 0, 0
    mismatch_examples = []
    for r in records:
        rid = r["response_id"]
        step19_rec = step19_by_id.get(rid)
        if step19_rec is None:
            continue
        a_final = r["conditions"]["A_reranked"]["final_decision"]
        step19_final = step19_rec["conditions"]["B_generated"]["final_decision"]
        if a_final == step19_final:
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_examples) < N_ERROR_EXAMPLES:
                mismatch_examples.append({"response_id": rid, "ablation_A_final": a_final, "step19_final": step19_final})
    cross_validation = {
        "n_compared": matched + mismatched, "matched": matched, "mismatched": mismatched,
        "match_rate": matched / (matched + mismatched) if (matched + mismatched) else None,
        "mismatch_examples": mismatch_examples,
        "note": ("Condition A reruns the SAME architecture as Step 19's Condition B_generated on the "
                 "same response_ids, using the same frozen deterministic configuration - a high match "
                 "rate confirms determinism/consistency across independent runs, not a new finding."),
    }
    print(json.dumps(cross_validation, indent=2))

    _section("CONDITION METRICS: A (reranked), B (no-reranker), C (FAISS top-1) - subset, n=%d" % n)
    condition_metrics = {}
    for cond in CONDITIONS:
        condition_metrics[cond] = _condition_metrics(records, cond)
        m = condition_metrics[cond]["attempt0_detection_metrics"]
        print(f"  {cond}: accuracy={m['accuracy']} macro_f1={m['macro_f1']} hallucinated_recall={m['hallucinated_recall']}")

    _section("CONDITION D (No-correction) and E (Verify-only) - REUSED from Step 19, zero new inference")
    condition_D = step19_results["overall_metrics_by_condition"]["B_generated"]["attempt0_detection_metrics"]
    condition_E = step19_results["overall_metrics_by_condition"]["C_verify_only"]["attempt0_detection_metrics"]
    print(f"  D (no-correction, n=2700, reused): accuracy={condition_D['accuracy']} macro_f1={condition_D['macro_f1']}")
    print(f"  E (verify-only, n=2700, reused): accuracy={condition_E['accuracy']} macro_f1={condition_E['macro_f1']}")

    _section("EVIDENCE MOVEMENT: does the reranker change WHICH evidence gets selected?")
    a_evidence = [r["conditions"]["A_reranked"]["attempts"][0]["evidence_corpus_ids"] for r in records]
    b_evidence = [r["conditions"]["B_no_reranker"]["attempts"][0]["evidence_corpus_ids"] for r in records]
    c_evidence = [r["conditions"]["C_faiss_top1"]["attempts"][0]["evidence_corpus_ids"] for r in records]
    movement_a_vs_b = ab_eval.evidence_movement_rate(a_evidence, b_evidence)
    movement_a_vs_c = ab_eval.evidence_movement_rate(a_evidence, c_evidence)
    print(f"  A vs B (reranked vs no-reranker) evidence-set changed rate: {movement_a_vs_b['changed_rate']}")
    print(f"  A vs C (reranked vs FAISS-top1) evidence-set changed rate: {movement_a_vs_c['changed_rate']}")

    _section("VERIFIER OVERCONFIDENCE: does the Step 15/16 pattern persist on RAGTruth? (full 2700, reused)")
    full_mapped0 = [r["conditions"]["B_generated"]["attempt0_mapped_decision"] for r in step19_full_records]
    full_conf = [r["conditions"]["B_generated"]["attempt0_confidence"] for r in step19_full_records]
    full_gold = [r["gold_has_hallucination"] for r in step19_full_records]
    full_confidence_by_outcome = ab_eval.confidence_by_outcome(full_mapped0, full_conf, full_gold)
    print(json.dumps(full_confidence_by_outcome, indent=2))

    _section("TRUTHFULQA SECONDARY OBSERVATIONS (reused from Step 20, zero new inference)")
    truthfulqa_secondary = {}
    tqa_results_path = TRUTHFULQA_DIR / "truthfulqa_eval_results.json"
    if tqa_results_path.exists():
        with tqa_results_path.open() as f:
            tqa = json.load(f)
        truthfulqa_secondary = {
            "mc_scores_overall": tqa["mc_scores_overall"],
            "free_form_decision_rates_overall": tqa["free_form_decision_rates_overall"],
            "note": "Reused directly from Step 20's truthfulqa_eval_results.json - not recomputed, not rerun.",
        }
        print(json.dumps(truthfulqa_secondary["mc_scores_overall"], indent=2))
    else:
        print("Step 20 TruthfulQA results not found - skipping (not required to rerun).")

    _section("ERROR ANALYSIS (deterministic selection, failure-source separated)")

    def top_by_response_id(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda r: r["response_id"])[:N_ERROR_EXAMPLES]

    def _preview(r: dict, cond: str) -> dict:
        c = r["conditions"][cond]
        return {
            "response_id": r["response_id"], "task_type": r["task_type"], "model": r["model"],
            "gold_has_hallucination": r["gold_has_hallucination"], "attempt0_mapped_decision": c["attempt0_mapped_decision"],
            "attempt0_confidence": c["attempt0_confidence"], "final_decision": c["final_decision"],
            "answer_preview": c["final_answer_preview"],
        }

    # 1. Retrieval failure: zero evidence returned (measurable directly).
    retrieval_failures = [r for r in records if r["conditions"]["A_reranked"]["attempt0_num_evidence"] == 0]
    # 2. Retrieval relevance failure: NOT identifiable without gold evidence annotations for RAGTruth.
    # 3. Verifier failure proxy: high confidence (>=0.9) but wrong (gold=False & CORRECT, or gold=True & ACCEPT).
    def is_verifier_overconfident_wrong(r, cond):
        c = r["conditions"][cond]
        gold = r["gold_has_hallucination"]
        wrong = (c["attempt0_mapped_decision"] == "ACCEPT" and gold) or (c["attempt0_mapped_decision"] == "CORRECT" and not gold)
        return wrong and (c["attempt0_confidence"] or 0.0) >= 0.9
    verifier_overconfident_wrong = [r for r in records if is_verifier_overconfident_wrong(r, "A_reranked")]
    # 4. Decision-policy failure proxy: zero-abstain-ever, confirmed directly from decision rates above.
    # 5. Correction failure: harmful correction (not_hallucinated_degraded category), Condition A.
    harmful_correction = [
        r for r in records
        if rt_eval.categorize_correction_outcome(
            r["gold_has_hallucination"], r["conditions"]["A_reranked"]["correction_occurred"],
            r["conditions"]["A_reranked"]["final_decision"],
        ) == "not_hallucinated_degraded"
    ]

    error_analysis = {
        "failure_source_separation": {
            "1_retrieval_failure_zero_evidence": {
                "count": len(retrieval_failures), "n": n,
                "examples": [_preview(r, "A_reranked") for r in top_by_response_id(retrieval_failures)],
            },
            "2_retrieval_relevance_failure": {
                "count": None,
                "note": ("NOT IDENTIFIABLE from the current evaluation - RAGTruth provides no per-"
                         "evidence-item gold-relevance annotation analogous to FEVER's gold evidence "
                         "sets, so whether retrieved-but-unflagged evidence is topically relevant "
                         "cannot be determined without inventing a semantic relevance metric, which "
                         "Step 21 instructions explicitly forbid."),
            },
            "3_verifier_overconfident_wrong": {
                "count": len(verifier_overconfident_wrong), "n": n,
                "examples": [_preview(r, "A_reranked") for r in top_by_response_id(verifier_overconfident_wrong)],
            },
            "4_decision_policy_zero_abstain": {
                "note": "See decision_rates above - attempt0_abstain_rate is 0.0 in ALL conditions "
                        "(A/B/C on this subset, and D/E on the full 2700) - the decision policy never "
                        "abstains regardless of evidence-selection method.",
            },
            "5_correction_failure_harmful": {
                "count": len(harmful_correction), "n": n,
                "examples": [_preview(r, "A_reranked") for r in top_by_response_id(harmful_correction)],
            },
        },
        "cross_validation": cross_validation,
    }
    with ERROR_ANALYSIS_PATH.open("w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, default=str)
    print(f"Saved {ERROR_ANALYSIS_PATH}")

    _section("SAVING AGGREGATE RESULTS")
    results = {
        "run_manifest_summary": {
            "subset_size": run_manifest["subset_size"], "full_ragtruth_test_size": run_manifest["full_ragtruth_test_size"],
            "model_identifiers": run_manifest["model_identifiers"], "frozen_config": run_manifest["frozen_config"],
            "ablation_conditions": run_manifest["ablation_conditions"],
        },
        "condition_metrics_A_B_C_subset": condition_metrics,
        "condition_D_no_correction_reused_full_2700": condition_D,
        "condition_E_verify_only_reused_full_2700": condition_E,
        "cross_validation_subset_A_vs_step19_full": cross_validation,
        "evidence_movement": {"A_vs_B_no_reranker": movement_a_vs_b, "A_vs_C_faiss_top1": movement_a_vs_c},
        "verifier_overconfidence_full_2700_reused": full_confidence_by_outcome,
        "truthfulqa_secondary_observations": truthfulqa_secondary,
        "reproducibility_check": run_manifest["reproducibility_check"],
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {RESULTS_PATH}")

    _section("WRITING COMPARISON CSV")
    with COMPARISON_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "n", "accuracy", "macro_f1", "hallucinated_recall", "hallucinated_precision",
                          "not_hallucinated_recall", "attempt0_abstain_rate", "correction_success_rate_among_attempted",
                          "max_attempts_reached_rate"])
        for cond in CONDITIONS:
            m = condition_metrics[cond]["attempt0_detection_metrics"]
            d = condition_metrics[cond]["decision_rates"]
            writer.writerow([cond, m["n"], m["accuracy"], m["macro_f1"], m["hallucinated_recall"],
                              m["hallucinated_precision"], m["not_hallucinated_recall"],
                              d["attempt0_abstain_rate"], d["correction_success_rate_among_attempted"],
                              d["max_attempts_reached_rate"]])
        d_rates = step19_results["overall_metrics_by_condition"]["B_generated"]["decision_rates"]
        writer.writerow(["D_no_correction (reused, n=2700)", condition_D["n"], condition_D["accuracy"],
                          condition_D["macro_f1"], condition_D["hallucinated_recall"], condition_D["hallucinated_precision"],
                          condition_D["not_hallucinated_recall"], d_rates["attempt0_abstain_rate"], None, None])
        e_rates = step19_results["overall_metrics_by_condition"]["C_verify_only"]["decision_rates"]
        writer.writerow(["E_verify_only (reused, n=2700)", condition_E["n"], condition_E["accuracy"],
                          condition_E["macro_f1"], condition_E["hallucinated_recall"], condition_E["hallucinated_precision"],
                          condition_E["not_hallucinated_recall"], e_rates["attempt0_abstain_rate"], None, None])
    print(f"Saved {COMPARISON_CSV_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
