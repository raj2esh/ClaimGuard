"""Step 20: aggregate metrics, category breakdowns, and error analysis
over the per-question TruthfulQA evaluation results already saved by
`scripts/evaluate_truthfulqa.py` (`truthfulqa_response_results.jsonl`).

Separate from the expensive GPU inference pass, same convention as
Step 19's `analyze_ragtruth_results.py`. Reads gold MC target labels
that are ALREADY IN the saved JSONL - never touches a model.

Usage:
    python scripts/analyze_truthfulqa_results.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.evaluation import truthfulqa_eval as eval_mod  # noqa: E402

OUT_DIR = cg_config.resolve_path("data/processed/evaluation/truthfulqa")
RESPONSE_JSONL_PATH = OUT_DIR / "truthfulqa_response_results.jsonl"
MANIFEST_PATH = OUT_DIR / "truthfulqa_eval_manifest.json"
RESULTS_PATH = OUT_DIR / "truthfulqa_eval_results.json"
ERROR_ANALYSIS_PATH = OUT_DIR / "truthfulqa_error_analysis.json"

N_ERROR_EXAMPLES_PER_CATEGORY = 5


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


def _mc_scores_for_record(r: dict) -> dict:
    """Recompute MC1/MC2/MC0 for one record from its saved raw
    log-likelihoods + gold targets - pure arithmetic, no model call."""
    out = {}
    for mc_name, targets_key in (("mc0", "gold_mc0_targets"), ("mc1", "gold_mc1_targets"), ("mc2", "gold_mc2_targets")):
        targets = r.get(targets_key)
        lls = r["mc_scores"][mc_name]["choice_log_likelihoods"]
        if not targets or not lls:
            out[mc_name] = None
            continue
        if mc_name == "mc2":
            out[mc_name] = eval_mod.compute_mc2_score(lls, targets)
        elif mc_name == "mc1":
            out[mc_name] = eval_mod.compute_mc1_score(lls, targets)
        else:
            out[mc_name] = eval_mod.compute_mc0_score(lls, targets)
    return out


def _mc_aggregate(records: list[dict]) -> dict:
    mc0_scores, mc1_scores, mc2_scores = [], [], []
    for r in records:
        scores = _mc_scores_for_record(r)
        if scores["mc0"] is not None:
            mc0_scores.append(scores["mc0"])
        if scores["mc1"] is not None:
            mc1_scores.append(scores["mc1"])
        if scores["mc2"] is not None:
            mc2_scores.append(scores["mc2"])
    return eval_mod.aggregate_mc_scores(mc1_scores, mc2_scores, mc0_scores)


def _decision_rates(records: list[dict]) -> dict:
    mapped0 = [r["free_form"]["condition_b_attempt0_mapped_decision"] for r in records]
    final_dec = [r["free_form"]["condition_c_final_decision"] for r in records]
    correction_occurred = [r["free_form"]["correction_occurred"] for r in records]
    max_attempts_flags = [r["free_form"]["termination_reason"] == "max_attempts_reached" for r in records]
    return eval_mod.decision_rate_breakdown(mapped0, final_dec, correction_occurred, max_attempts_flags)


def _latency_stats(records: list[dict]) -> dict:
    latencies = [r["free_form"]["total_latency_seconds"] for r in records]
    if not latencies:
        return {}
    s = sorted(latencies)
    return {
        "mean_seconds": statistics.mean(latencies), "median_seconds": statistics.median(latencies),
        "p95_seconds": s[int(len(s) * 0.95)] if len(s) > 1 else s[0],
        "min_seconds": s[0], "max_seconds": s[-1], "total_seconds": sum(latencies),
    }


def _breakdown_by_category(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["category"]].append(r)
    out = {}
    for cat, group_records in sorted(groups.items()):
        out[cat] = {
            "n": len(group_records),
            "mc_scores": _mc_aggregate(group_records),
            "decision_rates": _decision_rates(group_records),
            "note": None if len(group_records) >= 15 else "small sample - interpret with caution",
        }
    return out


def _error_analysis(records: list[dict]) -> dict:
    """Deterministic selection only. MC-track errors use the rigorous
    ground truth (MC1=0 means the model ranked an incorrect choice
    highest). Free-form-track categories are limited to what is
    measurable WITHOUT a truthfulness judgment (see module note below) -
    retrieval/process diagnostics, not truthful/untruthful outcomes."""

    def top_by_response_id(items: list[dict], key="question_id") -> list[dict]:
        return sorted(items, key=lambda r: r[key])[:N_ERROR_EXAMPLES_PER_CATEGORY]

    def _preview(r: dict, mc1_score) -> dict:
        return {
            "question_id": r["question_id"], "category": r["category"], "question": r["question"] if "question" in r else None,
            "mc1_score": mc1_score,
            "condition_a_baseline_preview": r["free_form"]["condition_a_baseline_answer_preview"],
            "condition_b_decision": r["free_form"]["condition_b_attempt0_mapped_decision"],
            "condition_c_final_decision": r["free_form"]["condition_c_final_decision"],
            "termination_reason": r["free_form"]["termination_reason"],
        }

    mc1_wrong = []
    for r in records:
        scores = _mc_scores_for_record(r)
        if scores["mc1"] == 0:
            mc1_wrong.append((r, scores["mc1"]))

    max_attempts_cases = [r for r in records if r["free_form"]["termination_reason"] == "max_attempts_reached"]
    retrieval_zero_evidence = [r for r in records if r["free_form"]["condition_b_attempt0_num_evidence"] == 0]
    inference_failures = [r for r in records if r["free_form"]["status"] == "failed"]

    return {
        "selection_rule": "First N by question_id after a stable sort - no manual example selection.",
        "not_computed_categories": {
            "categories_1_through_7_from_instructions": (
                "Categories referencing 'truthful'/'untruthful' free-form outcomes (baseline truthful -> "
                "ClaimGuard truthful/untruthful, corrected/still-untruthful, harmed, verifier accepted an "
                "untruthful answer, verifier corrected a truthful answer unnecessarily) require a ground-truth "
                "truthfulness judgment for FREE-FORM generated text. No approved automatic judge for this exists "
                "in this project (no human raters, no fine-tuned GPT-judge model, and embedding/string-similarity "
                "scoring would be exactly the fabricated 'semantic similarity threshold called truthfulness' the "
                "Step 20 instructions explicitly forbid). Reported as a documented protocol gap, not silently "
                "invented - see claimguard.evaluation.truthfulqa_eval module docstring."
            ),
        },
        "mc1_incorrect_examples": [
            _preview(r, s) for r, s in sorted(mc1_wrong, key=lambda x: x[0]["question_id"])[:N_ERROR_EXAMPLES_PER_CATEGORY]
        ],
        "mc1_incorrect_total_count": len(mc1_wrong),
        "max_attempts_examples": [_preview(r, _mc_scores_for_record(r)["mc1"]) for r in top_by_response_id(max_attempts_cases)],
        "max_attempts_total_count": len(max_attempts_cases),
        "retrieval_zero_evidence_examples": [
            _preview(r, _mc_scores_for_record(r)["mc1"]) for r in top_by_response_id(retrieval_zero_evidence)
        ],
        "retrieval_zero_evidence_total_count": len(retrieval_zero_evidence),
        "inference_failure_examples": [_preview(r, _mc_scores_for_record(r)["mc1"]) for r in top_by_response_id(inference_failures)],
        "inference_failure_total_count": len(inference_failures),
    }


def main() -> int:
    _section("LOADING PER-QUESTION RESULTS")
    records = load_records()
    print(f"Loaded {len(records)} question records from {RESPONSE_JSONL_PATH}")
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)

    _section("MC1 / MC2 / MC0 OVERALL (Qwen3-8B parametric calibration - no pipeline involved)")
    overall_mc = _mc_aggregate(records)
    print(json.dumps(overall_mc, indent=2))

    _section("FREE-FORM PIPELINE PROCESS BEHAVIOR (NOT a truthfulness judgment)")
    overall_decisions = _decision_rates(records)
    overall_latency = _latency_stats(records)
    print(json.dumps(overall_decisions, indent=2))

    _section("CATEGORY BREAKDOWN (37 categories)")
    category_breakdown = _breakdown_by_category(records)
    for cat, v in category_breakdown.items():
        print(f"  {cat}: n={v['n']} mc1={v['mc_scores']['mc1_accuracy']} mc2={v['mc_scores']['mc2_accuracy']} "
              f"attempt0_accept_rate={v['decision_rates'].get('attempt0_accept_rate')}")

    _section("RETRIEVAL DIAGNOSTICS (free-form track, attempt 0)")
    n_zero_evidence = sum(1 for r in records if r["free_form"]["condition_b_attempt0_num_evidence"] == 0)
    evidence_counts = [r["free_form"]["condition_b_attempt0_num_evidence"] for r in records]
    retrieval_diag = {
        "n_zero_evidence_retrieved": n_zero_evidence,
        "mean_evidence_count": statistics.mean(evidence_counts) if evidence_counts else None,
        "note": (
            "As in Step 19, the retrieval corpus is exactly the frozen FEVER-Wikipedia corpus (unchanged, not "
            "augmented with TruthfulQA sources or any external content). TruthfulQA's 37 categories span topics "
            "(law, misconceptions, fiction, proverbs, etc.) with no guarantee of FEVER-Wikipedia coverage. "
            "Successful (non-zero) retrieval is NOT interpreted as proof of relevant evidence - see Step 19's "
            "established corpus-mismatch finding, which this evaluation does not re-litigate or re-verify claim "
            "by claim."
        ),
    }
    print(json.dumps(retrieval_diag, indent=2))

    _section("ERROR ANALYSIS (deterministic selection)")
    error_analysis = _error_analysis(records)
    print(f"  mc1_incorrect: {error_analysis['mc1_incorrect_total_count']}, "
          f"max_attempts: {error_analysis['max_attempts_total_count']}, "
          f"retrieval_zero_evidence: {error_analysis['retrieval_zero_evidence_total_count']}, "
          f"inference_failures: {error_analysis['inference_failure_total_count']}")
    with ERROR_ANALYSIS_PATH.open("w", encoding="utf-8") as f:
        json.dump(error_analysis, f, indent=2, default=str)
    print(f"Saved {ERROR_ANALYSIS_PATH}")

    _section("DATA LEAKAGE / ISOLATION AUDIT")
    leakage_audit = {
        "truthfulqa_used_in_training": False,
        "truthfulqa_used_in_tuning": False,
        "truthfulqa_used_to_choose_thresholds": False,
        "truthfulqa_used_for_model_selection": False,
        "ragtruth_touched_in_step20": False,
        "gold_truthfulqa_labels_entered_inference": False,
        "correct_incorrect_answer_sets_passed_to_generator": False,
        "gold_answers_passed_to_retrieval": False,
        "truthfulqa_results_used_to_modify_thresholds": False,
        "external_web_retrieval_used": False,
        "note": (
            "run_correction_loop() and score_choice_log_likelihood() receive only `question`/choice TEXT - "
            "the 0/1 labels inside mc0/mc1/mc2_targets and best_answer/correct_answers/incorrect_answers are "
            "read only by this analysis script's own scoring functions, after all inference completed. See "
            "tests/test_truthfulqa_evaluation.py's structural guard and mock-based behavioral test confirming "
            "gold fields never appear in run_correction_loop's or score_choice_log_likelihood's actual call args."
        ),
    }
    print(json.dumps(leakage_audit, indent=2))

    _section("SAVING AGGREGATE RESULTS")
    results = {
        "manifest_summary": {
            "n_questions_evaluated": manifest["n_questions_evaluated"], "n_categories": manifest["n_categories"],
            "pilot_mode": manifest["pilot_mode"], "model_identifiers": manifest["model_identifiers"],
            "frozen_config": manifest["frozen_config"], "evaluation_tracks": manifest["evaluation_tracks"],
        },
        "mc_scores_overall": overall_mc,
        "free_form_decision_rates_overall": overall_decisions,
        "free_form_latency_overall": overall_latency,
        "category_breakdown": category_breakdown,
        "retrieval_diagnostics": retrieval_diag,
        "reproducibility_check": manifest["reproducibility_check"],
        "leakage_isolation_audit": leakage_audit,
        "scientific_scope_note": (
            "MC1/MC2/MC0 measure Qwen3-8B's OWN parametric calibration on TruthfulQA - they do not exercise "
            "retrieval, reranking, verification, or correction at all, so they say nothing about ClaimGuard's "
            "verification/correction layer specifically. The free-form pipeline track exercises the full "
            "ClaimGuard system but reports PROCESS behavior (decision rates, retrieval diagnostics, latency) "
            "only, not a truthfulness accuracy number - no approved automatic judge for free-form TruthfulQA "
            "truthfulness exists in this project (see claimguard.evaluation.truthfulqa_eval module docstring)."
        ),
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
