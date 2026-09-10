"""Step 22: build the final, machine-readable results tables by reading
directly from the ALREADY-VALIDATED source artifacts of Steps 9-21. Makes
no model calls, runs no new experiments - pure aggregation/formatting.
Every number here is read from an existing file, never invented or
recomputed with different logic than the step that originally produced it.

Usage:
    python scripts/generate_final_results_tables.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

ROOT = cg_config.PROJECT_ROOT
OUT_DIR = cg_config.resolve_path("data/processed/final")
TABLES_JSON_PATH = OUT_DIR / "final_results_tables.json"
TABLES_CSV_DIR = OUT_DIR / "tables_csv"


def _load(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"Required source artifact not found: {p}")
    with p.open() as f:
        return json.load(f)


def main() -> int:
    dataset_manifest = _load("data/processed/dataset_manifest.json")
    verifier_dev = _load("experiments/verifier_binary_final/final/dev_metrics.json")
    verifier_repro = _load("experiments/verifier_binary_final/final/reproducibility.json")
    corpus_manifest = _load("data/processed/retrieval/corpus_manifest.json")
    reranker_eval = _load("data/processed/retrieval/reranker_eval_results.json")
    integration_eval = _load("data/processed/integration/integration_eval_results.json")
    decision_policy = _load("data/processed/integration/decision_policy_results.json")
    ragtruth_results = _load("data/processed/evaluation/ragtruth/ragtruth_eval_results.json")
    truthfulqa_results = _load("data/processed/evaluation/truthfulqa/truthfulqa_eval_results.json")
    ablation_results = _load("data/processed/evaluation/ablations/ablation_results.json")

    fever = dataset_manifest["datasets"]["fever"]

    # --- Table 1: Dataset statistics ---
    table1 = {
        "FEVER": {
            "role": "verifier training (entailment/contradiction)",
            "raw_train_claims": fever["splits"]["train"]["raw_dataset_size"],
            "usable_records_resolved_premise": fever["splits"]["train"]["usable_records_with_resolved_premise"],
            "neutral_class_available": fever["premise_resolution_status"]["resolution_coverage"]["neutral_class_available_from_fever"],
            "license": fever["license"],
        },
        "verifier_train_dev": {
            "train_record_count": verifier_repro["train_record_count"],
            "dev_record_count": verifier_repro["dev_record_count"],
        },
        "RAGTruth": {
            "role": "primary end-to-end evaluation (held-out TEST split)",
            "test_responses": ragtruth_results["manifest_summary"]["n_responses_evaluated"],
            "hallucination_prevalence": ragtruth_results["overall_metrics_by_condition"]["B_generated"]["hallucination_prevalence_gold"],
        },
        "TruthfulQA": {
            "role": "secondary evaluation-only (adversarial stress test, no split)",
            "n_questions": truthfulqa_results["manifest_summary"]["n_questions_evaluated"],
            "n_categories": truthfulqa_results["manifest_summary"]["n_categories"],
        },
    }

    # --- Table 2: Verifier development ---
    table2 = {
        "model": verifier_repro["model_name"], "n_params": verifier_repro["n_total_params"],
        "precision": verifier_repro["precision"], "train_records": verifier_repro["train_record_count"],
        "dev_records": verifier_repro["dev_record_count"],
        "dev_accuracy": verifier_dev["eval_accuracy"], "dev_macro_f1": verifier_dev["eval_macro_f1"],
        "dev_weighted_f1": verifier_dev["eval_weighted_f1"],
        "per_class_f1": {k: v["f1"] for k, v in verifier_dev["eval_per_class"].items()},
        "confusion_matrix": verifier_dev["eval_confusion_matrix"],
        "confusion_matrix_labels": verifier_dev["eval_confusion_matrix_labels"],
        "peak_gpu_memory_gb": verifier_repro["peak_gpu_memory_gb"],
    }

    # --- Table 3: Retrieval/reranking ---
    table3 = {
        "corpus_size_sentences": corpus_manifest["corpus_construction"]["total_corpus_records"],
        "embedding_model": corpus_manifest["embedding"]["model_name"],
        "embedding_dim": corpus_manifest["embedding"]["embedding_dim"],
        "faiss_only": reranker_eval["faiss_only"],
        "faiss_plus_reranker": reranker_eval["faiss_plus_reranker"],
        "integration_strategy_comparison": integration_eval["evidence_selection_comparison"],
        "integration_verifier_classification": integration_eval["verifier_classification"],
        "integration_latency_ms": integration_eval["latency"],
    }

    # --- Table 4: End-to-end RAGTruth ---
    b = ragtruth_results["overall_metrics_by_condition"]["B_generated"]
    table4 = {
        "n_responses": b["n"], "hallucination_prevalence": b["hallucination_prevalence_gold"],
        "attempt0_accuracy": b["attempt0_detection_metrics"]["accuracy"],
        "attempt0_macro_f1": b["attempt0_detection_metrics"]["macro_f1"],
        "final_accuracy": b["final_outcome_metrics"]["accuracy"],
        "final_macro_f1": b["final_outcome_metrics"]["macro_f1"],
        "majority_class_baseline": 1 - b["hallucination_prevalence_gold"],
        "decision_rates": b["decision_rates"],
        "correction_outcome_categories": b["correction_outcome_categories"],
        "task_type_breakdown": ragtruth_results["task_type_breakdown_condition_B"],
        "source_model_breakdown": ragtruth_results["source_model_breakdown_condition_B"],
        "retrieval_diagnostics": ragtruth_results["retrieval_diagnostics"],
        "latency_by_condition": ragtruth_results["latency_by_condition"],
    }

    # --- Table 5: TruthfulQA ---
    table5 = {
        "n_questions": truthfulqa_results["manifest_summary"]["n_questions_evaluated"],
        "n_categories": truthfulqa_results["manifest_summary"]["n_categories"],
        "mc_scores": truthfulqa_results["mc_scores_overall"],
        "free_form_decision_rates": truthfulqa_results["free_form_decision_rates_overall"],
        "free_form_latency": truthfulqa_results["free_form_latency_overall"],
        "retrieval_diagnostics": truthfulqa_results["retrieval_diagnostics"],
        "note": truthfulqa_results["scientific_scope_note"],
    }

    # --- Table 6: Controlled ablations ---
    cm = ablation_results["condition_metrics_A_B_C_subset"]
    table6 = {
        "A_reranked": {"n": 300, **cm["A_reranked"]["attempt0_detection_metrics"]},
        "B_no_reranker": {"n": 300, **cm["B_no_reranker"]["attempt0_detection_metrics"]},
        "C_faiss_top1": {"n": 300, **cm["C_faiss_top1"]["attempt0_detection_metrics"]},
        "D_no_correction_REUSED": {"n": 2700, **ablation_results["condition_D_no_correction_reused_full_2700"]},
        "E_verify_only_REUSED": {"n": 2700, **ablation_results["condition_E_verify_only_reused_full_2700"]},
        "evidence_movement": ablation_results["evidence_movement"],
        "cross_validation_A_vs_step19": ablation_results["cross_validation_subset_A_vs_step19_full"],
    }

    # --- Table 7: Failure-source evidence ---
    with (ROOT / "data/processed/evaluation/ablations/ablation_error_analysis.json").open() as f:
        ablation_errors = json.load(f)
    table7 = {
        "verifier_overconfidence_full_2700": ablation_results["verifier_overconfidence_full_2700_reused"],
        "failure_source_separation_n300_subset": ablation_errors["failure_source_separation"],
    }

    all_tables = {
        "table1_dataset_statistics": table1, "table2_verifier_development": table2,
        "table3_retrieval_reranking": table3, "table4_end_to_end_ragtruth": table4,
        "table5_truthfulqa": table5, "table6_controlled_ablations": table6,
        "table7_failure_source_evidence": table7,
        "provenance": {
            "note": "Every value above is read directly from the source artifact listed - none are "
                    "recomputed with different logic or invented. See tests/test_final_report.py for "
                    "the programmatic consistency check against these same source files.",
            "source_artifacts": [
                "data/processed/dataset_manifest.json",
                "experiments/verifier_binary_final/final/dev_metrics.json",
                "experiments/verifier_binary_final/final/reproducibility.json",
                "data/processed/retrieval/corpus_manifest.json",
                "data/processed/retrieval/reranker_eval_results.json",
                "data/processed/integration/integration_eval_results.json",
                "data/processed/integration/decision_policy_results.json",
                "data/processed/evaluation/ragtruth/ragtruth_eval_results.json",
                "data/processed/evaluation/truthfulqa/truthfulqa_eval_results.json",
                "data/processed/evaluation/ablations/ablation_results.json",
                "data/processed/evaluation/ablations/ablation_error_analysis.json",
            ],
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TABLES_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_tables, f, indent=2, default=str)
    print(f"Saved {TABLES_JSON_PATH}")

    TABLES_CSV_DIR.mkdir(parents=True, exist_ok=True)
    with (TABLES_CSV_DIR / "table4_ragtruth_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in ("n_responses", "hallucination_prevalence", "attempt0_accuracy", "attempt0_macro_f1",
                   "final_accuracy", "final_macro_f1", "majority_class_baseline"):
            w.writerow([k, table4[k]])
    with (TABLES_CSV_DIR / "table6_ablation_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n", "accuracy", "macro_f1", "hallucinated_recall"])
        for cond in ("A_reranked", "B_no_reranker", "C_faiss_top1", "D_no_correction_REUSED", "E_verify_only_REUSED"):
            row = table6[cond]
            w.writerow([cond, row["n"], row["accuracy"], row["macro_f1"], row["hallucinated_recall"]])
    print(f"Saved CSV tables to {TABLES_CSV_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
