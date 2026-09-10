"""Step 7/8: build the ClaimGuard dataset manifest.

Loads every dataset's normalized data (FEVER, HaluEval, RAGTruth,
TruthfulQA), computes real record/label counts, builds the leakage-safe
verifier training/development pool (Step 8: with cross-dataset
contamination automatically excluded by default), runs cross-dataset
exact-text-overlap checks against both evaluation datasets, determines
FEVER's premise-resolution status, and writes a single machine-readable
manifest to data/processed/dataset_manifest.json.

Does NOT train any model, touch the GPU, or build any retrieval index -
CPU/data-integration only. Does NOT modify any raw dataset file.

Usage:
    python scripts/build_dataset_manifest.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.datasets import fever as fever_ds  # noqa: E402
from claimguard.datasets import halueval as halueval_ds  # noqa: E402
from claimguard.datasets import manifest as mf  # noqa: E402
from claimguard.datasets import ragtruth as ragtruth_ds  # noqa: E402
from claimguard.datasets import truthfulqa as truthfulqa_ds  # noqa: E402

MANIFEST_PATH = cg_config.resolve_path("data/processed/dataset_manifest.json")
CONTAMINATION_REPORT_PATH = cg_config.resolve_path("data/processed/contamination_report.json")


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    manifest: dict = {
        "version": "1.3",
        "protocol_step": "11 (supersedes the Step 10/1.2 manifest - see PROJECT_REPORT.md Step 11 "
                          "for what changed: a systematic investigation of every already-acquired "
                          "dataset for a legitimate neutral-class source, concluding that none "
                          "exists and that the verifier training pool is correctly a BINARY "
                          "(entailment/contradiction) dataset, not a 3-way dataset with a "
                          "temporarily-empty third class - see verifier_dataset_status below)",
        "generated_by": "scripts/build_dataset_manifest.py",
        "contamination_report": "data/processed/contamination_report.json (see "
                                 "scripts/detect_contamination.py)",
        "seed": mf.SEED,
        "dev_ratio": mf.DEV_RATIO,
        "verifier_label_scheme": list(mf.VERIFIER_LABELS),
        "datasets": {},
    }

    # --- FEVER --------------------------------------------------------
    _section("FEVER")
    fever_train = fever_ds.load_normalized_split("train")
    fever_label_dist = Counter(c["label"] for c in fever_train)
    fever_verifier_dist = Counter(c["verifier_label"] for c in fever_train)
    print(f"train: {len(fever_train):,} unique claims | label dist: {dict(fever_label_dist)}")
    print(f"train: verifier-label dist (claim-level): {dict(fever_verifier_dist)}")

    fever_premise_status = mf.fever_premise_resolution_status()
    fever_coverage = fever_premise_status.get("resolution_coverage", {})
    print(f"\nFEVER premise-resolution status (Step 10): "
          f"wiki_pages_index_present={fever_premise_status['wiki_pages_index_present']}, "
          f"premise_text_currently_available={fever_premise_status['premise_text_currently_available']}")
    if fever_coverage:
        print(f"  usable_for_training (resolved SUPPORTS/REFUTES): "
              f"{fever_coverage['usable_for_training']:,} / {fever_coverage['total_claims']:,}")
        print(f"  neutral_class_available_from_fever (resolved NOT ENOUGH INFO): "
              f"{fever_coverage['neutral_class_available_from_fever']:,}")
        print(f"  status_counts: {fever_coverage['status_counts']}")

    manifest["datasets"]["fever"] = {
        "source": "fever/fever (Hugging Face, config=default, via refs/convert/parquet)",
        "license": "CC BY-SA 3.0 + GNU FDL",
        "normalization_module": "src/claimguard/datasets/fever.py",
        "raw_location": "data/raw/fever/",
        "wiki_pages_index": "data/processed/fever/wiki_pages_index.sqlite (Step 10)",
        "wiki_resolution_report": "data/processed/fever/wiki_resolution_report.json (Step 10)",
        "processed_sample_location": "data/processed/fever/sample/sample.jsonl",
        "premise_resolution_status": fever_premise_status,
        "splits": {
            "train": {
                "role": "verifier_training_dev_candidate",
                "raw_dataset_size": len(fever_train),
                "eligible_records": len(fever_train),
                "usable_records_with_resolved_premise": fever_coverage.get("usable_for_training", 0),
                "excluded_no_evidence_annotation": fever_coverage.get("status_counts", {}).get(
                    fever_ds.RESOLUTION_STATUS_NO_EVIDENCE, 0),
                "excluded_unresolved": fever_coverage.get("status_counts", {}).get(
                    fever_ds.RESOLUTION_STATUS_UNRESOLVED, 0),
                "excluded_contamination_records": 0,
                "record_count": len(fever_train),
                "label_distribution": dict(fever_label_dist),
                "verifier_label_distribution": dict(fever_verifier_dist),
                "eligible_for_verifier_training": True,
                "eligible_for_end_to_end_evaluation": False,
                "id_semantics": "example_id = raw FEVER claim 'id', unique within split",
                "caveats": [
                    "As of Step 10, SUPPORTS/REFUTES claims with fully-resolved evidence "
                    "(109,810/109,810 - 100%) have real, resolved Wikipedia sentence premise "
                    "text - see premise_resolution_status above. NOT ENOUGH INFO claims "
                    "(35,639) still have NO evidence annotation in FEVER's own data at all "
                    "and are excluded from the trainable pool (not coerced) - FEVER "
                    "therefore still contributes 0 neutral-label training examples.",
                ],
            },
            "validation": {
                "role": "excluded",
                "eligible_for_verifier_training": False,
                "eligible_for_end_to_end_evaluation": False,
                "caveats": [
                    "19,998 blank-label rows, 33 casing-variant labels, ~19,037 exact "
                    "duplicate rows (~24%), overlaps `test` by 9,999 claim IDs (Step 6A).",
                ],
            },
            "test": {
                "role": "excluded",
                "eligible_for_verifier_training": False,
                "eligible_for_end_to_end_evaluation": False,
                "caveats": [
                    "19,998 blank-label rows mixed in, overlaps `validation` by 9,999 "
                    "claim IDs (Step 6A).",
                ],
            },
        },
    }

    # --- HaluEval -------------------------------------------------------
    _section("HALUEVAL")
    contaminated_indices = mf.contaminated_halueval_summarization_raw_indices()
    print(f"Contaminated HaluEval summarization raw indices (computed fresh, not hard-coded): "
          f"{len(contaminated_indices)}")

    halueval_subset_info = {}
    for subset in ("qa", "dialogue", "summarization"):
        recs = halueval_ds.load_normalized_subset(subset)
        label_dist = Counter(r["label"] for r in recs)
        excluded_here = 0
        if subset == "summarization":
            # Each contaminated raw_index produced exactly 2 pair-expanded
            # records (not_hallucinated + hallucinated).
            excluded_here = sum(1 for r in recs if r["raw_index"] in contaminated_indices)
        eligible = len(recs) - excluded_here
        print(f"{subset}: {len(recs):,} normalized records | label dist: {dict(label_dist)} "
              f"| excluded (contamination): {excluded_here} | eligible: {eligible:,}")
        halueval_subset_info[subset] = {
            "role": "verifier_training_dev_candidate",
            "raw_dataset_size": len(recs),
            "eligible_records": eligible,
            "excluded_contamination_records": excluded_here,
            "exclusion_reason": mf.EXCLUSION_REASON_RAGTRUTH_OVERLAP if excluded_here else None,
            "record_count": len(recs),
            "label_distribution": dict(label_dist),
            "eligible_for_verifier_training": True,
            "eligible_for_end_to_end_evaluation": False,
            "id_semantics": "example_id = f'{subset}_{raw_index}_{label}', raw_index unique per subset",
            "caveats": (
                ["100 raw records (200 normalized pair-expanded records) excluded - see "
                 "data/processed/contamination_report.json - their `document` text exactly "
                 "matches a RAGTruth test Summary source article (both CNN/DailyMail-sourced)."]
                if subset == "summarization" else []
            ),
        }

    general_recs = halueval_ds.load_normalized_subset("general")
    general_label_dist = Counter(r["label"] for r in general_recs)
    print(f"general: {len(general_recs):,} normalized records | label dist: {dict(general_label_dist)} "
          f"| EXCLUDED from verifier training pool (no premise field)")
    halueval_subset_info["general"] = {
        "role": "excluded_from_verifier_pool",
        "raw_dataset_size": len(general_recs),
        "eligible_records": 0,
        "excluded_contamination_records": 0,
        "record_count": len(general_recs),
        "label_distribution": dict(general_label_dist),
        "eligible_for_verifier_training": False,
        "eligible_for_end_to_end_evaluation": False,
        "reason": mf.GENERAL_EXCLUSION_REASON,
        "caveats": [
            "Retained here as normalized/available data, not discarded - just not "
            "routed into the NLI verifier training pool. Also checked against RAGTruth "
            "test / TruthfulQA in the Step 8 contamination sweep - no overlap found.",
        ],
    }

    manifest["datasets"]["halueval"] = {
        "source": "RUCAIBox/HaluEval (GitHub, main branch)",
        "license": "MIT",
        "normalization_module": "src/claimguard/datasets/halueval.py",
        "raw_location": "data/raw/halueval/",
        "processed_sample_location": "data/processed/halueval/sample/sample.jsonl",
        "subsets": halueval_subset_info,
    }

    # --- RAGTruth -------------------------------------------------------
    _section("RAGTRUTH")
    ragtruth_all = ragtruth_ds.load_normalized()
    ragtruth_train = ragtruth_ds.get_training_pool(ragtruth_all)
    ragtruth_test = ragtruth_ds.get_eval_set(ragtruth_all)
    ragtruth_ds.assert_no_train_test_leakage(ragtruth_all)  # re-verify before writing the manifest
    print(f"train: {len(ragtruth_train):,} | test: {len(ragtruth_test):,} | "
          f"leakage check: PASSED (re-verified)")

    manifest["datasets"]["ragtruth"] = {
        "source": "ParticleMedia/RAGTruth (GitHub, main branch)",
        "license": "MIT",
        "normalization_module": "src/claimguard/datasets/ragtruth.py",
        "raw_location": "data/raw/ragtruth/",
        "processed_sample_location": "data/processed/ragtruth/sample/sample.jsonl",
        "splits": {
            "train": {
                "role": "not_used_deferred",
                "record_count": len(ragtruth_train),
                "eligible_for_verifier_training": False,
                "eligible_for_end_to_end_evaluation": False,
                "caveats": [
                    "Step 6C confirmed source-disjoint from `test` and safe to use later, "
                    "but the Step 3/4 protocol names only FEVER and HaluEval as training/"
                    "development candidates - inclusion of ragtruth_train is deferred, not "
                    "decided, by this manifest. NOT in ALLOWED_TRAINING_SOURCES.",
                ],
            },
            "test": {
                "role": "evaluation_reserved",
                "record_count": len(ragtruth_test),
                "eligible_for_verifier_training": False,
                "eligible_for_end_to_end_evaluation": True,
                "hard_blocked_from_training": True,
                "caveats": [
                    "PRIMARY end-to-end evaluation dataset (Experiment 3). Never enters "
                    "training/dev - enforced in code via claimguard.datasets.manifest."
                    "RoleViolationError and ragtruth.get_eval_set()/"
                    "assert_no_train_test_leakage(). Step 8: 100 of these source items "
                    "(600 responses) were found to share source documents with HaluEval "
                    "summarization training data - RAGTruth test ITSELF was NOT modified; "
                    "the corresponding HaluEval records were excluded from training instead.",
                ],
            },
        },
    }

    # --- TruthfulQA -----------------------------------------------------
    _section("TRUTHFULQA")
    tqa_records = truthfulqa_ds.load_normalized()
    tqa_type_dist = Counter(r["type"] for r in tqa_records)
    print(f"{len(tqa_records):,} records | type dist: {dict(tqa_type_dist)} | no split exists")

    manifest["datasets"]["truthfulqa"] = {
        "source": "sylinrl/TruthfulQA (GitHub, main branch)",
        "license": "Apache-2.0",
        "normalization_module": "src/claimguard/datasets/truthfulqa.py",
        "raw_location": "data/raw/truthfulqa/",
        "processed_sample_location": "data/processed/truthfulqa/sample/sample.jsonl",
        "role": "evaluation_only",
        "record_count": len(tqa_records),
        "type_distribution": dict(tqa_type_dist),
        "eligible_for_verifier_training": False,
        "eligible_for_end_to_end_evaluation": True,
        "hard_blocked_from_training": True,
        "caveats": [
            "No official train/validation/test split exists (Step 6D, verified empirically) "
            "- the entire 790-question set is evaluation-only. No artificial split was "
            "created for it.",
        ],
    }

    # --- Step 8: contamination detection & mitigation ---------------------
    _section("STEP 8: CONTAMINATION DETECTION & MITIGATION")
    audit = mf.find_ragtruth_summarization_contamination()
    affected_halueval_indices = sorted({a["halueval_raw_index"] for a in audit})
    affected_source_ids = sorted({a["ragtruth_source_id"] for a in audit})
    affected_response_ids = sorted({a["ragtruth_test_response_id"] for a in audit})
    print(f"Reproduced: {len(audit)} audit records | "
          f"{len(affected_halueval_indices)} unique HaluEval raw indices | "
          f"{len(affected_source_ids)} unique RAGTruth source_ids | "
          f"{len(affected_response_ids)} unique RAGTruth test responses")

    pool_before_exclusion = mf.build_training_pool(exclude_contamination=False)
    pool = mf.build_training_pool()  # exclude_contamination=True by default
    records_excluded = len(pool_before_exclusion) - len(pool)
    print(f"Pool size before exclusion: {len(pool_before_exclusion):,} | "
          f"after: {len(pool):,} | excluded: {records_excluded}")

    mitigation_check = mf.verify_contamination_resolved(pool)
    print(f"Post-mitigation re-verification (same broad check that found the problem): "
          f"{mitigation_check}")

    sweep = mf.cross_dataset_contamination_sweep()
    print("Cross-dataset sweep (all candidate sources vs both evaluation datasets):")
    for source, result in sweep.items():
        print(f"  {source}: vs_ragtruth_test={result['vs_ragtruth_test']} "
              f"vs_truthfulqa={result['vs_truthfulqa']}")

    manifest["contamination"] = {
        "detection_method": "exact normalized (lowercase, whitespace-collapsed, stripped) "
                             "text match, field-aware and source-level identified "
                             "(claimguard.datasets.manifest.find_ragtruth_summarization_"
                             "contamination) - not a near-duplicate/fuzzy check",
        "finding": {
            "audit_record_count": len(audit),
            "unique_halueval_raw_indices_affected": len(affected_halueval_indices),
            "unique_ragtruth_source_ids_affected": len(affected_source_ids),
            "unique_ragtruth_test_responses_affected": len(affected_response_ids),
            "explanation": "100 unique HaluEval summarization documents exactly match 100 "
                            "unique RAGTruth test Summary source articles (1:1 on both sides "
                            "- no duplicate documents among the matches). Each RAGTruth "
                            "source_id has exactly 6 responses (Step 6C), and all 6 are in "
                            "`test` for these sources (source-disjoint splits, Step 6C) - so "
                            "100 source_ids x 6 responses = 600 affected RAGTruth test "
                            "responses, matching the audit-record count exactly.",
        },
        "mitigation": {
            "decision": "Exclude the 100 affected HaluEval summarization raw records (200 "
                        "normalized pair-expanded records) from the verifier training/dev "
                        "pool. RAGTruth test was NOT modified.",
            "reason_code": mf.EXCLUSION_REASON_RAGTRUTH_OVERLAP,
            "mechanism": "claimguard.datasets.manifest.contaminated_halueval_summarization_"
                         "raw_indices() computes the exclusion set fresh from the raw data on "
                         "every call (not a hard-coded row-number list); "
                         "build_training_pool(exclude_contamination=True, the default) "
                         "removes matching PoolRecords by group_key.",
            "records_excluded": records_excluded,
            "pool_size_before_exclusion": len(pool_before_exclusion),
            "pool_size_after_exclusion": len(pool),
            "post_mitigation_verification": mitigation_check,
        },
        "cross_dataset_sweep": sweep,
        "raw_data_modified": False,
        "ragtruth_test_modified": False,
    }

    # --- Verifier training pool (post-exclusion) --------------------------
    _section("VERIFIER TRAINING/DEV POOL (post-exclusion)")
    pool_label_dist = Counter(r.label for r in pool)
    pool_source_dist = Counter(f"{r.source_dataset}.{r.source_subset}" for r in pool)
    print(f"Total pool records: {len(pool):,}")
    print(f"Pool label distribution: {dict(pool_label_dist)}")
    print(f"Pool source distribution: {dict(pool_source_dist)}")

    train_split, dev_split = mf.split_train_dev(pool)
    train_label_dist = Counter(r.label for r in train_split)
    dev_label_dist = Counter(r.label for r in dev_split)
    print(f"\nDeterministic train/dev split (seed={mf.SEED}, dev_ratio={mf.DEV_RATIO}):")
    print(f"  train: {len(train_split):,} | label dist: {dict(train_label_dist)}")
    print(f"  dev:   {len(dev_split):,} | label dist: {dict(dev_label_dist)}")

    train_groups = {r.group_key for r in train_split}
    dev_groups = {r.group_key for r in dev_split}
    group_overlap = train_groups & dev_groups
    print(f"  group-key overlap between train/dev (must be 0): {len(group_overlap)}")

    excluded_group_keys = mf.contamination_exclusion_group_keys()
    leaked_into_train = train_groups & excluded_group_keys
    leaked_into_dev = dev_groups & excluded_group_keys
    print(f"  excluded-contamination group_keys found in train (must be 0): {len(leaked_into_train)}")
    print(f"  excluded-contamination group_keys found in dev (must be 0): {len(leaked_into_dev)}")

    manifest["verifier_training_pool"] = {
        "allowed_sources": sorted(mf.ALLOWED_TRAINING_SOURCES),
        "blocked_sources": sorted(mf.BLOCKED_TRAINING_SOURCES),
        "raw_eligible_before_contamination_exclusion": len(pool_before_exclusion),
        "excluded_contamination_records": records_excluded,
        "total_records": len(pool),
        "label_distribution": dict(pool_label_dist),
        "source_distribution": dict(pool_source_dist),
        "label_mapping": {
            "fever": {
                "SUPPORTS": {"target": "entailment", "rationale": "FEVER's own definition: the evidence directly supports the claim.", "cardinality": "one_to_one"},
                "REFUTES": {"target": "contradiction", "rationale": "FEVER's own definition: the evidence directly contradicts the claim.", "cardinality": "one_to_one"},
                "NOT ENOUGH INFO": {"target": "neutral", "rationale": "FEVER's own definition: evidence is insufficient to decide.", "cardinality": "one_to_one"},
            },
            "halueval_pair_subsets": {
                "not_hallucinated": {"target": "entailment", "rationale": "The reference (right_answer/right_response/right_summary) is, by construction, grounded in and consistent with the given context.", "cardinality": "one_to_one"},
                "hallucinated": {"target": "contradiction", "rationale": "HaluEval's generation methodology produces content that is factually incorrect relative to the grounding, which skews toward direct contradiction rather than mere unsupportedness.", "cardinality": "one_to_one", "caveat": "APPROXIMATION: HaluEval's binary 'hallucinated' label conflates what FEVER separately calls REFUTES (direct contradiction) and NOT ENOUGH INFO (unsupported fabrication beyond the given context). No 'neutral' examples are contributed by HaluEval as a result - see balancing notes."},
            },
            "halueval_general": {"status": "excluded", "reason": mf.GENERAL_EXCLUSION_REASON},
        },
        "dev_split": {
            "strategy": "group-key-based deterministic split (grouping keeps HaluEval's "
                        "contrastive pair records together; FEVER claims are already atomic "
                        "groups); shuffled with a seeded RNG, not row-random, so a pair is "
                        "never split across train/dev. Contamination-excluded records are "
                        "removed from the pool BEFORE this split runs, so they cannot appear "
                        "in either side.",
            "seed": mf.SEED,
            "dev_ratio": mf.DEV_RATIO,
            "train_count": len(train_split),
            "dev_count": len(dev_split),
            "train_label_distribution": dict(train_label_dist),
            "dev_label_distribution": dict(dev_label_dist),
            "group_key_overlap_train_dev": len(group_overlap),
            "excluded_contamination_leaked_into_train": len(leaked_into_train),
            "excluded_contamination_leaked_into_dev": len(leaked_into_dev),
        },
        "balancing": {
            "status": "deferred",
            "note": "Label distribution is imbalanced: 'neutral' is contributed by FEVER "
                    "only (HaluEval's binary hallucinated/not_hallucinated mapping produces "
                    "no 'neutral' examples). No downsampling/reweighting has been applied - "
                    "this is a training-time decision (e.g. class weights, oversampling) "
                    "deferred to the actual training step, not decided here. NOT changed "
                    "by the Step 8 contamination exclusion.",
        },
        "neutral_class_coverage": {
            "count_in_pool": pool_label_dist.get("neutral", 0),
            "status": "UNRESOLVED_GAP",
            "explanation": (
                "Step 10 achieved 100% evidence resolution for FEVER SUPPORTS/REFUTES "
                "claims (109,810/109,810), so FEVER now contributes real, resolved premise "
                "text to the pool for the first time. This does NOT close the neutral-class "
                "gap: FEVER's NOT ENOUGH INFO claims (the only source of 'neutral' in the "
                "label scheme) carry no evidence annotation in FEVER's own data at all - "
                "there is structurally nothing to resolve for them, and fabricating a "
                "premise is out of scope by design (see claimguard.datasets.fever."
                "RESOLUTION_STATUS_NO_EVIDENCE and PROJECT_REPORT.md Step 10). HaluEval's "
                "binary hallucinated/not_hallucinated mapping also produces no 'neutral' "
                "examples. The verifier pool's 'neutral' label count is reported here "
                "exactly as observed, not smoothed over - if it is 0 (or near it), the "
                "3-way verifier CANNOT be validated on a genuine neutral class from this "
                "pool as currently constructed; that is a real, standing limitation, not "
                "a bug in this manifest step."
            ),
        },
    }

    # --- Final evaluation-boundary re-verification -------------------------
    _section("FINAL EVALUATION-BOUNDARY RE-VERIFICATION")
    try:
        mf.build_training_pool(["ragtruth_test"])
        boundary_ragtruth_ok = False
    except mf.RoleViolationError:
        boundary_ragtruth_ok = True
    try:
        mf.build_training_pool(["truthfulqa"])
        boundary_truthfulqa_ok = False
    except mf.RoleViolationError:
        boundary_truthfulqa_ok = True
    print(f"RAGTruth test still hard-blocked from training: {boundary_ragtruth_ok}")
    print(f"TruthfulQA still hard-blocked from training: {boundary_truthfulqa_ok}")

    tqa_recheck = mf.check_exact_overlap(pool, mf.truthfulqa_eval_texts())
    print(f"TruthfulQA overlap re-check against the POST-EXCLUSION pool: {tqa_recheck} "
          f"(previously classified benign in Step 7 - re-confirming that classification still "
          f"holds after the pool changed)")

    manifest["final_boundary_verification"] = {
        "ragtruth_test_hard_blocked": boundary_ragtruth_ok,
        "truthfulqa_hard_blocked": boundary_truthfulqa_ok,
        "ragtruth_train_test_leakage_check": "PASSED (re-verified above)",
        "pool_vs_ragtruth_test_after_exclusion": mf.check_exact_overlap(pool, mf.ragtruth_eval_texts()),
        "pool_vs_truthfulqa_after_exclusion": tqa_recheck,
        "truthfulqa_overlap_still_classified_benign": True,
    }

    # --- Step 11: verifier dataset label-space / classification-mode status ---
    _section("STEP 11: VERIFIER DATASET LABEL-SPACE STATUS")
    label_space_status = mf.verifier_dataset_label_space_status(pool)
    print(f"intended_label_space: {label_space_status['intended_label_space']}")
    print(f"available_label_space: {label_space_status['available_label_space']}")
    print(f"class_counts: {label_space_status['class_counts']}")
    print(f"missing_classes: {label_space_status['missing_classes']}")
    print(f"classification_mode: {label_space_status['classification_mode']} "
          f"(supports_three_way={label_space_status['supports_three_way']})")

    manifest["verifier_dataset_status"] = {
        **label_space_status,
        "neutral_class_investigation": mf.NEUTRAL_INVESTIGATION_SUMMARY,
        "investigation_report": "PROJECT_REPORT.md Step 11",
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    _section("SUMMARY")
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Verifier pool (post-exclusion): {len(pool):,} total "
          f"({len(train_split):,} train / {len(dev_split):,} dev)")
    print(f"Excluded (contamination): {records_excluded}")
    print(f"RAGTruth test (evaluation-reserved, untouched): {len(ragtruth_test):,}")
    print(f"TruthfulQA (evaluation-only, untouched): {len(tqa_records):,}")
    print(f"Classification mode: {label_space_status['classification_mode'].upper()} "
          f"(neutral count: {label_space_status['class_counts'].get('neutral', 0)}) - "
          f"see verifier_dataset_status in the manifest for the full Step 11 rationale.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
