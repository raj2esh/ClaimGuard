"""Step 8: detect and report cross-dataset contamination.

Independently reproduces Step 7's discovery (100 exact document matches
between HaluEval `summarization` premise text and RAGTruth `test` source
articles), identifies it at the individual-record level (HaluEval
raw_index, RAGTruth source_id/response_id - not just a bare count), and
runs a systematic exact-text-match sweep of every other candidate training
source against both evaluation datasets (RAGTruth test, TruthfulQA).

Writes data/processed/contamination_report.json. Does NOT modify any raw
dataset file - read-only analysis only. Does NOT touch the GPU or load any
model.

Usage:
    python scripts/detect_contamination.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.datasets import manifest as mf  # noqa: E402

REPORT_PATH = cg_config.resolve_path("data/processed/contamination_report.json")


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    _section("REPRODUCING STEP 7's HALUEVAL-SUMMARIZATION / RAGTRUTH-TEST OVERLAP")
    audit = mf.find_ragtruth_summarization_contamination()
    print(f"Audit records (halueval_raw_index x ragtruth_test_response pairs): {len(audit)}")

    affected_halueval_indices = sorted({a["halueval_raw_index"] for a in audit})
    affected_source_ids = sorted({a["ragtruth_source_id"] for a in audit})
    affected_response_ids = sorted({a["ragtruth_test_response_id"] for a in audit})

    print(f"Unique HaluEval summarization raw indices affected: {len(affected_halueval_indices)}")
    print(f"Unique RAGTruth test source_ids affected: {len(affected_source_ids)}")
    print(f"Unique RAGTruth test response records affected: {len(affected_response_ids)}")
    print(f"(Not assumed equal to each other or to the audit-record count - reported "
          f"separately per the actual grouping found.)")

    if len(affected_halueval_indices) != len(affected_source_ids):
        print(f"NOTE: HaluEval-index count ({len(affected_halueval_indices)}) differs from "
              f"RAGTruth-source_id count ({len(affected_source_ids)}) - indicates a "
              f"non-1:1 text-to-record mapping on at least one side (duplicate documents). "
              f"See audit_records in the report for the exact mapping.")

    _section("CROSS-DATASET CONTAMINATION SWEEP (all candidate sources)")
    sweep = mf.cross_dataset_contamination_sweep()
    for source, result in sweep.items():
        print(f"{source}:")
        print(f"  vs RAGTruth test: {result['vs_ragtruth_test']}")
        print(f"  vs TruthfulQA:    {result['vs_truthfulqa']}")

    _section("MITIGATION - EXCLUSION MECHANISM")
    exclusion_keys = mf.contamination_exclusion_group_keys()
    print(f"Reason code: {mf.EXCLUSION_REASON_RAGTRUTH_OVERLAP}")
    print(f"Excluded group_keys (computed fresh from raw data, not hard-coded): {len(exclusion_keys)}")

    _section("VERIFYING THE MITIGATION - BEFORE vs AFTER")
    pool_before = mf.build_training_pool(exclude_contamination=False)
    pool_after = mf.build_training_pool(exclude_contamination=True)
    print(f"Pool size WITHOUT contamination exclusion: {len(pool_before):,}")
    print(f"Pool size WITH contamination exclusion (default, actual pool): {len(pool_after):,}")
    print(f"Records excluded: {len(pool_before) - len(pool_after):,}")

    verification = mf.verify_contamination_resolved(pool_after)
    print(f"Re-running Step 7's original (broader, all-task-type) overlap check against the "
          f"FILTERED pool: {verification}")
    if verification["remaining_premise_overlap_with_ragtruth_source_info_text"] != 0:
        print("WARNING: contamination NOT fully resolved by the exclusion - investigate before proceeding.")
    else:
        print("Confirmed: 0 remaining overlap. Mitigation verified effective using the same "
              "methodology that found the original problem.")

    report = {
        "method": "exact normalized (lowercase, whitespace-collapsed, stripped) text match only "
                  "- not a near-duplicate or fuzzy-match check",
        "reproduced_from": "Step 7 (PROJECT_REPORT.md / RESEARCH.md), originally found via the "
                            "aggregate pool-vs-ragtruth-test check",
        "halueval_ragtruth_summarization_overlap": {
            "audit_record_count": len(audit),
            "counts": {
                "unique_halueval_raw_indices_affected": len(affected_halueval_indices),
                "unique_ragtruth_source_ids_affected": len(affected_source_ids),
                "unique_ragtruth_test_responses_affected": len(affected_response_ids),
            },
            "affected_halueval_raw_indices": affected_halueval_indices,
            "affected_ragtruth_source_ids": affected_source_ids,
            "affected_ragtruth_test_response_ids": affected_response_ids,
            "audit_records": audit,
        },
        "exclusion": {
            "reason_code": mf.EXCLUSION_REASON_RAGTRUTH_OVERLAP,
            "mechanism": "claimguard.datasets.manifest.contaminated_halueval_summarization_raw_indices() "
                         "is computed fresh from the raw data on every call (not a hard-coded row-number "
                         "list); build_training_pool() excludes these by PoolRecord.group_key by default "
                         "(exclude_contamination=True is the default).",
            "excluded_group_keys_count": len(exclusion_keys),
        },
        "pool_before_vs_after": {
            "pool_size_without_exclusion": len(pool_before),
            "pool_size_with_exclusion": len(pool_after),
            "records_excluded": len(pool_before) - len(pool_after),
        },
        "mitigation_verification": verification,
        "cross_dataset_sweep": sweep,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    _section("SUMMARY")
    print(f"Report written to {REPORT_PATH}")
    print(f"Records excluded from training pool: {len(pool_before) - len(pool_after):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
