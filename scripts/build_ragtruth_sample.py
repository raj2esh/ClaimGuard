"""Step 6C: build a tiny RAGTruth development sample.

RAGTruth is ClaimGuard's PRIMARY end-to-end evaluation dataset - this
sample is built EXCLUSIVELY from the `train` split. Test-split records are
never read into this script's output at all, so the reserved evaluation set
cannot be contaminated or redefined by this step, by construction (not just
by convention).

Sampling strategy: RAGTruth's real structure has two dimensions worth
covering - task_type (Summary / Data2txt / QA, each with a differently
shaped `source_info`) and hallucination presence (labels present/absent,
including real span-level annotations with label_type/offsets when
present). The sample takes the first N train-split records (in original
file order - fully deterministic) for each of the 6
(task_type x has_hallucination) combinations, so every structural
combination the real data actually contains is represented at least once,
including genuine span-annotated examples.

Usage:
    python scripts/build_ragtruth_sample.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.datasets import ragtruth  # noqa: E402

SAMPLE_DIR = cg_config.resolve_path("data/processed/ragtruth/sample")
PER_COMBINATION_TARGET = 5


def main() -> int:
    print("Loading and joining RAGTruth (response.jsonl + source_info.jsonl)...")
    all_records = ragtruth.load_normalized()
    print(f"Loaded {len(all_records):,} total normalized records "
          f"({sum(1 for r in all_records if r['split'] == 'train'):,} train, "
          f"{sum(1 for r in all_records if r['split'] == 'test'):,} test)")

    # Explicit safeguard: verify the leakage property still holds before
    # sampling, and verify the sample-building process only ever touches
    # the train pool below.
    ragtruth.assert_no_train_test_leakage(all_records)
    print("Leakage check passed: no source_id spans both splits.")

    train_only = ragtruth.get_training_pool(all_records)
    print(f"Restricting sample construction to the {len(train_only):,} TRAIN-split records only "
          f"- test split is never read by this script's sampling logic.")

    buckets: dict[tuple[str, bool], list[dict]] = {}
    for record in train_only:
        key = (record.get("task_type"), record.get("has_hallucination"))
        buckets.setdefault(key, [])
        if len(buckets[key]) >= PER_COMBINATION_TARGET:
            continue
        issues = ragtruth.validate_record(record)
        if issues:
            continue
        buckets[key].append(record)

    all_selected: list[dict] = []
    for key in sorted(buckets.keys(), key=lambda k: (str(k[0]), k[1])):
        task_type, has_halluc = key
        recs = buckets[key]
        print(f"task_type={task_type!r}, has_hallucination={has_halluc}: {len(recs)} records")
        all_selected.extend(recs)

    # Double-check, at the point of writing, that nothing test-split ever
    # entered the selection (belt-and-suspenders, not just relying on the
    # upstream filter).
    assert all(r["split"] == "train" for r in all_selected), \
        "BUG: a non-train record reached the RAGTruth sample writer"
    assert all(r["eval_reserved"] is False for r in all_selected), \
        "BUG: an eval_reserved record reached the RAGTruth sample writer"

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / "sample.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_selected:
            f.write(json.dumps(r) + "\n")

    n_with_spans = sum(1 for r in all_selected if r["labels"])
    print(f"\nWrote {len(all_selected)} records to {out_path} "
          f"({n_with_spans} include real span-level annotations).")
    print("All records are from split='train'; eval_reserved is False for every record.")
    print("NOTE: this is a tiny development sample, NOT an evaluation benchmark, and it does "
          "NOT touch or redefine RAGTruth's reserved test split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
