"""Step 7/8/10: materialize the verifier training/development pool to disk.

Writes the deterministic, leakage-safe verifier training/dev pool (built by
claimguard.datasets.manifest from FEVER `train` (SUPPORTS/REFUTES claims
with resolved premise text as of Step 10 - NOT ENOUGH INFO claims are
excluded, see manifest.fever_train_pool_records) + HaluEval `qa`/`dialogue`/
`summarization`, with the 100 cross-dataset-contaminated HaluEval
summarization records excluded by default as of Step 8) to:

    data/processed/verifier_pool/train.jsonl
    data/processed/verifier_pool/dev.jsonl
    data/processed/verifier_pool/excluded_contamination.jsonl

The third file is written for audit purposes only - it is never read back
into train/dev, and exists so the exclusion is independently inspectable
rather than just asserted.

This is bookkeeping/reproducibility output only - it does NOT train any
model. Re-running this script with the same code produces byte-identical
group assignments (same seed, same deterministic split, same
deterministically-computed contamination exclusion).

Usage:
    python scripts/build_verifier_pool.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.datasets import manifest as mf  # noqa: E402

OUT_DIR = cg_config.resolve_path("data/processed/verifier_pool")


def _write(records: list[mf.PoolRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), default=str) + "\n")


def main() -> int:
    print(f"Building verifier training pool from: {sorted(mf.ALLOWED_TRAINING_SOURCES)}")
    pool_before = mf.build_training_pool(exclude_contamination=False)
    pool = mf.build_training_pool()  # exclude_contamination=True (default)
    excluded_keys = mf.contamination_exclusion_group_keys()
    excluded_records = [r for r in pool_before if r.group_key in excluded_keys]
    print(f"Total pool records before contamination exclusion: {len(pool_before):,}")
    print(f"Excluded (reason={mf.EXCLUSION_REASON_RAGTRUTH_OVERLAP}): {len(excluded_records):,}")
    print(f"Total pool records after exclusion: {len(pool):,}")

    train, dev = mf.split_train_dev(pool)
    print(f"Deterministic split (seed={mf.SEED}, dev_ratio={mf.DEV_RATIO}): "
          f"{len(train):,} train / {len(dev):,} dev")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write(train, OUT_DIR / "train.jsonl")
    _write(dev, OUT_DIR / "dev.jsonl")
    _write(excluded_records, OUT_DIR / "excluded_contamination.jsonl")

    for name, split in (("train", train), ("dev", dev)):
        label_dist = Counter(r.label for r in split)
        source_dist = Counter(f"{r.source_dataset}.{r.source_subset}" for r in split)
        print(f"\n{name}: {len(split):,} records")
        print(f"  label distribution: {dict(label_dist)}")
        print(f"  source distribution: {dict(source_dist)}")

    print(f"\nWrote {OUT_DIR / 'train.jsonl'}")
    print(f"Wrote {OUT_DIR / 'dev.jsonl'}")
    print(f"Wrote {OUT_DIR / 'excluded_contamination.jsonl'} (audit only, {len(excluded_records)} records)")
    neutral_count = sum(1 for r in train + dev if r.label == "neutral")
    print("\nNOTE: this pool is training/dev bookkeeping output only - no model has been "
          "trained. As of Step 10, FEVER SUPPORTS/REFUTES records in this pool have real, "
          "resolved Wikipedia sentence premise_text (premise_text_available=True) via the "
          "wiki_pages evidence resolution index. FEVER NOT ENOUGH INFO claims still have no "
          "evidence annotation in FEVER's own data at all and are excluded, not coerced - so "
          f"this pool's 'neutral' label count is {neutral_count} (a real, unresolved coverage "
          "gap - see PROJECT_REPORT.md Step 10, not a bug in this script). RAGTruth test and "
          "TruthfulQA were not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
