"""Step 6A: build a tiny FEVER development sample.

Creates a small sample (up to 10 SUPPORTS, 10 REFUTES, 10 NOT ENOUGH INFO)
from the FEVER train split for local development/testing of ClaimGuard's
verifier pipeline. This is NOT an evaluation benchmark - just enough data to
exercise code paths without processing the full ~311k-row train split.

Only the `train` split is used as the source: it is the one split with no
blank-label / casing-inconsistency issues (see
src/claimguard/datasets/fever.py docstring and PROJECT_REPORT.md Step 6A for
why `validation`/`test` are not used here).

Usage:
    python scripts/build_fever_sample.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.datasets import fever  # noqa: E402

SAMPLE_DIR = cg_config.resolve_path("data/processed/fever/sample")
PER_LABEL_TARGET = 10
SOURCE_SPLIT = "train"


def main() -> int:
    print(f"Loading and normalizing FEVER split: {SOURCE_SPLIT}")
    records = fever.load_normalized_split(SOURCE_SPLIT)
    print(f"Loaded {len(records):,} normalized claims from {SOURCE_SPLIT}")

    by_label: dict[str, list[dict]] = {lbl: [] for lbl in fever.VALID_FEVER_LABELS}
    skipped_malformed = 0

    for record in records:
        label = record.get("label")
        if label not in by_label or len(by_label[label]) >= PER_LABEL_TARGET:
            continue
        issues = fever.validate_record(record)
        if issues:
            skipped_malformed += 1
            continue
        by_label[label].append(record)
        if all(len(v) >= PER_LABEL_TARGET for v in by_label.values()):
            break

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / "sample.jsonl"
    total_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for label, recs in by_label.items():
            print(f"{label}: {len(recs)} / {PER_LABEL_TARGET} target")
            for r in recs:
                f.write(json.dumps(r) + "\n")
                total_written += 1

    print(
        f"Skipped {skipped_malformed} candidate records that failed validate_record() "
        f"while searching for clean examples (not silently dropped from the dataset as a "
        f"whole - just not included in this small clean sample)."
    )
    print(f"Wrote {total_written} records to {out_path}")
    print("NOTE: this is a tiny development sample, NOT an evaluation benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
