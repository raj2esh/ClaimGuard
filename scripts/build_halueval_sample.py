"""Step 6B: build a tiny HaluEval development sample.

Creates a small, deterministic sample covering ALL FOUR HaluEval subsets, not
just one - representative of the dataset's actual structure (two materially
different shapes: qa/dialogue/summarization contrastive pairs, and general's
single labeled responses), not of any particular label's real-world
frequency. Selection is always "first N in file order" - no randomness, so
the sample is exactly reproducible.

Sizing rationale:
  - qa / dialogue / summarization: first 5 raw records each, expanded to 10
    normalized records each (5 hallucinated + 5 not_hallucinated - the
    expansion in halueval.py always yields one of each per raw record, so
    this is naturally label-balanced with no extra selection logic needed).
  - general: first 5 "yes" (hallucinated) and first 5 "no" (not_hallucinated)
    records in file order, since general's real label distribution is
    imbalanced (~82% no / 18% yes - see PROJECT_REPORT.md Step 6B) and a
    naive "first 10 rows" would likely be almost all one label. This is a
    deliberate departure from pure "first N" for this subset only, needed to
    exercise both code paths - documented here rather than left implicit.

This is NOT an evaluation benchmark - just enough data, across every subset
shape, to exercise ClaimGuard's HaluEval-handling code paths in tests.

Usage:
    python scripts/build_halueval_sample.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.datasets import halueval  # noqa: E402

SAMPLE_DIR = cg_config.resolve_path("data/processed/halueval/sample")
PAIR_SUBSET_RAW_COUNT = 5  # -> 10 normalized records per subset (5 + 5)
GENERAL_PER_LABEL_TARGET = 5


def build_pair_subset_sample(subset: str) -> list[dict]:
    raw_records = halueval.load_raw_jsonl(halueval.PAIR_SUBSETS[subset]["file"])
    out = []
    for i, raw in enumerate(raw_records[:PAIR_SUBSET_RAW_COUNT]):
        out.extend(halueval.expand_pair_record(raw, subset, i))
    return out


def build_general_sample() -> list[dict]:
    raw_records = halueval.load_raw_jsonl(halueval.GENERAL_SUBSET_FILE)
    by_label: dict[str, list[dict]] = {"hallucinated": [], "not_hallucinated": []}
    for i, raw in enumerate(raw_records):
        record = halueval.normalize_general_record(raw, i)
        label = record["label"]
        if label in by_label and len(by_label[label]) < GENERAL_PER_LABEL_TARGET:
            by_label[label].append(record)
        if all(len(v) >= GENERAL_PER_LABEL_TARGET for v in by_label.values()):
            break
    return by_label["hallucinated"] + by_label["not_hallucinated"]


def main() -> int:
    all_records: list[dict] = []

    for subset in halueval.PAIR_SUBSETS:
        recs = build_pair_subset_sample(subset)
        print(f"{subset}: {len(recs)} normalized records "
              f"({PAIR_SUBSET_RAW_COUNT} raw records x 2 expansion)")
        all_records.extend(recs)

    general_recs = build_general_sample()
    n_hallucinated = sum(1 for r in general_recs if r["label"] == "hallucinated")
    n_not = sum(1 for r in general_recs if r["label"] == "not_hallucinated")
    print(f"general: {len(general_recs)} normalized records "
          f"({n_hallucinated} hallucinated, {n_not} not_hallucinated)")
    all_records.extend(general_recs)

    skipped = 0
    clean_records = []
    for r in all_records:
        issues = halueval.validate_record(r)
        if issues:
            skipped += 1
            print(f"  skipping {r['example_id']} from sample due to validate_record() issues: {issues}")
            continue
        clean_records.append(r)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / "sample.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in clean_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nSkipped {skipped} candidate records that failed validate_record() (not silently "
          f"dropped from the dataset as a whole - just not included in this clean sample).")
    print(f"Wrote {len(clean_records)} records to {out_path}")
    print("NOTE: this is a tiny development sample, NOT an evaluation benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
