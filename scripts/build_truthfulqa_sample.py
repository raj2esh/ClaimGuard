"""Step 6D: build a tiny TruthfulQA development sample.

TruthfulQA has NO official train/validation/test split (verified empirically
in Step 6D - no split-like column exists in either raw file). This sample is
therefore explicitly a development/inspection fixture only, covering the
dataset's real structure - it must not be read as, or presented as, an
official evaluation split.

Sampling strategy: one representative question per category (first
encountered in file order - fully deterministic), covering all 37 real
categories, so every category the real data actually contains is
represented at least once. This directly follows the requirement to give
"representative coverage across categories and answer-label structures" -
every sampled record also carries its full best/correct/incorrect answer
lists and all three mc0/mc1/mc2 multiple-choice representations, so all of
TruthfulQA's real answer-label structures are exercised too.

Usage:
    python scripts/build_truthfulqa_sample.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.datasets import truthfulqa  # noqa: E402

SAMPLE_DIR = cg_config.resolve_path("data/processed/truthfulqa/sample")


def main() -> int:
    print("Loading and joining TruthfulQA (TruthfulQA.csv + mc_task.json)...")
    all_records = truthfulqa.load_normalized()
    print(f"Loaded {len(all_records):,} normalized question records.")

    dupes = truthfulqa.find_duplicate_questions(all_records)
    print(f"Duplicate questions found: {len(dupes)} (expected 0, per Step 6D inspection).")

    seen_categories: set[str] = set()
    selected: list[dict] = []
    skipped = 0
    for record in all_records:
        category = record.get("category")
        if category in seen_categories:
            continue
        issues = truthfulqa.validate_record(record)
        if issues:
            skipped += 1
            print(f"  skipping first candidate for category={category!r} due to "
                  f"validate_record() issues: {issues} (will try the next one in that category)")
            continue
        seen_categories.add(category)
        selected.append(record)

    print(f"\nSelected {len(selected)} records covering {len(seen_categories)} distinct categories.")
    print(f"Skipped {skipped} candidate records that failed validate_record() while selecting "
          f"(not silently dropped from the dataset as a whole - just not chosen for this sample).")

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / "sample.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(selected)} records to {out_path}")
    print("NOTE: TruthfulQA has NO official split. This is a development/inspection sample "
          "only, NOT an official evaluation split - do not treat it as one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
