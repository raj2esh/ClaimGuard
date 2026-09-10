"""Step 13: build the FEVER-evidence-scoped retrieval corpus.

Does NOT touch RAGTruth or TruthfulQA. Does NOT modify raw FEVER data or
the wiki_pages index - read-only against both, writes only to
data/processed/retrieval/. See src/claimguard/retrieval/corpus.py for the
full corpus-scope rationale.

Usage:
    python scripts/build_retrieval_corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.retrieval import corpus as corpus_mod  # noqa: E402


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    _section("BUILDING RETRIEVAL CORPUS")
    records, stats = corpus_mod.build_corpus_records()
    print(f"Requested evidence page IDs: {stats['requested_evidence_page_ids']:,}")
    print(f"Resolved unique pages: {stats['resolved_unique_pages']:,}")
    print(f"Missing pages (investigate, not hidden): {stats['missing_page_count']}")
    if stats["missing_pages"]:
        print(f"  sample: {stats['missing_pages'][:10]}")
    print(f"Empty-text sentences skipped: {stats['empty_text_sentences_skipped']:,}")
    print(f"Total corpus records: {stats['total_corpus_records']:,}")

    _section("VALIDATING CORPUS")
    issues = corpus_mod.validate_corpus(records)
    print(f"Validation issues: {len(issues)}")
    if issues:
        print(f"  sample: {issues[:10]}")
        print("FAILED - refusing to write a corpus with integrity issues.")
        return 1

    _section("WRITING CORPUS")
    corpus_mod.write_corpus(records)
    print(f"Wrote {len(records):,} records to {corpus_mod.CORPUS_JSONL_PATH}")

    manifest = {
        "stage": "corpus",
        "corpus_construction": stats,
        "validation_issues": len(issues),
    }
    corpus_mod.CORPUS_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with corpus_mod.CORPUS_MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Wrote corpus manifest to {corpus_mod.CORPUS_MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
