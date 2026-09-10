"""ClaimGuard retrieval: corpus definition and construction (Step 13).

## Retrieval objective (see RESEARCH.md "Proposed Methodology" step 3 and
PROJECT_REPORT.md Step 13 for the full discussion)

RESEARCH.md's methodology retrieves evidence per claim via a BM25+FAISS
retriever against a Wikipedia-derived knowledge source; FEVER's own
`wiki_pages` corpus (acquired and indexed in Step 10 -
`data/processed/fever/wiki_pages_index.sqlite`, 5,416,536 pages /
42,041,084 sentences) is the only Wikipedia knowledge source this project
has acquired, and RESEARCH.md's Dataset Strategy table lists FEVER as the
verifier's training/validation source - so `wiki_pages` is the correct
retrieval corpus, not RAGTruth (evaluation-reserved, and its per-example
source documents are not a general knowledge base anyway) and not
TruthfulQA (no grounding corpus at all, evaluation-only).

## Corpus scope - a principled, bounded subset, not the full 42M sentences

The full `wiki_pages` corpus is NOT embedded in this step. Investigated and
quantified before deciding (not guessed):
  - Full corpus: 5,416,536 pages / 42,041,084 sentences - embedding this
    exhaustively would require many hours of GPU time and ~86GB+ of
    embedding storage even at fp16, on a SHARED GPU, for a step whose
    stated purpose is "establish retrieval INFRASTRUCTURE," not build a
    production-scale index.
  - The FEVER train split's SUPPORTS/REFUTES claims (the same claims this
    project's verifier was trained on, Steps 10-12) reference exactly
    **12,549 unique Wikipedia pages** as evidence (across ALL evidence
    sets recorded per claim, not just the one Step 10 ultimately
    resolved) - a bounded, already-validated scope tied directly to this
    project's existing experimental data.
  - Indexing every SENTENCE of those pages (not just the single cited
    evidence sentence - so retrieval must actually distinguish the right
    sentence from its neighbors, rather than trivially matching a
    memorized (claim, sentence) pair) yields **158,658 sentences** -
    tractable to embed fully within this step, while still being a real,
    substantial, correctly-provenanced Wikipedia corpus.

This corpus is the FEVER-evidence-page-scoped retrieval corpus for Step
13's infrastructure-building purpose. Expanding to the full `wiki_pages`
corpus (or a larger principled subset) is a natural, separate future
scaling step - not attempted here, and not silently substituted for
without saying so.

This module NEVER reads RAGTruth or TruthfulQA data - it does not even
import those dataset modules, so there is no code path by which their
content could reach the retrieval corpus.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .. import config as cg_config
from ..datasets import fever as fever_ds

WIKI_INDEX_PATH = fever_ds.WIKI_INDEX_PATH
CORPUS_DIR = cg_config.resolve_path("data/processed/retrieval")
CORPUS_JSONL_PATH = CORPUS_DIR / "corpus.jsonl"
CORPUS_MANIFEST_PATH = CORPUS_DIR / "corpus_manifest.json"

CORPUS_VERSION = "fever_evidence_pages_v1"


def corpus_id(page_id: str, sentence_id: int) -> str:
    """Stable, human-readable identifier - deterministic given (page_id,
    sentence_id), independent of any FAISS row position or file ordering.
    """
    return f"{page_id}::{sentence_id}"


def evidence_referenced_page_ids(raw_dir: Path | None = None) -> set[str]:
    """Every unique Wikipedia page_id referenced by ANY evidence set of ANY
    FEVER train SUPPORTS/REFUTES claim - the corpus scope's page set. NOT
    limited to the single evidence set Step 10's resolver ultimately used,
    so a claim with multiple alternative annotated evidence sets
    contributes every one of them to the retrieval corpus's page scope.
    """
    kwargs = {"raw_dir": raw_dir} if raw_dir else {}
    claims = fever_ds.load_normalized_split("train", **kwargs)
    page_ids: set[str] = set()
    for c in claims:
        if c["label"] not in fever_ds.RESOLVABLE_LABELS:
            continue
        for evidence_set in c["evidence"]:
            for sent_ref in evidence_set:
                page_ids.add(sent_ref["wiki_url"])
    return page_ids


def build_corpus_records(
    index_path: Path = WIKI_INDEX_PATH, raw_dir: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the corpus record list (deterministically ordered) and a stats
    dict documenting exactly what was included/excluded and why - never
    silent. Each record: {corpus_id, page_id, sentence_id, text,
    corpus_version}.

    Page lookup reuses `claimguard.datasets.fever`'s already-validated
    exact-match-then-NFC-fallback resolver (Step 10) rather than a fresh
    exact-only lookup, so this corpus benefits from the same deterministic
    canonicalization that achieved 100% FEVER evidence resolution.
    """
    page_ids = evidence_referenced_page_ids(raw_dir=raw_dir)
    conn = fever_ds.open_wiki_index(index_path)

    records: list[dict[str, Any]] = []
    resolved_page_ids: dict[str, str] = {}  # requested -> resolved (may differ under NFC)
    missing_pages: list[str] = []
    empty_text_skipped = 0

    for requested_page_id in sorted(page_ids):
        resolved = fever_ds._resolve_page_id(conn, requested_page_id)
        if resolved is None:
            missing_pages.append(requested_page_id)
            continue
        resolved_page_ids[requested_page_id] = resolved

    unique_resolved_pages = sorted(set(resolved_page_ids.values()))
    for page_id in unique_resolved_pages:
        rows = conn.execute(
            "SELECT sentence_id, text FROM sentences WHERE page_id = ? ORDER BY sentence_id",
            (page_id,),
        ).fetchall()
        for sentence_id, text in rows:
            if not text or not text.strip():
                empty_text_skipped += 1
                continue
            records.append({
                "corpus_id": corpus_id(page_id, sentence_id),
                "page_id": page_id,
                "sentence_id": sentence_id,
                "text": text,
                "corpus_version": CORPUS_VERSION,
            })

    conn.close()

    stats = {
        "corpus_version": CORPUS_VERSION,
        "requested_evidence_page_ids": len(page_ids),
        "resolved_unique_pages": len(unique_resolved_pages),
        "missing_pages": missing_pages,
        "missing_page_count": len(missing_pages),
        "empty_text_sentences_skipped": empty_text_skipped,
        "total_corpus_records": len(records),
        "source_wiki_pages_index": str(index_path),
        "source_fever_split": "train",
        "source_fever_labels": sorted(fever_ds.RESOLVABLE_LABELS),
    }
    return records, stats


def write_corpus(
    records: list[dict[str, Any]], path: Path = CORPUS_JSONL_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_corpus(path: Path = CORPUS_JSONL_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Retrieval corpus not found: {path}. Run scripts/build_retrieval_corpus.py first."
        )
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_corpus(records: list[dict[str, Any]]) -> list[str]:
    """Return a list of integrity issue codes - never raises or discards,
    callers decide what to do (matches the project's validate_record
    convention elsewhere)."""
    issues: list[str] = []
    seen_ids: set[str] = set()
    for r in records:
        if r["corpus_id"] in seen_ids:
            issues.append(f"duplicate_corpus_id:{r['corpus_id']}")
        seen_ids.add(r["corpus_id"])
        if not r.get("text") or not r["text"].strip():
            issues.append(f"empty_text:{r['corpus_id']}")
        if r.get("corpus_id") != corpus_id(r["page_id"], r["sentence_id"]):
            issues.append(f"corpus_id_mismatch:{r['corpus_id']}")
    return issues
