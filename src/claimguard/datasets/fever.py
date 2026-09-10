"""FEVER dataset loading and normalization utilities.

Loads the raw FEVER data acquired by scripts/inspect_fever.py from
data/raw/fever/<split>.jsonl and normalizes it into a per-claim schema for
ClaimGuard's verifier training/evaluation code to consume later. This module
does NOT implement model training and does NOT load any model - see
PROJECT_PLAN.md Step 11 for the actual verification module.

Real-data findings this module accounts for (full detail in PROJECT_REPORT.md
Step 6A and RESEARCH.md):
  - The raw data is DENORMALIZED: one row per evidence line, not one row per
    claim. Rows sharing the same `id` belong to the same claim; rows sharing
    the same (`id`, `evidence_annotation_id`) belong to the same evidence set
    (one of possibly several alternative evidence sets for that claim).
  - "No evidence" (NOT ENOUGH INFO claims) is encoded with SENTINEL values,
    not null: evidence_wiki_url == "" and evidence_id/evidence_sentence_id
    == -1. A naive `is None` check misses this.
  - The `validation` and `test` splits, as delivered by Hugging Face's
    auto-converted Parquet export of fever/fever (the script-based loader is
    blocked under datasets>=4: "Dataset scripts are no longer supported"),
    contain a mix of real labels, a blank label ("") representing
    blind/withheld-label rows carried over from the original unlabelled
    dev/test splits, and (validation only) a label casing inconsistency
    ("Not Enough Info" vs "NOT ENOUGH INFO"). `validation` and `test` also
    overlap by 9,999 claim IDs and validation has ~19k exact duplicate rows.
    `train` has none of these issues. Callers should filter to
    VALID_FEVER_LABELS (exact match) before treating a record as labeled
    data, and should not assume validation/test are clean or disjoint
    without further preprocessing (out of scope for this module - see
    PROJECT_REPORT.md Step 6A recommendations).
"""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .. import config as cg_config

RAW_DIR = cg_config.resolve_path("data/raw/fever")
WIKI_INDEX_PATH = cg_config.resolve_path("data/processed/fever/wiki_pages_index.sqlite")

VALID_FEVER_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}

# FEVER's 3-way label -> our DeBERTa verifier's 3-way label. The right-hand
# side is the id2label mapping empirically confirmed from the Step 5B smoke
# test: {0: 'entailment', 1: 'neutral', 2: 'contradiction'}.
FEVER_TO_VERIFIER_LABEL = {
    "SUPPORTS": "entailment",
    "REFUTES": "contradiction",
    "NOT ENOUGH INFO": "neutral",
}

NO_EVIDENCE_WIKI_URL = ""
NO_EVIDENCE_SENTINEL_ID = -1


def _row_has_evidence(row: dict[str, Any]) -> bool:
    url = row.get("evidence_wiki_url")
    return url is not None and url != NO_EVIDENCE_WIKI_URL


def load_raw_split(split: str, raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Load the raw (denormalized, one-row-per-evidence-line) JSONL for a split."""
    path = raw_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw FEVER split not found: {path}. Run scripts/inspect_fever.py first."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_claims(rows: Iterable[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Group denormalized evidence-line rows into one normalized record per claim.

    Output schema per record:
        example_id: int            - the raw 'id' field (claim id)
        claim: str
        label: str                 - raw FEVER label, as-is (may be "" or a
                                      casing variant on validation/test - see
                                      module docstring; not corrected here)
        verifier_label: str | None - FEVER_TO_VERIFIER_LABEL[label], or None
                                      if label is not in VALID_FEVER_LABELS
        evidence: list[list[dict]] - list of evidence SETS; each set is a
                                      list of {"wiki_url": str,
                                      "sentence_id": int} dicts (sentinel
                                      "no evidence" entries dropped, but an
                                      empty set [] is kept as a placeholder
                                      for e.g. a NOT ENOUGH INFO annotation)
        evidence_ids: list[int]    - the evidence_annotation_id of each
                                      evidence set, same order as `evidence`
        split: str
    """
    by_claim: dict[int, dict[str, Any]] = {}
    evidence_sets: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        cid = row["id"]
        if cid not in by_claim:
            by_claim[cid] = {
                "example_id": cid,
                "claim": row.get("claim", ""),
                "label": row.get("label", ""),
                "split": split,
            }
        ann_id = row.get("evidence_annotation_id")
        if _row_has_evidence(row):
            evidence_sets[cid][ann_id].append(
                {"wiki_url": row["evidence_wiki_url"], "sentence_id": row["evidence_sentence_id"]}
            )
        else:
            evidence_sets[cid][ann_id]  # noqa: B018 - touch to create the empty set

    records = []
    for cid, record in by_claim.items():
        ann_map = evidence_sets[cid]
        ordered_ann_ids = sorted(ann_map.keys())  # stable order for reproducibility
        record["evidence_ids"] = ordered_ann_ids
        record["evidence"] = [ann_map[a] for a in ordered_ann_ids]
        record["verifier_label"] = FEVER_TO_VERIFIER_LABEL.get(record["label"])
        records.append(record)
    return records


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of integrity issue codes for a normalized record.

    Does not raise or discard anything - callers decide what to do with the
    issues found (never silently discard problematic examples).
    """
    issues: list[str] = []
    if not record.get("claim"):
        issues.append("empty_or_missing_claim")

    label = record.get("label")
    if label not in VALID_FEVER_LABELS:
        issues.append(f"invalid_label:{label!r}")

    evidence = record.get("evidence", [])
    has_any_evidence = any(len(s) > 0 for s in evidence)
    if label in ("SUPPORTS", "REFUTES") and not has_any_evidence:
        issues.append("supports_or_refutes_without_evidence")
    if label == "NOT ENOUGH INFO" and has_any_evidence:
        issues.append("not_enough_info_with_evidence")

    return issues


def load_normalized_split(split: str, raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Load a split's raw data and return normalized per-claim records."""
    rows = load_raw_split(split, raw_dir=raw_dir)
    return group_claims(rows, split=split)


# ---------------------------------------------------------------------------
# Evidence -> premise text resolution (Step 10).
#
# The functions below resolve a claim's evidence references (wiki_url,
# sentence_id pairs recorded above) into actual Wikipedia sentence text, via
# the wiki_pages SQLite index built by scripts/build_wiki_pages_index.py
# (scripts/acquire_inspect_wiki_pages.py acquires the underlying corpus).
# This is the exact same deterministic method validated end-to-end at 100%
# resolution for FEVER train SUPPORTS/REFUTES claims in
# scripts/resolve_fever_evidence.py; see PROJECT_REPORT.md Step 10 for the
# full investigation (two real edge cases found and fixed: the
# sentence_id == -1 "whole-page evidence" sentinel, and Unicode NFC
# normalization mismatches between claims JSON and wiki_pages page ids).
#
# NOT ENOUGH INFO claims carry no evidence annotation in FEVER's own data at
# all (evidence == [[]] or [] - the sentinel-empty case from group_claims
# above) - there is nothing to resolve, and fabricating a premise for them
# is out of scope by design. Their resolution_status is "no_evidence_annotation",
# never "unresolved" - "unresolved" is reserved for a SUPPORTS/REFUTES claim
# whose evidence exists but could not be found in the wiki_pages index (a
# real gap), so the two cases are never confused with each other.
# ---------------------------------------------------------------------------

RESOLVABLE_LABELS = {"SUPPORTS", "REFUTES"}
NOT_ENOUGH_INFO_LABEL = "NOT ENOUGH INFO"

RESOLUTION_STATUS_RESOLVED = "resolved"
RESOLUTION_STATUS_NO_EVIDENCE = "no_evidence_annotation"
RESOLUTION_STATUS_UNRESOLVED = "unresolved"
RESOLUTION_STATUS_UNKNOWN_LABEL = "unknown_label"


def open_wiki_index(index_path: Path = WIKI_INDEX_PATH) -> sqlite3.Connection:
    """Open a read-only connection to the wiki_pages resolution index.

    Raises FileNotFoundError (rather than letting sqlite3 silently create an
    empty database file) if the index hasn't been built yet.
    """
    if not index_path.exists():
        raise FileNotFoundError(
            f"wiki_pages index not found: {index_path}. "
            "Run scripts/acquire_inspect_wiki_pages.py then "
            "scripts/build_wiki_pages_index.py first."
        )
    return sqlite3.connect(str(index_path))


def _resolve_page_id(conn: sqlite3.Connection, page_id: str) -> str | None:
    """Return the page_id actually present in the index for `page_id`.

    Tries an exact match first, then falls back to Unicode NFC-normalized
    equality. This is deterministic canonicalization (exact string equality
    after standard Unicode normalization), NOT fuzzy matching - validated as
    necessary and sufficient in Step 10 (all page ids that failed exact
    lookup resolved under NFC; none were genuinely absent). Returns None if
    the page is absent under both forms.
    """
    if conn.execute("SELECT 1 FROM pages WHERE page_id = ?", (page_id,)).fetchone() is not None:
        return page_id
    nfc = unicodedata.normalize("NFC", page_id)
    if nfc != page_id and conn.execute("SELECT 1 FROM pages WHERE page_id = ?", (nfc,)).fetchone() is not None:
        return nfc
    return None


def resolve_sentence(conn: sqlite3.Connection, page_id: str, sentence_id: int) -> tuple[str | None, str]:
    """Resolve one (page_id, sentence_id) evidence reference to text.

    Returns (text_or_None, failure_reason); failure_reason is "" on success.

    sentence_id == -1 is a real FEVER annotation sentinel meaning "evidence
    is the page as a whole" (confirmed in Step 10: every such reference in
    FEVER train points at a page that exists with real content) - resolved
    via the page's own `text` field, not a (nonexistent) negative-index
    sentence lookup, so this real convention is never misclassified as
    missing data.
    """
    resolved_page_id = _resolve_page_id(conn, page_id)

    if sentence_id == -1:
        if resolved_page_id is None:
            return None, "missing_page"
        page_row = conn.execute(
            "SELECT text FROM pages WHERE page_id = ?", (resolved_page_id,)
        ).fetchone()
        page_text = page_row[0]
        if not page_text or not page_text.strip():
            return None, "empty_resolved_text"
        return page_text, ""

    if resolved_page_id is None:
        return None, "missing_page"
    row = conn.execute(
        "SELECT text FROM sentences WHERE page_id = ? AND sentence_id = ?",
        (resolved_page_id, sentence_id),
    ).fetchone()
    if row is None:
        return None, "missing_sentence_id"
    text = row[0]
    if not text or not text.strip():
        return None, "empty_resolved_text"
    return text, ""


def resolve_claim_premise(conn: sqlite3.Connection, record: dict[str, Any]) -> dict[str, Any]:
    """Resolve one normalized claim record (from group_claims) to premise text.

    Tries each evidence set in `record["evidence"]` in the existing
    deterministic order (set 0 first); uses the first set that resolves
    COMPLETELY. Sentences from different evidence sets are never mixed into
    one premise, and a partially-resolved set is never used.

    Returns a dict with:
        resolution_status: one of RESOLUTION_STATUS_* above
        resolved_premise: str | None  - sentence texts from the resolved
                                         set, joined with a single space, in
                                         evidence-set order; None unless
                                         resolution_status == "resolved"
        resolved_evidence_set_index: int | None - which evidence set in
                                         record["evidence"] resolved; None
                                         unless status == "resolved"
        premise_text_available: bool  - convenience flag, True iff status
                                         == "resolved"

    Does not raise on failure and does not silently promote an unresolved
    or no-evidence claim into a "resolved" one - callers must check
    resolution_status / premise_text_available before treating a record as
    usable training data.
    """
    label = record.get("label")

    if label == NOT_ENOUGH_INFO_LABEL:
        return {
            "resolution_status": RESOLUTION_STATUS_NO_EVIDENCE,
            "resolved_premise": None,
            "resolved_evidence_set_index": None,
            "premise_text_available": False,
        }

    if label not in RESOLVABLE_LABELS:
        # e.g. blank/withheld labels on validation/test (see module docstring).
        return {
            "resolution_status": RESOLUTION_STATUS_UNKNOWN_LABEL,
            "resolved_premise": None,
            "resolved_evidence_set_index": None,
            "premise_text_available": False,
        }

    for set_idx, evidence_set in enumerate(record.get("evidence", [])):
        if not evidence_set:
            continue
        sentence_texts: list[str] = []
        set_ok = True
        for sent_ref in evidence_set:
            text, _failure_reason = resolve_sentence(
                conn, sent_ref["wiki_url"], sent_ref["sentence_id"]
            )
            if text is None:
                set_ok = False
                break
            sentence_texts.append(text)
        if set_ok:
            return {
                "resolution_status": RESOLUTION_STATUS_RESOLVED,
                "resolved_premise": " ".join(sentence_texts),
                "resolved_evidence_set_index": set_idx,
                "premise_text_available": True,
            }

    return {
        "resolution_status": RESOLUTION_STATUS_UNRESOLVED,
        "resolved_premise": None,
        "resolved_evidence_set_index": None,
        "premise_text_available": False,
    }


def load_resolved_split(
    split: str,
    raw_dir: Path = RAW_DIR,
    index_path: Path = WIKI_INDEX_PATH,
) -> list[dict[str, Any]]:
    """Load a split's normalized claims and attach resolved premise text.

    Each returned record is the same normalized record produced by
    load_normalized_split/group_claims (example_id, claim, label,
    verifier_label, evidence, evidence_ids, split - all preserved
    unchanged) plus the four resolution fields documented in
    resolve_claim_premise(): resolution_status, resolved_premise,
    resolved_evidence_set_index, premise_text_available.

    Raw data is never modified - this only adds derived fields to the
    in-memory normalized record. Unresolved and no-evidence-annotation
    records are returned like any other record (not dropped) so callers can
    decide how to filter them; premise_text_available is the single flag to
    check before using a record as a (premise, claim, label) training
    example.
    """
    records = load_normalized_split(split, raw_dir=raw_dir)
    conn = open_wiki_index(index_path)
    try:
        for record in records:
            record.update(resolve_claim_premise(conn, record))
    finally:
        conn.close()
    return records
