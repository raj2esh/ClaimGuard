"""Step 10: resolve FEVER evidence references against the wiki_pages index.

For every SUPPORTS/REFUTES claim in FEVER `train`, attempts to resolve one
of its evidence SETS (in the existing deterministic order already produced
by claimguard.datasets.fever.group_claims - evidence[0] first, then
evidence[1], etc.) into full sentence text via the wiki_pages SQLite index
(scripts/build_wiki_pages_index.py). A claim resolves if ANY ONE of its
evidence sets resolves COMPLETELY (every sentence in that set found, with
non-empty text) - sentences from DIFFERENT evidence sets are never mixed
into one premise, and a partially-resolved set is not used.

NOT ENOUGH INFO claims are NOT attempted here: they carry no evidence
annotation in FEVER's own data at all (evidence_wiki_url == "", confirmed
empirically in Step 6A and reconfirmed in Step 10) - there is nothing to
resolve, and fabricating a premise for them is out of scope by design (see
PROJECT_REPORT.md Step 10 for the full reasoning).

sentence_id == -1 is handled as a real FEVER sentinel ("whole-page
evidence") via the page's own `text` field, not treated as missing data -
see resolve_sentence() and PROJECT_REPORT.md Step 10 for the investigation
that established this (a genuine annotation convention, confirmed by
checking that the referenced pages exist with real content, not a resolver
bug left unfixed).

A page_id that isn't found on first (exact) lookup is retried once against
its Unicode NFC-normalized form. This is a deterministic canonicalization
(exact string equality after standard Unicode normalization), NOT fuzzy
matching - confirmed necessary and sufficient by investigation (Step 10):
all 123 unique page ids that failed exact lookup resolved successfully
under NFC normalization, and zero were genuinely absent from wiki_pages.

Produces data/processed/fever/wiki_resolution_report.json - a full,
investigable audit, not just a summary count. Does not modify any raw data.

Usage:
    python scripts/resolve_fever_evidence.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.datasets import fever as fever_ds  # noqa: E402

INDEX_PATH = cg_config.resolve_path("data/processed/fever/wiki_pages_index.sqlite")
REPORT_PATH = cg_config.resolve_path("data/processed/fever/wiki_resolution_report.json")

RESOLVABLE_LABELS = {"SUPPORTS", "REFUTES"}


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _resolve_page_id(conn: sqlite3.Connection, page_id: str) -> str | None:
    """Return the page_id actually present in the index, trying an exact
    match first and falling back to NFC-normalized equality (deterministic
    canonicalization, not fuzzy matching - see module docstring). Returns
    None if the page is absent under both forms.
    """
    if conn.execute("SELECT 1 FROM pages WHERE page_id = ?", (page_id,)).fetchone() is not None:
        return page_id
    nfc = unicodedata.normalize("NFC", page_id)
    if nfc != page_id and conn.execute("SELECT 1 FROM pages WHERE page_id = ?", (nfc,)).fetchone() is not None:
        return nfc
    return None


def resolve_sentence(conn: sqlite3.Connection, page_id: str, sentence_id: int) -> tuple[str | None, str]:
    """Returns (text_or_None, failure_reason). failure_reason is "" on success.

    sentence_id == -1 is a real FEVER annotation sentinel (confirmed
    empirically in Step 10: 100% of ~89k such references point at pages
    that exist with real content) meaning "evidence is the page as a
    whole" - resolved via the page's own `text` field, not a sentence
    lookup (a naive lookup would always fail, since no page has a sentence
    at a negative index, misclassifying a real convention as missing data).
    """
    resolved_page_id = _resolve_page_id(conn, page_id)

    if sentence_id == -1:
        if resolved_page_id is None:
            return None, "missing_page"
        page_row = conn.execute("SELECT text FROM pages WHERE page_id = ?", (resolved_page_id,)).fetchone()
        page_text = page_row[0]
        if not page_text or not page_text.strip():
            return None, "empty_resolved_text"
        return page_text, ""

    if resolved_page_id is None:
        return None, "missing_page"
    row = conn.execute(
        "SELECT text FROM sentences WHERE page_id = ? AND sentence_id = ?", (resolved_page_id, sentence_id)
    ).fetchone()
    if row is None:
        return None, "missing_sentence_id"
    text = row[0]
    if not text or not text.strip():
        return None, "empty_resolved_text"
    return text, ""


def resolve_claim(conn: sqlite3.Connection, record: dict) -> dict:
    """Try each evidence set in order; use the first that resolves
    completely. Returns a full audit entry for this claim."""
    attempts = []
    for set_idx, evidence_set in enumerate(record["evidence"]):
        if not evidence_set:
            attempts.append({"set_index": set_idx, "status": "empty_set", "sentences": []})
            continue
        set_sentences = []
        set_ok = True
        for sent_ref in evidence_set:
            page_id = sent_ref["wiki_url"]
            sentence_id = sent_ref["sentence_id"]
            text, failure_reason = resolve_sentence(conn, page_id, sentence_id)
            set_sentences.append({
                "wiki_url": page_id, "sentence_id": sentence_id,
                "resolved": text is not None, "failure_reason": failure_reason,
            })
            if text is None:
                set_ok = False
        attempts.append({
            "set_index": set_idx,
            "status": "fully_resolved" if set_ok else "partial_or_failed",
            "sentences": set_sentences,
        })
        if set_ok:
            # Lean record for the (large-volume) resolved case - full text
            # is recomputed deterministically later when building the pool,
            # not duplicated here. Full per-sentence "attempts" detail is
            # kept only for the unresolved case below, where it's actually
            # needed for investigation.
            return {
                "example_id": record["example_id"],
                "label": record["label"],
                "status": "resolved",
                "resolved_evidence_set_index": set_idx,
                "num_sentences_in_resolved_set": len(evidence_set),
            }

    return {
        "example_id": record["example_id"],
        "label": record["label"],
        "status": "unresolved",
        "resolved_evidence_set_index": None,
        "attempts": attempts,
    }


def main() -> int:
    _section("FEVER EVIDENCE RESOLUTION AUDIT")
    if not INDEX_PATH.exists():
        print(f"Index not found at {INDEX_PATH} - run scripts/build_wiki_pages_index.py first.")
        return 1

    print("Loading FEVER train claims...")
    claims = fever_ds.load_normalized_split("train")
    resolvable = [c for c in claims if c["label"] in RESOLVABLE_LABELS]
    nei = [c for c in claims if c["label"] == "NOT ENOUGH INFO"]
    print(f"Total FEVER train claims: {len(claims):,}")
    print(f"SUPPORTS/REFUTES (attempting resolution): {len(resolvable):,}")
    print(f"NOT ENOUGH INFO (NOT attempted - no evidence annotation exists for these; "
          f"see report for why): {len(nei):,}")

    conn = sqlite3.connect(str(INDEX_PATH))

    results = []
    failure_reason_counts: Counter = Counter()
    resolved_via_set_index: Counter = Counter()
    for i, record in enumerate(resolvable):
        result = resolve_claim(conn, record)
        results.append(result)
        if result["status"] == "unresolved":
            # Record the failure reason(s) from the last attempted set for summary counting.
            last_attempt = result["attempts"][-1] if result["attempts"] else None
            if last_attempt and last_attempt["sentences"]:
                for s in last_attempt["sentences"]:
                    if s["failure_reason"]:
                        failure_reason_counts[s["failure_reason"]] += 1
            elif last_attempt and last_attempt["status"] == "empty_set":
                failure_reason_counts["empty_evidence_set"] += 1
        else:
            resolved_via_set_index[result["resolved_evidence_set_index"]] += 1
        if (i + 1) % 20000 == 0:
            print(f"  ...{i + 1:,}/{len(resolvable):,} claims processed")

    conn.close()

    n_resolved = sum(1 for r in results if r["status"] == "resolved")
    n_unresolved = len(results) - n_resolved
    resolved_label_dist = Counter(r["label"] for r in results if r["status"] == "resolved")
    unresolved_label_dist = Counter(r["label"] for r in results if r["status"] == "unresolved")

    _section("RESOLUTION RESULTS")
    print(f"Resolved: {n_resolved:,} / {len(resolvable):,} ({n_resolved / len(resolvable) * 100:.2f}%)")
    print(f"Unresolved: {n_unresolved:,} ({n_unresolved / len(resolvable) * 100:.2f}%)")
    print(f"Resolved label distribution: {dict(resolved_label_dist)}")
    print(f"Unresolved label distribution: {dict(unresolved_label_dist)}")
    print(f"Resolved via evidence-set index (0 = first/preferred set worked): {dict(resolved_via_set_index)}")
    print(f"Failure reasons among unresolved (from last attempted set): {dict(failure_reason_counts)}")

    # Investigate a small sample of unresolved claims directly, per instructions.
    unresolved_samples = [r for r in results if r["status"] == "unresolved"][:5]
    _section("INVESTIGATING A SAMPLE OF UNRESOLVED CLAIMS (not assumed to be an upstream issue)")
    for r in unresolved_samples:
        print(f"  example_id={r['example_id']} label={r['label']}")
        for attempt in r["attempts"][:2]:
            print(f"    set {attempt['set_index']}: {attempt['status']}")
            for s in attempt.get("sentences", [])[:3]:
                print(f"      {s}")

    report = {
        "method": "deterministic exact resolution via (page_id, sentence_id) primary-key lookup "
                  "against the wiki_pages SQLite index - no fuzzy matching. Evidence sets tried "
                  "in existing deterministic order (evidence[0] first); a claim resolves only "
                  "if one full set resolves completely (no cross-set mixing).",
        "index_path": str(INDEX_PATH),
        "not_enough_info_handling": {
            "count": len(nei),
            "attempted": False,
            "reason": "FEVER's NOT ENOUGH INFO claims carry no evidence annotation at all "
                      "(evidence_wiki_url == '' for every such row, confirmed in Step 6A and "
                      "reconfirmed here) - there is no reference to resolve. Fabricating a "
                      "premise for these would not reflect any real FEVER annotation. See "
                      "PROJECT_REPORT.md Step 10 for the full discussion.",
        },
        "summary": {
            "total_resolvable_claims": len(resolvable),
            "resolved": n_resolved,
            "unresolved": n_unresolved,
            "resolution_rate": n_resolved / len(resolvable) if resolvable else 0.0,
            "resolved_label_distribution": dict(resolved_label_dist),
            "unresolved_label_distribution": dict(unresolved_label_dist),
            "resolved_via_evidence_set_index": dict(resolved_via_set_index),
            "unresolved_failure_reasons": dict(failure_reason_counts),
        },
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    _section("SUMMARY")
    print(f"Report written to {REPORT_PATH} ({REPORT_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
