"""ClaimGuard cross-dataset integration: roles, label mapping, and the
leakage-safe verifier training/development pool.

Reconciles Steps 6A-6D's real dataset findings into the data protocol
already established in RESEARCH.md / CLAIMGUARD_MODEL_SELECTION.md
(Step 3/4): "FEVER + HaluEval can contribute to verifier training/
development. RAGTruth remains the primary end-to-end evaluation dataset.
TruthfulQA is evaluation-only." This module enforces that protocol in code,
not just prose.

## Roles (do not change silently - see PROJECT_REPORT.md Step 7)

- FEVER `train` (only): verifier training/development candidate.
  `validation`/`test` are excluded - Step 6A found them not clean
  (blank labels, casing variants, ~24% duplicate rows in `validation`,
  9,999-claim overlap between `validation` and `test`).
- HaluEval `qa`/`dialogue`/`summarization` subsets: verifier training/
  development candidates. HaluEval `general` is normalized and available,
  but EXCLUDED from this pool specifically - see GENERAL_EXCLUSION_REASON.
- RAGTruth `test`: HARD-BLOCKED from ever entering training - this module
  will raise RoleViolationError rather than silently permit it.
  RAGTruth `train` is NOT included in this pool - Step 6C confirmed it is
  safe to use later (source-disjoint from `test`), but the current protocol
  (Step 3/4) names only FEVER and HaluEval as training/development
  candidates, so RAGTruth `train`'s inclusion is deferred, not decided here.
- TruthfulQA: entirely evaluation-only, never enters training. It has no
  split of its own (Step 6D), so there is no leakage-safe subset of it that
  could be used for training even if the protocol wanted to.

## A gap this module surfaces rather than hides (updated Step 10)

FEVER's normalized evidence (`claimguard.datasets.fever`) used to be only a
list of (`wiki_url`, `sentence_id`) REFERENCES, not resolved sentence text -
Step 6A correctly did not download the 5.4M-page `wiki_pages` corpus (out of
scope for acquisition/inspection at that time). Step 10 acquired and indexed
that corpus and achieved 100% evidence resolution for FEVER train's
SUPPORTS/REFUTES claims (109,810/109,810 - see PROJECT_REPORT.md Step 10),
so `fever_train_pool_records()` below now yields PoolRecords with real,
resolved Wikipedia sentence premise text (`premise_text_available=True`)
for those claims, exactly like HaluEval's `qa`/`dialogue`/`summarization`
records.

This does NOT mean FEVER's premise gap is fully closed, though: FEVER's
NOT ENOUGH INFO claims (35,639 on train) carry no evidence annotation in
FEVER's own data at all - there is nothing to resolve, and inventing a
premise for them is out of scope by design (see
`claimguard.datasets.fever.RESOLUTION_STATUS_NO_EVIDENCE`). Those claims are
therefore EXCLUDED from `fever_train_pool_records()` - not silently coerced
into trainable records - which means FEVER still contributes ZERO neutral-
label examples to the verifier pool, and (since HaluEval also never
produces a neutral label) the pool's overall neutral-class coverage remains
effectively zero after this step. See `fever_resolution_coverage()` and
PROJECT_REPORT.md Step 10 for the full, honest accounting of this gap.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import fever as fever_ds
from . import halueval as halueval_ds
from . import ragtruth as ragtruth_ds
from . import truthfulqa as truthfulqa_ds

SEED = 42  # matches configs/default.yaml's seed
DEV_RATIO = 0.10

VERIFIER_LABELS = ("entailment", "neutral", "contradiction")

# HaluEval's `general` subset has NO premise/grounding field at all (Step 6B:
# context == {} for every general record) - it cannot be safely mapped to
# the (premise, hypothesis, label) NLI format this project's verifier
# consumes without fabricating a premise that isn't in the data. Left as a
# candidate for a future, separate correction-trigger/response-quality
# classifier instead of forced into this pool.
GENERAL_EXCLUSION_REASON = (
    "HaluEval 'general' records have no premise/grounding field (context is "
    "always {} - see Step 6B); cannot be safely represented as a "
    "(premise, hypothesis, label) NLI training pair without inventing a "
    "premise absent from the source data. Excluded from the verifier "
    "training pool; candidate for a future correction-trigger classifier "
    "that does not require a premise."
)

ALLOWED_TRAINING_SOURCES = {"fever_train", "halueval_qa", "halueval_dialogue", "halueval_summarization"}
BLOCKED_TRAINING_SOURCES = {
    "ragtruth_test", "ragtruth_train", "truthfulqa",
    "halueval_general", "fever_validation", "fever_test",
}


class RoleViolationError(RuntimeError):
    """Raised when code attempts to route an evaluation-reserved or otherwise
    disallowed dataset/split into the verifier training pool."""


@dataclass
class PoolRecord:
    hypothesis: str
    label: str  # one of VERIFIER_LABELS
    original_label: str  # the raw source label before mapping
    source_dataset: str  # "fever" | "halueval"
    source_subset: str  # "train" (fever) | "qa"/"dialogue"/"summarization" (halueval)
    source_id: Any  # provenance: fever example_id, or halueval example_id
    premise_text: str | None = None
    premise_reference: Any = None  # e.g. FEVER's evidence sets (wiki_url/sentence_id)
    premise_text_available: bool = False
    resolution_status: str | None = None  # e.g. fever.RESOLUTION_STATUS_RESOLVED; None where not applicable (HaluEval)
    group_key: tuple = field(default=())  # for leakage-safe dev-split grouping

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "label": self.label,
            "original_label": self.original_label,
            "source_dataset": self.source_dataset,
            "source_subset": self.source_subset,
            "source_id": self.source_id,
            "premise_text": self.premise_text,
            "premise_reference": self.premise_reference,
            "premise_text_available": self.premise_text_available,
            "resolution_status": self.resolution_status,
        }


def normalize_text(s: str) -> str:
    """Lowercase, strip, collapse internal whitespace - for EXACT (not
    fuzzy/near-duplicate) overlap checks only."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# --- Pool construction -------------------------------------------------

def fever_train_pool_records(
    raw_dir: Path | None = None, index_path: Path | None = None
) -> list[PoolRecord]:
    """Build FEVER PoolRecords using Step 10's resolved premise text.

    Only claims with premise_text_available=True (resolution_status ==
    fever.RESOLUTION_STATUS_RESOLVED - a SUPPORTS/REFUTES claim whose
    evidence was found, in full, in the wiki_pages index) become trainable
    PoolRecords here: a PoolRecord without premise text cannot represent a
    real (premise, hypothesis, label) NLI example, so including one would
    require either inventing a premise (forbidden) or silently changing the
    task to hypothesis-only classification.

    NOT ENOUGH INFO claims (no evidence annotation exists in FEVER's own
    data - resolution_status == fever.RESOLUTION_STATUS_NO_EVIDENCE) and any
    genuinely unresolved SUPPORTS/REFUTES claims are therefore EXCLUDED
    here, not silently coerced. This is not hidden: see
    fever_resolution_coverage() for the full accounting of what was
    excluded, why, and what that means for the pool's label coverage.
    """
    kwargs: dict[str, Any] = {}
    if raw_dir:
        kwargs["raw_dir"] = raw_dir
    if index_path:
        kwargs["index_path"] = index_path
    claims = fever_ds.load_resolved_split("train", **kwargs)
    out = []
    for c in claims:
        if c["verifier_label"] is None:
            continue  # unmapped label - excluded, not coerced (none expected on train)
        if not c["premise_text_available"]:
            continue  # no resolved premise text - excluded, not coerced; see docstring above
        out.append(PoolRecord(
            hypothesis=c["claim"],
            label=c["verifier_label"],
            original_label=c["label"],
            source_dataset="fever",
            source_subset="train",
            source_id=c["example_id"],
            premise_text=c["resolved_premise"],
            premise_reference=c["evidence"],
            premise_text_available=True,
            resolution_status=c["resolution_status"],
            group_key=("fever", "train", c["example_id"]),
        ))
    return out


def fever_resolution_coverage(
    raw_dir: Path | None = None, index_path: Path | None = None
) -> dict[str, Any]:
    """Honest, real-data accounting of FEVER train claims by
    resolution_status and label - including the claims EXCLUDED from
    fever_train_pool_records() because they have no usable premise text.

    This directly answers "does the verifier pool now have neutral-class
    coverage from FEVER": as of Step 10 it does not, because NOT ENOUGH
    INFO claims have no evidence annotation at all (a structural property
    of FEVER's own data, not a resolver bug or acquisition gap - see
    PROJECT_REPORT.md Step 10).
    """
    kwargs: dict[str, Any] = {}
    if raw_dir:
        kwargs["raw_dir"] = raw_dir
    if index_path:
        kwargs["index_path"] = index_path
    claims = fever_ds.load_resolved_split("train", **kwargs)

    status_counts: Counter = Counter(c["resolution_status"] for c in claims)
    status_by_label: dict[str, Counter] = {}
    for c in claims:
        status_by_label.setdefault(c["label"], Counter())[c["resolution_status"]] += 1

    usable = status_counts.get(fever_ds.RESOLUTION_STATUS_RESOLVED, 0)
    neutral_usable = status_by_label.get("NOT ENOUGH INFO", Counter()).get(
        fever_ds.RESOLUTION_STATUS_RESOLVED, 0
    )
    return {
        "total_claims": len(claims),
        "status_counts": dict(status_counts),
        "status_by_label": {k: dict(v) for k, v in status_by_label.items()},
        "usable_for_training": usable,
        "neutral_class_available_from_fever": neutral_usable,
    }


_HALUEVAL_PAIR_LABEL_MAP = {"not_hallucinated": "entailment", "hallucinated": "contradiction"}


def halueval_pair_pool_records(subset: str, raw_dir: Path | None = None) -> list[PoolRecord]:
    if subset not in halueval_ds.PAIR_SUBSETS:
        raise ValueError(f"{subset!r} is not a HaluEval pair subset "
                          f"({sorted(halueval_ds.PAIR_SUBSETS)}); did you mean 'general'? "
                          f"'general' is intentionally excluded - see GENERAL_EXCLUSION_REASON.")
    kwargs = {"raw_dir": raw_dir} if raw_dir else {}
    records = halueval_ds.load_normalized_subset(subset, **kwargs)
    out = []
    for r in records:
        mapped = _HALUEVAL_PAIR_LABEL_MAP.get(r["label"])
        if mapped is None:
            continue  # should not occur - halueval pair labels are always one of the two
        premise_text = " ".join(v for v in r["context"].values() if v)
        out.append(PoolRecord(
            hypothesis=r["candidate_response"],
            label=mapped,
            original_label=r["label"],
            source_dataset="halueval",
            source_subset=subset,
            source_id=r["example_id"],
            premise_text=premise_text,
            premise_reference=None,
            premise_text_available=True,
            group_key=("halueval", subset, r["raw_index"]),
        ))
    return out


_SOURCE_BUILDERS = {
    "fever_train": fever_train_pool_records,
    "halueval_qa": lambda raw_dir=None: halueval_pair_pool_records("qa", raw_dir),
    "halueval_dialogue": lambda raw_dir=None: halueval_pair_pool_records("dialogue", raw_dir),
    "halueval_summarization": lambda raw_dir=None: halueval_pair_pool_records("summarization", raw_dir),
}


def build_training_pool(
    requested_sources: Iterable[str] = ALLOWED_TRAINING_SOURCES,
    exclude_contamination: bool = True,
) -> list[PoolRecord]:
    """Build the verifier training/development pool from the given sources.

    Raises RoleViolationError immediately (fails loudly) if any requested
    source is not in ALLOWED_TRAINING_SOURCES - this is the enforcement
    point for "RAGTruth test/TruthfulQA/HaluEval general can never enter
    training".

    By default (Step 8), also excludes any record whose group_key matches
    a known cross-dataset contamination case (currently: the 100
    HaluEval-summarization/RAGTruth-test-Summary document overlaps - see
    contamination_exclusion_group_keys()). Pass exclude_contamination=False
    only for auditing/before-vs-after comparisons, never for building an
    actual training pool.
    """
    requested = list(requested_sources)
    disallowed = set(requested) - ALLOWED_TRAINING_SOURCES
    if disallowed:
        raise RoleViolationError(
            f"Requested source(s) are not permitted in the verifier training pool: "
            f"{sorted(disallowed)}. Allowed sources: {sorted(ALLOWED_TRAINING_SOURCES)}."
        )
    pool: list[PoolRecord] = []
    for source in requested:
        pool.extend(_SOURCE_BUILDERS[source]())

    if exclude_contamination:
        exclude_keys = contamination_exclusion_group_keys()
        pool = [r for r in pool if r.group_key not in exclude_keys]

    return pool


# --- Deterministic train/dev split --------------------------------------

def split_train_dev(
    pool: list[PoolRecord], seed: int = SEED, dev_ratio: float = DEV_RATIO
) -> tuple[list[PoolRecord], list[PoolRecord]]:
    """Deterministically split the pool into train/dev.

    Splits by `group_key`, not by individual record, so that HaluEval's two
    pair-expanded records (not_hallucinated + hallucinated, sharing one
    underlying raw example) always land on the same side - never leaking
    one half of a contrastive pair into dev while its partner is in train.
    FEVER's group_key is already 1:1 with its records (one claim = one
    group), so this has no effect there beyond a stable split.
    """
    group_keys = sorted({r.group_key for r in pool}, key=str)
    rng = random.Random(seed)
    rng.shuffle(group_keys)
    n_dev_groups = max(1, int(len(group_keys) * dev_ratio))
    dev_group_set = set(group_keys[:n_dev_groups])

    train, dev = [], []
    for r in pool:
        (dev if r.group_key in dev_group_set else train).append(r)
    return train, dev


# --- Cross-dataset leakage checks ---------------------------------------

def ragtruth_eval_texts(raw_dir: Path | None = None) -> dict[str, set[str]]:
    """Normalized text sets from RAGTruth's reserved `test` split ONLY -
    uses ragtruth.get_eval_set(), the same accessor the eval-boundary
    safeguard in Step 6C relies on, so this never accidentally reads train.
    """
    kwargs = {"raw_dir": raw_dir} if raw_dir else {}
    all_records = ragtruth_ds.load_normalized(**kwargs)
    test_records = ragtruth_ds.get_eval_set(all_records)
    responses = {normalize_text(r["response_text"]) for r in test_records if r.get("response_text")}
    source_texts = set()
    for r in test_records:
        si = r.get("source_info")
        if isinstance(si, str):
            source_texts.add(normalize_text(si))
        elif isinstance(si, dict):
            for v in si.values():
                if isinstance(v, str):
                    source_texts.add(normalize_text(v))
    return {"response_text": responses, "source_info_text": source_texts}


def truthfulqa_eval_texts(raw_dir: Path | None = None) -> dict[str, set[str]]:
    kwargs = {"raw_dir": raw_dir} if raw_dir else {}
    records = truthfulqa_ds.load_normalized(**kwargs)
    questions = {normalize_text(r["question"]) for r in records if r.get("question")}
    answers = set()
    for r in records:
        for a in [r.get("best_answer"), r.get("best_incorrect_answer")] + \
                 r.get("correct_answers", []) + r.get("incorrect_answers", []):
            if a:
                answers.add(normalize_text(a))
    return {"question": questions, "answer": answers}


def check_exact_overlap(pool: list[PoolRecord], eval_texts: dict[str, set[str]]) -> dict[str, Any]:
    """Exact (normalized) text overlap between the pool's hypothesis/premise
    texts and a given evaluation dataset's text sets. EXACT matching only -
    this does not attempt near-duplicate/fuzzy detection.
    """
    pool_hypotheses = {normalize_text(r.hypothesis) for r in pool if r.hypothesis}
    pool_premises = {normalize_text(r.premise_text) for r in pool if r.premise_text}

    result: dict[str, Any] = {}
    for eval_field, eval_set in eval_texts.items():
        hyp_overlap = pool_hypotheses & eval_set
        prem_overlap = pool_premises & eval_set
        result[eval_field] = {
            "hypothesis_overlap_count": len(hyp_overlap),
            "premise_overlap_count": len(prem_overlap),
        }
    return result


# --- Step 8: contamination detection, source-level identification, ------
# --- and exclusion mechanism ---------------------------------------------
#
# Step 7's check_exact_overlap()/ragtruth_eval_texts() found 100 exact text
# matches between the training pool's premise text and RAGTruth test's
# `source_info_text` set - but that set pools ALL RAGTruth task types'
# source_info text together, so it does not by itself identify which
# RAGTruth records are implicated, nor whether 100 matching document
# strings implies exactly 100 affected records on each side. The functions
# below answer both questions precisely, from the real data, every time
# they are called (nothing here is a hard-coded list of row numbers).

EXCLUSION_REASON_RAGTRUTH_OVERLAP = "ragtruth_test_source_overlap"


def find_ragtruth_summarization_contamination(
    halueval_raw_dir: Path | None = None, ragtruth_raw_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Precisely identify every HaluEval `summarization` raw record whose
    `document` text exactly matches a RAGTruth `test`-split, task_type=
    "Summary" `source_info` text, with full provenance on both sides.

    Deliberately narrower/stricter than Step 7's aggregate check: this
    filters RAGTruth's source_info comparison set to task_type == "Summary"
    specifically (Step 7's set pooled every task type together), so every
    match returned here is confirmed - not assumed - to come from the
    CNN/DailyMail-sourced Summary task. Does not assume a 1:1 mapping
    between matching text and record: both sides are grouped by normalized
    text first, so a document that appears more than once on either side
    produces multiple audit entries, not one.
    """
    halueval_kwargs = {"raw_dir": halueval_raw_dir} if halueval_raw_dir else {}
    ragtruth_kwargs = {"raw_dir": ragtruth_raw_dir} if ragtruth_raw_dir else {}

    halueval_raw = halueval_ds.load_raw_jsonl("summarization_data.json", **halueval_kwargs)
    ragtruth_all = ragtruth_ds.load_normalized(**ragtruth_kwargs)
    ragtruth_test = ragtruth_ds.get_eval_set(ragtruth_all)

    halueval_by_text: dict[str, list[int]] = {}
    for i, r in enumerate(halueval_raw):
        doc = r.get("document")
        if doc:
            halueval_by_text.setdefault(normalize_text(doc), []).append(i)

    ragtruth_by_text: dict[str, list[dict[str, Any]]] = {}
    for r in ragtruth_test:
        if r.get("task_type") != "Summary":
            continue
        si = r.get("source_info")
        if not isinstance(si, str) or not si:
            continue
        ragtruth_by_text.setdefault(normalize_text(si), []).append(
            {"source_id": r["source_id"], "response_id": r["response_id"]}
        )

    shared_texts = set(halueval_by_text) & set(ragtruth_by_text)

    audit: list[dict[str, Any]] = []
    for text in sorted(shared_texts):
        for halueval_idx in halueval_by_text[text]:
            for rt_entry in ragtruth_by_text[text]:
                audit.append({
                    "halueval_raw_index": halueval_idx,
                    "halueval_subset": "summarization",
                    "ragtruth_source_id": rt_entry["source_id"],
                    "ragtruth_test_response_id": rt_entry["response_id"],
                    "matching_field": "halueval.document == ragtruth.source_info (task_type=Summary)",
                    "comparison_method": "exact_normalized_text_match",
                    "text_preview": text[:200],
                })
    return audit


def contaminated_halueval_summarization_raw_indices(**kwargs: Any) -> set[int]:
    """Deterministically COMPUTED (not hard-coded) set of HaluEval
    summarization raw_index values to exclude from the training pool."""
    audit = find_ragtruth_summarization_contamination(**kwargs)
    return {a["halueval_raw_index"] for a in audit}


def contamination_exclusion_group_keys(**kwargs: Any) -> set[tuple]:
    """The PoolRecord.group_key values that build_training_pool() removes
    by default. Exposed separately so tests/scripts can verify exclusion
    membership without re-deriving it independently."""
    indices = contaminated_halueval_summarization_raw_indices(**kwargs)
    return {("halueval", "summarization", i) for i in indices}


def verify_contamination_resolved(pool: list[PoolRecord]) -> dict[str, Any]:
    """Re-run the SAME (broader, all-task-type) overlap check Step 7 used
    to originally discover the problem, directly against an
    already-filtered pool - proving the exclusion actually removed the
    contaminated premise text, using the same methodology that found it,
    not a narrower one that could hide a partial fix."""
    rt_texts = ragtruth_eval_texts()
    pool_premises = {normalize_text(r.premise_text) for r in pool if r.premise_text}
    remaining = pool_premises & rt_texts["source_info_text"]
    return {"remaining_premise_overlap_with_ragtruth_source_info_text": len(remaining)}


# --- Step 8: systematic cross-dataset contamination sweep ---------------

ALL_HALUEVAL_SUBSET_SOURCES = ("halueval_qa", "halueval_dialogue", "halueval_general", "halueval_summarization")


def dataset_text_sets(source: str, raw_dir: Path | None = None) -> dict[str, set[str]]:
    """Normalized {'premise': set, 'hypothesis': set} text for a candidate
    source, for the systematic cross-dataset sweep below. Field-aware: uses
    each dataset's own natural grounding/candidate fields, not a generic
    stringification.
    """
    if source == "fever_train":
        kwargs = {"raw_dir": raw_dir} if raw_dir else {}
        claims = fever_ds.load_resolved_split("train", **kwargs)
        premises = {
            normalize_text(c["resolved_premise"])
            for c in claims if c.get("premise_text_available") and c.get("resolved_premise")
        }
        return {
            "premise": premises,
            "hypothesis": {normalize_text(c["claim"]) for c in claims if c.get("claim")},
        }
    if source.startswith("halueval_"):
        subset = source.split("_", 1)[1]
        kwargs = {"raw_dir": raw_dir} if raw_dir else {}
        recs = halueval_ds.load_normalized_subset(subset, **kwargs)
        premises = {
            normalize_text(" ".join(v for v in r["context"].values() if v))
            for r in recs if r.get("context")
        }
        premises.discard("")
        hyps = {normalize_text(r["candidate_response"]) for r in recs if r.get("candidate_response")}
        return {"premise": premises, "hypothesis": hyps}
    raise ValueError(f"Unknown source for dataset_text_sets: {source!r}")


def cross_dataset_contamination_sweep() -> dict[str, Any]:
    """Systematic exact-text-match sweep: FEVER train and EVERY HaluEval
    subset (including 'general', which is not in the training pool -
    included here for audit completeness per Step 8) vs RAGTruth test and
    TruthfulQA. EXACT match only - explicitly not a semantic or
    near-duplicate check, and not claimed to be one.
    """
    rt_texts = ragtruth_eval_texts()
    tqa_texts = truthfulqa_eval_texts()

    report: dict[str, Any] = {}
    for source in ("fever_train",) + ALL_HALUEVAL_SUBSET_SOURCES:
        texts = dataset_text_sets(source)
        vs_rt = {
            field: {
                "premise_overlap": len(texts["premise"] & eval_set),
                "hypothesis_overlap": len(texts["hypothesis"] & eval_set),
            }
            for field, eval_set in rt_texts.items()
        }
        vs_tqa = {
            field: {
                "premise_overlap": len(texts["premise"] & eval_set),
                "hypothesis_overlap": len(texts["hypothesis"] & eval_set),
            }
            for field, eval_set in tqa_texts.items()
        }
        report[source] = {"vs_ragtruth_test": vs_rt, "vs_truthfulqa": vs_tqa}
    return report


# --- Step 8: FEVER premise-resolution status (determined, not invented) --

def fever_premise_resolution_status(
    raw_dir: Path | None = None, index_path: Path | None = None
) -> dict[str, Any]:
    """Determine whether literal premise (evidence sentence) text is
    currently available for FEVER pool records, using the real wiki_pages
    resolution index and audit built in Step 10. Does NOT fetch, invent, or
    otherwise fabricate missing premise text - read-only determination
    against the actual index/data every time it is called.
    """
    idx_path = index_path or fever_ds.WIKI_INDEX_PATH
    index_present = idx_path.exists()
    if not index_present:
        return {
            "premise_text_currently_available": False,
            "wiki_pages_index_present": False,
            "conclusion": (
                f"wiki_pages resolution index not found at {idx_path}. Run "
                "scripts/acquire_inspect_wiki_pages.py then "
                "scripts/build_wiki_pages_index.py first (see PROJECT_REPORT.md Step 10)."
            ),
        }
    coverage = fever_resolution_coverage(raw_dir=raw_dir, index_path=index_path)
    return {
        "premise_text_currently_available": True,
        "wiki_pages_index_present": True,
        "resolution_coverage": coverage,
        "conclusion": (
            f"{coverage['usable_for_training']:,} of {coverage['total_claims']:,} FEVER train "
            "claims (SUPPORTS/REFUTES with fully-resolved evidence) now have real, resolved "
            "Wikipedia sentence premise text (Step 10) - up from 0 in Step 8/9. NOT ENOUGH INFO "
            "claims remain structurally unresolvable (no evidence annotation exists in FEVER's "
            "own data at all) - see PROJECT_REPORT.md Step 10. The verifier pool's neutral-class "
            f"contribution from FEVER therefore remains "
            f"{coverage['neutral_class_available_from_fever']} claims."
        ),
    }


# --- Step 11: neutral-class investigation and the resulting classification-
# --- mode determination ---------------------------------------------------
#
# Step 10 closed FEVER's SUPPORTS/REFUTES premise-text gap but left the
# 'neutral' class at 0 (FEVER NOT ENOUGH INFO claims carry no evidence
# annotation in FEVER's own data at all; HaluEval's binary mapping never
# produces 'neutral' either). Step 11 investigated every already-acquired
# dataset for a legitimate, non-fabricated neutral source (see
# PROJECT_REPORT.md Step 11 for the full investigation) and found none:
#   - FEVER train: verified empirically that ALL 47,609 NOT ENOUGH INFO
#     rows carry the pure sentinel (evidence_wiki_url="", evidence_id=-1,
#     evidence_sentence_id=-1) with zero exceptions - confirmed against the
#     authoritative FEVER documentation (fever.ai), which itself states NEI
#     claims are recorded with null evidence and that no officially
#     distributed predicted-evidence/retrieval-baseline file exists.
#   - FEVER validation DOES contain a small number (151/13,883) of NOT
#     ENOUGH INFO rows with a non-sentinel evidence_wiki_url - investigated
#     and found to be a data-quality artifact (e.g. the same wiki_url
#     'Starrcade' attached to several unrelated claims, evidence_annotation_id
#     == -1), consistent with validation's already-documented (Step 6A)
#     blank-label/duplicate-row/casing-inconsistency problems - NOT a
#     legitimate evidence source, and validation is out of the training
#     pool for independent, already-established reasons regardless.
#   - HaluEval: every subset is strictly binary (hallucinated/
#     not_hallucinated); no subset encodes an "insufficient evidence"
#     category - confirmed by reading claimguard.datasets.halueval's
#     documented real-data schema (Step 6B), not reinterpreted here.
#   - RAGTruth and TruthfulQA are both hard-blocked, evaluation-only
#     datasets under the Step 3/4 protocol - even if either had a
#     neutral-shaped category, using it for training would violate the
#     evaluation boundary this project depends on, so neither is a
#     candidate regardless of label content.
# Conclusion: no legitimate, premise-grounded, leakage-safe neutral source
# currently exists in this project's acquired data. The verifier training
# pool is therefore, correctly, a BINARY (entailment/contradiction) dataset
# - not a 3-way dataset with a currently-empty third class.

INTENDED_VERIFIER_LABEL_SPACE = ("entailment", "neutral", "contradiction")

CLASSIFICATION_MODE_THREE_WAY = "three_way"
CLASSIFICATION_MODE_BINARY = "binary"
CLASSIFICATION_MODE_INVALID = "invalid"


def verifier_dataset_label_space_status(pool: list[PoolRecord]) -> dict[str, Any]:
    """Determine the CURRENT, real classification mode of a verifier pool
    from its actual label counts - never from a stored claim or assumption.

    Returns a dict with `intended_label_space` (the eventual 3-way target,
    unchanged from the project's original design), `available_label_space`
    (labels with count > 0 in `pool`, computed fresh), `class_counts`,
    `missing_classes`, `classification_mode` (one of
    CLASSIFICATION_MODE_THREE_WAY / _BINARY / _INVALID), and a `rationale`
    string. `classification_mode` is only ever CLASSIFICATION_MODE_THREE_WAY
    if EVERY intended label has count > 0 - a pool with only two of three
    populated is BINARY, not "mostly three-way", so nothing downstream can
    round up to a claim this function does not actually support.
    """
    class_counts: Counter = Counter(r.label for r in pool)
    available = sorted(label for label in INTENDED_VERIFIER_LABEL_SPACE if class_counts.get(label, 0) > 0)
    missing = sorted(label for label in INTENDED_VERIFIER_LABEL_SPACE if class_counts.get(label, 0) == 0)
    unexpected = sorted(set(class_counts) - set(INTENDED_VERIFIER_LABEL_SPACE))

    if unexpected:
        mode = CLASSIFICATION_MODE_INVALID
        rationale = (
            f"Pool contains label(s) outside the intended verifier label space: {unexpected}. "
            "This is a data-integrity problem, not a valid classification mode."
        )
    elif not missing:
        mode = CLASSIFICATION_MODE_THREE_WAY
        rationale = (
            "All three intended labels (entailment, neutral, contradiction) have at least one "
            "example - the pool genuinely supports 3-way verifier training/evaluation."
        )
    elif set(available) == {"entailment", "contradiction"}:
        mode = CLASSIFICATION_MODE_BINARY
        rationale = (
            "Only entailment and contradiction have examples; 'neutral' has 0. Per Step 11's "
            "investigation (see PROJECT_REPORT.md), no legitimate, non-fabricated neutral "
            "source currently exists in this project's acquired data (FEVER NOT ENOUGH INFO "
            "claims carry no evidence annotation at all; HaluEval is strictly binary; "
            "RAGTruth/TruthfulQA are evaluation-only and hard-blocked from training regardless "
            "of label content). The pool is therefore a BINARY verifier dataset, not a 3-way "
            "dataset with a temporarily-empty third class - callers must not claim 3-way "
            "training/evaluation support from this pool."
        )
    else:
        mode = CLASSIFICATION_MODE_INVALID
        rationale = (
            f"Unexpected label coverage pattern: available={available}, missing={missing}. "
            "Neither a valid 3-way pool nor the expected entailment/contradiction-only binary "
            "pool - investigate before trusting this pool for training."
        )

    return {
        "intended_label_space": list(INTENDED_VERIFIER_LABEL_SPACE),
        "available_label_space": available,
        "class_counts": dict(class_counts),
        "missing_classes": missing,
        "classification_mode": mode,
        "supports_three_way": mode == CLASSIFICATION_MODE_THREE_WAY,
        "rationale": rationale,
    }


NEUTRAL_INVESTIGATION_SUMMARY = {
    "fever_train_not_enough_info": {
        "count": 35639,
        "finding": (
            "Verified empirically (Step 11): ALL 47,609 NOT ENOUGH INFO rows in the raw "
            "(denormalized) FEVER train file carry the pure sentinel "
            "(evidence_wiki_url='', evidence_id=-1, evidence_sentence_id=-1) with ZERO "
            "exceptions - grouped to 35,639 unique claims. Cross-checked against the "
            "authoritative FEVER documentation (fever.ai), which confirms NOT ENOUGH INFO "
            "claims are recorded with null evidence in the official release and that no "
            "officially-distributed predicted-evidence/retrieval-baseline artifact exists."
        ),
        "conclusion": "no_legitimate_premise_available",
    },
    "fever_validation_not_enough_info_anomaly": {
        "finding": (
            "FEVER validation (NOT a training-eligible split - excluded since Step 6A/7 for "
            "independent data-quality reasons) contains 151/13,883 NOT ENOUGH INFO rows with a "
            "non-sentinel evidence_wiki_url. Investigated: the same wiki_url ('Starrcade') "
            "appears attached to multiple unrelated claims with evidence_annotation_id=-1 - "
            "consistent with a parquet-export/indexing artifact, not genuine annotator-recorded "
            "evidence. Not used as a neutral source: both because it does not appear to be "
            "legitimate evidence, and because validation is already excluded from training for "
            "reasons unrelated to this step."
        ),
        "conclusion": "not_a_legitimate_source_and_out_of_scope",
    },
    "halueval_all_subsets": {
        "finding": (
            "Every HaluEval subset (qa/dialogue/summarization/general) is strictly binary "
            "(hallucinated/not_hallucinated - see claimguard.datasets.halueval.VALID_LABELS). "
            "No subset encodes an 'insufficient evidence'/'unknown' third category anywhere in "
            "the real data."
        ),
        "conclusion": "no_legitimate_source",
    },
    "ragtruth_and_truthfulqa": {
        "finding": (
            "Both are hard-blocked, evaluation-only datasets under the Step 3/4 research "
            "protocol (RoleViolationError on any training-pool inclusion attempt). Even if "
            "either contained neutral-shaped content, using it for training would violate the "
            "evaluation boundary this project's leakage-safety design depends on - so neither "
            "is a candidate regardless of label content."
        ),
        "conclusion": "protected_evaluation_only_not_a_candidate",
    },
    "overall_conclusion": (
        "The current authoritative datasets do not provide a leakage-safe, premise-grounded "
        "neutral training class. The verifier training pool is a binary (entailment/"
        "contradiction) dataset. See PROJECT_REPORT.md Step 11 for the full investigation."
    ),
}
