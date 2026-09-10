"""RAGTruth dataset loading and normalization utilities.

RAGTruth is ClaimGuard's PRIMARY end-to-end evaluation dataset (see
RESEARCH.md). This module is written accordingly: the evaluation boundary
(train vs. test) is made explicit and hard to bypass by accident, not just
documented in prose.

Loads the raw RAGTruth data acquired by scripts/inspect_ragtruth.py from
data/raw/ragtruth/{response,source_info}.jsonl and joins/normalizes it into
one record per response. This module does NOT implement model training and
does NOT load any model - see PROJECT_PLAN.md Step 13/17.

Real-data findings this module accounts for (full detail in
PROJECT_REPORT.md Step 6C):

  - Both files are JSON Lines, confirmed empirically (not assumed from the
    ".jsonl" extension, which happens to be accurate here unlike FEVER/
    HaluEval's misleading ".json").
  - response.jsonl: 17,790 records (15,090 train + 2,700 test), fields
    id/labels/model/quality/response/source_id/split/temperature, all
    present in every record, zero nulls, id is 100% unique.
  - source_info.jsonl: 2,965 records, fields
    prompt/source/source_id/source_info/task_type, all present in every
    record, source_id is 100% unique. `source_info` itself varies by
    task_type: a string (Summary), or a nested dict (Data2txt: business
    listing fields; QA: {"question", "passages"}).
  - Every response's source_id has exactly one matching source_info record
    (100% referential integrity, 0 orphans). Every source_info item has
    exactly 6 responses (one per model: gpt-4-0613, gpt-3.5-turbo-0613,
    mistral-7B-instruct, llama-2-7b-chat, llama-2-13b-chat, llama-2-70b-chat).
  - **Leakage check (the critical one for this dataset): confirmed 0
    source_info items have responses spanning more than one split.** Splits
    are cleanly source-disjoint - a source item's responses are entirely in
    train or entirely in test, never both. One exact response-TEXT match
    was found across splits, investigated, and confirmed benign: it is the
    generic stock refusal phrase "Unable to answer based on given
    passages.", produced verbatim by different models for different,
    legitimately-unrelated source items in both splits - not a content leak.
  - Label/span annotations were validated PROGRAMMATICALLY against the real
    response text: all 14,289 offset-bearing labels' response[start:end]
    slices exactly matched the label's own "text" field, with zero
    out-of-bounds or start>end offsets. Labels also carry `label_type`
    (Evident Conflict / Evident Baseless Info / Subtle Baseless Info /
    Subtle Conflict), `implicit_true` (bool), `meta`, and `due_to_null` -
    all preserved verbatim, not summarized away.
  - `quality` is "good" (17,617), "truncated" (29), or "incorrect_refusal"
    (144) - preserved verbatim; callers filtering for clean examples should
    consider excluding non-"good" quality, but this module does not do so
    by default (lossless).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import config as cg_config

RAW_DIR = cg_config.resolve_path("data/raw/ragtruth")

VALID_SPLITS = {"train", "test"}
KNOWN_TASK_TYPES = {"Summary", "Data2txt", "QA"}
KNOWN_QUALITY_VALUES = {"good", "truncated", "incorrect_refusal"}


def _load_jsonl(filename: str, raw_dir: Path) -> list[dict[str, Any]]:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Raw RAGTruth file not found: {path}. Run scripts/inspect_ragtruth.py first."
        )
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_raw_responses(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    return _load_jsonl("response.jsonl", raw_dir)


def load_raw_source_info(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    return _load_jsonl("source_info.jsonl", raw_dir)


def join_records(
    responses: list[dict[str, Any]], source_infos: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join response records to their source_info via source_id.

    Lossless: every field from both raw records is preserved (source_info
    fields are namespaced under `source_*` to avoid collision, response
    fields are kept at the top level as in the raw data). Adds two derived
    convenience fields: `has_hallucination` (bool) and `eval_reserved`
    (bool, True iff split == "test") - the latter is the explicit
    evaluation-boundary marker downstream code should check.
    """
    source_by_id = {s["source_id"]: s for s in source_infos}

    joined = []
    for r in responses:
        source_id = r.get("source_id")
        source = source_by_id.get(source_id)
        if source is None:
            # Preserve the orphan rather than dropping it - callers can see
            # source_missing=True and decide. (0 orphans found in the real
            # data, but this must not silently disappear if that changes.)
            record = {
                "response_id": r.get("id"),
                "source_id": source_id,
                "source_missing": True,
                "task_type": None,
                "source": None,
                "source_info": None,
                "prompt": None,
            }
        else:
            record = {
                "response_id": r.get("id"),
                "source_id": source_id,
                "source_missing": False,
                "task_type": source.get("task_type"),
                "source": source.get("source"),
                "source_info": source.get("source_info"),
                "prompt": source.get("prompt"),
            }

        labels = r.get("labels", [])
        record.update(
            split=r.get("split"),
            eval_reserved=(r.get("split") == "test"),
            model=r.get("model"),
            temperature=r.get("temperature"),
            quality=r.get("quality"),
            response_text=r.get("response"),
            labels=labels,
            has_hallucination=bool(labels),
        )
        joined.append(record)
    return joined


def load_normalized(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Load both raw files and return the joined, normalized records."""
    responses = load_raw_responses(raw_dir)
    source_infos = load_raw_source_info(raw_dir)
    return join_records(responses, source_infos)


def get_training_pool(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records from the `train` split only - safe for verifier development."""
    return [r for r in records if r.get("split") == "train"]


def get_eval_set(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records from the `test` split only - the reserved, primary end-to-end
    evaluation set. Do not use this for training/tuning anything.
    """
    return [r for r in records if r.get("eval_reserved") is True]


def find_duplicate_response_ids(records: list[dict[str, Any]]) -> dict[Any, int]:
    """Return {response_id: count} for any response_id appearing more than once.

    On the real data this returns {} - response.jsonl's `id` field was
    confirmed 100% unique (17,790 values, 17,790 unique) at inspection time.
    Kept as a reusable check rather than a one-off inspection-script finding.
    """
    counts: dict[Any, int] = {}
    for r in records:
        rid = r.get("response_id")
        counts[rid] = counts.get(rid, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def assert_no_train_test_leakage(records: list[dict[str, Any]]) -> None:
    """Raise AssertionError if any source_id's records span both splits.

    This is the programmatic form of the Step 6C leakage investigation -
    callers (e.g. before building a training batch) can invoke this as a
    safeguard rather than relying on the one-time inspection script finding
    staying true forever.
    """
    splits_by_source: dict[Any, set[str]] = {}
    for r in records:
        splits_by_source.setdefault(r.get("source_id"), set()).add(r.get("split"))
    leaking = {sid: splits for sid, splits in splits_by_source.items() if len(splits) > 1}
    if leaking:
        raise AssertionError(
            f"Train/test leakage detected: {len(leaking)} source_id(s) have records in "
            f"more than one split: {dict(list(leaking.items())[:5])}"
        )


def validate_offsets(record: dict[str, Any]) -> list[str]:
    """Re-validate each label's start/end offsets against this record's own
    response_text (defense in depth - the raw data was already validated at
    inspection time, but this makes the check reusable/testable per-record).
    """
    issues = []
    text = record.get("response_text")
    if not isinstance(text, str):
        return ["response_text_not_a_string"]
    for lbl in record.get("labels", []):
        if not isinstance(lbl, dict):
            issues.append("label_not_a_dict")
            continue
        start, end, label_text = lbl.get("start"), lbl.get("end"), lbl.get("text")
        if not isinstance(start, int) or not isinstance(end, int):
            issues.append("offset_not_int")
            continue
        if start < 0 or end > len(text) or start > end:
            issues.append(f"offset_out_of_bounds:[{start}:{end}]")
            continue
        if label_text is not None and text[start:end] != label_text:
            issues.append(f"offset_text_mismatch:[{start}:{end}]")
    return issues


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of integrity issue codes for a normalized record.

    Does not raise or discard anything - callers decide what to do with the
    issues found (never silently discard problematic examples).
    """
    issues: list[str] = []

    if not record.get("response_text"):
        issues.append("empty_or_missing_response_text")
    if record.get("split") not in VALID_SPLITS:
        issues.append(f"invalid_split:{record.get('split')!r}")
    if record.get("source_missing"):
        issues.append("source_missing")
    if record.get("task_type") is not None and record.get("task_type") not in KNOWN_TASK_TYPES:
        issues.append(f"unexpected_task_type:{record.get('task_type')!r}")
    if record.get("quality") is not None and record.get("quality") not in KNOWN_QUALITY_VALUES:
        issues.append(f"unexpected_quality:{record.get('quality')!r}")

    issues.extend(validate_offsets(record))
    return issues
