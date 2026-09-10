"""HaluEval dataset loading and normalization utilities.

Loads the raw HaluEval data acquired by scripts/inspect_halueval.py from
data/raw/halueval/*.json (actually JSON Lines, despite the extension - see
below) and normalizes it for ClaimGuard's verifier/correction-trigger
training and evaluation. This module does NOT implement model training and
does NOT load any model - see PROJECT_PLAN.md Step 11/12.

Real-data findings this module accounts for (full detail in
PROJECT_REPORT.md Step 6B):

  - All four files are JSON LINES (one JSON object per line), not a single
    JSON array, despite the ".json" extension. Confirmed empirically, not
    assumed from the filename.

  - HaluEval's four subsets have TWO materially different shapes, and this
    module deliberately does not force them into one schema:

    Group A - qa / dialogue / summarization ("contrastive pair" subsets):
      Each raw record is a grounding context plus a NON-hallucinated
      reference and a hallucinated counterpart - there is no direct
      per-example binary label in the raw data. Field names differ per
      subset (qa: knowledge/question/right_answer/hallucinated_answer;
      dialogue: knowledge/dialogue_history/right_response/
      hallucinated_response; summarization: document/right_summary/
      hallucinated_summary - note dialogue has TWO context fields, qa has
      one, summarization uses a different field name ("document") than
      qa/dialogue's "knowledge"). No id field exists in any of these three
      files. To get binary-labeled examples for verifier/correction-trigger
      training, `expand_pair_record()` below EXPANDS each raw record into
      two normalized records (one hallucinated, one not) - this is a
      normalization CHOICE, not something present in the raw data, and is
      documented as such rather than silently assumed.

    Group B - general (real single-response subset):
      Each record is one real ChatGPT response to a user query, with a
      genuine human-annotated binary label (`hallucination`: "yes"/"no",
      lowercase in the real data) and, for "yes" cases, span-level
      `hallucination_spans` (a list of the specific hallucinated substrings)
      - richer than a plain label. There is NO reference/"right" response in
      this subset (unlike Group A) - it evaluates a real response as-is.
      It has an `ID` field, but this field is NOT reliably unique or clean:
      real count is 4,507 records / 4,506 unique ID values - two records
      literally have ID == "ID" (the column-header string leaked in as a
      data value) and a separate record has ID == "" (empty). `raw_id`
      below preserves this field for traceability only; it must not be used
      as a uniqueness key.

  - Across all four files, no native id is trustworthy as a uniqueness key,
    so every normalized record carries `raw_index` (0-based position in the
    source file), which is always present and always unique per (subset,
    raw_index) pair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import config as cg_config

RAW_DIR = cg_config.resolve_path("data/raw/halueval")

# Group A: (filename, context_fields, right_field, hallucinated_field, query_field)
PAIR_SUBSETS: dict[str, dict[str, Any]] = {
    "qa": {
        "file": "qa_data.json",
        "context_fields": ["knowledge"],
        "query_field": "question",
        "right_field": "right_answer",
        "hallucinated_field": "hallucinated_answer",
    },
    "dialogue": {
        "file": "dialogue_data.json",
        "context_fields": ["knowledge", "dialogue_history"],
        "query_field": None,
        "right_field": "right_response",
        "hallucinated_field": "hallucinated_response",
    },
    "summarization": {
        "file": "summarization_data.json",
        "context_fields": ["document"],
        "query_field": None,
        "right_field": "right_summary",
        "hallucinated_field": "hallucinated_summary",
    },
}

GENERAL_SUBSET_FILE = "general_data.json"

VALID_LABELS = {"hallucinated", "not_hallucinated"}
# Values seen in the real general_data.json `ID` field that are known data
# artifacts, not genuine identifiers - flagged by validate_record(), not
# silently corrected.
SUSPICIOUS_RAW_IDS = {"", "ID"}


def load_raw_jsonl(filename: str, raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Load one HaluEval file. Real format is JSON Lines, not a JSON array."""
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Raw HaluEval file not found: {path}. Run scripts/inspect_halueval.py first."
        )
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def expand_pair_record(raw: dict[str, Any], subset: str, raw_index: int) -> list[dict[str, Any]]:
    """Expand one Group A (qa/dialogue/summarization) raw record into two
    normalized, binary-labeled records: one for the reference ("right")
    response and one for the hallucinated counterpart.
    """
    spec = PAIR_SUBSETS[subset]
    context = {field: raw.get(field, "") for field in spec["context_fields"]}
    query = raw.get(spec["query_field"]) if spec["query_field"] else None

    base = {
        "subset": subset,
        "context": context,
        "query": query,
        "hallucination_spans": None,  # Group A has no span-level annotation
        "raw_id": None,  # Group A files have no id field at all
        "raw_index": raw_index,
    }

    not_hallucinated = dict(base)
    not_hallucinated.update(
        example_id=f"{subset}_{raw_index}_not_hallucinated",
        candidate_response=raw.get(spec["right_field"], ""),
        label="not_hallucinated",
        source_field=spec["right_field"],
    )

    hallucinated = dict(base)
    hallucinated.update(
        example_id=f"{subset}_{raw_index}_hallucinated",
        candidate_response=raw.get(spec["hallucinated_field"], ""),
        label="hallucinated",
        source_field=spec["hallucinated_field"],
    )

    return [not_hallucinated, hallucinated]


def normalize_general_record(raw: dict[str, Any], raw_index: int) -> dict[str, Any]:
    """Normalize one Group B (general) raw record - already single-response,
    already labeled; no expansion needed.
    """
    label_raw = (raw.get("hallucination") or "").strip().lower()
    label = "hallucinated" if label_raw == "yes" else "not_hallucinated" if label_raw == "no" else None
    return {
        "example_id": f"general_{raw_index}",
        "subset": "general",
        "context": {},  # no grounding document/knowledge field in this subset
        "query": raw.get("user_query"),
        "candidate_response": raw.get("chatgpt_response", ""),
        "label": label,
        "hallucination_spans": raw.get("hallucination_spans"),
        "source_field": "chatgpt_response",
        "raw_id": raw.get("ID"),
        "raw_index": raw_index,
    }


def load_normalized_subset(subset: str, raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Load and normalize one subset by name ('qa', 'dialogue',
    'summarization', or 'general').
    """
    if subset == "general":
        raw_records = load_raw_jsonl(GENERAL_SUBSET_FILE, raw_dir=raw_dir)
        return [normalize_general_record(r, i) for i, r in enumerate(raw_records)]
    if subset in PAIR_SUBSETS:
        raw_records = load_raw_jsonl(PAIR_SUBSETS[subset]["file"], raw_dir=raw_dir)
        out: list[dict[str, Any]] = []
        for i, r in enumerate(raw_records):
            out.extend(expand_pair_record(r, subset, i))
        return out
    raise ValueError(f"Unknown HaluEval subset: {subset!r}. Expected one of "
                      f"{sorted(list(PAIR_SUBSETS.keys()) + ['general'])}.")


def load_all_normalized(raw_dir: Path = RAW_DIR) -> dict[str, list[dict[str, Any]]]:
    """Load and normalize every subset, keyed by subset name."""
    return {
        subset: load_normalized_subset(subset, raw_dir=raw_dir)
        for subset in list(PAIR_SUBSETS.keys()) + ["general"]
    }


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of integrity issue codes for a normalized record.

    Does not raise or discard anything - callers decide what to do with the
    issues found (never silently discard problematic examples).
    """
    issues: list[str] = []

    if not record.get("candidate_response"):
        issues.append("empty_or_missing_candidate_response")

    label = record.get("label")
    if label not in VALID_LABELS:
        issues.append(f"invalid_label:{label!r}")

    subset = record.get("subset")
    if subset != "general":
        context = record.get("context") or {}
        if not any(context.get(k) for k in context):
            issues.append("missing_context")

    spans = record.get("hallucination_spans")
    if spans is not None and not isinstance(spans, list):
        issues.append("hallucination_spans_not_a_list")
    if subset == "general" and label == "hallucinated" and spans is not None and len(spans) == 0:
        # Not necessarily wrong (spans could legitimately be empty), but
        # worth surfacing since it's unusual for a labeled-hallucinated
        # example to have zero flagged spans.
        issues.append("hallucinated_with_no_spans")

    raw_id = record.get("raw_id")
    if raw_id in SUSPICIOUS_RAW_IDS:
        issues.append(f"suspicious_raw_id:{raw_id!r}")

    return issues
