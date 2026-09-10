"""ClaimGuard verifier: dataset adapter.

Consumes the Step 8/10 leakage-safe verifier pool
(data/processed/verifier_pool/{train,dev}.jsonl) directly - does NOT
rebuild the dataset from raw sources. Filters to records with actual
premise text available (premise_text_available=True), since a valid NLI
(premise, hypothesis, label) example requires real premise text, not a
reference.

As of Step 9, this filter excluded ALL FEVER records (FEVER's evidence was
reference-only - wiki_url/sentence_id, not resolved text). As of Step 10,
FEVER's wiki_pages evidence corpus was acquired and resolved, so
SUPPORTS/REFUTES FEVER records now DO carry real premise text and pass
this same filter - no code change was needed here for that to happen, this
module was already correctly generic. FEVER NOT ENOUGH INFO claims still
have no evidence annotation at all (Step 11: no legitimate neutral source
exists) and are excluded upstream, at pool-construction time
(claimguard.datasets.manifest.fever_train_pool_records), not here. See
PROJECT_REPORT.md Steps 9-11 for the full history and exact counts.

Also enforces, as an explicit defense-in-depth safety guard (the Step 8
pool files should already be clean, but this module does not trust that
silently): every loaded record's (source_dataset, source_subset) must be
in the allowed training set, none may come from a forbidden
evaluation-reserved dataset (RAGTruth, TruthfulQA), and none may be a
known cross-dataset-contaminated HaluEval summarization record (re-checked
against claimguard.datasets.manifest's live contamination computation,
not trusted from the file alone). Violations raise loudly - never
silently filtered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import config as cg_config
from ..datasets import manifest as mf

POOL_DIR = cg_config.resolve_path("data/processed/verifier_pool")

ALLOWED_SOURCE_SUBSETS = {
    ("fever", "train"),
    ("halueval", "qa"),
    ("halueval", "dialogue"),
    ("halueval", "summarization"),
}
FORBIDDEN_SOURCE_DATASETS = {"ragtruth", "truthfulqa"}


class TrainingSafetyError(RuntimeError):
    """Raised when a record from a forbidden, evaluation-reserved, or
    known-contaminated source is detected in what should be the clean
    training-eligible pool. Training must not silently proceed past this."""


@dataclass
class VerifierExample:
    premise: str
    hypothesis: str
    label: str  # "entailment" | "neutral" | "contradiction"
    source_dataset: str
    source_subset: str
    source_id: Any
    original_label: str


def _extract_halueval_raw_index(source_id: str) -> int | None:
    """HaluEval pool source_id format: f'{subset}_{raw_index}_{label}'
    (e.g. "summarization_1234_hallucinated"). Extract the purely-numeric
    token - subset names and labels never contain digits, so this is
    unambiguous."""
    for part in str(source_id).split("_"):
        if part.isdigit():
            return int(part)
    return None


def _enforce_safety(record: dict[str, Any], contaminated_indices: set[int]) -> None:
    source_dataset = record.get("source_dataset")
    source_subset = record.get("source_subset")

    if source_dataset in FORBIDDEN_SOURCE_DATASETS:
        raise TrainingSafetyError(
            f"Forbidden source_dataset {source_dataset!r} found in what should be the "
            f"training-eligible pool - this must never happen. "
            f"Record source_id={record.get('source_id')!r}."
        )
    if (source_dataset, source_subset) not in ALLOWED_SOURCE_SUBSETS:
        raise TrainingSafetyError(
            f"Unrecognized (source_dataset, source_subset)=({source_dataset!r}, "
            f"{source_subset!r}) - not in the allowed training set "
            f"{sorted(ALLOWED_SOURCE_SUBSETS)}. Refusing to use this record."
        )
    if source_dataset == "halueval" and source_subset == "summarization":
        raw_index = _extract_halueval_raw_index(record.get("source_id", ""))
        if raw_index is not None and raw_index in contaminated_indices:
            raise TrainingSafetyError(
                f"Excluded-contamination record leaked into the pool file: HaluEval "
                f"summarization raw_index={raw_index} (reason="
                f"{mf.EXCLUSION_REASON_RAGTRUTH_OVERLAP!r}). Refusing to train on it."
            )


def load_pool_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Verifier pool file not found: {path}. Run scripts/build_verifier_pool.py first."
        )
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_baseline_examples(
    pool_path: Path, require_premise_text: bool = True
) -> tuple[list[VerifierExample], dict[str, Any]]:
    """Load one pool split (train or dev) and filter to baseline-usable
    examples. Returns (examples, stats) where stats documents exactly what
    was included/excluded and why - never silent.
    """
    raw_records = load_pool_jsonl(pool_path)
    total = len(raw_records)

    contaminated_indices = mf.contaminated_halueval_summarization_raw_indices()
    for r in raw_records:
        _enforce_safety(r, contaminated_indices)

    usable: list[VerifierExample] = []
    excluded_no_premise = 0
    excluded_by_source: dict[str, int] = {}
    for r in raw_records:
        has_premise = bool(r.get("premise_text_available")) and bool((r.get("premise_text") or "").strip())
        if require_premise_text and not has_premise:
            excluded_no_premise += 1
            key = f"{r['source_dataset']}.{r['source_subset']}"
            excluded_by_source[key] = excluded_by_source.get(key, 0) + 1
            continue
        usable.append(VerifierExample(
            premise=r["premise_text"] or "",
            hypothesis=r["hypothesis"],
            label=r["label"],
            source_dataset=r["source_dataset"],
            source_subset=r["source_subset"],
            source_id=r["source_id"],
            original_label=r["original_label"],
        ))

    stats = {
        "pool_path": str(pool_path),
        "total_records_in_pool_file": total,
        "usable_records": len(usable),
        "excluded_no_premise_text": excluded_no_premise,
        "excluded_by_source": excluded_by_source,
        "note": "As of Step 10, excluded_no_premise_text should be 0 for FEVER SUPPORTS/REFUTES "
                "records (resolved premise text now available) - any nonzero FEVER exclusion "
                "here would indicate an unresolved claim slipped into the pool file and should "
                "be investigated, not assumed benign. See PROJECT_REPORT.md Steps 9-11.",
    }
    return usable, stats
