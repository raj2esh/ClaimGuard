"""TruthfulQA dataset loading and normalization utilities.

Loads the raw TruthfulQA data acquired by scripts/inspect_truthfulqa.py from
data/raw/truthfulqa/{TruthfulQA.csv,mc_task.json} and joins/normalizes it
into one record per question. This module does NOT implement model training
and does NOT load any model.

TruthfulQA is NOT a claim/evidence-grounded dataset like FEVER/HaluEval/
RAGTruth, and is NOT forced into their shape here:
  - There is no retrieval evidence or grounding context - `source` is a
    reference URL for where the question/answers came from, not passable
    retrieval evidence.
  - There is no train/validation/test split of any kind in either raw file
    (verified empirically - no split-like column exists). This module does
    NOT invent one. There is no get_eval_set()/get_training_pool() pair
    here unlike claimguard.datasets.ragtruth, deliberately, because doing
    so would fabricate structure the real data does not have.
  - Each question has MULTIPLE free-form correct answers and MULTIPLE
    free-form incorrect answers (not a single label), plus a separate
    "best" answer of each kind, plus three DIFFERENT official
    multiple-choice reformulations (mc0/mc1/mc2, described below). Records
    are kept at question-level (not flattened into one row per answer
    candidate) since no downstream step in this project yet requires that
    transformation - see PROJECT_REPORT.md Step 6D for the reasoning.

Real-data findings this module accounts for (full detail in
PROJECT_REPORT.md Step 6D):
  - The real row/record count in both files is **790**, not the commonly
    cited 817 - the upstream repository has evidently grown since the
    original paper's published figure. Likewise there are **37** real
    categories in the data, not the commonly cited 38. Both counts are
    read from the real files, not hard-coded, so this module will reflect
    whatever the currently-downloaded files actually contain.
  - TruthfulQA.csv columns (verified, not assumed): Type, Category,
    Question, Best Answer, Best Incorrect Answer, Correct Answers,
    Incorrect Answers, Source. Answer-list columns use "; " as the item
    delimiter within a single CSV cell (confirmed empirically against the
    real data, not assumed from documentation).
  - mc_task.json has THREE targets dicts per question, not the two
    (mc1/mc2) described by the (non-original-author) Hugging Face mirror's
    dataset card: `mc0_targets` (exactly one correct option paired with
    exactly one incorrect option - a simple binary choice), `mc1_targets`
    (one correct option among several incorrect ones - single-answer
    multiple choice), and `mc2_targets` (potentially SEVERAL options
    marked correct among several incorrect ones - multi-label multiple
    choice, confirmed by inspecting real values with multiple "1"s). Each
    target dict maps an answer-choice string to 0 (incorrect) or 1
    (correct).
  - The two files join perfectly by exact question text: 790/790 questions
    in each file, 100% overlap, 0 orphans in either direction.
  - Two questions have the literal string "Unknown" present in BOTH their
    Correct Answers and Incorrect Answers lists (e.g. "What proportion of
    Canadians are straight?"). Investigated and confirmed intentional, not
    a data error: for genuinely unknowable statistics, "Unknown" can be
    scored as either an honest (correct) or insufficiently informative
    (incorrect) answer depending on evaluation methodology, and the
    original authors included it in both lists for such questions.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from .. import config as cg_config

RAW_DIR = cg_config.resolve_path("data/raw/truthfulqa")
ANSWER_LIST_DELIMITER = "; "
KNOWN_TYPES = {"Adversarial", "Non-Adversarial"}


def load_raw_csv(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    path = raw_dir / "TruthfulQA.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw TruthfulQA CSV not found: {path}. Run scripts/inspect_truthfulqa.py first."
        )
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_raw_mc_json(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    path = raw_dir / "mc_task.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw TruthfulQA mc_task.json not found: {path}. Run scripts/inspect_truthfulqa.py first."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def parse_answer_list(raw_value: str | None, delimiter: str = ANSWER_LIST_DELIMITER) -> list[str]:
    """Split a semicolon-delimited answer-list cell into individual answers."""
    if not raw_value:
        return []
    return [x.strip() for x in raw_value.split(delimiter) if x.strip()]


def join_records(
    csv_rows: list[dict[str, Any]], mc_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join CSV rows to mc_task.json records by exact question text.

    Lossless: every CSV field is preserved; mc0/mc1/mc2 target dicts are
    attached verbatim when a matching question is found. A question present
    in only one file is kept, not dropped (`has_mc_data` reflects whether
    the join succeeded) - the real data has 0 such cases, but this must not
    silently disappear if that changes.
    """
    mc_by_question = {r.get("question"): r for r in mc_records if isinstance(r, dict)}

    joined = []
    for i, row in enumerate(csv_rows):
        question = row.get("Question", "")
        mc = mc_by_question.get(question)

        record = {
            "question_id": i,  # synthetic - no native id field exists in either raw file
            "question": question,
            "category": row.get("Category", ""),
            "type": row.get("Type", ""),
            "source": row.get("Source", ""),
            "best_answer": row.get("Best Answer", ""),
            "best_incorrect_answer": row.get("Best Incorrect Answer", ""),
            "correct_answers": parse_answer_list(row.get("Correct Answers")),
            "incorrect_answers": parse_answer_list(row.get("Incorrect Answers")),
            "mc0_targets": mc.get("mc0_targets") if mc else None,
            "mc1_targets": mc.get("mc1_targets") if mc else None,
            "mc2_targets": mc.get("mc2_targets") if mc else None,
            "has_mc_data": mc is not None,
        }
        joined.append(record)
    return joined


def load_normalized(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    """Load both raw files and return the joined, normalized, question-level records."""
    csv_rows = load_raw_csv(raw_dir)
    mc_records = load_raw_mc_json(raw_dir)
    return join_records(csv_rows, mc_records)


def find_duplicate_questions(records: list[dict[str, Any]]) -> dict[str, int]:
    """Return {question: count} for any question text appearing more than once.

    On the real data this returns {} - 0 exact and 0 normalized duplicate
    questions were found at inspection time.
    """
    counts: dict[str, int] = {}
    for r in records:
        q = r.get("question", "")
        counts[q] = counts.get(q, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of integrity issue codes for a normalized record.

    Does not raise or discard anything - callers decide what to do with the
    issues found (never silently discard problematic examples).
    """
    issues: list[str] = []

    if not record.get("question"):
        issues.append("empty_or_missing_question")
    if not record.get("category"):
        issues.append("empty_or_missing_category")
    if record.get("type") not in KNOWN_TYPES:
        issues.append(f"unexpected_type:{record.get('type')!r}")
    if not record.get("best_answer"):
        issues.append("empty_or_missing_best_answer")
    if not record.get("correct_answers"):
        issues.append("empty_correct_answers_list")
    if not record.get("incorrect_answers"):
        issues.append("empty_incorrect_answers_list")

    for mc_field in ("mc0_targets", "mc1_targets", "mc2_targets"):
        targets = record.get(mc_field)
        if targets is None:
            if record.get("has_mc_data"):
                issues.append(f"missing_{mc_field}_despite_has_mc_data")
            continue
        if not isinstance(targets, dict) or not targets:
            issues.append(f"{mc_field}_not_a_nonempty_dict")
            continue
        values = set(targets.values())
        if not values.issubset({0, 1}):
            issues.append(f"{mc_field}_has_non_binary_values")
        if 1 not in values:
            issues.append(f"{mc_field}_has_no_correct_option")

    return issues
