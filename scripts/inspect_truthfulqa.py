"""Step 6D: TruthfulQA dataset acquisition and inspection.

Downloads (if not already present) the official TruthfulQA data files from
the sylinrl/TruthfulQA GitHub repository (the original authors' own repo -
Stephanie Lin, Jacob Hilton, Owain Evans - Apache-2.0 License) and writes
them, unmodified, to data/raw/truthfulqa/. Then inspects the REAL downloaded
data: format, schema, category/type distributions, answer-candidate
structure, duplicates, and whether any split semantics genuinely exist.

TruthfulQA is NOT forced into the FEVER/HaluEval/RAGTruth schemas - its task
semantics (question + multiple free-form correct/incorrect answers, plus a
separately-released multiple-choice reformulation) are materially different
and are inspected on their own terms.

Two official files are acquired, both from the original repo, representing
two different official task framings:
  - TruthfulQA.csv (root of repo)  - the generation-style benchmark: a
    question plus free-form lists of correct and incorrect reference answers.
  - data/mc_task.json               - the official multiple-choice
    reformulation of the same questions.

CPU / data-preparation only: does not load any ClaimGuard model, does not
touch the GPU, does not build any retrieval index.

Usage:
    python scripts/inspect_truthfulqa.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

RAW_DIR = cg_config.resolve_path("data/raw/truthfulqa")
BASE_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main"
FILES = {
    "TruthfulQA.csv": f"{BASE_URL}/TruthfulQA.csv",
    "mc_task.json": f"{BASE_URL}/data/mc_task.json",
}


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "claimguard-dataset-acquisition"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    dest.write_bytes(data)


def acquire() -> dict[str, Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fname, url in FILES.items():
        dest = RAW_DIR / fname
        if dest.exists():
            print(f"[acquire] {fname}: already present at {dest}, not re-downloading")
        else:
            print(f"[acquire] {fname}: downloading from {url} ...")
            try:
                _download(url, dest)
            except urllib.error.HTTPError as exc:
                print(f"[acquire] FAILED to download {fname}: HTTP {exc.code} {exc.reason} "
                      f"(assumed path may be wrong - not guessing an alternative, reporting as-is)")
                continue
            except urllib.error.URLError as exc:
                print(f"[acquire] FAILED to download {fname}: {exc!r}")
                continue
            size_kb = dest.stat().st_size / 1024
            print(f"[acquire] {fname}: saved {size_kb:.1f} KB -> {dest}")
        if dest.exists():
            paths[fname] = dest
    return paths


def inspect_csv(dest: Path) -> dict:
    _section("FILE: TruthfulQA.csv")
    raw_text = dest.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(raw_text))
    fieldnames = reader.fieldnames or []
    print(f"Header columns (exact, as found - not assumed): {fieldnames}")

    rows = list(reader)
    n = len(rows)
    print(f"Row count: {n:,}")

    print("\nSample records (first 2, truncated):")
    for r in rows[:2]:
        compact = {k: (v if not isinstance(v, str) or len(v) <= 150 else v[:150] + "...") for k, v in r.items()}
        print(f"  {compact}")

    # Missing/empty per column.
    missing = {col: sum(1 for r in rows if r.get(col) is None) for col in fieldnames}
    empty = {col: sum(1 for r in rows if r.get(col) == "") for col in fieldnames}
    print(f"\nMissing (key absent/None) per column: {missing}")
    print(f"Empty-string per column: {empty}")

    # Category / Type distributions.
    if "Category" in fieldnames:
        cat_dist = Counter(r.get("Category") for r in rows)
        print(f"\nCategory distribution ({len(cat_dist)} categories): {dict(cat_dist)}")
    if "Type" in fieldnames:
        type_dist = Counter(r.get("Type") for r in rows)
        print(f"Type distribution: {dict(type_dist)}")

    # Answer-list fields: detect the actual delimiter empirically rather
    # than assuming "; ".
    answer_list_cols = [c for c in fieldnames if "Answers" in c]
    print(f"\nAnswer-list-like columns detected: {answer_list_cols}")
    for col in answer_list_cols:
        sample_vals = [r.get(col, "") for r in rows[:5]]
        print(f"  Sample raw values for '{col}': {sample_vals}")

    # Try the conventional TruthfulQA delimiter ('; ') and report how many
    # rows produce >1 item vs staying as a single blob, so we can see
    # empirically whether the delimiter guess is right.
    delimiter = "; "
    counts_per_row: dict[str, list[int]] = {col: [] for col in answer_list_cols}
    for r in rows:
        for col in answer_list_cols:
            val = r.get(col, "") or ""
            items = [x for x in val.split(delimiter) if x.strip()] if val else []
            counts_per_row[col].append(len(items))
    for col in answer_list_cols:
        counts = counts_per_row[col]
        dist = Counter(counts)
        print(f"  '{col}' split on {delimiter!r}: item-count distribution across rows "
              f"(0 = empty, 1 = single/no-delimiter-found): {dict(sorted(dist.items()))}")

    # Best Answer / Best Incorrect Answer presence.
    for col in ("Best Answer", "Best Incorrect Answer"):
        if col in fieldnames:
            n_present = sum(1 for r in rows if r.get(col))
            print(f"'{col}': present (non-empty) in {n_present:,}/{n:,} rows")

    # Overlap between correct/incorrect answer lists within the same row
    # (a real anomaly if the identical string appears in both).
    if "Correct Answers" in fieldnames and "Incorrect Answers" in fieldnames:
        overlap_rows = []
        for r in rows:
            correct_items = set(x.strip() for x in (r.get("Correct Answers") or "").split(delimiter) if x.strip())
            incorrect_items = set(x.strip() for x in (r.get("Incorrect Answers") or "").split(delimiter) if x.strip())
            shared = correct_items & incorrect_items
            if shared:
                overlap_rows.append((r.get("Question"), shared))
        print(f"\nRows where the SAME answer string appears in both Correct Answers and "
              f"Incorrect Answers: {len(overlap_rows)}")
        if overlap_rows:
            print("  Investigating the actual rows before calling this a finding:")
            for question, shared in overlap_rows:
                print(f"    Question: {question!r}")
                print(f"    Shared string(s): {shared}")

    # Duplicate questions - exact and normalized (lowercased/stripped).
    if "Question" in fieldnames:
        questions = [r.get("Question", "") for r in rows]
        exact_dupes = {q: c for q, c in Counter(questions).items() if c > 1}
        print(f"\nExact duplicate question strings: {len(exact_dupes)} "
              f"(showing up to 3): {dict(list(exact_dupes.items())[:3])}")
        normalized = [q.strip().lower() for q in questions]
        norm_dupes = {q: c for q, c in Counter(normalized).items() if c > 1}
        print(f"Normalized (lowercased/stripped) duplicate questions: {len(norm_dupes)}")

    # Exact duplicate rows.
    seen = set()
    dup_count = 0
    for r in rows:
        key = json.dumps(r, sort_keys=True)
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    print(f"\nExact duplicate full rows: {dup_count}")

    # Split semantics check - explicitly look for and report absence of any
    # split-like column, rather than assuming none exists.
    split_like = [c for c in fieldnames if c.lower() in ("split", "set", "partition")]
    if split_like:
        print(f"\nSplit-like columns found: {split_like}")
    else:
        print("\nSplit-like columns found: NONE - confirms no official train/validation/test "
              "split column in this file")

    return {"fieldnames": fieldnames, "row_count": n, "answer_list_cols": answer_list_cols,
            "duplicate_rows": dup_count, "rows": rows}


def inspect_mc_json(dest: Path) -> dict:
    _section("FILE: mc_task.json")
    raw_text = dest.read_text(encoding="utf-8")

    # Determine actual format empirically - JSON array vs JSON Lines.
    try:
        parsed = json.loads(raw_text)
        fmt = "json_array" if isinstance(parsed, list) else "single_json_object"
        records = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        fmt = "jsonl"

    n = len(records)
    print(f"Parsed as: {fmt} | Record count: {n:,}")

    if not records:
        return {"record_count": 0}

    key_sets = [set(r.keys()) for r in records if isinstance(r, dict)]
    union_keys = set().union(*key_sets) if key_sets else set()
    intersection_keys = set.intersection(*key_sets) if key_sets else set()
    print(f"Field names - union: {sorted(union_keys)}")
    print(f"Field names - present in every record: {sorted(intersection_keys)}")

    print("\nSample record (first, truncated):")
    r0 = records[0]
    compact = {}
    for k, v in r0.items():
        s = str(v)
        compact[k] = s[:300] + "..." if len(s) > 300 else v
    print(f"  {compact}")

    # Inspect mc1_targets / mc2_targets structure empirically - do not
    # assume HF's re-described "dict with choices/labels" shape applies to
    # this original file without checking.
    for field in ("mc1_targets", "mc2_targets"):
        if field in union_keys:
            sample_val = r0.get(field)
            print(f"\n'{field}' actual type in this file: {type(sample_val).__name__}")
            print(f"'{field}' sample value: {sample_val}"[:500])

    # Duplicate questions.
    if "question" in union_keys:
        questions = [r.get("question", "") for r in records]
        exact_dupes = {q: c for q, c in Counter(questions).items() if c > 1}
        print(f"\nExact duplicate questions in mc_task.json: {len(exact_dupes)}")

    return {"union_keys": sorted(union_keys), "record_count": n, "records": records}


def cross_file_check(csv_report: dict, json_report: dict) -> None:
    _section("CROSS-FILE CHECK: TruthfulQA.csv vs mc_task.json question overlap")
    csv_questions = set(r.get("Question", "").strip() for r in csv_report.get("rows", []))
    json_questions = set(r.get("question", "").strip() for r in json_report.get("records", []) if isinstance(r, dict))
    print(f"Unique questions in TruthfulQA.csv: {len(csv_questions):,}")
    print(f"Unique questions in mc_task.json: {len(json_questions):,}")
    overlap = csv_questions & json_questions
    print(f"Exact question-text overlap between the two files: {len(overlap):,} "
          f"({len(overlap) / max(len(csv_questions), 1) * 100:.1f}% of CSV questions)")
    only_csv = csv_questions - json_questions
    only_json = json_questions - csv_questions
    print(f"Questions only in CSV (not in mc_task.json): {len(only_csv)}")
    print(f"Questions only in mc_task.json (not in CSV): {len(only_json)}")


def main() -> int:
    _section("TRUTHFULQA ACQUISITION")
    print("Source: sylinrl/TruthfulQA (GitHub, main branch) - the original authors' own "
          "repository (Lin, Hilton, Evans - \"TruthfulQA: Measuring How Models Mimic Human "
          "Falsehoods\"), Apache-2.0 License (verified before download). Preferred over the "
          "community-added Hugging Face mirror (truthfulqa/truthful_qa, added by a third "
          "party per its own dataset card, not the original authors).")
    print(f"Files requested: {list(FILES.keys())}")
    print(f"Raw storage directory: {RAW_DIR}")

    paths = acquire()

    csv_report = {}
    json_report = {}
    if "TruthfulQA.csv" in paths:
        csv_report = inspect_csv(paths["TruthfulQA.csv"])
    else:
        print("\nTruthfulQA.csv was not acquired - skipping its inspection.")

    if "mc_task.json" in paths:
        json_report = inspect_mc_json(paths["mc_task.json"])
    else:
        print("\nmc_task.json was not acquired - skipping its inspection.")

    if csv_report.get("rows") and json_report.get("records"):
        cross_file_check(csv_report, json_report)

    _section("SUMMARY")
    if csv_report:
        print(f"TruthfulQA.csv: {csv_report['row_count']:,} rows | duplicates={csv_report['duplicate_rows']}")
    if json_report:
        print(f"mc_task.json: {json_report.get('record_count', 0):,} records")
    print("\nInspection complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
