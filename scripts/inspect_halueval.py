"""Step 6B: HaluEval dataset acquisition and inspection.

Downloads (if not already present) the official HaluEval data files from the
RUCAIBox/HaluEval GitHub repository (the original project, MIT License) and
writes each, unmodified, to data/raw/halueval/. Then inspects the REAL
downloaded data and reports its actual structure - per-file schema, field
types, label/value distributions, sentinel/missing values, duplicates, and
cross-file overlap. Nothing about the schema is assumed from the README
prose or prior desk research; every claim below is checked against the real
data, and each subset is inspected on its own terms rather than forced into
a shared abstraction if the real data doesn't support one.

CPU / data-preparation only: does not load any ClaimGuard model, does not
touch the GPU, does not build any retrieval index.

Usage:
    python scripts/inspect_halueval.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

RAW_DIR = cg_config.resolve_path("data/raw/halueval")
BASE_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data"
# Exact filenames as documented in the official README - verified against the
# repo before writing this script (see PROJECT_REPORT.md Step 6B).
FILES = ["qa_data.json", "dialogue_data.json", "summarization_data.json", "general_data.json"]


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
    for fname in FILES:
        dest = RAW_DIR / fname
        if dest.exists():
            print(f"[acquire] {fname}: already present at {dest}, not re-downloading")
        else:
            url = f"{BASE_URL}/{fname}"
            print(f"[acquire] {fname}: downloading from {url} ...")
            try:
                _download(url, dest)
            except urllib.error.HTTPError as exc:
                print(f"[acquire] FAILED to download {fname}: HTTP {exc.code} {exc.reason}")
                continue
            size_kb = dest.stat().st_size / 1024
            print(f"[acquire] {fname}: saved {size_kb:.1f} KB -> {dest}")
        paths[fname] = dest
    return paths


def _parse_records(dest: Path) -> tuple[list[dict], str]:
    """Parse a raw file without assuming JSON-array vs JSON-Lines format.

    Returns (records, format_used) - do not assume based on the .json
    extension; try both and report which one actually worked.
    """
    raw_text = dest.read_text(encoding="utf-8")

    # Attempt 1: a single JSON array (or single JSON object) for the whole file.
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return parsed, "json_array"
        if isinstance(parsed, dict):
            return [parsed], "single_json_object"
    except json.JSONDecodeError:
        pass

    # Attempt 2: JSON Lines - one JSON object per line, despite the .json extension.
    records = []
    bad_lines = 0
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines += 1
    if records:
        fmt = "jsonl" if bad_lines == 0 else f"jsonl_with_{bad_lines}_unparseable_lines"
        return records, fmt

    raise ValueError(f"Could not parse {dest} as either a JSON array or JSON Lines")


def inspect_file(fname: str, dest: Path) -> dict:
    _section(f"FILE: {fname}")
    if not dest.exists():
        print("NOT DOWNLOADED - skipping inspection")
        return {"file": fname, "status": "missing"}

    records, fmt = _parse_records(dest)
    n = len(records)
    print(f"Parsed as: {fmt} | Record count: {n:,}")

    # Field names: union and intersection across ALL records (do not assume
    # every record has the same keys).
    key_sets = [set(r.keys()) for r in records if isinstance(r, dict)]
    non_dict_records = n - len(key_sets)
    union_keys = set().union(*key_sets) if key_sets else set()
    intersection_keys = set.intersection(*key_sets) if key_sets else set()
    print(f"Non-dict top-level records: {non_dict_records}")
    print(f"Field names - union across all records: {sorted(union_keys)}")
    print(f"Field names - present in EVERY record (intersection): {sorted(intersection_keys)}")
    if union_keys != intersection_keys:
        print(f"WARNING: inconsistent schema - fields not present in every record: "
              f"{sorted(union_keys - intersection_keys)}")

    # Field types (per field, the set of Python types observed).
    field_types: dict[str, set[str]] = {k: set() for k in union_keys}
    for r in records:
        if not isinstance(r, dict):
            continue
        for k in union_keys:
            if k in r:
                field_types[k].add(type(r[k]).__name__)
    print(f"Field types observed: {{{', '.join(f'{k}: {sorted(v)}' for k, v in sorted(field_types.items()))}}}")

    # Missing values per field (present as key but null/None, or key entirely absent).
    missing_null = {k: sum(1 for r in records if isinstance(r, dict) and r.get(k) is None) for k in union_keys}
    missing_absent = {k: sum(1 for r in records if isinstance(r, dict) and k not in r) for k in union_keys}
    print(f"Null/None values per field: {missing_null}")
    print(f"Field entirely absent from record, per field: {missing_absent}")

    # Empty-string sentinel check per string field.
    empty_string_counts = {}
    for k in union_keys:
        if "str" in field_types.get(k, set()):
            empty_string_counts[k] = sum(1 for r in records if isinstance(r, dict) and r.get(k) == "")
    print(f"Empty-string ('') values per string field: {empty_string_counts}")

    # Sample records (truncate long strings, cap how many we print).
    print("\nSample records (first 2, truncated):")
    for r in records[:2]:
        compact = {
            k: (v if not isinstance(v, str) or len(v) <= 120 else v[:120] + "...") for k, v in r.items()
        } if isinstance(r, dict) else r
        print(f"  {compact}")

    # Label / value distribution for any field that looks like a label
    # (small number of unique values relative to record count).
    label_like_fields = []
    for k in union_keys:
        values = [r.get(k) for r in records if isinstance(r, dict) and k in r]
        unique_vals = set(v for v in values if isinstance(v, (str, int, bool)) or v is None)
        if 0 < len(unique_vals) <= 10 and len(unique_vals) < n:
            label_like_fields.append(k)
    for k in label_like_fields:
        dist = Counter(r.get(k) for r in records if isinstance(r, dict))
        print(f"\nValue distribution for label-like field '{k}': {dict(dist)}")

    # Duplicate full-row check.
    seen = set()
    dup_count = 0
    for r in records:
        key = json.dumps(r, sort_keys=True, default=str)
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    print(f"\nExact duplicate records: {dup_count}")

    # Duplicate/unique ID check, if an id-like field exists. If duplicates are
    # found, show the actual colliding records rather than just the count -
    # need to see real values before calling this a genuine dataset issue.
    id_like_fields = [k for k in union_keys if k.lower() in ("id", "idx", "index", "qid", "example_id")]
    for k in id_like_fields:
        ids = [r.get(k) for r in records if isinstance(r, dict)]
        id_counts = Counter(ids)
        dup_ids = {v: c for v, c in id_counts.items() if c > 1}
        print(f"ID-like field '{k}': {len(ids):,} values, {len(set(ids)):,} unique")
        if dup_ids:
            print(f"  Duplicate ID values found: {dup_ids}")
            for dup_id, count in list(dup_ids.items())[:5]:
                matching = [r for r in records if isinstance(r, dict) and r.get(k) == dup_id]
                print(f"  Records with {k}={dup_id!r} ({count} of them):")
                for m in matching:
                    compact = {
                        kk: (vv if not isinstance(vv, str) or len(vv) <= 100 else vv[:100] + "...")
                        for kk, vv in m.items()
                    }
                    print(f"    {compact}")
    if not id_like_fields:
        print("No explicit id-like field found in this file's schema.")

    return {
        "file": fname,
        "format": fmt,
        "record_count": n,
        "union_keys": sorted(union_keys),
        "intersection_keys": sorted(intersection_keys),
        "field_types": {k: sorted(v) for k, v in field_types.items()},
        "duplicate_records": dup_count,
    }


def cross_file_checks(all_records: dict[str, list[dict]]) -> None:
    _section("CROSS-FILE CHECKS")
    # Overlap of exact question/claim-like text fields across files, where a
    # shared field name exists (e.g. if two files both have a 'knowledge' or
    # similar field, check for suspicious exact overlap - NOT expected to
    # mean much across semantically different tasks, but worth a quick check
    # for accidental duplication during file generation).
    text_field_candidates = ["knowledge", "question", "document"]
    for field in text_field_candidates:
        present_in = {fname: recs for fname, recs in all_records.items()
                      if recs and isinstance(recs[0], dict) and field in recs[0]}
        if len(present_in) > 1:
            names = list(present_in.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = names[i], names[j]
                    set_a = set(r.get(field) for r in present_in[a] if isinstance(r, dict))
                    set_b = set(r.get(field) for r in present_in[b] if isinstance(r, dict))
                    overlap = set_a & set_b
                    print(f"Overlap of '{field}' values between [{a}] and [{b}]: {len(overlap)}")


def main() -> int:
    _section("HALUEVAL ACQUISITION")
    print(f"Source: RUCAIBox/HaluEval (GitHub, main branch) - the original HaluEval project "
          f"repository, MIT License (verified before download).")
    print(f"Files requested: {FILES}")
    print(f"Raw storage directory: {RAW_DIR}")

    paths = acquire()

    all_records: dict[str, list[dict]] = {}
    reports = {}
    for fname, dest in paths.items():
        report = inspect_file(fname, dest)
        reports[fname] = report
        if dest.exists():
            try:
                records, _fmt = _parse_records(dest)
                all_records[fname] = records
            except ValueError as exc:
                print(f"[inspect] could not re-parse {fname} for cross-file checks: {exc!r}")

    cross_file_checks(all_records)

    _section("SUMMARY")
    for fname, rep in reports.items():
        if rep.get("status") == "missing":
            print(f"{fname}: NOT DOWNLOADED")
        else:
            print(f"{fname}: {rep['record_count']:,} records | format={rep['format']} | "
                  f"duplicates={rep['duplicate_records']}")
    print("\nInspection complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
