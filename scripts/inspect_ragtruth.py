"""Step 6C: RAGTruth dataset acquisition and inspection.

RAGTruth is ClaimGuard's PRIMARY end-to-end evaluation dataset - this script
is written with correspondingly higher scrutiny than the FEVER/HaluEval
inspection scripts, especially around cross-split leakage, since a leak here
would silently invalidate any effectiveness claim built on it.

Downloads (if not already present) the official RAGTruth data files from the
ParticleMedia/RAGTruth GitHub repository (the paper's own repo, MIT License)
and writes them, unmodified, to data/raw/ragtruth/. Then inspects the REAL
downloaded data: schema, labels, annotation/offset validity, and - most
importantly - whether the same underlying source document/question ever
contributes responses to both the train and test splits.

Nothing about the schema, split semantics, or field names is assumed from
the paper, README, or prior desk research; every claim below is checked
against the real data.

CPU / data-preparation only: does not load any ClaimGuard model, does not
touch the GPU, does not build any retrieval index.

Usage:
    python scripts/inspect_ragtruth.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

RAW_DIR = cg_config.resolve_path("data/raw/ragtruth")
BASE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
FILES = ["response.jsonl", "source_info.jsonl"]


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
            except urllib.error.URLError as exc:
                print(f"[acquire] FAILED to download {fname}: {exc!r}")
                continue
            size_kb = dest.stat().st_size / 1024
            print(f"[acquire] {fname}: saved {size_kb:.1f} KB -> {dest}")
        paths[fname] = dest
    return paths


def _parse_records(dest: Path) -> tuple[list[dict], str]:
    """Parse without assuming JSON-array vs JSON-Lines format."""
    raw_text = dest.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return parsed, "json_array"
        if isinstance(parsed, dict):
            return [parsed], "single_json_object"
    except json.JSONDecodeError:
        pass

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


def _schema_report(records: list[dict], label: str) -> dict:
    key_sets = [set(r.keys()) for r in records if isinstance(r, dict)]
    union_keys = set().union(*key_sets) if key_sets else set()
    intersection_keys = set.intersection(*key_sets) if key_sets else set()
    print(f"[{label}] Field names - union: {sorted(union_keys)}")
    print(f"[{label}] Field names - present in every record: {sorted(intersection_keys)}")
    if union_keys != intersection_keys:
        print(f"[{label}] WARNING: inconsistent schema - fields not in every record: "
              f"{sorted(union_keys - intersection_keys)}")

    field_types: dict[str, set[str]] = {k: set() for k in union_keys}
    for r in records:
        if not isinstance(r, dict):
            continue
        for k in union_keys:
            if k in r:
                field_types[k].add(type(r[k]).__name__)
    print(f"[{label}] Field types: "
          f"{{{', '.join(f'{k}: {sorted(v)}' for k, v in sorted(field_types.items()))}}}")

    missing_null = {k: sum(1 for r in records if isinstance(r, dict) and r.get(k) is None) for k in union_keys}
    print(f"[{label}] Null/None per field: {missing_null}")

    return {"union_keys": sorted(union_keys), "intersection_keys": sorted(intersection_keys),
            "field_types": {k: sorted(v) for k, v in field_types.items()}}


def inspect_response_file(records: list[dict]) -> dict:
    _section("FILE: response.jsonl")
    n = len(records)
    print(f"Record count: {n:,}")
    schema = _schema_report(records, "response")

    print("\nSample records (first 2, labels truncated, text truncated):")
    for r in records[:2]:
        compact = {}
        for k, v in r.items():
            if k == "labels" and isinstance(v, list):
                compact[k] = v[:2] if len(v) > 2 else v
                if len(v) > 2:
                    compact[k].append(f"...({len(v)} total)")
            elif isinstance(v, str) and len(v) > 150:
                compact[k] = v[:150] + "..."
            else:
                compact[k] = v
        print(f"  {compact}")

    # ID uniqueness.
    id_field = "id" if "id" in schema["union_keys"] else None
    if id_field:
        ids = [r.get(id_field) for r in records]
        print(f"\n'{id_field}' field: {len(ids):,} values, {len(set(ids)):,} unique")
        id_counts = Counter(ids)
        dups = {k: v for k, v in id_counts.items() if v > 1}
        if dups:
            print(f"  Duplicate id values: {dict(list(dups.items())[:10])}"
                  f"{' ...' if len(dups) > 10 else ''}")

    # split distribution.
    if "split" in schema["union_keys"]:
        split_dist = Counter(r.get("split") for r in records)
        print(f"\nSplit distribution: {dict(split_dist)}")

    # quality distribution.
    if "quality" in schema["union_keys"]:
        quality_dist = Counter(r.get("quality") for r in records)
        print(f"Quality field distribution: {dict(quality_dist)}")

    # model distribution (which LLM generated each response), if present.
    for candidate in ("model", "model_name"):
        if candidate in schema["union_keys"]:
            model_dist = Counter(r.get(candidate) for r in records)
            print(f"'{candidate}' distribution: {dict(model_dist)}")

    # task_type, if present directly on response records.
    if "task_type" in schema["union_keys"]:
        task_dist = Counter(r.get("task_type") for r in records)
        print(f"'task_type' distribution (on response records): {dict(task_dist)}")

    # Exact duplicate rows.
    seen = set()
    dup_count = 0
    for r in records:
        key = json.dumps(r, sort_keys=True, default=str)
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    print(f"\nExact duplicate response records: {dup_count}")

    # labels structure.
    labels_field = "labels" if "labels" in schema["union_keys"] else None
    if labels_field:
        _inspect_labels(records, labels_field)

    return {"count": n, "schema": schema, "id_field": id_field,
            "labels_field": labels_field, "duplicate_records": dup_count}


def _inspect_labels(records: list[dict], labels_field: str) -> None:
    _section("ANNOTATION / LABEL STRUCTURE (response.labels)")
    n_with_labels = sum(1 for r in records if r.get(labels_field))
    n_empty_labels = sum(1 for r in records if isinstance(r.get(labels_field), list) and len(r[labels_field]) == 0)
    n_non_list = sum(1 for r in records if not isinstance(r.get(labels_field), list))
    print(f"Responses with a non-empty labels list: {n_with_labels:,} / {len(records):,}")
    print(f"Responses with an EMPTY labels list []: {n_empty_labels:,} / {len(records):,}")
    print(f"Responses where '{labels_field}' is not a list: {n_non_list}")

    all_label_dicts = []
    for r in records:
        labels = r.get(labels_field)
        if isinstance(labels, list):
            all_label_dicts.extend(x for x in labels if isinstance(x, dict))

    print(f"Total individual label/span annotations across all responses: {len(all_label_dicts):,}")
    if not all_label_dicts:
        return

    label_key_sets = [set(x.keys()) for x in all_label_dicts]
    label_union = set().union(*label_key_sets)
    label_intersection = set.intersection(*label_key_sets)
    print(f"Label dict field names - union: {sorted(label_union)}")
    print(f"Label dict field names - present in every label: {sorted(label_intersection)}")

    if "label_type" in label_union:
        type_dist = Counter(x.get("label_type") for x in all_label_dicts)
        print(f"'label_type' distribution: {dict(type_dist)}")

    if "implicit_true" in label_union:
        implicit_dist = Counter(x.get("implicit_true") for x in all_label_dicts)
        print(f"'implicit_true' distribution: {dict(implicit_dist)}")

    # Offset sanity (start <= end, non-negative) - without response text yet.
    if "start" in label_union and "end" in label_union:
        bad_offsets = [x for x in all_label_dicts
                       if not isinstance(x.get("start"), int) or not isinstance(x.get("end"), int)
                       or x.get("start", 0) > x.get("end", 0) or x.get("start", 0) < 0]
        print(f"Labels with non-int or start>end or negative offsets (structural check only, "
              f"not yet validated against response text): {len(bad_offsets)}")
        if bad_offsets:
            print(f"  Example malformed offset label: {bad_offsets[0]}")


def inspect_source_info_file(records: list[dict]) -> dict:
    _section("FILE: source_info.jsonl")
    n = len(records)
    print(f"Record count: {n:,}")
    schema = _schema_report(records, "source_info")

    print("\nSample records (first 2, one per differing task_type if possible, truncated):")
    seen_task_types = set()
    shown = 0
    for r in records:
        tt = r.get("task_type") if isinstance(r, dict) else None
        if tt in seen_task_types and shown >= 2:
            continue
        seen_task_types.add(tt)
        compact = {
            k: (v if not isinstance(v, str) or len(v) <= 150 else v[:150] + "...")
            for k, v in r.items()
        }
        print(f"  {compact}")
        shown += 1
        if shown >= 6:
            break

    id_field = "source_id" if "source_id" in schema["union_keys"] else ("id" if "id" in schema["union_keys"] else None)
    if id_field:
        ids = [r.get(id_field) for r in records]
        print(f"\n'{id_field}' field: {len(ids):,} values, {len(set(ids)):,} unique")

    if "task_type" in schema["union_keys"]:
        task_dist = Counter(r.get("task_type") for r in records)
        print(f"'task_type' distribution: {dict(task_dist)}")

    seen = set()
    dup_count = 0
    for r in records:
        key = json.dumps(r, sort_keys=True, default=str)
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    print(f"\nExact duplicate source_info records: {dup_count}")

    return {"count": n, "schema": schema, "id_field": id_field}


def join_and_leakage_checks(response_records: list[dict], source_records: list[dict],
                             response_meta: dict, source_meta: dict) -> None:
    _section("JOIN + LEAKAGE INVESTIGATION (high scrutiny - primary eval dataset)")

    # Determine the linking field empirically rather than assuming
    # "source_id" - check what key(s) on response records match source
    # record ids.
    source_id_field = source_meta["id_field"]
    candidate_link_fields = [k for k in response_meta["schema"]["union_keys"]
                              if "source" in k.lower() or k == source_id_field]
    print(f"Candidate response->source_info linking fields: {candidate_link_fields}")

    source_ids = set(r.get(source_id_field) for r in source_records) if source_id_field else set()
    link_field = None
    for cand in candidate_link_fields:
        resp_vals = set(r.get(cand) for r in response_records)
        overlap = resp_vals & source_ids
        print(f"  '{cand}' on response records: {len(resp_vals):,} unique values, "
              f"{len(overlap):,} match a source_info id ({len(overlap)/max(len(resp_vals),1)*100:.1f}%)")
        if overlap and len(overlap) / max(len(resp_vals), 1) > 0.9:
            link_field = cand

    if not link_field:
        print("Could not confidently identify a response->source_info linking field "
              "(no candidate matched >90% of source_info ids) - skipping join-dependent checks.")
        return

    print(f"\nUsing '{link_field}' as the response->source_info link field for the checks below.")

    # Referential integrity.
    orphan_responses = [r for r in response_records if r.get(link_field) not in source_ids]
    print(f"Response records whose '{link_field}' has NO matching source_info record: {len(orphan_responses):,}")

    # How many responses per source (multiple annotations/models per source?).
    responses_per_source: dict = defaultdict(list)
    for r in response_records:
        responses_per_source[r.get(link_field)].append(r)
    counts = Counter(len(v) for v in responses_per_source.values())
    print(f"Distribution of #responses per source_info item: {dict(counts)}")

    # THE CRITICAL CHECK: does any single source_info item have responses in
    # BOTH the train and test splits?
    if "split" in response_meta["schema"]["union_keys"]:
        cross_split_sources = []
        for source_id, resp_list in responses_per_source.items():
            splits_seen = set(r.get("split") for r in resp_list)
            if len(splits_seen) > 1:
                cross_split_sources.append((source_id, splits_seen, len(resp_list)))
        print(f"\n*** Source_info items whose responses span MORE THAN ONE split value "
              f"(train/test leakage via shared source): {len(cross_split_sources)} ***")
        if cross_split_sources:
            print(f"  Example (first 3): {cross_split_sources[:3]}")
        else:
            print("  None found: every source_info item's responses all belong to a single split.")

        # Exact response-text overlap across splits.
        text_field = None
        for cand in ("response", "response_text", "text", "generated_text"):
            if cand in response_meta["schema"]["union_keys"]:
                text_field = cand
                break
        if text_field:
            by_split_text: dict = defaultdict(set)
            for r in response_records:
                by_split_text[r.get("split")].add(r.get(text_field))
            splits = list(by_split_text.keys())
            for i in range(len(splits)):
                for j in range(i + 1, len(splits)):
                    a, b = splits[i], splits[j]
                    overlap = by_split_text[a] & by_split_text[b]
                    print(f"Exact '{text_field}' text overlap between split={a!r} and split={b!r}: {len(overlap)}")
                    if overlap:
                        print(f"  Investigating the actual overlapping text(s) before calling this "
                              f"a finding (not just counting it):")
                        for text_val in overlap:
                            matches = [r for r in response_records if r.get(text_field) == text_val]
                            print(f"  Text: {text_val!r}")
                            for m in matches:
                                print(f"    -> id={m.get('id')} source_id={m.get('source_id')} "
                                      f"split={m.get('split')} model={m.get('model')} "
                                      f"quality={m.get('quality')} labels={m.get('labels')}")
        else:
            print("Could not identify the response-text field for cross-split text overlap check.")

        # source_id overlap across splits directly (same source_id appearing
        # attached to responses in different splits - same info as above,
        # reported directly in terms of source_id counts for clarity).
        source_ids_by_split: dict = defaultdict(set)
        for r in response_records:
            source_ids_by_split[r.get("split")].add(r.get(link_field))
        splits = list(source_ids_by_split.keys())
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                a, b = splits[i], splits[j]
                overlap = source_ids_by_split[a] & source_ids_by_split[b]
                print(f"source_id overlap between split={a!r} and split={b!r}: {len(overlap)} "
                      f"(should match the cross-split-source count above)")
    else:
        print("No 'split' field found on response records - cannot perform split-leakage checks.")


def validate_offsets_against_response_text(response_records: list[dict], response_meta: dict) -> None:
    _section("OFFSET VALIDATION (labels vs actual response text)")
    labels_field = response_meta["labels_field"]
    text_field = None
    for cand in ("response", "response_text", "text", "generated_text"):
        if cand in response_meta["schema"]["union_keys"]:
            text_field = cand
            break
    if not labels_field or not text_field:
        print(f"Cannot validate offsets: labels_field={labels_field}, text_field={text_field}")
        return

    total_labels = 0
    exact_match = 0
    mismatch = 0
    out_of_bounds = 0
    mismatch_examples = []
    for r in response_records:
        text = r.get(text_field)
        labels = r.get(labels_field)
        if not isinstance(labels, list) or not isinstance(text, str):
            continue
        for lbl in labels:
            if not isinstance(lbl, dict):
                continue
            start, end = lbl.get("start"), lbl.get("end")
            label_text = lbl.get("text")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            total_labels += 1
            if start < 0 or end > len(text) or start > end:
                out_of_bounds += 1
                continue
            actual_slice = text[start:end]
            if label_text is not None and actual_slice == label_text:
                exact_match += 1
            else:
                mismatch += 1
                if len(mismatch_examples) < 3:
                    mismatch_examples.append({
                        "start": start, "end": end,
                        "expected_text": label_text,
                        "actual_slice": actual_slice,
                    })

    print(f"Total offset-bearing labels checked: {total_labels:,}")
    print(f"Offsets exactly matching label['text'] via response[start:end]: {exact_match:,}")
    print(f"Offsets where response[start:end] != label['text']: {mismatch:,}")
    print(f"Offsets out of bounds or start>end against the actual response text: {out_of_bounds:,}")
    if mismatch_examples:
        print(f"Example mismatches (first 3): {mismatch_examples}")


def main() -> int:
    _section("RAGTRUTH ACQUISITION")
    print("Source: ParticleMedia/RAGTruth (GitHub, main branch) - the official repository for "
          "the RAGTruth paper (\"RAGTruth: A Hallucination Corpus for Developing Trustworthy "
          "Retrieval-Augmented Language Models\"), MIT License (verified before download). "
          "Selected over community Hugging Face mirrors of uncertain fidelity to the original "
          "release.")
    print(f"Files requested: {FILES}")
    print(f"Raw storage directory: {RAW_DIR}")

    paths = acquire()

    response_records, response_fmt = ([], "not_downloaded")
    source_records, source_fmt = ([], "not_downloaded")
    if paths.get("response.jsonl", Path()).exists():
        response_records, response_fmt = _parse_records(paths["response.jsonl"])
    if paths.get("source_info.jsonl", Path()).exists():
        source_records, source_fmt = _parse_records(paths["source_info.jsonl"])

    print(f"\nresponse.jsonl parsed as: {response_fmt}")
    print(f"source_info.jsonl parsed as: {source_fmt}")

    if not response_records or not source_records:
        print("One or both files failed to download/parse - stopping inspection.")
        return 1

    response_meta = inspect_response_file(response_records)
    source_meta = inspect_source_info_file(source_records)

    join_and_leakage_checks(response_records, source_records, response_meta, source_meta)
    validate_offsets_against_response_text(response_records, response_meta)

    _section("SUMMARY")
    print(f"response.jsonl: {response_meta['count']:,} records | duplicates={response_meta['duplicate_records']}")
    print(f"source_info.jsonl: {source_meta['count']:,} records | duplicates={source_meta['count'] and 'see above'}")
    print("\nInspection complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
