"""Step 6A: FEVER dataset acquisition and inspection.

Downloads (if not already present) the official FEVER v1.0 dataset from the
Hugging Face `fever/fever` repository (the FEVER project's own org, linked to
https://fever.ai) and writes each split, unmodified, to data/raw/fever/.
Then inspects the REAL downloaded data and reports its actual structure -
splits, schema, label distribution, evidence structure, and integrity
findings. Nothing about the schema is assumed; every claim below is checked
against the real data.

CPU / data-preparation only: does not load any ClaimGuard model, does not
touch the GPU, does not build any retrieval index.

Usage:
    python scripts/inspect_fever.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

RAW_DIR = cg_config.resolve_path("data/raw/fever")
DATASET_REPO = "fever/fever"
CONFIG_NAME = "v1.0"
# NOTE: earlier desk research (fever.ai docs) described a 6-way v1.0 split set
# (train / unlabelled_dev / labelled_dev / paper_dev / unlabelled_test /
# paper_test). The REAL data acquired here overrides that: this repo's script
# loader is blocked under datasets>=4 ("Dataset scripts are no longer
# supported"), and its Hugging Face auto-converted Parquet export (an
# official HF conversion, not a third-party mirror) only exposes a "default"
# config with 3 splits: train / validation / test. We acquire what actually
# exists rather than the names assumed before inspection.
SPLITS_TO_ACQUIRE = ["train", "validation", "test"]
EXPECTED_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
# Empirically confirmed from the real data: "no evidence" (NOT ENOUGH INFO)
# rows use sentinel values, NOT null/None - evidence_wiki_url="" and
# evidence_id/evidence_sentence_id=-1. Treating None as the "missing" marker
# (a reasonable a priori assumption) produced a false-positive "malformed
# evidence" count equal to the entire NOT ENOUGH INFO population on the first
# run; this constant fixes that.
NO_EVIDENCE_WIKI_URL = ""
NO_EVIDENCE_SENTINEL_ID = -1


def _has_evidence(row: dict) -> bool:
    url = row.get("evidence_wiki_url")
    return url is not None and url != NO_EVIDENCE_WIKI_URL


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def acquire():
    """Download each split if not already saved under data/raw/fever/."""
    import datasets

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    loaded = {}
    sources = {}

    for split in SPLITS_TO_ACQUIRE:
        raw_path = RAW_DIR / f"{split}.jsonl"
        if raw_path.exists():
            print(f"[acquire] {split}: already present at {raw_path}, loading from disk (not re-downloading)")
            ds = datasets.load_dataset("json", data_files=str(raw_path), split="train")
            loaded[split] = ds
            sources[split] = f"local raw file (previously downloaded from {DATASET_REPO} {CONFIG_NAME})"
            continue

        print(f"[acquire] {split}: downloading from {DATASET_REPO} (config={CONFIG_NAME})...")
        try:
            ds = datasets.load_dataset(DATASET_REPO, CONFIG_NAME, split=split)
            source_used = f"{DATASET_REPO} ({CONFIG_NAME}) via datasets.load_dataset (script-based loader)"
        except Exception as exc:  # noqa: BLE001
            print(f"[acquire] standard load_dataset failed for split={split}: {exc!r}")
            print("[acquire] falling back to the Hugging Face auto-converted Parquet export "
                  "(refs/convert/parquet) - this is HF's own official conversion, not a "
                  "third-party mirror.")
            try:
                ds = datasets.load_dataset(
                    DATASET_REPO, CONFIG_NAME, split=split, revision="refs/convert/parquet"
                )
                source_used = f"{DATASET_REPO} ({CONFIG_NAME}) via refs/convert/parquet"
            except ValueError as e2:
                print(f"[acquire] config={CONFIG_NAME!r} not present on the parquet export "
                      f"({e2!r}); discovering the actual available config/splits instead of "
                      f"assuming...")
                available_configs = datasets.get_dataset_config_names(
                    DATASET_REPO, revision="refs/convert/parquet"
                )
                print(f"[acquire] available configs on parquet export: {available_configs}")
                fallback_config = "default" if "default" in available_configs else available_configs[0]
                available_splits = datasets.get_dataset_split_names(
                    DATASET_REPO, fallback_config, revision="refs/convert/parquet"
                )
                print(f"[acquire] available splits for config={fallback_config!r}: {available_splits}")
                if split not in available_splits:
                    print(f"[acquire] requested split {split!r} not present under config "
                          f"{fallback_config!r}; skipping this split rather than guessing a "
                          f"substitute.")
                    continue
                ds = datasets.load_dataset(
                    DATASET_REPO, fallback_config, split=split, revision="refs/convert/parquet"
                )
                source_used = f"{DATASET_REPO} (config={fallback_config}) via refs/convert/parquet"

        print(f"[acquire] {split}: {len(ds):,} rows | source={source_used}")
        ds.to_json(str(raw_path))
        print(f"[acquire] {split}: saved raw, unmodified dump -> {raw_path}")
        loaded[split] = ds
        sources[split] = source_used

    return loaded, sources


def inspect_split(split_name: str, ds) -> dict:
    _section(f"SPLIT: {split_name}")
    print(f"Row count: {len(ds):,}")
    print(f"Columns: {ds.column_names}")
    print(f"Features (actual dtypes from the data): {ds.features}")

    cols = ds.column_names
    n = len(ds)

    # Missing-value counts per column (null/None).
    missing = {}
    for col in cols:
        col_values = ds[col]
        missing[col] = sum(1 for v in col_values if v is None)
    print(f"\nMissing values per column (true None/null): {missing}")

    # This schema does NOT use null for "no evidence" - it uses sentinel
    # values. Report those separately so they aren't confused with real nulls.
    if "evidence_wiki_url" in cols:
        sentinel_url = sum(1 for v in ds["evidence_wiki_url"] if v == NO_EVIDENCE_WIKI_URL)
        print(f"Rows with evidence_wiki_url == '' (sentinel 'no evidence' marker, NOT a null): {sentinel_url}")
    if "evidence_id" in cols:
        sentinel_id = sum(1 for v in ds["evidence_id"] if v == NO_EVIDENCE_SENTINEL_ID)
        print(f"Rows with evidence_id == -1 (sentinel marker): {sentinel_id}")

    # Label distribution (if a label column exists and has any non-null values).
    label_counts = Counter()
    if "label" in cols:
        label_counts = Counter(v for v in ds["label"] if v is not None)
        print(f"\nLabel distribution (raw column values): {dict(label_counts)}")
        unexpected_labels = set(label_counts.keys()) - EXPECTED_LABELS
        if unexpected_labels:
            print(f"WARNING: unexpected label values found (not in {EXPECTED_LABELS}): {unexpected_labels}")
        else:
            print(f"All observed labels are within the expected FEVER label set: {EXPECTED_LABELS}")
    else:
        print("\nNo 'label' column in this split (expected for blind test splits).")

    # One example per label, printed compactly (not dumping the whole dataset).
    print("\nOne example per label:")
    shown = set()
    for row in ds:
        lbl = row.get("label")
        if lbl and lbl not in shown:
            shown.add(lbl)
            compact = {k: (v if not isinstance(v, str) or len(v) <= 90 else v[:90] + "...") for k, v in row.items()}
            print(f"  [{lbl}] {compact}")
        if shown >= set(label_counts.keys()):
            break

    # Evidence structure analysis: is the data denormalized (one row per
    # evidence line) or nested (one row per claim with a list-valued evidence
    # field)? Determine this empirically from the actual column dtypes/values.
    evidence_related_cols = [c for c in cols if "evidence" in c.lower()]
    print(f"\nEvidence-related columns found: {evidence_related_cols}")

    id_col = "id" if "id" in cols else None
    evidence_structure = {}
    if id_col and "evidence_annotation_id" in cols:
        # Denormalized layout: reconstruct per-claim evidence-set structure by
        # grouping rows on (id, evidence_annotation_id).
        per_claim_annotation_sets = {}
        for row in ds:
            cid = row.get(id_col)
            ann_id = row.get("evidence_annotation_id")
            per_claim_annotation_sets.setdefault(cid, set()).add(ann_id)

        n_unique_claims = len(per_claim_annotation_sets)
        evidence_set_counts = Counter(len(v) for v in per_claim_annotation_sets.values())
        print(f"\nRow count ({n:,}) vs unique claim IDs ({n_unique_claims:,}) "
              f"-> {'DENORMALIZED (one row per evidence line)' if n_unique_claims < n else 'one row per claim'}")
        print(f"Distribution of #evidence-annotation-sets per claim: {dict(evidence_set_counts)}")
        evidence_structure = {
            "layout": "denormalized_per_evidence_line" if n_unique_claims < n else "one_row_per_claim",
            "unique_claims": n_unique_claims,
            "rows": n,
            "evidence_sets_per_claim_distribution": dict(evidence_set_counts),
        }
    elif id_col and evidence_related_cols:
        # Possibly a nested list-valued evidence field - inspect one example's type.
        sample_val = ds[0].get(evidence_related_cols[0])
        print(f"\nSample value of '{evidence_related_cols[0]}' column: {type(sample_val)} -> {sample_val!r}"[:300])
        evidence_structure = {"layout": "unknown_needs_manual_check", "sample_type": str(type(sample_val))}

    # Zero-evidence claims (label == NOT ENOUGH INFO expected to have no evidence).
    # Uses _has_evidence(), which accounts for this schema's real sentinel
    # encoding (evidence_wiki_url="" / evidence_id=-1), not a None check.
    if "evidence_wiki_url" in cols and "label" in cols:
        zero_evidence_labels = Counter(
            row["label"] for row in ds if not _has_evidence(row) and row.get("label")
        )
        print(f"\nRows with NO evidence (sentinel-encoded, not null), by label: {dict(zero_evidence_labels)}")

    return {
        "split": split_name,
        "row_count": n,
        "columns": cols,
        "missing_values": missing,
        "label_distribution": dict(label_counts),
        "evidence_structure": evidence_structure,
    }


def integrity_checks(all_splits: dict) -> dict:
    _section("DATA INTEGRITY CHECKS")
    findings = {}

    # Duplicate full-row check per split (exact duplicate rows).
    for split_name, ds in all_splits.items():
        seen = set()
        dup_count = 0
        for row in ds:
            key = json.dumps(row, sort_keys=True, default=str)
            if key in seen:
                dup_count += 1
            else:
                seen.add(key)
        findings[f"{split_name}_duplicate_rows"] = dup_count
        print(f"[{split_name}] exact duplicate rows: {dup_count}")

    # Missing claims (empty/None claim text).
    for split_name, ds in all_splits.items():
        missing_claims = sum(1 for row in ds if not row.get("claim"))
        findings[f"{split_name}_missing_claim_text"] = missing_claims
        print(f"[{split_name}] rows with missing/empty claim text: {missing_claims}")

    # Invalid labels.
    for split_name, ds in all_splits.items():
        if "label" not in ds.column_names:
            continue
        invalid = sum(1 for row in ds if row.get("label") and row["label"] not in EXPECTED_LABELS)
        findings[f"{split_name}_invalid_labels"] = invalid
        print(f"[{split_name}] rows with a label outside {EXPECTED_LABELS}: {invalid}")

    # Malformed evidence: SUPPORTS/REFUTES rows with NO evidence, or
    # NOT ENOUGH INFO rows WITH evidence (unexpected either way). Uses
    # _has_evidence(), which accounts for the real sentinel encoding
    # ("" / -1) confirmed empirically above - a naive `is None` check here
    # produced a false positive equal to the entire NOT ENOUGH INFO
    # population on the first run against this dataset.
    for split_name, ds in all_splits.items():
        if "label" not in ds.column_names or "evidence_wiki_url" not in ds.column_names:
            continue
        malformed = 0
        for row in ds:
            lbl = row.get("label")
            has_ev = _has_evidence(row)
            if lbl in ("SUPPORTS", "REFUTES") and not has_ev:
                malformed += 1
            elif lbl == "NOT ENOUGH INFO" and has_ev:
                malformed += 1
        findings[f"{split_name}_malformed_evidence"] = malformed
        print(f"[{split_name}] rows with unexpected evidence-vs-label combination: {malformed}")

    # Split contamination: overlap of claim IDs across splits.
    id_sets = {}
    for split_name, ds in all_splits.items():
        if "id" in ds.column_names:
            id_sets[split_name] = set(ds["id"])
    split_names = list(id_sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = id_sets[a] & id_sets[b]
            findings[f"overlap_{a}_vs_{b}"] = len(overlap)
            print(f"Claim-ID overlap between [{a}] and [{b}]: {len(overlap)}")

    return findings


def main() -> int:
    _section("FEVER ACQUISITION")
    print(f"Source: {DATASET_REPO} (config={CONFIG_NAME}) - the FEVER project's own Hugging "
          f"Face org, linked to https://fever.ai")
    print(f"Splits requested: {SPLITS_TO_ACQUIRE}")
    print(f"Raw storage directory: {RAW_DIR}")

    all_splits, sources = acquire()

    per_split_reports = {}
    for split_name, ds in all_splits.items():
        per_split_reports[split_name] = inspect_split(split_name, ds)

    integrity = integrity_checks(all_splits)

    _section("SUMMARY")
    for split_name, rep in per_split_reports.items():
        print(f"{split_name}: {rep['row_count']:,} rows | source: {sources[split_name]}")
    print("\nInspection complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
