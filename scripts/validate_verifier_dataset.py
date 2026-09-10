"""Step 11: final verifier training/dev dataset validation gate.

Re-derives the verifier training/dev pool from real data exactly the way
scripts/build_verifier_pool.py does, then checks it against every
established safety/scientific invariant this project depends on. This is
a READ-ONLY check - it does not modify any data, does not train a model,
and does not touch the GPU.

Exits 0 only if every check passes. Exits 1 (loudly, with the failing
check named) if any invariant is violated - in particular, it refuses to
let a future run silently claim "3-way" dataset support while the
'neutral' class is actually empty: `data/processed/dataset_manifest.json`'s
own `verifier_dataset_status.classification_mode` claim is cross-checked
against the classification mode computed HERE, fresh, from the real pool,
and any mismatch is a hard failure.

Usage:
    python scripts/validate_verifier_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.datasets import manifest as mf  # noqa: E402

MANIFEST_PATH = cg_config.resolve_path("data/processed/dataset_manifest.json")

ALLOWED_SOURCE_SUBSET_PAIRS = {
    ("fever", "train"),
    ("halueval", "qa"),
    ("halueval", "dialogue"),
    ("halueval", "summarization"),
}


class ValidationFailure(RuntimeError):
    """Raised (and caught at the top level) when a check fails - carries
    the check name so the failure is loud and specific, never silent."""


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_schema_and_labels(pool: list[mf.PoolRecord]) -> dict[str, Any]:
    issues = []
    for r in pool:
        if r.label not in mf.VERIFIER_LABELS:
            issues.append(f"record {r.source_dataset}/{r.source_id}: invalid label {r.label!r}")
        if not r.hypothesis:
            issues.append(f"record {r.source_dataset}/{r.source_id}: empty hypothesis")
        if r.premise_text_available:
            if not r.premise_text:
                issues.append(
                    f"record {r.source_dataset}/{r.source_id}: premise_text_available=True but "
                    "premise_text is empty/None"
                )
        else:
            if r.premise_text is not None:
                issues.append(
                    f"record {r.source_dataset}/{r.source_id}: premise_text_available=False but "
                    "premise_text is not None (should never happen - every pool record must "
                    "actually have usable premise text to be in the pool at all)"
                )
        if not r.group_key:
            issues.append(f"record {r.source_dataset}/{r.source_id}: empty group_key")
    if issues:
        raise ValidationFailure(f"schema_and_labels: {len(issues)} issue(s), e.g. {issues[:5]}")
    return {"checked": len(pool), "issues": 0}


def check_provenance(pool: list[mf.PoolRecord]) -> dict[str, Any]:
    issues = []
    for r in pool:
        pair = (r.source_dataset, r.source_subset)
        if pair not in ALLOWED_SOURCE_SUBSET_PAIRS:
            issues.append(f"record has disallowed (source_dataset, source_subset)={pair}")
        if r.source_id is None:
            issues.append(f"record from {pair} has source_id=None")
    if issues:
        raise ValidationFailure(f"provenance: {len(issues)} issue(s), e.g. {issues[:5]}")
    return {"checked": len(pool), "allowed_pairs": sorted(ALLOWED_SOURCE_SUBSET_PAIRS)}


def check_train_dev_disjointness(
    train: list[mf.PoolRecord], dev: list[mf.PoolRecord]
) -> dict[str, Any]:
    train_groups = {r.group_key for r in train}
    dev_groups = {r.group_key for r in dev}
    overlap = train_groups & dev_groups
    if overlap:
        raise ValidationFailure(
            f"train_dev_disjointness: {len(overlap)} group_key(s) present in BOTH train and dev, "
            f"e.g. {list(overlap)[:5]}"
        )
    return {"train_groups": len(train_groups), "dev_groups": len(dev_groups), "overlap": 0}


def check_halueval_pair_grouping(train: list[mf.PoolRecord], dev: list[mf.PoolRecord]) -> dict[str, Any]:
    by_split_group: dict[tuple, set[str]] = {}
    for split_name, records in (("train", train), ("dev", dev)):
        for r in records:
            if r.source_dataset != "halueval":
                continue
            by_split_group.setdefault(r.group_key, set()).add(split_name)
    bad = {gk: splits for gk, splits in by_split_group.items() if len(splits) > 1}
    if bad:
        raise ValidationFailure(
            f"halueval_pair_grouping: {len(bad)} group_key(s) split across train AND dev, "
            f"e.g. {list(bad.items())[:5]}"
        )
    return {"halueval_groups_checked": len(by_split_group), "split_pairs": 0}


def check_contamination_exclusion(pool: list[mf.PoolRecord]) -> dict[str, Any]:
    excluded_keys = mf.contamination_exclusion_group_keys()
    pool_keys = {r.group_key for r in pool}
    leaked = pool_keys & excluded_keys
    if leaked:
        raise ValidationFailure(
            f"contamination_exclusion: {len(leaked)} known-contaminated group_key(s) present in "
            f"the pool, e.g. {list(leaked)[:5]}"
        )
    return {"excluded_keys": len(excluded_keys), "leaked_into_pool": 0}


def check_ragtruth_boundary() -> dict[str, Any]:
    for source in ("ragtruth_test", "ragtruth_train"):
        try:
            mf.build_training_pool([source])
        except mf.RoleViolationError:
            continue
        raise ValidationFailure(f"ragtruth_boundary: build_training_pool([{source!r}]) did NOT raise")
    return {"ragtruth_test_blocked": True, "ragtruth_train_blocked": True}


def check_truthfulqa_boundary() -> dict[str, Any]:
    try:
        mf.build_training_pool(["truthfulqa"])
    except mf.RoleViolationError:
        return {"truthfulqa_blocked": True}
    raise ValidationFailure("truthfulqa_boundary: build_training_pool(['truthfulqa']) did NOT raise")


def check_class_coverage_and_manifest_consistency(pool: list[mf.PoolRecord]) -> dict[str, Any]:
    status = mf.verifier_dataset_label_space_status(pool)
    if status["classification_mode"] == mf.CLASSIFICATION_MODE_INVALID:
        raise ValidationFailure(f"class_coverage: INVALID classification mode - {status['rationale']}")

    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open() as f:
            manifest = json.load(f)
        declared = manifest.get("verifier_dataset_status", {}).get("classification_mode")
        if declared is not None and declared != status["classification_mode"]:
            raise ValidationFailure(
                f"class_coverage: manifest declares classification_mode={declared!r} but the "
                f"real, freshly-computed pool is {status['classification_mode']!r} - the "
                "manifest is STALE or WRONG. Re-run scripts/build_dataset_manifest.py. This "
                "check exists specifically to prevent a future run from silently claiming "
                "3-way dataset support while neutral is actually empty."
            )
        if declared == mf.CLASSIFICATION_MODE_THREE_WAY and status["class_counts"].get("neutral", 0) == 0:
            raise ValidationFailure(
                "class_coverage: manifest/pool claims THREE_WAY classification but neutral "
                "count is 0 - refusing to certify this dataset."
            )
    return status


def check_deterministic_counts() -> dict[str, Any]:
    pool_a = mf.build_training_pool()
    pool_b = mf.build_training_pool()
    train_a, dev_a = mf.split_train_dev(pool_a)
    train_b, dev_b = mf.split_train_dev(pool_b)
    keys_a = ([r.group_key for r in train_a], [r.group_key for r in dev_a])
    keys_b = ([r.group_key for r in train_b], [r.group_key for r in dev_b])
    if keys_a != keys_b:
        raise ValidationFailure("deterministic_counts: two independent pool builds produced "
                                 "different train/dev group_key membership or ordering")
    return {"pool_size": len(pool_a), "train": len(train_a), "dev": len(dev_a), "deterministic": True}


def main() -> int:
    _section("VERIFIER DATASET VALIDATION (Step 11)")
    pool = mf.build_training_pool()
    train, dev = mf.split_train_dev(pool)
    print(f"Pool: {len(pool):,} records ({len(train):,} train / {len(dev):,} dev)")

    checks = [
        ("schema_and_labels", lambda: check_schema_and_labels(pool)),
        ("provenance", lambda: check_provenance(pool)),
        ("train_dev_disjointness", lambda: check_train_dev_disjointness(train, dev)),
        ("halueval_pair_grouping", lambda: check_halueval_pair_grouping(train, dev)),
        ("contamination_exclusion", lambda: check_contamination_exclusion(pool)),
        ("ragtruth_boundary", check_ragtruth_boundary),
        ("truthfulqa_boundary", check_truthfulqa_boundary),
        ("class_coverage_and_manifest_consistency",
         lambda: check_class_coverage_and_manifest_consistency(pool)),
        ("deterministic_counts", check_deterministic_counts),
    ]

    results: dict[str, Any] = {}
    failed = []
    for name, fn in checks:
        try:
            result = fn()
            results[name] = {"status": "PASS", "detail": result}
            print(f"  [PASS] {name}: {result}")
        except ValidationFailure as e:
            results[name] = {"status": "FAIL", "detail": str(e)}
            print(f"  [FAIL] {name}: {e}")
            failed.append(name)

    _section("SUMMARY")
    class_status = results.get("class_coverage_and_manifest_consistency", {}).get("detail", {})
    if isinstance(class_status, dict) and "classification_mode" in class_status:
        print(f"Classification mode: {class_status['classification_mode'].upper()}")
        print(f"Class counts: {class_status.get('class_counts')}")

    if failed:
        print(f"\nVALIDATION FAILED: {len(failed)} check(s) failed: {failed}")
        return 1

    print(f"\nALL {len(checks)} CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
