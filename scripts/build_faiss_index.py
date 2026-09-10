"""Step 13: build, validate, save, and reload-check the FAISS index.

Usage:
    python scripts/build_faiss_index.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.retrieval import corpus as corpus_mod  # noqa: E402
from claimguard.retrieval import embed as embed_mod  # noqa: E402
from claimguard.retrieval import index as index_mod  # noqa: E402


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    _section("LOADING CORPUS AND EMBEDDINGS")
    records = corpus_mod.load_corpus()
    embeddings, corpus_ids = embed_mod.load_embeddings()
    print(f"Corpus records: {len(records):,}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Corpus IDs (embedding order): {len(corpus_ids):,}")

    if len(records) != len(corpus_ids) or embeddings.shape[0] != len(corpus_ids):
        print(f"FATAL: count mismatch - records={len(records)}, corpus_ids={len(corpus_ids)}, "
              f"embeddings={embeddings.shape[0]}. Refusing to build an inconsistent index.")
        return 1

    import numpy as np
    if not np.isfinite(embeddings).all():
        print("FATAL: embeddings contain non-finite values. Refusing to build index.")
        return 1

    _section("BUILDING FAISS INDEX")
    build_start = time.time()
    index = index_mod.build_index(embeddings)
    build_elapsed = time.time() - build_start
    print(f"Index type: {index_mod.INDEX_TYPE}, metric: {index_mod.METRIC}")
    print(f"Dimension: {index.d}, vector count: {index.ntotal}")
    print(f"Build time: {build_elapsed:.2f}s")

    embedding_cfg = embed_mod.load_embedding_config()
    embedding_info = {**embedding_cfg, "embedding_dim": embeddings.shape[1]}

    _section("SAVING INDEX")
    index_mod.save_index(index, records, corpus_ids, embedding_info)
    index_size_mb = index_mod.FAISS_INDEX_PATH.stat().st_size / (1024**2)
    metadata_size_mb = index_mod.INDEX_METADATA_PATH.stat().st_size / (1024**2)
    print(f"Saved index to {index_mod.FAISS_INDEX_PATH} ({index_size_mb:.1f} MB)")
    print(f"Saved metadata to {index_mod.INDEX_METADATA_PATH} ({metadata_size_mb:.1f} MB)")

    _section("VALIDATING SAVED INDEX (fresh reload)")
    reloaded_index, reloaded_metadata = index_mod.load_index()
    issues = index_mod.validate_index(reloaded_index, reloaded_metadata)
    print(f"Validation issues: {len(issues)}")
    if issues:
        print(f"  sample: {issues[:10]}")
        return 1

    _section("RELOAD CONSISTENCY CHECK")
    # Query the SAME vector both from the in-memory index and the reloaded
    # one, confirm identical top-k results.
    query_vec = embeddings[0:1]
    scores_a, idx_a = index.search(query_vec, 5)
    scores_b, idx_b = reloaded_index.search(query_vec, 5)
    identical = bool(np.array_equal(idx_a, idx_b)) and bool(np.allclose(scores_a, scores_b, atol=1e-5))
    print(f"In-memory vs. reloaded-index top-5 results identical: {identical}")
    if not identical:
        print(f"  in-memory: idx={idx_a} scores={scores_a}")
        print(f"  reloaded:  idx={idx_b} scores={scores_b}")
        return 1

    print("\nFAISS INDEX BUILD + VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
