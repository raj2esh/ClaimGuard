"""Step 13: embedding smoke test + full corpus embedding.

Usage:
    python scripts/embed_retrieval_corpus.py --smoke-test
    python scripts/embed_retrieval_corpus.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.retrieval import corpus as corpus_mod  # noqa: E402
from claimguard.retrieval import embed as embed_mod  # noqa: E402


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run_smoke_test() -> int:
    _section("EMBEDDING SMOKE TEST")
    embedding_cfg = embed_mod.load_embedding_config()
    print(f"Config: {embedding_cfg}")

    model, verified_info = embed_mod.load_embedding_model(embedding_cfg)
    print(f"Verified model info: {json.dumps(verified_info, indent=2)}")

    sample_texts = [
        "Paris is the capital city of France.",
        "The Eiffel Tower is located in Paris.",
        "Bananas are a good source of potassium.",
        "The mitochondria is the powerhouse of the cell.",
    ]
    emb = embed_mod.embed_texts(
        model, sample_texts, batch_size=embedding_cfg["batch_size"],
        normalize=embedding_cfg["normalize_embeddings"],
    )
    print(f"Embedding shape: {emb.shape}")
    assert emb.shape == (len(sample_texts), verified_info["embedding_dim"]), "shape mismatch"
    import numpy as np
    assert np.isfinite(emb).all(), "non-finite values in smoke-test embeddings"
    print("Finite check: PASS")

    if embedding_cfg["normalize_embeddings"]:
        norms = np.linalg.norm(emb, axis=1)
        print(f"L2 norms (should be ~1.0 if normalized): {norms}")
        assert np.allclose(norms, 1.0, atol=1e-3), "normalization check failed"
        print("Normalization check: PASS")

    determinism = embed_mod.smoke_test_determinism(
        model, sample_texts, embedding_cfg["batch_size"], embedding_cfg["normalize_embeddings"]
    )
    print(f"Determinism check (two independent encode calls): {determinism}")
    assert determinism["allclose_atol_1e-5_rtol_1e-4"], "determinism check failed"

    import torch
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"Peak GPU memory during smoke test: {peak_gb:.3f} GB")
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free_gb, total_gb = torch.cuda.mem_get_info()
        print(f"GPU memory after cleanup: free={free_gb / 1024**3:.2f} GB / total={total_gb / 1024**3:.2f} GB")

    print("\nSMOKE TEST: PASS")
    return 0


def run_full_embedding() -> int:
    _section("FULL CORPUS EMBEDDING")
    records = corpus_mod.load_corpus()
    print(f"Loaded {len(records):,} corpus records.")

    embedding_cfg = embed_mod.load_embedding_config()
    embeddings, corpus_ids, stats = embed_mod.embed_corpus(records, embedding_cfg)

    print(f"\nEmbedding stats: {json.dumps({k: v for k, v in stats.items() if k != 'failed_records'}, indent=2)}")
    if stats["failed_count"] > 0:
        print(f"FAILED RECORDS ({stats['failed_count']}): {stats['failed_records']}")
        print("Investigate before proceeding - not silently continuing with a partial corpus.")
        return 1

    embed_mod.save_embeddings(embeddings, corpus_ids)
    print(f"\nSaved embeddings to {embed_mod.EMBEDDINGS_PATH} (shape {embeddings.shape})")
    print(f"Saved corpus_id ordering to {embed_mod.CORPUS_IDS_PATH}")

    # Merge embedding stats into the corpus manifest.
    with corpus_mod.CORPUS_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["embedding"] = {k: v for k, v in stats.items() if k != "failed_records"}
    with corpus_mod.CORPUS_MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Updated corpus manifest with embedding stats: {corpus_mod.CORPUS_MANIFEST_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        return run_smoke_test()
    return run_full_embedding()


if __name__ == "__main__":
    sys.exit(main())
