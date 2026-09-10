"""Step 5D smoke test for the reranker model (BAAI/bge-reranker-large).

Loads the model exactly as configured in configs/models.yaml, reranks four
query-passage candidates, and reports load/inference diagnostics plus raw
scores. This is a smoke test only - it does not implement the ClaimGuard
reranking module (see PROJECT_PLAN.md Step 10). Loads ONLY the reranker - no
other ClaimGuard model is loaded in this process, no FAISS index is built.

Usage:
    python scripts/smoke_test_reranker.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

QUERY = "What is the capital of France?"
CANDIDATES = {
    "Candidate 1": "Paris is the capital and largest city of France.",
    "Candidate 2": "Berlin is the capital of Germany and a major European city.",
    "Candidate 3": "France is a country in Western Europe known for its culture and history.",
    "Candidate 4": "The Pacific Ocean is the largest and deepest ocean on Earth.",
}

SAFETY_FREE_VRAM_GB = 3.0  # bge-reranker-large is small (~560M); generous margin still enforced


def gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def main() -> int:
    models_cfg = cg_config.load_models_config()
    rr_cfg = models_cfg["reranker"]["primary"]
    model_name = rr_cfg["name"]
    dtype_name = rr_cfg.get("dtype", "float16")
    batch_size = rr_cfg.get("batch_size", 16)
    max_seq_length = rr_cfg.get("max_seq_length", 512)

    print(f"=== ClaimGuard Step 5D smoke test: {model_name} ===")
    print(f"torch: {torch.__version__} | cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available, aborting before load.")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    free_before_gb = gb(free_before)
    print(f"VRAM before load: free={free_before_gb:.2f} GB / total={gb(total):.2f} GB")

    if free_before_gb < SAFETY_FREE_VRAM_GB:
        print(
            f"FAIL: free VRAM ({free_before_gb:.2f} GB) below safety threshold "
            f"({SAFETY_FREE_VRAM_GB} GB) - aborting before load to avoid risking an OOM."
        )
        return 1

    torch_dtype = getattr(torch, dtype_name, torch.float16)

    print(
        f"Loading model: {model_name} (requested dtype={dtype_name}, device=cuda, "
        f"max_length={max_seq_length}, batch_size={batch_size})"
    )
    load_start = time.time()
    try:
        model = CrossEncoder(
            model_name,
            device="cuda",
            max_length=max_seq_length,
            model_kwargs={"torch_dtype": torch_dtype},
        )
    except TypeError:
        print("note: CrossEncoder does not accept model_kwargs in this ST version, "
              "retrying with automodel_args")
        model = CrossEncoder(
            model_name,
            device="cuda",
            max_length=max_seq_length,
            automodel_args={"torch_dtype": torch_dtype},
        )
    load_elapsed = time.time() - load_start
    print(f"Model loaded in {load_elapsed:.1f}s")

    underlying = model.model
    actual_dtype = next(underlying.parameters()).dtype
    if actual_dtype != torch_dtype:
        print(f"note: model loaded in {actual_dtype}, explicitly casting to {torch_dtype}")
        underlying.to(torch_dtype)
        actual_dtype = next(underlying.parameters()).dtype

    n_params = sum(p.numel() for p in underlying.parameters())
    actual_device = next(underlying.parameters()).device

    print(f"Parameter count: {n_params:,} (~{n_params / 1e6:.1f}M)")
    print(f"Model dtype: {actual_dtype}")
    print(f"Model device: {actual_device}")

    # Determine score type from the library's own activation function, rather
    # than assuming - do not label raw logits as probabilities.
    activation_fn = getattr(model, "activation_fn", None)
    activation_name = type(activation_fn).__name__ if activation_fn is not None else "unknown"
    print(f"CrossEncoder default activation_fn: {activation_name}")

    free_after, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"VRAM after load: free={gb(free_after):.2f} GB / total={gb(total):.2f} GB")
    print(f"torch allocated={gb(allocated):.2f} GB | reserved={gb(reserved):.2f} GB")

    labels = list(CANDIDATES.keys())
    pairs = [(QUERY, CANDIDATES[label]) for label in labels]

    print(f"\nQuery: {QUERY!r}")
    print("Running reranker on 4 (query, candidate) pairs (first, individually-timed call)...")
    first_start = time.time()
    default_scores = model.predict([pairs[0]])
    first_elapsed = time.time() - first_start
    print(f"First inference latency (1 pair): {first_elapsed * 1000:.1f} ms")

    print("\nScoring all 4 pairs with the library's default activation...")
    all_default_scores = model.predict(pairs, batch_size=batch_size)

    print("Scoring all 4 pairs again with activation_fn=Identity (guaranteed raw logits)...")
    raw_scores = model.predict(pairs, batch_size=batch_size, activation_fn=torch.nn.Identity())

    scores_are_raw = activation_name in ("Identity", "NoneType", "unknown")
    score_type_label = "RAW SCORE (logit, unbounded)" if scores_are_raw else f"PROBABILITY-LIKE (activation={activation_name})"

    print(f"\nScore type (from model's default_activation_function): {score_type_label}")

    results = []
    for label, score in zip(labels, all_default_scores):
        results.append({"candidate": label, "text": CANDIDATES[label], "score": float(score)})

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)

    print("\n--- Results (sorted by actual reranker score, descending) ---")
    for rank, r in enumerate(ranked, start=1):
        print(f"Rank {rank}: {r['candidate']} | score={r['score']:.4f} | text={r['text']!r}")

    print("\n--- Raw (Identity-activation) scores for comparison ---")
    for label, score in zip(labels, raw_scores):
        print(f"{label}: raw_score={float(score):.4f}")

    # Batch test.
    print("\nRunning explicit batch test (all 4 pairs, single predict call)...")
    batch_start = time.time()
    batch_scores = model.predict(pairs, batch_size=len(pairs))
    batch_elapsed = time.time() - batch_start
    batch_scores_arr = list(batch_scores)
    print(f"Batch output shape: ({len(batch_scores_arr)},)")
    print(f"Batch scores: {[round(float(s), 4) for s in batch_scores_arr]}")
    print(f"Batch inference latency ({len(pairs)} pairs): {batch_elapsed * 1000:.1f} ms")

    peak = torch.cuda.max_memory_allocated()
    free_end, total = torch.cuda.mem_get_info()
    print(f"\nPeak allocated VRAM (this process): {gb(peak):.2f} GB")
    print(f"VRAM after inference (pre-cleanup): free={gb(free_end):.2f} GB / total={gb(total):.2f} GB")

    # Cleanup: release model, empty CUDA cache, before exiting.
    del model, underlying
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_cleanup, total = torch.cuda.mem_get_info()
    delta = free_before_gb - gb(free_cleanup)
    print(f"VRAM after cleanup: free={gb(free_cleanup):.2f} GB / total={gb(total):.2f} GB")
    print(f"Delta vs. pre-load baseline ({free_before_gb:.2f} GB free): {delta:.2f} GB")

    print("\n=== Smoke test complete: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
