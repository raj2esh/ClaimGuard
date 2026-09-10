"""Step 5C smoke test for the embedding model (BAAI/bge-large-en-v1.5).

Loads the model exactly as configured in configs/models.yaml, encodes four test
sentences, and reports load/inference diagnostics plus cosine similarities. This
is a smoke test only - it does not implement the ClaimGuard retrieval module
(see PROJECT_PLAN.md Step 9). Loads ONLY the embedding model - no other model
is loaded in this process, and no FAISS index is built.

Usage:
    python scripts/smoke_test_bge.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

SENTENCES = {
    "A": "Paris is the capital city of France.",
    "B": "France's capital is Paris.",
    "C": "Berlin is the capital of Germany.",
    "D": "Bananas are a good source of potassium.",
}

SAFETY_FREE_VRAM_GB = 3.0  # bge-large-en-v1.5 is small (~335M); generous margin still enforced


def gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    # Embeddings are already L2-normalized, so dot product == cosine similarity.
    return float(torch.dot(a.float(), b.float()).item())


def main() -> int:
    models_cfg = cg_config.load_models_config()
    emb_cfg = models_cfg["embedding"]["primary"]
    model_name = emb_cfg["name"]
    dtype_name = emb_cfg.get("dtype", "float16")
    batch_size = emb_cfg.get("batch_size", 32)
    normalize = emb_cfg.get("normalize_embeddings", True)

    print(f"=== ClaimGuard Step 5C smoke test: {model_name} ===")
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

    print(f"Loading model: {model_name} (requested dtype={dtype_name}, device=cuda, "
          f"normalize_embeddings={normalize}, batch_size={batch_size})")
    load_start = time.time()
    model = SentenceTransformer(model_name, device="cuda", model_kwargs={"torch_dtype": torch_dtype})
    load_elapsed = time.time() - load_start
    print(f"Model loaded in {load_elapsed:.1f}s")

    underlying = model[0].auto_model
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

    free_after, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"VRAM after load: free={gb(free_after):.2f} GB / total={gb(total):.2f} GB")
    print(f"torch allocated={gb(allocated):.2f} GB | reserved={gb(reserved):.2f} GB")

    labels = list(SENTENCES.keys())
    texts = [SENTENCES[k] for k in labels]

    print(f"\nEncoding {len(texts)} sentences...")
    encode_start = time.time()
    embeddings = model.encode(
        texts, normalize_embeddings=normalize, convert_to_tensor=True, batch_size=batch_size
    )
    encode_elapsed = time.time() - encode_start

    emb_dim = embeddings.shape[-1]
    print(f"Embedding dimension (from actual model output): {emb_dim}")
    print(f"Encoding latency ({len(texts)} sentences): {encode_elapsed * 1000:.1f} ms")

    emb = {label: embeddings[i] for i, label in enumerate(labels)}
    sim_ab = cosine_sim(emb["A"], emb["B"])
    sim_ac = cosine_sim(emb["A"], emb["C"])
    sim_ad = cosine_sim(emb["A"], emb["D"])

    print(f"\nA-B similarity: {sim_ab:.4f}")
    print(f"A-C similarity: {sim_ac:.4f}")
    print(f"A-D similarity: {sim_ad:.4f}")

    # Explicit batch test: same four sentences encoded as a single batch call.
    batch_start = time.time()
    batch_embeddings = model.encode(
        texts, normalize_embeddings=normalize, convert_to_tensor=True, batch_size=len(texts)
    )
    batch_elapsed = time.time() - batch_start
    actual_shape = tuple(batch_embeddings.shape)
    expected_shape = (len(texts), emb_dim)
    print(f"\nBatch embedding tensor shape: {actual_shape}")
    print(f"Expected shape (num_sentences, embedding_dim): {expected_shape}")
    print(f"Shape matches expected: {actual_shape == expected_shape}")
    print(f"Batch encoding latency: {batch_elapsed * 1000:.1f} ms")

    peak = torch.cuda.max_memory_allocated()
    free_end, total = torch.cuda.mem_get_info()
    print(f"\nPeak allocated VRAM (this process): {gb(peak):.2f} GB")
    print(f"VRAM after inference (pre-cleanup): free={gb(free_end):.2f} GB / total={gb(total):.2f} GB")

    # Cleanup: release model and embeddings, empty CUDA cache, before exiting.
    del embeddings, batch_embeddings, emb, model, underlying
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
