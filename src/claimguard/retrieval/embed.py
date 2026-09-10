"""ClaimGuard retrieval: corpus embedding via the project's configured BGE model.

Uses ONLY the already-selected and Step 5C-smoke-tested embedding model
(`BAAI/bge-large-en-v1.5`, read from `configs/models.yaml` - never
hard-coded here) via `sentence-transformers`. Does not introduce a second
embedding model. Loading pattern mirrors `scripts/smoke_test_bge.py`
(Step 5C) for consistency: explicit dtype verification/cast, VRAM safety
check before load.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import config as cg_config

EMBEDDINGS_DIR = cg_config.resolve_path("data/processed/retrieval")
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
CORPUS_IDS_PATH = EMBEDDINGS_DIR / "embedding_corpus_ids.json"

SAFETY_FREE_VRAM_GB = 3.0  # bge-large-en-v1.5 is small (~335M) - generous margin, matches Step 5C


def load_embedding_config(models_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = models_cfg or cg_config.load_models_config()
    return cfg["embedding"]["primary"]


def load_embedding_model(embedding_cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Load the configured SentenceTransformer model and VERIFY (not
    assume) its actual dimension/dtype/device against the config. Returns
    (model, verified_info) where verified_info documents every checked
    fact (Step 13 instruction 9, items A-G).
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if embedding_cfg["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requests device=cuda but CUDA is not available.")
        free_bytes, _total = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        if free_gb < SAFETY_FREE_VRAM_GB:
            raise RuntimeError(
                f"Free VRAM ({free_gb:.2f} GB) below safety threshold "
                f"({SAFETY_FREE_VRAM_GB} GB) - refusing to load before risking an OOM."
            )

    requested_dtype = getattr(torch, embedding_cfg["dtype"])
    model = SentenceTransformer(
        embedding_cfg["name"], device=embedding_cfg["device"],
        model_kwargs={"torch_dtype": requested_dtype},
    )
    underlying = model[0].auto_model
    actual_dtype = next(underlying.parameters()).dtype
    if actual_dtype != requested_dtype:
        underlying.to(requested_dtype)
        actual_dtype = next(underlying.parameters()).dtype

    if hasattr(model, "get_embedding_dimension"):
        actual_dim = model.get_embedding_dimension()
    else:
        actual_dim = model.get_sentence_embedding_dimension()
    actual_device = str(next(underlying.parameters()).device)
    n_params = sum(p.numel() for p in underlying.parameters())

    verified_info = {
        "model_name": embedding_cfg["name"],
        "requested_dtype": embedding_cfg["dtype"],
        "actual_dtype": str(actual_dtype),
        "requested_device": embedding_cfg["device"],
        "actual_device": actual_device,
        "embedding_dim": actual_dim,
        "batch_size": embedding_cfg["batch_size"],
        "normalize_embeddings": embedding_cfg["normalize_embeddings"],
        "max_seq_length": embedding_cfg.get("max_seq_length"),
        "n_params": n_params,
    }
    return model, verified_info


def embed_texts(
    model, texts: list[str], batch_size: int, normalize: bool
) -> np.ndarray:
    """Embed a list of texts, returning a float32 numpy array (FAISS
    requires float32 contiguous arrays regardless of the model's internal
    compute dtype)."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def smoke_test_determinism(model, texts: list[str], batch_size: int, normalize: bool) -> dict[str, Any]:
    """Embed the same texts twice and verify the results are equivalent
    within a tight numerical tolerance - repeatability check (Step 13
    instruction 10)."""
    emb1 = embed_texts(model, texts, batch_size, normalize)
    emb2 = embed_texts(model, texts, batch_size, normalize)
    max_abs_diff = float(np.max(np.abs(emb1 - emb2)))
    allclose = bool(np.allclose(emb1, emb2, atol=1e-5, rtol=1e-4))
    return {
        "max_abs_diff": max_abs_diff,
        "allclose_atol_1e-5_rtol_1e-4": allclose,
        "shape": list(emb1.shape),
    }


def embed_corpus(
    records: list[dict[str, Any]],
    embedding_cfg: dict[str, Any],
    log_every: int = 20000,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Embed every corpus record's text, in deterministic corpus order.

    Never silently skips a record: if a batch fails to embed, it is
    retried at the individual-item level so failures are isolated and
    RECORDED (corpus_id + reason) rather than dropping the whole batch or
    the whole run. Raises if any produced embedding is non-finite (NaN/Inf)
    rather than silently including a corrupted vector.
    """
    model, verified_info = load_embedding_model(embedding_cfg)
    batch_size = embedding_cfg["batch_size"]
    normalize = embedding_cfg["normalize_embeddings"]

    all_embeddings: list[np.ndarray] = []
    ordered_ids: list[str] = []
    failed: list[dict[str, str]] = []

    start = time.time()
    n = len(records)
    for i in range(0, n, batch_size):
        batch = records[i:i + batch_size]
        texts = [r["text"] for r in batch]
        try:
            batch_emb = embed_texts(model, texts, batch_size=len(texts), normalize=normalize)
            all_embeddings.append(batch_emb)
            ordered_ids.extend(r["corpus_id"] for r in batch)
        except Exception as batch_exc:  # noqa: BLE001 - isolate to per-item retry, not a silent skip
            for r in batch:
                try:
                    single_emb = embed_texts(model, [r["text"]], batch_size=1, normalize=normalize)
                    all_embeddings.append(single_emb)
                    ordered_ids.append(r["corpus_id"])
                except Exception as item_exc:  # noqa: BLE001
                    failed.append({
                        "corpus_id": r["corpus_id"],
                        "reason": f"{type(item_exc).__name__}: {item_exc}",
                        "batch_error": f"{type(batch_exc).__name__}: {batch_exc}",
                    })
        if log_every and (i // batch_size) % max(1, log_every // batch_size) == 0:
            print(f"  ...embedded {min(i + batch_size, n):,}/{n:,} records "
                  f"({time.time() - start:.1f}s elapsed)")

    elapsed = time.time() - start
    embeddings = (
        np.concatenate(all_embeddings, axis=0) if all_embeddings
        else np.zeros((0, verified_info["embedding_dim"]), dtype=np.float32)
    )

    if embeddings.shape[0] > 0 and not np.isfinite(embeddings).all():
        n_bad = int((~np.isfinite(embeddings)).any(axis=1).sum())
        raise RuntimeError(
            f"{n_bad} embedding row(s) contain non-finite (NaN/Inf) values - refusing to "
            "silently proceed with a corrupted embedding matrix."
        )

    stats = {
        **verified_info,
        "total_records": n,
        "embedded_count": len(ordered_ids),
        "failed_count": len(failed),
        "failed_records": failed,
        "elapsed_seconds": elapsed,
        "records_per_second": (len(ordered_ids) / elapsed) if elapsed > 0 else None,
        "embeddings_shape": list(embeddings.shape),
        "all_finite": True,
    }
    return embeddings, ordered_ids, stats


def save_embeddings(embeddings: np.ndarray, corpus_ids: list[str], out_dir: Path = EMBEDDINGS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with CORPUS_IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(corpus_ids, f)


def load_embeddings(out_dir: Path = EMBEDDINGS_DIR) -> tuple[np.ndarray, list[str]]:
    if not EMBEDDINGS_PATH.exists() or not CORPUS_IDS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found under {out_dir}. Run scripts/embed_retrieval_corpus.py first."
        )
    embeddings = np.load(EMBEDDINGS_PATH)
    with CORPUS_IDS_PATH.open("r", encoding="utf-8") as f:
        corpus_ids = json.load(f)
    return embeddings, corpus_ids
