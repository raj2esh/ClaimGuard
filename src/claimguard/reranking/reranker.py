"""ClaimGuard reranking: BGE cross-encoder reranking of FAISS candidates (Step 14).

Uses ONLY the already-selected and Step 5D-smoke-tested reranker model
(`BAAI/bge-reranker-large`, read from `configs/models.yaml` - never
hard-coded here) via `sentence_transformers.CrossEncoder`. Does not
introduce a different reranker model.

## Score direction (verified, not assumed)

Step 5D empirically confirmed the model's default `activation_fn` is
`Sigmoid` (probability-like, bounded [0,1]) - HIGHER score means MORE
relevant (confirmed there by the correct-answer candidate scoring highest
by a wide margin). This module's `smoke_test_reranker()` re-verifies this
at runtime every time it is called (reads `model.activation_fn`'s actual
type; does not assume last step's finding still holds for a freshly loaded
model) rather than assuming Step 5D's finding still holds unconditionally.
Uses the model's DEFAULT activation (not forced `Identity()`/raw logits)
for all real reranking, since "reranker score" conventionally means the
calibrated, higher-is-better score a caller would sort by.

## Two-stage pipeline

    query -> BGE embedding -> FAISS top-K -> BGE reranker -> reranked top-N

The reranker NEVER searches the corpus directly - `rerank()` only ever
scores the candidate list it is given (the FAISS top-K), never queries the
index or the corpus itself. Every output record preserves full provenance
from the FAISS result (`corpus_id`, `text`, `page_id`, `sentence_id`,
`corpus_version`) plus the original FAISS score/rank and the new reranker
score/rank - never bare scores.
"""

from __future__ import annotations

from typing import Any

from .. import config as cg_config

SAFETY_FREE_VRAM_GB = 3.0  # bge-reranker-large is small (~560M) - generous margin, matches Step 5D


def load_reranker_config(models_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = models_cfg or cg_config.load_models_config()
    return cfg["reranker"]["primary"]


def load_reranker(reranker_cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Load the configured CrossEncoder and VERIFY (not assume) its actual
    dtype/device/param-count/score-direction against the config. Mirrors
    the Step 5D smoke-test loading pattern (model_kwargs with an
    automodel_args fallback, explicit dtype verification/cast, VRAM safety
    check before load) for consistency.
    """
    import torch
    from sentence_transformers import CrossEncoder

    if reranker_cfg["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requests device=cuda but CUDA is not available.")
        free_bytes, _total = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        if free_gb < SAFETY_FREE_VRAM_GB:
            raise RuntimeError(
                f"Free VRAM ({free_gb:.2f} GB) below safety threshold "
                f"({SAFETY_FREE_VRAM_GB} GB) - refusing to load before risking an OOM."
            )

    model_name = reranker_cfg["name"]
    requested_dtype = getattr(torch, reranker_cfg["dtype"])
    max_length = reranker_cfg["max_seq_length"]
    try:
        model = CrossEncoder(
            model_name, device=reranker_cfg["device"], max_length=max_length,
            model_kwargs={"torch_dtype": requested_dtype},
        )
    except TypeError:
        model = CrossEncoder(
            model_name, device=reranker_cfg["device"], max_length=max_length,
            automodel_args={"torch_dtype": requested_dtype},
        )

    underlying = model.model
    actual_dtype = next(underlying.parameters()).dtype
    if actual_dtype != requested_dtype:
        underlying.to(requested_dtype)
        actual_dtype = next(underlying.parameters()).dtype

    actual_device = str(next(underlying.parameters()).device)
    n_params = sum(p.numel() for p in underlying.parameters())

    activation_fn = getattr(model, "activation_fn", None)
    activation_name = type(activation_fn).__name__ if activation_fn is not None else "unknown"
    scores_are_raw = activation_name in ("Identity", "NoneType", "unknown")
    score_direction = "higher_is_more_relevant"  # true for both Sigmoid-probability and raw-logit CrossEncoder scores here

    verified_info = {
        "model_name": model_name,
        "requested_dtype": reranker_cfg["dtype"],
        "actual_dtype": str(actual_dtype),
        "requested_device": reranker_cfg["device"],
        "actual_device": actual_device,
        "max_length": max_length,
        "batch_size": reranker_cfg["batch_size"],
        "n_params": n_params,
        "activation_fn": activation_name,
        "score_type": "raw_logit" if scores_are_raw else f"probability_like_{activation_name.lower()}",
        "score_direction": score_direction,
    }
    return model, verified_info


def score(model, query: str, candidate_texts: list[str], batch_size: int) -> list[float]:
    """Score (query, candidate) pairs with the model's default (calibrated,
    higher-is-more-relevant) activation. Returns one float per candidate,
    in input order."""
    if not candidate_texts:
        return []
    pairs = [(query, text) for text in candidate_texts]
    scores = model.predict(pairs, batch_size=batch_size)
    return [float(s) for s in scores]


def rerank(
    model, query: str, candidates: list[dict[str, Any]], top_n: int, batch_size: int
) -> list[dict[str, Any]]:
    """Rerank a FAISS candidate list (already-retrieved, in FAISS-score
    order) against `query`, and return the top `top_n` by reranker score.

    Every input candidate is expected to carry the fields a
    `Retriever.retrieve()` result produces: corpus_id, text, score (the
    FAISS inner-product score), page_id, sentence_id, corpus_version.
    Every OUTPUT record preserves all of that (renamed `score` ->
    `faiss_score` to disambiguate from the new `reranker_score`) plus
    `original_rank` (1-based position in the input list) and
    `reranked_rank` (1-based position after sorting by reranker_score,
    descending). Never returns bare (text, score) pairs.

    Empty `candidates` returns `[]` without calling the model. Duplicate
    `corpus_id`s in the input (should not occur from a real FAISS result,
    which is unique by construction) are each scored and ranked
    independently - not merged or deduplicated here, since this function's
    job is to rank exactly the candidates it is given, not to second-guess
    the caller's candidate set.
    """
    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    reranker_scores = score(model, query, texts, batch_size=batch_size)

    enriched = []
    for original_rank, (c, r_score) in enumerate(zip(candidates, reranker_scores), start=1):
        enriched.append({
            "corpus_id": c["corpus_id"],
            "text": c["text"],
            "faiss_score": c.get("score", c.get("faiss_score")),
            "reranker_score": r_score,
            "original_rank": original_rank,
            "page_id": c["page_id"],
            "sentence_id": c["sentence_id"],
            "corpus_version": c.get("corpus_version"),
        })

    enriched.sort(key=lambda r: r["reranker_score"], reverse=True)
    for reranked_rank, r in enumerate(enriched, start=1):
        r["reranked_rank"] = reranked_rank

    return enriched[:top_n]


def retrieve_and_rerank(
    retriever, reranker_model, query: str, retrieval_top_k: int, rerank_top_n: int, batch_size: int
) -> dict[str, Any]:
    """The full two-stage pipeline for one query: FAISS top-K, then
    reranked top-N. Returns both stages' candidate lists so callers can
    compare them directly (never silently discards the pre-rerank order).
    """
    faiss_candidates = retriever.retrieve(query, top_k=retrieval_top_k)
    reranked_candidates = rerank(
        reranker_model, query, faiss_candidates, top_n=rerank_top_n, batch_size=batch_size
    )
    return {
        "query": query,
        "faiss_candidates": faiss_candidates,
        "reranked_candidates": reranked_candidates,
    }
