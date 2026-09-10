"""ClaimGuard verification: the integrated retrieval + reranking + binary
verifier pipeline (Step 15).

    query -> embed(query) -> FAISS top-K -> reranker -> reranked top-N
          -> verifier(premise=candidate, hypothesis=query) per candidate
          -> aggregation (max entailment probability) -> selected evidence

This is the FIRST integrated retrieval + reranking + verifier experiment.
It is NOT the final ClaimGuard end-to-end evaluation - no generator, no
correction loop, no RAGTruth/TruthfulQA evaluation happens here or
anywhere in this module.

Per Step 14's finding (reranker top-1 is WEAKER than FAISS top-1 on
Recall@1), this pipeline never assumes reranked-rank-1 is the final
evidence: it preserves the full reranked top-N candidate list and lets the
verifier's own entailment probability select the evidence
(`select_evidence` below), while also retaining the FAISS-top-1 and
reranker-top-1 candidates separately so the three selection strategies can
be compared directly (see PROJECT_REPORT.md Step 15, Section 14 of the
step instructions).

Gold labels/evidence are NEVER used to select evidence - only for
evaluation, applied strictly after `run_pipeline()` returns. `run_pipeline`
takes only `query` as its selection input; nothing about gold evidence or
gold labels can reach it structurally (no such parameter exists).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..reranking import reranker as reranker_mod
from . import verify as verify_mod

DEFAULT_ENTAILMENT_THRESHOLD = 0.5  # INITIAL threshold only - never tuned against any benchmark here


@dataclass
class PipelineResult:
    query: str
    faiss_candidates: list[dict[str, Any]]
    reranked_candidates: list[dict[str, Any]]          # full reranked permutation (all retrieval_top_k)
    verified_candidates: list[dict[str, Any]]           # top rerank_top_n candidates, with verifier scores
    faiss_top1: dict[str, Any] | None
    reranker_top1: dict[str, Any] | None
    selected_evidence: dict[str, Any] | None            # verifier-argmax selection (Strategy C)
    final_support_score: float | None                   # max entailment_probability among verified_candidates
    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD
    final_supported: bool | None = None                 # final_support_score >= entailment_threshold


def select_evidence(verified_candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None]:
    """Strategy C: the candidate with the MAXIMUM entailment probability
    among the verifier-scored candidates. Deterministic tie-break: first
    candidate (by input/reranked-rank order) among ties, so repeated runs
    on identical input never produce a different selection."""
    if not verified_candidates:
        return None, None
    best = max(verified_candidates, key=lambda c: (c["entailment_probability"], -c["reranked_rank"]))
    return best, best["entailment_probability"]


def run_pipeline(
    retriever, reranker_model, verifier_model, verifier_tokenizer, query: str,
    retrieval_top_k: int, rerank_top_n: int, reranker_batch_size: int, verifier_max_length: int,
    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
) -> PipelineResult:
    """The full pipeline for one query. `query` is the ONLY selection
    input - no gold-evidence or gold-label parameter exists on this
    function, so there is no structural way for gold information to
    influence evidence selection.
    """
    faiss_candidates = retriever.retrieve(query, top_k=retrieval_top_k)

    # Full reranked permutation of ALL retrieved candidates (never just the
    # top_n slice) - needed to report reranked Recall@k at every cutoff,
    # matching Step 14's methodology exactly.
    reranked_candidates = reranker_mod.rerank(
        reranker_model, query, faiss_candidates, top_n=len(faiss_candidates), batch_size=reranker_batch_size,
    )

    top_n_candidates = reranked_candidates[:rerank_top_n]
    verified_candidates = verify_mod.verify_candidates(
        verifier_model, verifier_tokenizer, query, top_n_candidates, max_length=verifier_max_length,
    )

    selected_evidence, final_support_score = select_evidence(verified_candidates)
    final_supported = (
        final_support_score >= entailment_threshold if final_support_score is not None else None
    )

    return PipelineResult(
        query=query,
        faiss_candidates=faiss_candidates,
        reranked_candidates=reranked_candidates,
        verified_candidates=verified_candidates,
        faiss_top1=faiss_candidates[0] if faiss_candidates else None,
        reranker_top1=reranked_candidates[0] if reranked_candidates else None,
        selected_evidence=selected_evidence,
        final_support_score=final_support_score,
        entailment_threshold=entailment_threshold,
        final_supported=final_supported,
    )
