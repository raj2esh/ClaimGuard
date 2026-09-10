"""Step 15: real end-to-end integration smoke test.

query -> embedding -> FAISS -> reranker -> verifier, on a small
deterministic sample. Verifies (not assumes) correct candidate count,
provenance, finite scores/logits at every stage, probabilities summing to
~1, a selected evidence exists, complete output schema, and reasonable GPU
memory. Does not process the full 2,000-query eval set - see
scripts/evaluate_integration.py for that.

Usage:
    python scripts/integration_smoke_test.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.retrieval import index as index_mod  # noqa: E402
from claimguard.reranking import reranker as reranker_mod  # noqa: E402
from claimguard.verification import pipeline as pipeline_mod  # noqa: E402
from claimguard.verification import verify as verify_mod  # noqa: E402

SAMPLE_QUERIES = [
    "Paris is the capital of France.",
    "The Eiffel Tower was built in Berlin.",
    "Water boils at 100 degrees Celsius at sea level.",
]


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    cfg = cg_config.load_integration_baseline_config()
    retrieval_top_k = cfg["pipeline"]["retrieval_top_k"]
    rerank_top_n = cfg["pipeline"]["rerank_top_n"]
    verifier_max_length = cfg["pipeline"]["verifier_max_length"]
    entailment_threshold = cfg["pipeline"]["entailment_threshold"]

    _section("LOADING MODELS (checkpoint/model loading validation)")
    retriever = index_mod.Retriever.from_disk()
    print(f"FAISS index vector count: {retriever.index.ntotal}")

    reranker_cfg = reranker_mod.load_reranker_config()
    reranker_model, reranker_info = reranker_mod.load_reranker(reranker_cfg)
    print(f"Reranker loaded: {reranker_info['model_name']}, score_direction={reranker_info['score_direction']}")

    checkpoint_dir = cg_config.resolve_path(cfg["verifier"]["checkpoint_dir"])
    verifier_model, verifier_tokenizer, verifier_info = verify_mod.load_verifier(checkpoint_dir)
    print(f"Verifier loaded: {verifier_info}")
    assert verifier_info["num_labels"] == 2, "expected a genuine binary head"

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _section("RUNNING END-TO-END PIPELINE ON SAMPLE QUERIES")
    for query in SAMPLE_QUERIES:
        print(f"\n--- Query: {query!r} ---")
        result = pipeline_mod.run_pipeline(
            retriever, reranker_model, verifier_model, verifier_tokenizer, query,
            retrieval_top_k=retrieval_top_k, rerank_top_n=rerank_top_n,
            reranker_batch_size=reranker_cfg["batch_size"], verifier_max_length=verifier_max_length,
            entailment_threshold=entailment_threshold,
        )

        # Candidate count
        assert len(result.faiss_candidates) == retrieval_top_k, (
            f"expected {retrieval_top_k} FAISS candidates, got {len(result.faiss_candidates)}"
        )
        assert len(result.reranked_candidates) == len(result.faiss_candidates), "reranking must not drop candidates"
        assert len(result.verified_candidates) == min(rerank_top_n, len(result.reranked_candidates)), (
            "expected exactly rerank_top_n verified candidates"
        )
        print(f"Candidate counts: faiss={len(result.faiss_candidates)} "
              f"reranked={len(result.reranked_candidates)} verified={len(result.verified_candidates)}")

        # Provenance + finiteness on every stage
        for c in result.faiss_candidates:
            assert math.isfinite(c["score"]), f"non-finite FAISS score: {c}"
            for field in ("corpus_id", "text", "page_id", "sentence_id", "corpus_version"):
                assert field in c, f"missing provenance field {field!r} in FAISS candidate"
        for c in result.reranked_candidates:
            assert math.isfinite(c["reranker_score"]), f"non-finite reranker score: {c}"
            assert math.isfinite(c["faiss_score"]), f"non-finite faiss_score carried into reranked candidate: {c}"
        for c in result.verified_candidates:
            assert math.isfinite(c["entailment_probability"]), f"non-finite entailment prob: {c}"
            assert math.isfinite(c["contradiction_probability"]), f"non-finite contradiction prob: {c}"
            prob_sum = c["entailment_probability"] + c["contradiction_probability"]
            assert abs(prob_sum - 1.0) < 1e-4, f"probabilities do not sum to ~1: {prob_sum} ({c})"
            for field in ("corpus_id", "text", "page_id", "sentence_id", "faiss_score",
                          "reranker_score", "original_rank", "reranked_rank"):
                assert field in c, f"missing field {field!r} in verified candidate"
        print("Finiteness / provenance / probability-sum checks: PASS")

        assert result.selected_evidence is not None, "selected_evidence must exist for a non-empty candidate list"
        assert result.final_support_score is not None
        print(f"Selected evidence corpus_id={result.selected_evidence['corpus_id']} "
              f"entailment_prob={result.final_support_score:.4f} "
              f"final_supported={result.final_supported} (threshold={entailment_threshold})")
        print(f"FAISS top-1: {result.faiss_top1['corpus_id']} | Reranker top-1: {result.reranker_top1['corpus_id']}")

    _section("EMPTY CANDIDATE / EDGE CASE HANDLING")
    empty_verified = verify_mod.verify_candidates(verifier_model, verifier_tokenizer, "q", [], max_length=verifier_max_length)
    assert empty_verified == []
    empty_selected, empty_score = pipeline_mod.select_evidence([])
    assert empty_selected is None and empty_score is None
    print("Empty-candidate handling: PASS")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\nPeak GPU memory during smoke test (all 3 models loaded simultaneously): {peak_gb:.3f} GB")
        del reranker_model, verifier_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free_gb, total_gb = torch.cuda.mem_get_info()
        print(f"GPU memory after cleanup: free={free_gb / 1024**3:.2f} GB / total={total_gb / 1024**3:.2f} GB")

    print("\nSMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
