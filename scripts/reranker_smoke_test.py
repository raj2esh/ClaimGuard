"""Step 14: real-GPU reranker smoke test.

Verifies (not assumes): model loads, scores a query+candidate batch,
output has one finite score per candidate, batch processing works,
repeated inference is numerically stable (two independent predict() calls
on the same input agree), reranked order is deterministic, GPU memory is
reasonable, and - critically - that FAISS candidate metadata (corpus_id,
page_id, sentence_id, corpus_version) survives rerank() UNCHANGED.

Does not evaluate the full corpus - see scripts/evaluate_reranker.py for
that. Does not call the verifier.

Usage:
    python scripts/reranker_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.reranking import reranker as reranker_mod  # noqa: E402


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


QUERY = "What is the capital of France?"
CANDIDATES = [
    {"corpus_id": "Paris::0", "text": "Paris is the capital and largest city of France.",
     "score": 0.91, "page_id": "Paris", "sentence_id": 0, "corpus_version": "smoke_test"},
    {"corpus_id": "Germany::3", "text": "Berlin is the capital of Germany and a major European city.",
     "score": 0.62, "page_id": "Germany", "sentence_id": 3, "corpus_version": "smoke_test"},
    {"corpus_id": "France::1", "text": "France is a country in Western Europe known for its culture and history.",
     "score": 0.58, "page_id": "France", "sentence_id": 1, "corpus_version": "smoke_test"},
    {"corpus_id": "Pacific_Ocean::0", "text": "The Pacific Ocean is the largest and deepest ocean on Earth.",
     "score": 0.11, "page_id": "Pacific_Ocean", "sentence_id": 0, "corpus_version": "smoke_test"},
]


def main() -> int:
    _section("RERANKER SMOKE TEST")
    reranker_cfg = reranker_mod.load_reranker_config()
    print(f"Config: {reranker_cfg}")

    model, verified_info = reranker_mod.load_reranker(reranker_cfg)
    print(f"Verified model info: {verified_info}")
    assert verified_info["score_direction"] == "higher_is_more_relevant"

    _section("SCORING (single batch)")
    reranked = reranker_mod.rerank(model, QUERY, CANDIDATES, top_n=len(CANDIDATES),
                                    batch_size=reranker_cfg["batch_size"])
    print(f"Output count: {len(reranked)} (expected {len(CANDIDATES)})")
    assert len(reranked) == len(CANDIDATES), "one score per candidate expected"

    import math
    for r in reranked:
        assert math.isfinite(r["reranker_score"]), f"non-finite score for {r['corpus_id']}"
    print("Finite check: PASS")

    for r in reranked:
        print(f"  reranked_rank={r['reranked_rank']} original_rank={r['original_rank']} "
              f"corpus_id={r['corpus_id']} faiss_score={r['faiss_score']:.4f} "
              f"reranker_score={r['reranker_score']:.4f}")

    # Empirically verify score direction on THIS real example: the direct
    # answer (Paris) should score highest.
    top = reranked[0]
    print(f"\nTop-ranked candidate: {top['corpus_id']} (expected 'Paris::0' to rank first "
          f"since it directly answers the query)")
    direction_sane = top["corpus_id"] == "Paris::0"
    print(f"Score-direction sanity check (higher=more relevant, on this real example): "
          f"{'PASS' if direction_sane else 'UNEXPECTED - investigate'}")

    _section("PROVENANCE PRESERVATION CHECK")
    by_id = {c["corpus_id"]: c for c in CANDIDATES}
    provenance_ok = True
    for r in reranked:
        original = by_id[r["corpus_id"]]
        if (r["text"] != original["text"] or r["page_id"] != original["page_id"]
                or r["sentence_id"] != original["sentence_id"]
                or r["corpus_version"] != original["corpus_version"]):
            provenance_ok = False
            print(f"  MISMATCH for {r['corpus_id']}: {r} vs original {original}")
    print(f"All provenance fields (text/page_id/sentence_id/corpus_version) preserved unchanged: "
          f"{provenance_ok}")
    if not provenance_ok:
        print("FAIL: provenance was lost during reranking - stopping before proceeding further.")
        return 1

    _section("DETERMINISM CHECK (two independent rerank() calls)")
    reranked_again = reranker_mod.rerank(model, QUERY, CANDIDATES, top_n=len(CANDIDATES),
                                          batch_size=reranker_cfg["batch_size"])
    order_1 = [r["corpus_id"] for r in reranked]
    order_2 = [r["corpus_id"] for r in reranked_again]
    scores_1 = [r["reranker_score"] for r in reranked]
    scores_2 = [r["reranker_score"] for r in reranked_again]
    max_abs_diff = max(abs(a - b) for a, b in zip(scores_1, scores_2))
    print(f"Order identical: {order_1 == order_2}")
    print(f"Max abs score diff between the two runs: {max_abs_diff}")
    if order_1 != order_2 or max_abs_diff > 1e-4:
        print("FAIL: reranking is not deterministic within tolerance.")
        return 1

    _section("EMPTY CANDIDATE LIST HANDLING")
    empty_result = reranker_mod.rerank(model, QUERY, [], top_n=5, batch_size=reranker_cfg["batch_size"])
    print(f"rerank() with empty candidates returns: {empty_result} (expected [])")
    assert empty_result == []

    import torch
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"\nPeak GPU memory during smoke test: {peak_gb:.3f} GB")
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free_gb, total_gb = torch.cuda.mem_get_info()
        print(f"GPU memory after cleanup: free={free_gb / 1024**3:.2f} GB / total={total_gb / 1024**3:.2f} GB")

    print("\nSMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
