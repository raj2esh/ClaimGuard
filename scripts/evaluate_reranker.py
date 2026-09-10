"""Step 14: FAISS-only vs. FAISS+BGE-reranker evaluation on the SAME query set.

Computes, for the identical set of FEVER-train-derived eval claims used in
Step 13 (same seed=42, sample_size=2000, same corpus - reconstructed
deterministically and cross-checked against Step 13's saved results):

- FAISS-only Recall@{1,5,10,20} (recomputed fresh here, not just re-read
  from Step 13's file, so this script is self-contained and verifiable).
- FAISS+reranker Recall@{1,5,10,20}, where the reranker reorders EXACTLY
  the same top-20 FAISS candidates per query (never searches the corpus
  itself) - so Recall@20 is a same-set-different-order invariant, checked
  explicitly.
- Rank-shift diagnostics (improved/unchanged/worsened, mean/median rank
  before/after, gold-in-top-20 rate).
- Latency (embedding+FAISS, reranker, end-to-end) and GPU memory.

Never uses RAGTruth or TruthfulQA. Never calls the verifier.

Usage:
    python scripts/evaluate_reranker.py
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.retrieval import eval as eval_mod  # noqa: E402
from claimguard.retrieval import index as index_mod  # noqa: E402
from claimguard.reranking import reranker as reranker_mod  # noqa: E402

RESULTS_PATH = cg_config.resolve_path("data/processed/retrieval/reranker_eval_results.json")
MANIFEST_PATH = cg_config.resolve_path("data/processed/retrieval/reranker_manifest.json")
STEP13_RESULTS_PATH = cg_config.resolve_path("data/processed/retrieval/retrieval_eval_results.json")


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    cfg = cg_config.load_reranker_baseline_config()
    retrieval_top_k = cfg["pipeline"]["retrieval_top_k"]
    rerank_top_n = cfg["pipeline"]["rerank_top_n"]
    eval_seed = cfg["evaluation"]["seed"]
    eval_sample_size = cfg["evaluation"]["sample_size"]

    _section("LOADING RETRIEVER AND RERANKER")
    retriever = index_mod.Retriever.from_disk()
    print(f"FAISS index vector count: {retriever.index.ntotal}")

    reranker_cfg = reranker_mod.load_reranker_config()
    model, reranker_info = reranker_mod.load_reranker(reranker_cfg)
    print(f"Reranker verified info: {json.dumps(reranker_info, indent=2)}")

    _section("BUILDING EVAL SET (same construction as Step 13)")
    corpus_ids_in_corpus = {r["corpus_id"] for r in retriever.row_metadata}
    eval_set = eval_mod.build_eval_set(corpus_ids_in_corpus, seed=eval_seed, sample_size=eval_sample_size)
    print(f"Eval set size: {len(eval_set):,} (seed={eval_seed}, sample_size={eval_sample_size})")

    if STEP13_RESULTS_PATH.exists():
        with STEP13_RESULTS_PATH.open() as f:
            step13_results = json.load(f)
        step13_ids = sorted(c["example_id"] for c in step13_results["retrieval_recall"]["per_claim"])
        this_ids = sorted(item["example_id"] for item in eval_set)
        same_set = step13_ids == this_ids
        print(f"Eval set identical to Step 13's saved eval set: {same_set}")
        if not same_set:
            print("FAIL: eval set does not match Step 13 - refusing to report a comparison "
                  "against a different query set.")
            return 1
    else:
        print("Step 13 results file not found - cannot cross-check identical eval set "
              "(proceeding, but this comparison's provenance is weaker).")

    _section("RUNNING FAISS RETRIEVAL + RERANKING FOR EVERY EVAL CLAIM")
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    faiss_ranks: list[int | None] = []
    reranked_ranks: list[int | None] = []
    per_claim: list[dict] = []
    faiss_latencies_ms: list[float] = []
    rerank_latencies_ms: list[float] = []
    examples_improved, examples_unchanged, examples_worsened = [], [], []

    start = time.time()
    for i, item in enumerate(eval_set):
        relevant = set(item["relevant_corpus_ids"])

        t0 = time.time()
        faiss_candidates = retriever.retrieve(item["claim"], top_k=retrieval_top_k)
        t1 = time.time()
        faiss_latencies_ms.append((t1 - t0) * 1000)

        faiss_ids = [c["corpus_id"] for c in faiss_candidates]
        f_rank = eval_mod.first_hit_rank(faiss_ids, relevant)

        t2 = time.time()
        reranked = reranker_mod.rerank(
            model, item["claim"], faiss_candidates, top_n=len(faiss_candidates),
            batch_size=reranker_cfg["batch_size"],
        )
        t3 = time.time()
        rerank_latencies_ms.append((t3 - t2) * 1000)

        reranked_ids = [c["corpus_id"] for c in reranked]
        r_rank = eval_mod.first_hit_rank(reranked_ids, relevant)

        faiss_ranks.append(f_rank)
        reranked_ranks.append(r_rank)
        per_claim.append({
            "example_id": item["example_id"], "faiss_rank": f_rank, "reranked_rank": r_rank,
            "num_relevant": len(relevant),
        })

        if f_rank is not None and r_rank is not None:
            if r_rank < f_rank and len(examples_improved) < 3:
                examples_improved.append({
                    "example_id": item["example_id"], "claim": item["claim"],
                    "faiss_rank": f_rank, "reranked_rank": r_rank,
                    "top_faiss_text": faiss_candidates[0]["text"],
                    "top_reranked_text": reranked[0]["text"],
                })
            elif r_rank > f_rank and len(examples_worsened) < 3:
                examples_worsened.append({
                    "example_id": item["example_id"], "claim": item["claim"],
                    "faiss_rank": f_rank, "reranked_rank": r_rank,
                    "top_faiss_text": faiss_candidates[0]["text"],
                    "top_reranked_text": reranked[0]["text"],
                })
            elif r_rank == f_rank and len(examples_unchanged) < 3:
                examples_unchanged.append({
                    "example_id": item["example_id"], "claim": item["claim"],
                    "faiss_rank": f_rank, "reranked_rank": r_rank,
                })

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1:,}/{len(eval_set):,} claims processed ({time.time() - start:.1f}s elapsed)")

    total_elapsed = time.time() - start
    peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    print(f"Total evaluation loop time: {total_elapsed:.1f}s | peak GPU memory: {peak_gpu_gb}")

    _section("FAISS-ONLY BASELINE (recomputed fresh)")
    faiss_recall = eval_mod.recall_at_cutoffs(faiss_ranks)
    for k in eval_mod.RECALL_CUTOFFS:
        print(f"  Recall@{k}: {faiss_recall[f'recall_at_{k}']:.4f}")

    _section("FAISS + RERANKER")
    reranked_recall = eval_mod.recall_at_cutoffs(reranked_ranks)
    for k in eval_mod.RECALL_CUTOFFS:
        print(f"  Recall@{k}: {reranked_recall[f'recall_at_{k}']:.4f}")

    invariant_ok = abs(
        faiss_recall[f"recall_at_{retrieval_top_k}"] - reranked_recall[f"recall_at_{retrieval_top_k}"]
    ) < 1e-9 if f"recall_at_{retrieval_top_k}" in faiss_recall else True
    print(f"\nSanity invariant (Recall@{retrieval_top_k} must match exactly - same candidate "
          f"set, only reordered): {'PASS' if invariant_ok else 'FAIL'}")

    _section("ABSOLUTE / RELATIVE IMPROVEMENT")
    improvement = {}
    for k in eval_mod.RECALL_CUTOFFS:
        f = faiss_recall[f"recall_at_{k}"]
        r = reranked_recall[f"recall_at_{k}"]
        abs_delta = r - f
        rel_delta = (abs_delta / f) if f > 0 else None
        improvement[f"recall_at_{k}"] = {"faiss": f, "reranked": r, "absolute_delta": abs_delta, "relative_delta": rel_delta}
        print(f"  Recall@{k}: faiss={f:.4f} reranked={r:.4f} "
              f"delta={abs_delta:+.4f} ({'+' if rel_delta and rel_delta >= 0 else ''}"
              f"{rel_delta * 100 if rel_delta is not None else float('nan'):.1f}%)")

    _section("RANK-SHIFT DIAGNOSTICS")
    gold_in_top_k = sum(1 for r in faiss_ranks if r is not None)
    improved = sum(1 for f, r in zip(faiss_ranks, reranked_ranks) if f is not None and r is not None and r < f)
    unchanged_ct = sum(1 for f, r in zip(faiss_ranks, reranked_ranks) if f == r)
    worsened = sum(1 for f, r in zip(faiss_ranks, reranked_ranks) if f is not None and r is not None and r > f)
    newly_lost = sum(1 for f, r in zip(faiss_ranks, reranked_ranks) if f is not None and r is None)
    newly_found = sum(1 for f, r in zip(faiss_ranks, reranked_ranks) if f is None and r is not None)

    faiss_hit_ranks_only = [r for r in faiss_ranks if r is not None]
    reranked_hit_ranks_only = [r for r in reranked_ranks if r is not None]

    diagnostics = {
        "gold_in_top_k_count": gold_in_top_k,
        "gold_in_top_k_rate": gold_in_top_k / len(eval_set) if eval_set else 0.0,
        "improved_count": improved,
        "unchanged_count": unchanged_ct,
        "worsened_count": worsened,
        "newly_lost_count": newly_lost,  # should be 0: same candidate set
        "newly_found_count": newly_found,  # should be 0: same candidate set
        "mean_rank_before": statistics.mean(faiss_hit_ranks_only) if faiss_hit_ranks_only else None,
        "mean_rank_after": statistics.mean(reranked_hit_ranks_only) if reranked_hit_ranks_only else None,
        "median_rank_before": statistics.median(faiss_hit_ranks_only) if faiss_hit_ranks_only else None,
        "median_rank_after": statistics.median(reranked_hit_ranks_only) if reranked_hit_ranks_only else None,
    }
    print(json.dumps(diagnostics, indent=2))
    if newly_lost or newly_found:
        print("WARNING: newly_lost/newly_found should be 0 for a same-candidate-set rerank - "
              "investigate before trusting these results.")

    _section("EXAMPLE INSPECTION")
    print(f"Improved examples ({len(examples_improved)}): {json.dumps(examples_improved, indent=2)}")
    print(f"Unchanged examples ({len(examples_unchanged)}): {json.dumps(examples_unchanged, indent=2)}")
    print(f"Worsened examples ({len(examples_worsened)}): {json.dumps(examples_worsened, indent=2)}")

    _section("LATENCY")
    def _pctl(values: list[float], p: float) -> float:
        s = sorted(values)
        return s[int(len(s) * p)] if s else float("nan")

    end_to_end_ms = [a + b for a, b in zip(faiss_latencies_ms, rerank_latencies_ms)]
    latency = {
        "faiss_avg_ms": statistics.mean(faiss_latencies_ms), "faiss_p50_ms": _pctl(faiss_latencies_ms, 0.5),
        "faiss_p95_ms": _pctl(faiss_latencies_ms, 0.95),
        "reranker_avg_ms": statistics.mean(rerank_latencies_ms), "reranker_p50_ms": _pctl(rerank_latencies_ms, 0.5),
        "reranker_p95_ms": _pctl(rerank_latencies_ms, 0.95),
        "end_to_end_avg_ms": statistics.mean(end_to_end_ms), "end_to_end_p50_ms": _pctl(end_to_end_ms, 0.5),
        "end_to_end_p95_ms": _pctl(end_to_end_ms, 0.95),
        "n_queries_measured": len(eval_set),
        "reranker_candidate_batch_size": retrieval_top_k,
    }
    print(json.dumps(latency, indent=2))

    _section("SAVING RESULTS")
    results = {
        "config": {"retrieval_top_k": retrieval_top_k, "rerank_top_n": rerank_top_n,
                   "eval_seed": eval_seed, "eval_sample_size": eval_sample_size},
        "faiss_only": faiss_recall,
        "faiss_plus_reranker": reranked_recall,
        "improvement": improvement,
        "recall_at_k_invariant_check_passed": invariant_ok,
        "rank_shift_diagnostics": diagnostics,
        "example_inspection": {
            "improved": examples_improved, "unchanged": examples_unchanged, "worsened": examples_worsened,
        },
        "latency": latency,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "per_claim": per_claim,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved evaluation results to {RESULTS_PATH}")

    manifest = {
        "reranker_info": reranker_info,
        "pipeline_config": {"retrieval_top_k": retrieval_top_k, "rerank_top_n": rerank_top_n},
        "evaluation_config": {"seed": eval_seed, "sample_size": eval_sample_size, "source": "fever_train"},
        "python_version": platform.python_version(),
        "torch_version": __import__("torch").__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "index_vector_count": retriever.index.ntotal,
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Saved reranker manifest to {MANIFEST_PATH}")

    if reranked_recall["recall_at_1"] > faiss_recall["recall_at_1"]:
        print("\nCONCLUSION: reranking IMPROVED Recall@1 over FAISS-only on this eval set.")
    elif reranked_recall["recall_at_1"] < faiss_recall["recall_at_1"]:
        print("\nCONCLUSION: reranking DEGRADED Recall@1 relative to FAISS-only on this eval "
              "set - reported honestly, not tuned away in this step.")
    else:
        print("\nCONCLUSION: reranking left Recall@1 UNCHANGED on this eval set.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
