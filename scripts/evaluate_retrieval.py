"""Step 13: retrieval-only evaluation (FEVER-derived, RAGTruth/TruthfulQA never used)
plus basic performance measurement (latency, corpus/index size).

Usage:
    python scripts/evaluate_retrieval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.retrieval import corpus as corpus_mod  # noqa: E402
from claimguard.retrieval import eval as eval_mod  # noqa: E402
from claimguard.retrieval import index as index_mod  # noqa: E402

RESULTS_PATH = cg_config.resolve_path("data/processed/retrieval/retrieval_eval_results.json")


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    _section("LOADING INDEX")
    retriever = index_mod.Retriever.from_disk()
    print(f"Index vector count: {retriever.index.ntotal}")

    _section("BUILDING RETRIEVAL EVAL SET (FEVER train only - never RAGTruth/TruthfulQA)")
    corpus_ids_in_corpus = {r["corpus_id"] for r in retriever.row_metadata}
    eval_set = eval_mod.build_eval_set(corpus_ids_in_corpus)
    print(f"Eval set size: {len(eval_set):,} claims (seed={eval_mod.DEFAULT_SEED}, "
          f"target sample_size={eval_mod.DEFAULT_SAMPLE_SIZE})")

    if not eval_set:
        print("No eligible claims found for retrieval evaluation - reporting this limitation "
              "explicitly rather than inventing an evaluation set.")
        recall_results = {"n_eval_claims": 0, "note": "no eligible claims - see log"}
    else:
        _section("RUNNING RETRIEVAL EVALUATION")
        eval_start = time.time()
        recall_results = eval_mod.evaluate_retrieval(retriever, eval_set)
        eval_elapsed = time.time() - eval_start
        print(f"Evaluated {recall_results['n_eval_claims']:,} claims in {eval_elapsed:.1f}s")
        for k in eval_mod.RECALL_CUTOFFS:
            print(f"  Recall@{k}: {recall_results[f'recall_at_{k}']:.4f}")

    _section("PERFORMANCE MEASUREMENT")
    corpus_records = corpus_mod.load_corpus()
    n_queries_for_latency = min(200, len(eval_set)) if eval_set else 0
    latencies_ms = []
    if n_queries_for_latency:
        for item in eval_set[:n_queries_for_latency]:
            t0 = time.time()
            retriever.retrieve(item["claim"], top_k=20)
            latencies_ms.append((time.time() - t0) * 1000)
        latencies_ms.sort()
        p50 = latencies_ms[len(latencies_ms) // 2]
        p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
        avg = sum(latencies_ms) / len(latencies_ms)
    else:
        p50 = p95 = avg = None

    index_size_mb = index_mod.FAISS_INDEX_PATH.stat().st_size / (1024**2)
    metadata_size_mb = index_mod.INDEX_METADATA_PATH.stat().st_size / (1024**2)

    performance = {
        "corpus_size": len(corpus_records),
        "index_vector_count": retriever.index.ntotal,
        "index_disk_size_mb": index_size_mb,
        "metadata_disk_size_mb": metadata_size_mb,
        "latency_n_queries_measured": n_queries_for_latency,
        "latency_avg_ms": avg,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
    }
    print(json.dumps(performance, indent=2))

    results = {"retrieval_recall": recall_results, "performance": performance}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
