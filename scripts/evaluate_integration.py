"""Step 15: full 2,000-query integrated retrieval + reranking + verifier evaluation.

Uses the IDENTICAL eval set as Steps 13/14 (cross-checked against Step 13's
saved results, not just re-derived with the same parameters). Never uses
RAGTruth, TruthfulQA, or gold evidence/labels to select evidence - gold
information is used ONLY after `run_pipeline()` returns, for evaluation.

Usage:
    python scripts/evaluate_integration.py
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
from claimguard.verification import pipeline as pipeline_mod  # noqa: E402
from claimguard.verification import verify as verify_mod  # noqa: E402

OUT_DIR = cg_config.resolve_path("data/processed/integration")
RESULTS_PATH = OUT_DIR / "integration_eval_results.json"
MANIFEST_PATH = OUT_DIR / "integration_manifest.json"
EXAMPLES_PATH = OUT_DIR / "integration_examples.jsonl"
STEP13_RESULTS_PATH = cg_config.resolve_path("data/processed/retrieval/retrieval_eval_results.json")

N_EXAMPLE_RECORDS_SAVED = 30  # a sample for manual inspection, not all 2000 (avoid bloat)


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _pctl(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[int(len(s) * p)] if s else float("nan")


def main() -> int:
    cfg = cg_config.load_integration_baseline_config()
    retrieval_top_k = cfg["pipeline"]["retrieval_top_k"]
    rerank_top_n = cfg["pipeline"]["rerank_top_n"]
    verifier_max_length = cfg["pipeline"]["verifier_max_length"]
    entailment_threshold = cfg["pipeline"]["entailment_threshold"]
    eval_seed = cfg["evaluation"]["seed"]
    eval_sample_size = cfg["evaluation"]["sample_size"]
    threshold_diag_points = cfg["evaluation"]["threshold_diagnostics"]

    _section("LOADING MODELS")
    t_load_start = time.time()
    retriever = index_mod.Retriever.from_disk()
    reranker_cfg = reranker_mod.load_reranker_config()
    reranker_model, reranker_info = reranker_mod.load_reranker(reranker_cfg)
    checkpoint_dir = cg_config.resolve_path(cfg["verifier"]["checkpoint_dir"])
    verifier_model, verifier_tokenizer, verifier_info = verify_mod.load_verifier(checkpoint_dir)
    model_load_seconds = time.time() - t_load_start
    print(f"All models loaded in {model_load_seconds:.1f}s")
    print(f"Verifier info: {verifier_info}")

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _section("BUILDING EVAL SET (identical to Steps 13/14)")
    corpus_ids_in_corpus = {r["corpus_id"] for r in retriever.row_metadata}
    eval_set = eval_mod.build_eval_set(corpus_ids_in_corpus, seed=eval_seed, sample_size=eval_sample_size)
    print(f"Eval set size: {len(eval_set):,}")

    with STEP13_RESULTS_PATH.open() as f:
        step13_results = json.load(f)
    step13_ids = sorted(c["example_id"] for c in step13_results["retrieval_recall"]["per_claim"])
    this_ids = sorted(item["example_id"] for item in eval_set)
    same_set = step13_ids == this_ids
    print(f"Eval set identical to Step 13's saved eval set: {same_set}")
    if not same_set:
        print("FAIL: eval set mismatch - refusing to report results against a different query set.")
        return 1

    _section("RUNNING INTEGRATED PIPELINE FOR EVERY EVAL CLAIM")
    faiss_ranks: list[int | None] = []
    reranked_ranks: list[int | None] = []
    selected_hit: list[bool] = []
    faiss_top1_hit: list[bool] = []
    reranker_top1_hit: list[bool] = []

    embed_faiss_latencies_ms: list[float] = []
    rerank_latencies_ms: list[float] = []
    verify_latencies_ms: list[float] = []

    true_labels: list[str] = []       # "entailment" (SUPPORTS) / "contradiction" (REFUTES)
    predicted_labels: list[str] = []  # selected evidence's verifier_label

    gold_supported_probs: list[float] = []   # entailment_prob of gold candidates, true label SUPPORTS
    gold_refuted_probs: list[float] = []     # entailment_prob of gold candidates, true label REFUTES
    non_gold_probs: list[float] = []         # entailment_prob of non-gold candidates (either true label)

    saved_examples: list[dict] = []

    start = time.time()
    for i, item in enumerate(eval_set):
        relevant = set(item["relevant_corpus_ids"])
        true_label = "entailment" if item["label"] == "SUPPORTS" else "contradiction"
        true_labels.append(true_label)

        t0 = time.time()
        faiss_candidates = retriever.retrieve(item["claim"], top_k=retrieval_top_k)
        t1 = time.time()
        embed_faiss_latencies_ms.append((t1 - t0) * 1000)

        reranked_candidates = reranker_mod.rerank(
            reranker_model, item["claim"], faiss_candidates, top_n=len(faiss_candidates),
            batch_size=reranker_cfg["batch_size"],
        )
        t2 = time.time()
        rerank_latencies_ms.append((t2 - t1) * 1000)

        top_n_candidates = reranked_candidates[:rerank_top_n]
        verified_candidates = verify_mod.verify_candidates(
            verifier_model, verifier_tokenizer, item["claim"], top_n_candidates, max_length=verifier_max_length,
        )
        t3 = time.time()
        verify_latencies_ms.append((t3 - t2) * 1000)

        faiss_ids = [c["corpus_id"] for c in faiss_candidates]
        reranked_ids = [c["corpus_id"] for c in reranked_candidates]
        f_rank = eval_mod.first_hit_rank(faiss_ids, relevant)
        r_rank = eval_mod.first_hit_rank(reranked_ids, relevant)
        faiss_ranks.append(f_rank)
        reranked_ranks.append(r_rank)

        faiss_top1_hit.append(faiss_candidates[0]["corpus_id"] in relevant if faiss_candidates else False)
        reranker_top1_hit.append(reranked_candidates[0]["corpus_id"] in relevant if reranked_candidates else False)

        selected, support_score = pipeline_mod.select_evidence(verified_candidates)
        is_selected_gold = bool(selected and selected["corpus_id"] in relevant)
        selected_hit.append(is_selected_gold)
        predicted_labels.append(selected["verifier_label"] if selected else "contradiction")

        for c in verified_candidates:
            is_gold = c["corpus_id"] in relevant
            if is_gold:
                if item["label"] == "SUPPORTS":
                    gold_supported_probs.append(c["entailment_probability"])
                else:
                    gold_refuted_probs.append(c["entailment_probability"])
            else:
                non_gold_probs.append(c["entailment_probability"])

        if len(saved_examples) < N_EXAMPLE_RECORDS_SAVED:
            saved_examples.append({
                "example_id": item["example_id"],
                "claim": item["claim"], "true_label": item["label"], "relevant_corpus_ids": sorted(relevant),
                "faiss_rank": f_rank, "reranked_rank": r_rank,
                "selected_evidence": selected, "selected_is_gold": is_selected_gold,
                "final_support_score": support_score,
                "faiss_top1_corpus_id": faiss_candidates[0]["corpus_id"] if faiss_candidates else None,
                "reranker_top1_corpus_id": reranked_candidates[0]["corpus_id"] if reranked_candidates else None,
                "verified_candidates": verified_candidates,
            })

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1:,}/{len(eval_set):,} claims processed ({time.time() - start:.1f}s elapsed)")

    total_loop_seconds = time.time() - start
    peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    print(f"Total loop time: {total_loop_seconds:.1f}s | peak GPU memory: {peak_gpu_gb}")

    _section("A. RETRIEVAL (FAISS-only) - consistency check against Step 13/14")
    faiss_recall = eval_mod.recall_at_cutoffs(faiss_ranks)
    for k in eval_mod.RECALL_CUTOFFS:
        print(f"  Recall@{k}: {faiss_recall[f'recall_at_{k}']:.4f}")

    _section("B. RERANKING - consistency check against Step 14")
    reranked_recall = eval_mod.recall_at_cutoffs(reranked_ranks)
    for k in eval_mod.RECALL_CUTOFFS:
        print(f"  Recall@{k}: {reranked_recall[f'recall_at_{k}']:.4f}")

    _section("C. VERIFIER EVIDENCE SELECTION")
    n = len(eval_set)
    strategy_a = sum(faiss_top1_hit) / n
    strategy_b = sum(reranker_top1_hit) / n
    strategy_c = sum(selected_hit) / n
    print(f"  Strategy A (FAISS top-1) hit rate:        {strategy_a:.4f}")
    print(f"  Strategy B (reranker top-1) hit rate:     {strategy_b:.4f}")
    print(f"  Strategy C (verifier-selected) hit rate:  {strategy_c:.4f}")

    _section("D. VERIFIER PREDICTION (using PIPELINE-SELECTED evidence, not gold)")
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=["entailment", "contradiction"], zero_division=0,
    )
    cm = confusion_matrix(true_labels, predicted_labels, labels=["entailment", "contradiction"])
    print("Positive class = 'entailment' (SUPPORTS / 'supported')")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (entailment): {precision[0]:.4f} | Recall: {recall[0]:.4f} | F1: {f1[0]:.4f}")
    print(f"  Precision (contradiction): {precision[1]:.4f} | Recall: {recall[1]:.4f} | F1: {f1[1]:.4f}")
    print(f"  Confusion matrix [[TP_ent, FN_ent],[FP_ent, TN_ent]] order=[entailment,contradiction]: {cm.tolist()}")
    print("NOTE: this is NOT the Step 12 gold-premise verifier accuracy (0.9585 macro F1) - this "
          "measures end-to-end pipeline accuracy using RETRIEVAL-SELECTED (not gold) evidence, "
          "so retrieval/reranking errors propagate into this number. Not comparable directly.")

    _section("THRESHOLD DIAGNOSTICS (not tuned, informational only)")
    def _diag(probs: list[float], label: str) -> dict:
        if not probs:
            return {"count": 0}
        d = {
            "count": len(probs), "mean": statistics.mean(probs), "median": statistics.median(probs),
        }
        for t in threshold_diag_points:
            d[f"fraction_above_{t}"] = sum(1 for p in probs if p >= t) / len(probs)
        print(f"  {label}: n={d['count']} mean={d['mean']:.4f} median={d['median']:.4f} "
              f"fractions>=thresh={[round(d[f'fraction_above_{t}'], 3) for t in threshold_diag_points]}")
        return d

    threshold_diagnostics = {
        "gold_supported_evidence": _diag(gold_supported_probs, "gold_supported_evidence (true=SUPPORTS)"),
        "gold_refuted_evidence": _diag(gold_refuted_probs, "gold_refuted_evidence (true=REFUTES)"),
        "non_gold_evidence": _diag(non_gold_probs, "non_gold_evidence"),
        "thresholds_checked": threshold_diag_points,
    }

    _section("LATENCY")
    verifier_total_ms = [v for v in verify_latencies_ms]
    end_to_end_ms = [a + b + c for a, b, c in zip(embed_faiss_latencies_ms, rerank_latencies_ms, verify_latencies_ms)]
    latency = {
        "embed_faiss_avg_ms": statistics.mean(embed_faiss_latencies_ms), "embed_faiss_p50_ms": _pctl(embed_faiss_latencies_ms, 0.5),
        "embed_faiss_p95_ms": _pctl(embed_faiss_latencies_ms, 0.95),
        "reranker_avg_ms": statistics.mean(rerank_latencies_ms), "reranker_p50_ms": _pctl(rerank_latencies_ms, 0.5),
        "reranker_p95_ms": _pctl(rerank_latencies_ms, 0.95),
        "verifier_avg_ms": statistics.mean(verifier_total_ms), "verifier_p50_ms": _pctl(verifier_total_ms, 0.5),
        "verifier_p95_ms": _pctl(verifier_total_ms, 0.95),
        "end_to_end_avg_ms": statistics.mean(end_to_end_ms), "end_to_end_p50_ms": _pctl(end_to_end_ms, 0.5),
        "end_to_end_p95_ms": _pctl(end_to_end_ms, 0.95),
        "n_queries_measured": n,
        "verifier_candidates_per_query": rerank_top_n,
        "model_load_seconds_one_time": model_load_seconds,
    }
    print(json.dumps(latency, indent=2))

    _section("SAVING RESULTS")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "retrieval_top_k": retrieval_top_k, "rerank_top_n": rerank_top_n,
            "verifier_max_length": verifier_max_length, "entailment_threshold": entailment_threshold,
            "eval_seed": eval_seed, "eval_sample_size": eval_sample_size,
        },
        "eval_set_identical_to_step13": same_set,
        "retrieval_recall": faiss_recall,
        "reranked_recall": reranked_recall,
        "evidence_selection_comparison": {
            "faiss_top1_hit_rate": strategy_a, "reranker_top1_hit_rate": strategy_b,
            "verifier_selected_hit_rate": strategy_c,
        },
        "verifier_classification": {
            "positive_class": "entailment", "accuracy": accuracy,
            "precision_entailment": precision[0], "recall_entailment": recall[0], "f1_entailment": f1[0],
            "precision_contradiction": precision[1], "recall_contradiction": recall[1], "f1_contradiction": f1[1],
            "confusion_matrix": cm.tolist(), "confusion_matrix_labels": ["entailment", "contradiction"],
            "note": "Uses pipeline-SELECTED (not gold) evidence - not comparable to Step 12's gold-premise dev metrics.",
        },
        "threshold_diagnostics": threshold_diagnostics,
        "latency": latency,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "total_loop_seconds": total_loop_seconds,
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {RESULTS_PATH}")

    manifest = {
        "verifier_info": verifier_info,
        "reranker_info": reranker_info,
        "pipeline_config": cfg["pipeline"],
        "evaluation_config": cfg["evaluation"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "index_vector_count": retriever.index.ntotal,
        "model_load_seconds": model_load_seconds,
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Saved {MANIFEST_PATH}")

    with EXAMPLES_PATH.open("w", encoding="utf-8") as f:
        for ex in saved_examples:
            f.write(json.dumps(ex, ensure_ascii=False, default=str) + "\n")
    print(f"Saved {len(saved_examples)} example records to {EXAMPLES_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
