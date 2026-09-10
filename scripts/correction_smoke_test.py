"""Step 18: real end-to-end smoke test for the ClaimGuard correction/
regeneration loop. Loads all four models ONCE (retriever/embedder,
reranker, verifier, generator), runs the bounded correction loop on a
tiny hand-written fixture (NOT RAGTruth, NOT TruthfulQA, NOT FEVER, NOT
HaluEval, no gold evidence, no evaluation-set tuning), checks
determinism by running one query through the WHOLE loop twice, measures
latency and peak/growth GPU memory, and saves a compact JSON artifact.

This is an ENGINEERING smoke test only - it validates that the loop
runs correctly, terminates, and produces well-formed output. It does NOT
measure or claim hallucination-detection research performance.

Usage:
    python scripts/correction_smoke_test.py
"""
from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.retrieval import index as index_mod  # noqa: E402
from claimguard.reranking import reranker as reranker_mod  # noqa: E402
from claimguard.verification import verify as verify_mod  # noqa: E402
from claimguard.generation import qwen3 as qwen3_mod  # noqa: E402
from claimguard.correction import correction as correction_mod  # noqa: E402

OUT_DIR = cg_config.resolve_path("data/processed/correction")
RESULTS_PATH = OUT_DIR / "correction_smoke_results.json"
SAFETY_FREE_VRAM_GB = 20.0

# Tiny hand-written smoke-test fixture only - see module docstring.
SMOKE_QUERIES = [
    "What is the capital of France?",
    "Where was the Eiffel Tower built?",
    "What year did the first person land on the Moon?",
]
DETERMINISM_CHECK_QUERY = SMOKE_QUERIES[0]


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def _attempt_summary(result) -> list[dict]:
    return [
        {
            "attempt_number": a.attempt_number, "step16_decision": a.step16_decision,
            "mapped_decision": a.mapped_decision, "confidence": a.step16_confidence,
            "num_evidence": len(a.evidence), "answer_preview": a.candidate_answer[:150],
        }
        for a in result.attempts
    ]


def main() -> int:
    import torch

    cfg = cg_config.load_correction_baseline_config()
    max_correction_attempts = cfg["max_correction_attempts"]
    pipeline_cfg = cfg["pipeline"]
    policy_cfg = cfg["decision_policy"]
    gen_cfg = cfg["generation"]

    _section("PREFLIGHT")
    print(f"torch: {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available, aborting before load.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    print(f"VRAM before load: free={gb(free_before):.2f} GB / total={gb(total):.2f} GB")
    if gb(free_before) < SAFETY_FREE_VRAM_GB:
        print(f"FAIL: free VRAM ({gb(free_before):.2f} GB) below safety threshold "
              f"({SAFETY_FREE_VRAM_GB} GB) - aborting before load.")
        return 1

    _section("LOADING ALL FOUR MODELS (once each, no duplicates)")
    t0 = time.time()
    retriever = index_mod.Retriever.from_disk()
    reranker_cfg = reranker_mod.load_reranker_config()
    reranker_model, reranker_info = reranker_mod.load_reranker(reranker_cfg)
    checkpoint_dir = cg_config.resolve_path(cfg["verifier"]["checkpoint_dir"])
    verifier_model, verifier_tokenizer, verifier_info = verify_mod.load_verifier(checkpoint_dir)
    generator_model, generator_tokenizer, generator_info = qwen3_mod.load_generator()
    load_seconds = time.time() - t0
    print(f"All four models loaded in {load_seconds:.1f}s")
    print(f"Generator: {generator_info['model_identifier']} ({generator_info['n_params']:,} params)")
    print(f"Verifier: {verifier_info['checkpoint_dir']}")
    print(f"Reranker: {reranker_info['model_name']}")

    free_after_load, total = torch.cuda.mem_get_info()
    print(f"VRAM after loading all models: free={gb(free_after_load):.2f} GB / total={gb(total):.2f} GB")

    def run(query: str):
        return correction_mod.run_correction_loop(
            retriever, reranker_model, reranker_cfg["batch_size"], verifier_model, verifier_tokenizer,
            generator_model, generator_tokenizer, query,
            generator_model_identifier=generator_info["model_identifier"],
            retrieval_top_k=pipeline_cfg["retrieval_top_k"], rerank_top_n=pipeline_cfg["rerank_top_n"],
            verifier_max_length=pipeline_cfg["verifier_max_length"], max_new_tokens=gen_cfg["max_new_tokens"],
            seed=gen_cfg["seed"], entailment_threshold=policy_cfg["entailment_threshold"],
            contradiction_threshold=policy_cfg["contradiction_threshold"],
            max_correction_attempts=max_correction_attempts,
        )

    _section(f"RUNNING CORRECTION LOOP ON {len(SMOKE_QUERIES)} SMOKE QUERIES (max_correction_attempts={max_correction_attempts})")
    query_results = []
    all_ok = True
    gpu_snapshots_gb = []
    for query in SMOKE_QUERIES:
        t_start = time.time()
        result = run(query)
        elapsed = time.time() - t_start
        allocated_now = gb(torch.cuda.memory_allocated())
        gpu_snapshots_gb.append(allocated_now)

        print(f"\nQuery: {query!r}")
        print(f"  final_decision={result.final_decision} status={result.status} "
              f"termination_reason={result.termination_reason} total_attempts={result.total_attempts} "
              f"correction_occurred={result.correction_occurred}")
        print(f"  final_answer: {result.final_answer!r}")
        print(f"  elapsed={elapsed:.2f}s current_allocated_vram={allocated_now:.2f}GB")

        ok = (
            result.status == "completed"
            and result.final_decision in ("ACCEPT", "ABSTAIN")
            and result.total_attempts >= 1
            and result.total_attempts <= 1 + max_correction_attempts
            and (result.final_answer is None or isinstance(result.final_answer, str))
        )
        if not ok:
            all_ok = False
            print("  FAIL: result validation failed for this query.")

        query_results.append({
            "query": query, "final_decision": result.final_decision, "status": result.status,
            "termination_reason": result.termination_reason, "total_attempts": result.total_attempts,
            "correction_occurred": result.correction_occurred,
            "final_answer_preview": (result.final_answer or "")[:200],
            "attempts": _attempt_summary(result), "total_latency_seconds": result.total_latency_seconds,
        })

    _section("GPU MEMORY GROWTH CHECK ACROSS REPEATED ATTEMPTS")
    print(f"Per-query allocated VRAM snapshots (GB): {[round(g, 2) for g in gpu_snapshots_gb]}")
    growth = gpu_snapshots_gb[-1] - gpu_snapshots_gb[0] if len(gpu_snapshots_gb) > 1 else 0.0
    print(f"Growth from first to last query: {growth:.3f} GB")
    no_growth_ok = abs(growth) < 0.5  # small tolerance for allocator fragmentation, not a hard leak
    if not no_growth_ok:
        all_ok = False
        print("FAIL: GPU memory grew noticeably across repeated correction-loop calls - possible leak.")

    _section("DETERMINISTIC REPEATABILITY CHECK (full loop, same query, twice)")
    r1 = run(DETERMINISM_CHECK_QUERY)
    r2 = run(DETERMINISM_CHECK_QUERY)
    identical = (
        r1.final_decision == r2.final_decision and r1.final_answer == r2.final_answer
        and r1.total_attempts == r2.total_attempts
        and [a.mapped_decision for a in r1.attempts] == [a.mapped_decision for a in r2.attempts]
    )
    print(f"Run 1: final_decision={r1.final_decision} final_answer={r1.final_answer!r} attempts={r1.total_attempts}")
    print(f"Run 2: final_decision={r2.final_decision} final_answer={r2.final_answer!r} attempts={r2.total_attempts}")
    print(f"Identical across two full-loop runs: {identical}")
    if not identical:
        all_ok = False
        print("FAIL: deterministic query produced different correction-loop results across two runs - "
              "investigate before declaring Step 18 complete (do not force determinism by post-processing).")

    _section("RESOURCE MEASUREMENT")
    peak_gpu_gb = gb(torch.cuda.max_memory_allocated())
    free_end, total = torch.cuda.mem_get_info()
    print(f"Peak allocated VRAM (this process, all four models + all generations): {peak_gpu_gb:.2f} GB")
    print(f"VRAM after all runs: free={gb(free_end):.2f} GB / total={gb(total):.2f} GB")

    _section("SAVING RESULTS")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model_identifiers": {
            "generator": generator_info["model_identifier"], "verifier": verifier_info["checkpoint_dir"],
            "reranker": reranker_info["model_name"],
        },
        "config": {
            "max_correction_attempts": max_correction_attempts, "pipeline": pipeline_cfg,
            "decision_policy": policy_cfg, "generation": gen_cfg,
        },
        "query_results": query_results,
        "deterministic_repeatability": {
            "query": DETERMINISM_CHECK_QUERY, "identical_across_two_runs": identical,
        },
        "gpu_snapshots_gb_per_query": gpu_snapshots_gb,
        "gpu_growth_gb": growth,
        "model_load_seconds": load_seconds,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "test_result": "PASS" if all_ok else "FAIL",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved {RESULTS_PATH}")
    print(f"\nTEST RESULT: {results['test_result']}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
