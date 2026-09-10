"""Step 19: RAGTruth end-to-end evaluation - the FIRST formal end-to-end
evaluation of the complete, FROZEN ClaimGuard system (Qwen3-8B generator,
BGE embedder, FAISS retrieval corpus, BGE reranker, DeBERTa binary
verifier, Step 16 decision policy, Step 18 correction loop) against the
2,700-response reserved RAGTruth TEST split.

ABSOLUTE RULE: nothing about the frozen configuration is changed based on
these results. Gold RAGTruth labels/spans are read ONLY by this script's
own scoring code, AFTER `run_correction_loop()` returns - never passed
into any inference call. See `tests/test_ragtruth_evaluation.py` for the
structural gold-isolation guard.

Three evaluation conditions per response (a fourth, "A", is procedurally
IDENTICAL to "D" as literally specified - see PROJECT_REPORT.md Step 19
Section 4 for why; its numbers are reported as equal to D's rather than
recomputed):

    B: fresh Qwen3-8B generation from RAGTruth's own prompt, then the
       full bounded correction loop (max_correction_attempts=2).
    C: the ORIGINAL RAGTruth response as the starting candidate, a
       SINGLE verify pass only (max_correction_attempts=0) - isolates
       raw detection behavior with no regeneration.
    D (= A): the ORIGINAL RAGTruth response as the starting candidate,
       full bounded correction loop (max_correction_attempts=2) -
       isolates whether correction improves an already-generated answer.

Usage:
    python scripts/evaluate_ragtruth.py [--pilot N]

`--pilot N` runs only the first N test responses (by response_id, stable
sort) - for a quick correctness/latency check before committing to the
full 2,700-response run. Omit for the full evaluation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.datasets import ragtruth as ragtruth_mod  # noqa: E402
from claimguard.retrieval import index as index_mod  # noqa: E402
from claimguard.reranking import reranker as reranker_mod  # noqa: E402
from claimguard.verification import verify as verify_mod  # noqa: E402
from claimguard.generation import qwen3 as qwen3_mod  # noqa: E402
from claimguard.correction import correction as correction_mod  # noqa: E402
from claimguard.evaluation import ragtruth_eval as eval_mod  # noqa: E402

OUT_DIR = cg_config.resolve_path("data/processed/evaluation/ragtruth")
RESULTS_PATH = OUT_DIR / "ragtruth_eval_results.json"
RESPONSE_JSONL_PATH = OUT_DIR / "ragtruth_response_results.jsonl"
ERROR_ANALYSIS_PATH = OUT_DIR / "ragtruth_error_analysis.json"
MANIFEST_PATH = OUT_DIR / "ragtruth_eval_manifest.json"

SAFETY_FREE_VRAM_GB = 20.0
CONDITIONS = ("B_generated", "C_verify_only", "D_original_full_loop")
ANSWER_PREVIEW_CHARS = 150
N_ERROR_EXAMPLES_PER_CATEGORY = 5
N_REPRODUCIBILITY_SUBSET = 10


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def _attempt_summaries(result) -> list[dict]:
    return [
        {
            "attempt_number": a.attempt_number, "step16_decision": a.step16_decision,
            "mapped_decision": a.mapped_decision, "confidence": a.step16_confidence,
            "num_evidence": len(a.evidence),
            "latency": {
                "generation_seconds": a.latency.generation_seconds,
                "retrieval_seconds": a.latency.retrieval_seconds,
                "reranking_seconds": a.latency.reranking_seconds,
                "verification_seconds": a.latency.verification_seconds,
            },
        }
        for a in result.attempts
    ]


def _condition_record(result) -> dict:
    """Compact, reproducible-without-rerunning-inference summary of one
    `run_correction_loop` result. Stores IDs/scores/short previews, NOT
    full retrieved evidence text or long candidate text (avoids
    unnecessarily duplicating copyrighted source material)."""
    attempt0 = result.attempts[0] if result.attempts else None
    return {
        "status": result.status,
        "final_decision": result.final_decision,
        "termination_reason": result.termination_reason,
        "correction_occurred": result.correction_occurred,
        "total_attempts": result.total_attempts,
        "max_correction_attempts": result.max_correction_attempts,
        "attempt0_mapped_decision": attempt0.mapped_decision if attempt0 else None,
        "attempt0_confidence": attempt0.step16_confidence if attempt0 else None,
        "attempt0_num_evidence": len(attempt0.evidence) if attempt0 else 0,
        "final_answer_preview": (result.final_answer or "")[:ANSWER_PREVIEW_CHARS],
        "total_latency_seconds": result.total_latency_seconds,
        "failure": (
            {"stage": result.failure.stage, "error_type": result.failure.error_type}
            if result.failure else None
        ),
        "attempts": _attempt_summaries(result),
    }


def run_one_response(models, r: dict, pipeline_cfg: dict, policy_cfg: dict, gen_cfg: dict, max_attempts: int) -> dict:
    """Runs Conditions B, C, D for ONE RAGTruth test response. `r` is a
    normalized record from `claimguard.datasets.ragtruth`; only `prompt`
    and `response_text` are ever passed into `run_correction_loop` -
    `has_hallucination`/`labels` are read by the CALLER after this
    function returns, never here."""
    retriever, reranker_model, reranker_batch_size, verifier_model, verifier_tokenizer, \
        generator_model, generator_tokenizer, generator_identifier = models

    prompt = r["prompt"] or ""
    original_response = r["response_text"] or ""

    common_kwargs = dict(
        generator_model_identifier=generator_identifier,
        retrieval_top_k=pipeline_cfg["retrieval_top_k"], rerank_top_n=pipeline_cfg["rerank_top_n"],
        verifier_max_length=pipeline_cfg["verifier_max_length"], max_new_tokens=gen_cfg["max_new_tokens"],
        seed=gen_cfg["seed"], entailment_threshold=policy_cfg["entailment_threshold"],
        contradiction_threshold=policy_cfg["contradiction_threshold"],
    )

    result_b = correction_mod.run_correction_loop(
        retriever, reranker_model, reranker_batch_size, verifier_model, verifier_tokenizer,
        generator_model, generator_tokenizer, prompt, initial_candidate=None,
        max_correction_attempts=max_attempts, **common_kwargs,
    )
    result_c = correction_mod.run_correction_loop(
        retriever, reranker_model, reranker_batch_size, verifier_model, verifier_tokenizer,
        generator_model, generator_tokenizer, prompt, initial_candidate=original_response,
        max_correction_attempts=0, **common_kwargs,
    )
    result_d = correction_mod.run_correction_loop(
        retriever, reranker_model, reranker_batch_size, verifier_model, verifier_tokenizer,
        generator_model, generator_tokenizer, prompt, initial_candidate=original_response,
        max_correction_attempts=max_attempts, **common_kwargs,
    )

    return {
        "response_id": r["response_id"], "source_id": r["source_id"], "task_type": r["task_type"],
        "model": r["model"], "temperature": r["temperature"], "quality": r["quality"],
        # Gold fields - present in the OUTPUT artifact for scoring, never used above this line.
        "gold_has_hallucination": r["has_hallucination"], "gold_n_labels": len(r["labels"]),
        "conditions": {
            "B_generated": _condition_record(result_b),
            "C_verify_only": _condition_record(result_c),
            "D_original_full_loop": _condition_record(result_d),
        },
    }


def main() -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=None, help="Evaluate only the first N test responses.")
    args = parser.parse_args()

    correction_cfg = cg_config.load_correction_baseline_config()
    pipeline_cfg = correction_cfg["pipeline"]
    policy_cfg = correction_cfg["decision_policy"]
    gen_cfg = correction_cfg["generation"]
    max_attempts = correction_cfg["max_correction_attempts"]

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

    _section("LOADING RAGTRUTH TEST SET (verified against the dataset module, not assumed)")
    records = ragtruth_mod.load_normalized()
    ragtruth_mod.assert_no_train_test_leakage(records)
    test_records = ragtruth_mod.get_eval_set(records)
    test_records = sorted(test_records, key=lambda r: r["response_id"])
    print(f"Test set size: {len(test_records)} (expected 2700)")
    print(f"Distinct source_ids: {len(set(r['source_id'] for r in test_records))} (expected 450)")
    if args.pilot:
        test_records = test_records[: args.pilot]
        print(f"PILOT MODE: evaluating only the first {len(test_records)} responses.")

    _section("LOADING ALL FOUR MODELS (once each, no duplicates)")
    t0 = time.time()
    retriever = index_mod.Retriever.from_disk()
    reranker_cfg = reranker_mod.load_reranker_config()
    reranker_model, reranker_info = reranker_mod.load_reranker(reranker_cfg)
    checkpoint_dir = cg_config.resolve_path(correction_cfg["verifier"]["checkpoint_dir"])
    verifier_model, verifier_tokenizer, verifier_info = verify_mod.load_verifier(checkpoint_dir)
    generator_model, generator_tokenizer, generator_info = qwen3_mod.load_generator()
    load_seconds = time.time() - t0
    print(f"All four models loaded in {load_seconds:.1f}s")
    models = (
        retriever, reranker_model, reranker_cfg["batch_size"], verifier_model, verifier_tokenizer,
        generator_model, generator_tokenizer, generator_info["model_identifier"],
    )

    _section(f"RUNNING {len(test_records)} RESPONSES x 3 CONDITIONS (max_correction_attempts={max_attempts})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    loop_start = time.time()
    n = len(test_records)
    with RESPONSE_JSONL_PATH.open("w", encoding="utf-8") as jf:
        for i, r in enumerate(test_records):
            rec = run_one_response(models, r, pipeline_cfg, policy_cfg, gen_cfg, max_attempts)
            jf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            jf.flush()
            if (i + 1) % 25 == 0 or (i + 1) == n:
                elapsed = time.time() - loop_start
                rate = elapsed / (i + 1)
                eta = rate * (n - (i + 1))
                print(f"  ...{i + 1}/{n} responses processed ({elapsed:.1f}s elapsed, "
                      f"{rate:.2f}s/response, ETA {eta / 60:.1f} min)", flush=True)

    total_loop_seconds = time.time() - loop_start
    peak_gpu_gb = gb(torch.cuda.max_memory_allocated())
    print(f"\nTotal loop time: {total_loop_seconds:.1f}s | peak GPU memory: {peak_gpu_gb:.2f} GB")

    _section("REPRODUCIBILITY CHECK (small deterministic subset, Condition B, rerun twice)")
    subset = test_records[:N_REPRODUCIBILITY_SUBSET]
    run1 = [run_one_response(models, r, pipeline_cfg, policy_cfg, gen_cfg, max_attempts) for r in subset]
    run2 = [run_one_response(models, r, pipeline_cfg, policy_cfg, gen_cfg, max_attempts) for r in subset]
    identical = all(
        a["conditions"]["B_generated"]["final_decision"] == b["conditions"]["B_generated"]["final_decision"]
        and a["conditions"]["B_generated"]["final_answer_preview"] == b["conditions"]["B_generated"]["final_answer_preview"]
        and a["conditions"]["B_generated"]["total_attempts"] == b["conditions"]["B_generated"]["total_attempts"]
        for a, b in zip(run1, run2)
    )
    print(f"Reproducibility subset size: {len(subset)} | identical across two full reruns: {identical}")

    _section("SAVING RAW RESULTS (aggregation happens in a separate pass)")
    manifest = {
        "model_identifiers": {
            "generator": generator_info["model_identifier"], "verifier": verifier_info["checkpoint_dir"],
            "reranker": reranker_info["model_name"],
        },
        "frozen_config": {
            "max_correction_attempts": max_attempts, "pipeline": pipeline_cfg,
            "decision_policy": policy_cfg, "generation": gen_cfg,
        },
        "conditions": {
            "B_generated": "Fresh Qwen3-8B generation from RAGTruth's own prompt, then the full bounded correction loop.",
            "C_verify_only": "Original RAGTruth response as starting candidate, single verify pass only (max_correction_attempts=0).",
            "D_original_full_loop": "Original RAGTruth response as starting candidate, full bounded correction loop.",
            "A_note": "Condition A, as literally specified, is procedurally identical to D - see PROJECT_REPORT.md Step 19 Section 4.",
        },
        "n_responses_evaluated": n, "pilot_mode": bool(args.pilot), "pilot_n": args.pilot,
        "model_load_seconds": load_seconds, "total_loop_seconds": total_loop_seconds,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "reproducibility_check": {"subset_size": len(subset), "identical_across_two_reruns": identical},
        "python_version": sys.version, "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Saved {MANIFEST_PATH}")
    print(f"Saved {RESPONSE_JSONL_PATH} ({n} lines)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
