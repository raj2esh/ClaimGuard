"""Step 21: controlled component ablations on the RAGTruth reserved test
set. Diagnostic experiment, NOT a tuning stage - no threshold/prompt/
generation-parameter search occurs anywhere in this script.

Ablation matrix (see PROJECT_REPORT.md Step 21 Section 3 for the full
table):

    A. Original frozen ClaimGuard    - retrieval + reranker + verifier + decision + correction
    B. No-reranker                   - retrieval (FAISS order, no rerank) + verifier + decision + correction
    C. FAISS top-1 direct            - retrieval top-1 only + verifier + decision + correction
    D. No-correction                 - REUSED from Step 19 (attempt-0 metrics), zero new inference
    E. Verify-only (decision w/o correction, isolated from generator) - REUSED from Step 19
       Condition C_verify_only, zero new inference

Conditions D and E are functionally identical to data ALREADY computed and
validated in Step 19's `ragtruth_eval_results.json` - this script does NOT
recompute them (see Step 21 instructions Section 14: "If two conditions
are functionally identical, do NOT execute duplicate computation").
`scripts/analyze_ablations.py` extracts D/E directly from that file.

For A/B/C, this script runs a SMALL DETERMINISTIC SUBSET of the RAGTruth
test set (not all 2,700 - Step 19 already measured the full-scale cost at
15.27 hours; re-running that 3x here would be the "unnecessary multi-hour
computation" the instructions explicitly warn against). For each response
in the subset:

    1. Generate the candidate answer ONCE (`qwen3_mod.generate`,
       deterministic - reproduces Step 19's exact stored answer for the
       same response_id, used as a consistency cross-check).
    2. Run `run_correction_loop(..., initial_candidate=<that text>,
       evidence_selection_mode=...)` three times - "reranked" (=A),
       "no_reranker" (=B), "faiss_top1" (=C) - holding the STARTING
       candidate text, generator, retrieval depth, decision policy, and
       correction prompt/budget completely fixed. Only the evidence-
       selection stage varies between A/B/C - a true controlled
       component ablation, not a hyperparameter search.

Usage:
    python scripts/run_ablations.py [--subset-size N]
"""
from __future__ import annotations

import argparse
import datetime
import json
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

OUT_DIR = cg_config.resolve_path("data/processed/evaluation/ablations")
RESPONSE_JSONL_PATH = OUT_DIR / "ablation_response_results.jsonl"
MANIFEST_PATH = OUT_DIR / "ablation_run_manifest.json"

SAFETY_FREE_VRAM_GB = 20.0
ANSWER_PREVIEW_CHARS = 150
DEFAULT_SUBSET_SIZE = 300
N_REPRODUCIBILITY_SUBSET = 10
CONDITIONS = ("A_reranked", "B_no_reranker", "C_faiss_top1")
MODE_BY_CONDITION = {"A_reranked": "reranked", "B_no_reranker": "no_reranker", "C_faiss_top1": "faiss_top1"}


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
            "evidence_corpus_ids": [e.corpus_id for e in a.evidence],
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
    attempt0 = result.attempts[0] if result.attempts else None
    return {
        "status": result.status, "final_decision": result.final_decision,
        "termination_reason": result.termination_reason, "correction_occurred": result.correction_occurred,
        "total_attempts": result.total_attempts, "max_correction_attempts": result.max_correction_attempts,
        "attempt0_mapped_decision": attempt0.mapped_decision if attempt0 else None,
        "attempt0_confidence": attempt0.step16_confidence if attempt0 else None,
        "attempt0_num_evidence": len(attempt0.evidence) if attempt0 else 0,
        "final_answer_preview": (result.final_answer or "")[:ANSWER_PREVIEW_CHARS],
        "total_latency_seconds": result.total_latency_seconds,
        "failure": ({"stage": result.failure.stage, "error_type": result.failure.error_type} if result.failure else None),
        "attempts": _attempt_summaries(result),
    }


def run_one_response(models, r: dict, pipeline_cfg: dict, policy_cfg: dict, gen_cfg: dict, max_attempts: int) -> dict:
    """Generates ONE candidate answer, then runs all 3 evidence-selection
    conditions (A/B/C) starting from that SAME text. `r` is a normalized
    RAGTruth record; `has_hallucination`/`labels` are never referenced
    above the return statement."""
    (retriever, reranker_model, reranker_batch_size, verifier_model, verifier_tokenizer,
     generator_model, generator_tokenizer, generator_identifier) = models

    prompt = r["prompt"] or ""

    gen_result = qwen3_mod.generate(
        generator_model, generator_tokenizer, prompt, model_identifier=generator_identifier,
        max_new_tokens=gen_cfg["max_new_tokens"], do_sample=False, seed=gen_cfg["seed"],
    )
    shared_candidate = gen_result.generated_text

    common_kwargs = dict(
        generator_model_identifier=generator_identifier,
        retrieval_top_k=pipeline_cfg["retrieval_top_k"], rerank_top_n=pipeline_cfg["rerank_top_n"],
        verifier_max_length=pipeline_cfg["verifier_max_length"], max_new_tokens=gen_cfg["max_new_tokens"],
        seed=gen_cfg["seed"], entailment_threshold=policy_cfg["entailment_threshold"],
        contradiction_threshold=policy_cfg["contradiction_threshold"], max_correction_attempts=max_attempts,
    )

    conditions_out = {}
    for cond_name, mode in MODE_BY_CONDITION.items():
        result = correction_mod.run_correction_loop(
            retriever, reranker_model, reranker_batch_size, verifier_model, verifier_tokenizer,
            generator_model, generator_tokenizer, prompt, initial_candidate=shared_candidate,
            evidence_selection_mode=mode, **common_kwargs,
        )
        conditions_out[cond_name] = _condition_record(result)

    return {
        "response_id": r["response_id"], "source_id": r["source_id"], "task_type": r["task_type"],
        "model": r["model"], "quality": r["quality"],
        # Gold fields - present in the OUTPUT for scoring only, never used above this line.
        "gold_has_hallucination": r["has_hallucination"], "gold_n_labels": len(r["labels"]),
        "shared_candidate_answer_preview": shared_candidate[:ANSWER_PREVIEW_CHARS],
        "conditions": conditions_out,
    }


def main() -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument("--pilot", type=int, default=None, help="Overrides --subset-size for a quick dry run.")
    args = parser.parse_args()
    subset_size = args.pilot or args.subset_size

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
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    print(f"VRAM before load: free={gb(free_before):.2f} GB / total={gb(total):.2f} GB")
    if gb(free_before) < SAFETY_FREE_VRAM_GB:
        print(f"FAIL: free VRAM ({gb(free_before):.2f} GB) below safety threshold - aborting.")
        return 1

    _section("LOADING RAGTRUTH TEST SET (identical construction to Step 19, not assumed)")
    records = ragtruth_mod.load_normalized()
    ragtruth_mod.assert_no_train_test_leakage(records)
    test_records = ragtruth_mod.get_eval_set(records)
    test_records = sorted(test_records, key=lambda r: r["response_id"])
    print(f"Full test set size: {len(test_records)} (expected 2700)")
    subset = test_records[:subset_size]
    print(f"Ablation subset size: {len(subset)} (first {subset_size} by response_id, stable sort - "
          f"same convention as every prior step's determinism subsets)")

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

    _section(f"RUNNING {len(subset)} RESPONSES x 3 EVIDENCE-SELECTION CONDITIONS (A/B/C)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    loop_start = time.time()
    n = len(subset)
    with RESPONSE_JSONL_PATH.open("w", encoding="utf-8") as jf:
        for i, r in enumerate(subset):
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

    _section("REPRODUCIBILITY CHECK (small deterministic subset, rerun twice)")
    repro_subset = subset[:N_REPRODUCIBILITY_SUBSET]
    run1 = [run_one_response(models, r, pipeline_cfg, policy_cfg, gen_cfg, max_attempts) for r in repro_subset]
    run2 = [run_one_response(models, r, pipeline_cfg, policy_cfg, gen_cfg, max_attempts) for r in repro_subset]
    identical = all(
        a["shared_candidate_answer_preview"] == b["shared_candidate_answer_preview"]
        and all(
            a["conditions"][c]["final_decision"] == b["conditions"][c]["final_decision"]
            and a["conditions"][c]["attempt0_num_evidence"] == b["conditions"][c]["attempt0_num_evidence"]
            for c in CONDITIONS
        )
        for a, b in zip(run1, run2)
    )
    print(f"Reproducibility subset size: {len(repro_subset)} | identical across two full reruns: {identical}")

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
        "ablation_conditions": {
            "A_reranked": "Original frozen ClaimGuard (retrieval+reranker+verifier+decision+correction) - "
                          "this is Step 19's Condition B architecture, run fresh here on the same subset "
                          "for a determinism cross-check against Step 19's full-run stored results.",
            "B_no_reranker": "Reranker DISABLED - FAISS top-k candidates taken directly in FAISS order "
                             "(first rerank_top_n), verified, decided, corrected exactly as in A.",
            "C_faiss_top1": "Only the single top-ranked FAISS candidate is verified (no reranking, no "
                            "top-n selection) - decided, corrected exactly as in A.",
            "D_no_correction_note": "REUSED from Step 19's ragtruth_eval_results.json "
                                     "(B_generated.attempt0_detection_metrics) - zero new inference.",
            "E_verify_only_note": "REUSED from Step 19's ragtruth_eval_results.json "
                                   "(C_verify_only) - zero new inference.",
        },
        "subset_size": n, "full_ragtruth_test_size": len(test_records),
        "subset_selection": "first N test responses by response_id, stable sort",
        "model_load_seconds": load_seconds, "total_loop_seconds": total_loop_seconds,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "reproducibility_check": {"subset_size": len(repro_subset), "identical_across_two_reruns": identical},
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
