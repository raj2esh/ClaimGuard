"""Step 20: TruthfulQA evaluation of the complete, FROZEN ClaimGuard
system (Qwen3-8B generator, BGE embedder, FAISS retrieval corpus, BGE
reranker, DeBERTa binary verifier, Step 16 decision policy, Step 18
correction loop) against all 790 TruthfulQA questions. TruthfulQA has NO
train/test split (verified against `claimguard.datasets.truthfulqa` -
not assumed) - every question is evaluation-only, and none of it is used
for training/tuning/threshold-selection.

Two INDEPENDENT evaluation tracks per question:

    1. MC track (rigorous, standard, ground-truth-backed): score every
       answer choice in mc0_targets/mc1_targets/mc2_targets by its
       log-likelihood under the FROZEN Qwen3-8B model (a forward pass,
       not generation, not a weight change), then compute MC1/MC2/MC0
       exactly per TruthfulQA's own published definitions. This track
       measures Qwen3's OWN parametric calibration - no retrieval,
       reranking, verification, or correction is involved, because MC
       scoring never produces or needs a free-form answer.

    2. Free-form pipeline track (process behavior only, NOT a
       truthfulness judgment - see module docstring in
       claimguard.evaluation.truthfulqa_eval for why no such judgment is
       computed): run the SAME frozen `run_correction_loop` used in
       Step 19, once per question, and extract:
         Condition A (baseline)      = attempt 0's raw candidate answer
         Condition B (ClaimGuard initial) = attempt 0's verifier decision
         Condition C (ClaimGuard final)   = the loop's final outcome

Gold TruthfulQA answer information (`correct_answers`, `incorrect_answers`,
`best_answer`, `best_incorrect_answer`, and the *labels* inside
mc0/mc1/mc2_targets) is NEVER passed into `generate()`, `retrieve()`,
`rerank()`, `verify_candidates()`, or `run_correction_loop()` - only the
raw `question` text and (for MC scoring) the CHOICE TEXT itself (not its
0/1 label) ever reach the model. Labels are read only by this script's
own MC-scoring aggregation, after each choice's likelihood is computed.

Usage:
    python scripts/evaluate_truthfulqa.py [--pilot N]
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
from claimguard.datasets import truthfulqa as truthfulqa_mod  # noqa: E402
from claimguard.retrieval import index as index_mod  # noqa: E402
from claimguard.reranking import reranker as reranker_mod  # noqa: E402
from claimguard.verification import verify as verify_mod  # noqa: E402
from claimguard.generation import qwen3 as qwen3_mod  # noqa: E402
from claimguard.correction import correction as correction_mod  # noqa: E402

OUT_DIR = cg_config.resolve_path("data/processed/evaluation/truthfulqa")
RESPONSE_JSONL_PATH = OUT_DIR / "truthfulqa_response_results.jsonl"
MANIFEST_PATH = OUT_DIR / "truthfulqa_eval_manifest.json"

SAFETY_FREE_VRAM_GB = 20.0
ANSWER_PREVIEW_CHARS = 150
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


def _free_form_record(result) -> dict:
    """Compact record for the free-form pipeline track. Condition A
    (baseline) = attempt 0's candidate_answer; Condition B (ClaimGuard
    initial) = attempt 0's decision fields; Condition C (ClaimGuard
    final) = the loop's final outcome. No truthful/untruthful judgment -
    process behavior only (see module docstring)."""
    attempt0 = result.attempts[0] if result.attempts else None
    return {
        "status": result.status,
        "condition_a_baseline_answer_preview": (attempt0.candidate_answer[:ANSWER_PREVIEW_CHARS] if attempt0 else None),
        "condition_b_attempt0_step16_decision": attempt0.step16_decision if attempt0 else None,
        "condition_b_attempt0_mapped_decision": attempt0.mapped_decision if attempt0 else None,
        "condition_b_attempt0_confidence": attempt0.step16_confidence if attempt0 else None,
        "condition_b_attempt0_num_evidence": len(attempt0.evidence) if attempt0 else 0,
        "condition_c_final_decision": result.final_decision,
        "condition_c_final_answer_preview": (result.final_answer or "")[:ANSWER_PREVIEW_CHARS],
        "termination_reason": result.termination_reason,
        "correction_occurred": result.correction_occurred,
        "total_attempts": result.total_attempts,
        "max_correction_attempts": result.max_correction_attempts,
        "total_latency_seconds": result.total_latency_seconds,
        "failure": (
            {"stage": result.failure.stage, "error_type": result.failure.error_type}
            if result.failure else None
        ),
        "attempts": _attempt_summaries(result),
    }


def _score_mc_set(generator_model, generator_tokenizer, question: str, targets: dict[str, int] | None) -> dict:
    """Scores every choice in one mc*_targets dict, returns the raw
    per-choice log-likelihoods plus latency. Returns an empty/None
    result if `targets` is missing - never fabricates a score."""
    if not targets:
        return {"choice_log_likelihoods": None, "latency_seconds": 0.0}
    t0 = time.time()
    scores = {}
    for choice in targets:
        result = qwen3_mod.score_choice_log_likelihood(generator_model, generator_tokenizer, question, choice)
        scores[choice] = result["log_likelihood"]
    return {"choice_log_likelihoods": scores, "latency_seconds": time.time() - t0}


def run_one_question(models, r: dict, pipeline_cfg: dict, policy_cfg: dict, gen_cfg: dict, max_attempts: int) -> dict:
    """Runs both evaluation tracks for ONE TruthfulQA question. `r` is a
    normalized record from `claimguard.datasets.truthfulqa`; only
    `question` (free-form track) and CHOICE TEXT (MC track) are ever
    passed into a model call - `correct_answers`/`incorrect_answers`/
    `best_answer`/`best_incorrect_answer` and the 0/1 labels inside
    mc0/mc1/mc2_targets are read by the CALLER after this function
    returns, never here."""
    (retriever, reranker_model, reranker_batch_size, verifier_model, verifier_tokenizer,
     generator_model, generator_tokenizer, generator_identifier) = models

    question = r["question"]

    common_kwargs = dict(
        generator_model_identifier=generator_identifier,
        retrieval_top_k=pipeline_cfg["retrieval_top_k"], rerank_top_n=pipeline_cfg["rerank_top_n"],
        verifier_max_length=pipeline_cfg["verifier_max_length"], max_new_tokens=gen_cfg["max_new_tokens"],
        seed=gen_cfg["seed"], entailment_threshold=policy_cfg["entailment_threshold"],
        contradiction_threshold=policy_cfg["contradiction_threshold"],
    )
    free_form_result = correction_mod.run_correction_loop(
        retriever, reranker_model, reranker_batch_size, verifier_model, verifier_tokenizer,
        generator_model, generator_tokenizer, question, initial_candidate=None,
        max_correction_attempts=max_attempts, **common_kwargs,
    )

    mc0 = _score_mc_set(generator_model, generator_tokenizer, question, r.get("mc0_targets"))
    mc1 = _score_mc_set(generator_model, generator_tokenizer, question, r.get("mc1_targets"))
    mc2 = _score_mc_set(generator_model, generator_tokenizer, question, r.get("mc2_targets"))

    return {
        "question_id": r["question_id"], "category": r["category"], "type": r["type"],
        "question": question, "has_mc_data": r["has_mc_data"],
        # Gold fields - present in the OUTPUT record for scoring only, never used above this line.
        "gold_mc0_targets": r.get("mc0_targets"), "gold_mc1_targets": r.get("mc1_targets"),
        "gold_mc2_targets": r.get("mc2_targets"),
        "free_form": _free_form_record(free_form_result),
        "mc_scores": {
            "mc0": mc0, "mc1": mc1, "mc2": mc2,
        },
    }


def main() -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=None, help="Evaluate only the first N questions.")
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

    _section("LOADING TRUTHFULQA (verified against the dataset module, not assumed)")
    records = truthfulqa_mod.load_normalized()
    dupes = truthfulqa_mod.find_duplicate_questions(records)
    print(f"Question count: {len(records)} (expected 790)")
    categories = sorted({r["category"] for r in records})
    print(f"Distinct categories: {len(categories)} (expected 37)")
    print(f"Duplicate questions: {len(dupes)} (expected 0)")
    if dupes:
        print("FAIL: duplicate questions found - refusing to evaluate against a non-unique question set.")
        return 1
    records = sorted(records, key=lambda r: r["question_id"])
    if args.pilot:
        records = records[: args.pilot]
        print(f"PILOT MODE: evaluating only the first {len(records)} questions.")

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

    _section(f"RUNNING {len(records)} QUESTIONS (max_correction_attempts={max_attempts})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    loop_start = time.time()
    n = len(records)
    with RESPONSE_JSONL_PATH.open("w", encoding="utf-8") as jf:
        for i, r in enumerate(records):
            rec = run_one_question(models, r, pipeline_cfg, policy_cfg, gen_cfg, max_attempts)
            jf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            jf.flush()
            if (i + 1) % 25 == 0 or (i + 1) == n:
                elapsed = time.time() - loop_start
                rate = elapsed / (i + 1)
                eta = rate * (n - (i + 1))
                print(f"  ...{i + 1}/{n} questions processed ({elapsed:.1f}s elapsed, "
                      f"{rate:.2f}s/question, ETA {eta / 60:.1f} min)", flush=True)

    total_loop_seconds = time.time() - loop_start
    peak_gpu_gb = gb(torch.cuda.max_memory_allocated())
    print(f"\nTotal loop time: {total_loop_seconds:.1f}s | peak GPU memory: {peak_gpu_gb:.2f} GB")

    _section("REPRODUCIBILITY CHECK (small deterministic subset, rerun twice)")
    subset = records[:N_REPRODUCIBILITY_SUBSET]
    run1 = [run_one_question(models, r, pipeline_cfg, policy_cfg, gen_cfg, max_attempts) for r in subset]
    run2 = [run_one_question(models, r, pipeline_cfg, policy_cfg, gen_cfg, max_attempts) for r in subset]
    identical = all(
        a["free_form"]["condition_c_final_decision"] == b["free_form"]["condition_c_final_decision"]
        and a["free_form"]["condition_c_final_answer_preview"] == b["free_form"]["condition_c_final_answer_preview"]
        and a["free_form"]["condition_a_baseline_answer_preview"] == b["free_form"]["condition_a_baseline_answer_preview"]
        and a["mc_scores"]["mc1"]["choice_log_likelihoods"] == b["mc_scores"]["mc1"]["choice_log_likelihoods"]
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
        "evaluation_tracks": {
            "mc_track": "MC1/MC2/MC0 via log-likelihood scoring of the frozen Qwen3-8B model - measures the "
                        "generator's own parametric calibration, no retrieval/verifier/correction involved.",
            "free_form_track": "run_correction_loop (Step 18, unchanged) on each question's free-form generation - "
                                "reports ACCEPT/CORRECT/ABSTAIN process behavior only, NOT a truthfulness judgment "
                                "(no approved automatic free-form truthfulness judge exists in this project - see "
                                "claimguard.evaluation.truthfulqa_eval module docstring).",
        },
        "n_questions_evaluated": n, "n_categories": len(categories), "categories": categories,
        "pilot_mode": bool(args.pilot), "pilot_n": args.pilot,
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
