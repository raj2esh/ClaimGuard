"""Step 17: real Qwen3-8B generator smoke test.

Loads the approved generator (configs/models.yaml generator.primary,
configs/generator_baseline.yaml generation settings), runs a tiny
hand-written smoke-test fixture (3 generic questions - NOT RAGTruth,
NOT TruthfulQA, NOT FEVER, NOT HaluEval, no gold evidence), checks
determinism by repeating one prompt, measures load/generation latency
and peak GPU memory, and saves a compact JSON artifact.

This is an ENGINEERING smoke test only - factual correctness of the
generated answers is not a research metric and is not scored here.

Usage:
    python scripts/generator_smoke_test.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.generation import qwen3 as qwen3_mod  # noqa: E402

OUT_DIR = cg_config.resolve_path("data/processed/generation")
RESULTS_PATH = OUT_DIR / "generator_smoke_results.json"

SAFETY_FREE_VRAM_GB = 20.0

# Tiny hand-written smoke-test fixture only - see module docstring.
SMOKE_PROMPTS = [
    {"category": "short_factual", "prompt": "What is the capital of France? Answer in one sentence."},
    {"category": "longer_explanatory", "prompt": "Explain in a short paragraph why the sky appears blue during the day."},
    {"category": "structured_response", "prompt": "List three renewable energy sources. One short sentence each."},
]
DETERMINISM_CHECK_PROMPT = SMOKE_PROMPTS[0]["prompt"]


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def main() -> int:
    import time

    import torch

    cfg = cg_config.load_generator_baseline_config()
    gen_settings = cfg["generation"]
    chat_settings = cfg["chat_template"]

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

    _section("LOADING GENERATOR")
    t0 = time.time()
    model, tokenizer, info = qwen3_mod.load_generator()
    load_seconds = time.time() - t0
    print(f"Loaded {info['model_identifier']} in {load_seconds:.1f}s")
    print(json.dumps(info, indent=2))

    free_after_load, total = torch.cuda.mem_get_info()
    print(f"VRAM after load: free={gb(free_after_load):.2f} GB / total={gb(total):.2f} GB")

    _section("SMOKE-TEST GENERATIONS (engineering only, not a research metric)")
    example_outputs = []
    per_prompt_latency = {}
    per_prompt_token_counts = {}
    all_ok = True

    for item in SMOKE_PROMPTS:
        result = qwen3_mod.generate(
            model, tokenizer, item["prompt"], model_identifier=info["model_identifier"],
            max_new_tokens=gen_settings["max_new_tokens"], do_sample=gen_settings["do_sample"],
            temperature=gen_settings["temperature"], top_p=gen_settings["top_p"],
            seed=gen_settings["seed"], enable_thinking=chat_settings["enable_thinking"],
            fallback_on_unsupported_kwarg=chat_settings["fallback_on_unsupported_kwarg"],
        )
        print(f"\n[{item['category']}] prompt: {item['prompt']!r}")
        print(f"  -> {result.generated_text!r}")
        print(f"  latency={result.latency_seconds:.2f}s prompt_tokens={result.prompt_token_count} "
              f"generated_tokens={result.generated_token_count}")

        ok = (
            isinstance(result.generated_text, str) and len(result.generated_text.strip()) > 0
            and item["prompt"] not in result.generated_text
            and result.generated_token_count > 0
        )
        if not ok:
            all_ok = False
            print("  FAIL: output validation failed for this prompt.")

        per_prompt_latency[item["category"]] = result.latency_seconds
        per_prompt_token_counts[item["category"]] = {
            "prompt_tokens": result.prompt_token_count, "generated_tokens": result.generated_token_count,
        }
        example_outputs.append({
            "category": item["category"], "prompt": item["prompt"],
            "generated_text": result.generated_text[:300],  # short only, per instructions
        })

    _section("DETERMINISTIC REPEATABILITY CHECK")
    r1 = qwen3_mod.generate(
        model, tokenizer, DETERMINISM_CHECK_PROMPT, model_identifier=info["model_identifier"],
        max_new_tokens=gen_settings["max_new_tokens"], do_sample=gen_settings["do_sample"],
        temperature=gen_settings["temperature"], top_p=gen_settings["top_p"],
        seed=gen_settings["seed"], enable_thinking=chat_settings["enable_thinking"],
        fallback_on_unsupported_kwarg=chat_settings["fallback_on_unsupported_kwarg"],
    )
    r2 = qwen3_mod.generate(
        model, tokenizer, DETERMINISM_CHECK_PROMPT, model_identifier=info["model_identifier"],
        max_new_tokens=gen_settings["max_new_tokens"], do_sample=gen_settings["do_sample"],
        temperature=gen_settings["temperature"], top_p=gen_settings["top_p"],
        seed=gen_settings["seed"], enable_thinking=chat_settings["enable_thinking"],
        fallback_on_unsupported_kwarg=chat_settings["fallback_on_unsupported_kwarg"],
    )
    identical = r1.generated_text == r2.generated_text
    print(f"Run 1: {r1.generated_text!r}")
    print(f"Run 2: {r2.generated_text!r}")
    print(f"Identical across two runs: {identical}")
    if not identical:
        all_ok = False
        print("FAIL: deterministic prompt produced different outputs across two runs - investigate before "
              "declaring Step 17 complete (do not force determinism by post-processing text).")

    _section("RESOURCE MEASUREMENT")
    peak_gpu_gb = gb(torch.cuda.max_memory_allocated())
    free_end, total = torch.cuda.mem_get_info()
    print(f"Peak allocated VRAM (this process): {peak_gpu_gb:.2f} GB")
    print(f"VRAM after generation: free={gb(free_end):.2f} GB / total={gb(total):.2f} GB")

    _section("RELEASING GPU RESOURCES")
    qwen3_mod.unload_generator(model)
    free_released, total = torch.cuda.mem_get_info()
    print(f"VRAM after unload: free={gb(free_released):.2f} GB / total={gb(total):.2f} GB")

    _section("SAVING RESULTS")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model_identifier": info["model_identifier"],
        "model_class": info["model_class"],
        "tokenizer_class": info["tokenizer_class"],
        "n_params": info["n_params"],
        "dtype": info["dtype"],
        "device": info["device"],
        "generation_config": gen_settings,
        "chat_template_config": chat_settings,
        "chat_template_fallback_triggered": r1.generation_config["chat_template_fallback_triggered"],
        "deterministic_repeatability": {
            "prompt": DETERMINISM_CHECK_PROMPT, "identical_across_two_runs": identical,
        },
        "smoke_test_latency_seconds": per_prompt_latency,
        "generated_token_counts": per_prompt_token_counts,
        "model_load_seconds": load_seconds,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "example_outputs": example_outputs,
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
