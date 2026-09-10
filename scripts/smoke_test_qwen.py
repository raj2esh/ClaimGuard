"""Step 5A smoke test for the primary generator model (Qwen3-8B).

Loads the model exactly as configured in configs/models.yaml, runs ONE short
generation, and reports load/generation diagnostics. This is a smoke test
only - it does not build the generation module (see PROJECT_PLAN.md Step 7).

Usage:
    python scripts/smoke_test_qwen.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

PROMPT = "What is the capital of France? Answer in one sentence."
SAFETY_FREE_VRAM_GB = 20.0  # minimum free VRAM required before attempting to load


def gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def main() -> int:
    models_cfg = cg_config.load_models_config()
    gen_cfg = models_cfg["generator"]["primary"]
    model_name = gen_cfg["name"]
    dtype_name = gen_cfg.get("dtype", "bfloat16")
    torch_dtype = getattr(torch, dtype_name)

    print(f"=== ClaimGuard Step 5A smoke test: {model_name} ===")
    print(f"torch: {torch.__version__} | cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available, aborting before load.")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    free_before_gb = gb(free_before)
    print(f"VRAM before load: free={free_before_gb:.2f} GB / total={gb(total):.2f} GB")

    if free_before_gb < SAFETY_FREE_VRAM_GB:
        print(
            f"FAIL: free VRAM ({free_before_gb:.2f} GB) below safety threshold "
            f"({SAFETY_FREE_VRAM_GB} GB) - aborting before load to avoid risking an OOM."
        )
        return 1

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name} (dtype={dtype_name}, device_map=auto)")
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_elapsed = time.time() - load_start
    print(f"Model loaded in {load_elapsed:.1f}s")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {n_params:,} (~{n_params / 1e9:.2f}B)")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Device map: {getattr(model, 'hf_device_map', 'n/a')}")

    free_after, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"VRAM after load: free={gb(free_after):.2f} GB / total={gb(total):.2f} GB")
    print(f"torch allocated={gb(allocated):.2f} GB | reserved={gb(reserved):.2f} GB")

    print(f"\nPrompt: {PROMPT!r}")
    messages = [{"role": "user", "content": PROMPT}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
    except TypeError:
        print("note: tokenizer chat template does not support enable_thinking kwarg, retrying without it")
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except Exception as exc:  # noqa: BLE001
        print(f"note: chat template unavailable ({exc!r}), falling back to raw prompt")
        encoded = tokenizer(PROMPT, return_tensors="pt")

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[1]

    gen_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False,
        )
    gen_elapsed = time.time() - gen_start

    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

    print(f"\nGenerated answer: {generated_text!r}")
    print(f"Generation latency: {gen_elapsed:.2f}s")

    peak = torch.cuda.max_memory_allocated()
    free_end, total = torch.cuda.mem_get_info()
    print(f"Peak allocated VRAM (this process): {gb(peak):.2f} GB")
    print(f"VRAM after generation: free={gb(free_end):.2f} GB / total={gb(total):.2f} GB")

    print("\n=== Smoke test complete: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
