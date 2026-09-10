"""Step 5B smoke test for the NLI claim verifier
(MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli).

Loads the model exactly as configured in configs/models.yaml, runs three small
NLI test cases (entailment / contradiction / neutral), and reports load/inference
diagnostics. This is a smoke test only - it does not implement the ClaimGuard
verification module (see PROJECT_PLAN.md Step 11). Loads ONLY the verifier -
no other model is loaded in this process.

Usage:
    python scripts/smoke_test_deberta.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402

TEST_CASES = [
    {
        "name": "entailment",
        "premise": "Paris is the capital city of France.",
        "hypothesis": "France's capital is Paris.",
        "expected": "ENTAILMENT",
    },
    {
        "name": "contradiction",
        "premise": "Paris is the capital city of France.",
        "hypothesis": "Berlin is the capital city of France.",
        "expected": "CONTRADICTION",
    },
    {
        "name": "neutral",
        "premise": "Paris is the capital city of France.",
        "hypothesis": "Paris has more than five million residents.",
        "expected": "UNKNOWN / NEUTRAL",
    },
]

SAFETY_FREE_VRAM_GB = 4.0  # DeBERTa-v3-large is small (~0.4B); generous margin still enforced


def gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def main() -> int:
    models_cfg = cg_config.load_models_config()
    ver_cfg = models_cfg["verifier"]["primary"]
    model_name = ver_cfg["name"]
    dtype_name = ver_cfg.get("dtype", "float32")

    print(f"=== ClaimGuard Step 5B smoke test: {model_name} ===")
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

    torch_dtype = getattr(torch, dtype_name, torch.float32)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name} (requested dtype={dtype_name})")
    load_start = time.time()
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch_dtype)
        used_dtype_name = dtype_name
    except Exception as exc:  # noqa: BLE001
        print(f"note: load with dtype={dtype_name} failed ({exc!r}), falling back to float32")
        model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float32)
        used_dtype_name = "float32"
    model = model.to("cuda:0")
    model.eval()
    load_elapsed = time.time() - load_start
    print(f"Model loaded in {load_elapsed:.1f}s (actual dtype used: {used_dtype_name})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {n_params:,} (~{n_params / 1e6:.1f}M)")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    # Verify the model's actual label mapping rather than assuming class indices.
    id2label = model.config.id2label
    print(f"id2label mapping (from model.config): {id2label}")

    free_after, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"VRAM after load: free={gb(free_after):.2f} GB / total={gb(total):.2f} GB")
    print(f"torch allocated={gb(allocated):.2f} GB | reserved={gb(reserved):.2f} GB")

    all_pass = True
    for case in TEST_CASES:
        premise = case["premise"]
        hypothesis = case["hypothesis"]

        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        infer_start = time.time()
        with torch.no_grad():
            logits = model(**inputs).logits
        infer_elapsed = time.time() - infer_start

        probs = F.softmax(logits.float(), dim=-1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_label = id2label[pred_idx]
        confidence = float(probs[pred_idx].item())
        class_probs = {id2label[i]: float(probs[i].item()) for i in range(len(id2label))}

        print(f"\n--- Test: {case['name']} (expected: {case['expected']}) ---")
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Predicted label: {pred_label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"All class probabilities: {class_probs}")
        print(f"Latency: {infer_elapsed * 1000:.1f} ms")

    peak = torch.cuda.max_memory_allocated()
    free_end, total = torch.cuda.mem_get_info()
    print(f"\nPeak allocated VRAM (this process): {gb(peak):.2f} GB")
    print(f"VRAM after inference: free={gb(free_end):.2f} GB / total={gb(total):.2f} GB")

    print(f"\n=== Smoke test complete: {'PASS' if all_pass else 'FAIL'} ===")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
