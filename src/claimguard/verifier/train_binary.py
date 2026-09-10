"""ClaimGuard verifier: FINAL binary (entailment/contradiction) training entry point.

Step 12 - trains the project's already-selected NLI verifier model
(MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, Step 3/5B) with
a genuine 2-class classification head (see binary_model.py) on the full
Step 10/11 leakage-safe pool: FEVER SUPPORTS/REFUTES (with resolved
Wikipedia-sentence premise text) plus HaluEval qa/dialogue/summarization
(post contamination-exclusion). This supersedes the Step 9 HaluEval-only
preliminary baseline - FEVER now contributes real premise text and is
included for the first time.

Per Step 11's finalized protocol, this is explicitly a BINARY
(entailment/contradiction) verifier. There is no neutral class in this
training run; none is fabricated; the pretrained model's original 3-class
head is not reused as-is (see binary_model.py for why and how the new
2-class head is constructed).

Uses transformers.Trainer - no custom training loop, no retrieval/FAISS/
reranking. Does not evaluate on RAGTruth or TruthfulQA (both remain
reserved for later, separate evaluation steps).

Usage:
    python -m claimguard.verifier.train_binary --config configs/verifier_binary_final.yaml
    python -m claimguard.verifier.train_binary --config configs/verifier_binary_final.yaml --smoke-test
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from .. import config as cg_config
from .binary_model import BINARY_ID2LABEL, BINARY_LABEL2ID, assert_binary_head, build_binary_model
from .dataset import VerifierExample, load_baseline_examples
from .evaluate import compute_metrics, compute_metrics_by_group
from .train import TokenizedVerifierDataset, set_all_seeds

VALIDATE_SCRIPT = cg_config.resolve_path("scripts/validate_verifier_dataset.py")


class TrainingBlockedError(RuntimeError):
    """Raised when a pre-training safety assertion fails. Training must
    stop, not proceed with a filtered-down or otherwise silently-adjusted
    dataset."""


def run_preflight_dataset_validation() -> None:
    """Run scripts/validate_verifier_dataset.py as a subprocess (the same,
    already-tested Step 11 validation gate) and hard-block training if it
    fails for any reason - schema, provenance, train/dev disjointness,
    HaluEval pair grouping, contamination exclusion, RAGTruth/TruthfulQA
    boundaries, class coverage, or determinism.
    """
    print(f"Running pre-training dataset validation: {VALIDATE_SCRIPT}")
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT)], capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise TrainingBlockedError(
            f"Pre-training dataset validation FAILED (exit code {result.returncode}) - "
            "refusing to start training. See output above for the failing check(s)."
        )
    print("Pre-training dataset validation PASSED.")


def assert_binary_labels_only(examples: list[VerifierExample], split_name: str) -> None:
    """Hard guard: every loaded example's label must be exactly
    'entailment' or 'contradiction' - never 'neutral' (Step 11: no
    legitimate neutral source exists; none may silently appear here) and
    never any other unexpected value."""
    bad = {e.label for e in examples} - set(BINARY_LABEL2ID)
    if bad:
        raise TrainingBlockedError(
            f"{split_name}: found label(s) outside the binary protocol {sorted(BINARY_LABEL2ID)}: "
            f"{sorted(bad)}. Refusing to train - this would mean the scientific protocol "
            "changed without this script being updated to match."
        )


def _source_group_key(example: VerifierExample) -> str:
    if example.source_dataset == "halueval":
        return f"halueval.{example.source_subset}"
    return example.source_dataset


def run(config_path: str, smoke_test: bool = False) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    set_all_seeds(seed)
    print(f"Seeds set (python/numpy/torch/cuda/transformers): {seed}")

    if not smoke_test:
        run_preflight_dataset_validation()
    else:
        print("SMOKE TEST: skipping the full pre-training dataset validation subprocess "
              "(it validates the full pool, not the truncated smoke-test slice) - the "
              "full run always validates before training.")

    model_name = cfg["model"]["name"]
    max_length = cfg["model"]["max_length"]
    if set(cfg["model"]["label_order"]) != set(BINARY_LABEL2ID):
        raise TrainingBlockedError(
            f"Config label_order {cfg['model']['label_order']} does not match the binary "
            f"protocol {sorted(BINARY_LABEL2ID)}."
        )

    train_path = cg_config.resolve_path(cfg["data"]["pool_train_path"])
    dev_path = cg_config.resolve_path(cfg["data"]["pool_dev_path"])
    require_premise_text = cfg["data"]["require_premise_text"]

    manifest_path = cg_config.resolve_path(cfg["data"]["manifest_path"])
    with manifest_path.open() as f:
        manifest = json.load(f)
    if manifest.get("version") != cfg["data"]["manifest_version_expected"]:
        raise TrainingBlockedError(
            f"Manifest version {manifest.get('version')!r} does not match expected "
            f"{cfg['data']['manifest_version_expected']!r} - re-run scripts/build_dataset_manifest.py "
            "or update this config."
        )
    classification_mode = manifest.get("verifier_dataset_status", {}).get("classification_mode")
    if classification_mode != "binary":
        raise TrainingBlockedError(
            f"Manifest verifier_dataset_status.classification_mode={classification_mode!r}, "
            "expected 'binary' - this script trains an explicit binary verifier and refuses "
            "to proceed if the manifest disagrees about what the current pool supports."
        )
    print(f"Manifest v{manifest['version']} confirms classification_mode='binary' - proceeding.")

    train_examples, train_stats = load_baseline_examples(train_path, require_premise_text)
    dev_examples, dev_stats = load_baseline_examples(dev_path, require_premise_text)
    assert_binary_labels_only(train_examples, "train")
    assert_binary_labels_only(dev_examples, "dev")
    print(f"Train stats: {train_stats}")
    print(f"Dev stats: {dev_stats}")

    # --- Report the final dataset composition from real, freshly-loaded data ---
    def _dataset_report(examples: list[VerifierExample], name: str) -> dict[str, Any]:
        from collections import Counter
        label_counts = Counter(e.label for e in examples)
        source_counts = Counter(_source_group_key(e) for e in examples)
        total = len(examples)
        return {
            "split": name,
            "total": total,
            "label_counts": dict(label_counts),
            "label_percentages": {k: round(100 * v / total, 2) for k, v in label_counts.items()} if total else {},
            "source_counts": dict(source_counts),
            "source_percentages": {k: round(100 * v / total, 2) for k, v in source_counts.items()} if total else {},
        }

    train_report = _dataset_report(train_examples, "train")
    dev_report = _dataset_report(dev_examples, "dev")
    print(f"FINAL DATASET REPORT (train): {json.dumps(train_report, indent=2)}")
    print(f"FINAL DATASET REPORT (dev): {json.dumps(dev_report, indent=2)}")

    if smoke_test:
        n_train = cfg["smoke_test"]["max_train_samples"]
        n_dev = cfg["smoke_test"]["max_dev_samples"]
        train_examples = train_examples[:n_train]
        dev_examples = dev_examples[:n_dev]
        print(f"SMOKE TEST: truncated to {len(train_examples)} train / {len(dev_examples)} dev examples")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_dtype = torch.bfloat16 if cfg["training"]["mixed_precision"] == "bf16" else None

    model, head_report = build_binary_model(model_name, dtype=torch_dtype)
    assert_binary_head(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_total:,} total, {n_trainable:,} trainable (full fine-tuning)")
    print(f"Binary head construction report: {json.dumps(head_report, indent=2)}")

    label2id = BINARY_LABEL2ID
    id2label = BINARY_ID2LABEL

    train_ds = TokenizedVerifierDataset(train_examples, tokenizer, max_length, label2id)
    dev_ds = TokenizedVerifierDataset(dev_examples, tokenizer, max_length, label2id)
    collator = DataCollatorWithPadding(tokenizer)

    output_dir = cg_config.resolve_path(cfg["training"]["output_dir"])
    if smoke_test:
        output_dir = output_dir.parent / f"{output_dir.name}_smoke_test"

    step_cfg = cfg["smoke_test"] if smoke_test else cfg["training"]

    effective_batch_size = cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation_steps"]
    steps_per_epoch = max(1, -(-len(train_examples) // effective_batch_size))  # ceil division
    total_steps = cfg["smoke_test"]["max_steps"] if smoke_test else steps_per_epoch * cfg["training"]["num_epochs"]
    warmup_steps = int(cfg["training"]["warmup_ratio"] * total_steps)
    print(f"Computed total_steps={total_steps}, warmup_steps={warmup_steps} "
          f"(from warmup_ratio={cfg['training']['warmup_ratio']})")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        num_train_epochs=cfg["training"]["num_epochs"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        bf16=(cfg["training"]["mixed_precision"] == "bf16"),
        max_grad_norm=cfg["training"]["max_grad_norm"],
        eval_strategy="steps",
        eval_steps=step_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=step_cfg["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        logging_steps=step_cfg["logging_steps"],
        max_steps=(cfg["smoke_test"]["max_steps"] if smoke_test else -1),
        metric_for_best_model=cfg["training"]["metric_for_best_model"],
        greater_is_better=cfg["training"]["greater_is_better"],
        load_best_model_at_end=True,
        report_to=[],
    )

    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics(labels, preds, id2label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )

    peak_mem_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Starting training...")
    train_result = trainer.train()
    print(f"train_result: {train_result}")

    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    )

    print("Running final dev evaluation...")
    metrics = trainer.evaluate()
    print(f"Final dev metrics: {metrics}")

    # --- Per-source-group dev metrics (never let aggregate hide a weak group) ---
    predict_output = trainer.predict(dev_ds)
    all_preds = np.argmax(predict_output.predictions, axis=-1)
    all_labels = np.asarray(predict_output.label_ids)
    group_keys = [_source_group_key(e) for e in dev_examples]
    overall_check = compute_metrics(all_labels, all_preds, id2label)
    per_group_metrics = compute_metrics_by_group(all_labels, all_preds, id2label, group_keys)
    # halueval_overall (all three subsets combined) alongside the per-subset breakdown
    halueval_mask = np.array([e.source_dataset == "halueval" for e in dev_examples])
    if halueval_mask.any():
        per_group_metrics["halueval_overall"] = compute_metrics(
            all_labels[halueval_mask], all_preds[halueval_mask], id2label
        )
    print(f"Per-source-group dev metrics: {json.dumps(per_group_metrics, indent=2)}")

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    with (final_dir / "training_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    with (final_dir / "dev_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    with (final_dir / "dev_metrics_overall_recomputed.json").open("w", encoding="utf-8") as f:
        json.dump(overall_check, f, indent=2, default=str)
    with (final_dir / "dev_metrics_by_source.json").open("w", encoding="utf-8") as f:
        json.dump(per_group_metrics, f, indent=2, default=str)
    with (final_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump({
            "train_pool_load_stats": train_stats, "dev_pool_load_stats": dev_stats,
            "train_final_report": train_report, "dev_final_report": dev_report,
        }, f, indent=2, default=str)
    with (final_dir / "binary_head_report.json").open("w", encoding="utf-8") as f:
        json.dump(head_report, f, indent=2, default=str)

    reproducibility = {
        "seed": seed,
        "model_name": model_name,
        "label2id": label2id,
        "id2label": id2label,
        "binary_head_construction": head_report,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_gb": peak_mem_gb,
        "n_total_params": n_total,
        "n_trainable_params": n_trainable,
        "manifest_version_expected": cfg["data"]["manifest_version_expected"],
        "manifest_version_actual": manifest["version"],
        "train_record_count": len(train_examples),
        "dev_record_count": len(dev_examples),
        "max_length": max_length,
        "optimizer": "AdamW (transformers.Trainer default)",
        "learning_rate": cfg["training"]["learning_rate"],
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "effective_batch_size": effective_batch_size,
        "precision": cfg["training"]["mixed_precision"],
        "num_epochs": cfg["training"]["num_epochs"],
        "lr_scheduler_type": cfg["training"]["lr_scheduler_type"],
        "warmup_ratio": cfg["training"]["warmup_ratio"],
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "max_grad_norm": cfg["training"]["max_grad_norm"],
        "weight_decay": cfg["training"]["weight_decay"],
        "metric_for_best_model": cfg["training"]["metric_for_best_model"],
        "greater_is_better": cfg["training"]["greater_is_better"],
        "checkpoint_selection_criterion": (
            f"{cfg['training']['metric_for_best_model']} "
            f"({'higher' if cfg['training']['greater_is_better'] else 'lower'} is better), "
            "load_best_model_at_end=True"
        ),
    }
    with (final_dir / "reproducibility.json").open("w", encoding="utf-8") as f:
        json.dump(reproducibility, f, indent=2, default=str)

    print(f"Saved final checkpoint + tokenizer + configs + all metrics to {final_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run(args.config, smoke_test=args.smoke_test)
