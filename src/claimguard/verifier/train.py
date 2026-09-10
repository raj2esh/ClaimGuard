"""ClaimGuard verifier: baseline training entry point.

Fine-tunes the project's already-selected NLI verifier model
(MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, see Step 3/5B)
on the Step 9 baseline pool (HaluEval qa/dialogue/summarization only - see
dataset.py for why FEVER is currently excluded). Uses transformers.Trainer
- no custom training loop, no custom architecture, no retrieval/FAISS/
reranking.

Usage:
    python -m claimguard.verifier.train --config configs/verifier_baseline.yaml
    python -m claimguard.verifier.train --config configs/verifier_baseline.yaml --smoke-test
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .. import config as cg_config
from .dataset import VerifierExample, load_baseline_examples
from .evaluate import compute_metrics


def set_all_seeds(seed: int) -> None:
    """Record every source of randomness explicitly, not just one."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # transformers convenience wrapper; also covers the above


def load_label2id_from_model(model_name: str, expected_labels: set[str]) -> dict[str, int]:
    """Verify (not assume) the model's own id2label matches our target
    verifier label scheme, and build label2id from it. Refuses to guess a
    mapping if they don't match."""
    config = AutoConfig.from_pretrained(model_name)
    id2label = {int(k): v.lower() for k, v in config.id2label.items()}
    found_labels = set(id2label.values())
    if found_labels != expected_labels:
        raise ValueError(
            f"Model {model_name!r}'s id2label {id2label} does not match the expected "
            f"verifier label scheme {expected_labels} - refusing to guess a mapping."
        )
    return {v: k for k, v in id2label.items()}


class TokenizedVerifierDataset(torch.utils.data.Dataset):
    def __init__(
        self, examples: list[VerifierExample], tokenizer, max_length: int, label2id: dict[str, int]
    ) -> None:
        self.examples = examples
        self.label2id = label2id
        encodings = tokenizer(
            [e.premise for e in examples],
            [e.hypothesis for e in examples],
            truncation=True,
            max_length=max_length,
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.token_type_ids = encodings.get("token_type_ids")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.label2id[self.examples[idx].label],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


def run(config_path: str, smoke_test: bool = False) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    set_all_seeds(seed)
    print(f"Seeds set (python/numpy/torch/cuda/transformers): {seed}")

    model_name = cfg["model"]["name"]
    max_length = cfg["model"]["max_length"]
    expected_labels = set(cfg["model"]["label_order"])

    label2id = load_label2id_from_model(model_name, expected_labels)
    id2label = {v: k for k, v in label2id.items()}
    print(f"Verified model label mapping (from the model's own config, not assumed): {label2id}")

    train_path = cg_config.resolve_path(cfg["data"]["pool_train_path"])
    dev_path = cg_config.resolve_path(cfg["data"]["pool_dev_path"])
    require_premise_text = cfg["data"]["require_premise_text"]

    train_examples, train_stats = load_baseline_examples(train_path, require_premise_text)
    dev_examples, dev_stats = load_baseline_examples(dev_path, require_premise_text)
    print(f"Train stats: {train_stats}")
    print(f"Dev stats: {dev_stats}")

    if smoke_test:
        n_train = cfg["smoke_test"]["max_train_samples"]
        n_dev = cfg["smoke_test"]["max_dev_samples"]
        train_examples = train_examples[:n_train]
        dev_examples = dev_examples[:n_dev]
        print(f"SMOKE TEST: truncated to {len(train_examples)} train / {len(dev_examples)} dev examples")

    print(f"Loading tokenizer/model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_dtype = torch.bfloat16 if cfg["training"]["mixed_precision"] == "bf16" else None
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch_dtype)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_total:,} total, {n_trainable:,} trainable "
          f"(full fine-tuning - pretrained classification head reused as-is, not "
          f"reinitialized, since its id2label already matched our target scheme)")

    train_ds = TokenizedVerifierDataset(train_examples, tokenizer, max_length, label2id)
    dev_ds = TokenizedVerifierDataset(dev_examples, tokenizer, max_length, label2id)
    collator = DataCollatorWithPadding(tokenizer)

    output_dir = cg_config.resolve_path(cfg["training"]["output_dir"])
    if smoke_test:
        output_dir = output_dir.parent / f"{output_dir.name}_smoke_test"

    step_cfg = cfg["smoke_test"] if smoke_test else cfg["training"]

    # This transformers version's TrainingArguments dropped `warmup_ratio`
    # (only `warmup_steps` remains, verified via introspection, not assumed)
    # - compute the equivalent step count ourselves from the config's ratio
    # so the documented warmup_ratio setting still has its intended effect.
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

    print("Starting training...")
    trainer.train()

    print("Running final dev evaluation...")
    metrics = trainer.evaluate()
    print(f"Final dev metrics: {metrics}")

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    with (final_dir / "training_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    with (final_dir / "dev_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    with (final_dir / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump({"train": train_stats, "dev": dev_stats}, f, indent=2, default=str)
    with (final_dir / "reproducibility.json").open("w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "model_name": model_name,
            "label2id": label2id,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "n_total_params": n_total,
            "n_trainable_params": n_trainable,
            "manifest_version_expected": cfg["data"]["manifest_version_expected"],
        }, f, indent=2, default=str)

    print(f"Saved final checkpoint + tokenizer + config + metrics to {final_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run(args.config, smoke_test=args.smoke_test)
