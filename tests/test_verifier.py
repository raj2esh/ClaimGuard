"""Tests for the ClaimGuard verifier baseline (claimguard.verifier).

Covers: label-mapping verification, dataset loading/filtering, provenance
retention, safety-guard rejection (forbidden sources, excluded
contamination), tokenized batch construction, a tiny model's forward/
backward pass and checkpoint save/load roundtrip, deterministic seeding,
and evaluation metrics. Uses a TINY randomly-initialized model for the
forward-pass/checkpoint tests (not the real 440M-param verifier) so this
suite stays fast and offline - no GPU required.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig, BertConfig, BertForSequenceClassification

from claimguard import config as cg_config
from claimguard.verifier.dataset import (
    TrainingSafetyError,
    VerifierExample,
    load_baseline_examples,
)
from claimguard.verifier.evaluate import compute_metrics
from claimguard.verifier.train import (
    TokenizedVerifierDataset,
    load_label2id_from_model,
    set_all_seeds,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _pool_row(
    source_dataset="halueval", source_subset="qa", source_id="qa_0_not_hallucinated",
    label="entailment", original_label="not_hallucinated",
    premise_text="some premise", premise_text_available=True, hypothesis="some hypothesis",
):
    return {
        "hypothesis": hypothesis, "label": label, "original_label": original_label,
        "source_dataset": source_dataset, "source_subset": source_subset, "source_id": source_id,
        "premise_text": premise_text, "premise_reference": None,
        "premise_text_available": premise_text_available,
    }


class TestConfigLoads(unittest.TestCase):
    def test_verifier_baseline_config_loads_with_required_keys(self) -> None:
        cfg = cg_config.load_verifier_baseline_config()
        self.assertIn("model", cfg)
        self.assertIn("data", cfg)
        self.assertIn("training", cfg)
        self.assertIn("smoke_test", cfg)
        self.assertEqual(set(cfg["model"]["label_order"]), {"entailment", "neutral", "contradiction"})


class TestLabelMappingVerification(unittest.TestCase):
    def test_matching_id2label_builds_correct_label2id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = BertConfig(
                vocab_size=100, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                intermediate_size=16, num_labels=3,
                id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
                label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
            )
            config.save_pretrained(td)
            label2id = load_label2id_from_model(td, {"entailment", "neutral", "contradiction"})
        self.assertEqual(label2id, {"entailment": 0, "neutral": 1, "contradiction": 2})

    def test_mismatched_id2label_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = BertConfig(
                vocab_size=100, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                intermediate_size=16, num_labels=2,
                id2label={0: "positive", 1: "negative"},
                label2id={"positive": 0, "negative": 1},
            )
            config.save_pretrained(td)
            with self.assertRaises(ValueError):
                load_label2id_from_model(td, {"entailment", "neutral", "contradiction"})


class TestDatasetLoadingAndFiltering(unittest.TestCase):
    def test_filters_out_records_without_premise_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pool.jsonl"
            _write_jsonl(path, [
                _pool_row(source_dataset="fever", source_subset="train", source_id=1,
                          premise_text=None, premise_text_available=False),
                _pool_row(source_dataset="halueval", source_subset="qa", source_id="qa_0_not_hallucinated"),
            ])
            examples, stats = load_baseline_examples(path, require_premise_text=True)
        self.assertEqual(len(examples), 1)
        self.assertEqual(stats["excluded_no_premise_text"], 1)
        self.assertEqual(stats["excluded_by_source"], {"fever.train": 1})

    def test_all_fever_excluded_matches_documented_step9_finding(self) -> None:
        # Reproduces the real Step 9 finding at small scale: FEVER
        # currently contributes zero usable records.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pool.jsonl"
            _write_jsonl(path, [
                _pool_row(source_dataset="fever", source_subset="train", source_id=i,
                          premise_text=None, premise_text_available=False)
                for i in range(5)
            ] + [
                _pool_row(source_dataset="halueval", source_subset="dialogue",
                          source_id=f"dialogue_{i}_hallucinated", label="contradiction")
                for i in range(3)
            ])
            examples, stats = load_baseline_examples(path, require_premise_text=True)
        self.assertEqual(stats["excluded_by_source"]["fever.train"], 5)
        self.assertEqual(len(examples), 3)
        self.assertTrue(all(e.source_dataset == "halueval" for e in examples))

    def test_provenance_fields_retained(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pool.jsonl"
            _write_jsonl(path, [
                _pool_row(source_dataset="halueval", source_subset="summarization",
                          source_id="summarization_7_hallucinated", label="contradiction",
                          original_label="hallucinated"),
            ])
            examples, _ = load_baseline_examples(path, require_premise_text=True)
        ex = examples[0]
        self.assertEqual(ex.source_dataset, "halueval")
        self.assertEqual(ex.source_subset, "summarization")
        self.assertEqual(ex.source_id, "summarization_7_hallucinated")
        self.assertEqual(ex.original_label, "hallucinated")
        self.assertEqual(ex.label, "contradiction")


class TestSafetyGuards(unittest.TestCase):
    def test_ragtruth_source_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pool.jsonl"
            _write_jsonl(path, [_pool_row(source_dataset="ragtruth", source_subset="test")])
            with self.assertRaises(TrainingSafetyError):
                load_baseline_examples(path)

    def test_truthfulqa_source_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pool.jsonl"
            _write_jsonl(path, [_pool_row(source_dataset="truthfulqa", source_subset=None)])
            with self.assertRaises(TrainingSafetyError):
                load_baseline_examples(path)

    def test_unrecognized_source_subset_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pool.jsonl"
            _write_jsonl(path, [_pool_row(source_dataset="halueval", source_subset="general")])
            with self.assertRaises(TrainingSafetyError):
                load_baseline_examples(path)

    def test_known_contaminated_record_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pool.jsonl"
            _write_jsonl(path, [
                _pool_row(source_dataset="halueval", source_subset="summarization",
                          source_id="summarization_42_hallucinated"),
            ])
            with mock.patch(
                "claimguard.verifier.dataset.mf.contaminated_halueval_summarization_raw_indices",
                return_value={42},
            ):
                with self.assertRaises(TrainingSafetyError):
                    load_baseline_examples(path)

    def test_non_contaminated_summarization_record_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pool.jsonl"
            _write_jsonl(path, [
                _pool_row(source_dataset="halueval", source_subset="summarization",
                          source_id="summarization_99_hallucinated"),
            ])
            with mock.patch(
                "claimguard.verifier.dataset.mf.contaminated_halueval_summarization_raw_indices",
                return_value={42},
            ):
                examples, _ = load_baseline_examples(path)
        self.assertEqual(len(examples), 1)


class TestTokenizedDataset(unittest.TestCase):
    def test_batch_construction_and_label_ids(self) -> None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
            )
        except Exception:
            raise unittest.SkipTest("verifier tokenizer not available locally/offline")

        examples = [
            VerifierExample("premise A", "hypothesis A", "entailment", "halueval", "qa", "id1", "not_hallucinated"),
            VerifierExample("premise B", "hypothesis B", "contradiction", "halueval", "qa", "id2", "hallucinated"),
        ]
        label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
        ds = TokenizedVerifierDataset(examples, tokenizer, max_length=32, label2id=label2id)
        self.assertEqual(len(ds), 2)
        item0 = ds[0]
        self.assertIn("input_ids", item0)
        self.assertIn("attention_mask", item0)
        self.assertEqual(item0["labels"], 0)
        item1 = ds[1]
        self.assertEqual(item1["labels"], 2)


def _make_tiny_model() -> BertForSequenceClassification:
    config = BertConfig(
        vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=32, num_labels=3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    )
    return BertForSequenceClassification(config)


class TestTinyModelForwardAndCheckpoint(unittest.TestCase):
    def test_forward_and_backward_pass(self) -> None:
        model = _make_tiny_model()
        model.train()
        input_ids = torch.randint(0, 100, (4, 8))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([0, 1, 2, 0])
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.assertIsNotNone(outputs.loss)
        outputs.loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_checkpoint_save_and_load_roundtrip(self) -> None:
        model = _make_tiny_model()
        with tempfile.TemporaryDirectory() as td:
            model.save_pretrained(td)
            loaded = AutoModelForSequenceClassification.from_pretrained(td)
        self.assertEqual(loaded.config.num_labels, 3)
        self.assertEqual(loaded.config.id2label[0], "entailment")
        self.assertEqual(loaded.config.id2label[2], "contradiction")


class TestDeterministicSeeding(unittest.TestCase):
    def test_seeding_produces_reproducible_torch_output(self) -> None:
        set_all_seeds(123)
        a = torch.randn(5)
        set_all_seeds(123)
        b = torch.randn(5)
        self.assertTrue(torch.equal(a, b))

    def test_different_seeds_produce_different_output(self) -> None:
        set_all_seeds(1)
        a = torch.randn(5)
        set_all_seeds(2)
        b = torch.randn(5)
        self.assertFalse(torch.equal(a, b))


class TestComputeMetrics(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        labels = np.array([0, 1, 2, 0, 1, 2])
        preds = np.array([0, 1, 2, 0, 1, 2])
        result = compute_metrics(labels, preds, id2label)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["num_examples"], 6)

    def test_known_confusion_matrix(self) -> None:
        id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        labels = np.array([0, 0, 1, 2])
        preds = np.array([0, 1, 1, 2])
        result = compute_metrics(labels, preds, id2label)
        self.assertLess(result["accuracy"], 1.0)
        self.assertIn("entailment", result["per_class"])
        self.assertIn("neutral", result["per_class"])
        self.assertIn("contradiction", result["per_class"])
        cm = result["confusion_matrix"]
        self.assertEqual(sum(sum(row) for row in cm), 4)


if __name__ == "__main__":
    unittest.main()
