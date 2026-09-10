"""Tests for the Step 12 FINAL binary verifier (claimguard.verifier.binary_model,
claimguard.verifier.train_binary, and the new compute_metrics_by_group helper).

Uses a TINY, locally-constructed DebertaV2ForSequenceClassification for the
binary-head-construction tests (same architecture family as the real
verifier model, so binary_model.py's architecture-specific checks are
exercised honestly) - never the real 440M-param model, so this suite stays
fast and offline. No GPU required.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

from claimguard import config as cg_config
from claimguard.verifier.binary_model import (
    BINARY_ID2LABEL,
    BINARY_LABEL2ID,
    assert_binary_head,
    build_binary_model,
    verified_source_label_indices,
)
from claimguard.verifier.dataset import VerifierExample
from claimguard.verifier.evaluate import compute_metrics_by_group
from claimguard.verifier.train_binary import TrainingBlockedError, assert_binary_labels_only


def _make_tiny_deberta_3class(hidden_size: int = 8) -> DebertaV2ForSequenceClassification:
    config = DebertaV2Config(
        vocab_size=100, hidden_size=hidden_size, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, max_position_embeddings=32, num_labels=3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    )
    return DebertaV2ForSequenceClassification(config)


class TestConfigLoads(unittest.TestCase):
    def test_verifier_binary_final_config_loads_with_required_keys(self) -> None:
        cfg = cg_config.load_verifier_binary_final_config()
        self.assertIn("model", cfg)
        self.assertIn("data", cfg)
        self.assertIn("training", cfg)
        self.assertEqual(cfg["model"]["num_labels"], 2)
        self.assertEqual(set(cfg["model"]["label_order"]), {"entailment", "contradiction"})
        self.assertEqual(cfg["data"]["manifest_version_expected"], "1.3")


class TestVerifiedSourceLabelIndices(unittest.TestCase):
    def test_finds_correct_indices_from_real_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model = _make_tiny_deberta_3class()
            model.config.save_pretrained(td)
            indices = verified_source_label_indices(td, {"entailment", "contradiction"})
        self.assertEqual(indices, {"entailment": 0, "contradiction": 2})

    def test_missing_required_label_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = DebertaV2Config(
                vocab_size=100, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                intermediate_size=16, num_labels=2,
                id2label={0: "positive", 1: "negative"},
                label2id={"positive": 0, "negative": 1},
            )
            config.save_pretrained(td)
            with self.assertRaises(ValueError):
                verified_source_label_indices(td, {"entailment", "contradiction"})


class TestAssertBinaryHead(unittest.TestCase):
    def test_accepts_correct_binary_head(self) -> None:
        config = DebertaV2Config(
            vocab_size=100, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, num_labels=2,
            id2label=BINARY_ID2LABEL, label2id=BINARY_LABEL2ID,
        )
        model = DebertaV2ForSequenceClassification(config)
        assert_binary_head(model)  # must not raise

    def test_rejects_three_class_head(self) -> None:
        model = _make_tiny_deberta_3class()
        with self.assertRaises(AssertionError):
            assert_binary_head(model)

    def test_rejects_binary_head_with_wrong_label_mapping(self) -> None:
        config = DebertaV2Config(
            vocab_size=100, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, num_labels=2,
            id2label={0: "positive", 1: "negative"},
            label2id={"positive": 0, "negative": 1},
        )
        model = DebertaV2ForSequenceClassification(config)
        with self.assertRaises(AssertionError):
            assert_binary_head(model)


class TestBuildBinaryModel(unittest.TestCase):
    def test_binary_model_has_correct_architecture_and_transplanted_weights(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            original = _make_tiny_deberta_3class()
            entailment_row = original.classifier.weight.data[0].clone()
            entailment_bias = original.classifier.bias.data[0].clone()
            contradiction_row = original.classifier.weight.data[2].clone()
            contradiction_bias = original.classifier.bias.data[2].clone()
            original.save_pretrained(td)

            model, report = build_binary_model(td, dtype=None)

        # Architecture
        assert_binary_head(model)  # must not raise
        self.assertEqual(model.config.num_labels, 2)
        self.assertEqual(model.classifier.weight.shape, torch.Size([2, 8]))

        # Weight transplant correctness - not randomly initialized
        self.assertTrue(torch.equal(model.classifier.weight.data[0], entailment_row))
        self.assertTrue(torch.equal(model.classifier.bias.data[0], entailment_bias))
        self.assertTrue(torch.equal(model.classifier.weight.data[1], contradiction_row))
        self.assertTrue(torch.equal(model.classifier.bias.data[1], contradiction_bias))

        # Report correctness
        self.assertEqual(report["source_label_indices"], {"entailment": 0, "contradiction": 2})
        self.assertFalse(report["classifier_rows_randomly_initialized"])
        self.assertTrue(report["encoder_reuse_verified_bit_identical"])
        self.assertTrue(report["pooler_reuse_verified_bit_identical"])

    def test_encoder_and_pooler_are_bit_identical_to_original(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            original = _make_tiny_deberta_3class()
            original.save_pretrained(td)
            original_params = dict(original.named_parameters())

            model, _ = build_binary_model(td, dtype=None)

        for name, param in model.named_parameters():
            if name.startswith("classifier."):
                continue
            self.assertTrue(
                torch.equal(original_params[name].data, param.data),
                f"parameter {name} differs from the original pretrained model",
            )

    def test_rejects_non_three_class_source_model(self) -> None:
        # build_binary_model expects to transplant FROM a 3-class model -
        # a 2-class (or any non-3) source should be rejected loudly, not
        # silently misinterpreted.
        with tempfile.TemporaryDirectory() as td:
            config = DebertaV2Config(
                vocab_size=100, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                intermediate_size=16, num_labels=2,
                id2label={"0": "entailment", "1": "contradiction"},
                label2id={"entailment": 0, "contradiction": 1},
            )
            model = DebertaV2ForSequenceClassification(config)
            model.save_pretrained(td)
            with self.assertRaises(RuntimeError):
                build_binary_model(td, dtype=None)


class TestAssertBinaryLabelsOnly(unittest.TestCase):
    def _example(self, label: str) -> VerifierExample:
        return VerifierExample("premise", "hypothesis", label, "halueval", "qa", "id1", "orig")

    def test_accepts_binary_only_examples(self) -> None:
        examples = [self._example("entailment"), self._example("contradiction")]
        assert_binary_labels_only(examples, "train")  # must not raise

    def test_rejects_neutral_label(self) -> None:
        examples = [self._example("entailment"), self._example("neutral")]
        with self.assertRaises(TrainingBlockedError):
            assert_binary_labels_only(examples, "train")

    def test_rejects_unexpected_label(self) -> None:
        examples = [self._example("hallucinated")]
        with self.assertRaises(TrainingBlockedError):
            assert_binary_labels_only(examples, "dev")

    def test_empty_list_does_not_raise(self) -> None:
        assert_binary_labels_only([], "train")  # must not raise


class TestComputeMetricsByGroup(unittest.TestCase):
    def test_splits_metrics_by_group_correctly(self) -> None:
        id2label = {0: "entailment", 1: "contradiction"}
        labels = np.array([0, 0, 1, 1, 0, 1])
        preds = np.array([0, 1, 1, 1, 0, 0])
        groups = ["fever", "fever", "halueval.qa", "halueval.qa", "fever", "halueval.qa"]
        result = compute_metrics_by_group(labels, preds, id2label, groups)

        self.assertEqual(set(result.keys()), {"fever", "halueval.qa"})
        self.assertEqual(result["fever"]["num_examples"], 3)
        self.assertEqual(result["halueval.qa"]["num_examples"], 3)
        # fever: labels [0,0,0] preds [0,1,0] -> 2/3 correct
        self.assertAlmostEqual(result["fever"]["accuracy"], 2 / 3)

    def test_omits_empty_groups(self) -> None:
        id2label = {0: "entailment", 1: "contradiction"}
        labels = np.array([0, 1])
        preds = np.array([0, 1])
        groups = ["fever", "fever"]
        result = compute_metrics_by_group(labels, preds, id2label, groups)
        self.assertEqual(set(result.keys()), {"fever"})


if __name__ == "__main__":
    unittest.main()
