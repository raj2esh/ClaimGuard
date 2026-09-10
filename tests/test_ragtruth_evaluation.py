"""Tests for Step 19 (RAGTruth end-to-end evaluation): the pure metric/
label-mapping functions in `claimguard.evaluation.ragtruth_eval`, the new
`initial_candidate` extension to `claimguard.correction.run_correction_loop`,
and a behavioral gold-label-isolation guard on the actual evaluation
orchestration code in `scripts/evaluate_ragtruth.py`.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

from claimguard import config as cg_config
from claimguard.correction import correction as correction_mod
from claimguard.datasets import ragtruth as ragtruth_mod
from claimguard.evaluation import ragtruth_eval as eval_mod


# --- Pure function tests ------------------------------------------------

class TestDetectionPredictions(unittest.TestCase):
    def test_attempt0_accept_is_not_hallucinated(self) -> None:
        self.assertEqual(eval_mod.attempt0_detection_prediction("ACCEPT"), "not_hallucinated")

    def test_attempt0_correct_is_hallucinated(self) -> None:
        self.assertEqual(eval_mod.attempt0_detection_prediction("CORRECT"), "hallucinated")

    def test_attempt0_abstain_is_hallucinated(self) -> None:
        self.assertEqual(eval_mod.attempt0_detection_prediction("ABSTAIN"), "hallucinated")

    def test_final_accept_is_not_hallucinated(self) -> None:
        self.assertEqual(eval_mod.final_outcome_prediction("ACCEPT"), "not_hallucinated")

    def test_final_abstain_is_hallucinated(self) -> None:
        self.assertEqual(eval_mod.final_outcome_prediction("ABSTAIN"), "hallucinated")


class TestConfusionCounts(unittest.TestCase):
    def test_known_confusion_matrix(self) -> None:
        gold = [True, True, False, False, True]
        pred = [True, False, False, True, True]
        counts = eval_mod.confusion_counts(gold, pred)
        self.assertEqual(counts, {"tp": 2, "fp": 1, "fn": 1, "tn": 1})

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            eval_mod.confusion_counts([True], [True, False])

    def test_all_correct(self) -> None:
        counts = eval_mod.confusion_counts([True, False], [True, False])
        self.assertEqual(counts, {"tp": 1, "fp": 0, "fn": 0, "tn": 1})


class TestDetectionMetrics(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        gold = [True, False, True, False]
        m = eval_mod.detection_metrics(gold, gold)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["hallucinated_f1"], 1.0)
        self.assertAlmostEqual(m["macro_f1"], 1.0)

    def test_never_only_accuracy(self) -> None:
        gold = [True, True, False]
        pred = [True, True, True]
        m = eval_mod.detection_metrics(gold, pred)
        for key in ("hallucinated_precision", "hallucinated_recall", "hallucinated_f1",
                    "not_hallucinated_precision", "not_hallucinated_recall", "not_hallucinated_f1", "macro_f1"):
            self.assertIn(key, m)

    def test_confusion_matrix_included(self) -> None:
        m = eval_mod.detection_metrics([True, False], [True, False])
        self.assertIn("confusion_matrix", m)
        self.assertEqual(m["confusion_matrix"]["tp"], 1)

    def test_empty_input(self) -> None:
        m = eval_mod.detection_metrics([], [])
        self.assertEqual(m["n"], 0)
        self.assertIsNone(m["accuracy"])

    def test_zero_precision_and_recall_yields_zero_f1(self) -> None:
        # One actual positive missed (gold=True, pred=False) and one false alarm
        # (gold=False, pred=True) -> precision=0/1=0.0 and recall=0/1=0.0, BOTH
        # well-defined (there IS an actual positive to have missed).
        m = eval_mod.detection_metrics([True, False], [False, True])
        self.assertEqual(m["hallucinated_precision"], 0.0)
        self.assertEqual(m["hallucinated_recall"], 0.0)
        self.assertEqual(m["hallucinated_f1"], 0.0)

    def test_recall_undefined_when_no_actual_positives_in_gold(self) -> None:
        # gold has ZERO actual positives -> recall's denominator (TP+FN) is 0,
        # so recall (and therefore F1) is mathematically undefined - reported
        # as None, never silently defaulted to 0.0.
        m = eval_mod.detection_metrics([False, False], [True, True])
        self.assertEqual(m["hallucinated_precision"], 0.0)
        self.assertIsNone(m["hallucinated_recall"])
        self.assertIsNone(m["hallucinated_f1"])


class TestDecisionRateBreakdown(unittest.TestCase):
    def test_rates_sum_correctly(self) -> None:
        mapped0 = ["ACCEPT", "CORRECT", "ABSTAIN", "CORRECT"]
        final = ["ACCEPT", "ACCEPT", "ABSTAIN", "ABSTAIN"]
        correction_occurred = [False, True, False, True]
        max_attempts = [False, False, False, True]
        d = eval_mod.decision_rate_breakdown(mapped0, final, correction_occurred, max_attempts)
        self.assertEqual(d["n"], 4)
        self.assertAlmostEqual(d["attempt0_accept_rate"], 0.25)
        self.assertAlmostEqual(d["attempt0_correct_rate"], 0.5)
        self.assertAlmostEqual(d["attempt0_abstain_rate"], 0.25)
        self.assertAlmostEqual(d["correction_attempted_rate"], 0.5)
        self.assertAlmostEqual(d["correction_success_rate_among_attempted"], 0.5)  # 1 of 2 corrected -> ACCEPT
        self.assertAlmostEqual(d["max_attempts_reached_rate"], 0.25)

    def test_empty_input(self) -> None:
        d = eval_mod.decision_rate_breakdown([], [], [], [])
        self.assertEqual(d["n"], 0)


class TestCategorizeCorrectionOutcome(unittest.TestCase):
    def test_hallucinated_corrected_successfully(self) -> None:
        self.assertEqual(
            eval_mod.categorize_correction_outcome(True, True, "ACCEPT"), "hallucinated_corrected_successfully",
        )

    def test_hallucinated_missed(self) -> None:
        self.assertEqual(eval_mod.categorize_correction_outcome(True, False, "ACCEPT"), "hallucinated_missed")

    def test_hallucinated_still_flagged(self) -> None:
        self.assertEqual(eval_mod.categorize_correction_outcome(True, True, "ABSTAIN"), "hallucinated_still_flagged")
        self.assertEqual(eval_mod.categorize_correction_outcome(True, False, "ABSTAIN"), "hallucinated_still_flagged")

    def test_not_hallucinated_preserved(self) -> None:
        self.assertEqual(eval_mod.categorize_correction_outcome(False, False, "ACCEPT"), "not_hallucinated_preserved")
        self.assertEqual(eval_mod.categorize_correction_outcome(False, True, "ACCEPT"), "not_hallucinated_preserved")

    def test_not_hallucinated_degraded(self) -> None:
        self.assertEqual(eval_mod.categorize_correction_outcome(False, False, "ABSTAIN"), "not_hallucinated_degraded")


class TestUnnecessaryCorrection(unittest.TestCase):
    def test_flags_correction_on_non_hallucinated(self) -> None:
        self.assertTrue(eval_mod.is_unnecessary_correction(False, True))

    def test_does_not_flag_correction_on_hallucinated(self) -> None:
        self.assertFalse(eval_mod.is_unnecessary_correction(True, True))

    def test_does_not_flag_when_no_correction_occurred(self) -> None:
        self.assertFalse(eval_mod.is_unnecessary_correction(False, False))


# --- Structural gold-isolation guard ------------------------------------

class TestEvaluationPackageImportSafety(unittest.TestCase):
    """`claimguard.evaluation` must never import an inference-path module -
    that would create a path for gold labels to reach a model call."""

    def test_no_inference_module_imports(self) -> None:
        import claimguard.evaluation as pkg

        banned = ("claimguard.retrieval", "claimguard.reranking", "claimguard.verification",
                  "claimguard.decision", "claimguard.generation", "claimguard.correction")
        for path in Path(pkg.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            for b in banned:
                self.assertFalse(
                    any(m.startswith(b) for m in imported), f"{path} imports inference module {b}: {imported}",
                )


# --- Behavioral gold-isolation guard on the actual orchestration script --

def _load_evaluate_ragtruth_module():
    script_path = Path(cg_config.PROJECT_ROOT) / "scripts" / "evaluate_ragtruth.py"
    spec = importlib.util.spec_from_file_location("evaluate_ragtruth_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGoldLabelIsolationInOrchestration(unittest.TestCase):
    """Confirms `run_one_response` (the function that calls
    `run_correction_loop` for all three conditions) never passes the gold
    `has_hallucination`/`labels` fields into any inference call - a
    behavioral test on the REAL orchestration code, not just a signature
    check."""

    def test_gold_fields_never_appear_in_run_correction_loop_calls(self) -> None:
        eval_script = _load_evaluate_ragtruth_module()

        sentinel_label_text = "SENTINEL_GOLD_SPAN_TEXT_MUST_NEVER_LEAK"
        fake_record = {
            "response_id": 999999, "source_id": 888888, "task_type": "QA", "model": "gpt-4-0613",
            "temperature": 0.7, "quality": "good",
            "prompt": "What is the capital of France?",
            "response_text": "The capital of France is Paris.",
            "has_hallucination": True,
            "labels": [{"text": sentinel_label_text, "start": 0, "end": 10, "label_type": "Evident Conflict"}],
        }

        pipeline_cfg = {"retrieval_top_k": 20, "rerank_top_n": 5, "verifier_max_length": 256}
        policy_cfg = {"entailment_threshold": 0.5, "contradiction_threshold": 0.5}
        gen_cfg = {"max_new_tokens": 256, "seed": 42}

        with mock.patch.object(eval_script.correction_mod, "run_correction_loop") as mocked:
            mocked.return_value = mock.MagicMock(
                status="completed", final_decision="ACCEPT", termination_reason="accepted",
                correction_occurred=False, total_attempts=1, max_correction_attempts=2,
                attempts=[mock.MagicMock(mapped_decision="ACCEPT", step16_confidence=0.9, evidence=[],
                                          latency=mock.MagicMock(generation_seconds=0.0, retrieval_seconds=0.0,
                                                                  reranking_seconds=0.0, verification_seconds=0.0))],
                final_answer="Paris.", total_latency_seconds=0.1, failure=None,
            )
            models = (object(), object(), 16, object(), object(), object(), object(), "fake/model")
            eval_script.run_one_response(models, fake_record, pipeline_cfg, policy_cfg, gen_cfg, max_attempts=2)

        self.assertEqual(mocked.call_count, 3, "expected exactly 3 run_correction_loop calls (B, C, D)")
        for call in mocked.call_args_list:
            all_args = list(call.args) + list(call.kwargs.values())
            for arg in all_args:
                self.assertNotEqual(arg, True, "gold has_hallucination=True boolean leaked into an inference call arg")
                self.assertNotIsInstance(arg, list, "labels list leaked into an inference call arg")
                if isinstance(arg, str):
                    self.assertNotIn(sentinel_label_text, arg, "gold label text leaked into an inference call arg")

    def test_output_record_still_contains_gold_fields_for_scoring(self) -> None:
        """The OUTPUT record (used only by the analysis script, after
        inference) legitimately DOES carry gold fields - confirms the
        isolation is about the inference CALL, not the result record."""
        eval_script = _load_evaluate_ragtruth_module()
        fake_record = {
            "response_id": 1, "source_id": 2, "task_type": "QA", "model": "gpt-4-0613",
            "temperature": 0.7, "quality": "good", "prompt": "Q?", "response_text": "A.",
            "has_hallucination": True, "labels": [{"text": "x", "start": 0, "end": 1}],
        }
        pipeline_cfg = {"retrieval_top_k": 20, "rerank_top_n": 5, "verifier_max_length": 256}
        policy_cfg = {"entailment_threshold": 0.5, "contradiction_threshold": 0.5}
        gen_cfg = {"max_new_tokens": 256, "seed": 42}
        with mock.patch.object(eval_script.correction_mod, "run_correction_loop") as mocked:
            mocked.return_value = mock.MagicMock(
                status="completed", final_decision="ACCEPT", termination_reason="accepted",
                correction_occurred=False, total_attempts=1, max_correction_attempts=2, attempts=[],
                final_answer="A.", total_latency_seconds=0.1, failure=None,
            )
            models = (object(), object(), 16, object(), object(), object(), object(), "fake/model")
            out = eval_script.run_one_response(models, fake_record, pipeline_cfg, policy_cfg, gen_cfg, max_attempts=2)
        self.assertEqual(out["gold_has_hallucination"], True)
        self.assertEqual(out["gold_n_labels"], 1)


# --- initial_candidate extension (Step 19 addition to correction.py) ----

class _FakeRetriever:
    def __init__(self):
        self.call_count = 0

    def retrieve(self, query, top_k):
        self.call_count += 1
        return [
            {"corpus_id": f"C{i}::0", "text": f"evidence {i}", "score": 1.0 - i * 0.01,
             "page_id": "P", "sentence_id": i, "corpus_version": "test"}
            for i in range(5)
        ]


class _FakeReranker:
    def predict(self, pairs, batch_size=None):
        return [float(len(pairs) - i) for i in range(len(pairs))]


class TestInitialCandidateExtension(unittest.TestCase):
    def test_none_preserves_generation_behavior(self) -> None:
        """Default (initial_candidate=None) must still call the generator -
        regression guard for Step 18 behavior."""
        import torch as _torch

        class _Model:
            def parameters(self):
                yield _torch.zeros(1)

            def generate(self, input_ids, attention_mask, **kw):
                return _torch.cat([input_ids, _torch.tensor([[7]])], dim=1)

        class _Tok:
            def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True, enable_thinking=None):
                ids = _torch.arange(1, 4).unsqueeze(0)
                return {"input_ids": ids, "attention_mask": _torch.ones_like(ids)}

            def decode(self, ids, skip_special_tokens=True):
                return "generated answer"

        with mock.patch("claimguard.verification.verify.verify_pair",
                         return_value={"entailment_probability": 0.95, "contradiction_probability": 0.05,
                                       "verifier_label": "entailment", "pipeline_label": "supported"}):
            result = correction_mod.run_correction_loop(
                _FakeRetriever(), _FakeReranker(), 16, object(), object(), _Model(), _Tok(),
                "a question", initial_candidate=None, generator_model_identifier="fake/model",
            )
        self.assertEqual(result.attempts[0].candidate_answer, "generated answer")  # proves the generation path ran

    def test_initial_candidate_skips_generation_on_attempt_zero(self) -> None:
        with mock.patch("claimguard.verification.verify.verify_pair",
                         return_value={"entailment_probability": 0.95, "contradiction_probability": 0.05,
                                       "verifier_label": "entailment", "pipeline_label": "supported"}):
            result = correction_mod.run_correction_loop(
                _FakeRetriever(), _FakeReranker(), 16, object(), object(), None, None,
                "a question", initial_candidate="pre-supplied answer text", generator_model_identifier="fake/model",
            )
        self.assertEqual(result.attempts[0].candidate_answer, "pre-supplied answer text")
        self.assertEqual(result.attempts[0].latency.generation_seconds, 0.0)
        self.assertEqual(result.final_decision, "ACCEPT")

    def test_max_correction_attempts_zero_yields_single_verify_only_pass(self) -> None:
        with mock.patch("claimguard.verification.verify.verify_pair",
                         return_value={"entailment_probability": 0.1, "contradiction_probability": 0.9,
                                       "verifier_label": "contradiction", "pipeline_label": "contradicted"}):
            result = correction_mod.run_correction_loop(
                _FakeRetriever(), _FakeReranker(), 16, object(), object(), None, None,
                "a question", initial_candidate="an answer that would trigger correction",
                max_correction_attempts=0, generator_model_identifier="fake/model",
            )
        self.assertEqual(result.total_attempts, 1, "verify-only mode must never regenerate")
        self.assertEqual(result.attempts[0].mapped_decision, "CORRECT", "the raw single-pass decision is preserved")
        self.assertEqual(result.final_decision, "ABSTAIN", "budget-exhausted CORRECT converts to terminal ABSTAIN")
        self.assertEqual(result.termination_reason, "max_attempts_reached")


# --- RAGTruth test-set verification (real data, no gold access issue -
# this is legitimately evaluation-layer code reading the dataset module) --

class TestRAGTruthTestSetDimensions(unittest.TestCase):
    """Confirms the numbers Step 19 instructions require verifying (not
    assuming) from the real dataset module."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.records = ragtruth_mod.load_normalized()
        except FileNotFoundError:
            cls.records = None

    def setUp(self) -> None:
        if self.records is None:
            raise unittest.SkipTest("Raw RAGTruth data not available in this environment.")

    def test_test_split_has_2700_responses(self) -> None:
        test = ragtruth_mod.get_eval_set(self.records)
        self.assertEqual(len(test), 2700)

    def test_test_split_has_450_source_items(self) -> None:
        test = ragtruth_mod.get_eval_set(self.records)
        self.assertEqual(len({r["source_id"] for r in test}), 450)

    def test_every_source_has_exactly_6_responses(self) -> None:
        from collections import Counter

        test = ragtruth_mod.get_eval_set(self.records)
        counts = Counter(r["source_id"] for r in test)
        self.assertTrue(all(c == 6 for c in counts.values()))

    def test_no_train_test_leakage(self) -> None:
        ragtruth_mod.assert_no_train_test_leakage(self.records)  # must not raise

    def test_raw_ragtruth_files_not_mutated_by_loading(self) -> None:
        """Loading/joining/normalizing must never write to data/raw/ragtruth/ -
        confirmed by hashing both files before and after a full load."""
        response_path = ragtruth_mod.RAW_DIR / "response.jsonl"
        source_path = ragtruth_mod.RAW_DIR / "source_info.jsonl"

        def _hash(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        before = (_hash(response_path), _hash(source_path))
        ragtruth_mod.load_normalized()
        ragtruth_mod.get_eval_set(self.records)
        after = (_hash(response_path), _hash(source_path))
        self.assertEqual(before, after)

    def test_dataset_module_exposes_no_write_function(self) -> None:
        write_like = [n for n in dir(ragtruth_mod) if any(w in n.lower() for w in ("save", "write", "dump"))]
        self.assertEqual(write_like, [], f"claimguard.datasets.ragtruth unexpectedly exposes: {write_like}")


if __name__ == "__main__":
    unittest.main()
