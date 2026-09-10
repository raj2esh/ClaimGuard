"""Tests for Step 20 (TruthfulQA evaluation): the pure MC1/MC2/MC0
arithmetic in `claimguard.evaluation.truthfulqa_eval`, the new
`score_choice_log_likelihood` function in `claimguard.generation.qwen3`,
and a behavioral gold-label-isolation guard on the actual evaluation
orchestration code in `scripts/evaluate_truthfulqa.py`.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import unittest
from pathlib import Path
from unittest import mock

from claimguard import config as cg_config
from claimguard.datasets import truthfulqa as truthfulqa_mod
from claimguard.evaluation import truthfulqa_eval as eval_mod
from claimguard.generation import qwen3 as qwen3_mod


# --- MC1/MC2/MC0 pure arithmetic ----------------------------------------

class TestMC1Score(unittest.TestCase):
    def test_correct_choice_has_highest_likelihood(self) -> None:
        lls = {"A": -1.0, "B": -5.0, "C": -3.0}
        targets = {"A": 1, "B": 0, "C": 0}
        self.assertEqual(eval_mod.compute_mc1_score(lls, targets), 1)

    def test_incorrect_choice_has_highest_likelihood(self) -> None:
        lls = {"A": -5.0, "B": -1.0, "C": -3.0}
        targets = {"A": 1, "B": 0, "C": 0}
        self.assertEqual(eval_mod.compute_mc1_score(lls, targets), 0)

    def test_tie_breaks_deterministically(self) -> None:
        lls = {"A": -1.0, "B": -1.0}
        targets = {"A": 1, "B": 0}
        # max() picks the first max encountered in dict iteration order - deterministic, not random.
        result1 = eval_mod.compute_mc1_score(lls, targets)
        result2 = eval_mod.compute_mc1_score(lls, targets)
        self.assertEqual(result1, result2)


class TestMC0Score(unittest.TestCase):
    def test_identical_rule_to_mc1_on_two_choices(self) -> None:
        lls = {"correct": -1.0, "incorrect": -4.0}
        targets = {"correct": 1, "incorrect": 0}
        self.assertEqual(eval_mod.compute_mc0_score(lls, targets), 1)

    def test_wrong_when_incorrect_choice_favored(self) -> None:
        lls = {"correct": -4.0, "incorrect": -1.0}
        targets = {"correct": 1, "incorrect": 0}
        self.assertEqual(eval_mod.compute_mc0_score(lls, targets), 0)


class TestMC2Score(unittest.TestCase):
    def test_all_mass_on_correct_choices_scores_near_one(self) -> None:
        lls = {"correct1": 0.0, "correct2": 0.0, "wrong1": -100.0, "wrong2": -100.0}
        targets = {"correct1": 1, "correct2": 1, "wrong1": 0, "wrong2": 0}
        score = eval_mod.compute_mc2_score(lls, targets)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_all_mass_on_wrong_choices_scores_near_zero(self) -> None:
        lls = {"correct1": -100.0, "wrong1": 0.0, "wrong2": 0.0}
        targets = {"correct1": 1, "wrong1": 0, "wrong2": 0}
        score = eval_mod.compute_mc2_score(lls, targets)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_uniform_likelihoods_scores_proportional_to_correct_count(self) -> None:
        # 2 correct, 2 incorrect, all EQUAL likelihood -> softmax is uniform -> correct mass = 2/4 = 0.5.
        lls = {"c1": -2.0, "c2": -2.0, "w1": -2.0, "w2": -2.0}
        targets = {"c1": 1, "c2": 1, "w1": 0, "w2": 0}
        score = eval_mod.compute_mc2_score(lls, targets)
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_score_is_continuous_not_binary(self) -> None:
        lls = {"c1": -1.0, "w1": -1.5}
        targets = {"c1": 1, "w1": 0}
        score = eval_mod.compute_mc2_score(lls, targets)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class TestAggregateMCScores(unittest.TestCase):
    def test_means_and_denominators(self) -> None:
        agg = eval_mod.aggregate_mc_scores(mc1_scores=[1, 0, 1], mc2_scores=[0.8, 0.2], mc0_scores=[1])
        self.assertAlmostEqual(agg["mc1_accuracy"], 2 / 3)
        self.assertEqual(agg["mc1_n"], 3)
        self.assertAlmostEqual(agg["mc2_accuracy"], 0.5)
        self.assertEqual(agg["mc2_n"], 2)
        self.assertEqual(agg["mc0_accuracy"], 1.0)
        self.assertEqual(agg["mc0_n"], 1)

    def test_empty_inputs_yield_none_not_zero(self) -> None:
        agg = eval_mod.aggregate_mc_scores([], [], [])
        self.assertIsNone(agg["mc1_accuracy"])
        self.assertIsNone(agg["mc2_accuracy"])
        self.assertIsNone(agg["mc0_accuracy"])
        self.assertEqual(agg["mc1_n"], 0)


class TestDecisionRateBreakdownReused(unittest.TestCase):
    """Confirms truthfulqa_eval re-exports (not duplicates) Step 19's
    dataset-agnostic decision_rate_breakdown."""

    def test_reexported_function_identity(self) -> None:
        from claimguard.evaluation import ragtruth_eval as ragtruth_eval_mod
        self.assertIs(eval_mod.decision_rate_breakdown, ragtruth_eval_mod.decision_rate_breakdown)


# --- score_choice_log_likelihood (new qwen3.py function) ----------------

class FakeCausalLMForScoring:
    """Uniform logits everywhere -> every token's log-softmax is the KNOWN
    constant -log(vocab_size), giving a hand-computable expected total."""

    def __init__(self, vocab_size: int = 10):
        self.vocab_size = vocab_size

    def parameters(self):
        import torch
        yield torch.zeros(1)

    def __call__(self, input_ids, attention_mask):
        import torch

        class _Output:
            pass

        out = _Output()
        out.logits = torch.zeros(1, input_ids.shape[1], self.vocab_size)
        return out


class FakeTokenizerForScoring:
    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt",
                             return_dict=True, enable_thinking=None):
        import torch
        n_tokens = sum(len(m["content"].split()) for m in messages) + 1
        ids = torch.arange(1, n_tokens + 1).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        import torch
        # Token ids must stay within FakeCausalLMForScoring's vocab_size (10 in these
        # tests) - using small ids here, distinct from the prompt's own id range but
        # still in-vocabulary, avoids a real (and correctly-raised) IndexError.
        n_tokens = max(1, len(text.split()))
        ids = torch.arange(2, 2 + n_tokens).unsqueeze(0)
        return {"input_ids": ids}


class TestScoreChoiceLogLikelihood(unittest.TestCase):
    def test_matches_hand_computed_expected_value(self) -> None:
        vocab_size = 10
        model = FakeCausalLMForScoring(vocab_size=vocab_size)
        tokenizer = FakeTokenizerForScoring()
        result = qwen3_mod.score_choice_log_likelihood(model, tokenizer, "a question", "a b c")
        expected_per_token = -math.log(vocab_size)
        self.assertEqual(result["num_tokens"], 3)
        self.assertAlmostEqual(result["log_likelihood"], 3 * expected_per_token, places=5)
        self.assertAlmostEqual(result["avg_log_likelihood"], expected_per_token, places=5)

    def test_longer_choice_has_more_negative_log_likelihood(self) -> None:
        model = FakeCausalLMForScoring(vocab_size=10)
        tokenizer = FakeTokenizerForScoring()
        short = qwen3_mod.score_choice_log_likelihood(model, tokenizer, "q", "a")
        long = qwen3_mod.score_choice_log_likelihood(model, tokenizer, "q", "a b c d")
        self.assertLess(long["log_likelihood"], short["log_likelihood"])

    def test_empty_choice_returns_zero(self) -> None:
        model = FakeCausalLMForScoring(vocab_size=10)
        tokenizer = FakeTokenizerForScoring()

        class _EmptyTokenizer(FakeTokenizerForScoring):
            def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
                import torch
                return {"input_ids": torch.zeros(1, 0, dtype=torch.long)}

        result = qwen3_mod.score_choice_log_likelihood(model, _EmptyTokenizer(), "q", "")
        self.assertEqual(result["log_likelihood"], 0.0)
        self.assertEqual(result["num_tokens"], 0)
        self.assertIsNone(result["avg_log_likelihood"])

    def test_no_gold_label_parameter(self) -> None:
        sig = inspect.signature(qwen3_mod.score_choice_log_likelihood)
        param_names = {p.lower() for p in sig.parameters}
        forbidden = ["gold", "label", "correct", "incorrect", "target"]
        offending = [p for p in param_names for f in forbidden if f in p]
        self.assertEqual(offending, [])


# --- Behavioral gold-isolation guard on the real orchestration script ---

def _load_evaluate_truthfulqa_module():
    script_path = Path(cg_config.PROJECT_ROOT) / "scripts" / "evaluate_truthfulqa.py"
    spec = importlib.util.spec_from_file_location("evaluate_truthfulqa_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGoldLabelIsolationInOrchestration(unittest.TestCase):
    def test_gold_targets_never_appear_in_inference_calls(self) -> None:
        eval_script = _load_evaluate_truthfulqa_module()

        sentinel = "SENTINEL_CORRECT_ANSWER_MUST_NEVER_LEAK"
        fake_record = {
            "question_id": 1, "category": "Misconceptions", "type": "Non-Adversarial",
            "question": "What happens if you crack your knuckles?",
            "best_answer": sentinel, "best_incorrect_answer": "x",
            "correct_answers": [sentinel], "incorrect_answers": ["x"],
            "mc0_targets": {sentinel: 1, "x": 0},
            "mc1_targets": {sentinel: 1, "x": 0, "y": 0},
            "mc2_targets": {sentinel: 1, "z": 1, "x": 0},
            "has_mc_data": True,
        }
        pipeline_cfg = {"retrieval_top_k": 20, "rerank_top_n": 5, "verifier_max_length": 256}
        policy_cfg = {"entailment_threshold": 0.5, "contradiction_threshold": 0.5}
        gen_cfg = {"max_new_tokens": 256, "seed": 42}

        fake_result = mock.MagicMock(
            status="completed", final_decision="ACCEPT", termination_reason="accepted",
            correction_occurred=False, total_attempts=1, max_correction_attempts=2,
            attempts=[mock.MagicMock(candidate_answer="Nothing harmful happens.", mapped_decision="ACCEPT",
                                      step16_decision="SUPPORTED", step16_confidence=0.9, evidence=[],
                                      latency=mock.MagicMock(generation_seconds=0.0, retrieval_seconds=0.0,
                                                              reranking_seconds=0.0, verification_seconds=0.0))],
            final_answer="Nothing harmful happens.", total_latency_seconds=0.1, failure=None,
        )
        fake_score_result = {"log_likelihood": -1.0, "num_tokens": 1, "avg_log_likelihood": -1.0,
                              "chat_template_fallback_triggered": False}

        with mock.patch.object(eval_script.correction_mod, "run_correction_loop", return_value=fake_result) as mocked_loop, \
             mock.patch.object(eval_script.qwen3_mod, "score_choice_log_likelihood", return_value=fake_score_result) as mocked_score:
            models = (object(), object(), 16, object(), object(), object(), object(), "fake/model")
            out = eval_script.run_one_question(models, fake_record, pipeline_cfg, policy_cfg, gen_cfg, max_attempts=2)

        for call in mocked_loop.call_args_list:
            for arg in list(call.args) + list(call.kwargs.values()):
                if isinstance(arg, str):
                    self.assertNotIn(sentinel, arg, "gold answer text leaked into run_correction_loop")
                self.assertNotIsInstance(arg, dict, "mc targets dict leaked into run_correction_loop")

        # score_choice_log_likelihood legitimately receives CHOICE TEXT (which happens to equal the
        # sentinel string here, since that's the correct answer) - but it must NEVER receive the
        # integer 0/1 LABEL, and the question passed must be the plain question text.
        for call in mocked_score.call_args_list:
            for arg in list(call.args) + list(call.kwargs.values()):
                self.assertNotIsInstance(arg, dict, "a targets dict (with labels) leaked into score_choice_log_likelihood")

        self.assertEqual(out["gold_mc1_targets"][sentinel], 1)  # legitimately present in the OUTPUT record

    def test_output_record_preserves_gold_for_scoring(self) -> None:
        eval_script = _load_evaluate_truthfulqa_module()
        fake_record = {
            "question_id": 2, "category": "Law", "type": "Adversarial", "question": "Q?",
            "best_answer": "A", "best_incorrect_answer": "B", "correct_answers": ["A"],
            "incorrect_answers": ["B"], "mc0_targets": {"A": 1, "B": 0}, "mc1_targets": None,
            "mc2_targets": None, "has_mc_data": True,
        }
        pipeline_cfg = {"retrieval_top_k": 20, "rerank_top_n": 5, "verifier_max_length": 256}
        policy_cfg = {"entailment_threshold": 0.5, "contradiction_threshold": 0.5}
        gen_cfg = {"max_new_tokens": 256, "seed": 42}
        fake_result = mock.MagicMock(
            status="completed", final_decision="ACCEPT", termination_reason="accepted",
            correction_occurred=False, total_attempts=1, max_correction_attempts=2, attempts=[],
            final_answer="A.", total_latency_seconds=0.1, failure=None,
        )
        with mock.patch.object(eval_script.correction_mod, "run_correction_loop", return_value=fake_result), \
             mock.patch.object(eval_script.qwen3_mod, "score_choice_log_likelihood",
                                return_value={"log_likelihood": -1.0, "num_tokens": 1, "avg_log_likelihood": -1.0,
                                              "chat_template_fallback_triggered": False}):
            models = (object(), object(), 16, object(), object(), object(), object(), "fake/model")
            out = eval_script.run_one_question(models, fake_record, pipeline_cfg, policy_cfg, gen_cfg, max_attempts=2)
        self.assertEqual(out["gold_mc0_targets"], {"A": 1, "B": 0})
        self.assertEqual(out["question"], "Q?")


# --- Real TruthfulQA dataset dimension checks ---------------------------

class TestTruthfulQADatasetDimensions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.records = truthfulqa_mod.load_normalized()
        except FileNotFoundError:
            cls.records = None

    def setUp(self) -> None:
        if self.records is None:
            raise unittest.SkipTest("Raw TruthfulQA data not available in this environment.")

    def test_exactly_790_questions(self) -> None:
        self.assertEqual(len(self.records), 790)

    def test_exactly_37_categories(self) -> None:
        categories = {r["category"] for r in self.records}
        self.assertEqual(len(categories), 37)

    def test_no_duplicate_questions(self) -> None:
        dupes = truthfulqa_mod.find_duplicate_questions(self.records)
        self.assertEqual(dupes, {})

    def test_no_split_field_present(self) -> None:
        for r in self.records[:50]:
            self.assertNotIn("split", r)

    def test_module_exposes_no_eval_set_or_training_pool_accessor(self) -> None:
        # Unlike claimguard.datasets.ragtruth, this module must NOT invent a split.
        self.assertFalse(hasattr(truthfulqa_mod, "get_eval_set"))
        self.assertFalse(hasattr(truthfulqa_mod, "get_training_pool"))

    def test_mc0_has_exactly_two_options_when_present(self) -> None:
        with_mc0 = [r for r in self.records if r.get("mc0_targets")]
        self.assertGreater(len(with_mc0), 0)
        for r in with_mc0[:50]:
            self.assertEqual(len(r["mc0_targets"]), 2)
            self.assertEqual(sum(r["mc0_targets"].values()), 1)

    def test_mc2_can_have_multiple_correct_options(self) -> None:
        multi_correct = [r for r in self.records if r.get("mc2_targets") and sum(r["mc2_targets"].values()) > 1]
        self.assertGreater(len(multi_correct), 0)

    def test_correct_and_incorrect_answers_associated_with_question(self) -> None:
        for r in self.records[:50]:
            self.assertIn("correct_answers", r)
            self.assertIn("incorrect_answers", r)
            self.assertEqual(r["question"], r["question"])  # sanity: same record, no cross-question mixing

    def test_dataset_module_exposes_no_write_function(self) -> None:
        write_like = [n for n in dir(truthfulqa_mod) if any(w in n.lower() for w in ("save", "write", "dump"))]
        self.assertEqual(write_like, [])


if __name__ == "__main__":
    unittest.main()
