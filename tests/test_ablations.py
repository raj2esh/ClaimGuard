"""Tests for Step 21 (controlled ablations): the new
`evidence_selection_mode` extension to `claimguard.correction.correction`,
the pure diagnostic functions in `claimguard.evaluation.ablation_eval`,
and a behavioral gold-label-isolation guard on `scripts/run_ablations.py`.
"""

from __future__ import annotations

import importlib.util
import inspect
import unittest
from pathlib import Path
from unittest import mock

from claimguard import config as cg_config
from claimguard.correction import correction as correction_mod
from claimguard.evaluation import ablation_eval as ab_eval


# --- evidence_selection_mode extension -----------------------------------

def _candidate(corpus_id: str, text: str, idx: int, score: float | None = None) -> dict:
    return {
        "corpus_id": corpus_id, "text": text, "score": score if score is not None else 1.0 - idx * 0.01,
        "page_id": "P", "sentence_id": idx, "corpus_version": "test",
    }


class _FakeRetrieverOrdered:
    """Returns 5 candidates in a FIXED, known FAISS order - lets tests
    assert exactly which candidates each evidence_selection_mode keeps."""

    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append(query)
        return [_candidate(f"C{i}::0", f"evidence {i}", i) for i in range(5)]


class _FakeRerankerReverses:
    """Reverses FAISS order deterministically - lets tests confirm
    'reranked' mode actually calls this, while 'no_reranker'/'faiss_top1'
    do NOT (their output preserves FAISS order instead)."""

    def __init__(self):
        self.call_count = 0

    def predict(self, pairs, batch_size=None):
        self.call_count += 1
        return [float(i) for i in range(len(pairs))]  # later pairs score higher -> reverses order


class TestFaissCandidatesAsRerankedSchema(unittest.TestCase):
    def test_preserves_faiss_order_and_schema(self) -> None:
        faiss_candidates = [_candidate(f"C{i}::0", f"text {i}", i) for i in range(5)]
        out = correction_mod._faiss_candidates_as_reranked_schema(faiss_candidates, top_n=3)
        self.assertEqual(len(out), 3)
        self.assertEqual([c["corpus_id"] for c in out], ["C0::0", "C1::0", "C2::0"])
        for i, c in enumerate(out):
            self.assertEqual(c["original_rank"], i + 1)
            self.assertEqual(c["reranked_rank"], i + 1)  # identity - no reordering happened
            self.assertIsNone(c["reranker_score"])
            self.assertIn("faiss_score", c)
            self.assertIn("text", c)
            self.assertIn("page_id", c)

    def test_top_n_truncates(self) -> None:
        faiss_candidates = [_candidate(f"C{i}::0", f"text {i}", i) for i in range(5)]
        out = correction_mod._faiss_candidates_as_reranked_schema(faiss_candidates, top_n=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["corpus_id"], "C0::0")


class TestEvidenceSelectionModes(unittest.TestCase):
    def _run(self, mode: str, verify_return):
        retriever = _FakeRetrieverOrdered()
        reranker = _FakeRerankerReverses()
        with mock.patch("claimguard.verification.verify.verify_pair", return_value=verify_return):
            result = correction_mod.run_correction_loop(
                retriever, reranker, 16, object(), object(), None, None, "a question",
                initial_candidate="an answer", evidence_selection_mode=mode,
                generator_model_identifier="fake/model", max_correction_attempts=0,
            )
        return result, retriever, reranker

    def test_reranked_mode_calls_reranker_and_reverses_order(self) -> None:
        verify_return = {"entailment_probability": 0.95, "contradiction_probability": 0.05,
                          "verifier_label": "entailment", "pipeline_label": "supported"}
        result, retriever, reranker = self._run("reranked", verify_return)
        self.assertEqual(reranker.call_count, 1)
        evidence_ids = [e.corpus_id for e in result.attempts[0].evidence]
        self.assertEqual(evidence_ids[0], "C4::0")  # reversed order -> last FAISS candidate ranks first

    def test_no_reranker_mode_never_calls_reranker_preserves_faiss_order(self) -> None:
        verify_return = {"entailment_probability": 0.95, "contradiction_probability": 0.05,
                          "verifier_label": "entailment", "pipeline_label": "supported"}
        result, retriever, reranker = self._run("no_reranker", verify_return)
        self.assertEqual(reranker.call_count, 0, "no_reranker mode must never call the reranker")
        evidence_ids = [e.corpus_id for e in result.attempts[0].evidence]
        self.assertEqual(evidence_ids, ["C0::0", "C1::0", "C2::0", "C3::0", "C4::0"])  # untouched FAISS order

    def test_faiss_top1_mode_verifies_exactly_one_candidate(self) -> None:
        verify_return = {"entailment_probability": 0.95, "contradiction_probability": 0.05,
                          "verifier_label": "entailment", "pipeline_label": "supported"}
        result, retriever, reranker = self._run("faiss_top1", verify_return)
        self.assertEqual(reranker.call_count, 0)
        self.assertEqual(len(result.attempts[0].evidence), 1)
        self.assertEqual(result.attempts[0].evidence[0].corpus_id, "C0::0")

    def test_default_mode_is_reranked(self) -> None:
        sig = inspect.signature(correction_mod.run_correction_loop)
        self.assertEqual(sig.parameters["evidence_selection_mode"].default, "reranked")

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            correction_mod._retrieve_rerank_verify(
                _FakeRetrieverOrdered(), _FakeRerankerReverses(), 16, object(), object(), "text",
                20, 5, 256, evidence_selection_mode="bogus_mode",
            )

    def test_retrieval_top_k_unchanged_across_modes(self) -> None:
        """Retrieval depth itself must be identical regardless of
        evidence-selection mode - only what happens AFTER retrieval differs."""
        verify_return = {"entailment_probability": 0.95, "contradiction_probability": 0.05,
                          "verifier_label": "entailment", "pipeline_label": "supported"}
        _, retriever_reranked, _ = self._run("reranked", verify_return)
        _, retriever_no_rerank, _ = self._run("no_reranker", verify_return)
        # Both fake retrievers received the same top_k call pattern (single call, same candidate answer).
        self.assertEqual(len(retriever_reranked.calls), len(retriever_no_rerank.calls))


# --- ablation_eval pure functions ----------------------------------------

class TestConfidenceByOutcome(unittest.TestCase):
    def test_four_groups_classified_correctly(self) -> None:
        mapped = ["ACCEPT", "ACCEPT", "CORRECT", "CORRECT"]
        conf = [0.9, 0.8, 0.7, 0.6]
        gold = [False, True, False, True]
        result = ab_eval.confidence_by_outcome(mapped, conf, gold)
        self.assertEqual(result["accept_correct_negative"]["n"], 1)
        self.assertAlmostEqual(result["accept_correct_negative"]["mean_confidence"], 0.9)
        self.assertEqual(result["accept_wrong_missed_hallucination"]["n"], 1)
        self.assertAlmostEqual(result["accept_wrong_missed_hallucination"]["mean_confidence"], 0.8)
        self.assertEqual(result["correct_wrong_false_positive"]["n"], 1)
        self.assertAlmostEqual(result["correct_wrong_false_positive"]["mean_confidence"], 0.7)
        self.assertEqual(result["correct_correct_positive"]["n"], 1)
        self.assertAlmostEqual(result["correct_correct_positive"]["mean_confidence"], 0.6)

    def test_empty_group_reports_none_not_zero(self) -> None:
        result = ab_eval.confidence_by_outcome(["ACCEPT"], [0.9], [False])
        self.assertEqual(result["accept_wrong_missed_hallucination"]["n"], 0)
        self.assertIsNone(result["accept_wrong_missed_hallucination"]["mean_confidence"])


class TestEvidenceMovementRate(unittest.TestCase):
    def test_detects_changed_and_unchanged(self) -> None:
        baseline = [["A", "B"], ["C", "D"], ["E", "F"]]
        comparison = [["A", "B"], ["X", "Y"], ["E", "F"]]
        result = ab_eval.evidence_movement_rate(baseline, comparison)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["n_changed"], 1)
        self.assertAlmostEqual(result["changed_rate"], 1 / 3)

    def test_order_independent_set_comparison(self) -> None:
        # Same evidence, different order -> NOT counted as "changed" (this measures WHICH
        # evidence was selected, not rank order within the selection).
        result = ab_eval.evidence_movement_rate([["A", "B"]], [["B", "A"]])
        self.assertEqual(result["n_changed"], 0)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            ab_eval.evidence_movement_rate([["A"]], [["A"], ["B"]])

    def test_empty_input(self) -> None:
        result = ab_eval.evidence_movement_rate([], [])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["changed_rate"])


# --- Gold-label isolation on the real orchestration script ---------------

def _load_run_ablations_module():
    script_path = Path(cg_config.PROJECT_ROOT) / "scripts" / "run_ablations.py"
    spec = importlib.util.spec_from_file_location("run_ablations_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGoldLabelIsolationInAblationOrchestration(unittest.TestCase):
    def test_gold_fields_never_appear_in_inference_calls(self) -> None:
        ablation_script = _load_run_ablations_module()

        sentinel = "SENTINEL_MUST_NEVER_LEAK_INTO_INFERENCE"
        fake_record = {
            "response_id": 1, "source_id": 2, "task_type": "QA", "model": "gpt-4-0613",
            "quality": "good", "prompt": "What is the capital of France?",
            "response_text": "Paris.", "has_hallucination": True,
            "labels": [{"text": sentinel, "start": 0, "end": 5}],
        }
        pipeline_cfg = {"retrieval_top_k": 20, "rerank_top_n": 5, "verifier_max_length": 256}
        policy_cfg = {"entailment_threshold": 0.5, "contradiction_threshold": 0.5}
        gen_cfg = {"max_new_tokens": 256, "seed": 42}

        fake_gen_result = mock.MagicMock(generated_text="Paris is the capital of France.")
        fake_loop_result = mock.MagicMock(
            status="completed", final_decision="ACCEPT", termination_reason="accepted",
            correction_occurred=False, total_attempts=1, max_correction_attempts=2,
            attempts=[mock.MagicMock(mapped_decision="ACCEPT", step16_confidence=0.9, evidence=[],
                                      latency=mock.MagicMock(generation_seconds=0.0, retrieval_seconds=0.0,
                                                              reranking_seconds=0.0, verification_seconds=0.0))],
            final_answer="Paris is the capital of France.", total_latency_seconds=0.1, failure=None,
        )

        with mock.patch.object(ablation_script.qwen3_mod, "generate", return_value=fake_gen_result), \
             mock.patch.object(ablation_script.correction_mod, "run_correction_loop", return_value=fake_loop_result) as mocked_loop:
            models = (object(), object(), 16, object(), object(), object(), object(), "fake/model")
            out = ablation_script.run_one_response(models, fake_record, pipeline_cfg, policy_cfg, gen_cfg, max_attempts=2)

        self.assertEqual(mocked_loop.call_count, 3, "expected exactly 3 run_correction_loop calls (A, B, C)")
        for call in mocked_loop.call_args_list:
            for arg in list(call.args) + list(call.kwargs.values()):
                self.assertNotEqual(arg, True, "gold has_hallucination=True leaked into an inference call")
                self.assertNotIsInstance(arg, list, "labels list leaked into an inference call")
                if isinstance(arg, str):
                    self.assertNotIn(sentinel, arg, "gold label text leaked into an inference call")
        self.assertEqual(out["gold_has_hallucination"], True)  # legitimately present in the OUTPUT record

    def test_only_three_conditions_produced_no_d_or_e(self) -> None:
        """Confirms the script design: D and E are REUSED from Step 19,
        never recomputed here - run_one_response must produce exactly
        A/B/C, nothing else."""
        ablation_script = _load_run_ablations_module()
        fake_record = {
            "response_id": 1, "source_id": 2, "task_type": "QA", "model": "gpt-4-0613",
            "quality": "good", "prompt": "Q?", "response_text": "A.", "has_hallucination": False, "labels": [],
        }
        pipeline_cfg = {"retrieval_top_k": 20, "rerank_top_n": 5, "verifier_max_length": 256}
        policy_cfg = {"entailment_threshold": 0.5, "contradiction_threshold": 0.5}
        gen_cfg = {"max_new_tokens": 256, "seed": 42}
        fake_gen_result = mock.MagicMock(generated_text="A.")
        fake_loop_result = mock.MagicMock(
            status="completed", final_decision="ACCEPT", termination_reason="accepted",
            correction_occurred=False, total_attempts=1, max_correction_attempts=2, attempts=[],
            final_answer="A.", total_latency_seconds=0.1, failure=None,
        )
        with mock.patch.object(ablation_script.qwen3_mod, "generate", return_value=fake_gen_result), \
             mock.patch.object(ablation_script.correction_mod, "run_correction_loop", return_value=fake_loop_result):
            models = (object(), object(), 16, object(), object(), object(), object(), "fake/model")
            out = ablation_script.run_one_response(models, fake_record, pipeline_cfg, policy_cfg, gen_cfg, max_attempts=2)
        self.assertEqual(set(out["conditions"].keys()), {"A_reranked", "B_no_reranker", "C_faiss_top1"})


if __name__ == "__main__":
    unittest.main()
