"""Tests for claimguard.correction (Step 18: bounded correction /
regeneration loop). Uses fake retriever/reranker/generator objects and a
mocked `verify_pair` (same pattern as tests/test_verification.py) so the
loop's ORCHESTRATION logic is exercised without any real GPU model. No
FEVER/RAGTruth/TruthfulQA/HaluEval data is used anywhere in this file.
"""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path
from unittest import mock

from claimguard import config as cg_config
from claimguard.correction import correction as correction_mod
from claimguard.correction.types import CorrectionResult
from claimguard.decision import policy as policy_mod


def _correction_source_files() -> list[Path]:
    import claimguard.correction as pkg
    return list(Path(pkg.__file__).parent.glob("*.py"))


class TestRAGTruthTruthfulQAExclusion(unittest.TestCase):
    def test_no_dataset_imports(self) -> None:
        import ast

        for path in _correction_source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                    imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            lowered = {m.lower() for m in imported}
            for banned in ("ragtruth", "truthfulqa", "fever", "halueval"):
                self.assertFalse(any(banned in m for m in lowered), f"{path} imports {banned}: {imported}")


class TestNoGoldAccess(unittest.TestCase):
    def test_run_correction_loop_has_no_gold_parameter(self) -> None:
        sig = inspect.signature(correction_mod.run_correction_loop)
        param_names = {p.lower() for p in sig.parameters}
        forbidden = ["gold", "label", "relevant", "hallucination", "fever", "ragtruth", "truthfulqa", "span"]
        offending = [p for p in param_names for f in forbidden if f in p]
        self.assertEqual(offending, [])

    def test_build_correction_prompt_has_no_gold_parameter(self) -> None:
        sig = inspect.signature(correction_mod.build_correction_prompt)
        param_names = {p.lower() for p in sig.parameters}
        forbidden = ["gold", "label", "relevant", "hallucination", "fever", "ragtruth", "truthfulqa", "span"]
        offending = [p for p in param_names for f in forbidden if f in p]
        self.assertEqual(offending, [])


class TestConfigLoads(unittest.TestCase):
    def test_correction_baseline_config_loads_with_required_keys(self) -> None:
        cfg = cg_config.load_correction_baseline_config()
        self.assertIn("max_correction_attempts", cfg)
        self.assertIn("pipeline", cfg)
        self.assertIn("decision_policy", cfg)
        self.assertIn("generation", cfg)

    def test_correction_baseline_matches_decision_policy_config(self) -> None:
        cfg = cg_config.load_correction_baseline_config()
        dp_cfg = cg_config.load_decision_policy_config()
        self.assertEqual(
            cfg["decision_policy"]["entailment_threshold"], dp_cfg["decision_policy"]["entailment_threshold"],
        )
        self.assertEqual(
            cfg["decision_policy"]["contradiction_threshold"], dp_cfg["decision_policy"]["contradiction_threshold"],
        )

    def test_correction_baseline_matches_generator_baseline_config(self) -> None:
        cfg = cg_config.load_correction_baseline_config()
        gen_cfg = cg_config.load_generator_baseline_config()
        self.assertEqual(cfg["generation"]["max_new_tokens"], gen_cfg["generation"]["max_new_tokens"])
        self.assertEqual(cfg["generation"]["seed"], gen_cfg["generation"]["seed"])
        self.assertFalse(cfg["generation"]["do_sample"])

    def test_missing_config_file_raises(self) -> None:
        with self.assertRaises(cg_config.ConfigError):
            cg_config.load_correction_baseline_config(config_dir=Path("/nonexistent/config/dir"))


# --- Fakes -------------------------------------------------------------

def _candidate(corpus_id: str, text: str, idx: int) -> dict:
    return {
        "corpus_id": corpus_id, "text": text, "score": 1.0 - idx * 0.01,
        "page_id": "P", "sentence_id": idx, "corpus_version": "test",
    }


class CountingRetriever:
    """Returns a fresh, call-count-suffixed candidate set every call, so
    tests can distinguish attempt-0 evidence from attempt-1+ evidence and
    confirm retrieval actually reran rather than reusing stale results."""

    def __init__(self):
        self.call_count = 0
        self.queries: list[str] = []

    def retrieve(self, query, top_k):
        self.queries.append(query)
        call_idx = self.call_count
        self.call_count += 1
        return [
            _candidate(f"C{i}_call{call_idx}::0", f"evidence {i} for call {call_idx}", i)
            for i in range(5)
        ]


class EmptyRetriever:
    def __init__(self):
        self.call_count = 0

    def retrieve(self, query, top_k):
        self.call_count += 1
        return []


class OrderPreservingRerankerModel:
    """predict() returns descending scores so rerank() preserves FAISS
    input order - deterministic, no randomness."""

    def predict(self, pairs, batch_size=None):
        return [float(len(pairs) - i) for i in range(len(pairs))]


class ScriptedGeneratorModel:
    """Fake causal LM whose `.generate()` returns a single 'response
    index' marker token after the prompt; paired with ScriptedTokenizer's
    `decode()`, this lets tests script exactly what each successive
    `generate()` call produces (a string, or raises an Exception)."""

    def __init__(self, responses: list):
        self.responses = responses
        self.call_count = 0

    def parameters(self):
        import torch
        yield torch.zeros(1)

    def generate(self, input_ids, attention_mask, **gen_kwargs):
        import torch

        idx = self.call_count
        self.call_count += 1
        if idx >= len(self.responses):
            raise RuntimeError("ScriptedGeneratorModel: ran out of scripted responses")
        response = self.responses[idx]
        if isinstance(response, Exception):
            raise response
        marker = torch.tensor([[idx]])
        return torch.cat([input_ids, marker], dim=1)


class ScriptedTokenizer:
    def __init__(self, responses: list):
        self.responses = responses

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors="pt",
                             return_dict=True, enable_thinking=None):
        import torch

        input_ids = torch.arange(1, 4).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids, skip_special_tokens=True):
        idx = int(ids[-1])
        return self.responses[idx]


def _verify_pair_by_round(retriever, decisions_by_round: dict[int, tuple[float, float]]):
    """Builds a `verify_pair` side_effect whose returned probabilities
    depend on which retrieval round is currently active (tracked via
    `retriever.call_count`, incremented once per attempt right before
    verification runs)."""

    def fake_verify_pair(model, tokenizer, premise, hypothesis, max_length=256):
        round_idx = retriever.call_count - 1
        ent, contra = decisions_by_round.get(round_idx, decisions_by_round[max(decisions_by_round)])
        label = "entailment" if ent >= contra else "contradiction"
        return {
            "entailment_probability": ent, "contradiction_probability": contra,
            "verifier_label": label, "pipeline_label": "supported" if label == "entailment" else "contradicted",
        }

    return fake_verify_pair


def _run_loop(retriever, reranker, responses, decisions_by_round, max_correction_attempts=2):
    generator = ScriptedGeneratorModel(responses)
    tokenizer = ScriptedTokenizer(responses)
    with mock.patch("claimguard.verification.verify.verify_pair",
                     side_effect=_verify_pair_by_round(retriever, decisions_by_round)):
        result = correction_mod.run_correction_loop(
            retriever, reranker, 16, object(), object(), generator, tokenizer,
            "What is the capital of France?", generator_model_identifier="fake/model",
            max_correction_attempts=max_correction_attempts,
        )
    return result, generator


class TestAcceptPath(unittest.TestCase):
    def test_accept_terminates_after_one_attempt(self) -> None:
        result, generator = _run_loop(
            CountingRetriever(), OrderPreservingRerankerModel(), ["Paris is the capital of France."],
            {0: (0.95, 0.05)},
        )
        self.assertEqual(result.final_decision, "ACCEPT")
        self.assertEqual(result.termination_reason, "accepted")
        self.assertEqual(result.total_attempts, 1)
        self.assertFalse(result.correction_occurred)
        self.assertEqual(result.status, "completed")
        self.assertEqual(generator.call_count, 1)
        self.assertEqual(result.attempts[0].step16_decision, "SUPPORTED")
        self.assertEqual(result.attempts[0].mapped_decision, "ACCEPT")


class TestAbstainPath(unittest.TestCase):
    def test_abstain_terminates_after_one_attempt_no_extra_generation(self) -> None:
        result, generator = _run_loop(
            CountingRetriever(), OrderPreservingRerankerModel(), ["Some uncertain answer."],
            {0: (0.3, 0.3)},
        )
        self.assertEqual(result.final_decision, "ABSTAIN")
        self.assertEqual(result.termination_reason, "policy_abstained")
        self.assertEqual(result.total_attempts, 1)
        self.assertFalse(result.correction_occurred)
        self.assertEqual(result.status, "completed")
        self.assertEqual(generator.call_count, 1, "ABSTAIN must not trigger any correction generation")

    def test_empty_evidence_set_yields_graceful_abstain_not_failure(self) -> None:
        retriever = EmptyRetriever()
        generator = ScriptedGeneratorModel(["An answer with no evidence available."])
        tokenizer = ScriptedTokenizer(generator.responses)
        result = correction_mod.run_correction_loop(
            retriever, OrderPreservingRerankerModel(), 16, object(), object(), generator, tokenizer,
            "An obscure question.", generator_model_identifier="fake/model",
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_decision, "ABSTAIN")
        self.assertEqual(generator.call_count, 1)


class TestCorrectPath(unittest.TestCase):
    def test_correct_triggers_one_regeneration_and_fresh_pipeline_rerun(self) -> None:
        retriever = CountingRetriever()
        result, generator = _run_loop(
            retriever, OrderPreservingRerankerModel(),
            ["The Eiffel Tower is in Berlin.", "The Eiffel Tower is in Paris, France."],
            {0: (0.1, 0.9), 1: (0.95, 0.05)},
        )
        self.assertEqual(result.final_decision, "ACCEPT")
        self.assertEqual(result.total_attempts, 2)
        self.assertTrue(result.correction_occurred)
        self.assertEqual(generator.call_count, 2, "correction must trigger exactly one new generation")
        self.assertEqual(retriever.call_count, 2, "correction must trigger a fresh retrieval, not reuse stale results")
        self.assertEqual(result.attempts[0].mapped_decision, "CORRECT")
        self.assertEqual(result.attempts[1].mapped_decision, "ACCEPT")
        # Attempt 1's evidence must come from the SECOND retrieval call, not attempt 0's.
        self.assertNotEqual(
            result.attempts[0].evidence[0].corpus_id, result.attempts[1].evidence[0].corpus_id,
        )
        self.assertEqual(result.final_answer, "The Eiffel Tower is in Paris, France.")

    def test_retrieval_query_is_the_candidate_answer_not_the_original_query(self) -> None:
        retriever = CountingRetriever()
        _run_loop(
            retriever, OrderPreservingRerankerModel(),
            ["candidate answer text", "corrected answer text"],
            {0: (0.1, 0.9), 1: (0.95, 0.05)},
        )
        self.assertEqual(retriever.queries[0], "candidate answer text")
        self.assertEqual(retriever.queries[1], "corrected answer text")
        self.assertNotIn("What is the capital of France?", retriever.queries)


class TestMaxAttempts(unittest.TestCase):
    def test_hard_termination_at_max_correction_attempts(self) -> None:
        retriever = CountingRetriever()
        result, generator = _run_loop(
            retriever, OrderPreservingRerankerModel(),
            ["answer v0", "answer v1", "answer v2"],
            {0: (0.1, 0.9), 1: (0.1, 0.9), 2: (0.1, 0.9)},
            max_correction_attempts=2,
        )
        self.assertEqual(result.final_decision, "ABSTAIN")
        self.assertEqual(result.termination_reason, "max_attempts_reached")
        self.assertEqual(result.total_attempts, 3)  # attempt 0, 1, 2
        self.assertTrue(result.correction_occurred)
        self.assertEqual(generator.call_count, 3, "must not exceed 1 + max_correction_attempts generations")
        self.assertEqual(result.status, "completed")
        for attempt in result.attempts:
            self.assertEqual(attempt.mapped_decision, "CORRECT")


class TestFailureHandling(unittest.TestCase):
    def test_generator_failure_returns_structured_failure(self) -> None:
        retriever = CountingRetriever()
        generator = ScriptedGeneratorModel([RuntimeError("simulated generator crash")])
        tokenizer = ScriptedTokenizer([])
        result = correction_mod.run_correction_loop(
            retriever, OrderPreservingRerankerModel(), 16, object(), object(), generator, tokenizer,
            "A question.", generator_model_identifier="fake/model",
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.final_decision, "ABSTAIN")
        self.assertIsNotNone(result.failure)
        self.assertEqual(result.failure.stage, "generation")
        self.assertEqual(result.failure.error_type, "RuntimeError")

    def test_empty_generated_answer_returns_structured_failure(self) -> None:
        retriever = CountingRetriever()
        generator = ScriptedGeneratorModel([""])
        tokenizer = ScriptedTokenizer([""])
        result = correction_mod.run_correction_loop(
            retriever, OrderPreservingRerankerModel(), 16, object(), object(), generator, tokenizer,
            "A question.", generator_model_identifier="fake/model",
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.termination_reason, "empty_generated_answer")

    def test_verifier_failure_returns_structured_failure(self) -> None:
        retriever = CountingRetriever()
        generator = ScriptedGeneratorModel(["some candidate answer"])
        tokenizer = ScriptedTokenizer(generator.responses)
        with mock.patch("claimguard.verification.verify.verify_pair",
                         side_effect=RuntimeError("simulated verifier crash")):
            result = correction_mod.run_correction_loop(
                retriever, OrderPreservingRerankerModel(), 16, object(), object(), generator, tokenizer,
                "A question.", generator_model_identifier="fake/model",
            )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure.stage, "retrieval_or_reranking_or_verification")

    def test_correction_generation_failure_returns_structured_failure(self) -> None:
        retriever = CountingRetriever()
        result, generator = _run_loop(
            retriever, OrderPreservingRerankerModel(),
            ["initial answer", RuntimeError("simulated correction-generation crash")],
            {0: (0.1, 0.9)},
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure.stage, "correction_generation")
        self.assertTrue(result.correction_occurred)

    def test_invalid_policy_decision_returns_structured_failure(self) -> None:
        retriever = CountingRetriever()
        generator = ScriptedGeneratorModel(["some candidate answer"])
        tokenizer = ScriptedTokenizer(generator.responses)
        bogus_decision = policy_mod.Decision(
            decision="BOGUS", confidence=0.5, selected_evidence=None, reason="test",
            max_entailment_probability=0.5, max_contradiction_probability=0.5,
            second_highest_entailment_probability=None, entailment_margin=None,
            num_candidates=5, num_above_entailment_threshold=0, num_above_contradiction_threshold=0,
        )
        with mock.patch("claimguard.verification.verify.verify_pair",
                         side_effect=_verify_pair_by_round(retriever, {0: (0.9, 0.1)})), \
             mock.patch.object(policy_mod.EvidenceDecisionPolicy, "decide", return_value=bogus_decision):
            result = correction_mod.run_correction_loop(
                retriever, OrderPreservingRerankerModel(), 16, object(), object(), generator, tokenizer,
                "A question.", generator_model_identifier="fake/model",
            )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure.stage, "decision_policy")


class TestProvenanceAndSerialization(unittest.TestCase):
    def test_evidence_provenance_preserved(self) -> None:
        result, _ = _run_loop(
            CountingRetriever(), OrderPreservingRerankerModel(), ["An answer."], {0: (0.95, 0.05)},
        )
        ev = result.attempts[0].evidence[0]
        self.assertEqual(ev.corpus_id, "C0_call0::0")
        self.assertEqual(ev.page_id, "P")
        self.assertEqual(ev.sentence_id, 0)
        self.assertEqual(ev.original_rank, 1)
        self.assertIn(ev.reranked_rank, range(1, 6))

    def test_selected_evidence_matches_max_entailment_candidate(self) -> None:
        result, _ = _run_loop(
            CountingRetriever(), OrderPreservingRerankerModel(), ["An answer."], {0: (0.95, 0.05)},
        )
        self.assertIsNotNone(result.attempts[0].selected_evidence)

    def test_state_is_json_serializable(self) -> None:
        result, _ = _run_loop(
            CountingRetriever(), OrderPreservingRerankerModel(), ["An answer."], {0: (0.95, 0.05)},
        )
        serialized = json.dumps(result.to_dict())
        reloaded = json.loads(serialized)
        self.assertEqual(reloaded["final_decision"], "ACCEPT")
        self.assertEqual(len(reloaded["attempts"]), 1)

    def test_result_is_correction_result_instance(self) -> None:
        result, _ = _run_loop(
            CountingRetriever(), OrderPreservingRerankerModel(), ["An answer."], {0: (0.95, 0.05)},
        )
        self.assertIsInstance(result, CorrectionResult)


class TestDeterminism(unittest.TestCase):
    def test_same_fixture_produces_identical_result_across_two_runs(self) -> None:
        def run_once():
            retriever = CountingRetriever()
            result, _ = _run_loop(
                retriever, OrderPreservingRerankerModel(),
                ["Eiffel Tower is in Berlin.", "Eiffel Tower is in Paris."],
                {0: (0.1, 0.9), 1: (0.95, 0.05)},
            )
            return result

        r1 = run_once()
        r2 = run_once()
        self.assertEqual(r1.final_decision, r2.final_decision)
        self.assertEqual(r1.final_answer, r2.final_answer)
        self.assertEqual(r1.total_attempts, r2.total_attempts)
        self.assertEqual(
            [a.mapped_decision for a in r1.attempts], [a.mapped_decision for a in r2.attempts],
        )


class TestPromptConstruction(unittest.TestCase):
    def test_correction_prompt_includes_query_answer_and_evidence(self) -> None:
        prompt = correction_mod.build_correction_prompt(
            "What is the capital of France?", "Berlin.", ["Paris is the capital of France."],
        )
        self.assertIn("What is the capital of France?", prompt)
        self.assertIn("Berlin.", prompt)
        self.assertIn("Paris is the capital of France.", prompt)

    def test_correction_prompt_handles_no_evidence(self) -> None:
        prompt = correction_mod.build_correction_prompt("Q?", "A.", [])
        self.assertIn("no supporting evidence", prompt.lower())


if __name__ == "__main__":
    unittest.main()
