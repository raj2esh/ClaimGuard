"""Tests for claimguard.verification (Step 15: integrated retrieval +
reranking + binary verifier pipeline).

Uses a real, tiny, LOCALLY-CONSTRUCTED DebertaV2ForSequenceClassification
binary checkpoint (saved to a temp dir and loaded via the real
`load_verifier()` path) for verifier tests - not the 440M-param model - so
this suite stays fast and offline while still exercising the real
checkpoint-loading code path. Pipeline-orchestration tests use lightweight
fakes for the retriever/reranker/verifier. Real-model tests (the actual
Step 12 checkpoint) are guarded by SkipTest if unavailable.
"""

from __future__ import annotations

import inspect
import math
import tempfile
import unittest
from pathlib import Path

from claimguard.verification import pipeline as pipeline_mod
from claimguard.verification import verify as verify_mod


def _retrieval_verification_source_files() -> list[Path]:
    import claimguard.verification as pkg
    return list(Path(pkg.__file__).parent.glob("*.py"))


class TestRAGTruthTruthfulQAExclusion(unittest.TestCase):
    def test_no_ragtruth_or_truthfulqa_imports(self) -> None:
        import ast

        for path in _retrieval_verification_source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
                    imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            lowered = {m.lower() for m in imported}
            self.assertFalse(any("ragtruth" in m for m in lowered), f"{path} imports ragtruth: {imported}")
            self.assertFalse(any("truthfulqa" in m for m in lowered), f"{path} imports truthfulqa: {imported}")


class TestNoGoldAccessDuringInference(unittest.TestCase):
    """Structural guard: run_pipeline's ONLY per-query selection input is
    `query` - no parameter exists through which gold evidence or gold
    labels could reach evidence selection."""

    def test_run_pipeline_has_no_gold_parameter(self) -> None:
        sig = inspect.signature(pipeline_mod.run_pipeline)
        param_names = {p.lower() for p in sig.parameters}
        forbidden_substrings = ["gold", "label", "relevant", "true_evidence", "answer"]
        offending = [p for p in param_names for f in forbidden_substrings if f in p]
        self.assertEqual(offending, [], f"run_pipeline has a suspicious parameter: {offending}")

    def test_select_evidence_only_takes_verified_candidates(self) -> None:
        sig = inspect.signature(pipeline_mod.select_evidence)
        self.assertEqual(list(sig.parameters.keys()), ["verified_candidates"])


def _candidate(corpus_id: str, text: str = "some evidence text", faiss_score: float = 0.5,
               reranker_score: float = 0.5, original_rank: int = 1, reranked_rank: int = 1,
               page_id: str = "P", sentence_id: int = 0):
    return {
        "corpus_id": corpus_id, "text": text, "faiss_score": faiss_score, "reranker_score": reranker_score,
        "original_rank": original_rank, "reranked_rank": reranked_rank, "page_id": page_id,
        "sentence_id": sentence_id, "corpus_version": "test",
    }


class TestSelectEvidence(unittest.TestCase):
    def test_selects_max_entailment_probability(self) -> None:
        candidates = [
            {**_candidate("A::0", reranked_rank=1), "entailment_probability": 0.3, "contradiction_probability": 0.7},
            {**_candidate("B::0", reranked_rank=2), "entailment_probability": 0.9, "contradiction_probability": 0.1},
            {**_candidate("C::0", reranked_rank=3), "entailment_probability": 0.5, "contradiction_probability": 0.5},
        ]
        selected, score = pipeline_mod.select_evidence(candidates)
        self.assertEqual(selected["corpus_id"], "B::0")
        self.assertAlmostEqual(score, 0.9)

    def test_empty_candidates_returns_none(self) -> None:
        selected, score = pipeline_mod.select_evidence([])
        self.assertIsNone(selected)
        self.assertIsNone(score)

    def test_deterministic_tie_break_prefers_earlier_rank(self) -> None:
        candidates = [
            {**_candidate("LATER::0", reranked_rank=3), "entailment_probability": 0.8, "contradiction_probability": 0.2},
            {**_candidate("EARLIER::0", reranked_rank=1), "entailment_probability": 0.8, "contradiction_probability": 0.2},
        ]
        selected, _ = pipeline_mod.select_evidence(candidates)
        self.assertEqual(selected["corpus_id"], "EARLIER::0")

    def test_deterministic_across_repeated_calls(self) -> None:
        candidates = [
            {**_candidate("A::0", reranked_rank=1), "entailment_probability": 0.4, "contradiction_probability": 0.6},
            {**_candidate("B::0", reranked_rank=2), "entailment_probability": 0.6, "contradiction_probability": 0.4},
        ]
        result_a, score_a = pipeline_mod.select_evidence(candidates)
        result_b, score_b = pipeline_mod.select_evidence(candidates)
        self.assertEqual(result_a["corpus_id"], result_b["corpus_id"])
        self.assertEqual(score_a, score_b)


def _faiss_candidate(corpus_id: str, text: str, score: float, page_id: str = "P", sentence_id: int = 0):
    """Matches the REAL claimguard.retrieval.index.Retriever.retrieve()
    output schema (key is "score", not "faiss_score" - that renaming only
    happens inside reranker.rerank())."""
    return {
        "corpus_id": corpus_id, "text": text, "score": score,
        "page_id": page_id, "sentence_id": sentence_id, "corpus_version": "test",
    }


class FakeRetriever:
    def __init__(self, n: int = 20):
        self.n = n

    def retrieve(self, query, top_k):
        return [
            _faiss_candidate(f"C{i}::0", text=f"candidate text {i}", score=1.0 - i * 0.01)
            for i in range(min(top_k, self.n))
        ]


class FakeRerankerModel:
    """Reverses FAISS order deterministically, so pipeline tests can check
    that reranked_rank differs meaningfully from original_rank."""

    def predict(self, pairs, batch_size=None):
        # Score = negative index in the pairs list -> later pairs score higher (reverses order).
        return [float(i) for i in range(len(pairs))]


class FakeVerifierModel:
    pass


class TestRunPipelineOrdering(unittest.TestCase):
    """Uses the real reranker.rerank()/verify.verify_candidates() logic
    with FAKE model objects (predict()-compatible / patched verify_pair)
    so pipeline ordering/provenance/candidate-count behavior is exercised
    without any real GPU model."""

    def test_pipeline_preserves_candidate_count_and_provenance(self) -> None:
        from unittest import mock

        retriever = FakeRetriever(n=20)
        reranker_model = FakeRerankerModel()

        def fake_verify_pair(model, tokenizer, premise, hypothesis, max_length=256):
            # Deterministic score derived from premise text so tests can
            # check select_evidence picks the right one.
            n = int(premise.split()[-1])
            return {
                "entailment_probability": n / 100.0, "contradiction_probability": 1 - n / 100.0,
                "verifier_label": "entailment" if n >= 50 else "contradiction",
                "pipeline_label": "supported" if n >= 50 else "contradicted",
            }

        with mock.patch("claimguard.verification.verify.verify_pair", side_effect=fake_verify_pair):
            result = pipeline_mod.run_pipeline(
                retriever, reranker_model, FakeVerifierModel(), verifier_tokenizer=None, query="test query",
                retrieval_top_k=20, rerank_top_n=5, reranker_batch_size=16, verifier_max_length=256,
            )

        self.assertEqual(len(result.faiss_candidates), 20)
        self.assertEqual(len(result.reranked_candidates), 20)  # full permutation, not truncated
        self.assertEqual(len(result.verified_candidates), 5)   # only top rerank_top_n verified
        self.assertIsNotNone(result.selected_evidence)
        self.assertIsNotNone(result.faiss_top1)
        self.assertIsNotNone(result.reranker_top1)

        # Every verified candidate must carry both retrieval AND reranking
        # provenance (faiss_score, reranker_score, both ranks) plus the new
        # verifier fields.
        for c in result.verified_candidates:
            for field in ("corpus_id", "text", "faiss_score", "reranker_score", "original_rank",
                          "reranked_rank", "page_id", "sentence_id", "entailment_probability",
                          "contradiction_probability", "verifier_label"):
                self.assertIn(field, c)

    def test_faiss_score_survives_reranking_and_verification_unchanged(self) -> None:
        from unittest import mock

        retriever = FakeRetriever(n=5)
        reranker_model = FakeRerankerModel()

        def fake_verify_pair(model, tokenizer, premise, hypothesis, max_length=256):
            return {"entailment_probability": 0.5, "contradiction_probability": 0.5,
                    "verifier_label": "entailment", "pipeline_label": "supported"}

        with mock.patch("claimguard.verification.verify.verify_pair", side_effect=fake_verify_pair):
            result = pipeline_mod.run_pipeline(
                retriever, reranker_model, FakeVerifierModel(), verifier_tokenizer=None, query="q",
                retrieval_top_k=5, rerank_top_n=5, reranker_batch_size=16, verifier_max_length=256,
            )

        faiss_scores_by_id = {c["corpus_id"]: c["score"] for c in result.faiss_candidates}
        for c in result.verified_candidates:
            self.assertAlmostEqual(c["faiss_score"], faiss_scores_by_id[c["corpus_id"]])

    def test_empty_faiss_candidates_yields_empty_pipeline_result(self) -> None:
        class EmptyRetriever:
            def retrieve(self, query, top_k):
                return []

        result = pipeline_mod.run_pipeline(
            EmptyRetriever(), FakeRerankerModel(), FakeVerifierModel(), verifier_tokenizer=None, query="q",
            retrieval_top_k=20, rerank_top_n=5, reranker_batch_size=16, verifier_max_length=256,
        )
        self.assertEqual(result.faiss_candidates, [])
        self.assertEqual(result.reranked_candidates, [])
        self.assertEqual(result.verified_candidates, [])
        self.assertIsNone(result.selected_evidence)
        self.assertIsNone(result.faiss_top1)
        self.assertIsNone(result.reranker_top1)
        self.assertIsNone(result.final_support_score)


def _make_tiny_binary_deberta(vocab_size: int = 100):
    """vocab_size defaults to a small placeholder for tests that never run
    a real forward pass (metadata-only checks). Tests that DO run a
    forward pass with the real DeBERTa-v3 tokenizer must pass
    vocab_size=len(tokenizer) - otherwise the tokenizer can legitimately
    emit token ids the tiny embedding table has no row for, raising a real
    IndexError (found via this exact mistake during Step 15's own test
    development - see test_verify_pair_probabilities_sum_to_one)."""
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
    from claimguard.verifier.binary_model import BINARY_ID2LABEL, BINARY_LABEL2ID

    config = DebertaV2Config(
        vocab_size=vocab_size, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=16, max_position_embeddings=32, num_labels=2,
        id2label=BINARY_ID2LABEL, label2id=BINARY_LABEL2ID,
    )
    return DebertaV2ForSequenceClassification(config)


class TestLoadVerifierRealCheckpointFormat(unittest.TestCase):
    """Uses a real, tiny, LOCALLY-SAVED binary checkpoint (same schema as
    the real Step 12 checkpoint) to exercise load_verifier()/verify_pair()
    end-to-end without needing the real 440M-param model or GPU."""

    def test_load_verifier_from_tiny_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model = _make_tiny_binary_deberta()
            model.save_pretrained(td)
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
            tok.save_pretrained(td)

            loaded_model, loaded_tokenizer, info = verify_mod.load_verifier(td, device="cpu", dtype="float32")

        self.assertEqual(info["num_labels"], 2)
        self.assertEqual(info["id2label"], {0: "entailment", 1: "contradiction"})

    def test_verify_pair_probabilities_sum_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
            # This test DOES run a real forward pass, so the tiny model's
            # embedding table must actually cover the real tokenizer's
            # vocabulary (see _make_tiny_binary_deberta's docstring).
            model = _make_tiny_binary_deberta(vocab_size=len(tok))
            model.save_pretrained(td)
            tok.save_pretrained(td)
            loaded_model, loaded_tokenizer, _ = verify_mod.load_verifier(td, device="cpu", dtype="float32")

        result = verify_mod.verify_pair(loaded_model, loaded_tokenizer, "some premise", "some hypothesis", max_length=32)
        total = result["entailment_probability"] + result["contradiction_probability"]
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertTrue(math.isfinite(result["entailment_probability"]))
        self.assertTrue(math.isfinite(result["contradiction_probability"]))
        self.assertIn(result["verifier_label"], ("entailment", "contradiction"))
        self.assertIn(result["pipeline_label"], ("supported", "contradicted"))

    def test_pipeline_label_mapping(self) -> None:
        self.assertEqual(verify_mod.VERIFIER_LABEL_TO_PIPELINE_LABEL["entailment"], "supported")
        self.assertEqual(verify_mod.VERIFIER_LABEL_TO_PIPELINE_LABEL["contradiction"], "contradicted")
        self.assertNotIn("neutral", verify_mod.VERIFIER_LABEL_TO_PIPELINE_LABEL)

    def test_verify_candidates_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model = _make_tiny_binary_deberta()
            model.save_pretrained(td)
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
            tok.save_pretrained(td)
            loaded_model, loaded_tokenizer, _ = verify_mod.load_verifier(td, device="cpu", dtype="float32")

        result = verify_mod.verify_candidates(loaded_model, loaded_tokenizer, "q", [], max_length=32)
        self.assertEqual(result, [])

    def test_load_verifier_rejects_non_binary_checkpoint(self) -> None:
        from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
        with tempfile.TemporaryDirectory() as td:
            config = DebertaV2Config(
                vocab_size=100, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                intermediate_size=16, num_labels=3,
                id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
                label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
            )
            model = DebertaV2ForSequenceClassification(config)
            model.save_pretrained(td)
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
            tok.save_pretrained(td)

            with self.assertRaises(AssertionError):
                verify_mod.load_verifier(td, device="cpu", dtype="float32")


class TestConfigLoads(unittest.TestCase):
    def test_integration_baseline_config_loads_with_required_keys(self) -> None:
        from claimguard import config as cg_config

        cfg = cg_config.load_integration_baseline_config()
        self.assertIn("pipeline", cfg)
        self.assertIn("verifier", cfg)
        self.assertIn("evaluation", cfg)
        self.assertIn("retrieval_top_k", cfg["pipeline"])
        self.assertIn("rerank_top_n", cfg["pipeline"])
        self.assertIn("entailment_threshold", cfg["pipeline"])


class TestLoadRealVerifierCheckpoint(unittest.TestCase):
    """Guarded by SkipTest if the real Step 12 checkpoint isn't available
    in this environment."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.model, cls.tokenizer, cls.info = verify_mod.load_verifier()
        except Exception as exc:  # noqa: BLE001
            cls.model = None
            cls.skip_reason = str(exc)

    def setUp(self) -> None:
        if self.model is None:
            raise unittest.SkipTest(f"Real Step 12 verifier checkpoint not available: {self.skip_reason}")

    def test_real_checkpoint_is_binary(self) -> None:
        self.assertEqual(self.info["num_labels"], 2)
        self.assertEqual(self.info["id2label"], {0: "entailment", 1: "contradiction"})

    def test_real_checkpoint_scores_a_supporting_pair_as_entailment(self) -> None:
        result = verify_mod.verify_pair(
            self.model, self.tokenizer,
            premise="Paris is the capital and largest city of France.",
            hypothesis="Paris is the capital of France.",
            max_length=256,
        )
        self.assertEqual(result["verifier_label"], "entailment")
        self.assertGreater(result["entailment_probability"], 0.5)


if __name__ == "__main__":
    unittest.main()
