"""Tests for claimguard.reranking (Step 14: BGE reranker integration).

Uses lightweight synthetic fixtures + mocks for scoring-logic tests (fast,
offline). Real-model tests (actual BAAI/bge-reranker-large) are guarded by
SkipTest if unavailable in this environment, matching the project's
established convention for embedding/verifier-model tests elsewhere.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest import mock

from claimguard.reranking import reranker as reranker_mod


def _retrieval_reranking_source_files() -> list[Path]:
    import claimguard.reranking as pkg
    return list(Path(pkg.__file__).parent.glob("*.py"))


class TestRAGTruthTruthfulQAExclusion(unittest.TestCase):
    """Same structural guard as Step 13's retrieval package: the reranking
    package must never IMPORT ragtruth/truthfulqa dataset modules."""

    def test_no_ragtruth_or_truthfulqa_imports(self) -> None:
        import ast

        for path in _retrieval_reranking_source_files():
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


def _candidate(corpus_id: str, text: str, score: float = 0.5, page_id: str = "P", sentence_id: int = 0):
    return {
        "corpus_id": corpus_id, "text": text, "score": score,
        "page_id": page_id, "sentence_id": sentence_id, "corpus_version": "test",
    }


class _FakeModel:
    """A fake CrossEncoder-like object: model.predict(pairs, batch_size=...)
    returns a deterministic score derived from candidate text length, so
    tests don't need the real 560M-param model."""

    def __init__(self, score_fn=None):
        self.score_fn = score_fn or (lambda q, t: float(len(t)))
        self.calls = []

    def predict(self, pairs, batch_size=None):
        self.calls.append((list(pairs), batch_size))
        return [self.score_fn(q, t) for q, t in pairs]


class TestScore(unittest.TestCase):
    def test_score_returns_one_float_per_candidate(self) -> None:
        model = _FakeModel(score_fn=lambda q, t: 1.0)
        scores = reranker_mod.score(model, "query", ["a", "bb", "ccc"], batch_size=8)
        self.assertEqual(scores, [1.0, 1.0, 1.0])

    def test_score_empty_candidates_returns_empty_without_calling_model(self) -> None:
        model = _FakeModel()
        scores = reranker_mod.score(model, "query", [], batch_size=8)
        self.assertEqual(scores, [])
        self.assertEqual(model.calls, [])


class TestRerank(unittest.TestCase):
    def test_output_shape_one_record_per_candidate(self) -> None:
        candidates = [_candidate("A::0", "short", 0.5), _candidate("B::0", "a bit longer", 0.9)]
        model = _FakeModel(score_fn=lambda q, t: float(len(t)))
        result = reranker_mod.rerank(model, "q", candidates, top_n=2, batch_size=8)
        self.assertEqual(len(result), 2)

    def test_finite_scores(self) -> None:
        candidates = [_candidate("A::0", "x", 0.5)]
        model = _FakeModel(score_fn=lambda q, t: 3.14)
        result = reranker_mod.rerank(model, "q", candidates, top_n=1, batch_size=8)
        self.assertTrue(math.isfinite(result[0]["reranker_score"]))

    def test_reranks_by_score_descending(self) -> None:
        candidates = [
            _candidate("SHORT::0", "hi", 0.9),        # highest FAISS score, but will score LOW
            _candidate("LONG::0", "a much longer candidate text here", 0.1),  # lowest FAISS, scores HIGH
        ]
        model = _FakeModel(score_fn=lambda q, t: float(len(t)))  # longer text -> higher reranker score
        result = reranker_mod.rerank(model, "q", candidates, top_n=2, batch_size=8)
        self.assertEqual(result[0]["corpus_id"], "LONG::0")
        self.assertEqual(result[1]["corpus_id"], "SHORT::0")

    def test_faiss_score_preserved(self) -> None:
        candidates = [_candidate("A::0", "x", 0.777)]
        model = _FakeModel(score_fn=lambda q, t: 1.0)
        result = reranker_mod.rerank(model, "q", candidates, top_n=1, batch_size=8)
        self.assertEqual(result[0]["faiss_score"], 0.777)

    def test_provenance_preserved(self) -> None:
        candidates = [_candidate("A::7", "some text", 0.5, page_id="PageA", sentence_id=7)]
        model = _FakeModel(score_fn=lambda q, t: 1.0)
        result = reranker_mod.rerank(model, "q", candidates, top_n=1, batch_size=8)
        r = result[0]
        self.assertEqual(r["text"], "some text")
        self.assertEqual(r["page_id"], "PageA")
        self.assertEqual(r["sentence_id"], 7)
        self.assertEqual(r["corpus_version"], "test")

    def test_original_and_reranked_rank_tracked(self) -> None:
        candidates = [
            _candidate("FIRST::0", "z", 0.9),   # original_rank 1, will score lowest
            _candidate("SECOND::0", "zzzzzzzz", 0.1),  # original_rank 2, will score highest
        ]
        model = _FakeModel(score_fn=lambda q, t: float(len(t)))
        result = reranker_mod.rerank(model, "q", candidates, top_n=2, batch_size=8)
        by_id = {r["corpus_id"]: r for r in result}
        self.assertEqual(by_id["FIRST::0"]["original_rank"], 1)
        self.assertEqual(by_id["FIRST::0"]["reranked_rank"], 2)
        self.assertEqual(by_id["SECOND::0"]["original_rank"], 2)
        self.assertEqual(by_id["SECOND::0"]["reranked_rank"], 1)

    def test_top_n_truncates_output(self) -> None:
        candidates = [_candidate(f"C{i}::0", "x" * i, 0.5) for i in range(1, 11)]
        model = _FakeModel(score_fn=lambda q, t: float(len(t)))
        result = reranker_mod.rerank(model, "q", candidates, top_n=3, batch_size=8)
        self.assertEqual(len(result), 3)
        # top_n=3 must be the 3 HIGHEST-scoring, not just the first 3 input candidates.
        self.assertEqual([r["corpus_id"] for r in result], ["C10::0", "C9::0", "C8::0"])

    def test_top_n_larger_than_candidates_returns_all(self) -> None:
        candidates = [_candidate("A::0", "x", 0.5)]
        model = _FakeModel(score_fn=lambda q, t: 1.0)
        result = reranker_mod.rerank(model, "q", candidates, top_n=100, batch_size=8)
        self.assertEqual(len(result), 1)

    def test_empty_candidates_returns_empty_list(self) -> None:
        model = _FakeModel()
        result = reranker_mod.rerank(model, "q", [], top_n=5, batch_size=8)
        self.assertEqual(result, [])
        self.assertEqual(model.calls, [])

    def test_duplicate_corpus_id_candidates_both_scored_independently(self) -> None:
        candidates = [_candidate("DUP::0", "x", 0.5), _candidate("DUP::0", "x", 0.5)]
        model = _FakeModel(score_fn=lambda q, t: 1.0)
        result = reranker_mod.rerank(model, "q", candidates, top_n=2, batch_size=8)
        self.assertEqual(len(result), 2)  # not deduplicated
        self.assertEqual(result[0]["original_rank"], 1)
        self.assertEqual(result[1]["original_rank"], 2)

    def test_deterministic_ranking_given_deterministic_model(self) -> None:
        candidates = [_candidate(f"C{i}::0", "x" * i, 0.5) for i in range(1, 6)]
        model = _FakeModel(score_fn=lambda q, t: float(len(t)))
        result_a = reranker_mod.rerank(model, "q", candidates, top_n=5, batch_size=8)
        result_b = reranker_mod.rerank(model, "q", candidates, top_n=5, batch_size=8)
        self.assertEqual(
            [r["corpus_id"] for r in result_a], [r["corpus_id"] for r in result_b],
        )


class TestRetrieveAndRerank(unittest.TestCase):
    def test_returns_both_stages(self) -> None:
        class FakeRetriever:
            def retrieve(self, query, top_k):
                return [_candidate(f"C{i}::0", "x" * i, 1.0 / i) for i in range(1, top_k + 1)]

        model = _FakeModel(score_fn=lambda q, t: float(len(t)))
        result = reranker_mod.retrieve_and_rerank(
            FakeRetriever(), model, "q", retrieval_top_k=5, rerank_top_n=2, batch_size=8,
        )
        self.assertEqual(len(result["faiss_candidates"]), 5)
        self.assertEqual(len(result["reranked_candidates"]), 2)
        self.assertEqual(result["query"], "q")


class TestConfigLoads(unittest.TestCase):
    def test_reranker_baseline_config_loads_with_required_keys(self) -> None:
        from claimguard import config as cg_config

        cfg = cg_config.load_reranker_baseline_config()
        self.assertIn("reranker", cfg)
        self.assertIn("pipeline", cfg)
        self.assertIn("evaluation", cfg)
        self.assertEqual(cfg["reranker"]["name"], "BAAI/bge-reranker-large")
        self.assertIn("retrieval_top_k", cfg["pipeline"])
        self.assertIn("rerank_top_n", cfg["pipeline"])


class TestLoadRerankerRealModel(unittest.TestCase):
    """Guarded by SkipTest if the real reranker can't be loaded (offline,
    no GPU, etc.) - otherwise verifies the REAL configured model, matching
    the project's convention for embedding/tokenizer tests elsewhere."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            reranker_cfg = reranker_mod.load_reranker_config()
            cls.model, cls.verified_info = reranker_mod.load_reranker(reranker_cfg)
            cls.reranker_cfg = reranker_cfg
        except Exception as exc:  # noqa: BLE001
            cls.model = None
            cls.skip_reason = str(exc)

    def setUp(self) -> None:
        if self.model is None:
            raise unittest.SkipTest(f"BGE reranker model not available: {self.skip_reason}")

    def test_score_direction_is_higher_is_more_relevant(self) -> None:
        self.assertEqual(self.verified_info["score_direction"], "higher_is_more_relevant")

    def test_real_model_scores_correct_answer_highest(self) -> None:
        candidates = [
            _candidate("wrong::0", "Berlin is the capital of Germany."),
            _candidate("right::0", "Paris is the capital of France."),
            _candidate("unrelated::0", "Bananas are a good source of potassium."),
        ]
        result = reranker_mod.rerank(
            self.model, "What is the capital of France?", candidates, top_n=3,
            batch_size=self.reranker_cfg["batch_size"],
        )
        self.assertEqual(result[0]["corpus_id"], "right::0")

    def test_real_model_scores_are_finite(self) -> None:
        candidates = [_candidate("a::0", "some evidence text")]
        result = reranker_mod.rerank(
            self.model, "a query", candidates, top_n=1, batch_size=self.reranker_cfg["batch_size"],
        )
        self.assertTrue(math.isfinite(result[0]["reranker_score"]))


if __name__ == "__main__":
    unittest.main()
