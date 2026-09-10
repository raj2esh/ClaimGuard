"""Tests for claimguard.retrieval (Step 13: corpus, embeddings, FAISS index,
retriever API, retrieval evaluation).

Uses synthetic fixtures for corpus/index logic (fast, offline, deterministic).
Real-data tests (actual wiki_pages index, actual BGE model) are guarded by
SkipTest if the underlying artifact isn't present in this environment,
matching the project's established convention elsewhere.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from claimguard.retrieval import corpus as corpus_mod
from claimguard.retrieval import embed as embed_mod
from claimguard.retrieval import eval as eval_mod
from claimguard.retrieval import index as index_mod


def _retrieval_module_source_files() -> list[Path]:
    import claimguard.retrieval as pkg
    pkg_dir = Path(pkg.__file__).parent
    return list(pkg_dir.glob("*.py"))


class TestRAGTruthTruthfulQAExclusion(unittest.TestCase):
    """Static, robust guard: the retrieval package must never IMPORT the
    RAGTruth or TruthfulQA dataset modules - so there is no code path by
    which their content could reach the corpus, embeddings, or evaluation.

    Checks actual import statements only (via `ast`), not mere mentions of
    the words "ragtruth"/"truthfulqa" - those words legitimately appear in
    this package's own docstrings, explaining WHY those datasets are
    excluded (see corpus.py/eval.py module docstrings), which is exactly
    the kind of documentation this project wants, not a violation."""

    def test_no_ragtruth_or_truthfulqa_imports_anywhere_in_retrieval_package(self) -> None:
        import ast

        for path in _retrieval_module_source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported_modules: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_modules.add(node.module)
                    imported_modules.update(f"{node.module}.{alias.name}" for alias in node.names)
            lowered = {m.lower() for m in imported_modules}
            self.assertFalse(
                any("ragtruth" in m for m in lowered), f"{path} imports a ragtruth module: {imported_modules}"
            )
            self.assertFalse(
                any("truthfulqa" in m for m in lowered), f"{path} imports a truthfulqa module: {imported_modules}"
            )


class TestCorpusId(unittest.TestCase):
    def test_format(self) -> None:
        self.assertEqual(corpus_mod.corpus_id("Page_A", 3), "Page_A::3")

    def test_deterministic(self) -> None:
        self.assertEqual(corpus_mod.corpus_id("X", 1), corpus_mod.corpus_id("X", 1))

    def test_distinct_for_different_sentences(self) -> None:
        self.assertNotEqual(corpus_mod.corpus_id("X", 1), corpus_mod.corpus_id("X", 2))


class TestValidateCorpus(unittest.TestCase):
    def test_detects_duplicate_corpus_id(self) -> None:
        records = [
            {"corpus_id": "A::0", "page_id": "A", "sentence_id": 0, "text": "hi"},
            {"corpus_id": "A::0", "page_id": "A", "sentence_id": 0, "text": "hi again"},
        ]
        issues = corpus_mod.validate_corpus(records)
        self.assertTrue(any("duplicate_corpus_id" in i for i in issues))

    def test_detects_empty_text(self) -> None:
        records = [{"corpus_id": "A::0", "page_id": "A", "sentence_id": 0, "text": "   "}]
        issues = corpus_mod.validate_corpus(records)
        self.assertTrue(any("empty_text" in i for i in issues))

    def test_detects_corpus_id_mismatch(self) -> None:
        records = [{"corpus_id": "WRONG::99", "page_id": "A", "sentence_id": 0, "text": "hi"}]
        issues = corpus_mod.validate_corpus(records)
        self.assertTrue(any("corpus_id_mismatch" in i for i in issues))

    def test_clean_corpus_has_no_issues(self) -> None:
        records = [{
            "corpus_id": corpus_mod.corpus_id("A", 0), "page_id": "A", "sentence_id": 0, "text": "hi",
        }]
        self.assertEqual(corpus_mod.validate_corpus(records), [])


class TestCorpusWriteLoadRoundtrip(unittest.TestCase):
    def test_write_then_load_preserves_records(self) -> None:
        records = [
            {"corpus_id": corpus_mod.corpus_id("A", 0), "page_id": "A", "sentence_id": 0,
             "text": "hi", "corpus_version": "test"},
            {"corpus_id": corpus_mod.corpus_id("A", 1), "page_id": "A", "sentence_id": 1,
             "text": "there", "corpus_version": "test"},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "corpus.jsonl"
            corpus_mod.write_corpus(records, path)
            loaded = corpus_mod.load_corpus(path)
        self.assertEqual(loaded, records)

    def test_load_missing_corpus_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                corpus_mod.load_corpus(Path(td) / "does_not_exist.jsonl")


class TestCorpusRealData(unittest.TestCase):
    """Guarded by SkipTest if the real wiki_pages index/full FEVER data
    aren't present - verifies corpus construction against the actual
    acquired data."""

    def test_build_corpus_records_from_real_index(self) -> None:
        try:
            records, stats = corpus_mod.build_corpus_records()
        except FileNotFoundError:
            raise unittest.SkipTest("wiki_pages index or full FEVER data not present")
        self.assertGreater(len(records), 0)
        self.assertEqual(stats["total_corpus_records"], len(records))
        issues = corpus_mod.validate_corpus(records)
        self.assertEqual(issues, [], f"real corpus has integrity issues: {issues[:5]}")
        # Every record must carry full provenance.
        for r in records[:100]:
            self.assertIn("page_id", r)
            self.assertIn("sentence_id", r)
            self.assertIn("text", r)
            self.assertIn("corpus_version", r)

    def test_deterministic_construction(self) -> None:
        try:
            records_a, _ = corpus_mod.build_corpus_records()
            records_b, _ = corpus_mod.build_corpus_records()
        except FileNotFoundError:
            raise unittest.SkipTest("wiki_pages index or full FEVER data not present")
        self.assertEqual(
            [r["corpus_id"] for r in records_a], [r["corpus_id"] for r in records_b],
        )


class TestEmbedTexts(unittest.TestCase):
    """Guarded by SkipTest if the real BGE model can't be loaded (offline,
    no GPU, etc.) - otherwise uses the REAL configured embedding model on
    a tiny sample, matching the project's convention for embedding/tokenizer
    tests elsewhere."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            embedding_cfg = embed_mod.load_embedding_config()
            cls.model, cls.verified_info = embed_mod.load_embedding_model(embedding_cfg)
            cls.embedding_cfg = embedding_cfg
        except Exception as exc:  # noqa: BLE001
            cls.model = None
            cls.skip_reason = str(exc)

    def setUp(self) -> None:
        if self.model is None:
            raise unittest.SkipTest(f"BGE embedding model not available: {self.skip_reason}")

    def test_embedding_dimension_matches_verified_config(self) -> None:
        emb = embed_mod.embed_texts(self.model, ["a test sentence"], batch_size=1, normalize=True)
        self.assertEqual(emb.shape, (1, self.verified_info["embedding_dim"]))

    def test_embeddings_are_finite(self) -> None:
        emb = embed_mod.embed_texts(
            self.model, ["one", "two", "three"], batch_size=3, normalize=True,
        )
        self.assertTrue(np.isfinite(emb).all())

    def test_normalized_embeddings_have_unit_norm(self) -> None:
        emb = embed_mod.embed_texts(self.model, ["normalize me"], batch_size=1, normalize=True)
        norm = np.linalg.norm(emb[0])
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_dtype_is_float32_for_faiss_compatibility(self) -> None:
        emb = embed_mod.embed_texts(self.model, ["dtype check"], batch_size=1, normalize=True)
        self.assertEqual(emb.dtype, np.float32)

    def test_determinism_two_encode_calls_agree(self) -> None:
        result = embed_mod.smoke_test_determinism(
            self.model, ["deterministic check"], batch_size=1, normalize=True,
        )
        self.assertTrue(result["allclose_atol_1e-5_rtol_1e-4"])


class _TinyIndexFixture:
    @staticmethod
    def make(n: int = 5, dim: int = 8, seed: int = 42):
        rng = np.random.RandomState(seed)
        embeddings = rng.randn(n, dim).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        records = [
            {
                "corpus_id": corpus_mod.corpus_id(f"Page{i}", 0), "page_id": f"Page{i}",
                "sentence_id": 0, "text": f"sentence {i}", "corpus_version": "test",
            }
            for i in range(n)
        ]
        corpus_ids = [r["corpus_id"] for r in records]
        return records, corpus_ids, embeddings


class TestBuildIndex(unittest.TestCase):
    def test_vector_count_and_dimension(self) -> None:
        records, corpus_ids, embeddings = _TinyIndexFixture.make(n=6, dim=8)
        index = index_mod.build_index(embeddings)
        self.assertEqual(index.ntotal, 6)
        self.assertEqual(index.d, 8)

    def test_search_returns_self_as_top_result(self) -> None:
        records, corpus_ids, embeddings = _TinyIndexFixture.make(n=6, dim=8)
        index = index_mod.build_index(embeddings)
        scores, indices = index.search(embeddings[0:1], 1)
        self.assertEqual(indices[0][0], 0)  # a vector's nearest neighbor (IP) is itself


class TestIndexSaveLoad(unittest.TestCase):
    def _with_temp_paths(self, td: str):
        out_dir = Path(td)
        orig_index_path = index_mod.FAISS_INDEX_PATH
        orig_meta_path = index_mod.INDEX_METADATA_PATH
        index_mod.FAISS_INDEX_PATH = out_dir / "faiss_index.bin"
        index_mod.INDEX_METADATA_PATH = out_dir / "index_metadata.json"
        return out_dir, orig_index_path, orig_meta_path

    def test_save_load_roundtrip_identical_retrieval(self) -> None:
        records, corpus_ids, embeddings = _TinyIndexFixture.make()
        index = index_mod.build_index(embeddings)
        with tempfile.TemporaryDirectory() as td:
            out_dir, orig_index_path, orig_meta_path = self._with_temp_paths(td)
            try:
                index_mod.save_index(index, records, corpus_ids, {"model_name": "test"})
                reloaded_index, metadata = index_mod.load_index(out_dir)
                issues = index_mod.validate_index(reloaded_index, metadata)
                self.assertEqual(issues, [])
                scores_a, idx_a = index.search(embeddings[0:1], 3)
                scores_b, idx_b = reloaded_index.search(embeddings[0:1], 3)
                self.assertTrue(np.array_equal(idx_a, idx_b))
                self.assertTrue(np.allclose(scores_a, scores_b, atol=1e-5))
            finally:
                index_mod.FAISS_INDEX_PATH = orig_index_path
                index_mod.INDEX_METADATA_PATH = orig_meta_path

    def test_load_missing_index_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir, orig_index_path, orig_meta_path = self._with_temp_paths(td)
            try:
                with self.assertRaises(FileNotFoundError):
                    index_mod.load_index(out_dir)
            finally:
                index_mod.FAISS_INDEX_PATH = orig_index_path
                index_mod.INDEX_METADATA_PATH = orig_meta_path


class TestValidateIndex(unittest.TestCase):
    def test_detects_count_mismatch(self) -> None:
        records, corpus_ids, embeddings = _TinyIndexFixture.make()
        index = index_mod.build_index(embeddings)
        metadata = {"row_metadata": records[:-1], "dimension": embeddings.shape[1], "vector_count": index.ntotal}
        issues = index_mod.validate_index(index, metadata)
        self.assertTrue(any("vector_metadata_count_mismatch" in i for i in issues))

    def test_detects_dimension_mismatch(self) -> None:
        records, corpus_ids, embeddings = _TinyIndexFixture.make()
        index = index_mod.build_index(embeddings)
        metadata = {"row_metadata": records, "dimension": 999, "vector_count": index.ntotal}
        issues = index_mod.validate_index(index, metadata)
        self.assertTrue(any("dimension_mismatch" in i for i in issues))

    def test_detects_duplicate_corpus_id_in_metadata(self) -> None:
        records, corpus_ids, embeddings = _TinyIndexFixture.make()
        index = index_mod.build_index(embeddings)
        bad_records = records[:-1] + [records[0]]
        metadata = {"row_metadata": bad_records, "dimension": embeddings.shape[1], "vector_count": index.ntotal}
        issues = index_mod.validate_index(index, metadata)
        self.assertTrue(any("duplicate_corpus_id_in_index" in i for i in issues))

    def test_clean_index_has_no_issues(self) -> None:
        records, corpus_ids, embeddings = _TinyIndexFixture.make()
        index = index_mod.build_index(embeddings)
        metadata = {"row_metadata": records, "dimension": embeddings.shape[1], "vector_count": index.ntotal}
        self.assertEqual(index_mod.validate_index(index, metadata), [])


class TestRetrieverAPI(unittest.TestCase):
    def _make_retriever(self, model_name: str = "test") -> index_mod.Retriever:
        records, corpus_ids, embeddings = _TinyIndexFixture.make(n=4, dim=8)
        index = index_mod.build_index(embeddings)
        metadata = {
            "row_metadata": records, "dimension": embeddings.shape[1], "vector_count": index.ntotal,
            # "name" - matches configs/models.yaml's embedding.primary.name field,
            # NOT "model_name" (a real bug: index.py once read this dict with the
            # wrong key and only surfaced at real retrieve() time - see
            # test_ensure_model_loaded_reads_embedding_info_with_correct_key below).
            "embedding_info": {"name": model_name},
        }
        return index_mod.Retriever(index, metadata)

    def test_get_by_corpus_id_returns_full_provenance(self) -> None:
        retriever = self._make_retriever()
        record = retriever.get_by_corpus_id("Page0::0")
        self.assertEqual(record["page_id"], "Page0")
        self.assertEqual(record["sentence_id"], 0)

    def test_get_by_missing_corpus_id_raises(self) -> None:
        retriever = self._make_retriever()
        with self.assertRaises(KeyError):
            retriever.get_by_corpus_id("does_not_exist::0")

    def test_top_k_larger_than_corpus_does_not_crash(self) -> None:
        records, corpus_ids, embeddings = _TinyIndexFixture.make(n=3, dim=8)
        index = index_mod.build_index(embeddings)
        scores, indices = index.search(embeddings[0:1], 10)  # top_k > ntotal
        # FAISS pads with -1 for missing slots; must not crash, and real
        # slots must be a subset of valid row indices.
        valid = [i for i in indices[0] if i >= 0]
        self.assertLessEqual(len(valid), 3)
        self.assertTrue(all(0 <= i < 3 for i in valid))

    def test_ensure_model_loaded_reads_embedding_info_with_correct_key(self) -> None:
        # Regression test for a real bug: _ensure_model_loaded() once read
        # self.metadata["embedding_info"]["model_name"], but the actual key
        # written by build_faiss_index.py (and configs/models.yaml's own
        # schema) is "name" - this raised KeyError at real retrieve() time,
        # never caught by the other synthetic-fixture tests above because
        # none of them exercise _ensure_model_loaded(). Verified here
        # WITHOUT loading the real model: a matching name must reach
        # load_embedding_model (mocked), and a mismatched name must raise
        # RuntimeError - not KeyError - before any load is attempted.
        from unittest import mock

        retriever = self._make_retriever(model_name="BAAI/bge-large-en-v1.5")
        fake_cfg = {"name": "BAAI/bge-large-en-v1.5", "normalize_embeddings": True}
        with mock.patch(
            "claimguard.retrieval.embed.load_embedding_config", return_value=fake_cfg,
        ), mock.patch(
            "claimguard.retrieval.embed.load_embedding_model", return_value=(mock.Mock(), {}),
        ) as mocked_load:
            retriever._ensure_model_loaded()  # must not raise KeyError
            mocked_load.assert_called_once()

    def test_ensure_model_loaded_rejects_mismatched_model(self) -> None:
        from unittest import mock

        retriever = self._make_retriever(model_name="index-was-built-with-model-A")
        fake_cfg = {"name": "config-now-says-model-B", "normalize_embeddings": True}
        with mock.patch(
            "claimguard.retrieval.embed.load_embedding_config", return_value=fake_cfg,
        ):
            with self.assertRaises(RuntimeError):
                retriever._ensure_model_loaded()


class TestRetrievalEval(unittest.TestCase):
    def test_recall_computation_with_synthetic_retriever(self) -> None:
        class FakeRetriever:
            def retrieve(self, query, top_k):
                # Always returns a fixed ranking; "hit" is engineered at rank 2.
                return [
                    {"corpus_id": "wrong::0"},
                    {"corpus_id": "target::0"},
                    {"corpus_id": "wrong::1"},
                ][:top_k]

        eval_set = [{"example_id": 1, "claim": "x", "relevant_corpus_ids": ["target::0"]}]
        result = eval_mod.evaluate_retrieval(FakeRetriever(), eval_set, max_k=3)
        self.assertEqual(result["recall_at_1"], 0.0)  # hit is at rank 2, not within top-1
        self.assertEqual(result["n_eval_claims"], 1)

    def test_no_hit_when_relevant_never_retrieved(self) -> None:
        class FakeRetriever:
            def retrieve(self, query, top_k):
                return [{"corpus_id": "irrelevant::0"}]

        eval_set = [{"example_id": 1, "claim": "x", "relevant_corpus_ids": ["target::0"]}]
        result = eval_mod.evaluate_retrieval(FakeRetriever(), eval_set, max_k=1)
        self.assertEqual(result["per_claim"][0]["first_hit_rank"], None)


if __name__ == "__main__":
    unittest.main()
