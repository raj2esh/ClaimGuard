"""Tests for the RAGTruth dataset loader/normalizer (claimguard.datasets.ragtruth).

RAGTruth is ClaimGuard's PRIMARY end-to-end evaluation dataset, so these
tests place extra emphasis on the evaluation-boundary and leakage
safeguards, not just schema/normalization correctness.

Uses the small sample under data/processed/ragtruth/sample/ (built by
scripts/build_ragtruth_sample.py, TRAIN-split only by construction) plus
small in-memory fixtures for anything that needs test-split or leaking data
(which the real sample deliberately never contains). No model loading, no
GPU.
"""

from __future__ import annotations

import json
import unittest

from claimguard import config as cg_config
from claimguard.datasets import ragtruth

SAMPLE_PATH = cg_config.resolve_path("data/processed/ragtruth/sample/sample.jsonl")


def _load_sample() -> list[dict]:
    if not SAMPLE_PATH.exists():
        raise unittest.SkipTest(
            f"RAGTruth sample not found at {SAMPLE_PATH}; run scripts/build_ragtruth_sample.py first."
        )
    records = []
    with SAMPLE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_response(rid, source_id, split, labels=None, response="hello world"):
    return {
        "id": rid, "source_id": source_id, "model": "test-model", "temperature": 0.0,
        "labels": labels or [], "split": split, "quality": "good", "response": response,
    }


def _make_source(source_id, task_type="QA"):
    return {
        "source_id": source_id, "task_type": task_type, "source": "test-source",
        "source_info": {"question": "q", "passages": "p"}, "prompt": "prompt text",
    }


class TestRAGTruthSampleLoads(unittest.TestCase):
    def test_sample_file_loads_and_is_nonempty(self) -> None:
        records = _load_sample()
        self.assertGreater(len(records), 0)

    def test_sample_covers_multiple_task_types_and_hallucination_states(self) -> None:
        records = _load_sample()
        task_types = {r["task_type"] for r in records}
        halluc_states = {r["has_hallucination"] for r in records}
        self.assertGreater(len(task_types), 1)
        self.assertEqual(halluc_states, {True, False})


class TestRAGTruthSchema(unittest.TestCase):
    def test_expected_fields_exist(self) -> None:
        records = _load_sample()
        expected_fields = {
            "response_id", "source_id", "source_missing", "task_type", "source",
            "source_info", "prompt", "split", "eval_reserved", "model", "temperature",
            "quality", "response_text", "labels", "has_hallucination",
        }
        for r in records:
            self.assertTrue(expected_fields.issubset(r.keys()))


class TestRAGTruthJoin(unittest.TestCase):
    def test_join_attaches_source_info_fields(self) -> None:
        responses = [_make_response("r1", "s1", "train")]
        sources = [_make_source("s1", task_type="Data2txt")]
        joined = ragtruth.join_records(responses, sources)
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]["task_type"], "Data2txt")
        self.assertEqual(joined[0]["source_missing"], False)

    def test_join_preserves_orphan_response_without_dropping_it(self) -> None:
        responses = [_make_response("r1", "missing_source", "train")]
        sources = [_make_source("s1")]
        joined = ragtruth.join_records(responses, sources)
        self.assertEqual(len(joined), 1)  # not dropped
        self.assertTrue(joined[0]["source_missing"])


class TestRAGTruthLabelHandling(unittest.TestCase):
    def test_has_hallucination_true_when_labels_present(self) -> None:
        responses = [_make_response("r1", "s1", "train", labels=[
            {"start": 0, "end": 5, "text": "hello", "label_type": "Evident Conflict",
             "meta": "", "implicit_true": False, "due_to_null": False},
        ])]
        joined = ragtruth.join_records(responses, [_make_source("s1")])
        self.assertTrue(joined[0]["has_hallucination"])

    def test_has_hallucination_false_when_labels_empty(self) -> None:
        responses = [_make_response("r1", "s1", "train", labels=[])]
        joined = ragtruth.join_records(responses, [_make_source("s1")])
        self.assertFalse(joined[0]["has_hallucination"])


class TestRAGTruthOffsetValidation(unittest.TestCase):
    def test_valid_offset_passes(self) -> None:
        record = {
            "response_text": "hello world",
            "labels": [{"start": 0, "end": 5, "text": "hello"}],
        }
        self.assertEqual(ragtruth.validate_offsets(record), [])

    def test_offset_text_mismatch_detected(self) -> None:
        record = {
            "response_text": "hello world",
            "labels": [{"start": 0, "end": 5, "text": "WRONG"}],
        }
        issues = ragtruth.validate_offsets(record)
        self.assertTrue(any("offset_text_mismatch" in i for i in issues))

    def test_out_of_bounds_offset_detected(self) -> None:
        record = {
            "response_text": "hi",
            "labels": [{"start": 0, "end": 100, "text": "hi"}],
        }
        issues = ragtruth.validate_offsets(record)
        self.assertTrue(any("offset_out_of_bounds" in i for i in issues))

    def test_sample_labels_have_no_offset_issues(self) -> None:
        # Reproduces the real Step 6C finding: all 14,289 real offset-bearing
        # labels validated exactly against response text.
        records = _load_sample()
        for r in records:
            issues = ragtruth.validate_offsets(r)
            self.assertEqual(issues, [], f"record {r['response_id']} has offset issues: {issues}")


class TestRAGTruthDuplicateDetection(unittest.TestCase):
    def test_finds_real_duplicate_response_ids(self) -> None:
        records = [
            {"response_id": "a"}, {"response_id": "a"}, {"response_id": "b"},
        ]
        dups = ragtruth.find_duplicate_response_ids(records)
        self.assertEqual(dups, {"a": 2})

    def test_no_false_positive_on_unique_ids(self) -> None:
        records = [{"response_id": "a"}, {"response_id": "b"}]
        self.assertEqual(ragtruth.find_duplicate_response_ids(records), {})

    def test_sample_has_no_duplicate_response_ids(self) -> None:
        records = _load_sample()
        self.assertEqual(ragtruth.find_duplicate_response_ids(records), {})


class TestRAGTruthEvaluationBoundary(unittest.TestCase):
    def _mixed_fixture(self) -> list[dict]:
        responses = [
            _make_response("r1", "s1", "train"),
            _make_response("r2", "s2", "test"),
        ]
        sources = [_make_source("s1"), _make_source("s2")]
        return ragtruth.join_records(responses, sources)

    def test_get_training_pool_excludes_test_split(self) -> None:
        records = self._mixed_fixture()
        pool = ragtruth.get_training_pool(records)
        self.assertTrue(all(r["split"] == "train" for r in pool))

    def test_get_eval_set_only_contains_reserved_records(self) -> None:
        records = self._mixed_fixture()
        eval_set = ragtruth.get_eval_set(records)
        self.assertTrue(all(r["eval_reserved"] is True for r in eval_set))
        self.assertTrue(all(r["split"] == "test" for r in eval_set))

    def test_no_leakage_fixture_passes_assertion(self) -> None:
        records = self._mixed_fixture()  # s1 only in train, s2 only in test
        ragtruth.assert_no_train_test_leakage(records)  # should not raise

    def test_leaking_fixture_raises(self) -> None:
        # Same source_id appearing in both splits - the exact condition the
        # real Step 6C inspection confirmed does NOT occur in RAGTruth
        # itself, but the safeguard must catch it if it ever did.
        responses = [
            _make_response("r1", "shared_source", "train"),
            _make_response("r2", "shared_source", "test"),
        ]
        sources = [_make_source("shared_source")]
        records = ragtruth.join_records(responses, sources)
        with self.assertRaises(AssertionError):
            ragtruth.assert_no_train_test_leakage(records)

    def test_real_sample_contains_only_train_and_is_not_eval_reserved(self) -> None:
        # The sample-builder script deliberately never reads test-split
        # data - verify that invariant holds in the actual written sample.
        records = _load_sample()
        self.assertTrue(all(r["split"] == "train" for r in records))
        self.assertTrue(all(r["eval_reserved"] is False for r in records))

    def test_real_data_has_no_cross_split_leakage(self) -> None:
        # Full-dataset version of the Step 6C leakage finding - skipped if
        # the full raw files aren't present (this test needs more than the
        # train-only sample to be meaningful).
        try:
            all_records = ragtruth.load_normalized()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw RAGTruth files not present")
        ragtruth.assert_no_train_test_leakage(all_records)  # should not raise


class TestRAGTruthValidateRecord(unittest.TestCase):
    def test_invalid_split_detected(self) -> None:
        record = {
            "response_text": "hi", "split": "validation", "source_missing": False,
            "task_type": "QA", "quality": "good", "labels": [],
        }
        issues = ragtruth.validate_record(record)
        self.assertIn("invalid_split:'validation'", issues)

    def test_empty_response_text_detected(self) -> None:
        record = {
            "response_text": "", "split": "train", "source_missing": False,
            "task_type": "QA", "quality": "good", "labels": [],
        }
        issues = ragtruth.validate_record(record)
        self.assertIn("empty_or_missing_response_text", issues)

    def test_source_missing_detected(self) -> None:
        record = {
            "response_text": "hi", "split": "train", "source_missing": True,
            "task_type": None, "quality": "good", "labels": [],
        }
        issues = ragtruth.validate_record(record)
        self.assertIn("source_missing", issues)

    def test_clean_record_has_no_issues(self) -> None:
        record = {
            "response_text": "hello", "split": "train", "source_missing": False,
            "task_type": "QA", "quality": "good", "labels": [{"start": 0, "end": 5, "text": "hello"}],
        }
        self.assertEqual(ragtruth.validate_record(record), [])


if __name__ == "__main__":
    unittest.main()
