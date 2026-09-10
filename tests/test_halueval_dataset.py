"""Tests for the HaluEval dataset loader/normalizer (claimguard.datasets.halueval).

Uses the small sample under data/processed/halueval/sample/ (built by
scripts/build_halueval_sample.py) plus small in-memory fixtures, so most
tests don't require the full raw HaluEval files to be present. No model
loading, no GPU.
"""

from __future__ import annotations

import json
import unittest

from claimguard import config as cg_config
from claimguard.datasets import halueval

SAMPLE_PATH = cg_config.resolve_path("data/processed/halueval/sample/sample.jsonl")


def _load_sample() -> list[dict]:
    if not SAMPLE_PATH.exists():
        raise unittest.SkipTest(
            f"HaluEval sample not found at {SAMPLE_PATH}; run scripts/build_halueval_sample.py first."
        )
    records = []
    with SAMPLE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestHaluEvalSampleLoads(unittest.TestCase):
    def test_sample_file_loads_and_is_nonempty(self) -> None:
        records = _load_sample()
        self.assertGreater(len(records), 0)

    def test_all_four_subsets_represented(self) -> None:
        records = _load_sample()
        subsets = {r["subset"] for r in records}
        self.assertEqual(subsets, {"qa", "dialogue", "summarization", "general"})


class TestHaluEvalSchema(unittest.TestCase):
    def test_expected_fields_exist(self) -> None:
        records = _load_sample()
        expected_fields = {
            "example_id", "subset", "context", "query", "candidate_response",
            "label", "hallucination_spans", "source_field", "raw_id", "raw_index",
        }
        for r in records:
            self.assertTrue(expected_fields.issubset(r.keys()))

    def test_raw_index_present_and_unique_per_subset(self) -> None:
        records = _load_sample()
        by_subset: dict[str, set[int]] = {}
        for r in records:
            by_subset.setdefault(r["subset"], set())
            # raw_index is unique per RAW record, not per normalized record,
            # so qa/dialogue/summarization will see each raw_index twice
            # (once per expanded label) - that's expected, not a bug.
            by_subset[r["subset"]].add(r["raw_index"])
        for subset, indices in by_subset.items():
            self.assertGreater(len(indices), 0, f"no raw_index values found for {subset}")


class TestHaluEvalPairExpansion(unittest.TestCase):
    def test_expand_pair_record_yields_one_of_each_label(self) -> None:
        raw = {
            "knowledge": "k", "question": "q",
            "right_answer": "correct answer", "hallucinated_answer": "wrong answer",
        }
        records = halueval.expand_pair_record(raw, "qa", raw_index=0)
        self.assertEqual(len(records), 2)
        labels = {r["label"] for r in records}
        self.assertEqual(labels, {"hallucinated", "not_hallucinated"})

    def test_context_fields_differ_by_subset(self) -> None:
        qa = halueval.expand_pair_record(
            {"knowledge": "k", "question": "q", "right_answer": "a", "hallucinated_answer": "b"},
            "qa", 0,
        )
        self.assertEqual(set(qa[0]["context"].keys()), {"knowledge"})

        dialogue = halueval.expand_pair_record(
            {"knowledge": "k", "dialogue_history": "h", "right_response": "a", "hallucinated_response": "b"},
            "dialogue", 0,
        )
        self.assertEqual(set(dialogue[0]["context"].keys()), {"knowledge", "dialogue_history"})

        summarization = halueval.expand_pair_record(
            {"document": "d", "right_summary": "a", "hallucinated_summary": "b"},
            "summarization", 0,
        )
        self.assertEqual(set(summarization[0]["context"].keys()), {"document"})

    def test_sample_pair_records_have_correct_source_field(self) -> None:
        records = _load_sample()
        for r in records:
            if r["subset"] in halueval.PAIR_SUBSETS:
                if r["label"] == "not_hallucinated":
                    self.assertEqual(r["source_field"], halueval.PAIR_SUBSETS[r["subset"]]["right_field"])
                else:
                    self.assertEqual(r["source_field"], halueval.PAIR_SUBSETS[r["subset"]]["hallucinated_field"])


class TestHaluEvalGeneralSubset(unittest.TestCase):
    def test_label_derivation_from_yes_no(self) -> None:
        yes_record = halueval.normalize_general_record(
            {"ID": "1", "user_query": "q", "chatgpt_response": "r", "hallucination": "yes", "hallucination_spans": ["r"]},
            raw_index=0,
        )
        no_record = halueval.normalize_general_record(
            {"ID": "2", "user_query": "q", "chatgpt_response": "r", "hallucination": "no", "hallucination_spans": []},
            raw_index=1,
        )
        self.assertEqual(yes_record["label"], "hallucinated")
        self.assertEqual(no_record["label"], "not_hallucinated")

    def test_unmapped_hallucination_value_yields_none_label(self) -> None:
        record = halueval.normalize_general_record(
            {"ID": "3", "user_query": "q", "chatgpt_response": "r", "hallucination": "maybe", "hallucination_spans": []},
            raw_index=2,
        )
        self.assertIsNone(record["label"])

    def test_general_has_no_context_fields(self) -> None:
        records = _load_sample()
        general = [r for r in records if r["subset"] == "general"]
        self.assertGreater(len(general), 0)
        for r in general:
            self.assertEqual(r["context"], {})


class TestHaluEvalMalformedRecordDetection(unittest.TestCase):
    def test_empty_candidate_response_detected(self) -> None:
        record = {
            "example_id": "x", "subset": "qa", "context": {"knowledge": "k"}, "query": "q",
            "candidate_response": "", "label": "hallucinated", "hallucination_spans": None,
            "source_field": "hallucinated_answer", "raw_id": None, "raw_index": 0,
        }
        issues = halueval.validate_record(record)
        self.assertIn("empty_or_missing_candidate_response", issues)

    def test_invalid_label_detected(self) -> None:
        record = {
            "example_id": "x", "subset": "qa", "context": {"knowledge": "k"}, "query": "q",
            "candidate_response": "resp", "label": "maybe", "hallucination_spans": None,
            "source_field": "hallucinated_answer", "raw_id": None, "raw_index": 0,
        }
        issues = halueval.validate_record(record)
        self.assertIn("invalid_label:'maybe'", issues)

    def test_missing_context_detected_for_non_general_subset(self) -> None:
        record = {
            "example_id": "x", "subset": "qa", "context": {"knowledge": ""}, "query": "q",
            "candidate_response": "resp", "label": "hallucinated", "hallucination_spans": None,
            "source_field": "hallucinated_answer", "raw_id": None, "raw_index": 0,
        }
        issues = halueval.validate_record(record)
        self.assertIn("missing_context", issues)

    def test_missing_context_not_flagged_for_general_subset(self) -> None:
        record = {
            "example_id": "x", "subset": "general", "context": {}, "query": "q",
            "candidate_response": "resp", "label": "hallucinated", "hallucination_spans": ["resp"],
            "source_field": "chatgpt_response", "raw_id": "5", "raw_index": 0,
        }
        issues = halueval.validate_record(record)
        self.assertNotIn("missing_context", issues)

    def test_suspicious_raw_id_detected(self) -> None:
        # Reproduces the real anomaly found in general_data.json: ID == "ID"
        # (the header string leaked in as a data value) and ID == "".
        for bad_id in ("ID", ""):
            record = {
                "example_id": "x", "subset": "general", "context": {}, "query": "q",
                "candidate_response": "resp", "label": "not_hallucinated", "hallucination_spans": [],
                "source_field": "chatgpt_response", "raw_id": bad_id, "raw_index": 0,
            }
            issues = halueval.validate_record(record)
            self.assertIn(f"suspicious_raw_id:{bad_id!r}", issues)

    def test_clean_record_has_no_issues(self) -> None:
        record = {
            "example_id": "x", "subset": "qa", "context": {"knowledge": "k"}, "query": "q",
            "candidate_response": "resp", "label": "hallucinated", "hallucination_spans": None,
            "source_field": "hallucinated_answer", "raw_id": None, "raw_index": 0,
        }
        issues = halueval.validate_record(record)
        self.assertEqual(issues, [])


class TestHaluEvalSampleIntegrity(unittest.TestCase):
    def test_all_sample_records_pass_validation(self) -> None:
        records = _load_sample()
        for r in records:
            issues = halueval.validate_record(r)
            self.assertEqual(issues, [], f"sample record {r['example_id']} has issues: {issues}")


if __name__ == "__main__":
    unittest.main()
