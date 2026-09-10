"""Tests for the TruthfulQA dataset loader/normalizer (claimguard.datasets.truthfulqa).

Uses the small sample under data/processed/truthfulqa/sample/ (built by
scripts/build_truthfulqa_sample.py) plus small in-memory fixtures. No model
loading, no GPU. TruthfulQA has no official split - several tests here
specifically verify this module does NOT invent one.
"""

from __future__ import annotations

import json
import unittest

from claimguard import config as cg_config
from claimguard.datasets import truthfulqa

SAMPLE_PATH = cg_config.resolve_path("data/processed/truthfulqa/sample/sample.jsonl")


def _load_sample() -> list[dict]:
    if not SAMPLE_PATH.exists():
        raise unittest.SkipTest(
            f"TruthfulQA sample not found at {SAMPLE_PATH}; run scripts/build_truthfulqa_sample.py first."
        )
    records = []
    with SAMPLE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _make_csv_row(question, category="Health", type_="Adversarial",
                   correct="A; B; C", incorrect="X; Y"):
    return {
        "Type": type_, "Category": category, "Question": question,
        "Best Answer": "A", "Best Incorrect Answer": "X",
        "Correct Answers": correct, "Incorrect Answers": incorrect,
        "Source": "https://example.com",
    }


def _make_mc_record(question):
    return {
        "question": question,
        "mc0_targets": {"A": 1, "X": 0},
        "mc1_targets": {"A": 1, "X": 0, "Y": 0},
        "mc2_targets": {"A": 1, "B": 1, "X": 0, "Y": 0},
    }


class TestTruthfulQASampleLoads(unittest.TestCase):
    def test_sample_file_loads_and_is_nonempty(self) -> None:
        records = _load_sample()
        self.assertGreater(len(records), 0)

    def test_sample_covers_multiple_categories(self) -> None:
        records = _load_sample()
        categories = {r["category"] for r in records}
        self.assertGreater(len(categories), 1)
        # One record per category by construction - no category repeats.
        self.assertEqual(len(categories), len(records))


class TestTruthfulQASchema(unittest.TestCase):
    def test_expected_fields_exist(self) -> None:
        records = _load_sample()
        expected_fields = {
            "question_id", "question", "category", "type", "source", "best_answer",
            "best_incorrect_answer", "correct_answers", "incorrect_answers",
            "mc0_targets", "mc1_targets", "mc2_targets", "has_mc_data",
        }
        for r in records:
            self.assertTrue(expected_fields.issubset(r.keys()))

    def test_no_split_field_is_present(self) -> None:
        # TruthfulQA has no official split - the schema must not invent one.
        records = _load_sample()
        for r in records:
            self.assertNotIn("split", r)

    def test_module_does_not_expose_split_accessors(self) -> None:
        # Unlike claimguard.datasets.ragtruth, this module must not fabricate
        # get_eval_set()/get_training_pool() - no real split exists to back them.
        self.assertFalse(hasattr(truthfulqa, "get_eval_set"))
        self.assertFalse(hasattr(truthfulqa, "get_training_pool"))


class TestTruthfulQAParsing(unittest.TestCase):
    def test_parse_answer_list_splits_on_semicolon(self) -> None:
        result = truthfulqa.parse_answer_list("A; B; C")
        self.assertEqual(result, ["A", "B", "C"])

    def test_parse_answer_list_handles_single_item(self) -> None:
        self.assertEqual(truthfulqa.parse_answer_list("Only one"), ["Only one"])

    def test_parse_answer_list_handles_empty_or_none(self) -> None:
        self.assertEqual(truthfulqa.parse_answer_list(""), [])
        self.assertEqual(truthfulqa.parse_answer_list(None), [])


class TestTruthfulQAJoin(unittest.TestCase):
    def test_join_attaches_mc_targets(self) -> None:
        csv_rows = [_make_csv_row("Q1?")]
        mc_records = [_make_mc_record("Q1?")]
        joined = truthfulqa.join_records(csv_rows, mc_records)
        self.assertEqual(len(joined), 1)
        self.assertTrue(joined[0]["has_mc_data"])
        self.assertEqual(joined[0]["mc0_targets"], {"A": 1, "X": 0})

    def test_join_preserves_question_without_mc_match(self) -> None:
        csv_rows = [_make_csv_row("Unmatched question?")]
        mc_records = [_make_mc_record("Different question?")]
        joined = truthfulqa.join_records(csv_rows, mc_records)
        self.assertEqual(len(joined), 1)  # not dropped
        self.assertFalse(joined[0]["has_mc_data"])
        self.assertIsNone(joined[0]["mc1_targets"])

    def test_answer_lists_parsed_correctly_through_join(self) -> None:
        csv_rows = [_make_csv_row("Q1?", correct="A; B; C", incorrect="X; Y")]
        joined = truthfulqa.join_records(csv_rows, [])
        self.assertEqual(joined[0]["correct_answers"], ["A", "B", "C"])
        self.assertEqual(joined[0]["incorrect_answers"], ["X", "Y"])


class TestTruthfulQAMultipleChoiceStructure(unittest.TestCase):
    def test_mc2_can_have_multiple_correct_options(self) -> None:
        # Reproduces the real finding: mc2_targets allows multiple 1s
        # (multi-label), unlike mc0/mc1 (single correct option).
        records = _load_sample()
        found_multi_correct = any(
            sum(v for v in r["mc2_targets"].values()) > 1
            for r in records if r.get("mc2_targets")
        )
        self.assertTrue(found_multi_correct, "expected at least one sample record with "
                         "multiple correct options in mc2_targets")

    def test_mc0_has_exactly_two_options(self) -> None:
        records = _load_sample()
        for r in records:
            if r.get("mc0_targets"):
                self.assertEqual(len(r["mc0_targets"]), 2)


class TestTruthfulQADuplicateDetection(unittest.TestCase):
    def test_finds_real_duplicate_questions(self) -> None:
        records = [{"question": "same?"}, {"question": "same?"}, {"question": "other?"}]
        dups = truthfulqa.find_duplicate_questions(records)
        self.assertEqual(dups, {"same?": 2})

    def test_no_false_positive_on_unique_questions(self) -> None:
        records = [{"question": "a?"}, {"question": "b?"}]
        self.assertEqual(truthfulqa.find_duplicate_questions(records), {})

    def test_sample_has_no_duplicate_questions(self) -> None:
        records = _load_sample()
        self.assertEqual(truthfulqa.find_duplicate_questions(records), {})


class TestTruthfulQAValidateRecord(unittest.TestCase):
    def test_clean_record_has_no_issues(self) -> None:
        record = {
            "question": "Q?", "category": "Health", "type": "Adversarial",
            "best_answer": "A", "correct_answers": ["A"], "incorrect_answers": ["X"],
            "mc0_targets": {"A": 1, "X": 0}, "mc1_targets": {"A": 1, "X": 0},
            "mc2_targets": {"A": 1, "X": 0}, "has_mc_data": True,
        }
        self.assertEqual(truthfulqa.validate_record(record), [])

    def test_empty_question_detected(self) -> None:
        record = {
            "question": "", "category": "Health", "type": "Adversarial",
            "best_answer": "A", "correct_answers": ["A"], "incorrect_answers": ["X"],
            "mc0_targets": None, "mc1_targets": None, "mc2_targets": None, "has_mc_data": False,
        }
        self.assertIn("empty_or_missing_question", truthfulqa.validate_record(record))

    def test_unexpected_type_detected(self) -> None:
        record = {
            "question": "Q?", "category": "Health", "type": "Sometimes",
            "best_answer": "A", "correct_answers": ["A"], "incorrect_answers": ["X"],
            "mc0_targets": None, "mc1_targets": None, "mc2_targets": None, "has_mc_data": False,
        }
        issues = truthfulqa.validate_record(record)
        self.assertIn("unexpected_type:'Sometimes'", issues)

    def test_mc_targets_with_no_correct_option_detected(self) -> None:
        record = {
            "question": "Q?", "category": "Health", "type": "Adversarial",
            "best_answer": "A", "correct_answers": ["A"], "incorrect_answers": ["X"],
            "mc0_targets": {"A": 0, "X": 0}, "mc1_targets": None, "mc2_targets": None,
            "has_mc_data": True,
        }
        issues = truthfulqa.validate_record(record)
        self.assertIn("mc0_targets_has_no_correct_option", issues)

    def test_sample_records_pass_validation(self) -> None:
        records = _load_sample()
        for r in records:
            issues = truthfulqa.validate_record(r)
            self.assertEqual(issues, [], f"sample record {r['question_id']} has issues: {issues}")


if __name__ == "__main__":
    unittest.main()
