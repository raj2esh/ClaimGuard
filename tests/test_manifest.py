"""Tests for the ClaimGuard dataset integration/manifest layer
(claimguard.datasets.manifest).

Covers: pool construction from permitted sources, role-boundary enforcement
(prohibited sources fail loudly), label mapping determinism, provenance
preservation, deterministic dev-split construction (including pair-grouping
for HaluEval), and cross-dataset leakage-check behavior. No model loading,
no GPU.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from claimguard.datasets import manifest as mf
from claimguard.datasets import ragtruth as ragtruth_ds
from claimguard.datasets import truthfulqa as truthfulqa_ds


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _build_tiny_wiki_index(db_path: Path) -> None:
    """Same minimal fixture index used in test_fever_dataset.py - just
    enough pages/sentences to resolve the small FEVER fixtures below,
    without requiring the real ~6.4GB wiki_pages_index.sqlite."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE sentences (page_id TEXT NOT NULL, sentence_id INTEGER NOT NULL, "
        "text TEXT NOT NULL, PRIMARY KEY (page_id, sentence_id))"
    )
    conn.execute(
        "CREATE TABLE pages (page_id TEXT PRIMARY KEY, has_content INTEGER NOT NULL, text TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO pages VALUES ('Page', 1, 'Page intro text.')")
    conn.execute("INSERT INTO sentences VALUES ('Page', 0, 'Page sentence zero.')")
    conn.execute("INSERT INTO pages VALUES ('Page2', 1, 'Page2 intro text.')")
    conn.execute("INSERT INTO sentences VALUES ('Page2', 0, 'Page2 sentence zero.')")
    conn.commit()
    conn.close()


class TestFeverPoolConstruction(unittest.TestCase):
    def test_fever_pool_includes_resolved_claims_and_excludes_not_enough_info(self) -> None:
        # Step 10: fever_train_pool_records only yields claims with real,
        # resolved premise text - SUPPORTS/REFUTES claims whose evidence
        # resolves against the wiki_pages index. NOT ENOUGH INFO claims
        # have no evidence annotation at all and are excluded, not coerced
        # into a trainable record with an invented premise.
        with tempfile.TemporaryDirectory() as td:
            raw_dir = Path(td) / "raw"
            raw_dir.mkdir()
            index_path = Path(td) / "wiki_pages_index.sqlite"
            _build_tiny_wiki_index(index_path)
            _write_jsonl(raw_dir / "train.jsonl", [
                {"id": 1, "label": "SUPPORTS", "claim": "c1", "evidence_annotation_id": 10,
                 "evidence_id": 100, "evidence_wiki_url": "Page", "evidence_sentence_id": 0},
                {"id": 2, "label": "REFUTES", "claim": "c2", "evidence_annotation_id": 20,
                 "evidence_id": 200, "evidence_wiki_url": "Page2", "evidence_sentence_id": 0},
                {"id": 3, "label": "NOT ENOUGH INFO", "claim": "c3", "evidence_annotation_id": 30,
                 "evidence_id": -1, "evidence_wiki_url": "", "evidence_sentence_id": -1},
            ])
            records = mf.fever_train_pool_records(raw_dir=raw_dir, index_path=index_path)
        self.assertEqual(len(records), 2)  # NOT ENOUGH INFO excluded, not coerced
        labels = {r.source_id: r.label for r in records}
        self.assertEqual(labels[1], "entailment")
        self.assertEqual(labels[2], "contradiction")
        self.assertNotIn(3, labels)

    def test_fever_records_have_resolved_premise_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_dir = Path(td) / "raw"
            raw_dir.mkdir()
            index_path = Path(td) / "wiki_pages_index.sqlite"
            _build_tiny_wiki_index(index_path)
            _write_jsonl(raw_dir / "train.jsonl", [
                {"id": 1, "label": "SUPPORTS", "claim": "c1", "evidence_annotation_id": 10,
                 "evidence_id": 100, "evidence_wiki_url": "Page", "evidence_sentence_id": 0},
            ])
            records = mf.fever_train_pool_records(raw_dir=raw_dir, index_path=index_path)
        self.assertTrue(records[0].premise_text_available)
        self.assertEqual(records[0].premise_text, "Page sentence zero.")
        self.assertIsNotNone(records[0].premise_reference)  # reference preserved, not discarded

    def test_fever_pool_excludes_unresolvable_claims(self) -> None:
        # A SUPPORTS/REFUTES claim whose evidence is NOT found in the index
        # must also be excluded (resolution_status == "unresolved"), never
        # silently promoted to a trainable record without real premise text.
        with tempfile.TemporaryDirectory() as td:
            raw_dir = Path(td) / "raw"
            raw_dir.mkdir()
            index_path = Path(td) / "wiki_pages_index.sqlite"
            _build_tiny_wiki_index(index_path)
            _write_jsonl(raw_dir / "train.jsonl", [
                {"id": 1, "label": "SUPPORTS", "claim": "c1", "evidence_annotation_id": 10,
                 "evidence_id": 100, "evidence_wiki_url": "Nonexistent_Page", "evidence_sentence_id": 0},
            ])
            records = mf.fever_train_pool_records(raw_dir=raw_dir, index_path=index_path)
        self.assertEqual(records, [])


class TestHaluEvalPoolConstruction(unittest.TestCase):
    def test_pair_subset_yields_both_labels_with_premise_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_dir = Path(td)
            _write_jsonl(raw_dir / "qa_data.json", [
                {"knowledge": "k1", "question": "q1", "right_answer": "correct",
                 "hallucinated_answer": "wrong"},
            ])
            records = mf.halueval_pair_pool_records("qa", raw_dir=raw_dir)
        self.assertEqual(len(records), 2)
        labels = {r.label for r in records}
        self.assertEqual(labels, {"entailment", "contradiction"})
        for r in records:
            self.assertTrue(r.premise_text_available)
            self.assertIn("k1", r.premise_text)

    def test_general_subset_rejected_by_pair_builder(self) -> None:
        with self.assertRaises(ValueError):
            mf.halueval_pair_pool_records("general")


class TestRoleBoundaryEnforcement(unittest.TestCase):
    def test_ragtruth_test_cannot_enter_training(self) -> None:
        with self.assertRaises(mf.RoleViolationError):
            mf.build_training_pool(["ragtruth_test"])

    def test_ragtruth_train_cannot_enter_training(self) -> None:
        with self.assertRaises(mf.RoleViolationError):
            mf.build_training_pool(["ragtruth_train"])

    def test_truthfulqa_cannot_enter_training(self) -> None:
        with self.assertRaises(mf.RoleViolationError):
            mf.build_training_pool(["truthfulqa"])

    def test_halueval_general_cannot_enter_training(self) -> None:
        with self.assertRaises(mf.RoleViolationError):
            mf.build_training_pool(["halueval_general"])

    def test_fever_validation_and_test_cannot_enter_training(self) -> None:
        with self.assertRaises(mf.RoleViolationError):
            mf.build_training_pool(["fever_validation"])
        with self.assertRaises(mf.RoleViolationError):
            mf.build_training_pool(["fever_test"])

    def test_mixed_valid_and_invalid_sources_still_rejected(self) -> None:
        # A single disallowed source anywhere in the request must fail the
        # whole call - not silently drop just the bad one.
        with self.assertRaises(mf.RoleViolationError):
            mf.build_training_pool(["halueval_qa", "ragtruth_test"])

    def test_allowed_sources_are_exactly_fever_train_and_halueval_pairs(self) -> None:
        self.assertEqual(mf.ALLOWED_TRAINING_SOURCES,
                          {"fever_train", "halueval_qa", "halueval_dialogue", "halueval_summarization"})


class TestDeterministicSplit(unittest.TestCase):
    def _fixture_pool(self) -> list[mf.PoolRecord]:
        pool = []
        for i in range(20):
            pool.append(mf.PoolRecord(
                hypothesis=f"h{i}a", label="entailment", original_label="not_hallucinated",
                source_dataset="halueval", source_subset="qa", source_id=f"qa_{i}_not_hallucinated",
                premise_text="ctx", premise_text_available=True, group_key=("halueval", "qa", i),
            ))
            pool.append(mf.PoolRecord(
                hypothesis=f"h{i}b", label="contradiction", original_label="hallucinated",
                source_dataset="halueval", source_subset="qa", source_id=f"qa_{i}_hallucinated",
                premise_text="ctx", premise_text_available=True, group_key=("halueval", "qa", i),
            ))
        return pool

    def test_split_is_deterministic_across_calls(self) -> None:
        pool = self._fixture_pool()
        train1, dev1 = mf.split_train_dev(pool, seed=7, dev_ratio=0.2)
        train2, dev2 = mf.split_train_dev(pool, seed=7, dev_ratio=0.2)
        self.assertEqual([r.source_id for r in train1], [r.source_id for r in train2])
        self.assertEqual([r.source_id for r in dev1], [r.source_id for r in dev2])

    def test_different_seed_can_produce_different_split(self) -> None:
        pool = self._fixture_pool()
        _, dev_a = mf.split_train_dev(pool, seed=1, dev_ratio=0.2)
        _, dev_b = mf.split_train_dev(pool, seed=2, dev_ratio=0.2)
        self.assertNotEqual({r.source_id for r in dev_a}, {r.source_id for r in dev_b})

    def test_pair_records_never_split_across_train_and_dev(self) -> None:
        # The critical leakage-prevention property: a HaluEval pair's two
        # expanded records (same group_key) must always land together.
        pool = self._fixture_pool()
        train, dev = mf.split_train_dev(pool, seed=7, dev_ratio=0.3)
        train_groups = {r.group_key for r in train}
        dev_groups = {r.group_key for r in dev}
        self.assertEqual(train_groups & dev_groups, set())

    def test_split_covers_entire_pool_with_no_duplication(self) -> None:
        pool = self._fixture_pool()
        train, dev = mf.split_train_dev(pool, seed=3, dev_ratio=0.25)
        all_ids = [r.source_id for r in train] + [r.source_id for r in dev]
        self.assertEqual(len(all_ids), len(pool))
        self.assertEqual(len(set(all_ids)), len(pool))  # no duplication


class TestLeakageCheck(unittest.TestCase):
    def test_check_exact_overlap_detects_real_overlap(self) -> None:
        pool = [mf.PoolRecord(
            hypothesis="The sky is blue", label="entailment", original_label="not_hallucinated",
            source_dataset="halueval", source_subset="qa", source_id="x",
            premise_text="Some context here", premise_text_available=True, group_key=("h", "qa", 0),
        )]
        eval_texts = {"response_text": {mf.normalize_text("The sky is blue")}}
        result = mf.check_exact_overlap(pool, eval_texts)
        self.assertEqual(result["response_text"]["hypothesis_overlap_count"], 1)

    def test_check_exact_overlap_zero_when_no_overlap(self) -> None:
        pool = [mf.PoolRecord(
            hypothesis="Totally unrelated text", label="entailment", original_label="not_hallucinated",
            source_dataset="halueval", source_subset="qa", source_id="x",
            premise_text="Some context", premise_text_available=True, group_key=("h", "qa", 0),
        )]
        eval_texts = {"response_text": {mf.normalize_text("Something completely different")}}
        result = mf.check_exact_overlap(pool, eval_texts)
        self.assertEqual(result["response_text"]["hypothesis_overlap_count"], 0)

    def test_normalize_text_is_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(mf.normalize_text("  Hello   World  "), mf.normalize_text("hello world"))


class TestProvenancePreservation(unittest.TestCase):
    def test_fever_record_provenance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_dir = Path(td) / "raw"
            raw_dir.mkdir()
            index_path = Path(td) / "wiki_pages_index.sqlite"
            _build_tiny_wiki_index(index_path)
            _write_jsonl(raw_dir / "train.jsonl", [
                {"id": 42, "label": "SUPPORTS", "claim": "c", "evidence_annotation_id": 1,
                 "evidence_id": 1, "evidence_wiki_url": "Page", "evidence_sentence_id": 0},
            ])
            records = mf.fever_train_pool_records(raw_dir=raw_dir, index_path=index_path)
        r = records[0]
        self.assertEqual(r.source_dataset, "fever")
        self.assertEqual(r.source_subset, "train")
        self.assertEqual(r.source_id, 42)
        self.assertEqual(r.original_label, "SUPPORTS")

    def test_halueval_record_provenance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw_dir = Path(td)
            _write_jsonl(raw_dir / "dialogue_data.json", [
                {"knowledge": "k", "dialogue_history": "h", "right_response": "r",
                 "hallucinated_response": "hr"},
            ])
            records = mf.halueval_pair_pool_records("dialogue", raw_dir=raw_dir)
        for r in records:
            self.assertEqual(r.source_dataset, "halueval")
            self.assertEqual(r.source_subset, "dialogue")
            self.assertIn(r.original_label, {"not_hallucinated", "hallucinated"})


class TestRealDataIntegration(unittest.TestCase):
    """Guarded by SkipTest if the full raw datasets aren't present - these
    verify the manifest logic against the actual acquired data, not just
    fixtures."""

    def test_ragtruth_eval_set_remains_intact_and_reserved(self) -> None:
        try:
            all_records = ragtruth_ds.load_normalized()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw RAGTruth files not present")
        eval_set = ragtruth_ds.get_eval_set(all_records)
        self.assertTrue(all(r["split"] == "test" for r in eval_set))
        self.assertTrue(all(r["eval_reserved"] for r in eval_set))
        self.assertGreater(len(eval_set), 0)

    def test_truthfulqa_has_no_split_field_and_stays_out_of_pool(self) -> None:
        try:
            records = truthfulqa_ds.load_normalized()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw TruthfulQA files not present")
        for r in records[:5]:
            self.assertNotIn("split", r)
        with self.assertRaises(mf.RoleViolationError):
            mf.build_training_pool(["truthfulqa"])

    def test_manifest_pool_counts_match_real_normalized_data(self) -> None:
        try:
            pool = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        from claimguard.datasets import halueval as halueval_ds
        try:
            fever_coverage = mf.fever_resolution_coverage()
        except FileNotFoundError:
            raise unittest.SkipTest("wiki_pages resolution index not present")
        # Step 10: FEVER contributes only claims with resolved premise text
        # (usable_for_training), not every normalized train claim - NOT
        # ENOUGH INFO and any genuinely-unresolved claims are excluded.
        expected = fever_coverage["usable_for_training"]
        for subset in ("qa", "dialogue", "summarization"):
            expected += len(halueval_ds.load_normalized_subset(subset))
        # Step 8: build_training_pool() excludes contaminated records by
        # default (exclude_contamination=True) - the raw normalized-data sum
        # above does not account for that. Each contaminated raw_index
        # removes 2 pool records (not_hallucinated + hallucinated), so
        # subtract that, not just the number of raw indices.
        expected -= 2 * len(mf.contaminated_halueval_summarization_raw_indices())
        self.assertEqual(len(pool), expected)


class TestFeverResolutionCoverageStep10(unittest.TestCase):
    """Guarded by SkipTest if the real wiki_pages index/full FEVER data
    aren't present - verifies the Step 10 resolution-coverage reporting
    against the actual acquired data, and that the honest neutral-class
    coverage gap is surfaced rather than hidden."""

    def test_fever_resolution_coverage_reports_real_counts(self) -> None:
        try:
            coverage = mf.fever_resolution_coverage()
        except FileNotFoundError:
            raise unittest.SkipTest("wiki_pages resolution index or full FEVER data not present")
        self.assertGreater(coverage["usable_for_training"], 0)
        self.assertIn("NOT ENOUGH INFO", coverage["status_by_label"])

    def test_neutral_class_from_fever_remains_zero(self) -> None:
        # The single most important Step 10 finding: 100% evidence
        # resolution for SUPPORTS/REFUTES does NOT create any usable
        # neutral-class (NOT ENOUGH INFO) examples, because those claims
        # have no evidence annotation in FEVER's own data at all. This test
        # exists so a future change that silently started fabricating or
        # coercing NEI premises would be caught immediately.
        try:
            coverage = mf.fever_resolution_coverage()
        except FileNotFoundError:
            raise unittest.SkipTest("wiki_pages resolution index or full FEVER data not present")
        self.assertEqual(coverage["neutral_class_available_from_fever"], 0)

    def test_verifier_pool_has_no_neutral_examples_yet(self) -> None:
        # Documents the standing limitation at the pool level too: since
        # HaluEval also never produces 'neutral', the full training pool's
        # neutral-class count mirrors FEVER's (currently zero).
        try:
            pool = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        neutral_records = [r for r in pool if r.label == "neutral"]
        self.assertEqual(len(neutral_records), 0)

    def test_fever_premise_resolution_status_reflects_real_index(self) -> None:
        status = mf.fever_premise_resolution_status()
        if not status["wiki_pages_index_present"]:
            raise unittest.SkipTest("wiki_pages resolution index not present")
        self.assertTrue(status["premise_text_currently_available"])
        self.assertGreater(status["resolution_coverage"]["usable_for_training"], 0)


class TestContaminationDetection(unittest.TestCase):
    """Step 8: cross-dataset contamination detection and exclusion."""

    def test_detects_known_document_overlap(self) -> None:
        # Small synthetic reproduction of the real finding's shape: one
        # shared document should produce exactly one audit record.
        import tempfile as _tempfile
        with _tempfile.TemporaryDirectory() as halueval_td, _tempfile.TemporaryDirectory() as ragtruth_td:
            halueval_dir = Path(halueval_td)
            ragtruth_dir = Path(ragtruth_td)
            shared_doc = "This is a shared CNN article about something specific."
            _write_jsonl(halueval_dir / "summarization_data.json", [
                {"document": shared_doc, "right_summary": "r1", "hallucinated_summary": "h1"},
                {"document": "An unrelated document.", "right_summary": "r2", "hallucinated_summary": "h2"},
            ])
            _write_jsonl(ragtruth_dir / "response.jsonl", [
                {"id": "resp1", "source_id": "src1", "model": "m", "temperature": 0.0,
                 "labels": [], "split": "test", "quality": "good", "response": "some response"},
            ])
            _write_jsonl(ragtruth_dir / "source_info.jsonl", [
                {"source_id": "src1", "task_type": "Summary", "source": "CNN/DM",
                 "source_info": shared_doc, "prompt": "Summarize this"},
            ])
            audit = mf.find_ragtruth_summarization_contamination(
                halueval_raw_dir=halueval_dir, ragtruth_raw_dir=ragtruth_dir,
            )
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["halueval_raw_index"], 0)
        self.assertEqual(audit[0]["ragtruth_source_id"], "src1")
        self.assertEqual(audit[0]["ragtruth_test_response_id"], "resp1")

    def test_no_overlap_produces_empty_audit(self) -> None:
        with tempfile.TemporaryDirectory() as halueval_td, tempfile.TemporaryDirectory() as ragtruth_td:
            halueval_dir = Path(halueval_td)
            ragtruth_dir = Path(ragtruth_td)
            _write_jsonl(halueval_dir / "summarization_data.json", [
                {"document": "Document A.", "right_summary": "r", "hallucinated_summary": "h"},
            ])
            _write_jsonl(ragtruth_dir / "response.jsonl", [
                {"id": "resp1", "source_id": "src1", "model": "m", "temperature": 0.0,
                 "labels": [], "split": "test", "quality": "good", "response": "resp"},
            ])
            _write_jsonl(ragtruth_dir / "source_info.jsonl", [
                {"source_id": "src1", "task_type": "Summary", "source": "CNN/DM",
                 "source_info": "Completely different document B.", "prompt": "p"},
            ])
            audit = mf.find_ragtruth_summarization_contamination(
                halueval_raw_dir=halueval_dir, ragtruth_raw_dir=ragtruth_dir,
            )
        self.assertEqual(audit, [])

    def test_non_summary_task_type_not_matched(self) -> None:
        # A matching text under a non-Summary task_type must NOT be treated
        # as contamination - the detector is specifically scoped to Summary.
        with tempfile.TemporaryDirectory() as halueval_td, tempfile.TemporaryDirectory() as ragtruth_td:
            halueval_dir = Path(halueval_td)
            ragtruth_dir = Path(ragtruth_td)
            shared_text = "Some text that happens to match."
            _write_jsonl(halueval_dir / "summarization_data.json", [
                {"document": shared_text, "right_summary": "r", "hallucinated_summary": "h"},
            ])
            _write_jsonl(ragtruth_dir / "response.jsonl", [
                {"id": "resp1", "source_id": "src1", "model": "m", "temperature": 0.0,
                 "labels": [], "split": "test", "quality": "good", "response": "resp"},
            ])
            _write_jsonl(ragtruth_dir / "source_info.jsonl", [
                {"source_id": "src1", "task_type": "QA", "source": "MARCO",
                 "source_info": {"question": shared_text, "passages": "p"}, "prompt": "p"},
            ])
            audit = mf.find_ragtruth_summarization_contamination(
                halueval_raw_dir=halueval_dir, ragtruth_raw_dir=ragtruth_dir,
            )
        self.assertEqual(audit, [])


class TestContaminationExclusionInPool(unittest.TestCase):
    """Guarded by SkipTest if the full raw datasets aren't present - these
    verify the exclusion mechanism against the actual acquired data."""

    def test_known_contaminated_indices_are_excluded_from_pool(self) -> None:
        try:
            indices = mf.contaminated_halueval_summarization_raw_indices()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        self.assertEqual(len(indices), 100)  # the real, previously-reported count

        excluded_keys = {("halueval", "summarization", i) for i in indices}
        pool = mf.build_training_pool()  # exclude_contamination=True by default
        pool_keys = {r.group_key for r in pool}
        self.assertEqual(pool_keys & excluded_keys, set())

    def test_excluded_records_absent_from_both_train_and_dev(self) -> None:
        try:
            pool = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        excluded_keys = mf.contamination_exclusion_group_keys()
        train, dev = mf.split_train_dev(pool)
        train_keys = {r.group_key for r in train}
        dev_keys = {r.group_key for r in dev}
        self.assertEqual(train_keys & excluded_keys, set())
        self.assertEqual(dev_keys & excluded_keys, set())

    def test_pair_grouping_still_intact_after_exclusion(self) -> None:
        try:
            pool = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        train, dev = mf.split_train_dev(pool)
        train_groups = {r.group_key for r in train}
        dev_groups = {r.group_key for r in dev}
        self.assertEqual(train_groups & dev_groups, set())

    def test_deterministic_rebuild_produces_identical_membership(self) -> None:
        try:
            pool_a = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        pool_b = mf.build_training_pool()
        train_a, dev_a = mf.split_train_dev(pool_a)
        train_b, dev_b = mf.split_train_dev(pool_b)
        self.assertEqual([r.group_key for r in train_a], [r.group_key for r in train_b])
        self.assertEqual([r.group_key for r in dev_a], [r.group_key for r in dev_b])

    def test_mitigation_resolves_the_original_broad_check(self) -> None:
        try:
            pool = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        result = mf.verify_contamination_resolved(pool)
        self.assertEqual(result["remaining_premise_overlap_with_ragtruth_source_info_text"], 0)

    def test_exclude_contamination_false_still_includes_contaminated_records(self) -> None:
        # Sanity check on the toggle itself - with the flag off, the
        # contaminated records ARE present (proves the default-True path is
        # actually doing something, not a no-op).
        try:
            pool_off = mf.build_training_pool(exclude_contamination=False)
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        excluded_keys = mf.contamination_exclusion_group_keys()
        pool_off_keys = {r.group_key for r in pool_off}
        self.assertTrue(excluded_keys.issubset(pool_off_keys))


class TestFinalBoundaryReVerification(unittest.TestCase):
    """Step 8 point 7: re-verify evaluation boundaries after the pool changed."""

    def test_ragtruth_test_still_hard_blocked_after_mitigation(self) -> None:
        with self.assertRaises(mf.RoleViolationError):
            mf.build_training_pool(["ragtruth_test"])

    def test_truthfulqa_still_hard_blocked_after_mitigation(self) -> None:
        with self.assertRaises(mf.RoleViolationError):
            mf.build_training_pool(["truthfulqa"])

    def test_ragtruth_test_source_ids_remain_disjoint_from_training(self) -> None:
        try:
            all_records = ragtruth_ds.load_normalized()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw RAGTruth files not present")
        ragtruth_ds.assert_no_train_test_leakage(all_records)  # should not raise

    def test_truthfulqa_generic_overlap_still_classified_benign(self) -> None:
        try:
            pool = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        result = mf.check_exact_overlap(pool, mf.truthfulqa_eval_texts())
        # These overlaps (short generic tokens like "yes"/"no"/country
        # names) were confirmed benign in Step 7/8 - re-verify the count is
        # still small/consistent with that classification, not a sudden
        # spike that would indicate a real regression.
        answer_overlap = result["answer"]["hypothesis_overlap_count"]
        self.assertLess(answer_overlap, 200, "unexpectedly large TruthfulQA overlap - "
                         "investigate before treating as benign")


class TestManifestConsistency(unittest.TestCase):
    """Step 8 point 11: manifest counts match actual files; exclusion report
    matches the final manifest. Guarded by SkipTest if the manifest/report
    files haven't been generated in this environment."""

    def test_manifest_pool_count_matches_written_pool_files(self) -> None:
        from claimguard import config as cg_config
        manifest_path = cg_config.resolve_path("data/processed/dataset_manifest.json")
        train_path = cg_config.resolve_path("data/processed/verifier_pool/train.jsonl")
        dev_path = cg_config.resolve_path("data/processed/verifier_pool/dev.jsonl")
        if not (manifest_path.exists() and train_path.exists() and dev_path.exists()):
            raise unittest.SkipTest("manifest/pool files not generated in this environment")

        with manifest_path.open() as f:
            manifest = json.load(f)
        with train_path.open() as f:
            train_count = sum(1 for _ in f)
        with dev_path.open() as f:
            dev_count = sum(1 for _ in f)

        self.assertEqual(manifest["verifier_training_pool"]["dev_split"]["train_count"], train_count)
        self.assertEqual(manifest["verifier_training_pool"]["dev_split"]["dev_count"], dev_count)
        self.assertEqual(
            manifest["verifier_training_pool"]["total_records"], train_count + dev_count,
        )

    def test_exclusion_report_matches_manifest_contamination_section(self) -> None:
        from claimguard import config as cg_config
        manifest_path = cg_config.resolve_path("data/processed/dataset_manifest.json")
        report_path = cg_config.resolve_path("data/processed/contamination_report.json")
        if not (manifest_path.exists() and report_path.exists()):
            raise unittest.SkipTest("manifest/contamination report not generated in this environment")

        with manifest_path.open() as f:
            manifest = json.load(f)
        with report_path.open() as f:
            report = json.load(f)

        manifest_excluded = manifest["verifier_training_pool"]["excluded_contamination_records"]
        report_excluded = report["pool_before_vs_after"]["records_excluded"]
        self.assertEqual(manifest_excluded, report_excluded)


class TestVerifierDatasetLabelSpaceStep11(unittest.TestCase):
    """Step 11: neutral-class strategy and classification-mode determination.

    Covers: synthetic-fixture correctness of verifier_dataset_label_space_status
    for binary/three-way/invalid pools (so a future silent label
    reinterpretation - e.g. someone starts mapping a HaluEval field to
    'neutral' - would be caught as INVALID rather than accepted), and
    real-data confirmation that the current pool is correctly binary with
    zero fabricated neutral examples.
    """

    @staticmethod
    def _record(label: str, source_dataset: str = "halueval", idx: int = 0) -> mf.PoolRecord:
        return mf.PoolRecord(
            hypothesis=f"h{idx}", label=label, original_label=label,
            source_dataset=source_dataset, source_subset="qa", source_id=idx,
            premise_text="ctx", premise_text_available=True, group_key=(source_dataset, "qa", idx),
        )

    def test_binary_pool_two_labels_reports_binary_mode(self) -> None:
        pool = [self._record("entailment", idx=0), self._record("contradiction", idx=1)]
        status = mf.verifier_dataset_label_space_status(pool)
        self.assertEqual(status["classification_mode"], mf.CLASSIFICATION_MODE_BINARY)
        self.assertFalse(status["supports_three_way"])
        self.assertEqual(status["missing_classes"], ["neutral"])

    def test_all_three_labels_present_reports_three_way_mode(self) -> None:
        pool = [
            self._record("entailment", idx=0),
            self._record("contradiction", idx=1),
            self._record("neutral", idx=2),
        ]
        status = mf.verifier_dataset_label_space_status(pool)
        self.assertEqual(status["classification_mode"], mf.CLASSIFICATION_MODE_THREE_WAY)
        self.assertTrue(status["supports_three_way"])
        self.assertEqual(status["missing_classes"], [])

    def test_only_neutral_present_is_invalid_not_binary(self) -> None:
        # Guards against a future bug silently treating "not the documented
        # two-class pattern" as if it were the known-good binary case.
        pool = [self._record("neutral", idx=0)]
        status = mf.verifier_dataset_label_space_status(pool)
        self.assertEqual(status["classification_mode"], mf.CLASSIFICATION_MODE_INVALID)

    def test_out_of_scheme_label_is_invalid_prohibited_relabeling_detected(self) -> None:
        # Simulates a prohibited label reinterpretation (e.g. code somewhere
        # starts emitting "hallucinated" directly as a verifier label
        # instead of the mapped entailment/contradiction/neutral scheme) -
        # must be flagged INVALID, never silently accepted as a valid mode.
        pool = [self._record("entailment", idx=0), self._record("hallucinated", idx=1)]
        status = mf.verifier_dataset_label_space_status(pool)
        self.assertEqual(status["classification_mode"], mf.CLASSIFICATION_MODE_INVALID)

    def test_empty_pool_reports_binary_gap_correctly_as_invalid(self) -> None:
        status = mf.verifier_dataset_label_space_status([])
        self.assertEqual(status["classification_mode"], mf.CLASSIFICATION_MODE_INVALID)
        self.assertEqual(status["available_label_space"], [])

    def test_real_pool_is_currently_binary_with_zero_neutral(self) -> None:
        try:
            pool = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        status = mf.verifier_dataset_label_space_status(pool)
        self.assertEqual(status["classification_mode"], mf.CLASSIFICATION_MODE_BINARY)
        self.assertEqual(status["class_counts"].get("neutral", 0), 0)
        self.assertFalse(status["supports_three_way"])

    def test_no_neutral_examples_are_fabricated_in_real_pool(self) -> None:
        # Direct anti-fabrication check: not one PoolRecord in the real
        # pool carries label == "neutral" - if this ever becomes nonzero,
        # it must be because a legitimate source was found and documented
        # (Step 11+), never a silent relabeling.
        try:
            pool = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        neutral_records = [r for r in pool if r.label == "neutral"]
        self.assertEqual(neutral_records, [])

    def test_fever_nei_claims_remain_excluded_from_real_pool(self) -> None:
        try:
            pool = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        fever_original_labels = {r.original_label for r in pool if r.source_dataset == "fever"}
        self.assertNotIn("NOT ENOUGH INFO", fever_original_labels)

    def test_manifest_declared_mode_matches_freshly_computed_mode(self) -> None:
        from claimguard import config as cg_config
        manifest_path = cg_config.resolve_path("data/processed/dataset_manifest.json")
        if not manifest_path.exists():
            raise unittest.SkipTest("manifest not generated in this environment")
        with manifest_path.open() as f:
            manifest = json.load(f)
        declared = manifest.get("verifier_dataset_status", {}).get("classification_mode")
        if declared is None:
            raise unittest.SkipTest("manifest predates Step 11's verifier_dataset_status block")
        try:
            pool = mf.build_training_pool()
        except FileNotFoundError:
            raise unittest.SkipTest("full raw dataset files not present")
        actual = mf.verifier_dataset_label_space_status(pool)["classification_mode"]
        self.assertEqual(
            declared, actual,
            "manifest's declared classification_mode is stale/wrong relative to the real pool - "
            "re-run scripts/build_dataset_manifest.py",
        )


if __name__ == "__main__":
    unittest.main()
