"""Tests for the FEVER dataset loader/normalizer (claimguard.datasets.fever).

Uses the small sample under data/processed/fever/sample/ (built by
scripts/build_fever_sample.py) plus small in-memory fixtures, so tests don't
require the full ~311k-row raw FEVER train file. No model loading, no GPU.

The Step 10 resolution tests build a tiny, self-contained SQLite index (a
handful of pages/sentences) in a temp directory - not the real ~6.4GB
wiki_pages_index.sqlite - so they stay fast and don't require the full
corpus to be downloaded/built to run.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from claimguard import config as cg_config
from claimguard.datasets import fever

SAMPLE_PATH = cg_config.resolve_path("data/processed/fever/sample/sample.jsonl")


def _load_sample() -> list[dict]:
    if not SAMPLE_PATH.exists():
        raise unittest.SkipTest(
            f"FEVER sample not found at {SAMPLE_PATH}; run scripts/build_fever_sample.py first."
        )
    records = []
    with SAMPLE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestFeverSampleLoads(unittest.TestCase):
    def test_sample_file_loads_and_is_nonempty(self) -> None:
        records = _load_sample()
        self.assertGreater(len(records), 0)


class TestFeverSchema(unittest.TestCase):
    def test_expected_fields_exist(self) -> None:
        records = _load_sample()
        expected_fields = {
            "example_id", "claim", "label", "verifier_label", "evidence", "evidence_ids", "split",
        }
        for r in records:
            self.assertTrue(expected_fields.issubset(r.keys()))


class TestFeverLabelNormalization(unittest.TestCase):
    def test_labels_normalize_correctly(self) -> None:
        records = _load_sample()
        for r in records:
            self.assertIn(r["label"], fever.VALID_FEVER_LABELS)
            self.assertEqual(r["verifier_label"], fever.FEVER_TO_VERIFIER_LABEL[r["label"]])

    def test_unmapped_label_yields_none_verifier_label(self) -> None:
        # Reproduces the real blank-label rows found in validation/test.
        rows = [
            {
                "id": 1, "claim": "x", "label": "", "evidence_annotation_id": 1,
                "evidence_id": -1, "evidence_wiki_url": "", "evidence_sentence_id": -1,
            }
        ]
        records = fever.group_claims(rows, split="test_fixture")
        self.assertIsNone(records[0]["verifier_label"])


class TestFeverEvidenceNormalization(unittest.TestCase):
    def test_supports_records_have_nonempty_evidence(self) -> None:
        records = _load_sample()
        for r in records:
            if r["label"] == "SUPPORTS":
                has_any = any(len(s) > 0 for s in r["evidence"])
                self.assertTrue(has_any, f"SUPPORTS record {r['example_id']} has no evidence")

    def test_not_enough_info_evidence_sets_are_empty(self) -> None:
        records = _load_sample()
        for r in records:
            if r["label"] == "NOT ENOUGH INFO":
                for s in r["evidence"]:
                    self.assertEqual(s, [])

    def test_group_claims_groups_denormalized_rows_by_claim(self) -> None:
        rows = [
            {
                "id": 1, "claim": "c1", "label": "SUPPORTS", "evidence_annotation_id": 10,
                "evidence_id": 100, "evidence_wiki_url": "Page_A", "evidence_sentence_id": 0,
            },
            {
                "id": 1, "claim": "c1", "label": "SUPPORTS", "evidence_annotation_id": 10,
                "evidence_id": 101, "evidence_wiki_url": "Page_A", "evidence_sentence_id": 1,
            },
            {
                "id": 2, "claim": "c2", "label": "REFUTES", "evidence_annotation_id": 20,
                "evidence_id": 102, "evidence_wiki_url": "Page_B", "evidence_sentence_id": 0,
            },
        ]
        records = fever.group_claims(rows, split="test_fixture")
        self.assertEqual(len(records), 2)
        c1 = next(r for r in records if r["example_id"] == 1)
        self.assertEqual(len(c1["evidence"]), 1)  # one evidence (annotation) set
        self.assertEqual(len(c1["evidence"][0]), 2)  # two evidence lines in that set


class TestFeverMalformedRecordDetection(unittest.TestCase):
    def test_invalid_label_detected(self) -> None:
        record = {"example_id": 1, "claim": "x", "label": "Not Enough Info", "evidence": [], "evidence_ids": []}
        issues = fever.validate_record(record)
        self.assertIn("invalid_label:'Not Enough Info'", issues)

    def test_empty_claim_detected(self) -> None:
        record = {
            "example_id": 1, "claim": "", "label": "SUPPORTS",
            "evidence": [[{"wiki_url": "Page", "sentence_id": 0}]], "evidence_ids": [1],
        }
        issues = fever.validate_record(record)
        self.assertIn("empty_or_missing_claim", issues)

    def test_supports_without_evidence_detected(self) -> None:
        record = {"example_id": 1, "claim": "x", "label": "SUPPORTS", "evidence": [[]], "evidence_ids": [1]}
        issues = fever.validate_record(record)
        self.assertIn("supports_or_refutes_without_evidence", issues)

    def test_not_enough_info_with_evidence_detected(self) -> None:
        record = {
            "example_id": 1, "claim": "x", "label": "NOT ENOUGH INFO",
            "evidence": [[{"wiki_url": "Page", "sentence_id": 0}]], "evidence_ids": [1],
        }
        issues = fever.validate_record(record)
        self.assertIn("not_enough_info_with_evidence", issues)

    def test_clean_record_has_no_issues(self) -> None:
        record = {
            "example_id": 1, "claim": "x", "label": "SUPPORTS",
            "evidence": [[{"wiki_url": "Page", "sentence_id": 0}]], "evidence_ids": [1],
        }
        issues = fever.validate_record(record)
        self.assertEqual(issues, [])


def _build_tiny_wiki_index(db_path: Path) -> None:
    """Build a minimal wiki_pages index for resolution tests: a couple of
    pages with sentences, an accented page title stored in NFC form (as the
    real wiki_pages corpus does - confirmed in Step 10's investigation, see
    check_unicode_normalization.py), and an intentionally absent page."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE sentences (page_id TEXT NOT NULL, sentence_id INTEGER NOT NULL, "
        "text TEXT NOT NULL, PRIMARY KEY (page_id, sentence_id))"
    )
    conn.execute(
        "CREATE TABLE pages (page_id TEXT PRIMARY KEY, has_content INTEGER NOT NULL, text TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO pages VALUES ('Page_A', 1, 'Page A intro text.')")
    conn.execute("INSERT INTO sentences VALUES ('Page_A', 0, 'Page A sentence zero.')")
    conn.execute("INSERT INTO sentences VALUES ('Page_A', 1, 'Page A sentence one.')")

    conn.execute("INSERT INTO pages VALUES ('Page_B', 1, 'Page B intro text.')")
    conn.execute("INSERT INTO sentences VALUES ('Page_B', 0, 'Page B sentence zero.')")

    # NFC-stored accented title (composed form, matching the real
    # wiki_pages corpus) - claim JSON evidence references will use the NFD
    # (decomposed) form below, exercising the NFC-fallback path exactly as
    # Step 10's real investigation found (claims JSON title != wiki_pages
    # title only in normalization form, never in content).
    import unicodedata
    nfc_title = unicodedata.normalize("NFC", "André_Téchiné")
    conn.execute("INSERT INTO pages VALUES (?, 1, 'Accented page intro.')", (nfc_title,))
    conn.execute("INSERT INTO sentences VALUES (?, 0, 'Accented page sentence zero.')", (nfc_title,))

    conn.commit()
    conn.close()


class TestFeverResolveSentence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.index_path = Path(self._tmpdir.name) / "wiki_pages_index.sqlite"
        _build_tiny_wiki_index(self.index_path)
        self.conn = sqlite3.connect(str(self.index_path))

    def tearDown(self) -> None:
        self.conn.close()
        self._tmpdir.cleanup()

    def test_resolves_real_sentence(self) -> None:
        text, reason = fever.resolve_sentence(self.conn, "Page_A", 0)
        self.assertEqual(text, "Page A sentence zero.")
        self.assertEqual(reason, "")

    def test_sentence_id_negative_one_resolves_via_page_text(self) -> None:
        text, reason = fever.resolve_sentence(self.conn, "Page_A", -1)
        self.assertEqual(text, "Page A intro text.")
        self.assertEqual(reason, "")

    def test_missing_page_reports_missing_page(self) -> None:
        text, reason = fever.resolve_sentence(self.conn, "Nonexistent_Page", 0)
        self.assertIsNone(text)
        self.assertEqual(reason, "missing_page")

    def test_missing_sentence_id_reports_missing_sentence_id(self) -> None:
        text, reason = fever.resolve_sentence(self.conn, "Page_A", 99)
        self.assertIsNone(text)
        self.assertEqual(reason, "missing_sentence_id")

    def test_nfd_title_resolves_against_nfc_stored_page(self) -> None:
        # The index stores the NFC (composed) form, exactly as the real
        # wiki_pages corpus does; the "claim" evidence reference below uses
        # NFD (decomposed) - deterministic canonicalization via the
        # NFC-fallback in _resolve_page_id, not fuzzy matching, per the
        # module docstring and Step 10's real investigation.
        import unicodedata
        nfd_title = unicodedata.normalize("NFD", "André_Téchiné")
        text, reason = fever.resolve_sentence(self.conn, nfd_title, 0)
        self.assertEqual(text, "Accented page sentence zero.")
        self.assertEqual(reason, "")

    def test_truly_absent_title_under_any_normalization_is_missing(self) -> None:
        text, reason = fever.resolve_sentence(self.conn, "Nonexistent_École", 0)
        self.assertIsNone(text)
        self.assertEqual(reason, "missing_page")


class TestFeverResolveClaimPremise(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.index_path = Path(self._tmpdir.name) / "wiki_pages_index.sqlite"
        _build_tiny_wiki_index(self.index_path)
        self.conn = sqlite3.connect(str(self.index_path))

    def tearDown(self) -> None:
        self.conn.close()
        self._tmpdir.cleanup()

    def test_not_enough_info_gets_no_evidence_annotation_status(self) -> None:
        record = {
            "example_id": 1, "claim": "x", "label": "NOT ENOUGH INFO",
            "evidence": [[]], "evidence_ids": [1],
        }
        result = fever.resolve_claim_premise(self.conn, record)
        self.assertEqual(result["resolution_status"], fever.RESOLUTION_STATUS_NO_EVIDENCE)
        self.assertIsNone(result["resolved_premise"])
        self.assertFalse(result["premise_text_available"])

    def test_unknown_label_is_not_confused_with_unresolved(self) -> None:
        record = {"example_id": 1, "claim": "x", "label": "", "evidence": [[]], "evidence_ids": [1]}
        result = fever.resolve_claim_premise(self.conn, record)
        self.assertEqual(result["resolution_status"], fever.RESOLUTION_STATUS_UNKNOWN_LABEL)
        self.assertFalse(result["premise_text_available"])

    def test_single_set_fully_resolves(self) -> None:
        record = {
            "example_id": 1, "claim": "x", "label": "SUPPORTS",
            "evidence": [[{"wiki_url": "Page_A", "sentence_id": 0}, {"wiki_url": "Page_A", "sentence_id": 1}]],
            "evidence_ids": [1],
        }
        result = fever.resolve_claim_premise(self.conn, record)
        self.assertEqual(result["resolution_status"], fever.RESOLUTION_STATUS_RESOLVED)
        self.assertTrue(result["premise_text_available"])
        self.assertEqual(result["resolved_premise"], "Page A sentence zero. Page A sentence one.")
        self.assertEqual(result["resolved_evidence_set_index"], 0)

    def test_first_partial_set_is_skipped_no_cross_set_mixing(self) -> None:
        # Set 0 is partially resolvable (one sentence missing) - must NOT be
        # used even partially. Set 1 fully resolves and must be used whole,
        # never mixed with any sentence from set 0.
        record = {
            "example_id": 1, "claim": "x", "label": "SUPPORTS",
            "evidence": [
                [{"wiki_url": "Page_A", "sentence_id": 0}, {"wiki_url": "Page_A", "sentence_id": 99}],
                [{"wiki_url": "Page_B", "sentence_id": 0}],
            ],
            "evidence_ids": [1, 2],
        }
        result = fever.resolve_claim_premise(self.conn, record)
        self.assertEqual(result["resolution_status"], fever.RESOLUTION_STATUS_RESOLVED)
        self.assertEqual(result["resolved_premise"], "Page B sentence zero.")
        self.assertEqual(result["resolved_evidence_set_index"], 1)

    def test_all_sets_unresolvable_yields_unresolved_status(self) -> None:
        record = {
            "example_id": 1, "claim": "x", "label": "REFUTES",
            "evidence": [[{"wiki_url": "Nonexistent_Page", "sentence_id": 0}]],
            "evidence_ids": [1],
        }
        result = fever.resolve_claim_premise(self.conn, record)
        self.assertEqual(result["resolution_status"], fever.RESOLUTION_STATUS_UNRESOLVED)
        self.assertIsNone(result["resolved_premise"])
        self.assertFalse(result["premise_text_available"])


class TestFeverLoadResolvedSplit(unittest.TestCase):
    """Wiring test: load_resolved_split correctly attaches resolution
    fields onto the normal group_claims output, without altering the
    preserved fields (claim, label, evidence, etc.)."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_dir = Path(self._tmpdir.name)
        self.raw_dir = tmp_dir / "raw"
        self.raw_dir.mkdir()
        self.index_path = tmp_dir / "wiki_pages_index.sqlite"
        _build_tiny_wiki_index(self.index_path)

        rows = [
            {
                "id": 1, "claim": "c1 supports", "label": "SUPPORTS", "evidence_annotation_id": 10,
                "evidence_id": 100, "evidence_wiki_url": "Page_A", "evidence_sentence_id": 0,
            },
            {
                "id": 2, "claim": "c2 nei", "label": "NOT ENOUGH INFO", "evidence_annotation_id": 20,
                "evidence_id": -1, "evidence_wiki_url": "", "evidence_sentence_id": -1,
            },
        ]
        with (self.raw_dir / "train.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_resolved_split_preserves_original_fields_and_adds_resolution_fields(self) -> None:
        records = fever.load_resolved_split("train", raw_dir=self.raw_dir, index_path=self.index_path)
        self.assertEqual(len(records), 2)

        r1 = next(r for r in records if r["example_id"] == 1)
        self.assertEqual(r1["claim"], "c1 supports")
        self.assertEqual(r1["label"], "SUPPORTS")
        self.assertEqual(r1["verifier_label"], "entailment")
        self.assertEqual(r1["resolution_status"], fever.RESOLUTION_STATUS_RESOLVED)
        self.assertTrue(r1["premise_text_available"])
        self.assertEqual(r1["resolved_premise"], "Page A sentence zero.")

        r2 = next(r for r in records if r["example_id"] == 2)
        self.assertEqual(r2["label"], "NOT ENOUGH INFO")
        self.assertEqual(r2["resolution_status"], fever.RESOLUTION_STATUS_NO_EVIDENCE)
        self.assertFalse(r2["premise_text_available"])
        self.assertIsNone(r2["resolved_premise"])

    def test_open_wiki_index_raises_when_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            fever.open_wiki_index(Path(self._tmpdir.name) / "does_not_exist.sqlite")


if __name__ == "__main__":
    unittest.main()
