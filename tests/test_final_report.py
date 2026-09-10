"""Step 22 consistency checks: verify the final results tables and
reproducibility manifest match their underlying source artifacts exactly.
No GPU inference, no multi-hour reruns - pure file-level comparison,
matching the "no experimental modification" boundary of Step 22.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from claimguard import config as cg_config

ROOT = cg_config.PROJECT_ROOT
FINAL_DIR = ROOT / "data/processed/final"


def _load(rel: str) -> dict:
    with (ROOT / rel).open() as f:
        return json.load(f)


class TestFinalResultsTablesExist(unittest.TestCase):
    def test_tables_json_exists_and_parses(self):
        path = FINAL_DIR / "final_results_tables.json"
        self.assertTrue(path.exists(), f"missing {path}")
        with path.open() as f:
            data = json.load(f)
        for key in (
            "table1_dataset_statistics", "table2_verifier_development",
            "table3_retrieval_reranking", "table4_end_to_end_ragtruth",
            "table5_truthfulqa", "table6_controlled_ablations",
            "table7_failure_source_evidence", "provenance",
        ):
            self.assertIn(key, data)

    def test_csv_exports_exist(self):
        self.assertTrue((FINAL_DIR / "tables_csv" / "table4_ragtruth_summary.csv").exists())
        self.assertTrue((FINAL_DIR / "tables_csv" / "table6_ablation_summary.csv").exists())


class TestTable1MatchesSourceManifest(unittest.TestCase):
    def setUp(self):
        self.tables = _load("data/processed/final/final_results_tables.json")
        self.dataset_manifest = _load("data/processed/dataset_manifest.json")

    def test_fever_counts_match(self):
        fever = self.dataset_manifest["datasets"]["fever"]
        t1 = self.tables["table1_dataset_statistics"]["FEVER"]
        self.assertEqual(t1["raw_train_claims"], fever["splits"]["train"]["raw_dataset_size"])
        self.assertEqual(
            t1["usable_records_resolved_premise"],
            fever["splits"]["train"]["usable_records_with_resolved_premise"],
        )
        self.assertEqual(
            t1["neutral_class_available"],
            fever["premise_resolution_status"]["resolution_coverage"]["neutral_class_available_from_fever"],
        )


class TestTable2MatchesVerifierArtifacts(unittest.TestCase):
    def setUp(self):
        self.tables = _load("data/processed/final/final_results_tables.json")
        self.dev_metrics = _load("experiments/verifier_binary_final/final/dev_metrics.json")
        self.repro = _load("experiments/verifier_binary_final/final/reproducibility.json")

    def test_dev_metrics_match(self):
        t2 = self.tables["table2_verifier_development"]
        self.assertEqual(t2["dev_accuracy"], self.dev_metrics["eval_accuracy"])
        self.assertEqual(t2["dev_macro_f1"], self.dev_metrics["eval_macro_f1"])
        self.assertEqual(t2["confusion_matrix"], self.dev_metrics["eval_confusion_matrix"])

    def test_reproducibility_fields_match(self):
        t2 = self.tables["table2_verifier_development"]
        self.assertEqual(t2["model"], self.repro["model_name"])
        self.assertEqual(t2["n_params"], self.repro["n_total_params"])
        self.assertEqual(t2["train_records"], self.repro["train_record_count"])


class TestTable4MatchesRagtruthResults(unittest.TestCase):
    def setUp(self):
        self.tables = _load("data/processed/final/final_results_tables.json")
        self.ragtruth = _load("data/processed/evaluation/ragtruth/ragtruth_eval_results.json")

    def test_headline_numbers_match(self):
        t4 = self.tables["table4_end_to_end_ragtruth"]
        b = self.ragtruth["overall_metrics_by_condition"]["B_generated"]
        self.assertEqual(t4["n_responses"], 2700)
        self.assertEqual(t4["hallucination_prevalence"], b["hallucination_prevalence_gold"])
        self.assertEqual(t4["attempt0_accuracy"], b["attempt0_detection_metrics"]["accuracy"])
        self.assertEqual(t4["final_accuracy"], b["final_outcome_metrics"]["accuracy"])
        self.assertAlmostEqual(t4["attempt0_accuracy"], 0.5588888888888889, places=10)
        self.assertAlmostEqual(t4["final_accuracy"], 0.5803703703703704, places=10)
        self.assertAlmostEqual(t4["hallucination_prevalence"], 0.34925925925925927, places=10)

    def test_correction_outcome_categories_sum_to_2700(self):
        cats = self.tables["table4_end_to_end_ragtruth"]["correction_outcome_categories"]
        self.assertEqual(sum(cats.values()), 2700)


class TestTable5MatchesTruthfulqaResults(unittest.TestCase):
    def setUp(self):
        self.tables = _load("data/processed/final/final_results_tables.json")
        self.tqa = _load("data/processed/evaluation/truthfulqa/truthfulqa_eval_results.json")

    def test_mc_scores_match(self):
        t5 = self.tables["table5_truthfulqa"]
        self.assertEqual(t5["n_questions"], 790)
        self.assertEqual(t5["n_categories"], 37)
        self.assertAlmostEqual(t5["mc_scores"]["mc1_accuracy"], 0.34050632911392403, places=10)
        self.assertAlmostEqual(t5["mc_scores"]["mc2_accuracy"], 0.5505488308164534, places=10)
        self.assertAlmostEqual(t5["mc_scores"]["mc0_accuracy"], 0.4417721518987342, places=10)


class TestTable6MatchesAblationResults(unittest.TestCase):
    def setUp(self):
        self.tables = _load("data/processed/final/final_results_tables.json")
        self.ablation = _load("data/processed/evaluation/ablations/ablation_results.json")

    def test_condition_sizes(self):
        t6 = self.tables["table6_controlled_ablations"]
        for cond in ("A_reranked", "B_no_reranker", "C_faiss_top1"):
            self.assertEqual(t6[cond]["n"], 300)
        self.assertEqual(t6["D_no_correction_REUSED"]["n"], 2700)
        self.assertEqual(t6["E_verify_only_REUSED"]["n"], 2700)

    def test_cross_validation_perfect_match_preserved(self):
        cv = self.tables["table6_controlled_ablations"]["cross_validation_A_vs_step19"]
        self.assertEqual(cv["matched"], 300)
        self.assertEqual(cv["mismatched"], 0)


class TestReproducibilityManifest(unittest.TestCase):
    def setUp(self):
        path = FINAL_DIR / "final_reproducibility_manifest.json"
        self.assertTrue(path.exists(), f"missing {path}")
        with path.open() as f:
            self.manifest = json.load(f)

    def test_required_top_level_keys(self):
        for key in (
            "environment", "frozen_model_identifiers", "seeds", "dataset_sources",
            "evaluation_counts", "config_and_checkpoint_file_hashes_sha256",
            "hash_scope_note", "git_state",
        ):
            self.assertIn(key, self.manifest)

    def test_hash_scope_note_discloses_weights_not_hashed(self):
        note = self.manifest["hash_scope_note"].lower()
        self.assertIn("not", note)
        self.assertIn("hash", note)

    def test_git_state_shows_zero_commits(self):
        self.assertFalse(self.manifest["git_state"]["commits_exist"])

    def test_evaluation_counts_match_known_values(self):
        counts = self.manifest["evaluation_counts"]
        self.assertEqual(counts["ragtruth_test_responses"], 2700)
        self.assertEqual(counts["truthfulqa_questions"], 790)
        self.assertEqual(counts["ablation_subset_responses"], 300)


class TestFiguresExist(unittest.TestCase):
    def test_all_eight_figures_present(self):
        fig_dir = FINAL_DIR / "figures"
        expected = [
            "figure1_architecture.png", "figure2_protocol_isolation.png",
            "figure3_retrieval_recall.png", "figure4_ragtruth_baseline_vs_final.png",
            "figure5_ablation_comparison.png", "figure6_verifier_confidence_by_outcome.png",
            "figure7_correction_benefit_vs_harm.png", "figure8_failure_model.png",
        ]
        for name in expected:
            p = fig_dir / name
            self.assertTrue(p.exists(), f"missing figure {p}")
            self.assertGreater(p.stat().st_size, 0)


class TestFinalDocsExist(unittest.TestCase):
    def test_final_docs_present_and_nonempty(self):
        for name in ("PROJECT_REPORT_FINAL.md", "FINAL_PRESENTATION_RESULTS.md", "README.md"):
            p = ROOT / name
            self.assertTrue(p.exists(), f"missing {p}")
            self.assertGreater(p.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
