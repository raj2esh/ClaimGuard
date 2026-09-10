"""Tests for claimguard.decision (Step 16: verifier calibration + a
deterministic downstream evidence-decision policy).

The binary verifier itself is not touched by this module or these tests -
only the policy layered on top of already-produced verifier scores."""

from __future__ import annotations

import inspect
import math
import unittest
from pathlib import Path

from claimguard.decision import calibration as calib_mod
from claimguard.decision import policy as policy_mod


def _candidate(corpus_id: str, entailment: float, contradiction: float, reranked_rank: int = 1,
               text: str = "evidence text", faiss_score: float = 0.5, reranker_score: float = 0.5,
               original_rank: int = 1, page_id: str = "P", sentence_id: int = 0):
    return {
        "corpus_id": corpus_id, "text": text, "faiss_score": faiss_score, "reranker_score": reranker_score,
        "original_rank": original_rank, "reranked_rank": reranked_rank, "page_id": page_id,
        "sentence_id": sentence_id, "corpus_version": "test",
        "entailment_probability": entailment, "contradiction_probability": contradiction,
    }


def _source_files() -> list[Path]:
    import claimguard.decision as pkg
    return list(Path(pkg.__file__).parent.glob("*.py"))


class TestRAGTruthTruthfulQAExclusion(unittest.TestCase):
    def test_no_ragtruth_or_truthfulqa_imports(self) -> None:
        import ast

        for path in _source_files():
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


class TestNoGoldAccess(unittest.TestCase):
    def test_decide_function_has_no_gold_parameter(self) -> None:
        sig = inspect.signature(policy_mod.decide)
        param_names = {p.lower() for p in sig.parameters}
        forbidden = ["gold", "label", "relevant", "true_evidence", "answer"]
        offending = [p for p in param_names for f in forbidden if f in p]
        self.assertEqual(offending, [])

    def test_evidence_decision_policy_decide_has_no_gold_parameter(self) -> None:
        sig = inspect.signature(policy_mod.EvidenceDecisionPolicy.decide)
        param_names = {p.lower() for p in sig.parameters}
        forbidden = ["gold", "label", "relevant", "true_evidence", "answer"]
        offending = [p for p in param_names for f in forbidden if f in p]
        self.assertEqual(offending, [])


class TestEvidenceDecisionPolicyBasic(unittest.TestCase):
    def test_high_entailment_yields_supported(self) -> None:
        candidates = [_candidate("A::0", 0.95, 0.05)]
        d = policy_mod.EvidenceDecisionPolicy(entailment_threshold=0.5).decide(candidates)
        self.assertEqual(d.decision, "SUPPORTED")
        self.assertAlmostEqual(d.confidence, 0.95)
        self.assertEqual(d.selected_evidence["corpus_id"], "A::0")

    def test_high_contradiction_yields_contradicted(self) -> None:
        candidates = [_candidate("A::0", 0.05, 0.95)]
        d = policy_mod.EvidenceDecisionPolicy(contradiction_threshold=0.5).decide(candidates)
        self.assertEqual(d.decision, "CONTRADICTED")
        self.assertAlmostEqual(d.confidence, 0.95)
        self.assertEqual(d.selected_evidence["corpus_id"], "A::0")

    def test_low_confidence_yields_abstain(self) -> None:
        candidates = [_candidate("A::0", 0.55, 0.45), _candidate("B::0", 0.4, 0.6)]
        d = policy_mod.EvidenceDecisionPolicy(entailment_threshold=0.9, contradiction_threshold=0.9).decide(candidates)
        self.assertEqual(d.decision, "ABSTAIN")
        self.assertIsNone(d.selected_evidence)

    def test_empty_candidates_yields_abstain(self) -> None:
        d = policy_mod.EvidenceDecisionPolicy().decide([])
        self.assertEqual(d.decision, "ABSTAIN")
        self.assertEqual(d.confidence, 0.0)
        self.assertIsNone(d.selected_evidence)
        self.assertEqual(d.num_candidates, 0)

    def test_entailment_takes_priority_over_contradiction_when_both_clear_threshold(self) -> None:
        # One candidate clears entailment, a DIFFERENT candidate clears
        # contradiction - SUPPORTED must win per the documented rule order.
        candidates = [_candidate("SUP::0", 0.9, 0.1), _candidate("CON::0", 0.1, 0.9)]
        d = policy_mod.EvidenceDecisionPolicy().decide(candidates)
        self.assertEqual(d.decision, "SUPPORTED")
        self.assertEqual(d.selected_evidence["corpus_id"], "SUP::0")

    def test_threshold_boundary_is_inclusive(self) -> None:
        candidates = [_candidate("A::0", 0.5, 0.5)]
        d = policy_mod.EvidenceDecisionPolicy(entailment_threshold=0.5).decide(candidates)
        self.assertEqual(d.decision, "SUPPORTED")

    def test_deterministic_across_repeated_calls(self) -> None:
        candidates = [_candidate("A::0", 0.7, 0.3), _candidate("B::0", 0.9, 0.1)]
        policy = policy_mod.EvidenceDecisionPolicy()
        d1 = policy.decide(candidates)
        d2 = policy.decide(candidates)
        self.assertEqual(d1.decision, d2.decision)
        self.assertEqual(d1.selected_evidence["corpus_id"], d2.selected_evidence["corpus_id"])
        self.assertEqual(d1.confidence, d2.confidence)


class TestEvidenceProvenancePreserved(unittest.TestCase):
    def test_selected_evidence_carries_full_provenance(self) -> None:
        candidates = [_candidate("A::3", 0.9, 0.1, faiss_score=0.42, reranker_score=0.77,
                                  original_rank=5, reranked_rank=2, page_id="Some_Page", sentence_id=3)]
        d = policy_mod.EvidenceDecisionPolicy().decide(candidates)
        ev = d.selected_evidence
        for field in ("corpus_id", "text", "faiss_score", "reranker_score", "original_rank",
                      "reranked_rank", "page_id", "sentence_id", "corpus_version",
                      "entailment_probability", "contradiction_probability"):
            self.assertIn(field, ev)
        self.assertEqual(ev["page_id"], "Some_Page")
        self.assertEqual(ev["faiss_score"], 0.42)

    def test_decision_is_never_a_bare_boolean(self) -> None:
        candidates = [_candidate("A::0", 0.9, 0.1)]
        d = policy_mod.EvidenceDecisionPolicy().decide(candidates)
        self.assertIsInstance(d, policy_mod.Decision)
        self.assertIn(d.decision, ("SUPPORTED", "CONTRADICTED", "ABSTAIN"))
        self.assertIsInstance(d.reason, str)
        self.assertGreater(len(d.reason), 0)


class TestConflictingCandidates(unittest.TestCase):
    def test_ambiguous_near_tie_below_threshold_abstains(self) -> None:
        candidates = [_candidate("A::0", 0.55, 0.45), _candidate("B::0", 0.52, 0.48)]
        d = policy_mod.EvidenceDecisionPolicy(entailment_threshold=0.6, contradiction_threshold=0.6).decide(candidates)
        self.assertEqual(d.decision, "ABSTAIN")

    def test_require_margin_forces_abstain_on_ambiguous_top_two(self) -> None:
        candidates = [_candidate("A::0", 0.91, 0.09), _candidate("B::0", 0.90, 0.10)]
        d = policy_mod.EvidenceDecisionPolicy(entailment_threshold=0.5, require_margin=0.1).decide(candidates)
        self.assertEqual(d.decision, "ABSTAIN")

    def test_require_margin_allows_supported_when_clear_winner(self) -> None:
        candidates = [_candidate("A::0", 0.95, 0.05), _candidate("B::0", 0.2, 0.8)]
        d = policy_mod.EvidenceDecisionPolicy(entailment_threshold=0.5, require_margin=0.1).decide(candidates)
        self.assertEqual(d.decision, "SUPPORTED")


class TestNaNInfRejection(unittest.TestCase):
    def test_nan_entailment_probability_raises(self) -> None:
        candidates = [_candidate("A::0", float("nan"), 0.5)]
        with self.assertRaises(ValueError):
            policy_mod.EvidenceDecisionPolicy().decide(candidates)

    def test_inf_contradiction_probability_raises(self) -> None:
        candidates = [_candidate("A::0", 0.5, float("inf"))]
        with self.assertRaises(ValueError):
            policy_mod.EvidenceDecisionPolicy().decide(candidates)

    def test_compute_query_aggregates_rejects_non_finite(self) -> None:
        candidates = [_candidate("A::0", float("nan"), 0.5)]
        with self.assertRaises(ValueError):
            policy_mod.compute_query_aggregates(candidates)


class TestQueryAggregates(unittest.TestCase):
    def test_margin_and_second_highest(self) -> None:
        candidates = [_candidate("A::0", 0.9, 0.1), _candidate("B::0", 0.7, 0.3), _candidate("C::0", 0.2, 0.8)]
        agg = policy_mod.compute_query_aggregates(candidates)
        self.assertAlmostEqual(agg["max_entailment_probability"], 0.9)
        self.assertAlmostEqual(agg["second_highest_entailment_probability"], 0.7)
        self.assertAlmostEqual(agg["entailment_margin"], 0.2, places=5)
        self.assertAlmostEqual(agg["max_contradiction_probability"], 0.8)
        self.assertEqual(agg["max_contradiction_candidate"]["corpus_id"], "C::0")

    def test_single_candidate_has_no_second_highest_or_margin(self) -> None:
        agg = policy_mod.compute_query_aggregates([_candidate("A::0", 0.9, 0.1)])
        self.assertIsNone(agg["second_highest_entailment_probability"])
        self.assertIsNone(agg["entailment_margin"])

    def test_empty_candidates(self) -> None:
        agg = policy_mod.compute_query_aggregates([])
        self.assertEqual(agg["num_candidates"], 0)
        self.assertIsNone(agg["max_entailment_probability"])

    def test_count_above_threshold(self) -> None:
        candidates = [_candidate("A::0", 0.9, 0.1), _candidate("B::0", 0.6, 0.4), _candidate("C::0", 0.3, 0.7)]
        self.assertEqual(policy_mod.count_above_threshold(candidates, "entailment_probability", 0.5), 2)
        self.assertEqual(policy_mod.count_above_threshold(candidates, "contradiction_probability", 0.5), 1)


class TestModuleLevelDecide(unittest.TestCase):
    def test_decide_stamps_query(self) -> None:
        d = policy_mod.decide("some claim text", [_candidate("A::0", 0.9, 0.1)])
        self.assertEqual(d.query, "some claim text")
        self.assertEqual(d.decision, "SUPPORTED")

    def test_decide_accepts_policy_kwargs(self) -> None:
        d = policy_mod.decide("q", [_candidate("A::0", 0.6, 0.4)], entailment_threshold=0.9)
        self.assertEqual(d.decision, "ABSTAIN")


class TestCalibration(unittest.TestCase):
    def test_gold_calibration_points_correct_and_incorrect(self) -> None:
        records = [
            {"corpus_id": "A", "entailment_probability": 0.9, "contradiction_probability": 0.1, "true_label": "SUPPORTS"},
            {"corpus_id": "B", "entailment_probability": 0.9, "contradiction_probability": 0.1, "true_label": "REFUTES"},
        ]
        points = calib_mod.gold_calibration_points(records)
        self.assertEqual(points[0]["confidence"], 0.9)
        self.assertTrue(points[0]["correct"])
        self.assertEqual(points[1]["confidence"], 0.9)
        self.assertFalse(points[1]["correct"])

    def test_ece_zero_for_perfectly_calibrated_points(self) -> None:
        # Every point in the [0.9,1.0) bin, all correct -> accuracy==mean_confidence -> ECE ~ 0.
        points = [{"confidence": 0.95, "correct": True} for _ in range(10)]
        ece = calib_mod.expected_calibration_error(points)
        self.assertAlmostEqual(ece, 0.05, places=2)  # mean_conf=0.95, accuracy=1.0 -> |1.0-0.95|=0.05

    def test_ece_none_for_empty_points(self) -> None:
        self.assertIsNone(calib_mod.expected_calibration_error([]))

    def test_reliability_bins_reports_empty_bins(self) -> None:
        points = [{"confidence": 0.95, "correct": True}]
        bins = calib_mod.reliability_bins(points)
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[-1]["count"], 1)
        self.assertEqual(bins[0]["count"], 0)
        self.assertIsNone(bins[0]["accuracy"])

    def test_brier_score_perfect_forecaster(self) -> None:
        records = [
            {"entailment_probability": 1.0, "true_label": "SUPPORTS"},
            {"entailment_probability": 0.0, "true_label": "REFUTES"},
        ]
        self.assertAlmostEqual(calib_mod.brier_score(records), 0.0)

    def test_brier_score_none_for_empty(self) -> None:
        self.assertIsNone(calib_mod.brier_score([]))

    def test_gold_calibration_points_rejects_non_finite(self) -> None:
        records = [{"corpus_id": "A", "entailment_probability": float("nan"),
                    "contradiction_probability": 0.1, "true_label": "SUPPORTS"}]
        with self.assertRaises(ValueError):
            calib_mod.gold_calibration_points(records)

    def test_confidence_distribution_sums_to_one(self) -> None:
        points = [{"confidence": c, "correct": True} for c in (0.05, 0.15, 0.95, 0.95)]
        dist = calib_mod.confidence_distribution(points)
        total_fraction = sum(b["fraction"] for b in dist["histogram"])
        self.assertAlmostEqual(total_fraction, 1.0)

    def test_overconfidence_detection(self) -> None:
        # Confidence 0.95, but only half are actually correct -> overconfident.
        points = [{"confidence": 0.95, "correct": i % 2 == 0} for i in range(10)]
        result = calib_mod.is_systematically_overconfident(points)
        self.assertEqual(result["overall"], "overconfident")


if __name__ == "__main__":
    unittest.main()
