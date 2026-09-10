"""Step 16: verifier calibration + deterministic evidence-decision-policy
analysis over the IDENTICAL 2,000-query FEVER-train eval set as Steps
13/14/15 (cross-checked against Step 13's saved results, not just
re-derived with matching parameters).

Reuses (does NOT duplicate) Steps 13-15's evaluation logic:
    - claimguard.retrieval.eval.build_eval_set / first_hit_rank / recall_at_cutoffs
    - claimguard.retrieval.index.Retriever
    - claimguard.reranking.reranker
    - claimguard.verification.verify.load_verifier
    - claimguard.verification.pipeline.run_pipeline / select_evidence

Adds ONLY the new Step 16 analysis on top: 4-way gold/non-gold x
SUPPORTS/REFUTES score distributions, candidate-level threshold
precision/recall/F1/FPR/FNR, gold-evidence calibration (ECE/Brier/
reliability bins), evidence-selection policy comparison (A/B/C/D),
deterministic-policy abstention/false-support/false-contradiction rates
at each swept threshold, and margin-based evidence-consistency analysis.

The binary verifier is NOT modified, retrained, or fine-tuned anywhere in
this script. No RAGTruth, no TruthfulQA, no gold evidence/labels used
during pipeline inference - gold information is used only AFTER
`run_pipeline()` returns, for analysis.

Usage:
    python scripts/analyze_decision_policy.py
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.retrieval import eval as eval_mod  # noqa: E402
from claimguard.retrieval import index as index_mod  # noqa: E402
from claimguard.reranking import reranker as reranker_mod  # noqa: E402
from claimguard.verification import pipeline as pipeline_mod  # noqa: E402
from claimguard.verification import verify as verify_mod  # noqa: E402
from claimguard.decision import calibration as calib_mod  # noqa: E402
from claimguard.decision import policy as policy_mod  # noqa: E402

OUT_DIR = cg_config.resolve_path("data/processed/integration")
CALIBRATION_PATH = OUT_DIR / "calibration_results.json"
DECISION_PATH = OUT_DIR / "decision_policy_results.json"
STEP13_RESULTS_PATH = cg_config.resolve_path("data/processed/retrieval/retrieval_eval_results.json")


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    s = sorted(values)
    return {
        "count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p10": s[int(len(s) * 0.10)], "p90": s[int(len(s) * 0.90)],
        "min": s[0], "max": s[-1],
    }


def _threshold_confusion(records: list[dict], threshold: float) -> dict:
    """Candidate-level binary detection: predicted positive = entailment_probability
    >= threshold; actual positive (true support) = is_gold AND true_label=='SUPPORTS'.
    See claimguard.decision.policy module docstring for the full definition."""
    tp = fp = fn = tn = 0
    for r in records:
        predicted_positive = r["entailment_probability"] >= threshold
        actual_positive = r["is_gold"] and r["true_label"] == "SUPPORTS"
        if predicted_positive and actual_positive:
            tp += 1
        elif predicted_positive and not actual_positive:
            fp += 1
        elif not predicted_positive and actual_positive:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    fnr = fn / (fn + tp) if (fn + tp) else None
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fpr, "false_negative_rate": fnr,
        "n_accepted_support": tp + fp, "n_rejected": fn + tn,
    }


def main() -> int:
    cfg = cg_config.load_decision_policy_config()
    retrieval_top_k = cfg["pipeline"]["retrieval_top_k"]
    rerank_top_n = cfg["pipeline"]["rerank_top_n"]
    verifier_max_length = cfg["pipeline"]["verifier_max_length"]
    eval_seed = cfg["evaluation"]["seed"]
    eval_sample_size = cfg["evaluation"]["sample_size"]
    thresholds_swept = cfg["thresholds_swept"]
    n_bins = cfg["calibration"]["n_bins"]

    _section("LOADING MODELS")
    t_load_start = time.time()
    retriever = index_mod.Retriever.from_disk()
    reranker_cfg = reranker_mod.load_reranker_config()
    reranker_model, reranker_info = reranker_mod.load_reranker(reranker_cfg)
    checkpoint_dir = cg_config.resolve_path(cfg["verifier"]["checkpoint_dir"])
    verifier_model, verifier_tokenizer, verifier_info = verify_mod.load_verifier(checkpoint_dir)
    model_load_seconds = time.time() - t_load_start
    print(f"All models loaded in {model_load_seconds:.1f}s")

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _section("BUILDING EVAL SET (identical to Steps 13/14/15)")
    corpus_ids_in_corpus = {r["corpus_id"] for r in retriever.row_metadata}
    eval_set = eval_mod.build_eval_set(corpus_ids_in_corpus, seed=eval_seed, sample_size=eval_sample_size)
    print(f"Eval set size: {len(eval_set):,}")

    with STEP13_RESULTS_PATH.open() as f:
        step13_results = json.load(f)
    step13_ids = sorted(c["example_id"] for c in step13_results["retrieval_recall"]["per_claim"])
    this_ids = sorted(item["example_id"] for item in eval_set)
    same_set = step13_ids == this_ids
    print(f"Eval set identical to Step 13's saved eval set: {same_set}")
    if not same_set:
        print("FAIL: eval set mismatch - refusing to report results against a different query set.")
        return 1

    _section("RUNNING PIPELINE FOR EVERY EVAL CLAIM (collecting full per-candidate records)")
    all_candidate_records: list[dict] = []   # every verified candidate, every query
    gold_records: list[dict] = []            # subset: is_gold candidates only (calibration input)
    per_query: list[dict] = []

    # Per-threshold running policy-decision tallies (built inline, no extra GPU passes).
    policies = {t: policy_mod.EvidenceDecisionPolicy(entailment_threshold=t, contradiction_threshold=t) for t in thresholds_swept}
    policy_tallies = {t: {"n_supported": 0, "n_contradicted": 0, "n_abstain": 0,
                           "n_false_support": 0, "n_false_contradiction": 0} for t in thresholds_swept}

    start = time.time()
    for i, item in enumerate(eval_set):
        relevant = set(item["relevant_corpus_ids"])
        true_label = item["label"]  # "SUPPORTS" or "REFUTES"

        result = pipeline_mod.run_pipeline(
            retriever, reranker_model, verifier_model, verifier_tokenizer, item["claim"],
            retrieval_top_k=retrieval_top_k, rerank_top_n=rerank_top_n,
            reranker_batch_size=reranker_cfg["batch_size"], verifier_max_length=verifier_max_length,
        )
        verified = result.verified_candidates

        for c in verified:
            is_gold = c["corpus_id"] in relevant
            rec = {
                "entailment_probability": c["entailment_probability"],
                "contradiction_probability": c["contradiction_probability"],
                "is_gold": is_gold, "true_label": true_label, "corpus_id": c["corpus_id"],
            }
            all_candidate_records.append(rec)
            if is_gold:
                gold_records.append(rec)

        # Strategy A/B/C (A, B reuse Step 13/14/15 first_hit_rank-derived
        # top-1 hit; C reuses pipeline_mod.select_evidence exactly).
        faiss_top1_hit = bool(result.faiss_top1 and result.faiss_top1["corpus_id"] in relevant)
        reranker_top1_hit = bool(result.reranker_top1 and result.reranker_top1["corpus_id"] in relevant)
        selected, support_score = pipeline_mod.select_evidence(verified)
        strategy_c_hit = bool(selected and selected["corpus_id"] in relevant)

        agg = policy_mod.compute_query_aggregates(verified)

        per_query.append({
            "example_id": item["example_id"], "true_label": true_label,
            "faiss_top1_hit": faiss_top1_hit, "reranker_top1_hit": reranker_top1_hit,
            "strategy_c_hit": strategy_c_hit,
            "max_entailment_probability": agg["max_entailment_probability"],
            "second_highest_entailment_probability": agg["second_highest_entailment_probability"],
            "entailment_margin": agg["entailment_margin"],
            "max_contradiction_probability": agg["max_contradiction_probability"],
        })

        for t, pol in policies.items():
            d = pol.decide(verified)
            tally = policy_tallies[t]
            if d.decision == "SUPPORTED":
                tally["n_supported"] += 1
                true_support = bool(d.selected_evidence and d.selected_evidence["corpus_id"] in relevant
                                     and true_label == "SUPPORTS")
                if not true_support:
                    tally["n_false_support"] += 1
            elif d.decision == "CONTRADICTED":
                tally["n_contradicted"] += 1
                true_contra = bool(d.selected_evidence and d.selected_evidence["corpus_id"] in relevant
                                    and true_label == "REFUTES")
                if not true_contra:
                    tally["n_false_contradiction"] += 1
            else:
                tally["n_abstain"] += 1

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1:,}/{len(eval_set):,} claims processed ({time.time() - start:.1f}s elapsed)")

    total_loop_seconds = time.time() - start
    peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    print(f"Total loop time: {total_loop_seconds:.1f}s | peak GPU memory: {peak_gpu_gb}")
    n = len(eval_set)

    _section("1. SCORE DISTRIBUTIONS (4 groups)")
    groups = {
        "gold_supports": [r["entailment_probability"] for r in all_candidate_records if r["is_gold"] and r["true_label"] == "SUPPORTS"],
        "gold_refutes": [r["entailment_probability"] for r in all_candidate_records if r["is_gold"] and r["true_label"] == "REFUTES"],
        "non_gold_for_supports_claims": [r["entailment_probability"] for r in all_candidate_records if not r["is_gold"] and r["true_label"] == "SUPPORTS"],
        "non_gold_for_refutes_claims": [r["entailment_probability"] for r in all_candidate_records if not r["is_gold"] and r["true_label"] == "REFUTES"],
    }
    score_distributions = {name: _stats(vals) for name, vals in groups.items()}
    for name, s in score_distributions.items():
        print(f"  {name}: n={s.get('count')} mean={s.get('mean')} median={s.get('median')}")

    _section("2. CANDIDATE-LEVEL THRESHOLD ANALYSIS (characterization only, no threshold selected)")
    threshold_analysis = [_threshold_confusion(all_candidate_records, t) for t in thresholds_swept]
    for row in threshold_analysis:
        print(f"  t={row['threshold']}: precision={row['precision']} recall={row['recall']} "
              f"f1={row['f1']} fpr={row['false_positive_rate']} fnr={row['false_negative_rate']}")

    _section("3. CALIBRATION (gold-evidence candidates only)")
    gold_points = calib_mod.gold_calibration_points(gold_records)
    ece = calib_mod.expected_calibration_error(gold_points, n_bins=n_bins)
    brier = calib_mod.brier_score(gold_records)
    reliability = calib_mod.reliability_bins(gold_points, n_bins=n_bins)
    overconf = calib_mod.is_systematically_overconfident(gold_points, n_bins=n_bins)
    non_gold_confidence = _stats([r["entailment_probability"] for r in all_candidate_records if not r["is_gold"]])
    print(f"  ECE={ece} Brier={brier} overall_verdict={overconf['overall']}")

    _section("4. EVIDENCE-SELECTION POLICY COMPARISON (A/B/C/D)")
    strategy_a = sum(q["faiss_top1_hit"] for q in per_query) / n
    strategy_b = sum(q["reranker_top1_hit"] for q in per_query) / n
    strategy_c = sum(q["strategy_c_hit"] for q in per_query) / n
    strategy_d = {}
    for t in thresholds_swept:
        accepted = [q for q in per_query if q["max_entailment_probability"] is not None and q["max_entailment_probability"] >= t]
        n_accepted = len(accepted)
        n_accepted_and_gold = sum(1 for q in accepted if q["strategy_c_hit"])
        strategy_d[str(t)] = {
            "coverage": n_accepted / n,
            "abstention_rate": 1 - n_accepted / n,
            "precision_among_accepted": (n_accepted_and_gold / n_accepted) if n_accepted else None,
            "overall_hit_rate_incl_abstained_as_miss": n_accepted_and_gold / n,
        }
    print(f"  A (FAISS top-1): {strategy_a:.4f} | B (reranker top-1): {strategy_b:.4f} | C (verifier max-ent): {strategy_c:.4f}")
    for t, d in strategy_d.items():
        print(f"  D (t={t}): coverage={d['coverage']:.4f} precision_among_accepted={d['precision_among_accepted']}")

    _section("5. DETERMINISTIC POLICY: SUPPORTED/CONTRADICTED/ABSTAIN + false-support/false-contradiction rates")
    policy_outcomes = {}
    for t in thresholds_swept:
        tally = policy_tallies[t]
        policy_outcomes[str(t)] = {
            **tally,
            "supported_rate": tally["n_supported"] / n, "contradicted_rate": tally["n_contradicted"] / n,
            "abstain_rate": tally["n_abstain"] / n,
            "false_support_rate": (tally["n_false_support"] / tally["n_supported"]) if tally["n_supported"] else None,
            "false_contradiction_rate": (tally["n_false_contradiction"] / tally["n_contradicted"]) if tally["n_contradicted"] else None,
        }
        print(f"  t={t}: supported={tally['n_supported']} contradicted={tally['n_contradicted']} "
              f"abstain={tally['n_abstain']} false_support_rate={policy_outcomes[str(t)]['false_support_rate']}")

    _section("6. EVIDENCE CONSISTENCY (margin) ANALYSIS")
    margins = [q["entailment_margin"] for q in per_query if q["entailment_margin"] is not None]
    margin_median = statistics.median(margins) if margins else None
    high_margin = [q for q in per_query if q["entailment_margin"] is not None and q["entailment_margin"] >= (margin_median or 0)]
    low_margin = [q for q in per_query if q["entailment_margin"] is not None and q["entailment_margin"] < (margin_median or 0)]

    def _precision_among_accepted_at_default(rows: list[dict], t: float = 0.5) -> float | None:
        accepted = [q for q in rows if q["max_entailment_probability"] is not None and q["max_entailment_probability"] >= t]
        if not accepted:
            return None
        return sum(1 for q in accepted if q["strategy_c_hit"]) / len(accepted)

    consistency_analysis = {
        "margin_median": margin_median,
        "margin_distribution": _stats(margins),
        "high_margin_group": {"n": len(high_margin), "precision_among_accepted_at_0.5": _precision_among_accepted_at_default(high_margin)},
        "low_margin_group": {"n": len(low_margin), "precision_among_accepted_at_0.5": _precision_among_accepted_at_default(low_margin)},
        "note": "Descriptive grouping only - no learned fusion. Compares whether queries with a larger "
                "gap between the top-1 and top-2 entailment probability have higher evidence-selection precision.",
    }
    print(f"  high_margin precision={consistency_analysis['high_margin_group']['precision_among_accepted_at_0.5']} "
          f"low_margin precision={consistency_analysis['low_margin_group']['precision_among_accepted_at_0.5']}")

    _section("SAVING RESULTS")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    calibration_results = {
        "config": {"eval_seed": eval_seed, "eval_sample_size": eval_sample_size, "n_bins": n_bins,
                    "verifier_checkpoint_dir": str(checkpoint_dir)},
        "eval_set_identical_to_step13": same_set,
        "score_distributions": score_distributions,
        "non_gold_confidence_descriptive_only": non_gold_confidence,
        "gold_calibration": {
            "n_gold_points": len(gold_points), "ece": ece, "brier_score": brier,
            "reliability_bins": reliability, "overconfidence_verdict": overconf,
        },
        "note": "ECE/Brier computed ONLY over gold-evidence candidates - see "
                "claimguard.decision.calibration module docstring for why non-gold candidates have no "
                "legitimate ground-truth 'correct' label.",
    }
    with CALIBRATION_PATH.open("w", encoding="utf-8") as f:
        json.dump(calibration_results, f, indent=2, default=str)
    print(f"Saved {CALIBRATION_PATH}")

    decision_policy_results = {
        "config": {
            "retrieval_top_k": retrieval_top_k, "rerank_top_n": rerank_top_n,
            "verifier_max_length": verifier_max_length, "eval_seed": eval_seed,
            "eval_sample_size": eval_sample_size, "thresholds_swept": thresholds_swept,
        },
        "eval_set_identical_to_step13": same_set,
        "threshold_analysis_candidate_level": threshold_analysis,
        "evidence_selection_comparison": {
            "strategy_a_faiss_top1_hit_rate": strategy_a,
            "strategy_b_reranker_top1_hit_rate": strategy_b,
            "strategy_c_verifier_max_entailment_hit_rate": strategy_c,
            "strategy_d_verifier_with_min_confidence": strategy_d,
        },
        "deterministic_policy_outcomes": policy_outcomes,
        "evidence_consistency_analysis": consistency_analysis,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "total_loop_seconds": total_loop_seconds,
        "n_queries": n,
        "limitations": [
            "Thresholds are characterized, not selected - no 'best' threshold is declared final.",
            "Candidate-level threshold analysis treats gold-REFUTES and non-gold candidates identically "
            "as negatives for the 'accepted support' detector - see module docstring for the exact definition.",
            "Calibration (ECE/Brier) is restricted to gold-evidence candidates only; non-gold confidence is "
            "reported descriptively without an accuracy label.",
            "No RAGTruth/TruthfulQA evaluation performed here.",
        ],
    }
    with DECISION_PATH.open("w", encoding="utf-8") as f:
        json.dump(decision_policy_results, f, indent=2, default=str)
    print(f"Saved {DECISION_PATH}")

    manifest_extra = {
        "verifier_info": verifier_info, "reranker_info": reranker_info,
        "python_version": platform.python_version(), "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_load_seconds": model_load_seconds,
    }
    print(json.dumps(manifest_extra, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
