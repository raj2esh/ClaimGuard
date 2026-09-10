"""Step 16: real end-to-end smoke test for the deterministic evidence
decision policy - runs the real Step 15 pipeline on a small deterministic
sample, then applies `claimguard.decision.policy.decide` to the resulting
verified candidates. Does not process the full 2,000-query eval set - see
scripts/analyze_decision_policy.py for that.

Usage:
    python scripts/decision_policy_smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimguard import config as cg_config  # noqa: E402
from claimguard.retrieval import index as index_mod  # noqa: E402
from claimguard.reranking import reranker as reranker_mod  # noqa: E402
from claimguard.verification import pipeline as pipeline_mod  # noqa: E402
from claimguard.verification import verify as verify_mod  # noqa: E402
from claimguard.decision import policy as policy_mod  # noqa: E402

SAMPLE_QUERIES = [
    "Paris is the capital of France.",           # should be confidently SUPPORTED
    "The Eiffel Tower was built in Berlin.",      # should be confidently CONTRADICTED
    "Bananas are a good source of potassium.",    # plausible SUPPORTED or ABSTAIN depending on corpus coverage
]


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    cfg = cg_config.load_decision_policy_config()
    retrieval_top_k = cfg["pipeline"]["retrieval_top_k"]
    rerank_top_n = cfg["pipeline"]["rerank_top_n"]
    verifier_max_length = cfg["pipeline"]["verifier_max_length"]
    policy_cfg = cfg["decision_policy"]

    _section("LOADING MODELS")
    retriever = index_mod.Retriever.from_disk()
    reranker_cfg = reranker_mod.load_reranker_config()
    reranker_model, reranker_info = reranker_mod.load_reranker(reranker_cfg)
    checkpoint_dir = cg_config.resolve_path(cfg["verifier"]["checkpoint_dir"])
    verifier_model, verifier_tokenizer, verifier_info = verify_mod.load_verifier(checkpoint_dir)
    print(f"Verifier loaded: {verifier_info}")
    assert verifier_info["num_labels"] == 2, "expected a genuine binary head"

    policy = policy_mod.EvidenceDecisionPolicy(
        entailment_threshold=policy_cfg["entailment_threshold"],
        contradiction_threshold=policy_cfg["contradiction_threshold"],
    )

    _section("RUNNING PIPELINE + DECISION POLICY ON SAMPLE QUERIES")
    for query in SAMPLE_QUERIES:
        result = pipeline_mod.run_pipeline(
            retriever, reranker_model, verifier_model, verifier_tokenizer, query,
            retrieval_top_k=retrieval_top_k, rerank_top_n=rerank_top_n,
            reranker_batch_size=reranker_cfg["batch_size"], verifier_max_length=verifier_max_length,
        )
        decision = policy_mod.decide(query, result.verified_candidates, policy=policy)
        print(f"\nQuery: {query!r}")
        print(f"  decision={decision.decision} confidence={decision.confidence:.4f} reason={decision.reason}")
        print(f"  selected_evidence corpus_id={decision.selected_evidence['corpus_id'] if decision.selected_evidence else None}")
        assert decision.decision in ("SUPPORTED", "CONTRADICTED", "ABSTAIN")
        assert isinstance(decision.reason, str) and decision.reason
        assert decision.query == query
        if decision.decision != "ABSTAIN":
            assert decision.selected_evidence is not None
            for field in ("corpus_id", "text", "faiss_score", "reranker_score", "page_id", "sentence_id"):
                assert field in decision.selected_evidence

    _section("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
