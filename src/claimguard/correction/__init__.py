"""ClaimGuard correction/regeneration loop (Step 18): the bounded
ACCEPT / CORRECT / ABSTAIN orchestration over the already-built
generation (Step 17), retrieval (Step 13), reranking (Step 14), verifier
(Step 12/15), and decision-policy (Step 16) components.

    query -> Qwen3-8B -> candidate -> retrieve -> rerank -> verify ->
        Step 16 decision policy -> ACCEPT / CORRECT (bounded) / ABSTAIN

See `correction.py` for the orchestration and `types.py` for the
structured state/result types. This is an engineering/integration
validation step, not the final hallucination-detection evaluation -
RAGTruth (Step 19) and TruthfulQA (Step 20) evaluation are out of scope
here.
"""
