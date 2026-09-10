"""ClaimGuard reranking: BGE cross-encoder reranking of FAISS candidates (Step 14).

Isolated evaluation of retrieval-quality vs. reranking-quality - does NOT
call the verifier (that integration is a later, separate controlled step).
See `reranker.py` for the model loading, scoring, and two-stage
(FAISS top-K -> reranker top-N) pipeline.
"""
