"""ClaimGuard retrieval: corpus, embeddings, and FAISS index (Step 13).

Retrieval infrastructure only - no reranking (Step 14), no verifier
integration, no RAGTruth/TruthfulQA evaluation. See `corpus.py` for the
retrieval-objective and corpus-scope rationale, `embed.py` for the BGE
embedding pipeline, `index.py` for the FAISS index and query-time
`Retriever` API, and `eval.py` for the FEVER-only retrieval evaluation.
"""
