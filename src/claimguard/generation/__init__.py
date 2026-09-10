"""ClaimGuard generation: the Qwen3-8B answer-generation component (Step 17).

Strictly the generator role in `query -> generator -> candidate answer`.
Does NOT implement retrieval, reranking, verification, evidence decision,
or the correction/regeneration loop (that integration is Step 18+). See
`qwen3.py` for model loading/generation and `types.py` for the structured
result type.

The generator knows nothing about FEVER/HaluEval/RAGTruth/TruthfulQA
labels, gold evidence, or hallucination spans - its only inputs are
generic prompt text and generation parameters.
"""
