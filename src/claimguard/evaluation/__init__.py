"""ClaimGuard evaluation: metrics and scoring for hallucination/factuality
evaluation. `ragtruth_eval.py` (Step 19) implements RAGTruth's response-
level detection-prediction derivation, correction-outcome categorization,
and metric computation. `truthfulqa_eval.py` (Step 20) implements
TruthfulQA's MC1/MC2/MC0 arithmetic. `ablation_eval.py` (Step 21)
implements confidence-by-outcome and evidence-movement diagnostics for
controlled component ablations. This package contains ONLY evaluation-
layer logic - it never imports an inference-path module (retrieval/
reranking/verification/decision/generation/correction), so gold
evaluation labels can never accidentally reach an inference call through
it.
"""
