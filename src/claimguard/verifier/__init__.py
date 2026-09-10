"""ClaimGuard verifier: NLI-style verifier training/evaluation.

- Step 9 (train.py): preliminary baseline - trains the project's
  already-selected NLI model
  (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) with its
  original pretrained 3-class head on HaluEval qa/dialogue/summarization
  only (FEVER excluded at the time - no resolved premise text available).
- Step 12 (train_binary.py, binary_model.py): FINAL binary verifier -
  trains the same base model with a genuine 2-class (entailment/
  contradiction) head on FEVER SUPPORTS/REFUTES (resolved premise text,
  Step 10) plus HaluEval qa/dialogue/summarization. No neutral class (Step
  11: no legitimate source exists in the current data).

No retrieval, no FAISS, no reranking, no custom architecture beyond the
documented classification-head reshaping in binary_model.py.
"""
