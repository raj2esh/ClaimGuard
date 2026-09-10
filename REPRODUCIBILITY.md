# ClaimGuard — Reproducibility Guide

This document explains how to reproduce ClaimGuard from source **without**
the ~40GB of locally generated artifacts (raw datasets, model checkpoints,
FAISS index, embeddings, training pool) that are intentionally excluded from
this repository — see `ARTIFACTS.md` for exactly what is excluded and why.

This guide describes the steps and scripts used to build the project. It is
**not** a claim that running these steps end-to-end has been re-verified
from a clean checkout as part of writing this document — the pipeline was
built and validated incrementally over Steps 1-22 (see `PROJECT_REPORT.md`),
each step's outputs read the previous step's already-produced artifacts.

**Driver script**: `scripts/run_full_pipeline.sh` sequences every stage
below (Sections 5, 11-16) using exactly the scripts referenced in this
document — it does not add any new logic. Run
`scripts/run_full_pipeline.sh --list` to see all stages,
`--dry-run` to print the exact commands without running anything, or
`--fast` to run a small pilot/smoke-test-scale pass (where the underlying
script supports one) instead of the full multi-hour run. It does not
download raw datasets for you (Section 5) and does not replace the
judgment calls in Sections 2-4 (environment/GPU prerequisites).

## 1. Project overview

ClaimGuard is a retrieval-augmented hallucination-detection and
self-correction pipeline: retrieve evidence for a candidate answer (FAISS
over a FEVER/Wikipedia corpus), rerank it, verify it with a fine-tuned NLI
model, apply a deterministic decision policy (ACCEPT/CORRECT/ABSTAIN), and
optionally run a bounded evidence-grounded correction loop. See
`PROJECT_REPORT_FINAL.md` for the full scientific summary, including the
project's negative/mixed findings — this is documentation for reproducing
the *system*, not a claim that the system reliably detects hallucinations.

## 2. Python / environment requirements

- Python 3.12.14
- Dependencies pinned in `requirements.txt` (also declared via `pyproject.toml`,
  installable with `pip install -e .`)
- Key libraries: `torch==2.11.0+cu128`, `transformers==5.16.1`,
  `sentence-transformers==6.0.1`, `faiss-cpu==1.15.0`, `datasets==5.0.1`,
  `scikit-learn==1.9.0`, `matplotlib==3.11.1`

```bash
conda create -n claimguard python=3.12
conda activate claimguard
pip install -r requirements.txt
pip install -e . --no-deps
```

## 3. GPU requirements

- A CUDA 12.8-capable GPU with enough free VRAM to hold the full inference
  pipeline concurrently: generator (Qwen3-8B, bf16) + verifier (DeBERTa-v3-large)
  + embedder (BGE-large) + reranker (BGE-reranker-large) — roughly 23-25GB
  VRAM for inference; this project used an NVIDIA RTX PRO 5000 Blackwell
  (~48GB) shared with another unrelated workload.
- Verifier training (`src/claimguard/verifier/train_binary.py`) additionally
  needs headroom for gradient/optimizer state — run in isolation from the
  rest of the pipeline if VRAM is tight.
- CPU-only steps (dataset manifest building, contamination detection, final
  results-table/figure generation) do not need a GPU at all.

## 4. Main model dependencies

All model identifiers are configured in `configs/models.yaml`, never
hard-coded — see `CLAIMGUARD_MODEL_SELECTION.md` for the full selection
rationale.

| Role | Model |
|---|---|
| Generator | `Qwen/Qwen3-8B` |
| Verifier | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (fine-tuned to a binary entailment/contradiction head — see `experiments/verifier_binary_final/final/reproducibility.json`) |
| Embedding | `BAAI/bge-large-en-v1.5` |
| Reranker | `BAAI/bge-reranker-large` |

## 5. Dataset acquisition instructions

Raw datasets are **not** included in this repository (see `ARTIFACTS.md`).
Acquire each from its authoritative source, then run this project's
inspection/sampling scripts against your local copy:

- `scripts/inspect_fever.py`, `scripts/build_fever_sample.py`,
  `scripts/resolve_fever_evidence.py`, `scripts/acquire_inspect_wiki_pages.py`,
  `scripts/build_wiki_pages_index.py` — FEVER acquisition, evidence
  resolution against Wikipedia `wiki_pages`, and the retrieval-corpus SQLite
  index build.
- `scripts/inspect_halueval.py`, `scripts/build_halueval_sample.py` — HaluEval.
- `scripts/inspect_ragtruth.py`, `scripts/build_ragtruth_sample.py` — RAGTruth.
- `scripts/inspect_truthfulqa.py`, `scripts/build_truthfulqa_sample.py` — TruthfulQA.
- `scripts/build_dataset_manifest.py` — builds `data/processed/dataset_manifest.json`
  from all four acquired datasets (tracked in this repo — see it for the
  exact per-dataset record counts this project reports).
- `scripts/detect_contamination.py` — reproduces the cross-dataset
  contamination sweep (`data/processed/contamination_report.json`, tracked).

## 6. Dataset licenses

| Dataset | License |
|---|---|
| FEVER | CC BY-SA 3.0 + GNU FDL |
| HaluEval | MIT |
| RAGTruth | MIT |
| TruthfulQA | Apache-2.0 |

This project does not claim ownership of any of these datasets and does not
redistribute their raw content — see Section 5 for acquisition and
`ARTIFACTS.md` for what generated-from-them artifacts are excluded.

## 7. FEVER

- Authoritative source: the FEVER shared task (`fever.ai`) / the `fever`
  dataset on Hugging Face Datasets.
- License: CC BY-SA 3.0 + GNU FDL.
- **Wiki reconstruction requirement**: FEVER's evidence is reference-only
  (`evidence_wiki_url` / `evidence_sentence_id`) — the actual Wikipedia
  `wiki_pages` text must be separately acquired and resolved against those
  references (`scripts/acquire_inspect_wiki_pages.py`,
  `scripts/resolve_fever_evidence.py`) before evidence sentences exist as
  usable text. This project's resolution status/coverage is recorded in
  `data/processed/dataset_manifest.json`.

## 8. HaluEval

- Authoritative source: `RUCAIBox/HaluEval`.
- License: MIT.
- Used as a verifier-training supplement (qa/dialogue/summarization
  subsets), normalized to a common `(premise, claim, label)` schema before
  pooling with FEVER (see `scripts/build_verifier_pool.py`). 200 pool records
  were excluded due to a confirmed exact-match overlap with RAGTruth test
  sources — see Section 9 and `data/processed/contamination_report.json`.

## 9. RAGTruth

- Authoritative source: `ParticleMedia/RAGTruth`.
- License: MIT.
- **Test-set reservation**: RAGTruth's `test` split is this project's
  primary held-out end-to-end benchmark and was never used for training —
  enforced by a `RoleViolationError` guard in
  `src/claimguard/datasets/manifest.py`'s `build_training_pool()`, not just
  convention. `scripts/evaluate_ragtruth.py` reproduces the full 2,700-response
  evaluation (`data/processed/evaluation/ragtruth/ragtruth_eval_results.json`,
  tracked).

## 10. TruthfulQA

- Authoritative source: `sylinrl/TruthfulQA`.
- License: Apache-2.0.
- Evaluation-only (no train/dev role at all). `scripts/evaluate_truthfulqa.py`
  reproduces the full 790-question evaluation
  (`data/processed/evaluation/truthfulqa/truthfulqa_eval_results.json`, tracked).

## 11. Retrieval index construction

1. `scripts/build_retrieval_corpus.py` — builds the sentence-level FEVER/
   Wikipedia retrieval corpus (`data/processed/retrieval/corpus.jsonl`,
   excluded from Git — large generated artifact).
2. `scripts/embed_retrieval_corpus.py` — embeds the corpus with
   `BAAI/bge-large-en-v1.5` (`embeddings.npy`, excluded).
3. `scripts/build_faiss_index.py` — builds the FAISS index
   (`faiss_index.bin`, excluded).
4. `scripts/evaluate_retrieval.py`, `scripts/evaluate_reranker.py` — produce
   the tracked recall@K summaries (`retrieval_eval_results.json`,
   `reranker_eval_results.json`).

## 12. Verifier training

1. `scripts/build_dataset_manifest.py` + `scripts/detect_contamination.py`
   — dataset manifest and contamination resolution (Section 5).
2. `scripts/build_verifier_pool.py` — builds the contamination-filtered
   train/dev pool (`data/processed/verifier_pool/{train,dev}.jsonl`,
   excluded — large, fully reproducible from this step).
3. `src/claimguard/verifier/train_binary.py` — trains the binary
   (entailment/contradiction) verifier head on top of
   `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, reusing the
   pretrained encoder/pooler and transplanting (not randomly initializing)
   the entailment/contradiction classifier rows. Run as a script
   (`argparse`-based); see `experiments/verifier_binary_final/final/training_config.yaml`
   (tracked) for the exact hyperparameters this project's reported checkpoint
   was trained with (seed 42, lr 2e-5, effective batch size 32, bf16, 1 epoch).
4. `scripts/validate_verifier_dataset.py` — dataset/schema validation used
   before training.

The trained checkpoint's weights (`model.safetensors`) are excluded from
Git; its small metadata/metrics files
(`experiments/verifier_binary_final/final/{config.json, dev_metrics.json,
reproducibility.json, training_config.yaml, ...}`) are tracked and record
everything needed to verify a retrained checkpoint against this project's
reported numbers.

## 13. Generator configuration

`Qwen/Qwen3-8B`, loaded bf16, `enable_thinking=False` throughout (no hidden
reasoning trace generated or stored). Configuration in
`configs/generator_baseline.yaml`. Smoke-tested via `scripts/generator_smoke_test.py`
and `scripts/smoke_test_qwen.py`.

## 14. Evaluation pipeline

- `scripts/evaluate_integration.py` — retrieval + reranking + verification
  integration metrics (`data/processed/integration/integration_eval_results.json`,
  tracked) and calibration/decision-policy analysis
  (`calibration_results.json`, `decision_policy_results.json`, tracked).
- `scripts/decision_policy_smoke_test.py`, `scripts/correction_smoke_test.py`,
  `scripts/integration_smoke_test.py` — smaller integration/engineering
  validation checks.
- `scripts/evaluate_ragtruth.py` — full RAGTruth end-to-end evaluation
  (attempt0 detection + post-correction final outcome), reading the trained
  verifier, retrieval index, decision policy, and correction loop
  (`src/claimguard/correction/correction.py`'s `run_correction_loop()`).
  Writes the tracked `data/processed/evaluation/ragtruth/` result/manifest/
  error-analysis JSON files (the large per-response `*_response_results.jsonl`
  dump is excluded from Git).
- `scripts/evaluate_truthfulqa.py` — equivalent full evaluation on
  TruthfulQA's 790 questions.
- `scripts/analyze_ragtruth_results.py`, `scripts/analyze_truthfulqa_results.py`,
  `scripts/analyze_decision_policy.py` — post-hoc analysis/error-breakdown
  scripts that read the above result files.

## 15. Ablation evaluation

- `scripts/run_ablations.py` — runs the controlled ablation conditions
  (A: reranked, B: no reranker, C: FAISS top-1 only) on a 300-response
  RAGTruth subset; Conditions D (no-correction) and E (verify-only) are
  extracted directly from the already-computed full-2,700 RAGTruth results
  rather than recomputed.
- `scripts/analyze_ablations.py` — produces
  `data/processed/evaluation/ablations/{ablation_results.json,
  ablation_error_analysis.json, ablation_comparison.csv}` (all tracked).

## 16. Expected major reported metrics

(All traced to tracked artifacts — see `data/processed/final/final_results_tables.json`
and `PROJECT_REPORT_FINAL.md` for the full set.)

| Metric | Value | Source |
|---|---|---|
| Verifier dev accuracy / macro F1 | 96.20% / 0.9585 | `experiments/verifier_binary_final/final/dev_metrics.json` |
| FAISS recall@1/5/10/20 | 28.00% / 56.85% / 66.20% / 73.25% | `data/processed/retrieval/reranker_eval_results.json` |
| RAGTruth final accuracy / macro F1 | 58.04% / 0.5798 | `data/processed/evaluation/ragtruth/ragtruth_eval_results.json` |
| RAGTruth majority-class baseline | 65.07% | (same file) |
| TruthfulQA MC1 / MC2 / MC0 | 34.05% / 55.05% / 44.18% | `data/processed/evaluation/truthfulqa/truthfulqa_eval_results.json` |

## 17. Important scientific caveats

- **This system does not reliably detect or correct hallucinations
  out-of-domain.** RAGTruth end-to-end accuracy is below a trivial
  majority-class baseline; the correction loop degrades ~37x more responses
  than it successfully fixes (923 degraded vs. 25 corrected, full 2,700-response
  test set). See `PROJECT_REPORT_FINAL.md` Sections 14, 17, 20 for the full,
  conservatively-worded discussion.
- The verifier's strong 96.20%/0.9585 dev metrics are a **gold-premise,
  in-domain (FEVER-distribution) evaluation** — not comparable to, and not
  predictive of, the end-to-end pipeline numbers above.
- No approved automatic truthfulness judge exists in this project for
  free-form generated text (TruthfulQA) — free-form ACCEPT/CORRECT/ABSTAIN
  rates are reported as process observations only, never as a truthfulness
  accuracy score.
- Retrieval-relevance failure (as opposed to zero-evidence failure) could
  not be directly measured on RAGTruth — no gold-relevance annotation exists
  for it, and no such metric was invented to fill that gap.
- Reproducing this project's exact reported numbers requires the exact
  seeds, model versions, and configuration recorded in
  `data/processed/final/final_reproducibility_manifest.json` (tracked) —
  minor library-version drift may shift results slightly.
