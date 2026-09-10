# ClaimGuard — Artifact Inclusion / Exclusion Reference

This document explains exactly what is tracked in this Git repository and
what is intentionally excluded (via `.gitignore`), so a researcher cloning
this repo understands what to expect on disk and how to regenerate anything
missing. Nothing described here as "excluded" was deleted from the original
research machine — exclusion is a Git-tracking decision, not a
data-destruction one.

## A. Included in Git

- **Source code**: `src/claimguard/` (generation, retrieval, reranking,
  verification, decision, correction, evaluation, datasets, config, utils).
- **Tests**: `tests/` (the full test suite).
- **Scripts**: `scripts/` (dataset acquisition/inspection, corpus/index
  construction, training, evaluation, ablation, and Step 22 report-generation
  scripts).
- **Configs**: `configs/` (model identifiers, training/eval hyperparameters
  — small YAML files, no secrets).
- **Documentation**: `README.md`, `PROJECT_PLAN.md`, `PROJECT_REPORT.md`,
  `PROJECT_REPORT_FINAL.md`, `RESEARCH.md`, `CLAIMGUARD_MODEL_SELECTION.md`,
  `FINAL_PRESENTATION_RESULTS.md`, `REPRODUCIBILITY.md` (this file's
  companion), `ARTIFACTS.md` (this file).
- **Presentation**: `presentation/` (guide/advisor presentation, its build
  scripts, and its speaker-notes Markdown — no large embedded data).
- **Small, curated research result summaries**:
  - `data/processed/dataset_manifest.json`, `contamination_report.json`
  - `data/processed/retrieval/{corpus_manifest.json, reranker_manifest.json,
    retrieval_eval_results.json, reranker_eval_results.json}`
  - `data/processed/integration/{integration_eval_results.json,
    integration_manifest.json, calibration_results.json,
    decision_policy_results.json}`
  - `data/processed/evaluation/{ragtruth,truthfulqa,ablations}/` — the
    aggregated `*_results.json`, `*_manifest.json`, `*_error_analysis.json`,
    and `ablation_comparison.csv` files (NOT the large per-response
    `*_response_results.jsonl` dumps — see Section B)
  - `data/processed/final/` — the complete Step 22 deliverable package
    (`final_results_tables.json`, `final_reproducibility_manifest.json`,
    `tables_csv/*.csv`, `figures/*.png`)
  - `data/processed/{correction,generation}/*_smoke_results.json`
  - `experiments/verifier_binary_final/final/` — small config/metric/
    reproducibility metadata only (NOT the model weights — see Section B)

## B. Intentionally excluded from Git

| Artifact | Why excluded | How to regenerate |
|---|---|---|
| `data/raw/` (all four datasets) | Raw third-party dataset content — not ours to redistribute | `REPRODUCIBILITY.md` §5-10 (acquire from authoritative source + this project's `inspect_*.py`/`build_*_sample.py` scripts) |
| `data/processed/fever/wiki_pages_index.sqlite` | Large generated retrieval index (~9.1GB) built from the acquired Wikipedia `wiki_pages` dump | `scripts/acquire_inspect_wiki_pages.py` + `scripts/build_wiki_pages_index.py` |
| `data/processed/fever/wiki_resolution_report.json` | Full per-page resolution dump (19MB), not a summary | Regenerated alongside the index build above |
| `data/processed/*/sample/*.jsonl` | Raw dataset excerpts extracted for manual inspection | `scripts/build_{fever,halueval,ragtruth,truthfulqa}_sample.py` |
| `data/processed/integration/integration_examples.jsonl` | Generated pipeline execution examples (candidate answers + evidence text), not a curated summary | `scripts/evaluate_integration.py` |
| `data/processed/retrieval/{corpus.jsonl, embeddings.npy, faiss_index.bin, index_metadata.json, embedding_corpus_ids.json}` | Large generated retrieval artifacts (corpus dump ~31MB, embeddings + FAISS index ~430MB each, metadata ~31MB) | `REPRODUCIBILITY.md` §11 (`build_retrieval_corpus.py` → `embed_retrieval_corpus.py` → `build_faiss_index.py`) |
| `data/processed/verifier_pool/{train.jsonl, dev.jsonl, excluded_contamination.jsonl}` | Generated verifier training/dev pool (~280MB total) | `scripts/build_verifier_pool.py` (after dataset manifest + contamination detection) |
| `data/processed/evaluation/{ragtruth,truthfulqa,ablations}/*_response_results.jsonl` | Large per-response raw output dumps (1.5-9.5MB each); the aggregated `*_results.json`/`*_error_analysis.json` files already contain every reported breakdown | `scripts/evaluate_ragtruth.py`, `scripts/evaluate_truthfulqa.py`, `scripts/run_ablations.py` |
| `experiments/verifier_binary_final/final/{model.safetensors, tokenizer.json, training_args.bin}` | Model weights and training-state binaries (~830MB+) | `src/claimguard/verifier/train_binary.py` per `training_config.yaml` (tracked) |
| `experiments/verifier_binary_final/checkpoint-4500/`, `checkpoint-4773/` | Intermediate training checkpoints (weights + optimizer/scheduler state, ~2.5GB each) | Regenerated as a byproduct of the training run above |
| `experiments/verifier_baseline/`, `verifier_baseline_smoke_test/`, `verifier_binary_final_smoke_test/` (entire trees, ~5.7GB each) | Superseded baseline/smoke-test checkpoints, not the model this project's results are based on | Not needed to reproduce reported results; smoke-test scripts (`scripts/*_smoke_test.py`) regenerate their small result summaries if re-run |
| Hugging Face / pip / model caches | Machine-local, multi-GB, trivially re-downloaded | Standard `transformers`/`huggingface_hub` caching on first model load |
| `*.log`, temporary/runtime files | Machine-specific, not research artifacts | N/A |

## C. Notably excluded — explicit callouts

- **`data/processed/fever/wiki_pages_index.sqlite`** is intentionally
  excluded because it is a large (~9.1GB) generated retrieval index, not a
  research result — it is fully rebuildable from the acquired FEVER wiki
  dump via `scripts/build_wiki_pages_index.py`.
- **`experiments/`** (the directory as a whole, ~23GB) contains local
  generated experiment artifacts — trained model checkpoints, optimizer
  state, and superseded baseline runs — and is intentionally excluded from
  version control except for the small, curated metric/config/reproducibility
  files under `experiments/verifier_binary_final/final/` listed in Section A.

## D. Not claimed

This repository does not claim to include everything needed to byte-for-byte
reproduce every number in `PROJECT_REPORT_FINAL.md` with zero additional
compute — reproducing the full pipeline (dataset acquisition, retrieval
index construction, verifier training, and the multi-hour RAGTruth/TruthfulQA/
ablation evaluations) requires GPU time and the raw datasets, as documented
in `REPRODUCIBILITY.md`. What this repository does guarantee is that every
number reported in the project's documentation is traceable to a small,
tracked, machine-readable artifact in `data/processed/` or
`experiments/verifier_binary_final/final/`.
