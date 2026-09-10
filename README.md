# ClaimGuard

Hallucination Detection and Self-Correction for LLMs.

## Research Objective

Investigate whether claim-level evidence verification, combined with iterative
retrieval and targeted self-correction, can significantly reduce factual
hallucinations in LLM-generated responses.

## Research Question

> Can claim-level evidence verification combined with iterative retrieval and
> targeted self-correction significantly reduce factual hallucinations in
> LLM-generated responses?

See `RESEARCH.md` for the full problem statement, hypothesis, methodology,
and evaluation plan. **See `PROJECT_REPORT_FINAL.md` for the final,
standalone scientific summary (Steps 1-22 complete) and
`FINAL_PRESENTATION_RESULTS.md` for a presentation-ready results summary.**
`PROJECT_REPORT.md` remains the full incremental step-by-step log. **See
`REPRODUCIBILITY.md` for how to reproduce this project from source (datasets,
environment, training, evaluation) and `ARTIFACTS.md` for what large
generated artifacts are intentionally excluded from this repository and how
to regenerate them.**

## High-Level Architecture

```
 user prompt
     │
     ▼
 [Generator LLM] ──► draft response
     │
     ▼
 [Claim extraction]  ──► atomic factual claims
     │
     ▼
 [Retrieval: BM25 + FAISS]  ──► candidate evidence
     │
     ▼
 [Reranking]  ──► top evidence per claim
     │
     ▼
 [Verification: NLI]  ──► SUPPORTS / REFUTES / NOT ENOUGH INFO
     │
     ▼
 [Targeted self-correction]  ──► revised response (loop, bounded iterations)
     │
     ▼
 final response
```

**All stages above are implemented and evaluated (Steps 1-21).** The actual
final implementation uses FAISS + BGE-reranker (not BM25) for retrieval, and
a deterministic (non-learned) decision policy between verification and
correction — see `PROJECT_REPORT_FINAL.md` §5 for the exact final
architecture and `data/processed/final/figures/figure1_architecture.png` for
a diagram.

## Current Model Stack

See `CLAIMGUARD_MODEL_SELECTION.md` for full rationale. Summary:

| Role | Model |
|---|---|
| Generator (primary) | `Qwen/Qwen3-8B` |
| Generator (secondary, robustness) | `meta-llama/Llama-3.1-8B-Instruct` |
| Embedding | `BAAI/bge-large-en-v1.5` |
| Reranker | `BAAI/bge-reranker-large` |
| Verifier (NLI) | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Retriever | FAISS (BGE-large-en-v1.5 embeddings) over a FEVER/Wikipedia corpus |
| Correction | Generator itself, prompted with claim + verdict + evidence |

Model identifiers are configured in `configs/models.yaml`, not hard-coded.
(FActScore/SelfCheckGPT-WikiBio/FEVEROUS/BM25-hybrid retrieval, listed as
candidates in early planning, were not used in the final implementation —
FAISS + BGE-reranker over FEVER/Wikipedia was used throughout instead. This
is a historical planning note, not a correction to earlier reports.)

## Dataset Plan (final — see `PROJECT_REPORT_FINAL.md` §6-7)

| Dataset | Role | Final size |
|---|---|---|
| FEVER | Verifier training + dev | 152,720 train / 16,890 dev (post-contamination-exclusion, post-neutral-class decision) |
| HaluEval | Verifier training supplement (qa/dialogue/summarization) | contamination-filtered: 200 records excluded (see contamination protocol) |
| RAGTruth | **Primary held-out end-to-end test** (never in training) | 2,700 test responses evaluated |
| TruthfulQA | Evaluation-only (adversarial stress test) | 790 questions, 37 categories |

All datasets were acquired and used (Steps 6-8, 10-12, 19-20).
`FActScore`/`SelfCheckGPT WikiBio`/`FEVEROUS` (listed above as originally
planned candidates) were **not used** — the project's held-out evaluation
relied on RAGTruth and TruthfulQA only, per Steps 19-20's scope. A
contamination sweep (Step 8) found and excluded 200 HaluEval records
overlapping RAGTruth test sources; RAGTruth test itself was never modified.
FEVER's neutral (NOT ENOUGH INFO) class was investigated (Step 11) and found
to have no legitimate usable source for the verifier's decision boundary in
this design, so the final verifier is strictly binary
(entailment/contradiction).

## Hardware

- NVIDIA RTX PRO 5000 Blackwell, ~48GB VRAM

## Project Structure

```text
ClaimGuard/
├── README.md
├── PROJECT_PLAN.md
├── RESEARCH.md
├── CLAIMGUARD_MODEL_SELECTION.md
├── PROJECT_REPORT.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── PROJECT_REPORT_FINAL.md   # final standalone scientific summary (Step 22)
├── FINAL_PRESENTATION_RESULTS.md  # presentation-ready results summary
├── configs/            # default.yaml, models.yaml, experiments.yaml, etc.
├── data/
│   ├── raw/             # acquired FEVER, HaluEval, RAGTruth, TruthfulQA
│   └── processed/       # dataset_manifest.json, retrieval/, integration/,
│                        # evaluation/{ragtruth,truthfulqa,ablations}/,
│                        # final/ (Step 22: final_results_tables.json,
│                        # final_reproducibility_manifest.json, figures/)
├── src/claimguard/      # generation, retrieval, reranking, verification/
│                        # (verifier), decision (calibration/policy),
│                        # correction, evaluation, datasets, config, utils
├── scripts/             # dataset build, training, eval, and Step 22
│                        # report-generation scripts
├── experiments/         # verifier_binary_final/final/ (trained checkpoint
│                        # + reproducibility.json + dev metrics)
├── notebooks/
├── tests/                # full suite, see "Testing" below
└── demo/
```

## Environment Activation

```bash
ssh <GPU_USERNAME>@<REMOTE_SERVER>
source ~/miniconda3/etc/profile.d/conda.sh
conda activate claimguard
cd ~/RJ/ClaimGuard
```

## Results (final — Steps 1-22 complete)

Full numbers: `data/processed/final/final_results_tables.json` (7 tables),
figures: `data/processed/final/figures/` (8 figures), narrative:
`PROJECT_REPORT_FINAL.md`. Headline results:

- Verifier (in-domain, FEVER dev, n=16,890): **96.20% accuracy, 0.9585 macro F1.**
- RAGTruth end-to-end (primary held-out benchmark, n=2,700): final accuracy
  **58.04%**, macro F1 **0.5798** — **below** the 65.07% majority-class
  baseline.
- Correction outcome: 25 hallucinations corrected successfully vs. **923
  correct answers degraded** by the correction loop.
- TruthfulQA (n=790, evaluation-only): MC1 34.05%, MC2 55.05%, MC0 44.18%.

**Main conclusion (stated conservatively, see `PROJECT_REPORT_FINAL.md` §20
for the full statement and explicit list of claims NOT being made):** this
architecture, verifier-trained on FEVER, does not reliably detect or correct
hallucinations on out-of-domain text (RAGTruth). The best-supported cause is
a domain mismatch between the FEVER/Wikipedia retrieval corpus and RAGTruth's
actual source text, compounded by a verifier discrimination (not
calibration) failure on out-of-domain evidence — not a defect in any single
component (reranker, correction logic) in isolation. This is a negative/mixed
result, preserved in full rather than downplayed.

## Reproduction

```bash
ssh <GPU_USERNAME>@<REMOTE_SERVER> && source ~/miniconda3/etc/profile.d/conda.sh && \
  conda activate claimguard && cd ~/RJ/ClaimGuard
python -m unittest discover -s tests               # run the test suite
python scripts/generate_final_results_tables.py    # rebuild Table 1-7 JSON/CSVs
python scripts/build_final_reproducibility_manifest.py
python scripts/generate_final_figures.py            # rebuild Figures 1-8
```

For rebuilding the full pipeline from raw datasets (acquisition, verifier
training, retrieval index, end-to-end evaluation) rather than just
regenerating the Step 22 report from already-saved results, see
`REPRODUCIBILITY.md` and `scripts/run_full_pipeline.sh --list`.

See `data/processed/final/final_reproducibility_manifest.json` for the exact
environment, seeds, model identifiers, and config/checkpoint-metadata hashes
this project's results were produced with. Re-running the multi-hour
RAGTruth/TruthfulQA/ablation GPU evaluations themselves is not required to
reproduce or verify the final report — `tests/test_final_report.py` checks
the final tables/manifest against the already-saved source artifacts instead.

## Development Status

**Steps 1-22 complete.** All pipeline stages (retrieval, reranking,
verification, decision policy, generation, correction loop) are implemented
and evaluated end-to-end on RAGTruth (primary) and TruthfulQA (secondary).
Step 22 (this update) is the final documentation/reporting stage — no new
experiments, model changes, or tuning occurred in it. See `PROJECT_PLAN.md`
for the original staged plan and `PROJECT_REPORT.md` for the full
step-by-step incremental log (Steps 0-21) that produced these results.
