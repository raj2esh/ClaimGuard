# ClaimGuard — Staged Development Plan

Status legend: COMPLETE / CURRENT / PENDING. Only steps actually finished are
marked COMPLETE — no future step is marked complete ahead of time.

| Step | Description | Status |
|---|---|---|
| 1 | Environment audit (local + remote) | COMPLETE |
| 2 | Environment setup (`claimguard` conda env, packages) | COMPLETE |
| 3 | Model / dataset selection | COMPLETE |
| 4 | Project structure and configuration skeleton | CURRENT |
| 5 | Model smoke tests (load configs only / minimal load checks) | PENDING |
| 6 | Dataset acquisition and preprocessing | PENDING |
| 7 | Baseline LLM (generator, no pipeline) | PENDING |
| 8 | Claim decomposition | PENDING |
| 9 | Retrieval (BM25 + FAISS) | PENDING |
| 10 | Reranking | PENDING |
| 11 | Verification (NLI) | PENDING |
| 12 | Correction | PENDING |
| 13 | End-to-end pipeline | PENDING |
| 14 | Baseline experiments | PENDING |
| 15 | Ablation experiments | PENDING |
| 16 | Error analysis | PENDING |
| 17 | Final evaluation | PENDING |
| 18 | Demo | PENDING |

## Notes

- Each step is performed only when explicitly requested, and each is
  documented in `PROJECT_REPORT.md` (what was done and why) once complete.
- Step 4 (this step) creates structure and configuration only — no models are
  downloaded, no datasets are downloaded, no training or GPU inference runs.
- Experiment placeholders for Steps 14–15 already exist in
  `configs/experiments.yaml`, all marked `status: not_run`.
