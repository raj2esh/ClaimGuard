# ClaimGuard Guide Presentation

A research-guide/advisor presentation summarizing the completed ClaimGuard
project (Steps 1-22). Generated as a documentation artifact — no new
experiments, model calls, or tuning were performed to build it.

## Files

- `ClaimGuard_Guide_Presentation.pptx` — the presentation, 25 slides (19 core
  content slides + 1 title slide + 1 closing slide + a 5-slide Q&A appendix).
- `ClaimGuard_Guide_Presentation_Notes.md` — speaker notes for every slide,
  extracted directly from the notes embedded in the `.pptx` (the two files
  cannot drift out of sync, since the Markdown is derived from the `.pptx`,
  not authored separately).
- `README.md` — this file.

## How it was generated

1. `generate_presentation_assets.py` — builds the diagram PNGs (architecture,
   contamination protocol, decision/correction flow, domain-mismatch and
   failure-model diagrams, etc.) with `matplotlib`. Pure documentation
   rendering, no model calls.
2. `build_presentation.py` — assembles the `.pptx` with `python-pptx`: title/
   bullet text, tables, five native (editable) bar charts, the diagram
   images, and per-slide speaker notes.
3. `extract_notes_md.py` — re-reads the saved `.pptx` and writes
   `ClaimGuard_Guide_Presentation_Notes.md` from its embedded notes, so the
   Markdown notes file is always in sync with the actual deck.

All three scripts live alongside this README for reproducibility and live in
version control as part of the presentation package.

## Source artifacts every number traces back to

Every metric in the deck is copied verbatim from already-validated Step 9-22
artifacts — nothing was invented, recomputed with different logic, or tuned
for the presentation:

- `experiments/verifier_binary_final/final/dev_metrics.json`,
  `reproducibility.json` — verifier development results (Slide 8).
- `data/processed/retrieval/reranker_eval_results.json` — FAISS/reranker
  recall@K (Slide 9).
- `data/processed/integration/integration_eval_results.json`,
  `calibration_results.json` — integrated pipeline metrics and gold/non-gold
  confidence (Slide 10).
- `data/processed/evaluation/ragtruth/ragtruth_eval_results.json` — RAGTruth
  end-to-end results and correction outcomes (Slide 12).
- `data/processed/evaluation/truthfulqa/truthfulqa_eval_results.json` —
  TruthfulQA MC scores and free-form decision rates (Slide 14).
- `data/processed/evaluation/ablations/ablation_results.json`,
  `ablation_error_analysis.json` — controlled ablation and failure-source
  numbers (Slides 15-16).
- `data/processed/final/final_results_tables.json`,
  `final_reproducibility_manifest.json` — the Step 22 aggregated tables and
  reproducibility manifest used to cross-check every figure above.
- `PROJECT_REPORT.md` (Steps 8, 11, 15, 16, 18, 21 sections),
  `PROJECT_REPORT_FINAL.md`, `RESEARCH.md`, `CLAIMGUARD_MODEL_SELECTION.md`,
  `PROJECT_PLAN.md` — narrative, methodology, and Q&A-appendix source
  material (contamination protocol counts, model-selection rationale,
  calibration ECE/Brier figures, correction-loop design).

## How to update the results if source artifacts change

This presentation should only ever be regenerated from already-validated
artifacts — never used to introduce new numbers ahead of the underlying
experiments being run and documented in `PROJECT_REPORT.md`.

1. Update the relevant hard-coded value(s) in `build_presentation.py` (search
   for the metric by name — each chart/table call lists its values inline)
   to match the new source artifact.
2. Re-run `python3 generate_presentation_assets.py` if any diagram's static
   labels changed (e.g. updated failure-rate percentages baked into
   `slide16_failure_model.png`).
3. Re-run `python3 build_presentation.py`.
4. Re-run `python3 extract_notes_md.py` to refresh the notes Markdown.
5. Re-render to PDF/PNG (`libreoffice --headless --convert-to pdf ...` then
   `pdftoppm`) and visually re-check any slide whose numbers changed.

## Presentation duration and audience

**Target audience:** a research guide/advisor project-review or thesis
evaluation meeting.

**Target duration:** 15-20 minutes for the 19 core content slides (Slides
2-19, plus the title and closing slides) at roughly 30-60 seconds per slide,
per the embedded speaker notes. The 5-slide Q&A appendix is reference
material for the Q&A portion of the meeting and is not part of the timed
walkthrough.

**Recommended pacing:**
- Slides 1-5 (title, motivation, problem, gap, architecture): ~4 minutes
- Slides 6-11 (methodology: datasets, contamination, verifier, retrieval,
  integration, decision/correction): ~6 minutes
- Slides 12-14 (RAGTruth and TruthfulQA results): ~4 minutes
- Slides 15-17 (ablations, failure analysis, key findings): ~4 minutes
- Slides 18-20 (limitations, future work, reproducibility, closing): ~2-3
  minutes
