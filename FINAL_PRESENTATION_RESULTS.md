# ClaimGuard — Final Presentation Results

A ~15-slide presentation outline. Every number is read from
`data/processed/final/final_results_tables.json` and the step artifacts it
aggregates (see that file's `provenance` field). No marketing language; the
negative/mixed results are stated plainly, as required.

---

### Slide 1 — Title / Research Question

**ClaimGuard: Hallucination Detection and Self-Correction for LLMs**

> Can claim-level evidence verification, combined with iterative retrieval and
> targeted self-correction, significantly reduce factual hallucinations in
> LLM-generated responses?

---

### Slide 2 — Problem / Motivation

- LLMs generate fluent but sometimes factually unsupported text.
- Retrieval-augmented verify-and-correct is a natural, testable mitigation.
- This project implements and *honestly evaluates* a full instance of that
  strategy end-to-end, on held-out, contamination-checked data.

---

### Slide 3 — Architecture

(Figure 1: `figures/figure1_architecture.png`)

FAISS retrieval → BGE reranker → DeBERTa NLI verifier → deterministic
decision policy (ACCEPT/CORRECT/ABSTAIN) → bounded Qwen3-8B correction loop
(max 2 correction attempts, 3 generations total).

---

### Slide 4 — Datasets and Protocol

| Dataset | Role | n |
|---|---|---|
| FEVER | Verifier train/dev | 152,720 / 16,890 |
| HaluEval | Verifier train/dev supplement | (contamination-filtered) |
| RAGTruth | **Primary held-out test** | 2,700 responses |
| TruthfulQA | Secondary evaluation-only | 790 questions, 37 categories |

Gold-label isolation enforced by AST + behavioral tests: gold labels never
reach any inference call, only the output record.

---

### Slide 5 — Contamination Protocol

(Figure 2: `figures/figure2_protocol_isolation.png`)

- Found: 100 HaluEval `summarization` records overlapping 100 RAGTruth test
  sources (600 affected response records).
- Fixed by excluding those 200 pool records from verifier training —
  **RAGTruth test left byte-for-byte untouched.**
- Full cross-dataset sweep confirmed no other contamination.

---

### Slide 6 — Verifier Development

- `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, binary head
  (transplanted from the pretrained 3-class classifier, not random init).
- Dev set (n=16,890): **accuracy 96.20%, macro F1 0.9585.**
- Strong **in-domain** (FEVER-distribution) performance.

---

### Slide 7 — Retrieval and Reranking

(Figure 3: `figures/figure3_retrieval_recall.png`)

- 109,350-sentence FEVER/Wikipedia corpus, BGE embeddings, FAISS index.
- FAISS-only recall@5 = 56.9%, recall@10 = 66.2%.
- Reranker **lowers** recall@1 (28.0% → 24.5%); no measurable effect on
  end-to-end RAGTruth detection metrics (Step 21 ablation).

---

### Slide 8 — RAGTruth Results (Primary Benchmark)

(Figure 4: `figures/figure4_ragtruth_baseline_vs_final.png`)

| | Accuracy | Macro F1 |
|---|---|---|
| Attempt0 (pre-correction) | 55.9% | 0.559 |
| Final (post-correction) | 58.0% | 0.580 |
| **Majority-class baseline** | **65.1%** | — |

**The full pipeline underperforms a trivial baseline** on its primary
held-out benchmark.

---

### Slide 9 — TruthfulQA Results

- MC1 = 34.05%, MC2 = 55.05%, MC0 = 44.18% (n=790) — Qwen3-8B's own
  answer-likelihood calibration, independent of the pipeline.
- Free-form: 0% attempt0-abstain rate, 95.3% final ACCEPT rate.
- **No approved automatic truthfulness judge exists** for free-form text in
  this project — disclosed gap, not resolved with an invented metric.

---

### Slide 10 — Ablation Study

(Figure 5: `figures/figure5_ablation_comparison.png`)

- Reranked (A) vs. no-reranker (B): **identical** to 4+ decimals.
- FAISS-top-1-only (C): measurably worse (accuracy 51.0% vs. 58.7%).
- **Pool size matters more than candidate ordering.**

---

### Slide 11 — Failure Analysis

(Figure 6: `figures/figure6_verifier_confidence_by_outcome.png`)

Verifier confidence is high (0.82–0.98) **regardless of whether the decision
was correct** — it does not discriminate right from wrong. This pattern is
identical across every evidence-selection ablation tested → the failure is
upstream of retrieval/reranking choices.

---

### Slide 12 — Correction: Benefit vs. Harm

(Figure 7: `figures/figure7_correction_benefit_vs_harm.png`)

- 25 hallucinations corrected successfully.
- **923 correct answers degraded by "correction."**
- Harm outweighs benefit ~**4x** — a downstream consequence of upstream
  verifier overconfidence, not an independently broken correction mechanism.

---

### Slide 13 — Key Findings

1. In-domain verifier quality is strong; it does **not** transfer out-of-domain.
2. Reranking is not the bottleneck (H2 not supported).
3. Verifier overconfidence (H3) and zero-abstention (H4) are strongly
   supported and upstream of every other component.
4. Correction causes net harm on RAGTruth (H5 supported, correctly
   attributed to H1/H3, not the correction logic itself).
5. Retrieval-corpus/RAGTruth domain mismatch (H1) is the best-supported root
   cause of the overall negative result.

---

### Slide 14 — Limitations and Future Work

(Figure 8: `figures/figure8_failure_model.png` — evidence-supported failure
model, explicitly not a proven causal DAG)

- Ablation subset is 300/2,700 (11.1%) of RAGTruth, cross-validated but
  smaller than the full run.
- Retrieval-relevance failure is unmeasurable with current annotations.
- Future work (not implemented here): in-domain retrieval corpus, isolate H3
  from H1, learned decision policy, approved free-form truthfulness judge,
  per-evidence relevance annotation.

---

### Slide 15 — Conclusion / Reproducibility

- **Conservative main claim**: this architecture, trained on FEVER, does not
  reliably detect or correct hallucinations out-of-domain; the leading cause
  is retrieval-corpus/verifier-domain mismatch, not a single broken
  component.
- Full reproducibility manifest: `data/processed/final/final_reproducibility_manifest.json`.
- Zero git commits throughout the project (working-tree only, per governing
  instructions) — see manifest `git_state`.
