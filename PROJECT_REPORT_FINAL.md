# ClaimGuard — Final Research Report

Steps 1–22, 2026-08-31 to 2026-09-10. This document is the final, standalone
scientific summary of the ClaimGuard project. It does not replace
`PROJECT_REPORT.md` (the full, step-by-step incremental log — the authoritative
source for procedural detail) or `RESEARCH.md` (the original problem
statement/hypotheses); it synthesizes them into one report reflecting the
final, frozen state of the system. All numbers here are read from, and
programmatically checked against, the saved artifacts in
`data/processed/final/` and the underlying step artifacts they were built from
(see `tests/test_final_report.py`).

## 1. Abstract

ClaimGuard is a retrieval-augmented hallucination-detection and
self-correction pipeline built around a fine-tuned NLI verifier
(`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, binary
entailment/contradiction head) layered on FAISS + BGE-reranker retrieval over
a FEVER/Wikipedia evidence corpus, with a Qwen3-8B generator and a bounded
evidence-grounded correction loop. The verifier itself is strong in-domain
(96.2% accuracy, 0.959 macro F1 on a 16,890-example FEVER-derived dev set).
However, end-to-end on RAGTruth's held-out test split (2,700 responses, the
project's primary out-of-domain benchmark), the full pipeline's hallucination
detection **underperforms the majority-class baseline** (58.0% final accuracy
vs. 65.1% majority-class), and the correction loop harms roughly 4x more
responses than it fixes (923 correct answers degraded vs. 25 hallucinations
successfully corrected, out of 2,700). Controlled ablations (Step 21) isolate
the dominant cause: the verifier is confidently wrong about as often as it is
confidently right (mean confidence 0.82–0.98 across all four
correct/incorrect × accept/flag outcome groups), a pattern that persists
identically whether or not the reranker is used or bypassed — implicating a
retrieval-corpus domain mismatch (FEVER/Wikipedia vs. RAGTruth's
news/business/QA text) combined with a verifier discrimination failure on
out-of-domain evidence, not a defect in the reranker or the correction logic
itself. TruthfulQA multiple-choice results (MC1 34.05%, MC2 55.05%, MC0
44.18%, n=790) further evidence that Qwen3-8B alone is not a reliably
truthful oracle. **The main, conservatively-stated finding of this project is
negative-to-mixed**: this particular claim-level verification architecture,
trained on FEVER, does not generalize to reliably detect or correct
hallucinations on out-of-domain text, and the dominant identified cause is
retrieval-corpus/verifier-domain mismatch rather than any single component
being "broken." This report preserves that negative result in full, alongside
what worked (in-domain verifier quality, correct/negative retrieval
diagnostics, well-isolated evaluation protocol) and does not overstate the
system's capability.

## 2. Motivation

LLM-generated text frequently contains hallucinated claims not supported by
any evidence, which limits deployment in factuality-sensitive settings.
Retrieval-augmented verification (retrieve evidence, verify the claim against
it, correct if contradicted) is a natural mitigation strategy. This project
investigates a concrete, fully-implemented instance of that strategy rather
than a purely theoretical one, and evaluates it honestly — including when it
fails — against held-out, contamination-checked data.

## 3. Research question

> Can claim-level evidence verification, combined with iterative retrieval and
> targeted self-correction, significantly reduce factual hallucinations in
> LLM-generated responses?

(Unchanged from `RESEARCH.md`'s original framing — see that file for the full
original hypothesis set H1–H5, which Step 21 tested directly; see Section 17
below for how each hypothesis resolved.)

## 4. Related methodological context

FEVER (claim verification against Wikipedia evidence) is the standard source
of NLI-style verifier training data; RAGTruth and TruthfulQA are established
hallucination/truthfulness benchmarks. This project's contribution is not a
new model or a new benchmark, but an end-to-end, contamination-checked,
gold-label-isolated integration of an off-the-shelf strong verifier with
retrieval, reranking, a deterministic decision policy, and a bounded
correction loop — evaluated honestly on out-of-domain data rather than only
on the training-adjacent FEVER distribution.

## 5. System architecture

```
query/claim
   -> FAISS retrieval (BAAI/bge-large-en-v1.5 embeddings, 109,350-sentence
      FEVER/Wikipedia corpus)
   -> BGE reranker (BAAI/bge-reranker-large) over top FAISS candidates
   -> Verifier (DeBERTa-v3-large NLI, binary entailment/contradiction head)
   -> Decision policy (deterministic thresholds -> ACCEPT / CORRECT / ABSTAIN)
   -> [if CORRECT] Qwen3-8B evidence-grounded regeneration, re-verify,
      bounded to max_correction_attempts=2 (3 generations total)
   -> final output (ACCEPT the candidate, or terminal ABSTAIN if budget
      exhausted while still CORRECT)
```

See Figure 1 (`data/processed/final/figures/figure1_architecture.png`). Every
stage's model identifier is configured in `configs/models.yaml`, never
hard-coded. All four models (retriever embedder, reranker, verifier,
generator) are loaded once by the calling script and passed into
`run_correction_loop()` already-loaded — no stage loads its own model copy.

## 6. Dataset construction

| Dataset | Role | Size |
|---|---|---|
| FEVER | Verifier training/dev (entailment/contradiction) | 145,449 raw train claims; 109,810 usable after premise resolution; final verifier train/dev split 152,720 / 16,890 |
| HaluEval | Verifier training/dev supplement (qa, dialogue, summarization) | contributes to the same train/dev pool as FEVER, contamination-filtered (Section 7) |
| RAGTruth | **Primary held-out end-to-end test benchmark** (never used in training) | 2,700 test responses evaluated |
| TruthfulQA | **Secondary evaluation-only** adversarial stress test (no train/dev role at all) | 790 questions, 37 categories |

FEVER's neutral (NOT ENOUGH INFO) class was investigated (Step 11) but found
to have no legitimate source usable for the verifier's decision boundary in
this design, so the verifier is trained strictly binary
(entailment/contradiction); `neutral_class_available_from_fever = 0` in the
final dataset manifest, and this is stated rather than silently worked
around. FEVER's evidence sentences were resolved against acquired
`wiki_pages` (Step 10) — no premise text was invented.

## 7. Contamination protocol

An exact-match, field-aware sweep (Step 8, independently re-derived and
verified stricter than Step 7's original check) found **100 unique RAGTruth
test source documents overlapping with 100 unique HaluEval `summarization`
raw records** (expanding to 600 affected RAGTruth response records, since
each RAGTruth source has exactly 6 model responses). Per the standing
decision rule — the held-out evaluation benchmark is never modified — the
100 HaluEval source records (200 pool records, both contrastive-pair halves)
were **excluded from the verifier training/dev pool**; RAGTruth test itself
was left byte-for-byte untouched. A broader sweep confirmed no other
contamination: FEVER train vs. RAGTruth test/TruthfulQA = 0 overlap;
HaluEval qa vs. TruthfulQA = 70 benign generic-token matches (documented, not
treated as leakage); HaluEval dialogue vs. TruthfulQA = 1 benign match;
HaluEval general = 0 both ways. Both `RoleViolationError` guards
(`build_training_pool(["ragtruth_test"])` / `(["truthfulqa"])`) were
re-verified to still raise after the pool rebuild. This protocol and its
exact counts are unchanged since Step 8 and were not touched in Step 22.

## 8. Verifier development

`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`'s pretrained
3-class MNLI head was reduced to a 2-class (entailment/contradiction) head by
**transplanting** (not randomly reinitializing) the original entailment and
contradiction classifier rows, with the encoder and pooler reused unchanged
(verified bit-identical). Trained 1 epoch (152,720 train / 16,890 dev
records, seed 42, lr 2e-5, effective batch size 32, bf16, linear schedule,
warmup ratio 0.06) with `macro_f1` as the checkpoint-selection metric.

**Final dev metrics**: accuracy **0.9620**, macro F1 **0.9585**, weighted F1
**0.9621**; per-class F1 — entailment 0.9705, contradiction 0.9465; confusion
matrix `[[10570, 431], [211, 5678]]` (rows/cols = entailment, contradiction);
peak GPU memory 14.36GB; 435,063,810 total parameters. This is strong
in-domain (FEVER-distribution) performance; Section 16/17 show it does not
transfer to out-of-domain text.

## 9. Retrieval system

109,350-sentence FEVER/Wikipedia corpus (from 12,549 wiki pages), embedded
with `BAAI/bge-large-en-v1.5` (1024-dim, float16) into a FAISS index.
FAISS-only recall on a 2,000-claim held-out eval set: recall@1 **0.280**,
recall@5 **0.569**, recall@10 **0.662**, recall@20 **0.733**. This retrieval
corpus is FEVER/Wikipedia-derived throughout the project — it was never
rebuilt or augmented for RAGTruth's/TruthfulQA's actual (news, business
listing, QA-passage) source domains, which Section 17 identifies as the
leading contributor to end-to-end failure.

## 10. Reranking

Adding `BAAI/bge-reranker-large` on top of FAISS **lowers** recall@1 (0.280
→ 0.2445) while modestly improving recall@5/@10 (0.569→0.591, 0.662→0.701);
recall@20 is unchanged (0.733 both). This recall@1 regression was first
found in Step 14 and reconfirmed unchanged in Steps 15, 16, and 21 — it is a
stable, real property of this reranker/corpus/query-distribution combination,
not measurement noise. Step 21's controlled ablation further showed the
reranker changes WHICH evidence is selected for 100% of end-to-end RAGTruth
responses but produces **no measurable change** in final detection accuracy,
macro F1, precision, or recall (identical to 4+ decimal places between
Conditions A and B) — reranking is not the bottleneck limiting end-to-end
performance, and is also not a fix for it.

## 11. Decision policy

A deterministic (non-learned) policy (`src/claimguard/decision/policy.py`)
maps verifier entailment/contradiction probabilities to
ACCEPT/CORRECT/ABSTAIN via fixed thresholds, chosen from Step 16's
calibration analysis — not tuned against RAGTruth or TruthfulQA. On
gold-evidence FEVER candidates, the verifier is well-calibrated (**ECE =
0.0092, Brier = 0.0178**) — when ground truth exists, confidence tracks
accuracy closely. The practical failure is a **discrimination** failure, not
a calibration failure: non-gold evidence topically/lexically similar to a
SUPPORTS claim is scored with confidence indistinguishable from genuine gold
evidence (median 0.9940 vs. 0.9987), across the entire threshold sweep tested
(0.50–0.95) — precision stays in a narrow 0.14–0.17 band regardless of
threshold. Under this policy, `attempt0_abstain_rate = 0.0` in every
end-to-end condition tested (RAGTruth full run, all three Step 21 ablation
conditions) — the policy essentially never abstains in practice, because the
verifier's own confidence distribution is extreme/bimodal rather than
graduated near the threshold boundary.

## 12. Generator

Qwen3-8B (bf16), `enable_thinking=False` throughout (no hidden reasoning
trace generated or stored, for both auditability and latency). Used both to
produce the initial candidate answer and, unchanged, to perform
evidence-grounded correction (Section 13).

## 13. Correction loop

`run_correction_loop()`: a plain bounded `while True` loop with explicit
`return` at every termination point (never an unbounded loop, never a hidden
retry path). `max_correction_attempts = 2` (a fresh, explicitly-labeled
engineering default, not tuned against any evaluation data) bounds the loop
to at most 3 total generations. Every correction attempt re-runs the FULL
retrieve → rerank → verify sequence on the corrected candidate text (never
cached, never the original query) — verified by dedicated tests. A
budget-exhausted CORRECT is **always** converted to a terminal ABSTAIN, never
silently treated as ACCEPT. The correction prompt supplies only the actual
retrieved evidence text (never fabricated, never gold) and instructs the
generator to preserve supported content and correct only unsupported
content.

## 14. RAGTruth evaluation

Full 2,700-response held-out test split, three source LLMs (gpt-3.5-turbo,
gpt-4, llama-2-13b/7b), three task types (Data2txt, QA, Summary). Gold
hallucination prevalence **34.93%**. Attempt0 (pre-correction) detection:
accuracy **55.89%**, macro F1 **0.5589**. Final (post-correction) outcome:
accuracy **58.04%**, macro F1 **0.5798** — both **below** the majority-class
baseline of **65.07%** (always predicting "not hallucinated"). Decision
rates: `attempt0_correct_rate = 65.3%` (the policy flags the majority of
responses for correction), `attempt0_abstain_rate = 0.0%`,
`correction_success_rate_among_attempted = 6.1%`. Correction-outcome
breakdown (n=2,700): 733 hallucinated-still-flagged, 834
not-hallucinated-preserved, **25 hallucinated-corrected-successfully**, **923
not-hallucinated-degraded**, 185 hallucinated-missed — correction harms
roughly **4x** more responses than it helps. Performance varies substantially
by task type (Data2txt accuracy 63.7%, QA 69.3%, Summary 41.1% final) and by
source model (llama-2-13b 66.4% vs. gpt-4 47.1% final accuracy) — these
breakdowns are preserved in `final_results_tables.json` Table 4 rather than
averaged away.

## 15. TruthfulQA evaluation

790 questions, 37 categories, evaluation-only (no train/dev role).
Multiple-choice: **MC1 = 34.05%**, **MC2 = 55.05%**, **MC0 = 44.18%** — these
measure Qwen3-8B's own answer-likelihood calibration directly and do not
involve the retrieval/verification pipeline. Free-form generation through
the full pipeline: attempt0 abstain rate 0.0% (same zero-abstain pattern as
RAGTruth), final ACCEPT rate 95.3%. **No approved automatic truthfulness
judge exists in this project for free-form generated text** (no human
raters, no GPT-judge available, and inventing a semantic-similarity heuristic
was explicitly out of scope per the governing instructions) — this is a
disclosed protocol gap, not silently resolved, and the free-form ACCEPT-rate
figure is reported as a process observation only, not interpreted as evidence
of higher truthfulness than RAGTruth.

## 16. Ablation study

Step 21 tested three evidence-selection strategies (A: reranked top-5, B: no
reranker/FAISS top-5, C: FAISS top-1 only) on a 300-response deterministic
RAGTruth subset, cross-validated 100%-matched (n=300) against Step 19's full
run for Condition A. Results: A accuracy 0.5867/macro F1 0.5813, B identical
to A to 4+ decimals, **C measurably worse** (accuracy 0.5100/macro F1
0.4874) — candidate **pool size** (5 vs. 1) matters more than candidate
**ordering** (reranked vs. not). Conditions D (no-correction) and E
(verify-only) were reused directly from Step 19's full 2,700-response data
with zero new inference (D accuracy 0.5589 = identical to attempt0 metrics,
by construction; E accuracy 0.4874, macro F1 0.4811 — verify-only performs
worse than the full pipeline, showing the correction step does contribute
some net signal despite Section 14's harm/benefit imbalance).

## 17. Failure analysis

See Figure 8 (`data/processed/final/figures/figure8_failure_model.png`), an
**evidence-supported failure model — explicitly not a proven causal DAG**.
Verifier confidence by outcome group (full 2,700, response-level): correctly
accepted 0.833 mean, correctly flagged 0.982 mean, **wrongly** accepted
(missed hallucination) 0.821 mean, **wrongly** flagged (false positive) 0.971
mean — confidence does not discriminate correct from incorrect decisions.
This mirrors Step 16's FEVER-domain gold-evidence overconfidence pattern
(Section 11), confirmed here to persist identically regardless of
evidence-selection strategy (Section 16). Failure-source separation (n=300
subset): retrieval-zero-evidence failures = 0/300 (not the bottleneck);
retrieval-relevance failure = **not identifiable** (RAGTruth has no
per-evidence gold-relevance annotation, and inventing a semantic-relevance
metric was out of scope — reported honestly as unmeasured rather than
guessed); verifier-overconfident-and-wrong (confidence ≥0.90) = 115/300
(38.3%); decision-policy zero-abstention confirmed structurally (Section 11);
correction-caused-harm = 101/300 (33.7%, consistent with the full-population
34.2% rate). **Hypothesis resolution against `RESEARCH.md`'s original H1–H5**:
H1 (retrieval-corpus domain mismatch) is the best-supported root cause; H2
(reranker is the bottleneck) is **not supported** — reranking changes
selection but not outcomes; H3 (verifier overconfidence) is strongly
supported and persists upstream of every ablation tested; H4 (decision
policy never abstains) is strongly supported; H5 (correction causes net
harm) is supported, but attributed correctly as a **downstream consequence**
of H1/H3, not an independently broken correction mechanism.

## 18. Limitations

- Conditions A/B/C (Step 21) were measured on an 11.1% (300/2,700)
  deterministic subset, not the full test set; Section 8's cross-validation
  confirms the underlying architecture matches Step 19's full-population
  numbers exactly on matching response_ids, but absolute subset figures carry
  more sampling variance than the full-2,700 RAGTruth figures.
- Retrieval-relevance failure (as opposed to zero-evidence failure) could not
  be measured directly — no valid ground-truth-free metric exists without
  fabricating a semantic-relevance heuristic, which was explicitly out of
  scope throughout.
- No approved automatic truthfulness judge exists for free-form TruthfulQA
  generations in this project; free-form results are reported only as process
  observations (decision rates, latency), never as a truthfulness score.
- The verifier's calibration diagnostics (ECE/Brier) are computable only on
  FEVER gold-evidence candidates (the only source with ground-truth
  entailment/contradiction labels for non-claim-extraction pairs); no
  equivalent calibration curve exists for RAGTruth's out-of-domain evidence.
- The retrieval corpus (FEVER/Wikipedia) was never rebuilt for RAGTruth's or
  TruthfulQA's actual source domains — this is the single largest,
  identified, and *addressable* limitation, and is the basis for the future
  work in Section 21.
- `max_correction_attempts=2` and the decision-policy thresholds are
  engineering defaults, not tuned against any evaluation set; results might
  differ under different (still-untuned, still not-tuned-on-eval-data)
  defaults, but that was explicitly out of scope for this project.

## 19. Reproducibility

See `data/processed/final/final_reproducibility_manifest.json` for the full
machine-readable manifest (environment, frozen model identifiers, seeds,
dataset sources, evaluation counts, SHA-256 hashes of small config/metadata
files, and git state). Summary: Python 3.12.14, torch 2.11.0+cu128,
transformers 5.16.1, CUDA 12.8, single NVIDIA RTX PRO 5000 Blackwell GPU
(shared with an unrelated job, never disturbed). Seed 42 used consistently
across dataset splitting, retrieval-eval sampling, and generation throughout.
**Full model weight files were not hashed** (multi-GB safetensors shards;
infeasible and non-standard for a documentation-only step) — this is stated
explicitly rather than implied; model identity/provenance instead rests on
exact HuggingFace model identifiers plus the architecture-construction checks
already verified in Steps 9–17 (bit-identical encoder/pooler reuse,
transplanted-not-random classifier rows). A 10-response deterministic subset
was independently re-run twice through the full pipeline in both Step 19 and
Step 21 pilots, producing identical outputs both times.

## 20. Conclusions

Claim-level evidence verification with iterative retrieval and targeted
self-correction, as implemented here, **does not significantly reduce
factual hallucinations** on RAGTruth's out-of-domain held-out test split —
end-to-end accuracy and macro F1 fall below a trivial majority-class
baseline, and the correction step causes substantially more harm than
benefit. This is not attributable to any single broken component: the
verifier is strong in-domain, the retrieval pipeline behaves as measured
throughout, and the correction loop faithfully executes what the decision
policy instructs. The dominant, best-supported explanation is a **domain
mismatch between the FEVER/Wikipedia retrieval corpus and RAGTruth's actual
source text**, compounded by a verifier **discrimination failure** (not a
calibration failure) on topically-irrelevant-but-lexically-similar evidence.
This negative result is reported in full, per this project's standing
evidence-first, no-fabrication methodology.

## 21. Future work

The following are identified, plausible next steps **and are explicitly NOT
implemented, attempted, tuned, or evaluated as part of this project** (see
`FINAL_PRESENTATION_RESULTS.md` and the Step 22 instructions' "future work
must NOT implement" boundary):

- Rebuilding or augmenting the retrieval corpus with in-domain sources
  (news, business-listing, QA-passage text matching RAGTruth's actual
  distribution) to directly test H1 in isolation.
- Investigating whether verifier overconfidence (H3) persists when retrieval
  quality is held at a known-good level, by re-examining Step 15/16's
  FEVER-domain, non-gold-but-topically-related evidence — a re-analysis
  question answerable from already-existing artifacts, requiring no new data
  collection.
- A learned (rather than fixed-threshold) decision policy, calibrated
  specifically for out-of-domain evidence discrimination rather than
  in-domain FEVER calibration.
- An approved, validated automatic truthfulness judge for free-form
  TruthfulQA generations (human raters or an approved GPT-judge protocol),
  to close the disclosed evaluation gap in Section 15.
- A per-evidence-item relevance annotation protocol for RAGTruth (or an
  equivalent out-of-domain benchmark), to make retrieval-relevance failure
  (Section 17, category 2) directly measurable rather than unidentifiable.

None of the above should be started without a fresh, explicit go-ahead — this
project's Step 22 boundary is documentation only.
