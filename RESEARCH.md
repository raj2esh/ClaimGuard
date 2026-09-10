# ClaimGuard — Research Document

Status: design document. **No experiments have been run. All results below are
marked `NOT RUN`.**

## Problem Statement

Large language models frequently produce fluent but factually incorrect
statements ("hallucinations") when generating open-ended text, including
answers to factual questions and long-form generations such as summaries or
biographies. Existing mitigations (RAG, self-consistency checks) reduce but
do not eliminate this, and few systems combine explicit claim-level
verification with a targeted, bounded self-correction loop.

## Motivation

A practical hallucination-mitigation system needs to do more than retrieve
evidence — it needs to (a) decompose a response into checkable units, (b)
verify each unit against evidence with a calibrated 3-way verdict
(supported / refuted / not enough info), and (c) correct only the
unsupported parts, rather than regenerating wholesale. This is closer to how
a human fact-checker works, and is hypothesized to be more sample-efficient
and more interpretable than end-to-end retraining approaches.

## Research Question

> Can claim-level evidence verification combined with iterative retrieval and
> targeted self-correction significantly reduce factual hallucinations in
> LLM-generated responses?

## Hypothesis

Adding claim-level verification and targeted correction on top of standard
RAG will reduce the measured hallucination rate (as scored by an external
verifier / FActScore-style atomic-fact accuracy) beyond what retrieval alone
achieves, without materially degrading fluency or answer relevance — and
this effect will hold across at least two different generator model
families, not just one.

**This is a hypothesis to be tested, not a finding.** No supporting or
refuting evidence has been collected yet.

## Proposed Methodology

1. Generate a draft response from the generator LLM for a given prompt.
2. Decompose the response into atomic factual claims (FActScore-style).
3. Retrieve candidate evidence per claim via a BM25 + FAISS hybrid retriever.
4. Rerank retrieved evidence with a cross-encoder.
5. Verify each claim against top evidence with an NLI model
   (SUPPORTS / REFUTES / NOT ENOUGH INFO).
6. For claims verified as REFUTES or NOT ENOUGH INFO, apply targeted
   self-correction: re-prompt the generator with the claim, the verdict, and
   the evidence, bounded to a small number of iterations
   (`correction.max_iterations` in `configs/default.yaml`).
7. Reassemble the corrected response.

Full detail on each component's model choice is in
`CLAIMGUARD_MODEL_SELECTION.md`.

## Baselines

Defined as placeholders in `configs/experiments.yaml`, none run yet:

- `vanilla_llm` — generator only, no pipeline.
- `standard_rag` — generator + retrieval + reranking, no verification/correction.
- `rag_verification` — adds verification, still no correction.
- `claimguard_full` — the complete proposed pipeline.

## Evaluation Metrics (planned, not yet implemented)

- Claim-level verification accuracy of the pipeline against held-out labeled
  claim/evidence pairs.
- FActScore-style atomic-fact support rate on generated long-form text.
- Hallucination rate on HaluEval-style binary hallucinated/not classification.
- TruthfulQA MC1/MC2 as an adversarial stress-test signal (not a primary
  effectiveness metric — see leakage concerns below).
- Fluency / answer-relevance sanity checks to confirm correction doesn't
  degrade output quality (metric TBD — e.g. length ratio, qualitative review,
  or a secondary judge model — not finalized).

## Dataset Strategy

| Dataset | Use |
|---|---|
| FEVER | Verifier training + validation |
| HaluEval | Verifier / correction-trigger training + validation |
| RAGTruth | Verifier training/validation + **primary held-out pipeline effectiveness test** |
| TruthfulQA | Evaluation-only stress test |
| FActScore / FActScore-Bio | Evaluation-only + claim-extraction design template |
| SelfCheckGPT WikiBio | Evaluation / baseline comparison |
| FEVEROUS | Optional, out of initial scope (no tabular evidence planned) |

## Leakage Concerns

FEVER (2018) and TruthfulQA (2021) are old, extremely well-known benchmarks
very likely present in the pretraining corpora of current open-weight
generator LLMs (Qwen, Llama, Mistral, Gemma families). Using them to measure
**the generator's own hallucination rate** risks measuring memorization
rather than the effect of the ClaimGuard pipeline. This does not equally
compromise using FEVER to train/evaluate the separate NLI verifier
component — that is in-distribution training, FEVER's intended use case.

RAGTruth and SelfCheckGPT's WikiBio set are treated as cleaner end-to-end
test signals, since their hallucinations are naturalistic LLM outputs rather
than synthetic, memorizable benchmark claims. The primary claim of pipeline
effectiveness should rest on RAGTruth-style held-out data, not FEVER or
TruthfulQA scores alone.

### FEVER acquisition findings (Step 6A, 2026-09-01)

The considerations above were written before FEVER was actually downloaded
and inspected. Having now acquired the real data (source:
`fever/fever` on Hugging Face - the FEVER project's own org, linked to
https://fever.ai; see `PROJECT_REPORT.md` Step 6A for full detail), several
concrete findings reinforce - and in one respect sharpen - the protocol
already selected in Step 3/4. **No part of the previously selected protocol
is being changed; this section adds evidence to it.**

- **Is FEVER suitable for training the verifier?** Yes, for the `train`
  split specifically. It is clean: correct 3-way labels only, no missing
  claim text, no malformed evidence-vs-label combinations, only 6 exact
  duplicate rows out of 311,431. This confirms `train` is appropriate as a
  verifier training/validation source, as planned.
- **Why FEVER should NOT be treated as a clean end-to-end generator
  evaluation benchmark - two independent reasons now, not one:**
  1. (As already documented above) FEVER is an old, widely-circulated
     benchmark very likely present in candidate generator LLMs' pretraining
     data, so a low hallucination rate measured on it may reflect
     memorization rather than the ClaimGuard pipeline's effect.
  2. (New, from real-data inspection) The `validation` and `test` splits, as
     delivered via Hugging Face's auto-converted Parquet export (the
     official script-based loader is blocked under the installed
     `datasets` 5.0.1: "Dataset scripts are no longer supported"), are
     **not clean, single-purpose splits as delivered**: both contain 19,998
     rows with a blank (`""`) label - almost certainly blind/withheld-label
     rows folded in during the automatic conversion - `validation`
     additionally has a label-casing inconsistency (33 rows labeled
     `"Not Enough Info"` instead of `"NOT ENOUGH INFO"`) and 19,037 exact
     duplicate rows (~24% of that split), and `validation`/`test` overlap by
     9,999 claim IDs, so they are not even disjoint from each other as
     delivered. None of this affects `train`. This is a second, independent
     reason not to treat FEVER's `test` split as a clean benchmark: even
     setting aside pretraining-contamination concerns, the split itself
     needs deliberate cleaning before it could mean anything as a metric.
- **Our selected verifier's own FEVER exposure:** as already selected in
  Step 3, `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` was
  trained directly on FEVER-NLI. Combined with the point above, evaluating
  *that* model's accuracy on FEVER's `test` split would be doubly
  uninformative - the model has already seen FEVER-NLI data during its own
  training (in-distribution, not a generalization test), and the delivered
  `test` split itself is not clean without further filtering. Using `train`
  to further validate/fine-tune the verifier remains appropriate (that is
  FEVER's intended use and the verifier's own training regime); claiming a
  "FEVER test accuracy" number for the verifier would not be.
- **Implication for the protocol:** unchanged from Step 3/4 - RAGTruth
  remains the primary source for measuring end-to-end pipeline
  effectiveness (Experiment 3), not FEVER. FEVER's `train` split is used for
  verifier training/validation only. If a clean FEVER `test`/`validation`
  signal is ever wanted later, it would require explicit deduplication and
  blank/casing-label filtering first - not attempted in Step 6A, which was
  acquisition and inspection only.

### HaluEval acquisition findings (Step 6B, 2026-09-01)

As with FEVER, the real downloaded data sharpens (without changing) the
protocol already selected in Step 3/4. Full acquisition/inspection detail is
in `PROJECT_REPORT.md` Step 6B; this section is the leakage/protocol-relevant
subset. **No part of the previously selected protocol is being changed.**

- **Structural finding relevant to leakage reasoning:** three of HaluEval's
  four subsets (qa, dialogue, summarization) are not naturalistic model
  outputs at all - each raw record is a *synthetically constructed*
  contrastive pair (a real reference response plus an algorithmically
  generated hallucinated counterpart, per the original HaluEval paper's
  sampling method), with no direct per-example label in the raw data. This
  reinforces the Step 3 decision to treat HaluEval as a **training/validation
  signal for the verifier and correction-trigger, not as a naturalistic
  end-to-end test** of ClaimGuard's effect on our own candidate generators
  (Qwen3-8B / Llama-3.1-8B-Instruct) - the hallucinations here were
  synthesized by a different, older process, not produced by the models
  ClaimGuard will actually be evaluated on.
- **The `general` subset is a partial exception:** its 4,507 records are
  real ChatGPT responses to real user queries with genuine human-annotated
  labels (not synthetic contrastive pairs), which makes it a somewhat
  stronger naturalistic signal than the other three subsets - but it is
  still specific to ChatGPT's outputs, not our own generator LLMs', so it
  remains a training/validation resource rather than a substitute for
  RAGTruth's role as the primary end-to-end effectiveness test.
- **Implication for the protocol:** unchanged - RAGTruth remains the primary
  source for Experiment 3 (end-to-end pipeline effectiveness). HaluEval
  (all four subsets, using the pair-expansion and general-label
  normalization designed in Step 6B) is added to the verifier/
  correction-trigger training and validation pool alongside FEVER's `train`
  split, exactly as planned in Step 3/4.

### RAGTruth acquisition findings (Step 6C, 2026-09-01) — primary evaluation dataset

RAGTruth is ClaimGuard's primary end-to-end evaluation dataset (Experiment 3
in the Research Protocol above), so this inspection was held to a higher
standard, specifically around leakage. Full acquisition/inspection detail is
in `PROJECT_REPORT.md` Step 6C. **The protocol is not changed here — it is
validated against the real data and confirmed sound.**

- **Is RAGTruth's own train/test split safe to use as designed (train for
  verifier development, test reserved for final evaluation)? Yes —
  empirically confirmed, not assumed.** RAGTruth's 2,965 underlying source
  items (documents/questions/business records) each generate exactly 6
  model responses (one per model: gpt-4-0613, gpt-3.5-turbo-0613,
  mistral-7B-instruct, llama-2-7b/13b/70b-chat), and **every single
  source item's 6 responses belong entirely to one split** — confirmed
  by joining response.jsonl to source_info.jsonl via `source_id` and
  checking, for all 2,965 source items, whether their responses' `split`
  values are ever mixed. Zero were. This means the `train`/`test` split is
  genuinely source-disjoint, not just row-disjoint — the property that
  actually matters for a clean evaluation boundary.
- **One apparent cross-split overlap was found and investigated, not
  reported at face value:** the exact same response *text* appeared in both
  `train` and `test`. Tracing the actual records showed this was the
  generic stock refusal phrase `"Unable to answer based on given
  passages."`, produced verbatim by different models for different,
  unrelated source items (different `source_id`s) in both splits — a
  coincidental short boilerplate string, not shared source content. No
  actual leak.
- **Annotation quality validated programmatically, not trusted from the
  README:** all 14,289 real span-level hallucination labels'
  `response_text[start:end]` slices were checked against each label's own
  `text` field — 100% exact match, zero out-of-bounds or malformed offsets.
  This gives high confidence in using RAGTruth's span annotations directly
  (e.g. for evaluating whether ClaimGuard's verifier/correction stage
  locates hallucinated spans correctly), not just its response-level
  presence/absence of hallucination.
- **Evaluation-boundary enforcement:** `src/claimguard/datasets/ragtruth.py`
  makes the boundary explicit in code, not just in a document — every
  normalized record carries `eval_reserved` (`True` iff `split == "test"`),
  `get_eval_set()`/`get_training_pool()` are the only sanctioned accessors,
  and `assert_no_train_test_leakage()` is a reusable, tested safeguard
  callable before any future training step. The Step 6C development sample
  was built exclusively from `train` (verified by an in-script assertion at
  write time) — RAGTruth's `test` split has not been read into any sample,
  processed file, or code path other than the one-time inspection script.
- **Implication for the protocol:** confirmed, unchanged. RAGTruth `test`
  (2,700 responses across 450 source items) remains the untouched,
  reserved end-to-end evaluation set for Experiment 3. RAGTruth `train`
  (15,090 responses) may be added to the verifier/correction-trigger
  training pool alongside FEVER `train` and HaluEval if a future step
  chooses to use it that way — that decision is deferred, not made here;
  Step 6C's only claim is that the split itself is safe to use for that
  purpose if chosen.

### TruthfulQA acquisition findings (Step 6D, 2026-09-01)

Full acquisition/inspection detail is in `PROJECT_REPORT.md` Step 6D. **This
section confirms, and if anything strengthens, TruthfulQA's Step 3/4
designation as evaluation-only — it is explicitly NOT added to the
verifier/correction-trigger training pool, unlike FEVER and HaluEval.**

- **No official split exists — verified empirically, not assumed.** Neither
  raw TruthfulQA file (`TruthfulQA.csv`, the original authors' generation-
  style benchmark, nor `data/mc_task.json`, their official multiple-choice
  reformulation of the same 790 questions) contains any split-like column.
  This is not a gap in the acquisition — TruthfulQA was designed and
  published as a single fixed benchmark set, not a trainable dataset with
  held-out partitions.
- **Why this reinforces evaluation-only status:** because there is no
  train/test partition of TruthfulQA's own, there is no leakage-safe subset
  of it that could be used for verifier training without consuming part of
  the only copy of the benchmark that exists. Unlike FEVER (which has a
  large, separate `train` split distinct from its `test`/`validation`) or
  RAGTruth (source-disjoint `train`/`test`, confirmed in Step 6C),
  TruthfulQA offers no such safe-to-train-on portion. Using any of it for
  training would mean the same 790 questions could no longer serve as a
  clean stress test later. The Step 3/4 decision to treat TruthfulQA as
  **evaluation-only** is therefore not just a policy choice but the only
  structurally sound way to use this particular dataset.
- **Scale check:** 790 questions (37 categories) is small relative to
  FEVER/HaluEval/RAGTruth's tens of thousands of examples — reinforcing
  that TruthfulQA's role is a targeted adversarial stress test, not a
  general-purpose training source, consistent with how it was already
  described in the Research Protocol above ("adversarial stress test").
  Note the real counts (790 questions, 37 categories) differ from the
  commonly-cited 817/38 — see PROJECT_REPORT.md Step 6D; this project's
  own artifacts use the real, currently-downloaded counts, not the
  historical paper figures.
- **Implication for the protocol:** unchanged, and now on firmer footing —
  TruthfulQA remains evaluation-only (Experiment 2's baseline stress test
  and any later full-pipeline stress test), never part of the
  verifier/correction-trigger training pool that FEVER `train`, HaluEval,
  and (optionally, later) RAGTruth `train` may contribute to.

### Dataset integration and protocol validation (Step 7, 2026-09-01)

Steps 6A-6D acquired and inspected each dataset in isolation. Step 7
reconciles those real findings into one enforced protocol - a machine-
readable manifest (`data/processed/dataset_manifest.json`) and a code layer
(`src/claimguard/datasets/manifest.py`) that make the roles below
structurally hard to violate, not just documented in prose. **No dataset
role established in Step 3/4 or Steps 6A-6D was changed here** - this
section records what was newly decided (the label mapping and the pool
construction/split mechanics) and what was newly discovered (real
cross-dataset contamination between two training-eligible sources and one
evaluation dataset).

**Verifier training/development pool - final composition:**
FEVER `train` (145,449 claims) + HaluEval `qa`/`dialogue`/`summarization`
(20,000 pair-expanded records each, 60,000 total) = **205,449 records**.
HaluEval `general` is normalized and available but **excluded** from this
pool: it has no premise/grounding field at all (context is always `{}`,
confirmed in Step 6B), so it cannot be represented as a
(premise, hypothesis, label) NLI training pair without fabricating a
premise the source data doesn't contain. It remains a candidate for a
future, separate correction-trigger/response-quality classifier that
doesn't require a premise - not decided or built here.

**Label mapping (verifier target: `{entailment, neutral, contradiction}`,
matching the selected DeBERTa verifier's own `id2label`):**

| Source | Native label | Target | Cardinality | Rationale |
|---|---|---|---|---|
| FEVER | SUPPORTS | entailment | 1:1 | FEVER's own definition: evidence directly supports the claim |
| FEVER | REFUTES | contradiction | 1:1 | FEVER's own definition: evidence directly contradicts the claim |
| FEVER | NOT ENOUGH INFO | neutral | 1:1 | FEVER's own definition: evidence is insufficient to decide |
| HaluEval (pair subsets) | not_hallucinated | entailment | 1:1 | The reference answer/response/summary is, by construction, grounded in the given context |
| HaluEval (pair subsets) | hallucinated | contradiction | 1:1, **approximation** | HaluEval's generation methodology produces content that is factually incorrect relative to the grounding - skews toward direct contradiction, but conflates what FEVER separately calls REFUTES and NOT ENOUGH INFO. No `neutral` examples come from HaluEval as a result. |
| HaluEval (general) | n/a | **not mapped** | excluded | No premise field exists to pair with a hypothesis - see above |

**A gap surfaced, not hidden:** FEVER's normalized evidence
(`claimguard.datasets.fever`) is a (`wiki_url`, `sentence_id`) *reference*,
not resolved sentence text - Step 6A correctly did not download the
5.4M-page `wiki_pages` corpus (out of scope for acquisition/inspection).
Every FEVER pool record is therefore marked `premise_text_available=False`;
actual training on the FEVER portion of this pool requires a future step to
resolve evidence text. HaluEval's pair-subset records, by contrast, already
carry full literal premise text (`premise_text_available=True`) and are
immediately usable.

**Cross-dataset contamination check (exact text match only - not a
near-duplicate/fuzzy check):**
- Pool vs. RAGTruth test: 0 hypothesis or premise matches against RAGTruth
  responses. **100 exact matches** between HaluEval `summarization`
  documents (premise text, in the training pool) and RAGTruth test's
  `Summary`-task source articles (`source_info` text) - both draw from
  CNN/DailyMail. **Investigated, not just counted: confirmed a genuine
  shared news article**, not a coincidental short-string match. This is
  real, quantified cross-dataset document contamination - 100 of the 2,700
  RAGTruth test responses may be grounded in an article the verifier's
  training data has also seen (as HaluEval context). **Mitigation
  recommended for the training step (not applied here):** exclude those
  100 HaluEval `summarization` records from the training pool before
  actual training, or exclude the affected RAGTruth test responses from
  the evaluation set.
- Pool vs. TruthfulQA: 70 exact hypothesis-vs-answer matches. **Investigated
  and confirmed benign** - all are short generic tokens (`"no"`, `"yes"`,
  country names like `"uk"`/`"japan"`/`"spain"`), not substantive shared
  claims.

**Development split:** built from the 205,449-record pool only (never
RAGTruth `test`, never TruthfulQA) - 90/10, seed 42 (matching
`configs/default.yaml`), grouped by `(dataset, subset, raw_index)` so that
HaluEval's two pair-expanded records from the same underlying example are
never split across train/dev. Result: 184,922 train / 20,527 dev,
group-key overlap between them confirmed 0.

**Balancing: deferred, not applied.** Pool label distribution is
`entailment` 110,035 / `contradiction` 59,775 / `neutral` 35,639 - `neutral`
is under-represented because only FEVER contributes it (HaluEval's binary
mapping produces none). No downsampling/reweighting was applied; this is
left as an explicit training-time decision (e.g. class weights,
oversampling), not silently resolved by trimming the dataset now.

**Implication for the protocol:** confirmed and operationalized, not
changed. FEVER `train` + HaluEval `qa`/`dialogue`/`summarization` are the
verifier training/development pool, exactly as Step 3/4 named them.
RAGTruth `test` and TruthfulQA remain hard-blocked from training
(`claimguard.datasets.manifest.RoleViolationError` on any attempt).
RAGTruth `train` remains deferred, not included. The one new,
protocol-relevant finding - the HaluEval/RAGTruth CNN/DailyMail overlap -
was resolved in Step 8, below.

### Cross-dataset contamination resolution (Step 8, 2026-09-02)

Step 7's finding (100 exact document matches between HaluEval `summarization`
premise text and RAGTruth `test` Summary source articles) was reproduced
independently, identified at the individual-record level, and resolved.
**No dataset role changed; only downstream training eligibility of 200
specific HaluEval records did. RAGTruth test itself was not modified.**

- **Reproduced exactly:** the same aggregate check found 100 matching
  documents, confirmed via a stricter, source-level check scoped to
  RAGTruth's `Summary` task_type specifically (Step 7's original check
  pooled all task types' `source_info` text together; this step confirmed
  every match genuinely belongs to `Summary`).
- **Scope precisely quantified, not assumed:** 100 unique HaluEval raw
  indices and 100 unique RAGTruth `test` source_ids (1:1 - no duplicate
  documents among the matches on either side), but **600 unique RAGTruth
  test response records** affected, because each source_id has exactly 6
  model responses (Step 6C) and all 6 sit in `test` for these particular
  sources (source-disjoint splits, also Step 6C) - 100 x 6 = 600, matching
  the audit-record count exactly.
- **Mitigation (per the decision criterion - RAGTruth test is the primary
  evaluation benchmark, so it must stay untouched):** excluded the 100
  affected HaluEval `summarization` raw records (200 normalized
  pair-expanded pool records - both the `not_hallucinated` and
  `hallucinated` half of each contrastive pair) from the verifier
  training/dev pool. The exclusion set is **computed fresh from the raw
  data on every run** (`claimguard.datasets.manifest.
  contaminated_halueval_summarization_raw_indices()`), not a hard-coded
  list of row numbers - reproducible and auditable via
  `data/processed/contamination_report.json`.
- **Verified effective using the same methodology that found the
  problem:** re-ran Step 7's original (broader) overlap check directly
  against the post-exclusion pool - **0 remaining overlap**, down from 100.
- **Additional sweep found nothing else:** FEVER, HaluEval `qa`, `dialogue`,
  and `general` (checked even though `general` isn't in the training pool,
  for audit completeness) all show **zero** exact overlap with RAGTruth
  test. The pre-existing HaluEval-`qa`/70 and HaluEval-`dialogue`/1
  TruthfulQA answer overlaps were re-confirmed benign (generic short
  tokens) - unchanged by this step.
- **FEVER premise-text status - determined, not solved:** `wiki_pages` is
  still not downloaded locally (confirmed by direct filesystem check, not
  assumed); FEVER pool records remain reference-only and are not yet
  trainable. Acquiring `wiki_pages` (or an equivalent resolved Wikipedia
  source) remains a separate future step, exactly as Step 7 already
  documented - not silently solved here.
- **Final pool:** 205,249 records (205,449 minus 200 excluded) - 184,723
  train / 20,526 dev, same seed (42) and dev_ratio (0.10), pair-grouping
  and the group-key-based split mechanics unchanged. Both evaluation
  boundaries (RAGTruth test hard-blocked, TruthfulQA hard-blocked) were
  re-verified after the pool changed, not just assumed to still hold.

### Verifier baseline and a real scope limitation on Experiment 1 (Step 9, 2026-09-02)

Step 9 built and trained the first verifier baseline. **A finding here
materially narrows Experiment 1's near-term scope**, and is recorded
explicitly rather than glossed over: of the 205,249-record leakage-safe
pool finalized in Step 8, **only the 59,800 HaluEval-derived records
(qa/dialogue/summarization, post-contamination-exclusion) currently have
literal premise text.** All 145,449 FEVER records are reference-only
(`evidence_wiki_url`/`evidence_sentence_id`, not resolved sentence text) -
correctly so, since Step 6A did not download the 5.4M-page `wiki_pages`
corpus (out of scope for acquisition). No premise text was fabricated to
work around this.

Consequence for the Research Protocol's Experiment 1 ("Verifier
validation... FEVER dev + held-out RAGTruth-derived triples"): **the
current baseline validates the verifier on HaluEval only, not FEVER.** A
FEVER-inclusive verifier validation is blocked on a future `wiki_pages` (or
equivalent Wikipedia text) acquisition step - not attempted here, and
explicitly not silently worked around (e.g. by using the claim text alone,
or the wiki page title as a fake premise, both of which were considered
and rejected as invalid representations).

**Baseline result:** fine-tuning the project's already-selected verifier
(`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) for 1 epoch on
the 53,816-example HaluEval-only train set reached 91.6% accuracy / 91.6%
weighted F1 on the 5,984-example dev set, with entailment F1 91.5% and
contradiction F1 91.7%. **The `neutral` class has zero support in this dev
set** - an expected, documented consequence of the label mapping (HaluEval's
binary hallucinated/not_hallucinated mapping never produces `neutral`; only
FEVER does, and FEVER is excluded from this baseline) - so the reported
`macro_f1` (0.611) is mechanically depressed by an undefined third class
and should not be read as "the model is bad at detecting insufficient
evidence." `weighted_f1` is the more meaningful single-number summary for
this baseline's actual (binary, in practice) task.

This is a legitimate, non-fabricated baseline for what HaluEval alone can
validate; it is not yet a validation of the verifier's FEVER-style 3-way
NLI behavior, and the Research Protocol's Experiment 1 should be understood
as partially complete until `wiki_pages` acquisition happens.

### FEVER wiki_pages acquisition and evidence-to-premise resolution (Step 10, 2026-09-02)

Step 9 identified that FEVER's evidence is reference-only
(`wiki_url`/`sentence_id`), blocking FEVER from contributing to the
verifier training pool. Step 10 closes that gap for the SUPPORTS/REFUTES
majority of FEVER, while honestly surfacing that it does **not** close the
gap for the `neutral` class.

**Acquisition:** downloaded the authoritative FEVER `wiki-pages.zip`
(fever.ai, CC BY-SA 3.0 + GNU FDL, same license as the FEVER claims data)
to `data/raw/fever/wiki_pages/`. The zip contains 224 entries; 109 are the
real per-shard JSONL files, the rest are macOS AppleDouble (`._*`)
resource-fork sidecars from how the archive was packaged - filtered out
after a first crash exposed them (see Obstacles below), not silently
ignored without investigation. Real, empirically confirmed corpus size:
**5,416,537 pages, 42,041,086 sentence lines** (0 duplicate page IDs;
20,431 pages with empty content; 507 malformed line entries; 1 page with an
empty ID - none of which blocked resolution, see below).

**Index:** built a deterministic SQLite resolution index
(`data/processed/fever/wiki_pages_index.sqlite`, ~6.4GB) with `(page_id,
sentence_id)` as the sentence table's primary key and each page's full
intro `text` also stored (added after the first resolution pass - see
Obstacles). 5,416,536 pages / 42,041,086 sentences indexed in ~138s;
78,260 duplicate `(page_id, sentence_id)` collisions resolved
first-encountered-wins (deterministic given fixed file-processing order).

**Resolution method:** exact `(page_id, sentence_id)` primary-key lookup
against the index - explicitly **not** fuzzy/near-duplicate matching. A
FEVER claim resolves if **one full evidence set** resolves completely (no
partial sets used, no mixing sentences across different evidence sets - the
existing deterministic evidence-set ordering from `group_claims` is
preserved). Only SUPPORTS/REFUTES claims were attempted; NOT ENOUGH INFO
was **not** attempted (see below).

**Two real resolution obstacles found and fixed by investigation, not
assumed to be upstream data problems (per instructions):**
1. **The `sentence_id == -1` sentinel.** First pass reached only 50.44%.
   Investigating a sample of "missing_sentence_id" failures (before
   labeling it upstream) found 100% had `sentence_id == -1`, and 100% of
   those referenced pages existed with real, non-empty content - a genuine
   FEVER annotation convention meaning "the evidence is the page as a
   whole," which the resolver hadn't accounted for. Fixed by storing each
   page's full text and resolving `sentence_id == -1` against it. Result:
   99.10%.
2. **Unicode normalization mismatch.** The remaining 991 failures were all
   `missing_page`. Investigating the 123 unique page ids involved found all
   123 resolved exactly under Unicode NFC normalization (0 truly absent) -
   the FEVER claims JSON and wiki-pages.zip use different normalization
   forms for the same accented titles (e.g. `André_Téchiné`). Fixed with a
   deterministic exact-match-then-NFC-fallback page lookup (still not
   fuzzy - exact string equality after standard canonicalization). Result:
   **100.00% (109,810/109,810) SUPPORTS/REFUTES FEVER train claims now
   resolve to real, resolved Wikipedia sentence premise text** (80,035
   SUPPORTS / 29,775 REFUTES).

**What did NOT change - the neutral-class gap, stated plainly:** FEVER's
35,639 NOT ENOUGH INFO train claims carry **no evidence annotation in
FEVER's own data at all** (`evidence_wiki_url == ""` for every such row -
confirmed in Step 6A, reconfirmed here). There is structurally nothing to
resolve for them, and fabricating a premise was explicitly out of scope.
`claimguard.datasets.fever.load_resolved_split()` marks these
`resolution_status="no_evidence_annotation"` (never `"unresolved"`, which
is reserved for a SUPPORTS/REFUTES claim whose evidence genuinely could not
be found - a case that did not occur here) and
`claimguard.datasets.manifest.fever_train_pool_records()` excludes them
from the trainable pool rather than coercing them in. **Consequence: FEVER
still contributes 0 `neutral`-label examples**, and since HaluEval's binary
mapping also never produces `neutral`, **the verifier training pool's
`neutral` class remains at 0 examples after this step.** Per the standing
instruction, this is reported honestly here rather than glossed over: the
3-way verifier cannot yet be validated on a genuine neutral class from this
project's data as currently constructed.

**Updated verifier training/dev pool** (`data/processed/dataset_manifest.json`
v1.2, `data/processed/verifier_pool/{train,dev}.jsonl`, rebuilt from real
data via `scripts/build_dataset_manifest.py` and
`scripts/build_verifier_pool.py`): **169,610 total records** (109,810 FEVER
+ 59,800 HaluEval, unchanged Step 8 contamination exclusion of 200
records) - **152,720 train / 16,890 dev**, same seed (42), same dev_ratio
(0.10), HaluEval pair-grouping and deterministic group-key split mechanics
unchanged and re-verified (0 group-key overlap between train/dev). Pool
label distribution: `{'entailment': 109,935, 'contradiction': 59,675,
'neutral': 0}`.

**Leakage re-check, extended for the first time to FEVER's real premise
text** (FEVER previously had no premise text to check): FEVER premise text
vs. RAGTruth test (`response_text`/`source_info_text`) and vs. TruthfulQA
(`question`/`answer`) - **zero overlap on every field**. HaluEval's
pre-existing overlaps (100 RAGTruth-test contamination cases, already
excluded since Step 8; 70/1 generic TruthfulQA answer-token overlaps,
previously classified benign) are unchanged by this step.

**Tests:** 18 new tests across `tests/test_fever_dataset.py` (sentence/page
resolution including the `-1` sentinel and NFC-fallback paths, per-claim
resolution including no-cross-set-mixing, NOT ENOUGH INFO vs. unresolved
vs. unknown-label status distinctions, `load_resolved_split` field
preservation) and `tests/test_manifest.py` (pool construction now requiring
real resolved premise text, exclusion of NOT ENOUGH INFO and genuinely
unresolved claims, the neutral-class-remains-zero finding asserted
explicitly so a future regression toward silent fabrication would be
caught). Full suite: 157/157 passed (139 before Step 10, +18 new).

**Not done in this step (explicitly deferred):** no model retraining
(deferred to Step 11 - the pool is ready, but training itself is out of
this step's scope), no retrieval/FAISS/BM25/reranking, no RAGTruth/TruthfulQA
evaluation runs, no hyperparameter tuning.

### Neutral-class strategy and final three-way verifier dataset validation (Step 11, 2026-09-03)

Step 10 closed FEVER's SUPPORTS/REFUTES premise-text gap (100% resolution)
but left `neutral` at 0 in the verifier pool. Step 11's purpose was to
determine, from the project's own already-acquired data and the
authoritative FEVER documentation, whether a legitimate (non-fabricated,
leakage-safe, premise-grounded) neutral source exists - and to report
honestly if it does not, rather than force a solution.

**Defining neutral (unchanged from FEVER's own semantics, stated
explicitly this step):** a neutral example is one where the available
premise does not support the claim, but also does not establish it as
false - not "hallucinated," and not automatically "contradiction." This
project has never equated the two, and this step re-confirms that HaluEval's
binary label was never reinterpreted to manufacture a third class.

**FEVER NOT ENOUGH INFO investigated deeper, not re-assumed from Step 6A/10:**
verified, exhaustively (every row, not a sample), that all 47,609 raw
NOT ENOUGH INFO rows in FEVER **train** (35,639 unique claims) carry the
pure sentinel (`evidence_wiki_url=""`, `evidence_id=-1`,
`evidence_sentence_id=-1`) with **zero exceptions** - no latent evidence
field, no hidden candidate page. Cross-checked against fever.ai's own
documentation: NOT ENOUGH INFO claims are officially recorded with null
evidence, and **no officially-distributed predicted-evidence/retrieval-
baseline artifact exists** for this dataset. FEVER **validation** (already
excluded from training since Step 6A/7 for independent data-quality
reasons - blank labels, duplicate rows, casing variants) does contain
151/13,883 NOT ENOUGH INFO rows with a non-sentinel `evidence_wiki_url` -
investigated and found to be a data artifact (the same wiki_url,
`Starrcade`, attached to several unrelated claims with
`evidence_annotation_id=-1`), not genuine annotator-recorded evidence, and
out of scope regardless since validation isn't a training-eligible split.
**Conclusion: the authoritative FEVER release provides no legitimate
premise for NOT ENOUGH INFO claims.**

**HaluEval investigated for a hidden third category:** confirmed (by
reading, not reinterpreting, the real schema already documented in
`claimguard.datasets.halueval`) that every subset - qa, dialogue,
summarization, general - is strictly binary
(`hallucinated`/`not_hallucinated`). No subset encodes an "insufficient
evidence" or "unknown" category anywhere in the real data.

**RAGTruth and TruthfulQA considered and excluded on protocol grounds, not
content grounds:** both remain hard-blocked, evaluation-only datasets under
the Step 3/4 research protocol. Even if either contained neutral-shaped
content, using it for training would violate the evaluation boundary this
project's entire leakage-safety design depends on - so neither was a
candidate regardless of label content, and this step did not weaken that
boundary to go looking.

**No new dataset was proposed or downloaded.** Per the standing instruction
to check the existing research plan and preserve evaluation boundaries
before considering acquisition, and since the investigation above
concluded no already-acquired dataset can legitimately supply neutral
examples, the correct next action (if a 3-way verifier is later required)
would be a **separate, explicitly-scoped dataset-acquisition step** -
proposing and downloading a new source was explicitly out of this step's
scope, and was not done.

**Final decision: the verifier training/dev pool is a BINARY
(entailment/contradiction) dataset, not a 3-way dataset with a
temporarily-empty third class.** This is now enforced in code, not just
documented in prose:
`claimguard.datasets.manifest.verifier_dataset_label_space_status(pool)`
computes the real classification mode fresh from the pool's actual label
counts every time it is called (`binary` only when exactly
entailment+contradiction are populated and neutral is genuinely absent;
`three_way` only when all three are populated; `invalid` for anything
else, e.g. an out-of-scheme label from a future bug). `data/processed/
dataset_manifest.json` (bumped to **v1.3**) now carries a
`verifier_dataset_status` block with this real status plus the full
neutral-class investigation summary, and
`scripts/validate_verifier_dataset.py` (new) cross-checks the manifest's
declared mode against a freshly-rebuilt pool and **fails loudly** if they
ever disagree - specifically to prevent a future run from silently
claiming 3-way support while `neutral` is actually empty.

**Stated plainly, per the explicit instruction not to hide this: "The
current authoritative datasets do not provide a leakage-safe,
premise-grounded neutral training class."** This is a scientific
limitation of the currently-acquired data, not a software defect - the
resolution pipeline built in Step 10 works correctly and completely; there
is simply nothing legitimate for it to resolve for the neutral class. The
Step 9/10 binary baseline protocol is retained and confirmed correct;
Step 12's training step should proceed as an explicitly binary verifier
baseline, not be silently upgraded to "3-way" in any report or config.

**Tests:** 9 new tests in `tests/test_manifest.py`
(`TestVerifierDatasetLabelSpaceStep11`) covering synthetic binary/
three-way/invalid label-space fixtures (including a simulated "prohibited
relabeling" case that must resolve to `invalid`, never silently accepted),
real-pool confirmation that zero fabricated neutral examples exist and
FEVER NOT ENOUGH INFO claims remain excluded, and manifest/pool
classification-mode consistency. Full suite: 166/166 passed (157 before
Step 11, +9 new).

**Not done in this step (explicitly deferred):** no verifier retraining, no
hyperparameter/class-weight/oversampling/downsampling changes, no synthetic
neutral generation, no RAGTruth/TruthfulQA evaluation, no retrieval/FAISS/
reranking work, no new dataset acquisition.

### Final binary ClaimGuard verifier training and baseline evaluation (Step 12, 2026-09-03)

Step 11 concluded the verifier is scientifically restricted to binary
(entailment/contradiction) classification for the current data. Step 12
trained the first FINAL binary verifier on the full Step 10/11 pool
(FEVER SUPPORTS/REFUTES with resolved premise text + HaluEval
qa/dialogue/summarization), superseding Step 9's preliminary,
HaluEval-only baseline.

**A genuine 2-class head, not the old 3-class head trained on 2 labels.**
Step 9 fine-tuned the pretrained model's original 3-class
(entailment/neutral/contradiction) classification head while only ever
supplying labels 0 and 2 - leaving the `neutral` output dimension
completely unsupervised while still shipping a nominally 3-class
architecture. Step 12 instead constructs a real 2-row classifier
(`claimguard.verifier.binary_model.build_binary_model`): the pretrained
encoder and pooler are reused (**verified bit-identical** to the original
checkpoint, not assumed), and the two new classifier rows are initialized
from the ORIGINAL pretrained model's own entailment/contradiction rows
(indices verified from the model's own config, not assumed to be `[0, 2]`)
- not randomly initialized, not trained from scratch. Confirmed
empirically: 435,063,810 total parameters (exactly 1,025 fewer than the
3-class model's 435,064,835 - the removed third classifier row + bias).

**Dataset (verified from disk, not assumed):** 169,610 total records -
152,720 train (98,934 entailment [64.8%] / 53,786 contradiction [35.2%];
98,758 FEVER [64.7%] / 53,962 HaluEval [35.3%, split ~evenly across
qa/dialogue/summarization]) and 16,890 dev (11,001 entailment / 5,889
contradiction; 11,052 FEVER / 5,838 HaluEval). Pre-training, the Step 11
validation gate (`scripts/validate_verifier_dataset.py`) was re-run fresh
and all 9 checks passed; `train_binary.py` additionally hard-asserts
`classification_mode == "binary"` from the manifest and that every loaded
label is exactly `entailment`/`contradiction` before training can start.

**Training:** same conservative Step 9 starting configuration (seed 42,
batch 16, grad-accum 2, lr 2e-5, wd 0.01, 1 epoch, warmup ratio 0.06
linear, bf16, max_grad_norm 1.0, no class weighting/oversampling/
downsampling, no hyperparameter tuning) - only the dataset and the
classification head changed. Smoke test (real GPU, tiny slice) passed
cleanly first; the one full run then ran 4,773 steps (1 epoch) in 1,964s
(~32.7 min), peak GPU memory 14.4GB, with a clean monotonic loss decrease
and no NaN/Inf/OOM/divergence.

**Result - a strong, honest binary baseline:** dev accuracy 96.20%, macro
F1 0.9585 (now a meaningful number, unlike Step 9's neutral-distorted
0.6107), weighted F1 0.9621; entailment F1 0.9705, contradiction F1
0.9465. **Per-source breakdown matters:** FEVER alone reaches macro F1
0.9704, while HaluEval alone reaches only 0.9344 (dialogue specifically
the weakest subset at 0.8695) - the aggregate number would have hidden
this gap if reported alone.

**Step 9 comparison - explicitly NOT apples-to-apples, verified rather
than assumed:** Step 9's dev set contained 5,984 usable HaluEval examples;
the current pool's dev.jsonl contains 5,838 HaluEval records - **a
different set**, because Step 10 changed the FEVER pool's group-key
membership (dropping NOT ENOUGH INFO claims from the buildable pool),
which changes the seeded shuffle's output for every group key, including
HaluEval's, even though the seed (42) is unchanged. The original Step 9
`dev.jsonl` was also overwritten in place by Step 10's pool rebuild, so an
exact historical replay isn't possible. Reported side by side, clearly
labeled as two non-identical conditions: Step 9 (HaluEval-only
architecture and data) reached 91.58% accuracy / 0.9160 weighted F1; Step
12's own HaluEval-only dev slice (a same-domain, not same-examples,
internal comparison) reaches 93.44% accuracy / 0.9344 weighted F1 - a
plausible but not rigorously isolated improvement, consistent with (not
proof of) the genuine binary head and/or FEVER co-training helping, but
not claimed as a controlled ablation result.

**Checkpoint:** `experiments/verifier_binary_final/final/` (best-by-macro_f1,
essentially tied with the final step) plus `checkpoint-4500` and
`checkpoint-4773` retained. Full reproducibility metadata (Python/PyTorch/
Transformers/CUDA versions, exact binary-head construction report, dataset
manifest snapshot, git status snapshot) saved alongside.

**This is a binary supported-vs-contradicted/hallucinated classifier for
the current training protocol. It is not a three-way entailment/
contradiction/neutral classifier.** RAGTruth test and TruthfulQA remain
untouched, reserved for later evaluation - not used for model selection or
tuning here.

**Tests:** 15 new tests (`tests/test_verifier_binary.py`) covering the
binary head construction (architecture verification, weight-transplant
correctness against a real tiny DebertaV2 fixture, encoder/pooler
bit-identity verification, rejection of a non-3-class source model),
`assert_binary_head`'s rejection of non-binary/mismatched configurations,
`assert_binary_labels_only`'s rejection of `neutral` or any unexpected
label, and `compute_metrics_by_group`'s per-source breakdown correctness.
Full suite: 181/181 passed (both before and after the training run).

### ClaimGuard retrieval corpus, BGE embeddings, and FAISS index infrastructure (Step 13, 2026-09-03)

Step 12 finalized the binary verifier; Step 13 built the first piece of the
retrieval side of the pipeline - a corpus, embeddings, a FAISS index, and a
retrieval-only evaluation - with no reranking, no verifier integration, and
no RAGTruth/TruthfulQA evaluation yet.

**Retrieval objective, defined from the existing protocol, not guessed:**
RESEARCH.md's own methodology (step 3) retrieves evidence via a BM25+FAISS
retriever against a Wikipedia-derived knowledge source. FEVER's
`wiki_pages` corpus (Step 10) is the only Wikipedia knowledge source this
project has acquired, so it is the retrieval corpus - not RAGTruth (its
per-example source documents are evaluation-reserved and not a general
knowledge base anyway) and not TruthfulQA (no grounding corpus at all).

**Corpus scope - a principled, bounded subset, quantified before deciding:**
the full `wiki_pages` corpus (5,416,536 pages / 42,041,084 sentences) was
NOT fully embedded - that would cost many hours of GPU time on a shared
machine for an infrastructure-building step. Instead, the corpus is scoped
to every sentence of every Wikipedia page that FEVER train's SUPPORTS/
REFUTES claims reference as evidence (12,549 unique pages, across all
annotated evidence sets, not just the one Step 10 ultimately resolved) -
**109,350 sentences** after removing 50,901 empty/boilerplate lines. This
keeps the corpus tied directly to the project's already-validated FEVER
verifier-training scope while remaining fully tractable. Expanding to the
full corpus is a natural, separate future scaling step - not attempted
here.

**Retrieval unit:** the individual FEVER/wiki_pages sentence (not a
chunk) - this preserves FEVER's own sentence identity and provenance
exactly, and no chunking was necessary since native sentences are already
well within BGE's 512-token limit. Every record carries a stable
`corpus_id` (`"{page_id}::{sentence_id}"`), independent of FAISS row
position, mapping back to page_id/sentence_id/exact text/corpus_version.

**Page resolution reused Step 10's proven resolver** (exact-match-then-
NFC-fallback) rather than a fresh lookup - result: **all 12,549 requested
evidence pages resolved, 0 missing**, 0 corpus validation issues.

**Embedding model - unchanged, no second model introduced:**
`BAAI/bge-large-en-v1.5` (Step 5C's already-selected/smoke-tested model),
read from `configs/models.yaml`, fp16, 1024-dim (verified from the actual
model output), `normalize_embeddings: true`. A real-GPU smoke test
(dimension/dtype/device/finite/normalization/determinism all verified,
including two independent encode calls agreeing exactly, max_abs_diff=0.0)
passed before the full run. Full-corpus embedding: **109,350/109,350
embedded, 0 failures, all finite, 65.5s wall-clock (1,670 records/sec)**.

**FAISS index:** `IndexFlatIP` (exact, flat, inner-product) - justified by
BGE's L2-normalized output (inner product = cosine similarity) and by the
corpus's modest scale (an approximate index would be premature
optimization at 109K vectors). 427.1MB on disk (index) + 30.7MB (metadata
mapping every FAISS row to its `corpus_id`/text/provenance). Save/reload
validated bit-for-bit identical top-k results.

**Retrieval evaluation (FEVER train only, never RAGTruth/TruthfulQA):**
2,000 deterministically-sampled (seed 42) FEVER train claims, each with a
ground-truth relevant-sentence set from its own annotated evidence.
Relevance criterion: a retrieval at cutoff k is a HIT if any one gold
evidence sentence appears in the top-k for that claim's raw text used as
query.

| Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|
| 0.2800 | 0.5685 | 0.6620 | 0.7325 |

Query latency: avg 21.7ms / p50 21.5ms / p95 22.0ms (single-query, exact
search over 109K vectors) - a real baseline, not optimized.

**A real implementation bug found and fixed during this step:** the
`Retriever`'s query-time model-match check read
`metadata["embedding_info"]["model_name"]`, but the field actually written
(matching `configs/models.yaml`'s own schema) is `"name"` - this raised
`KeyError` at the first real `retrieve()` call, caught only when actually
running retrieval evaluation (the synthetic-fixture unit tests didn't
exercise this code path). Fixed, and two regression tests added
(`test_ensure_model_loaded_reads_embedding_info_with_correct_key`,
`test_ensure_model_loaded_rejects_mismatched_model`) using mocks so the
real model is never needed to catch this class of bug again.

**Tests:** 32 new tests (`tests/test_retrieval.py`) covering corpus_id
determinism, corpus validation (duplicate/empty/mismatched IDs),
write/load roundtrip, real-data corpus construction and determinism
(guarded), real BGE embedding dimension/finite/normalization/determinism
(guarded), FAISS build/save/load/reload-consistency, index validation,
the `Retriever` API (`get_by_corpus_id`, missing-id rejection, top-k
overflow safety, the embedding-info key-mismatch regression above), and
retrieval-evaluation recall computation on synthetic fixtures. A static
AST-based check confirms the retrieval package never imports the
`ragtruth`/`truthfulqa` dataset modules. Full suite: 213/213 passed.

This is retrieval infrastructure only - no reranking integrated yet, the
verifier was not retrained or modified, and RAGTruth test/TruthfulQA
remain completely untouched, reserved for later evaluation steps.

### ClaimGuard BGE reranker integration and evaluation (Step 14, 2026-09-03)

Step 13 built FAISS retrieval infrastructure in isolation; Step 14 adds the
second retrieval-side component from RESEARCH.md's methodology (a
cross-encoder reranker on top of FAISS candidates) and evaluates it
independently of both the verifier and any full pipeline integration.

**Reranker model - unchanged, verified, not silently swapped:**
`BAAI/bge-reranker-large` (Step 5D's already-selected/smoke-tested model),
`sentence_transformers.CrossEncoder`, fp16, 559,891,457 parameters
(matches Step 5D exactly). **Score direction re-verified at runtime, not
assumed to still hold:** `activation_fn` is `Sigmoid` (probability-like,
[0,1]), confirmed empirically on both a real toy example (the direct
answer ranked highest) and the real evaluation data - higher score =
more relevant.

**Two-stage pipeline:** `query -> BGE embedding -> FAISS top-20 -> BGE
reranker -> reranked top-5` (config: `configs/reranker_baseline.yaml`,
`retrieval_top_k=20`, `rerank_top_n=5`, one controlled baseline, no
parameter sweep). The reranker only ever scores the FAISS candidate set
it is given - it never searches the corpus itself. Every output record
preserves full provenance (`corpus_id`, `text`, `page_id`, `sentence_id`,
`corpus_version`) plus both the original FAISS score/rank and the new
reranker score/rank.

**Smoke test (real GPU) passed cleanly:** correct output shape, all
finite, provenance preserved exactly, fully deterministic across two
independent calls (`max_abs_diff = 0.0`), empty-candidate handling
correct, peak GPU memory 1.06GB, GPU cleanly released after.

**Evaluation - the SAME 2,000-claim eval set as Step 13** (cross-checked:
identical example_id set, not just the same construction parameters):

| Metric | FAISS-only | FAISS+reranker | Δ absolute | Δ relative |
|---|---|---|---|---|
| Recall@1 | 0.2800 | 0.2445 | **−0.0355** | **−12.7%** |
| Recall@5 | 0.5685 | 0.5905 | +0.0220 | +3.9% |
| Recall@10 | 0.6620 | 0.7010 | +0.0390 | +5.9% |
| Recall@20 | 0.7325 | 0.7325 | 0.0000 | 0.0% (exact invariant, verified: same 20 candidates, only reordered) |

**Reranking made Recall@1 measurably WORSE on this eval set - reported
honestly, not tuned away.** Rank-shift diagnostics explain the mixed
picture: of the 1,465 claims with gold evidence in the FAISS top-20
(73.25%), reranking improved 527, left 960 unchanged, and worsened 513;
mean rank improved slightly (3.91 → 3.46) but median stayed flat (2 → 2).
Manual inspection of worsened examples shows a real, sensible pattern: the
reranker sometimes prefers a page's broad topical/definitional sentence
over the specific, narrower factual sentence FEVER's annotators actually
cited (e.g. for "Machu Picchu was built with brick walls," FAISS correctly
ranked the exact evidence sentence first; the reranker demoted it to rank
11 in favor of Machu Picchu's general introductory sentence) - a
cross-encoder trained on general relevance judgment does not automatically
match FEVER's narrow single-sentence annotation convention.

**Latency:** FAISS avg 28.1ms / p50 23.0ms / p95 31.4ms; reranker (20
candidates/query) avg 24.7ms / p50 23.5ms / p95 30.1ms; end-to-end avg
52.9ms / p50 49.1ms / p95 57.8ms. Peak GPU memory during the full
2,000-query evaluation: 1.84GB.

**Tests:** 19 new tests (`tests/test_reranking.py`) covering scoring,
reranking (shape/finiteness/direction/determinism/top-N/empty/duplicate
candidates/provenance/rank-tracking), the two-stage pipeline wrapper, config
loading, and (guarded, real model) score-direction and correctness checks.
Full suite: 232/232 passed.

**The verifier was NOT called or modified in this step** - retrieval and
reranking quality were evaluated in complete isolation from claim
verification, exactly as planned; that integration is a separate future
controlled step.

### Retrieval + reranking + binary verifier integration (Step 15, 2026-09-03)

Steps 13-14 validated retrieval and reranking in isolation; Step 15 is the
FIRST integrated experiment combining Step 13's FAISS retrieval, Step 14's
reranking, and Step 12's binary verifier - explicitly NOT the final
ClaimGuard end-to-end evaluation (no generator, no correction loop).

**Architecture:** `query -> BGE embedding -> FAISS top-20 -> BGE reranker
-> reranked top-5 -> binary DeBERTa verifier (per candidate) ->
aggregation (max entailment probability) -> selected evidence`. Per
Step 14's finding that reranker top-1 is *weaker* than FAISS top-1 on
Recall@1, the pipeline never assumes reranked-rank-1 is final: it verifies
all 5 top-reranked candidates and lets the verifier's own entailment
probability select evidence (Strategy C), while separately retaining
FAISS-top-1 (Strategy A) and reranker-top-1 (Strategy B) for direct
comparison.

**All three models load simultaneously** (peak GPU 2.65GB across the full
2,000-query run) - no sequential loading/offloading needed at this scale,
measured and documented rather than assumed necessary.

**Evaluation set:** the IDENTICAL 2,000-claim FEVER-train-derived eval set
from Steps 13/14 (cross-checked: identical example_id set). Gold
evidence/labels are used only for evaluation, strictly after pipeline
execution - `run_pipeline()`'s only per-query input is the raw claim text
(structurally verified by a dedicated test inspecting its signature).

**Consistency checks (both PASS, confirming no evaluation drift):**
retrieval Recall@{1,5,10,20} = {0.2800, 0.5685, 0.6620, 0.7325} and
reranked Recall@{1,5,10,20} = {0.2445, 0.5905, 0.7010, 0.7325} exactly
reproduce Steps 13/14.

**Evidence-selection strategy comparison (gold-hit rate):**

| Strategy | Hit rate |
|---|---|
| A: FAISS top-1 | 0.2800 |
| B: Reranker top-1 | 0.2445 |
| **C: Verifier-selected (max entailment among top-5)** | **0.3180** |

Letting the verifier choose among the reranked top-5 (by its own
entailment probability) selects gold evidence more often than either
naive top-1 strategy - the first concrete evidence that preserving
multiple candidates + verifier-based selection is the right design
response to Step 14's reranker-top-1 weakness.

**End-to-end classification (using PIPELINE-selected evidence, not gold -
positive class = entailment/"supported"):** accuracy 0.8860; entailment
precision/recall/F1 = 0.8763/0.9873/0.9285; contradiction
precision/recall/F1 = 0.9389/0.5828/0.7192. **Not comparable to Step 12's
0.9585 gold-premise macro F1** - this number includes retrieval/reranking
error propagation by construction. The REFUTES/contradiction recall gap
(0.58) is explained by the threshold diagnostics below.

**Threshold diagnostics (informational, NOT tuned):**

| Group | n | Mean entailment prob | Median |
|---|---|---|---|
| Gold evidence, true=SUPPORTS | 1,024 | 0.9826 | 0.9987 |
| Gold evidence, true=REFUTES | 252 | 0.0807 | 0.0217 |
| Non-gold evidence (either label) | 8,724 | 0.6968 | 0.9758 |

The verifier behaves exactly as intended on GOLD evidence (near-1.0 for
genuine support, near-0.0 for genuine refutation). But **non-gold
evidence's median entailment probability is 0.9758** - nearly as high as
genuine gold support - meaning the verifier is frequently overconfident
that superficially-related-but-wrong retrieved text "supports" the claim.
This directly explains the depressed contradiction-class recall: for many
REFUTES claims, the pipeline retrieves no genuinely refuting sentence
among its top-5, and the verifier then over-readily calls a topically
adjacent (but non-refuting) candidate "supported" instead.

**Latency (2,000 queries):** embed+FAISS avg 29.7ms; reranker avg 26.4ms;
verifier avg 102.9ms (5 candidates/query); end-to-end avg 159.1ms / p50
147.6ms / p95 199.3ms. One-time model load: 10.7s for all three models.

**Tests:** 18 new tests (`tests/test_verification.py`) covering pipeline
ordering/provenance/candidate-count, aggregation correctness and
determinism, structural no-gold-access verification, binary label
semantics, and (guarded, tiny-checkpoint and real-checkpoint) verifier
loading/scoring. A real test-fixture bug was found and fixed during this
step's own development (a locally-built tiny model's embedding table was
too small for the real tokenizer's vocabulary, causing an `IndexError` -
fixed by sizing the fixture's vocab to `len(tokenizer)`). Full suite:
250/250 passed.

This is the first integrated retrieval + reranking + verifier experiment.
It is NOT the final ClaimGuard end-to-end evaluation - no generator, no
correction loop, no RAGTruth/TruthfulQA evaluation.

### Verifier calibration and evidence decision policy (Step 16, 2026-09-03)

Step 15 found the verifier is frequently overconfident on non-gold
evidence (median entailment prob 0.9758). Step 16 formally characterizes
this behavior and asks whether a deterministic (non-learned) downstream
decision policy - SUPPORTED / CONTRADICTED / ABSTAIN - can be built on
top of it, without retraining the verifier. **The binary verifier is
unchanged; Step 16 adds only a downstream evidence-decision policy**
(`src/claimguard/decision/`).

**4-way score distribution (splitting "non-gold" by the claim's true
label, refining Step 15's combined 3-group breakdown):**

| Group | n | Mean | Median |
|---|---|---|---|
| Gold evidence, true=SUPPORTS | 1,024 | 0.9826 | 0.9987 |
| Gold evidence, true=REFUTES | 252 | 0.0807 | 0.0217 |
| Non-gold evidence, SUPPORTS claims | 6,471 | 0.8556 | **0.9940** |
| Non-gold evidence, REFUTES claims | 2,253 | 0.2407 | 0.0337 |

**New finding not visible in Step 15's combined non-gold statistic:** the
overconfidence problem is concentrated almost entirely in non-gold
evidence retrieved for SUPPORTS claims (median 0.9940, nearly
indistinguishable from genuine gold-SUPPORTS evidence at 0.9987).
Non-gold evidence for REFUTES claims scores low (median 0.0337), much
closer to genuine gold-REFUTES evidence. The verifier's failure mode is
one-sided: it readily calls topically-related-but-wrong evidence
"supporting", but rarely calls it "refuting".

**Calibration (gold-evidence candidates only - see
`claimguard.decision.calibration` for why non-gold candidates have no
legitimate ground-truth "correct" label): ECE = 0.0092, Brier = 0.0178.**
Both are low - **the verifier is well-calibrated on cases where ground
truth exists.** The practical problem is therefore not raw
miscalibration but a **discrimination failure**: the verifier cannot
separate "genuinely correct evidence" from "topically similar but wrong
evidence" - both score confidently high.

**Candidate-level "accepted support" threshold analysis** (predicted
positive = entailment_probability ≥ t; actual positive = gold evidence
for a SUPPORTS claim), swept at t ∈ {0.50, 0.60, 0.70, 0.80, 0.90, 0.95}:
precision stays in **0.14-0.17** and recall stays **≥0.97** across the
entire range - raising the threshold to 0.95 barely moves precision,
because non-gold-for-SUPPORTS-claims evidence clusters at the same
near-1.0 confidence as genuine gold evidence. **No single entailment
threshold meaningfully separates correct from incorrect evidence.**

**Deterministic policy outcomes** (`EvidenceDecisionPolicy`, thresholds
swept 0.50-0.95): false-support rate stays **62-65%** across the entire
threshold range, and abstention barely rises (0/2,000 at t=0.5 to
43/2,000 at t=0.95). **Raising the threshold does not buy safety** - it
only shaves off the least-confident errors while leaving the bulk of
overconfident false positives untouched.

**Evidence-consistency (margin) analysis** is the one lever that helped
meaningfully: queries where the top-1 entailment probability beats the
runner-up by more than the median margin have selection precision
**0.4607** vs **0.2765** for low-margin queries (a ~67% relative
improvement) - still far from a "safe" policy, but the most promising
signal found.

**Honest conclusion (per explicit instruction not to force a policy that
doesn't exist): no single-threshold deterministic policy on entailment
probability alone is safe for production evidence selection.** A
deterministic `EvidenceDecisionPolicy` (SUPPORTED/CONTRADICTED/ABSTAIN)
was still implemented, correctly, as the required interface - but its
default configuration is explicitly labeled a *characterization
baseline*, not a validated safe policy. Margin-based consistency
filtering is the most promising direction for follow-up work, not yet
sufficient on its own.

Artifacts: `data/processed/integration/calibration_results.json`,
`data/processed/integration/decision_policy_results.json`,
`configs/decision_policy.yaml`. Tests: 33 new
(`tests/test_decision.py`). Full suite: 283/283 passed.

### Qwen3-8B generator integration (Step 17, 2026-09-03)

Integrated `Qwen/Qwen3-8B` (the approved primary generator per
`CLAIMGUARD_MODEL_SELECTION.md`/`configs/models.yaml`, no ambiguity) as a
standalone, independently-callable answer-generation component
(`src/claimguard/generation/`). Strictly the generator box in the future
`query -> generator -> candidate answer -> [Step 18] retrieval ->
reranking -> verifier -> decision -> correction` architecture - **no
correction/regeneration loop, no retrieval/reranking/verifier
integration implemented here.**

**Loading:** reuses the exact pattern already validated in Step 5A's
`scripts/smoke_test_qwen.py` (`AutoModelForCausalLM`/`AutoTokenizer`,
BF16, `device_map="auto"`, `low_cpu_mem_usage=True`) - not re-derived.
Model identity comes from `configs/models.yaml`'s `generator.primary`
section (never hard-coded); `configs/generator_baseline.yaml` holds
generation-specific settings and is tested to stay consistent with
`models.yaml` (no silent drift).

**Chat template / determinism:** uses the tokenizer's own
`apply_chat_template` (never a manually-assumed prompt format), with
Qwen3's "thinking" mode explicitly disabled (`enable_thinking=False`) -
ClaimGuard never extracts or stores hidden reasoning, only the final
answer text. A `TypeError` fallback (retry without `enable_thinking`)
reuses Step 5A's validated handling for tokenizer/template version
differences. Deterministic decoding (`do_sample=False`, fixed seed)
confirmed to produce byte-identical output across two repeated real-model
calls.

**Smoke test results (real GPU, real model, PASS):** 8.19B params
(matches Step 5A exactly), BF16, loaded in 9.6s (weights already cached
locally). Three hand-written prompts (short factual, longer explanatory,
structured list) all produced non-empty, prompt-free, sensible answers -
engineering validation only, not a research/hallucination metric. Peak
allocated VRAM 15.29GB (matches Step 5A's 15.27GB). GPU resources
explicitly released after testing; confirmed via `nvidia-smi
--query-compute-apps` (not just the script's own free-VRAM print, which
reflects the SHARED GPU's total free memory and was misleading on its
own since `exp42`'s concurrent training job's memory usage fluctuates
independently) that zero GPU memory remained attributed to the generator
process once it exited.

**Generator knows nothing about datasets:** `generate()`/`load_generator()`
have no FEVER/HaluEval/RAGTruth/TruthfulQA/gold-evidence/hallucination-span
parameter (structurally verified via `inspect.signature`); no such module
is importable from `src/claimguard/generation/` (AST-checked).

Artifacts: `data/processed/generation/generator_smoke_results.json`,
`configs/generator_baseline.yaml`. Tests: 24 new
(`tests/test_generation.py`, includes one guarded real-Qwen3-8B test
class). Full suite: 307/307 passed (283 pre-existing + 24 new).

**Explicitly not done in this step:** no fine-tuning, no LoRA/PEFT, no
quantization (BF16 only, per the approved protocol), no correction/
regeneration loop, no RAGTruth evaluation, no TruthfulQA evaluation, no
retrieval/reranker/verifier/decision-policy changes.

### ClaimGuard correction/regeneration loop (Step 18, 2026-09-03)

Implemented the bounded orchestration connecting every prior component:

    query -> Qwen3-8B -> candidate answer -> retrieve -> rerank -> verify
        -> Step 16 decision policy -> ACCEPT / CORRECT (bounded) / ABSTAIN

`src/claimguard/correction/` (using the placeholder package created back
in Step 4's scaffolding). **This is an ENGINEERING/INTEGRATION validation
step, not the final hallucination-detection evaluation** - RAGTruth
(Step 19) and TruthfulQA (Step 20) remain untouched.

**Terminology mapping (documented, not invented):** Step 16's
`EvidenceDecisionPolicy` still outputs SUPPORTED/CONTRADICTED/ABSTAIN
unchanged. Step 18 relabels these, at the integration layer only, as
ACCEPT/CORRECT/ABSTAIN: `SUPPORTED->ACCEPT`, `CONTRADICTED->CORRECT`,
`ABSTAIN->ABSTAIN`. Step 16's actual decision rule (thresholds, margin
logic) is never modified or retuned.

**Bounded loop:** `max_correction_attempts=2` (conservative engineering
baseline, never tuned against evaluation data) - terminates after at
most 3 total generations (1 initial + 2 corrections). If the budget is
exhausted while the policy still says CORRECT, the final decision is
explicitly **ABSTAIN**, never a silently-accepted unverified answer.

**What gets retrieved/verified at each attempt:** the CURRENT CANDIDATE
ANSWER text, never the raw user question - the user's original question
is preserved separately and used only for the first generation call and
for building later correction prompts. Every corrected answer triggers a
completely fresh retrieval -> reranking -> verification pass; no stale
verifier result is ever reused across attempts.

**Real end-to-end smoke test (3 hand-written questions, PASS):** all
three reached ACCEPT on the first attempt - no CORRECT/ABSTAIN was
naturally triggered by these simple, well-covered factual questions
against the FEVER-wiki-derived corpus. CORRECT-path, ABSTAIN-path, and
max-attempts-termination behavior were validated via 10 synthetic mocked
fixtures instead (`tests/test_correction.py`), not forced in the real
smoke run. Peak VRAM with all four models (embedder + reranker + verifier
+ Qwen3-8B) loaded simultaneously: 17.81GB - well within budget and
consistent with `CLAIMGUARD_MODEL_SELECTION.md`'s ~23-25GB estimate.
Zero GPU memory growth measured across 3 repeated correction-loop calls.
Deterministic repeatability confirmed: identical `final_decision`/
`final_answer`/attempt sequence across two full-loop runs of the same
query.

**Multi-claim limitation (documented, not silently assumed away):** the
correction loop preserves Step 16's answer/evidence-level abstraction -
a candidate answer is checked as a single unit against retrieved
evidence, not decomposed into independently-verified atomic claims. No
new claim-extraction model was introduced.

Artifacts: `data/processed/correction/correction_smoke_results.json`,
`configs/correction_baseline.yaml`. Tests: 25 new
(`tests/test_correction.py`). Full suite: 332/332 passed (307
pre-existing + 25 new; 2 of the pre-existing real-Qwen3-8B-guarded tests
skipped gracefully during one run due to a transient VRAM dip from the
shared GPU's concurrent unrelated job - not a regression, confirmed by
the skip reason logged and the same tests passing cleanly in Step 17).

**Explicitly not done in this step:** Qwen3, DeBERTa, the BGE reranker,
and the FAISS retrieval corpus/index all remain frozen; no fine-tuning;
no RAGTruth evaluation; no TruthfulQA evaluation; no learned fusion; no
decision-policy retuning; no new claim-decomposition model.

### RAGTruth end-to-end evaluation (Step 19, 2026-09-04/05)

**The first formal end-to-end evaluation of the complete, FROZEN
ClaimGuard system** against the reserved 2,700-response RAGTruth TEST
split (450 source items x 6 models, verified empirically - not assumed -
against `claimguard.datasets.ragtruth`). Nothing was tuned based on
these results; all thresholds/prompts/configuration were frozen before
the run and were not altered afterward.

**Conditions:** B (fresh Qwen3-8B generation from RAGTruth's own prompt,
then the full correction loop), C (original RAGTruth response, single
verify-only pass, `max_correction_attempts=0`), D (original response,
full correction loop). A, as literally specified, is procedurally
identical to D and its numbers are reported as equal to D's rather than
wastefully recomputed. C/D reuse the Step 18 correction loop UNCHANGED
via one small, backward-compatible extension (`initial_candidate`,
default `None` - lets a caller start the loop from an already-existing
answer instead of generating one; `max_correction_attempts=0` combined
with it yields a single verify-only pass) - Step 18's decision rule
itself was not touched.

**Critical scoping finding, anticipated before running the evaluation:**
ClaimGuard's retrieval corpus (Step 13) is built exclusively from FEVER's
Wikipedia evidence pages. RAGTruth's actual source documents (news
articles, Yelp-style business listings, QA passages) are **not in this
corpus at all**. Retrieval never returns zero candidates (mean 5
evidence items per query, exactly `rerank_top_n`), but for most RAGTruth
responses the retrieved evidence is topically unrelated to the actual
source material - a known architectural mismatch, not a bug, and the
primary lens for interpreting everything below.

**Headline result (Condition B, the true end-to-end generate+verify+
correct condition, n=2,700, gold prevalence 34.9% hallucinated):**
attempt-0 detection accuracy **55.9%**, macro F1 **0.559** (hallucinated
F1 0.560, not-hallucinated F1 0.558). The trivial always-predict-
majority-class baseline (always guess "not hallucinated", at 65.1%
prevalence) would itself score **65.1% accuracy** - ClaimGuard's 55.9%
is **below** that trivial baseline. After the correction loop runs,
final-outcome accuracy rises slightly to 58.0% (macro F1 0.580) - a
small, real improvement from correction, but still below the trivial
baseline.

**Striking finding: ABSTAIN never fires - 0% of all 2,700 responses,
across all three conditions.** Every decision is ACCEPT or CORRECT
(Step 16's SUPPORTED/CONTRADICTED), never Step 16's own ABSTAIN. This
sharply confirms Step 16's own finding (verifier confidence clusters
near the extremes, rarely lands in the uncertain middle) generalizes
out-of-domain: on RAGTruth, the verifier is *always* confident, just
frequently confidently wrong given irrelevant evidence.

**Correction rarely resolves a CORRECT verdict:** of responses where
correction was attempted, only 6.1% (Condition B) / 18.4% (Condition D)
ended up ACCEPT within the 2-attempt budget - 61.3% (B) / 62.1% (D) of
*all* responses hit `max_attempts_reached`. Consistent with the corpus
mismatch: regenerating the candidate answer doesn't change which
(irrelevant) evidence gets retrieved, so the verifier's verdict rarely
flips.

**Task-type breakdown (Condition B) is the clearest evidence for the
corpus-mismatch explanation:** Data2txt (business-listing data, zero
FEVER-wiki overlap) - macro F1 **0.398**, not-hallucinated recall a mere
**0.9%** (ClaimGuard flags 98.7% of Data2txt responses as CORRECT
regardless of whether they're actually hallucinated). Summary (news
articles) - macro F1 **0.372**, accuracy 37.3%, *worse than the trivial
77%-accuracy always-predict-not-hallucinated baseline* for that subset.
QA - the best of the three (macro F1 **0.496**), plausibly because some
QA content brushes against general-knowledge territory FEVER-wiki
covers, but hallucinated-class recall is still only 23.8%.

**Source-model breakdown** shows the same pattern from a different
angle: gpt-4-0613/gpt-3.5-turbo-0613 (genuinely low hallucination
prevalence, ~9-10%) get ClaimGuard's *worst* macro F1 (~0.39-0.42) -
the system over-flags the vast majority of their actually-good responses
(gpt-4: hallucinated-recall 95%, but precision only 13.6%). The weaker,
higher-hallucination-rate open models (mistral-7B-instruct,
llama-2-\*-chat, prevalence 38-56%) score notably better (macro F1
0.59-0.65) - not because ClaimGuard discriminates better on them, but
because its systematic "flag as unsupported" bias happens to numerically
align with truth more often when there's genuinely more to flag.

**Span-level evaluation: not computed** (documented limitation, not
fabricated) - Step 16's decision policy operates at the whole-answer
level with no mechanism to attribute a verdict to a specific substring;
inventing a span-alignment heuristic would misrepresent what ClaimGuard
actually does.

**Engineering robustness:** 0 inference failures and 0 zero-evidence
retrievals across all 8,100 correction-loop calls (2,700 responses x 3
conditions). Determinism reconfirmed on real RAGTruth data (identical
across two full reruns of a 10-response subset). Peak GPU memory 18.39GB
- unchanged from Step 18's smoke test, confirming no leak across a
~15.3-hour, 2,700-response run.

**Honest scientific conclusion:** ClaimGuard, run end-to-end and
unmodified against RAGTruth's reserved test split, does **not**
demonstrate reliable hallucination detection - its accuracy is at or
below trivial baselines on 2 of 3 task types, and its one clear
behavioral signature (near-zero ABSTAIN, systematic over-flagging) is a
direct, already-anticipated consequence of evaluating a FEVER-Wikipedia-
grounded verifier against source documents the retrieval corpus was
never built to cover. The correction loop provides a small, consistent,
but modest improvement over verify-only, never a fix for the underlying
evidence mismatch. **This is a poor result, reported honestly, not spun
as a success** - and it isolates WHY (retrieval corpus scope) rather
than leaving the cause ambiguous.

Artifacts: `data/processed/evaluation/ragtruth/{ragtruth_eval_results.json,
ragtruth_response_results.jsonl, ragtruth_error_analysis.json,
ragtruth_eval_manifest.json}`. Tests: 36 new
(`tests/test_ragtruth_evaluation.py`). Full suite: 368/368 passed.

### TruthfulQA evaluation (Step 20, 2026-09-05)

Evaluated the complete, FROZEN ClaimGuard system against all **790** TruthfulQA
questions (**37** categories, empirically verified - not assumed - against
`claimguard.datasets.truthfulqa`; no train/test split exists or was invented).
Nothing was tuned based on these results.

**Two independent, deliberately separate tracks**, because TruthfulQA supports
one rigorously and does not support the other without inventing something:

1. **MC track** (rigorous, standard, ground-truth-backed): score every
   mc0/mc1/mc2_targets choice by its LOG-LIKELIHOOD under the frozen Qwen3-8B
   model (one new function, `score_choice_log_likelihood` - a teacher-forced
   forward pass, not generation, not a weight change, not a new classifier),
   then compute MC1/MC2/MC0 exactly per TruthfulQA's own published
   definitions. This measures **Qwen3's own parametric calibration** - no
   retrieval, reranking, verification, or correction is involved at all.
2. **Free-form pipeline track**: run the SAME frozen `run_correction_loop`
   used in Step 19 (unchanged), once per question, extracting Condition A
   (baseline Qwen3 answer), B (ClaimGuard's attempt-0 verification decision),
   C (final post-correction outcome). Reports PROCESS behavior only
   (ACCEPT/CORRECT/ABSTAIN rates, retrieval diagnostics, latency) - **not** a
   truthfulness judgment.

**Deliberate, disclosed protocol gap:** no approved automatic judge exists in
this project for scoring FREE-FORM generated text against TruthfulQA's
human-annotated answer sets (no human raters, no fine-tuned GPT-judge model,
and embedding/string-similarity scoring would be exactly the fabricated
"semantic similarity threshold called truthfulness" the instructions
explicitly forbid). This is reported honestly as a limitation, not papered
over with an invented heuristic - see
`claimguard.evaluation.truthfulqa_eval`'s module docstring.

**MC results (n=790, Qwen3-8B's own calibration):** MC1 accuracy **34.05%**,
MC2 accuracy **55.05%**, MC0 accuracy **44.18%**. A modest, unremarkable
result - well above trivial 1-in-several-choices random guessing but far
from strong calibrated truthfulness. **Caveat already on record in
`CLAIMGUARD_MODEL_SELECTION.md` before this evaluation ran**: TruthfulQA is
an old, ubiquitous public benchmark plausibly present in Qwen3's pretraining
data, so this score may partly reflect memorization rather than genuine
calibration - not something this evaluation can distinguish.

**Free-form pipeline process behavior (n=790) - a striking CONTRAST with
Step 19's RAGTruth numbers:** attempt-0 ACCEPT rate 80.0% (vs RAGTruth's
34.7%), ABSTAIN rate **0.0%** (same "always confident" pattern as Step 19 and
Step 16), correction attempted on 20.0% of questions, but
**correction-success-among-attempted is 76.6%** - dramatically higher than
RAGTruth's 6.1-18.4%. Final accept rate 95.3%; only 4.7% hit
max-attempts-reached (vs RAGTruth's 61.3%). **This does NOT mean ClaimGuard
is "more truthful" on TruthfulQA** - no truthfulness ground truth was
computed for this track (see the protocol gap above). It means the verifier
more often accepts TruthfulQA's short, general-knowledge-style answers
(original or regenerated) than RAGTruth's long, domain-specific documents -
plausibly because short answers to well-known questions have a higher
chance of landing near SOME plausible-looking Wikipedia sentence in the
frozen FEVER corpus, not because they are more accurate.

**Category breakdown** (37 categories) shows wide MC1 variance: Stereotypes
0.875, Logical Falsehood 0.857, Statistics 0.80 (n=5, small) vs Confusion:
Other/Confusion: People/Finance at MC1=0.0 (Qwen3 gets every question wrong
in these categories). Small-n categories (<15) are explicitly flagged, not
overinterpreted.

**Retrieval diagnostics:** 0 zero-evidence questions, mean 5.0 evidence/query
- identical pattern to Step 19. TruthfulQA's 37 categories (law, fiction,
proverbs, paranormal, etc.) have no guaranteed FEVER-Wikipedia coverage;
successful retrieval is not interpreted as proof of relevant evidence.

**Engineering robustness:** 0 inference failures across 790 questions (one
pipeline pass each, plus ~2-8 MC choices scored per mc0/mc1/mc2 set).
Reproducibility reconfirmed on real TruthfulQA data (identical across two
full reruns of a 10-question subset). Total runtime 5,092.5s (1.41 hours) -
790 x cheaper-than-RAGTruth per-question cost, since MC scoring needs only
short forward passes and most questions accept on the first pipeline pass.
Peak GPU memory 17.93GB, consistent with Steps 18-19.

Artifacts: `data/processed/evaluation/truthfulqa/{truthfulqa_eval_results.json,
truthfulqa_response_results.jsonl, truthfulqa_error_analysis.json,
truthfulqa_eval_manifest.json}`. Tests: 27 new
(`tests/test_truthfulqa_evaluation.py`). Full suite: 395/395 passed.

### Controlled ablations and failure-source analysis (Step 21, 2026-09-09/10)

A DIAGNOSTIC experiment, not a tuning stage - no threshold/prompt/generation-
parameter search anywhere in this step. Isolates which existing, FROZEN
components contribute to Step 19's below-baseline RAGTruth result, via
controlled component removal (never hyperparameter search).

**Ablation matrix:** A (original frozen ClaimGuard), B (no-reranker - FAISS
order taken directly), C (FAISS top-1 direct - single candidate only), D
(no-correction), E (verify-only, decision isolated from the generator). A/B/C
required NEW inference (a 300-response deterministic RAGTruth subset, first
300 by `response_id`) via one small, additive extension to
`run_correction_loop` (`evidence_selection_mode: "reranked"|"no_reranker"|
"faiss_top1"`, default unchanged - all 411 pre-existing tests still pass).
D and E are FUNCTIONALLY IDENTICAL to data Step 19 already computed for all
2,700 responses (no-correction = attempt-0 detection metrics; verify-only =
Step 19's `C_verify_only`) - reused directly, zero new inference, per the
explicit "do not duplicate computation" instruction.

**Cross-validation (300/300 matched, 100%):** Condition A rerun on the same
300 response_ids as Step 19 produced BYTE-IDENTICAL final decisions to
Step 19's stored full-run results - strong confirmation of determinism
across independent runs, not a new finding on its own.

**Headline finding: the reranker changes WHICH evidence is selected but NOT
the downstream decision.** Evidence-set changed in 100% of responses between
reranked and no-reranker conditions, yet A and B's detection metrics are
IDENTICAL to many decimal places (accuracy 0.5867, macro F1 0.5813,
hallucinated recall 0.8974, both n=300). Removing the reranker entirely made
no measurable difference to classification quality - consistent with the
Step 19 corpus-mismatch finding: when the retrieval corpus has no genuinely
relevant content for the query domain, reordering irrelevant candidates
doesn't change how confidently (and wrongly) the verifier scores them.
**H2 (reranking does not reliably improve evidence selection) is
supported; reranking is not the bottleneck, but it is not helping either.**

**Reducing the candidate pool DOES hurt:** Condition C (FAISS top-1, only
ONE candidate reaches the verifier) scored notably worse - accuracy 0.51 vs
0.5867, macro F1 0.487 vs 0.581, not-hallucinated recall 0.246 vs 0.388
(more false positives). Having 5 candidates to pick the max-entailment
evidence from helps, independent of whether those 5 are reranked or raw
FAISS order.

**Verifier overconfidence persists on RAGTruth, unrelated to correctness**
(full 2,700, reused): mean decision confidence is 0.83-0.98 across ALL FOUR
outcome groups (correct accepts, WRONGLY-accepted hallucinations, WRONGLY-
flagged clean answers, correctly-flagged hallucinations) - the verifier's
own confidence does not distinguish right from wrong decisions AT ALL. Of
the 300-response subset, **115 (38.3%) were confidently (>=90%) wrong.**
**H3 (DeBERTa is overconfident on irrelevant evidence) is strongly
supported.**

**Zero-abstain confirmed across every condition tested** (A/B/C on the
subset, D/E on the full 2,700): the decision policy NEVER abstains at
attempt 0, regardless of evidence-selection method, correction status, or
candidate source. **H4 (insufficient abstention) is strongly supported.**

**Correction harm is substantial and reproduces Step 19's proportion:** 101
of 300 (33.7%) Condition-A responses show harmful correction (a genuinely
clean answer ending ABSTAIN) - matching Step 19's full-2700 rate (34.2%)
closely. Correction success (hallucinated -> accepted) remains rare: 8.3%
of attempted corrections (A) vs 2.8% (B, no-reranker) - correction dynamics
are somewhat WORSE without reranking, a secondary, smaller effect.
**H5 (correction causes substantial harmful changes) is supported.**

**Not identifiable from the current evaluation:** retrieval RELEVANCE
failure (evidence returned but doesn't actually support the response) -
RAGTruth provides no per-evidence gold-relevance annotation analogous to
FEVER's gold evidence sets, so this cannot be measured without inventing a
semantic relevance metric, which this step explicitly does not do.

**Main scientific finding:** the evidence points to **H1 (FEVER/Wikipedia
corpus is the dominant end-to-end bottleneck) and H3/H4 (verifier
overconfidence + zero-abstention) as the best-supported, interacting failure
sources (H6)** - not H2 (reranking is a secondary/non-factor) and not H5
alone (correction harm is real but is a downstream CONSEQUENCE of upstream
overconfidence, not an independent root cause). No component should be
"blamed" in isolation from the others.

Artifacts: `data/processed/evaluation/ablations/{ablation_results.json,
ablation_comparison.csv, ablation_error_analysis.json,
ablation_run_manifest.json}`. Tests: 16 new (`tests/test_ablations.py`).
Full suite: 411/411 passed.

## Ablation Plan

**Update (Step 21):** `ablation_no_reranker` and `ablation_no_correction`
(plus two new conditions not in the original plan - FAISS-top-1-direct and
verify-only) were executed - see the Step 21 section above.
`ablation_no_bm25`/`ablation_no_dense`/`ablation_no_verifier` remain
`not_run`: no BM25 component was ever built (dense/FAISS-only retrieval was
the actual implemented design, see Step 13), and removing the verifier
entirely is out of Step 21's explicit scope. This section's original text is
preserved below as the historical planning record.

Placeholders defined in `configs/experiments.yaml` (all `status: not_run`):

- `ablation_no_bm25` — remove BM25, dense retrieval only.
- `ablation_no_dense` — remove dense/FAISS retrieval, BM25 only.
- `ablation_no_reranker` — remove the reranking stage.
- `ablation_no_verifier` — remove claim verification.
- `ablation_no_correction` — remove self-correction (verify only, no rewrite).

Each isolates one pipeline component's contribution relative to
`claimguard_full`.

## Expected Contributions

- An empirical comparison of hallucination rates across
  vanilla-LLM / standard-RAG / RAG+verification / full-ClaimGuard conditions,
  on a naturalistic (non-benchmark-memorizable) test set.
- A component-level ablation showing which stage(s) of the pipeline
  contribute most to hallucination reduction.
- A cross-model-family robustness check (Qwen3-8B vs. Llama-3.1-8B-Instruct)
  of whether the effect generalizes.

These are anticipated contributions, contingent on results — **NOT
established findings**.

## Limitations

- Single-GPU, shared-hardware constraint limits model scale (7B–14B
  generators only) and concurrent-experiment throughput.
- FEVER/TruthfulQA contamination risk limits which datasets can support a
  "clean" end-to-end effectiveness claim (see Leakage Concerns).
- Correction quality depends on the verifier's precision; verifier errors
  will propagate into unnecessary or incorrect corrections — not yet
  measured.
- No human evaluation is currently planned; automated metrics only, which
  may not fully capture answer quality/fluency tradeoffs.
- Results section: **NOT RUN.**
