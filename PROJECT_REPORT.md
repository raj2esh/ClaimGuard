# ClaimGuard — Project Report

Ongoing log of what was done at each step and why. Appended after each new phase.

---

## Step 0 — Project initialization (2026-08-31)

**What:** Created a local placeholder directory (`~/hallucination-detection-correction`) and,
once the remote GPU server was identified as the real working environment, a remote project
directory `~/RJ/ClaimGuard/` on `<GPU_USERNAME>@<REMOTE_SERVER>`.

**Why:** ClaimGuard needed a home before any environment or code work could start. The remote
server was chosen because it's the same machine already hosting the SAM-RNet project and has
the only GPU available (a local Intel-integrated-graphics-only machine can't do this work) —
`~/RJ/` is the established project-root convention on that server.

---

## Step 1 — Environment audit (2026-08-31)

**What:** Read-only inspection of both the local machine and the remote server: Python/PyTorch
versions, CUDA availability, GPU model and VRAM, current GPU memory usage and running
processes, presence of NLP/ML packages (transformers, datasets, torch, accelerate, peft,
bitsandbytes, sentence-transformers, faiss, scipy, scikit-learn), disk space, and git status.
No installs, downloads, or changes were made.

**Why:** Before building anything, we needed to know what hardware and software was actually
available and, critically, whether the GPU was already in use by other work (it was — the
SAM-RNet `exp42` training job, ~12GB VRAM) so nothing would be built or launched that could
collide with or disrupt it.

**Findings:** Local machine has no NVIDIA GPU (Intel UHD 620 only) and no ML stack — not
viable for this project. Remote server has an NVIDIA RTX PRO 5000 Blackwell (~48GB VRAM,
driver 580.178.04, ~36GB free at audit time), and a separate conda env (`exp42`) already
running SAM-RNet training with torch 2.11.0+cu128 confirmed working on this exact GPU — this
became the reference point for choosing a CUDA build in Step 2. No ClaimGuard-specific
environment existed yet on the remote server.

---

## Step 2 — ClaimGuard environment setup (2026-08-31)

**What:** Created an isolated conda environment `claimguard` (Python 3.12) on the remote
server, separate from `exp42`. Installed PyTorch via the official PyTorch pip index
(cu128 build) plus transformers, datasets, accelerate, peft, sentence-transformers, scipy,
scikit-learn, faiss-cpu, pyyaml, tqdm, pandas, numpy, matplotlib, jupyter, and ipykernel.
Verified the install with a real CUDA tensor computation. Wrote `requirements.txt` (full
`pip freeze`) and a `.gitignore` appropriate for an ML project, and ran `git init` in
`~/RJ/ClaimGuard/` (no commits made).

**Why:** A dedicated environment avoids any risk of breaking the SAM-RNet `exp42` setup or its
in-progress training job. The cu128 PyTorch build was chosen — via the official install
method, not guessed — specifically because it was already proven to work with
`torch.cuda.is_available() == True` on this exact Blackwell GPU in the `exp42` env, which is
strong evidence of driver/CUDA compatibility before spending time on a full install.
`faiss-cpu` (not a GPU faiss build) was chosen deliberately: the GPU budget is meant for the
generator LLM, and CPU-side vector search is standard practice that avoids extra CUDA-runtime
coupling.

**Findings/results:** Python 3.12.14, PyTorch 2.11.0+cu128, CUDA 12.8, `torch.cuda.is_available()
== True`, GPU correctly detected as "NVIDIA RTX PRO 5000 Blackwell" with 47.27GB VRAM, and a
2000×2000 GPU matmul test passed. The `exp42` process (PID 1529809, ~12.25GB VRAM) was
confirmed still running and untouched throughout. Temporary setup scripts/logs used during
installation were cleaned up afterward, leaving only `requirements.txt` and `.gitignore` as
project artifacts.

---

## Step 3 — Model and dataset research (2026-08-31)

**What:** Researched (via live web search, not assumptions) current open-weight generator
LLMs (7B–14B range), NLI/claim-verification models, embedding models, rerankers, and
factuality/hallucination datasets, and selected a concrete stack. Wrote the full research
writeup and a concise `CLAIMGUARD_MODEL_SELECTION.md`. No weights or datasets were downloaded.

**Why:** Before implementing the pipeline, the project needed a deliberate, justified choice
of every model component and every dataset's role (train/validation/test/evaluation-only),
rather than defaulting to "the biggest model" or the first dataset that comes to mind. Live
research was used specifically because model releases move fast (8 months had passed since
this assistant's knowledge cutoff) and because license terms, benchmark exposure, and VRAM
budgets all needed to be checked against current, real information rather than fabricated.

**Findings/decisions (see `CLAIMGUARD_MODEL_SELECTION.md` for full rationale):**
- **Generator:** Qwen3-8B (primary, Apache 2.0, smallest footprint) / Llama-3.1-8B-Instruct
  (secondary, cross-family robustness check). Avoided Gemma-4-12B-it (Unified) — unneeded
  multimodal complexity and unverified tooling maturity for this task.
- **Verifier:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` — already trained
  on FEVER-NLI + MNLI + ANLI + LingNLI + WANLI, MIT license, tiny (<2GB VRAM), 3-way labels
  map directly to SUPPORTS/REFUTES/NEI.
- **Embedding / reranker:** BAAI/bge-large-en-v1.5 + BAAI/bge-reranker-large — English-only
  task doesn't need multilingual/sparse machinery, both are small and native to
  sentence-transformers.
- **Retriever:** BM25 + FAISS hybrid over the bge-large embeddings.
- **Correction model:** the generator itself (Qwen3-8B), prompted with the claim, verifier
  verdict, and retrieved evidence — avoids adding a second large model until there's enough
  training data to justify a dedicated correction adapter.
- **Datasets:** FEVER + HaluEval + RAGTruth for verifier/correction-trigger training and
  validation; RAGTruth held out as the primary end-to-end effectiveness test (its
  hallucinations are naturalistic LLM outputs, not memorizable benchmark claims); TruthfulQA,
  FActScore-Bio, and SelfCheckGPT's WikiBio set reserved as evaluation-only stress tests. FEVER
  and TruthfulQA were flagged as likely present in candidate generator LLMs' pretraining
  data — appropriate for training the separate verifier, but not appropriate as a "clean"
  held-out test of the generator's own hallucination rate.
- **GPU feasibility:** full inference pipeline (generator + verifier + embedder + reranker)
  concurrently estimated at ~23–25GB, safely inside the ~35GB currently free alongside
  `exp42`. QLoRA fine-tuning of the generator (~10–14GB additional) should run in isolation,
  not concurrently with the full pipeline.

---

## Step 4 — Project structure and configuration skeleton (2026-08-31)

**What:** Built the full ClaimGuard directory layout (`configs/`, `data/{raw,processed,evaluation}`,
`src/claimguard/` with 11 not-yet-implemented submodules — models, generation, claims,
retrieval, reranking, verification, correction, evaluation, datasets, pipelines, utils —
`scripts/`, `experiments/{baseline,ablation,results}`, `notebooks/`, `tests/`, `demo/`),
plus `configs/default.yaml`, `configs/models.yaml`, `configs/experiments.yaml`,
`pyproject.toml`, `README.md`, `PROJECT_PLAN.md`, and `RESEARCH.md`. Wrote a minimal
`src/claimguard/config.py` (YAML loading + required-key validation + path resolution,
dependency-light — only `pyyaml` + stdlib) and structural unit tests
(`tests/test_structure.py`, using the stdlib `unittest`, not `pytest`, since `pytest` wasn't
already installed). Installed the package editable (`pip install -e . --no-deps`) in the
`claimguard` conda env and ran the test suite.

**Why:** Steps 5 onward (model smoke tests, dataset work, pipeline components) all need
somewhere to live and a way to load configuration without importing torch/transformers or
touching the GPU — separating config plumbing from model code means later steps can be
tested structurally before any model is ever loaded. Model identifiers were put in
`configs/models.yaml` (not hard-coded) specifically so later code reads configuration rather
than embedding repo IDs throughout the codebase. `--no-deps` on the editable install avoided
any risk of pip silently pulling in new/updated packages beyond what Step 2 already
installed. `unittest` was chosen over `pytest` for the same reason — no new dependency needed
to satisfy the testing requirement.

**Results:** `pip install -e .` succeeded (`claimguard-0.1.0`); `python -c "import
claimguard"` succeeded; all 9 unit tests passed (package import, all three config files load,
required default-config keys present, missing-key detection works, project paths and data
subdirectories resolve). `git status` shows all new files as untracked — no commit made.
`exp42`'s GPU process was not touched; no model weights, datasets, or GPU inference were
involved in this step.

---

## Step 5A — Qwen3-8B model smoke test (2026-08-31 to 2026-09-01)

**What:** Wrote `scripts/smoke_test_qwen.py` (reads the generator config from
`configs/models.yaml`, no hard-coded model name), loaded `Qwen/Qwen3-8B` via
`transformers.AutoModelForCausalLM.from_pretrained` (BF16, `device_map="auto"`,
`low_cpu_mem_usage=True`, eval mode), and ran one short chat-template-formatted generation
("What is the capital of France? Answer in one sentence.", `max_new_tokens=50`,
`do_sample=False`).

**Why:** Before building any pipeline component, we needed to confirm the primary generator
actually loads and runs correctly in this environment — correct parameter count, correct
BF16 dtype, no CUDA/Blackwell incompatibility, and a sane VRAM footprint — on real hardware
rather than assuming compatibility from the research in Step 3.

**GPU safety handling (notable):** on the first attempt, the shared GPU had only ~14.7GB
free — a second, unrelated process from another user (`manish`,
`Camouflaged-Object-Segmention`, ~20.7GB) had started alongside `exp42` since Step 4, and
`exp42` itself had, separately and without any action from this session, stopped running.
Per the explicit instruction to stop rather than risk an OOM, the smoke test was *not*
attempted at that VRAM level. Per the user's choice, GPU memory was polled read-only
(`nvidia-smi` only, no process ever touched) until free VRAM rose above a 20GB safety
threshold; it did (freed up to ~35GB, then ~47GB once idle), at which point the smoke test
proceeded. The script itself also has this same 20GB pre-load safety check built in, so
future runs will self-abort under similar conditions rather than needing an external poll.

**Bug found and fixed:** the first script version failed post-load with
`TypeError: ones_like(): argument 'input' ... must be Tensor, not BatchEncoding` — calling
`tokenizer.apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding`, not a
raw tensor, when other kwargs are present. Fixed by passing `return_dict=True` explicitly and
reading `input_ids`/`attention_mask` from the resulting dict. Also switched the deprecated
`torch_dtype=` argument to `dtype=` on `from_pretrained` (a `transformers` 5.16.1 deprecation
warning on the first run) — a correctness fix in our own script, not a package change.

**Results — PASS:**
- Environment: Python 3.12.14, torch 2.11.0+cu128, `claimguard` conda env, CUDA available.
- GPU: NVIDIA RTX PRO 5000 Blackwell.
- VRAM before load: 46.93 GB free / 47.27 GB total (GPU was idle at this point).
- VRAM after load: 31.67 GB free (15.26 GB allocated/reserved by the model).
- Peak VRAM during generation: 15.27 GB allocated; 31.58 GB free afterward.
- Parameter count: 8,190,735,360 (~8.19B).
- Dtype: `torch.bfloat16`, confirmed on loaded parameters.
- Model load time: 132.8s on first run (includes ~16GB download from the HF Hub, no
  `HF_TOKEN` set — public model, no auth needed, just a rate-limit warning); 4.9s on the
  rerun with weights cached locally.
- Generated answer: `"The capital of France is Paris."` — factually correct.
- Generation latency: 0.76s for the 50-token-budget generation.
- No CUDA errors, no Blackwell-specific incompatibilities, no BF16 issues. One informational
  HF Hub rate-limit warning (expected, no token configured) and the one deprecation warning
  described above (fixed in-script). No memory, tokenizer, or model-loading issues once the
  chat-template bug was fixed.
- After the process exited, GPU returned to fully free (48.4GB) with zero compute processes
  — no leaked memory. `exp42` was not touched at any point (it was not running during this
  step at all, independent of anything done here).

---

## Step 5B — DeBERTa NLI verifier smoke test (2026-09-01)

**What:** Checked GPU state first (`nvidia-smi`, idle: 48.9GB free, 0 processes — safe to
proceed). Updated `configs/models.yaml`'s verifier entry from `float32` to `bfloat16` per this
step's instruction to use BF16 if supported. Wrote `scripts/smoke_test_deberta.py` (reads the
verifier config from `configs/models.yaml`, loads only this one model — no Qwen, no other
model in the same process) and loaded
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` via
`AutoModelForSequenceClassification.from_pretrained` on a single CUDA device, with a
try/except fallback to float32 if BF16 loading failed (it did not — BF16 worked directly).
Ran the three specified NLI cases (entailment / contradiction / neutral) and printed the full
3-way probability distribution for each, using the model's own `id2label` mapping rather than
assuming class index order.

**Why:** Same reasoning as Step 5A — confirm the verifier component actually loads and
produces sane, well-calibrated probability outputs on real hardware before it's wired into any
pipeline code, and confirm the label-index mapping empirically rather than hard-coding an
assumed order (DeBERTa NLI checkpoints don't always use the same index convention).

**Results — PASS:**
- Environment: Python 3.12.14, torch 2.11.0+cu128, `claimguard` env, CUDA available.
- GPU: NVIDIA RTX PRO 5000 Blackwell.
- VRAM before load: 46.93 GB free / 47.27 GB total.
- VRAM after load: 46.12 GB free (0.81 GB allocated/reserved — as expected for a ~435M-param
  model).
- Peak VRAM during inference: 0.82 GB allocated; 46.03 GB free afterward.
- Parameter count: 435,064,835 (~435.1M) — matches the DeBERTa-v3-large scale.
- Dtype: `torch.bfloat16` (requested and used successfully, no fallback needed), device
  `cuda:0`.
- Model load time: 12.8s.
- **id2label mapping (verified from `model.config`, not assumed):**
  `{0: 'entailment', 1: 'neutral', 2: 'contradiction'}`.
- **Test 1 (entailment):** predicted `entailment`, confidence 0.9993
  (entailment=0.99928, neutral=0.00060, contradiction=0.00013). Latency 463.1ms (includes
  one-time CUDA warmup on first inference call).
- **Test 2 (contradiction):** predicted `contradiction`, confidence 0.9970
  (entailment=0.00056, neutral=0.00247, contradiction=0.99697). Latency 122.8ms.
- **Test 3 (neutral):** predicted `neutral`, confidence 0.9995
  (entailment=0.00030, neutral=0.99952, contradiction=0.00018). Latency 89.8ms.
- All three predictions matched the expected semantic relationship on this one example each
  — **not a benchmark result**, just confirmation the model, tokenizer, and label mapping
  behave sensibly end to end.
- No CUDA errors, no Blackwell incompatibilities, no BF16 loading issues (unlike the Qwen3-8B
  script's chat-template bug in Step 5A, this script ran correctly on the first attempt). One
  informational HF Hub rate-limit warning (expected, no `HF_TOKEN` set).
- After the process exited, GPU returned to fully free (48.4GB free / 48.9GB total idle
  baseline) with zero compute processes — no leaked memory. No other model was loaded in this
  process; no other GPU process was touched.

---

## Step 5C — BGE embedding model smoke test (2026-09-01)

**What:** Checked GPU state first (`nvidia-smi`, idle: 48.9GB free, 0 processes — safe to
proceed). Added a `normalize_embeddings: true` field to `configs/models.yaml`'s embedding
entry (model name, device, dtype, batch size were already present from Step 4). Wrote
`scripts/smoke_test_bge.py` (reads the embedding config from `configs/models.yaml`, loads
only `BAAI/bge-large-en-v1.5` via `sentence-transformers` — no Qwen, no DeBERTa, no reranker,
no FAISS index built). Encoded four test sentences individually and as an explicit batch,
computed pairwise cosine similarities (A-B, A-C, A-D), verified the embedding dimension from
the actual model output rather than assuming 1024, and released the model/embeddings with
`gc.collect()` + `torch.cuda.empty_cache()` before exit.

**Why:** Same reasoning as Steps 5A/5B — confirm the embedding component loads correctly, in
the intended precision, on real hardware, and that its output shape/dimension match
expectations empirically before it's wired into retrieval code. Verifying GPU memory returns
to baseline after cleanup also confirms the model can be loaded/unloaded repeatedly later
without leaking VRAM across pipeline stages.

**Results — PASS:**
- Environment: Python 3.12.14, torch 2.11.0+cu128, `claimguard` env, CUDA available.
- GPU: NVIDIA RTX PRO 5000 Blackwell.
- VRAM before load: 46.93 GB free / 47.27 GB total.
- VRAM after load: 46.30 GB free (0.62 GB allocated / 0.63 GB reserved — as expected for a
  ~335M-param model).
- Peak VRAM during inference: 0.63 GB.
- Parameter count: 335,141,888 (~335.1M) — matches the bge-large scale.
- Dtype: `torch.float16` (requested and used successfully), device `cuda:0`.
- Model load time: 21.1s.
- **Embedding dimension (verified from actual model output, not assumed): 1024** — matches
  the expected bge-large-en-v1.5 dimension.
- **A-B similarity: 0.9516** (two paraphrases of the same fact — highest, as expected)
- **A-C similarity: 0.7888** (related domain — European capitals — but different fact)
- **A-D similarity: 0.4478** (unrelated topic — lowest, as expected)
- Similarity ordering (A-B > A-C > A-D) matches semantic expectations on this one example set
  — **not a retrieval benchmark**, just a sanity check that the embedding space is sensibly
  structured.
- **Batch embedding tensor shape: (4, 1024)** — matches expected `(num_sentences,
  embedding_dim)` exactly.
- Encoding latency: 445.6ms for the first (individual) encode call (includes CUDA warmup),
  12.0ms for the second (explicit batch) call.
- **Cleanup:** after `del` + `gc.collect()` + `torch.cuda.empty_cache()`, VRAM returned to
  46.85 GB free — within 0.09 GB of the 46.93 GB pre-load baseline (residual is normal CUDA
  context overhead, not a leak). No other GPU process was affected.
- No CUDA errors, no Blackwell incompatibilities, no FP16 loading issues. One informational HF
  Hub rate-limit warning (expected, no `HF_TOKEN` set).

---

## Step 5D — BGE reranker smoke test (2026-09-01)

**What:** Checked GPU state first (`nvidia-smi`, idle: 48.9GB free, 0 processes — safe to
proceed). `configs/models.yaml`'s reranker entry already had all required fields (name,
device, dtype, max_seq_length, batch_size) from Step 4, so no config changes were needed.
Wrote `scripts/smoke_test_reranker.py` (reads the reranker config from
`configs/models.yaml`, loads only `BAAI/bge-reranker-large` via
`sentence_transformers.CrossEncoder` — no Qwen, no DeBERTa, no embedding model, no FAISS
index). Reranked one query against four candidate passages, sorted by actual score
(no hard-coded expected ranking), ran an explicit batch call, and released the
model/CUDA cache before exit.

**Why:** Same reasoning as the prior smoke tests, plus one reranker-specific question the
task explicitly called out: whether `CrossEncoder.predict()`'s output is a raw logit or an
already-calibrated probability, since mislabeling one as the other would corrupt downstream
verification-confidence logic later. Rather than assuming, the script reads the model's own
`activation_fn` attribute and additionally forces `activation_fn=torch.nn.Identity()` on a
second pass to capture the true raw logits for comparison.

**Results — PASS:**
- Environment: Python 3.12.14, torch 2.11.0+cu128, `claimguard` env, CUDA available.
- GPU: NVIDIA RTX PRO 5000 Blackwell.
- VRAM before load: 46.93 GB free / 47.27 GB total.
- VRAM after load: 45.88 GB free (1.04 GB allocated / 1.05 GB reserved — as expected for a
  ~560M-param model).
- Peak VRAM during inference: 1.05 GB.
- Parameter count: 559,891,457 (~559.9M) — matches the bge-reranker-large scale.
- Dtype: `torch.float16` (requested and used successfully), device `cuda:0`.
- Model load time: 37.7s.
- **Score type — verified, not assumed:** the model's default `activation_fn` is `Sigmoid`,
  so `CrossEncoder.predict()`'s default output is **probability-like** (bounded [0,1]), not a
  raw logit. Forcing `activation_fn=Identity()` recovered the true raw logits for comparison:
  Candidate 1 = 8.2969, Candidate 2 = -5.9219, Candidate 3 = -2.8438, Candidate 4 = -6.7148
  (unbounded, as expected for raw logits).
- **Ranking (sorted by actual default/probability-like score, descending):**
  1. Candidate 1 ("Paris is the capital and largest city of France.") — 0.9998
  2. Candidate 3 ("France is a country in Western Europe...") — 0.0550
  3. Candidate 2 ("Berlin is the capital of Germany...") — 0.0027
  4. Candidate 4 ("The Pacific Ocean is the largest...") — 0.0012
  The direct answer ranked highest by a wide margin on this one example — **not a benchmark
  result**, just confirmation the model, tokenizer, and scoring behave sensibly end to end.
- **Batch output shape: (4,)** — one score per pair, as expected. Batch scores:
  `[0.9998, 0.0027, 0.055, 0.0012]` (order matches input candidate order 1-4).
- Latency: 322.7ms for the first single-pair inference (includes CUDA warmup), 10.8ms for the
  4-pair batch call.
- **Cleanup:** after `del` + `gc.collect()` + `torch.cuda.empty_cache()`, VRAM returned to
  46.85 GB free — within 0.09 GB of the 46.93 GB pre-load baseline (same residual as prior
  steps; not a leak). No other GPU process was affected.
- No CUDA errors, no Blackwell incompatibilities, no FP16 loading issues. One informational HF
  Hub rate-limit warning (expected, no `HF_TOKEN` set).

---

## Step 6A — FEVER dataset acquisition and inspection (2026-09-01)

**What:** CPU/data-only step, no models or GPU involved. Verified the authoritative source
before downloading (web research: `fever/fever` on Hugging Face, the FEVER project's own org,
linked to fever.ai; license CC BY-SA 3.0 + GNU FDL for the underlying Wikipedia text).
Wrote `scripts/inspect_fever.py` (acquisition + inspection combined) and
`src/claimguard/datasets/fever.py` (loading/normalization utilities), plus
`scripts/build_fever_sample.py` and `tests/test_fever_dataset.py`. Downloaded FEVER's `train`,
`validation`, and `test` splits to `data/raw/fever/*.jsonl` (raw, unmodified), inspected real
schema/labels/evidence structure, ran integrity checks, built a 30-record (10/10/10) sample at
`data/processed/fever/sample/sample.jsonl`, and ran the full test suite.

**Why:** Before designing the verifier training format, we needed to know FEVER's actual
structure empirically rather than trust the split names and shapes found during earlier desk
research — and two real complications did in fact surface only once the real data was
downloaded (detailed below), directly validating that approach.

**Acquisition — two real obstacles hit and resolved:**
1. `fever/fever`'s official script-based loader failed outright under the installed
   `datasets` 5.0.1: `RuntimeError: Dataset scripts are no longer supported, but found
   fever.py` — script-based dataset loading was removed in recent `datasets` versions.
   Fell back to Hugging Face's own auto-converted Parquet export (`refs/convert/parquet`) —
   an official HF-hosted conversion of the same authoritative repo, not a third-party mirror.
2. The parquet export's config name isn't `v1.0` as the original loading script exposed —
   only a single `default` config exists there, and it exposes just 3 splits
   (`train`/`validation`/`test`), not the 6-way split set (train / unlabelled_dev /
   labelled_dev / paper_dev / unlabelled_test / paper_test) described in earlier desk
   research. The acquisition script now discovers available configs/splits at runtime via
   `datasets.get_dataset_config_names`/`get_dataset_split_names` rather than assuming, and
   skips (does not guess-substitute) any requested split that isn't actually present.

**Dataset size by split (real, from the downloaded data):**
- `train`: 311,431 rows / **145,449 unique claims**
- `validation`: 78,947 rows / 21,172 unique claims
- `test`: 38,565 rows / 29,997 unique claims

**Schema (verified from actual data, all splits):** columns
`id, label, claim, evidence_annotation_id, evidence_id, evidence_wiki_url,
evidence_sentence_id`. The data is **denormalized — one row per evidence line, not one row
per claim** (row count > unique claim count in every split). Rows sharing `id` belong to the
same claim; rows sharing `(id, evidence_annotation_id)` belong to the same evidence set (a
claim can have several alternative evidence sets, any one of which suffices — `train`'s
distribution ranges from 1 set for most claims up to 41 sets for a few outliers).

**Labels — confirmed, with a real-data caveat:** `train`'s labels are exactly
`{SUPPORTS: 193,756, REFUTES: 70,066, NOT ENOUGH INFO: 47,609}` — clean, matches the expected
FEVER label set exactly. **`validation` and `test` are not clean as delivered**: alongside
real labels, both contain 19,998 rows with a **blank (`""`) label** (almost certainly
blind/withheld-label rows folded in by the auto-conversion), and `validation` additionally has
33 rows labeled `"Not Enough Info"` (wrong casing) instead of `"NOT ENOUGH INFO"`. The
FEVER→verifier label mapping (`SUPPORTS→entailment, REFUTES→contradiction, NOT ENOUGH
INFO→neutral`, matching the DeBERTa verifier's empirically-confirmed `id2label` from Step 5B)
was applied only to `train`'s samples for this reason; `fever.py`'s `group_claims()` maps
these unexpected labels to `verifier_label=None` rather than guessing.

**Evidence structure:** confirmed via `evidence_wiki_url`/`evidence_id`/`evidence_sentence_id`.
Critically, **"no evidence" is encoded with sentinel values, not null** —
`evidence_wiki_url == ""` and `evidence_id`/`evidence_sentence_id == -1` — every `NOT ENOUGH
INFO` row across all splits uses this sentinel, and a `None`/null check misses it entirely.

**Integrity findings (all splits, real counts — nothing fabricated):**
- Exact duplicate rows: `train` 6, **`validation` 19,037 (~24% of that split)**, `test` 0.
- Missing/empty claim text: 0 in every split.
- Invalid (non-empty, non-conforming) labels: `train` 0, `validation` 33
  (`"Not Enough Info"` casing), `test` 0.
- Malformed evidence-vs-label combinations: `train` 0 (after fixing a false-positive — see
  below), `validation` 151, `test` 0.
- Claim-ID overlap: `train`↔`validation` 0, `train`↔`test` 0, **`validation`↔`test` 9,999**
  (these two splits are not disjoint as delivered).
- **A bug in the inspection script itself was caught and fixed before being reported as a
  finding:** the first run's "malformed evidence" check assumed `None` meant "no evidence,"
  which flagged all 47,609 `NOT ENOUGH INFO` rows in `train` as malformed — a false positive
  caused by the wrong null-representation assumption, not a real data problem. Fixed to use
  the actual sentinel encoding (`""`/`-1`) discovered above; the corrected check reports 0
  malformed rows in `train`, confirming the data is actually clean there.

**Normalization design (`src/claimguard/datasets/fever.py`):** `load_raw_split()` reads
`data/raw/fever/<split>.jsonl`; `group_claims()` aggregates the denormalized rows into one
record per claim — `example_id, claim, label, verifier_label, evidence (list of evidence
sets, each a list of {wiki_url, sentence_id}), evidence_ids, split`; `validate_record()`
reports issues (empty claim, invalid label, evidence/label mismatch) without discarding
anything — callers decide. This schema was written *after* inspecting the real data, not
before.

**Sample created:** `data/processed/fever/sample/sample.jsonl` — 10 SUPPORTS + 10 REFUTES +
10 NOT ENOUGH INFO (30 total), built from `train` only (the one split with no blank/casing
label issues), all passing `validate_record()` with zero issues skipped. Explicitly a
development/testing fixture, not an evaluation benchmark.

**Tests:** `tests/test_fever_dataset.py`, 12 new tests (package import/config tests from
Step 4 also re-run) — **21/21 passed**: sample loads, expected fields present, labels
normalize correctly (including the blank-label → `None` case), evidence normalization
(grouping + sentinel-dropping) works on both the real sample and a synthetic fixture, and all
four malformed-record cases (invalid label, empty claim, SUPPORTS/REFUTES without evidence,
NOT ENOUGH INFO with evidence) are correctly detected.

**Storage:** raw data in `data/raw/fever/*.jsonl` (63MB + 15.7MB + 7.5MB, untouched after
download); processed sample in `data/processed/fever/sample/` — kept separate per
instructions, and both are covered by `.gitignore`'s `data/` rule from Step 4.

**Documentation:** `RESEARCH.md` updated with a new "FEVER acquisition findings (Step 6A)"
subsection under Leakage Concerns — reinforces, does not change, the protocol already
selected in Step 3/4 (RAGTruth remains the primary end-to-end effectiveness test; FEVER
`train` is for verifier training/validation only).

**Issues/recommendations for future steps:** the `validation`/`test` splits as delivered need
explicit deduplication and blank/casing-label filtering before they could be used for
anything beyond `train`-based verifier work — not attempted here, as Step 6A was acquisition
and inspection only. `wiki_pages` (the 5.4M-page Wikipedia evidence corpus) was not
downloaded — out of scope for this step, needed only once retrieval implementation begins.

**Git status:** still no commits — all new/changed files (`scripts/inspect_fever.py`,
`scripts/build_fever_sample.py`, `src/claimguard/datasets/fever.py`,
`tests/test_fever_dataset.py`, updated `RESEARCH.md`) show as untracked/modified.

---

## Step 6B — HaluEval dataset acquisition and inspection (2026-09-01)

**What:** CPU/data-only step, no models or GPU involved. Verified the authoritative source
before downloading (web research: no official HaluEval Hugging Face org exists — only
community re-uploads like `pminervini/HaluEval` — so the original
`RUCAIBox/HaluEval` GitHub repository was used instead, confirmed MIT-licensed via its
`LICENSE` file). Wrote `scripts/inspect_halueval.py` (acquisition + inspection),
`src/claimguard/datasets/halueval.py` (normalization utilities), `scripts/build_halueval_sample.py`,
and `tests/test_halueval_dataset.py`. Downloaded all four HaluEval data files to
`data/raw/halueval/*.json` (raw, unmodified), inspected real schema/format/labels/duplicates
per file, built a 40-record sample spanning all four subsets at
`data/processed/halueval/sample/sample.jsonl`, and ran the full existing test suite (not
just the new tests).

**Why:** Same evidence-first approach as Step 6A — HaluEval's real structure needed to be
established from the actual files before any normalization code was written, and (as with
FEVER) real obstacles and a real data anomaly did surface only once the data was in hand.

**Acquisition — obstacles hit and resolved:**
1. A transient `ConnectionResetError` on the 4th file (`general_data.json`) during the first
   run — not a structural issue, just a network hiccup. The acquisition script is idempotent
   (skips files already saved), so a plain retry completed the download cleanly.
2. **Format assumption avoided, not made:** despite the `.json` extension, all four files are
   actually **JSON Lines** (one JSON object per line), not a single JSON array. The
   inspection script tries JSON-array parsing first and falls back to JSONL only if that
   fails, rather than assuming either format from the filename — confirmed empirically as
   `jsonl` for all four files.

**File sizes / row counts (real, from the downloaded data):**
- `qa_data.json`: 10,000 records
- `dialogue_data.json`: 10,000 records
- `summarization_data.json`: 10,000 records
- `general_data.json`: **4,507 records** (not the commonly-cited 5,000 — the real file has
  4,507)

**Schema — two materially different shapes found, and NOT forced into one abstraction:**
- **qa / dialogue / summarization ("contrastive pair" subsets):** each raw record is a
  grounding context plus a non-hallucinated reference and a hallucinated counterpart —
  **no direct per-example binary label exists in the raw data**, and **no id field exists in
  any of the three files**. Field names differ meaningfully per subset: `qa` has one context
  field (`knowledge`); `dialogue` has *two* (`knowledge` **and** `dialogue_history`);
  `summarization` uses a differently-named context field (`document`) rather than
  `knowledge`. All three: 0 null values, 0 empty strings, 0 exact duplicate records, 100%
  schema consistency across every record (union keys == intersection keys) — clean.
- **general (real single-response subset):** structurally different — one real ChatGPT
  response per record with a genuine human-annotated label (`hallucination`: lowercase
  `"yes"`/`"no"` in the real data, not the `"Yes/No"` casing the README's prose implied), plus
  **span-level `hallucination_spans`** (a list of the specific hallucinated substrings) for
  `"yes"` cases — richer than a plain binary label. **No reference/"right" field at all** in
  this subset (unlike the other three) — it judges a real response as-is. Label distribution:
  `no` 3,692 / `yes` 815.

**A real data anomaly found and investigated, not assumed:** `general_data.json`'s `ID`
field has 4,507 values but only 4,506 unique. Rather than reporting "1 duplicate ID" as a bare
number, the inspection script was extended to print the actual colliding records — this
revealed **two separate records whose `ID` value is literally the string `"ID"`** (the
column-header text itself, leaked in as a data value), plus a separate, distinct record with
`ID == ""` (empty). Confirmed as a genuine upstream data artifact in the source file (not a
parsing bug on our side) by inspecting the raw values directly, per the instruction to
investigate before labeling something an integrity issue. `halueval.py`'s
`SUSPICIOUS_RAW_IDS = {"", "ID"}` and `validate_record()`'s `suspicious_raw_id` check
reproduce this exact finding as a machine-readable, tested check (see
`test_suspicious_raw_id_detected`). Because of this, the native `ID` field is preserved for
traceability (`raw_id`) but is **not** used as ClaimGuard's uniqueness key — every normalized
record instead carries `raw_index` (0-based file position), which is reliable across all four
files.

**Integrity findings (real counts — nothing fabricated):**
- Exact duplicate records: 0 in all four files.
- Missing/null values: 0 in all four files, all fields.
- Empty-string values: 0 in all fields except `general_data.json`'s `ID` (1 occurrence — the
  anomaly above).
- Duplicate IDs: only `general_data.json` has an id-like field at all, and it has exactly one
  colliding value (`"ID"`, appearing twice) — investigated and explained above.
- Cross-file overlap: checked `knowledge`/`question`/`document` field values across files
  where the field name is shared (only `knowledge`, shared by `qa` and `dialogue`) — 0
  overlap.

**Normalization design (`src/claimguard/datasets/halueval.py`):** deliberately **two
different normalization paths**, not one, matching the two real shapes found:
- `expand_pair_record()` — for qa/dialogue/summarization, expands each raw record into **two**
  normalized binary-labeled records (one `not_hallucinated` from the reference field, one
  `hallucinated` from the hallucinated field). This is an explicit normalization *choice*
  (the raw data has no direct label), documented as such in the module docstring rather than
  presented as something the data already provided.
- `normalize_general_record()` — for `general`, a 1:1 mapping (already single-response,
  already labeled), deriving `label` from the lowercase `hallucination` field and preserving
  `hallucination_spans` as-is.
- Shared normalized schema across both paths: `example_id, subset, context (dict, keys vary
  honestly by subset), query, candidate_response, label, hallucination_spans, source_field,
  raw_id, raw_index`. `context` is always a dict rather than a string specifically so
  `dialogue`'s two context fields and `general`'s complete absence of context fields are both
  represented truthfully rather than papered over.
- `validate_record()` checks: empty candidate response, invalid label, missing context
  (subset-aware — not flagged for `general`, which legitimately has none), malformed
  `hallucination_spans` type, and the `suspicious_raw_id` check described above. Nothing is
  silently discarded — callers decide what to do with reported issues.

**Sample construction (`data/processed/halueval/sample/sample.jsonl`, 40 records):**
deliberately spans **all four subsets**, not just one, because the dataset itself is not
homogeneous — a sample from only one subset wouldn't exercise the other normalization path at
all. qa/dialogue/summarization: first 5 raw records each (→10 normalized records each, 5+5
label-balanced automatically by the pair-expansion). `general`: first 5 `"yes"` and first 5
`"no"` records **in file order** — a deliberate, documented departure from pure "first N" for
this subset only, since general's real label distribution is imbalanced (~82%/18%) and a
naive first-10 slice would likely miss the minority label entirely. Selection is fully
deterministic (no randomness) and reproducible. 0 candidate records were skipped for failing
`validate_record()`. Explicitly a development/testing fixture, not an evaluation benchmark.

**Tests:** `tests/test_halueval_dataset.py`, 17 new tests. **Ran the full existing suite, not
just the new tests: 38/38 passed** (17 new HaluEval tests + 12 FEVER tests from Step 6A + 9
structural tests from Step 4). New tests cover: schema/field presence, all-four-subsets
representation, pair-expansion label balance, per-subset context field differences
(qa/dialogue/summarization each verified independently), general label derivation (including
the unmapped-value → `None` case), the exact real duplicate-ID anomaly reproduced as a test
fixture, missing-context detection (correctly subset-aware), and full-sample validation (every
sample record passes `validate_record()` with zero issues).

**Storage:** raw data in `data/raw/halueval/*.json` (6.0MB + 6.8MB + 45.9MB + 3.2MB, untouched
after download); processed sample in `data/processed/halueval/sample/` — kept separate, both
covered by `.gitignore`'s `data/` rule from Step 4.

**Documentation:** `RESEARCH.md` updated with a new "HaluEval acquisition findings (Step 6B)"
subsection under Leakage Concerns — reinforces, does not change, the Step 3/4 protocol
(HaluEval — all four subsets — joins FEVER `train` in the verifier/correction-trigger
training pool; RAGTruth remains the primary end-to-end effectiveness test, since three of
HaluEval's four subsets are synthetically constructed contrastive pairs rather than
naturalistic model outputs).

**Unresolved issues:** none blocking. Noted for future steps: `summarization_data.json` is
much larger (45.9MB) than the other three despite the same 10,000-record count — its
`document`/summary fields are simply longer text on average; worth accounting for in any
future batch-size/context-length planning for the verifier, but not an integrity problem.

**Git status:** still no commits — all new/changed files (`scripts/inspect_halueval.py`,
`scripts/build_halueval_sample.py`, `src/claimguard/datasets/halueval.py`,
`tests/test_halueval_dataset.py`, updated `RESEARCH.md`) show as untracked/modified.

---

## Step 6C — RAGTruth dataset acquisition and inspection (2026-09-01)

**What:** CPU/data-only step, no models or GPU involved. RAGTruth is ClaimGuard's **primary
end-to-end evaluation dataset**, so this step was held to a higher standard than 6A/6B,
especially on leakage. Verified the authoritative source before downloading — no single
"official" Hugging Face mirror exists with certain fidelity, so used the paper's own GitHub
repository, `ParticleMedia/RAGTruth` (MIT License, verified via its `LICENSE` file), in
preference to community re-uploads of uncertain provenance. Wrote
`scripts/inspect_ragtruth.py` (acquisition + inspection, including a join between the two raw
files and dedicated leakage/offset-validation logic), `src/claimguard/datasets/ragtruth.py`
(normalization with an explicit, code-level evaluation-boundary safeguard), and
`scripts/build_ragtruth_sample.py` / `tests/test_ragtruth_dataset.py`. Downloaded
`response.jsonl` and `source_info.jsonl` to `data/raw/ragtruth/` (raw, unmodified), inspected
schema/labels/annotation offsets, ran an explicit cross-split leakage investigation, built a
30-record **train-only** sample, and ran the full existing test suite.

**Why:** Because RAGTruth is the dataset the project's actual "did ClaimGuard reduce
hallucinations" claim will rest on (per Step 3/4's research protocol), a leak between its
train and test data — even a subtle one via shared underlying source documents rather than
literal duplicate rows — would silently invalidate that claim. The higher scrutiny (joining
files, checking source-level rather than only row-level overlap, validating offsets against
real text) was applied for that reason.

**Acquisition:** both files downloaded cleanly on the first attempt (no retries needed this
time). Both are genuinely JSON Lines, matching their `.jsonl` extension (unlike FEVER's/
HaluEval's misleadingly-named files) — confirmed empirically via the same
array-then-JSONL-fallback parser used in Steps 6A/6B, not assumed from the extension.

**Real files / row counts:**
- `response.jsonl`: **17,790 records** (split exactly: 15,090 `train` + 2,700 `test` — matches
  the paper's published figures, a useful authenticity check).
- `source_info.jsonl`: **2,965 records**.
- Every source item has **exactly 6** responses (2,965 × 6 = 17,790, exact) — one per model:
  `gpt-4-0613`, `gpt-3.5-turbo-0613`, `mistral-7B-instruct`, `llama-2-7b-chat`,
  `llama-2-13b-chat`, `llama-2-70b-chat` (2,965 each).

**Schema (verified from actual data):**
- `response.jsonl` fields (all present in every record, zero nulls): `id, source_id, model,
  temperature, labels, split, quality, response`. `id` is **100% unique** (17,790/17,790) —
  cleaner than HaluEval's flawed `ID` field. `quality`: `good` 17,617 / `truncated` 29 /
  `incorrect_refusal` 144.
- `source_info.jsonl` fields (all present in every record, zero nulls): `source_id, task_type,
  source, source_info, prompt`. `source_id` is 100% unique (2,965/2,965). `source_info` itself
  is **not one fixed shape** — a plain string for `task_type="Summary"` (the article text), a
  nested dict for `"Data2txt"` (business listing: name/address/categories/hours/attributes/
  review_info), and a differently-shaped dict for `"QA"` (`{"question", "passages"}`).
  `task_type` distribution: `Summary` 943 / `Data2txt` 1,033 / `QA` 989.

**Join integrity:** 100% referential — every response's `source_id` matches exactly one
`source_info` record, 0 orphans.

**Annotation/label structure:** 7,664 of 17,790 responses have ≥1 label (hallucinated span);
10,126 have an empty `labels` list (no hallucination flagged) — sums correctly. 14,289 total
individual span annotations. Each label dict has `start, end, text, label_type, implicit_true,
meta, due_to_null`, present in every label with no missing keys. `label_type` distribution:
`Evident Baseless Info` 6,237 / `Evident Conflict` 5,324 / `Subtle Baseless Info` 2,527 /
`Subtle Conflict` 201. `implicit_true`: `False` 12,361 / `True` 1,928.

**Offsets validated programmatically, not trusted from the README:** for all 14,289
offset-bearing labels, `response[start:end]` was checked against the label's own `text`
field — **100% exact match (14,289/14,289), 0 mismatches, 0 out-of-bounds or start>end
offsets.** This is real, verified data quality, not an assumption.

**Leakage investigation (the high-scrutiny part of this step):**
- Checked whether any of the 2,965 source items have responses spanning **more than one**
  split (the real leakage risk, since row-level `id`s are already 100% unique and thus
  trivially non-overlapping) — **result: 0**. Every source item's 6 responses belong entirely
  to `train` or entirely to `test`. Also directly verified via `source_id` set overlap between
  splits: 0.
- **One suspicious result was investigated before being reported, not taken at face value:**
  exact response-*text* overlap between splits initially showed 1 match. Extended the
  inspection script to print the actual colliding records rather than just the count — the
  matching text was the boilerplate refusal `"Unable to answer based on given passages."`,
  produced by different models for different, legitimately unrelated source items (different
  `source_id`s) in both splits. **Confirmed benign — a coincidental short stock phrase, not a
  content leak** — before it was written up as a finding.
- No exact duplicate rows in either file (0 in `response.jsonl`, 0 in `source_info.jsonl`).

**Primary evaluation boundary — explicit statement:** **RAGTruth's `test` split (2,700
responses across 450 source items) is reserved as the untouched, primary end-to-end
evaluation set and was not used to build the Step 6C sample or any processed artifact beyond
the one-time inspection script.** `train` (15,090 responses, 2,515 source items) is available
for future verifier/correction-trigger development if a later step chooses to use it — Step
6C validates that the split is safe to use this way (source-disjoint, confirmed above) but
does not itself move any RAGTruth data into a training pool, consistent with the instruction
not to do so without an explicit project-protocol requirement.

**Normalization design (`src/claimguard/datasets/ragtruth.py`):** `join_records()` joins
`response.jsonl` to `source_info.jsonl` via `source_id`, producing one lossless record per
response with **every original field preserved** (nothing summarized away) plus two derived
fields: `has_hallucination` (bool) and, critically, **`eval_reserved`** (`True` iff
`split == "test"`) — the evaluation boundary made explicit and machine-checkable rather than
left as a convention. `get_training_pool()`/`get_eval_set()` are the only two accessors
downstream code is expected to use; `assert_no_train_test_leakage()` is a reusable safeguard
(raises `AssertionError` if any `source_id` ever spans both splits) that can be invoked before
any future training step, not just a one-off script check. `validate_offsets()` re-validates
each record's labels against its own `response_text` (defense in depth). Orphan responses
(source_id with no matching source_info) are preserved with `source_missing=True` rather than
silently dropped — 0 occur in the real data, but the code does not assume that stays true.

**Sample construction (`data/processed/ragtruth/sample/sample.jsonl`, 30 records):** built
**exclusively from `train`** — `test`-split records are never read into this script's
selection logic at all, and the script asserts this at write time
(`assert all(r["split"] == "train" ...)`) as a belt-and-suspenders check beyond the upstream
filter. Covers all 6 (`task_type` × `has_hallucination`) combinations the real data contains
(5 records each, first-in-file-order — fully deterministic), so every structural shape
(Summary/Data2txt/QA × hallucinated/not) is represented, including 15 records with real
span-level annotations.

**Tests:** `tests/test_ragtruth_dataset.py`, 24 new tests. **Ran the full existing suite:
62/62 passed** (24 new RAGTruth tests + 17 HaluEval + 12 FEVER + 9 structural). New tests
specifically cover: schema, join behavior (including the orphan-preservation case via a
fixture), label/hallucination-flag derivation, offset validation (valid, mismatched, and
out-of-bounds fixtures, plus a full check that every real sample label has zero offset
issues), duplicate-ID detection, and — the emphasis of this step —
**evaluation-boundary protection**: `get_training_pool`/`get_eval_set` filtering correctness,
`assert_no_train_test_leakage` both passing on a clean fixture and correctly raising on a
deliberately-leaking fixture, confirming the real sample is 100% `train`/non-reserved, and
**re-running the leakage assertion against the full real 17,790-record dataset** (not just
the sample) as part of the automated test suite.

**Obstacles encountered:** none beyond the investigation described above — both files
downloaded cleanly, and no format/config-discovery problems arose (unlike FEVER's blocked
script loader or HaluEval's transient connection reset).

**Bugs discovered in our own inspection code:** none this time — the one "suspicious result"
(cross-split text overlap) was investigated and resolved as a genuine-but-benign data
property, not a bug in the inspection logic itself.

**Unresolved issues:** none blocking.

**Storage:** raw data in `data/raw/ragtruth/response.jsonl` (20.5MB) and `source_info.jsonl`
(14.4MB), untouched after download; processed sample in `data/processed/ragtruth/sample/` —
both covered by `.gitignore`'s `data/` rule from Step 4.

**Documentation:** `RESEARCH.md` updated with a new "RAGTruth acquisition findings (Step 6C)"
subsection under Leakage Concerns — this one **validates and confirms**, rather than merely
reinforces, the Step 3/4 protocol: RAGTruth's split is empirically source-disjoint and safe to
use exactly as planned (test reserved for Experiment 3, train available if later chosen for
verifier development).

**Git status:** still no commits — all new/changed files (`scripts/inspect_ragtruth.py`,
`scripts/build_ragtruth_sample.py`, `src/claimguard/datasets/ragtruth.py`,
`tests/test_ragtruth_dataset.py`, updated `RESEARCH.md`) show as untracked/modified.

---

## Step 6D — TruthfulQA dataset acquisition and inspection (2026-09-01)

**What:** CPU/data-only step, no models or GPU involved. Verified the authoritative source
before downloading: the original authors' own GitHub repository, `sylinrl/TruthfulQA`
(Stephanie Lin, Jacob Hilton, Owain Evans — "TruthfulQA: Measuring How Models Mimic Human
Falsehoods"), Apache-2.0 License, chosen over the community-added Hugging Face mirror
(`truthfulqa/truthful_qa`, added by a third party per its own dataset card, not the original
authors, though it does cite them). Wrote `scripts/inspect_truthfulqa.py`,
`src/claimguard/datasets/truthfulqa.py`, `scripts/build_truthfulqa_sample.py`, and
`tests/test_truthfulqa_dataset.py`. Downloaded both official files to `data/raw/truthfulqa/`
(raw, unmodified), inspected schema/categories/answer structure, built a 37-record sample (one
per category), and ran the full existing test suite.

**Why:** TruthfulQA needed to be treated as its own task shape from the start — it has no
retrieval evidence and (as confirmed below) no split of any kind, so forcing it into any of
the FEVER/HaluEval/RAGTruth patterns would have been wrong. The instruction to explicitly
verify rather than invent split semantics was followed literally: the inspection checks for
a split-like column and reports its absence as a fact, rather than assuming one exists or
manufacturing one for convenience.

**Acquisition — one real obstacle, resolved:** `TruthfulQA.csv` hit a transient
`ConnectionResetError` on the first attempt (network hiccup, not a structural issue); the
acquisition script is idempotent, so a plain retry completed it. `mc_task.json` downloaded
cleanly both times. One bug in our own script was caught before it ran on real data: an
f-string with a string literal broken incorrectly across two lines caused a `SyntaxError` on
the very first run — fixed immediately (a script bug, not a dataset finding).

**Real files / counts — and a real discrepancy from commonly-cited figures:**
- `TruthfulQA.csv`: **790 rows**, columns `Type, Category, Question, Best Answer, Best
  Incorrect Answer, Correct Answers, Incorrect Answers, Source` (exact, read from the header
  row, not assumed). Zero nulls; 2 empty-string `Source` values.
- `data/mc_task.json`: **790 records**, fields `question, mc0_targets, mc1_targets,
  mc2_targets` (all present in every record).
- **Both real counts are 790, not the commonly-cited 817** (questions) **or 38** (the CSV has
  **37** real categories) — the upstream repository has evidently grown/been revised since
  the original paper's published figures (its README separately notes recent updates,
  including a January-2025-dated addition of the `Best Incorrect Answer` column). This
  project's artifacts reflect the real, currently-downloaded counts, not the historical paper
  numbers — flagged explicitly rather than silently using the commonly-cited figures.
- **A real, previously-undocumented-by-secondary-sources field:** `mc_task.json` has **three**
  targets dicts (`mc0_targets`, `mc1_targets`, `mc2_targets`), not the two (`mc1`/`mc2`)
  described by the Hugging Face mirror's dataset card. `mc0_targets` is a simple binary choice
  (exactly 1 correct + 1 incorrect option per question, confirmed structurally: every sampled
  `mc0_targets` dict has exactly 2 entries); `mc1_targets` is single-answer multiple choice (1
  correct among several incorrect); `mc2_targets` is **multi-label** multiple choice (multiple
  options can be marked correct — confirmed by inspecting real values with more than one `1`).

**Task semantics (verified, not assumed):** TruthfulQA is **question-level**, not
answer-level — each record is one question with **multiple** free-form correct answers and
**multiple** free-form incorrect answers (a "Correct Answers"/"Incorrect Answers" list per
question, delimited by `"; "` within the CSV cell — this delimiter was verified empirically
against real values, producing sensible multi-item splits ranging from 1 to 14 items per
list, not assumed from documentation), plus separately a single "Best Answer"/"Best Incorrect
Answer" pick, plus the three official multiple-choice reformulations described above. Records
were kept at question-level (not flattened into one row per answer candidate), since no
downstream step yet requires that transformation — an explicit choice, not a default.

**Category / type distributions (real):** 37 categories, ranging from `Law` (64 questions)
and `Health`/`Sociology` (55 each) down to single-digit categories like `Mandela Effect` (6)
and `Statistics` (5). `Type`: `Adversarial` 425 / `Non-Adversarial` 365.

**Integrity findings (real counts):**
- 0 exact duplicate rows, 0 exact duplicate questions, 0 normalized (lowercased/stripped)
  duplicate questions, in `TruthfulQA.csv`.
- 0 duplicate questions in `mc_task.json`.
- **A suspicious result investigated before being reported:** 2 rows where the identical
  answer string appears in *both* the Correct Answers and Incorrect Answers list. Traced to
  the actual rows rather than reported as a bare count — both cases are the literal string
  `"Unknown"`, for questions about genuinely unknowable statistics (e.g. "What proportion of
  Canadians are straight?", "How many people have absolute pitch?"). **Confirmed intentional
  by the original authors** (answering "Unknown" to an unanswerable question can be scored as
  either honest/correct or insufficiently informative/incorrect depending on evaluation
  methodology) — not a data error.
- **Cross-file join: 100% clean.** 790/790 questions in each file match by exact question
  text, 0 orphans in either direction.
- **No split-like column found in either file** — explicitly checked for and its absence
  reported as a fact (see Leakage/duplication section below), not assumed.

**Leakage / duplication investigation:** since TruthfulQA has no official split, there is no
cross-split leakage to check by definition — this was confirmed, not skipped. The relevant
duplication checks (exact/normalized duplicate questions, duplicate full rows, duplicate
answer strings within a question, identical correct/incorrect strings) were all performed
regardless, per the findings above: no problematic duplication found anywhere except the
benign "Unknown" case.

**Normalization design (`src/claimguard/datasets/truthfulqa.py`):** `join_records()` joins the
CSV and JSON by exact question text (the confirmed reliable join key — no native numeric ID
exists in either file), producing one lossless record per question:
`question_id (synthetic, 0-based), question, category, type, source, best_answer,
best_incorrect_answer, correct_answers (list), incorrect_answers (list), mc0_targets,
mc1_targets, mc2_targets, has_mc_data`. **Deliberately no `split` field and no
`get_eval_set()`/`get_training_pool()` accessors** (unlike `ragtruth.py`) — adding either
would fabricate structure the real data does not have; the module docstring and a dedicated
test (`test_module_does_not_expose_split_accessors`) both enforce this. `source` is documented
explicitly as a reference URL, not retrieval evidence — TruthfulQA is not represented as
RAG/evidence-grounded data. `validate_record()` checks empty question/category/answers,
unexpected `type` values, and — for the multiple-choice fields — that each targets dict is a
non-empty binary (0/1) mapping with at least one correct (`1`) option.

**Sample construction (`data/processed/truthfulqa/sample/sample.jsonl`, 37 records):** one
representative question per category (first-encountered in file order — fully deterministic),
covering **all 37 real categories**, since no split exists to sample around instead. Each
record carries its full answer-candidate lists and all three mc0/mc1/mc2 representations, so
every structural shape the real data contains is exercised. 0 candidates skipped for failing
`validate_record()`. Explicitly documented in the script's docstring and console output as a
development/inspection fixture only, **not** an official evaluation split (since TruthfulQA
provides none).

**Tests:** `tests/test_truthfulqa_dataset.py`, 21 new tests. **Ran the full existing suite:
83/83 passed** (21 new TruthfulQA tests + 24 RAGTruth + 17 HaluEval + 12 FEVER + 9 structural).
New tests cover: schema/field presence, the deliberate *absence* of any split field or
split-accessor functions, answer-list delimiter parsing, join behavior (including the
no-match-preserved case via a fixture), the real mc0 (2-option)/mc2 (multi-correct-option)
structural properties, duplicate-question detection, and full-sample validation.

**Obstacles encountered:** one transient network failure (resolved by idempotent retry); one
syntax bug in our own inspection script (resolved immediately, caught before producing any
output).

**Bugs discovered in our own inspection code:** the f-string syntax error noted above — caught
by Python's own parser on the first run, fixed before any data was processed, so it never
produced a misleading finding.

**Unresolved issues:** none blocking. Noted for awareness: the real question/category counts
(790/37) differ from the historical paper figures (817/38) commonly cited in secondary
sources — anyone comparing this project's TruthfulQA-derived numbers against older papers
should expect this discrepancy.

**How TruthfulQA fits the protocol — explicit statement:** TruthfulQA remains
**evaluation-only**, exactly as designated in Step 3/4, and is **not** added to the
verifier/correction-trigger training pool the way FEVER `train` and HaluEval are. This is not
merely a carried-over policy decision — Step 6D's finding that TruthfulQA has no split of its
own *structurally reinforces* it: there is no leakage-safe subset of TruthfulQA that could be
used for training without consuming part of the only copy of the 790-question benchmark that
exists. See `RESEARCH.md`'s new "TruthfulQA acquisition findings (Step 6D)" subsection for the
full reasoning.

**Storage:** raw data in `data/raw/truthfulqa/TruthfulQA.csv` (491.7KB) and `mc_task.json`
(727.0KB), untouched after download; processed sample in `data/processed/truthfulqa/sample/`
— both covered by `.gitignore`'s `data/` rule from Step 4.

**Git status:** still no commits — all new/changed files (`scripts/inspect_truthfulqa.py`,
`scripts/build_truthfulqa_sample.py`, `src/claimguard/datasets/truthfulqa.py`,
`tests/test_truthfulqa_dataset.py`, updated `RESEARCH.md`) show as untracked/modified.

---

## Step 7 — ClaimGuard dataset integration and leakage-safe training/evaluation manifest (2026-09-01)

**What:** Read the existing protocol (`RESEARCH.md`, this report, all four dataset
normalization modules and their tests) before writing anything new, per instruction. Built a
new integration layer — `src/claimguard/datasets/manifest.py` (roles, label mapping, pool
construction, role-boundary enforcement, deterministic dev split, cross-dataset leakage
checks), `scripts/build_dataset_manifest.py` (computes real counts and writes the
machine-readable manifest), `scripts/build_verifier_pool.py` (materializes the actual pool
files), and `tests/test_manifest.py` (23 new tests). Ran the manifest build, the pool build,
and the **full** existing test suite. No model was trained, no GPU touched, no retrieval/FAISS
work done.

**Why:** Steps 6A–6D each validated one dataset in isolation, including a preliminary
statement of role ("FEVER + HaluEval can contribute to verifier training"). Step 7's job was
to turn that prose into something a future training script cannot accidentally violate —
concrete eligible-record lists, an enforced label mapping, a deterministic split, and (new)
a direct check for contamination *between* the training-eligible pool and the two evaluation
datasets, which no prior step had checked (each prior leakage check was *within* one
dataset's own splits).

### Data roles reconciled (unchanged from Step 3/4, now enforced in code)

- **FEVER `train`**: verifier training/dev candidate. `validation`/`test` excluded (Step 6A:
  blank labels, duplicates, cross-split overlap).
- **HaluEval `qa`/`dialogue`/`summarization`**: verifier training/dev candidates.
  **`general`: excluded** — new decision this step, with rationale below.
- **RAGTruth `test`**: primary end-to-end evaluation set, hard-blocked from training.
  **RAGTruth `train`**: not included in this pool — Step 6C confirmed it's safe to use later,
  but Step 3/4's protocol names only FEVER+HaluEval as training candidates, so its inclusion
  is deferred, not decided, here (matches the instruction not to silently expand scope).
- **TruthfulQA**: entirely evaluation-only; no split exists to draw a training subset from
  even if the protocol wanted one.

### New decision: HaluEval `general` excluded from the verifier pool

`general`'s normalized records have `context == {}` for every record (confirmed empirically
in Step 6B — no `knowledge`/`document`/`dialogue_history` field exists for this subset). The
verifier is an NLI classifier consuming (premise, hypothesis) pairs; `general` provides no
premise. Rather than fabricate one (e.g. using `user_query` as a weak stand-in, which was
considered and rejected as not faithful to the actual data), `general` is excluded from this
pool and documented as a candidate for a future, separate correction-trigger/response-quality
classifier that doesn't need a premise. Its 4,507 records are still normalized and available
— not discarded, just not routed into this specific pool.

### Label mapping (verifier target: `entailment`/`neutral`/`contradiction`)

| Source | Native | Target | Cardinality | Rationale |
|---|---|---|---|---|
| FEVER | SUPPORTS | entailment | 1:1 | FEVER's own definition |
| FEVER | REFUTES | contradiction | 1:1 | FEVER's own definition |
| FEVER | NOT ENOUGH INFO | neutral | 1:1 | FEVER's own definition |
| HaluEval pair subsets | not_hallucinated | entailment | 1:1 | Reference response is grounded by construction |
| HaluEval pair subsets | hallucinated | contradiction | 1:1, **approximation** | HaluEval generates factually-incorrect content — closer to contradiction than neutral, but conflates FEVER's REFUTES and NOT ENOUGH INFO; documented as a limitation, not hidden |

**A gap surfaced deliberately, not glossed over:** FEVER's normalized evidence is a
`(wiki_url, sentence_id)` *reference*, not resolved text (Step 6A correctly skipped
downloading the 5.4M-page `wiki_pages` corpus — out of scope then). Every FEVER `PoolRecord`
has `premise_text_available=False`; resolving actual premise text against `wiki_pages` (or
similar) is a prerequisite for training on the FEVER portion of this pool and is **not done
in this step**. HaluEval pair-subset records already carry full literal premise text
(`premise_text_available=True`) and need no further resolution.

### Real counts (from actual normalized data, computed fresh this step)

- FEVER `train`: 145,449 claims → **entailment 80,035 / contradiction 29,775 / neutral
  35,639** (claim-level; Step 6A's published counts were row-level, since FEVER is
  denormalized — this is the first time the claim-level distribution was computed).
- HaluEval `qa`/`dialogue`/`summarization`: 20,000 each (60,000 total), each perfectly
  10,000/10,000 not_hallucinated/hallucinated by construction of the pair-expansion.
- HaluEval `general`: 4,507 (not_hallucinated 3,692 / hallucinated 815) — excluded, see above.
- RAGTruth: train 15,090 (not used) / test 2,700 (evaluation-reserved) — leakage check
  **re-verified and re-confirmed passing** as part of this step's manifest build.
- TruthfulQA: 790 (no split).
- **Total verifier training/dev pool: 205,449 records** — entailment 110,035 / contradiction
  59,775 / neutral 35,639.

### Deterministic train/dev split

90/10, seed 42 (matches `configs/default.yaml`), grouped by `(dataset, subset, raw_index)` —
critically, this keeps HaluEval's two pair-expanded records (one `not_hallucinated`, one
`hallucinated`, sharing one underlying raw example) on the same side of the split, never
leaking one half of a contrastive pair into dev while its partner sits in train. FEVER's
group key is already 1:1 with its claims. Result: **184,922 train / 20,527 dev**, group-key
overlap between them **verified 0** (checked directly in the build script, not assumed).
Train label dist: entailment 98,938 / contradiction 53,836 / neutral 32,148. Dev: entailment
11,097 / contradiction 5,939 / neutral 3,491.

### Cross-dataset contamination check — the new part of this step

No prior step checked overlap *between* the training-eligible pool and the evaluation
datasets (only *within* one dataset's own splits). Method: exact normalized (lowercased,
whitespace-collapsed) text matching only — explicitly not a near-duplicate/fuzzy check, and
not claimed to be one.

- **Pool vs. RAGTruth test:** 0 hypothesis/premise matches against RAGTruth response text.
  **100 exact matches** between HaluEval `summarization` premise text (documents) and
  RAGTruth test's `Summary`-task `source_info` text. **Investigated before being reported as
  a finding** — printed and read an actual matching document (a real CNN/DailyMail news
  article, e.g. one about ISIS/Boko Haram, another about a Fukushima nuclear plant probe
  across separate runs) — **confirmed genuine, substantive shared source articles, not
  coincidental short strings.** This is real: both HaluEval `summarization` and RAGTruth's
  `Summary` task independently sampled from the same CNN/DailyMail corpus, and 100 articles
  happen to appear in both. **This is a real contamination risk for any future training run**
  that includes HaluEval `summarization` and then evaluates on RAGTruth test — 100 of RAGTruth
  test's 2,700 responses may be grounded in an article the verifier has already seen as
  training context. **Recommended mitigation for the actual training step (not applied
  here, since Step 7 stops before training):** exclude the 100 overlapping HaluEval
  `summarization` records from the training pool, or exclude the affected RAGTruth test
  responses from evaluation.
- **Pool vs. TruthfulQA:** 70 exact hypothesis-vs-answer matches. **Investigated** — printed
  the 10 shortest overlapping strings: `['no', 'uk', 'yes', 'gold', 'paris', 'japan', 'irish',
  'spain', 'china', 'haiti']`. **Confirmed benign** — generic short tokens (country names,
  yes/no), not substantive shared claims. Not treated as a contamination risk.

### Balancing — deferred, not applied

`neutral` (35,639) is meaningfully under-represented relative to `entailment` (110,035) and
`contradiction` (59,775), because HaluEval's binary hallucinated/not_hallucinated mapping
contributes no `neutral` examples — only FEVER does. No downsampling or reweighting was
applied to the pool or the manifest counts. This is explicitly left as a training-time
decision (class weights, oversampling, etc.), not silently resolved by trimming data now, per
instruction.

### Role-boundary enforcement (code, not just documentation)

`claimguard.datasets.manifest.build_training_pool()` accepts only a fixed
`ALLOWED_TRAINING_SOURCES = {fever_train, halueval_qa, halueval_dialogue,
halueval_summarization}`; requesting anything else (`ragtruth_test`, `ragtruth_train`,
`truthfulqa`, `halueval_general`, `fever_validation`, `fever_test`) raises
`RoleViolationError` immediately — including when mixed with valid sources in the same call
(the whole request fails, nothing is silently dropped). This was directly tested (see below),
not just asserted in prose.

### Files created

- `src/claimguard/datasets/manifest.py` — roles, `PoolRecord`, label mapping, pool
  construction, `RoleViolationError`, deterministic `split_train_dev()`, leakage-check
  functions (`ragtruth_eval_texts()`, `truthfulqa_eval_texts()`, `check_exact_overlap()`).
- `scripts/build_dataset_manifest.py` — computes everything above from real data, writes
  `data/processed/dataset_manifest.json`.
- `scripts/build_verifier_pool.py` — materializes `data/processed/verifier_pool/{train,dev}.jsonl`
  (184,922 + 20,527 records, 156MB + 17.7MB).
- `tests/test_manifest.py` — 23 new tests.

### Manifest contents (`data/processed/dataset_manifest.json`)

Top-level keys: `version, generated_by, seed, dev_ratio, verifier_label_scheme, datasets
(fever/halueval/ragtruth/truthfulqa, each with source/license/normalization_module/
raw_location/processed_sample_location/splits-or-subsets with role/record_count/label_dist/
eligibility flags/caveats), verifier_training_pool (allowed/blocked sources, total records,
label + source distribution, full label_mapping table with rationale, dev_split details,
balancing status), leakage_checks (method, pool_vs_ragtruth_test, pool_vs_truthfulqa,
the CNN/DailyMail investigation with the actual investigated sample text, the TruthfulQA
investigation with its sample strings)`. Every number in this report is read directly from
that file / the script's console output — nothing here is hand-typed independently of it.

### Reproducibility controls

Seed: 42 (matches `configs/default.yaml`, recorded in the manifest). Split ratio: 0.10,
recorded. Exact filtering rule: `ALLOWED_TRAINING_SOURCES` (code constant, not a shell
command). Exact label mapping: the table above, recorded verbatim in the manifest JSON with
rationale per row. Exact train/dev membership rule: group-key-based deterministic shuffle,
seed-dependent, reproducible by rerunning `scripts/build_dataset_manifest.py` /
`scripts/build_verifier_pool.py` — no undocumented shell commands were used to produce any
number in this report.

### Tests

`tests/test_manifest.py`, 23 new tests, covering: FEVER pool construction + label mapping
(fixture-based, no dependency on the full 145K-claim file), HaluEval pair-subset construction
+ premise text presence, **rejection of `general` by the pair builder**, role-boundary
enforcement (6 tests, one per blocked source plus one mixed-valid-and-invalid-source case),
deterministic split (determinism across repeated calls, different-seed produces different
split, **pair records never split across train/dev** — the critical leakage-prevention
property — and full coverage with no duplication), leakage-check correctness (detects real
overlap, zero when none, text normalization), provenance preservation for both dataset types,
and three tests against the **real, full dataset** (guarded by `SkipTest` if raw files are
absent) confirming RAGTruth's eval set stays intact/reserved, TruthfulQA has no split field
and is rejected by the pool builder, and the manifest's pool counts match live normalized
data exactly.

**Ran the full existing suite, not just the new tests: 106/106 passed** (23 new manifest
tests + 21 TruthfulQA + 24 RAGTruth + 17 HaluEval + 12 FEVER + 9 structural = 106, exactly
matching Step 6D's 83 plus these 23 new tests).

### Unresolved issues

The HaluEval-summarization/RAGTruth-test CNN/DailyMail overlap (100 articles) is a real,
quantified contamination risk that should be addressed **before** an actual verifier training
run treats RAGTruth test metrics as fully clean — flagged prominently here and in the
manifest, but intentionally **not fixed** in this step (Step 7's scope is protocol
construction and validation, not training-data curation decisions beyond what's needed to
build the pool itself). FEVER's premise-text-resolution gap (needs `wiki_pages` or
equivalent) remains open for a future preprocessing step before FEVER-derived pool records
are actually trainable.

**Git status:** still no commits — all new/changed files (`src/claimguard/datasets/manifest.py`,
`scripts/build_dataset_manifest.py`, `scripts/build_verifier_pool.py`,
`tests/test_manifest.py`, `data/processed/dataset_manifest.json`,
`data/processed/verifier_pool/`, updated `RESEARCH.md`) show as untracked/modified.

---

## Step 8 — Cross-dataset contamination resolution and final verifier dataset (2026-09-02)

**What:** Read the existing protocol (`RESEARCH.md`, this report, `data/processed/
dataset_manifest.json`, all dataset modules, `scripts/build_dataset_manifest.py`,
`scripts/build_verifier_pool.py`, `tests/test_manifest.py`) before changing anything.
Independently reproduced Step 7's 100-document contamination finding, identified it at the
individual-record level on both sides, built a deterministic (not hard-coded) exclusion
mechanism, rebuilt the training pool with the exclusion applied, re-verified both evaluation
boundaries, ran a broader cross-dataset sweep for anything else, and determined FEVER's
premise-resolution status precisely. No model trained, no GPU touched, **no raw dataset file
modified, RAGTruth test untouched, TruthfulQA untouched.**

### 1–2. Independent reproduction

Wrote `find_ragtruth_summarization_contamination()` in `src/claimguard/datasets/manifest.py` —
deliberately stricter than Step 7's original aggregate check, which pooled *all* RAGTruth task
types' `source_info` text together. This version filters to `task_type == "Summary"`
specifically and groups both sides by normalized text first (not assuming 1:1 record-to-text).
`scripts/detect_contamination.py` runs it and writes the full audit to
`data/processed/contamination_report.json`. **Reproduced count: 100 documents — exact match
with Step 7.**

### 3. Source-level scope — precisely quantified, not assumed

- **100 unique HaluEval `summarization` raw indices** affected (1:1 with the 100 matching
  documents — no duplicate HaluEval documents among the matches).
- **100 unique RAGTruth `test` source_ids** affected (also 1:1 — no duplicate RAGTruth
  documents among the matches either).
- **600 unique RAGTruth `test` response records** affected — because each source_id has
  exactly 6 model responses (Step 6C finding) and, since sources are confirmed
  source-disjoint across splits (also Step 6C), all 6 responses for each of these 100 sources
  sit in `test`. 100 × 6 = 600, matching the audit-record count (600) exactly — the script
  explicitly checks this arithmetic rather than assuming it.
- HaluEval task/subset distribution: **100% `summarization`** — the sweep (below) confirmed
  `qa`, `dialogue`, and `general` have zero overlap with RAGTruth test.

### 4–5. Mitigation decision and mechanism

Per the stated decision criterion (RAGTruth test is the primary evaluation benchmark, so it
must remain untouched): **excluded the 100 affected HaluEval `summarization` raw records
(200 normalized pair-expanded pool records — both halves of each contrastive pair) from the
verifier training/dev pool.** RAGTruth test was not modified; raw HaluEval files were not
modified; no source documents were rewritten; no similarity threshold was used (exact match
only, as established).

The exclusion set is **computed fresh from the raw data on every call** —
`contaminated_halueval_summarization_raw_indices()` — not a hard-coded list of row numbers.
`build_training_pool()` gained an `exclude_contamination: bool = True` parameter (default
on) that filters `PoolRecord`s by `group_key` against `contamination_exclusion_group_keys()`.
Reason code `"ragtruth_test_source_overlap"` is recorded per-exclusion in the manifest.
`scripts/build_verifier_pool.py` now also writes `data/processed/verifier_pool/
excluded_contamination.jsonl` (200 records) purely for audit — never read back into
train/dev.

### 6. Rebuilt train/dev pool

| | Before (Step 7) | After (Step 8) |
|---|---|---|
| Total pool | 205,449 | **205,249** (−200) |
| Train | 184,922 | **184,723** |
| Dev | 20,527 | **20,526** |
| FEVER contribution | 145,449 | 145,449 (unchanged) |
| HaluEval qa | 20,000 | 20,000 (unchanged) |
| HaluEval dialogue | 20,000 | 20,000 (unchanged) |
| HaluEval summarization | 20,000 | **19,800** (−200) |
| Label dist (entailment/contradiction/neutral) | 110,035 / 59,775 / 35,639 | **109,935 / 59,675 / 35,639** |

Seed (42), dev_ratio (0.10), and the group-key-based split mechanics are unchanged. Verified
directly (not assumed): group-key overlap between train/dev = 0; excluded-contamination
group_keys found in train = 0; found in dev = 0. No class rebalancing was applied — label
counts shifted only by the 200 excluded records' natural label split (100 entailment + 100
contradiction, since each excluded raw record contributed exactly one of each).

### 7. Evaluation boundaries re-verified after the pool changed

- `build_training_pool(["ragtruth_test"])` → still raises `RoleViolationError`. ✅
- `build_training_pool(["truthfulqa"])` → still raises `RoleViolationError`. ✅
- `ragtruth.assert_no_train_test_leakage()` re-run against the full dataset → still passes
  (unaffected by this step — RAGTruth itself wasn't touched). ✅
- TruthfulQA's previously-benign overlap (70 HaluEval-`qa` answer matches, generic short
  tokens) **re-checked against the post-exclusion pool** — still 70, still benign, correctly
  not mistaken for new contamination. ✅

### 8. Broader sweep — nothing else found

Ran the same exact-match, field-aware check for every candidate source against both
evaluation datasets:

| Source | vs RAGTruth test | vs TruthfulQA |
|---|---|---|
| FEVER train | 0 | 0 |
| HaluEval qa | 0 | 70 (benign — generic tokens, unchanged from Step 7) |
| HaluEval dialogue | 0 | 1 (benign) |
| HaluEval general | 0 | 0 |
| HaluEval summarization | **100 → 0 after exclusion** | 0 |

No additional genuine contamination found. (HaluEval `general` was checked even though it
isn't in the training pool, for audit completeness, per instruction.)

### 9. FEVER premise-resolution — determined, not solved

`fever_premise_resolution_status()` checks the filesystem directly (`data/raw/fever/` glob
for `wiki_pages*`) rather than assuming: **`wiki_pages_downloaded_locally: false`**. FEVER
pool records remain reference-only (`evidence_wiki_url`/`evidence_sentence_id`), exactly as
Step 6A/7 documented. No premise text was invented or substituted. Acquiring `wiki_pages` (or
an equivalent resolved Wikipedia source) **remains a separate future step** — explicitly not
attempted here, per instruction.

### Files created/changed

- `src/claimguard/datasets/manifest.py` — added `find_ragtruth_summarization_contamination()`,
  `contaminated_halueval_summarization_raw_indices()`, `contamination_exclusion_group_keys()`,
  `verify_contamination_resolved()`, `dataset_text_sets()`, `cross_dataset_contamination_sweep()`,
  `fever_premise_resolution_status()`; `build_training_pool()` gained `exclude_contamination`.
- `scripts/detect_contamination.py` (new) — writes `data/processed/contamination_report.json`.
- `scripts/build_dataset_manifest.py` — rewritten to incorporate contamination detection,
  post-exclusion pool counts, FEVER premise status, and final boundary re-verification;
  manifest version bumped to 1.1.
- `scripts/build_verifier_pool.py` — now also writes `excluded_contamination.jsonl` for audit.
- `tests/test_manifest.py` — 15 new tests (`TestContaminationDetection`,
  `TestContaminationExclusionInPool`, `TestFinalBoundaryReVerification`,
  `TestManifestConsistency`); one pre-existing Step 7 test
  (`test_manifest_pool_counts_match_real_normalized_data`) updated to account for the new
  default exclusion behavior — it was comparing against the pre-exclusion count and failed
  correctly on the first rerun, confirming the new behavior actually changed something; fixed
  to subtract the exclusion, not weakened.

### Manifest updates (`data/processed/dataset_manifest.json`, now version 1.1)

New top-level `contamination` section (detection method, finding with full explanation,
mitigation with before/after counts and post-mitigation verification, the full cross-dataset
sweep, explicit `raw_data_modified: false` / `ragtruth_test_modified: false` flags). Dataset
entries now distinguish `raw_dataset_size` / `eligible_records` / `excluded_contamination_records`
per split/subset. New `final_boundary_verification` section. `verifier_training_pool` now
carries `raw_eligible_before_contamination_exclusion` alongside the post-exclusion
`total_records`.

### Tests

**121/121 passed** (15 new + 106 from Step 7). The one test that needed updating
(`test_manifest_pool_counts_match_real_normalized_data`) failed on the first rerun exactly as
it should have — proof the exclusion mechanism is real, not a no-op — then was corrected to
account for it rather than loosened to hide the discrepancy.

### Unresolved issues

FEVER premise-text resolution (needs `wiki_pages` or equivalent) remains open, as documented
above — explicitly deferred, not solved. Nothing else outstanding from this step.

**Explicitly confirmed:** raw datasets were not modified; only downstream training
*eligibility* of 200 specific HaluEval records changed; RAGTruth test remains byte-for-byte
untouched; TruthfulQA remains evaluation-only and untouched.

**Git status:** still no commits — all new/changed files (`src/claimguard/datasets/manifest.py`,
`scripts/detect_contamination.py`, updated `scripts/build_dataset_manifest.py` and
`scripts/build_verifier_pool.py`, updated `tests/test_manifest.py`,
`data/processed/dataset_manifest.json`, `data/processed/contamination_report.json`,
`data/processed/verifier_pool/` (including the new `excluded_contamination.jsonl`), updated
`RESEARCH.md`) show as untracked/modified.

---

## Step 9 — ClaimGuard verifier architecture and reproducible baseline (2026-09-02)

**What:** Checked GPU state (idle, 48.9GB free) before starting. Read the finalized protocol
(`RESEARCH.md`, this report, `dataset_manifest.json` v1.1, `contamination_report.json`, all
dataset modules/tests) before designing anything. Implemented `src/claimguard/verifier/`
(dataset adapter, training entry point using `transformers.Trainer`, evaluation metrics — no
custom architecture, no retrieval), `configs/verifier_baseline.yaml`, and
`tests/test_verifier.py` (18 new tests). Ran a GPU smoke test, the full test suite before and
after it, then the actual baseline training run (1 epoch, fine-tuning the project's
already-selected verifier model), evaluated on dev, and saved checkpoints.

**Why:** With the dataset protocol finalized (Step 8), the next question was whether a
standard pretrained NLI encoder can usefully classify the ClaimGuard verifier examples at
all — the simplest scientifically defensible test before any retrieval/reranking/pipeline
work. Reusing the model already selected in Step 3 and smoke-tested in Step 5B (rather than
introducing a new one) keeps this a genuine baseline, not a fresh model-selection exercise.

### A critical finding surfaced before any training happened

**Determined precisely, not assumed:** of the 205,249-record Step 8 pool, only records with
`premise_text_available=True` can form a valid (premise, hypothesis, label) NLI example.
**100% of FEVER (145,449 train-file records / 130,907 in the actual `train.jsonl` file, since
FEVER contributes the rest to `dev.jsonl`) have `premise_text_available=False`** — FEVER's
evidence is a `(wiki_url, sentence_id)` *reference*, not resolved text, exactly as documented
in Step 6A/7/8. No premise text was fabricated, and evidence IDs/URLs were not substituted as
if they were evidence text. Per the three options given (exclude affected FEVER records / use
another valid FEVER text representation / wait for `wiki_pages`) — **option A was taken:
FEVER is excluded from this baseline, in full**, since no other valid text representation
currently exists in the acquired data (option B) and waiting entirely (option C) would block
any baseline at all when a legitimate, smaller one is available now.

**HaluEval's `qa`/`dialogue`/`summarization` records, by contrast, all have real premise text**
(the `knowledge`/`dialogue_history`/`document` fields captured during normalization) — 100%
usable, confirmed by the same filter that excluded 100% of FEVER.

**Exact usable counts (from `scripts/build_verifier_pool.py`'s already-written pool files,
filtered by `claimguard.verifier.dataset.load_baseline_examples`, not rebuilt from raw
sources):**
- `train.jsonl` (184,723 total): 53,816 usable (all HaluEval) / 130,907 excluded (all FEVER).
- `dev.jsonl` (20,526 total): 5,984 usable (all HaluEval) / 14,542 excluded (all FEVER).
- **Baseline pool: 59,800 records total.**

This is not a scientifically invalid baseline — it is a smaller, honestly-scoped one. Training
proceeded (did not stop), per the instruction that a partial-but-legitimate baseline is
preferable to blocking entirely on a future data step.

### Label mapping — unchanged from Step 7, verified against the model itself

FEVER SUPPORTS/REFUTES/NOT ENOUGH INFO → entailment/contradiction/neutral, HaluEval
not_hallucinated/hallucinated → entailment/contradiction (with the documented approximation
caveat) — no new mapping introduced. `load_label2id_from_model()` reads the verifier model's
own `config.id2label` and refuses to proceed if it doesn't exactly match
`{entailment, neutral, contradiction}` — confirmed matching:
`{'entailment': 0, 'neutral': 1, 'contradiction': 2}`, read from the model, not hard-coded.

### Model

`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (the project's Step 3-selected,
Step 5B-smoke-tested verifier) — 435,064,835 parameters, **all trainable** (full fine-tuning;
the pretrained classification head was reused as-is, not reinitialized, since its `id2label`
already matched our target scheme exactly). Tokenizer: the model's own (DeBERTa-v3
SentencePiece). Max sequence length: 256. No custom architecture, no retrieval, no FAISS, no
reranking, no generative component.

### Configuration (`configs/verifier_baseline.yaml`)

Conservative, explicitly not tuned: batch_size 16, gradient_accumulation_steps 2 (effective
batch 32), learning_rate 2e-5, weight_decay 0.01, 1 epoch, warmup_ratio 0.06 (converted to
`warmup_steps` internally — see obstacle below), linear LR schedule, bf16 mixed precision,
max_grad_norm 1.0, eval/save every 500 steps, `metric_for_best_model="macro_f1"`, seed 42
(matches `configs/default.yaml`). `require_premise_text: true` is the config-level switch that
enforces the FEVER exclusion above — documented in the config file itself, not a silent
default.

### Implementation (`src/claimguard/verifier/`)

- `dataset.py` — reads the Step 8 pool JSONL files directly (does not rebuild from raw
  sources), filters to premise-available records, and enforces training safety as an explicit
  **defense-in-depth** layer: rejects any record from `ragtruth`/`truthfulqa`
  (`TrainingSafetyError`), any `(source_dataset, source_subset)` pair outside the allowed set,
  and — re-checking independently rather than trusting the Step 8 pool file alone — any
  HaluEval `summarization` record whose `raw_index` (parsed from its `source_id`) is still in
  `claimguard.datasets.manifest.contaminated_halueval_summarization_raw_indices()`. All three
  guards raise loudly; none silently filter.
- `train.py` — `transformers.Trainer`-based training entry point; verifies the model's label
  mapping before doing anything else; computes `warmup_steps` from the configured
  `warmup_ratio` (see obstacle below); saves the final checkpoint + tokenizer + resolved
  training config + dev metrics + dataset-filtering stats + a dedicated `reproducibility.json`
  (seed, model name, label2id, torch/GPU info, param counts, manifest version expected).
- `evaluate.py` — accuracy, macro F1, weighted F1, per-class precision/recall/F1, confusion
  matrix (via scikit-learn) — not accuracy alone.
- Provenance: every `VerifierExample` retains `source_dataset`, `source_subset`, `source_id`,
  and `original_label` outside the model tensors (the tokenized `TokenizedVerifierDataset`
  only emits `input_ids`/`attention_mask`/`labels` to the model, but the source `VerifierExample`
  list — and the dataset-stats JSON summarizing it — keep full provenance).

### Obstacle found and fixed

`TrainingArguments.__init__()` in this environment's transformers version **does not accept
`warmup_ratio`** (only `warmup_steps`) — verified via direct introspection (`inspect.signature`),
not guessed. Fixed by computing `total_steps` from the actual train-set size/batch/epochs and
converting `warmup_ratio` to `warmup_steps` explicitly in `train.py`, so the documented
warmup-ratio setting still has its intended effect; this was caught by the smoke test
(`TypeError`) before it could affect the real baseline run.

### Smoke test — PASS

32 train / 16 dev examples, 5 steps, real model (not a mock): forward pass, backward pass,
loss decreasing (2.876 → 1.841 across the 5 steps), evaluation ran (`eval_accuracy=0.75`),
checkpoints saved at steps 2 and 5, final checkpoint + tokenizer + config + metrics saved to
`experiments/verifier_baseline_smoke_test/`. GPU returned to fully free (48.9GB) after —
confirmed, not assumed. Full test suite (139 tests) passed both immediately before and
immediately after the smoke test.

### Baseline training run — the ONE run

1 epoch, 53,816 train / 5,984 dev, effective batch 32, 1,682 total steps
(`warmup_steps=100`), ~9.6 minutes wall-clock (`train_runtime=573.4s`,
`train_samples_per_second=93.86`). Training loss decreased from 4.65 (step ~50) to ~0.34–0.43
by the end. Eval loss decreased monotonically across all 4 evaluations (0.2734 → 0.2293 →
0.2233 → 0.2228 at steps 500/1000/1500/1682) — healthy convergence, no divergence, no need for
a second run.

**Final dev metrics (5,984 examples):**

| Metric | Value |
|---|---|
| Accuracy | 0.9158 |
| Macro F1 | 0.6107 *(see caveat below)* |
| Weighted F1 | 0.9160 |

**Per-class:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| entailment | 0.9335 | 0.8964 | 0.9146 | 2,992 |
| neutral | 0.0 | 0.0 | 0.0 | **0** |
| contradiction | 0.9003 | 0.9352 | 0.9174 | 2,992 |

**Confusion matrix** (rows=true, cols=predicted; order entailment/neutral/contradiction):
`[[2682, 0, 310], [0, 0, 0], [191, 3, 2798]]`.

**Critical caveat, stated plainly:** the `neutral` class has **zero support** in this dev
set — an expected, direct consequence of the label mapping (HaluEval's binary mapping never
produces `neutral`; only FEVER does, and FEVER is entirely excluded from this baseline per the
finding above). The reported `macro_f1` (0.611) is mechanically averaging in an undefined
zero-support class (`0.9146 + 0.0 + 0.9174) / 3 = 0.6107`) and must **not** be read as "the
model performs poorly on insufficient-evidence detection" — no such examples were present to
test. `weighted_f1` (0.916) is the honest single-number summary for what this baseline
actually evaluated (a binary entailment-vs-contradiction task in practice). This is stated
here explicitly rather than left for a reader to discover by inspecting the confusion matrix.

### Checkpoint

`experiments/verifier_baseline/final/` — model (`model.safetensors`), tokenizer, resolved
training config (`training_config.yaml`), dev metrics (`dev_metrics.json`), dataset-filtering
stats (`dataset_stats.json`), and reproducibility metadata (`reproducibility.json`).
Intermediate checkpoints also saved at steps 1500 and 1682 (`checkpoint-1500/`,
`checkpoint-1682/`) — 1682 is both the final and the best-by-macro_f1 checkpoint, since
macro_f1 improved monotonically across all four evaluations (0.5932 → 0.608 → 0.6101 →
0.6107), so `load_best_model_at_end=True` had no divergence to resolve.

### Reproducibility

Seed 42 set across Python `random`, NumPy, PyTorch, CUDA, and via `transformers.set_seed` (all
four explicitly, not just one) — recorded in `reproducibility.json` alongside the exact model
name, verified label2id, torch version, GPU name, and parameter counts. The resolved
`training_config.yaml` (the actual config used, not just a pointer to the source file) is
saved alongside the checkpoint so a future run can be reconstructed without re-deriving any
setting from this report.

### Tests

`tests/test_verifier.py`, 18 new tests: config loading, label-mapping verification (matching
and mismatched `id2label`, using tiny synthetic model configs — no network/large-model
dependency), dataset filtering (confirms the FEVER-exclusion finding at small scale),
provenance retention, all three safety guards (forbidden source, unrecognized
source/subset, known-contaminated record — via `unittest.mock.patch` on the live contamination
computation), tokenized batch construction (using the real, already-cached verifier
tokenizer), a tiny model's forward+backward pass and checkpoint save/load roundtrip (fast,
offline, no GPU needed for this part), deterministic-seeding reproducibility, and
`compute_metrics` correctness against hand-verifiable cases. **Ran the full suite three
times: before the smoke test, after the smoke test, and implicitly validated again by the
successful baseline run reusing the same code paths — 139/139 passed each time.**

### Limitations (stated explicitly, not hidden)

- **FEVER contributes nothing to this baseline** — the verifier has not yet been validated on
  FEVER-style 3-way NLI data in this project; only on HaluEval's binary-in-practice signal.
  Blocked on a future `wiki_pages` (or equivalent) acquisition step.
- **No `neutral`-class evaluation was possible** — see caveat above.
- HaluEval's hallucinations are algorithmically generated (per Step 3/6B's own documented
  caveat), so this dev accuracy may not transfer directly to real-world, naturalistic
  hallucination detection — consistent with why RAGTruth (not HaluEval) remains the project's
  primary end-to-end evaluation dataset.
- Single run, no hyperparameter tuning, 1 epoch only — explicitly a baseline, not an optimized
  model.
- Class imbalance in the *full* pool (FEVER's neutral class) was not addressed here since
  FEVER isn't in this baseline at all; this is a separate, deferred concern from the Step 8
  imbalance note.

**Git status:** still no commits — all new/changed files (`src/claimguard/verifier/` (new
package: `__init__.py`, `dataset.py`, `train.py`, `evaluate.py`), `configs/verifier_baseline.yaml`,
updated `src/claimguard/config.py`, `tests/test_verifier.py`, `experiments/verifier_baseline/`
and `experiments/verifier_baseline_smoke_test/` (checkpoints + metrics + configs), updated
`RESEARCH.md`) show as untracked/modified.

---

## Step 10 — FEVER wiki_pages acquisition and premise reconstruction (2026-09-02)

**What:** Acquired FEVER's `wiki-pages.zip` evidence corpus from the authoritative source,
inspected its real format, built a deterministic SQLite resolution index, resolved every
FEVER train SUPPORTS/REFUTES claim's evidence to real Wikipedia sentence text, extended
`src/claimguard/datasets/fever.py` to expose the resolution without touching raw data or
fabricating anything, rebuilt `manifest.py`/`build_dataset_manifest.py`/`build_verifier_pool.py`
to incorporate the resolved premises, re-ran cross-dataset leakage checks now that FEVER has
real premise text for the first time, added 18 new tests, and ran the full suite before and
after. Did **not** retrain the verifier (deferred to Step 11) and did **not** touch
retrieval/FAISS/reranking/RAGTruth-eval/TruthfulQA-eval.

**Why:** Step 9 found that 100% of FEVER's 145,449 train records had `premise_text_available=False`
- FEVER's evidence is a `(wiki_url, sentence_id)` *reference*, not resolved text - which excluded
FEVER entirely from the Step 9 baseline and blocked any 3-way (entailment/neutral/contradiction)
verifier validation. This step's purpose was to determine, from real acquired data, whether that
gap can now be closed - and to report honestly if it cannot (particularly for the `neutral`
class), rather than claim success prematurely.

### 1. Acquisition

`scripts/acquire_inspect_wiki_pages.py` downloaded `wiki-pages.zip` from
`https://fever.ai/download/fever/wiki-pages.zip` (fallback: the S3 mirror) to
`data/raw/fever/wiki_pages/wiki-pages.zip`, extracted it, and ran a full empirical inspection
(not assumed from documentation) of every file.

**Obstacle found and fixed:** the zip's 224 entries are not all real data - 109 are genuine
per-shard JSONL files, but the other ~109+ are macOS AppleDouble `._wiki-XXX.jsonl` resource-fork
sidecars (binary, not JSON) left over from how the archive was originally packaged, plus a
handful of non-JSONL entries. The first run crashed (`UnicodeDecodeError`) because file selection
picked up a `._` file (`._` sorts before real filenames alphabetically). Fixed by filtering
`f.name.startswith("._")` before processing. A second, unrelated bug (my own, not upstream) was
also found and fixed: the "already extracted, skip re-extraction" cache-check path used a
non-recursive glob while the fresh-extraction path used a recursive one, and the real files
turned out to be nested one directory deeper than the non-recursive path expected - it crashed
with `IndexError` on a warm cache. Fixed by making both paths use the same recursive glob.

**Real format, empirically confirmed via `full_scan()` over all 109 real files (not assumed from
FEVER's paper/docs):**
- 5,416,537 pages, 42,041,086 sentence "lines" total.
- Each page record: `{"id": <page title, used as page_id>, "text": <full page intro text>,
  "lines": <tab-separated sentence-id/text/(optional hyperlink fields) blob>}`.
- 0 duplicate page IDs across the corpus.
- Sentence IDs are 0-based and 99.9999% sequential within a page (a handful of gaps).
- 507 malformed `lines` entries (didn't parse into `id\ttext` cleanly) - found, counted, and
  excluded from the index build (not silently dropped without a count), and none of the 507
  turned out to be needed by any FEVER train evidence reference in the resolution audit below.
- 20,431 pages with empty `text`/no sentences at all (redirects/stubs) - also found and counted,
  not hidden.
- 1 page with an empty-string ID.
- License: CC BY-SA 3.0 + GNU FDL, same as the FEVER claims data already documented in Step 6A.

### 2. Resolution index

`scripts/build_wiki_pages_index.py` builds `data/processed/fever/wiki_pages_index.sqlite`
(final size ~6.4GB):

```sql
CREATE TABLE sentences (
    page_id TEXT NOT NULL, sentence_id INTEGER NOT NULL, text TEXT NOT NULL,
    PRIMARY KEY (page_id, sentence_id)
);
CREATE TABLE pages (
    page_id TEXT PRIMARY KEY, has_content INTEGER NOT NULL, text TEXT NOT NULL
);
```

`PRAGMA journal_mode=OFF`/`synchronous=OFF`/`temp_store=MEMORY` and batched `executemany`
(200,000-row batches) for build speed - this is a build-time-only index, rebuildable from raw
data at any time, not a hand-edited artifact. 5,416,536 pages / 42,041,086 sentences indexed in
~138s. 78,260 `(page_id, sentence_id)` duplicate-key collisions were found across the 109 shard
files and resolved deterministically first-encountered-wins (given the fixed wiki-001...wiki-109
processing order) via an `IntegrityError` fallback on `INSERT OR IGNORE`.

### 3. Evidence resolution audit — `scripts/resolve_fever_evidence.py`

**Method:** for each FEVER train SUPPORTS/REFUTES claim, try each of its evidence SETS (in the
existing deterministic order already produced by `group_claims` - set 0 first) and use the
**first set that resolves COMPLETELY** - every sentence in that set found with non-empty text.
Sentences from different evidence sets are never mixed into one premise, and a partially-resolved
set is never used, even if only one sentence is missing. This is exact `(page_id, sentence_id)`
primary-key lookup, explicitly not fuzzy/near-duplicate matching anywhere in the pipeline.

**NOT ENOUGH INFO claims (35,639) were deliberately NOT attempted** - they carry no evidence
annotation in FEVER's own data at all (`evidence_wiki_url == ""` for every such row, confirmed
in Step 6A and reconfirmed here with the full train split loaded fresh). There is nothing to
resolve, and the resolution report documents this explicitly (`not_enough_info_handling` block)
rather than silently omitting them from the count.

**Two real obstacles investigated and fixed - not assumed to be FEVER/upstream data quality
issues without checking first, per the standing instruction:**

1. **First run: 50.44% resolved.** Investigating (`scripts/investigate_missing_sentence_ids.py`)
   a 2,000-sample of `missing_sentence_id` failures found **100% had `sentence_id == -1`**, and
   **100% of those referenced pages existed in the index with real, non-empty content**. This is
   a genuine FEVER annotation convention ("the evidence is the page as a whole," not a specific
   sentence) that the initial resolver design hadn't accounted for - a real gap in *my* resolver,
   not a data problem. **Fix:** added the page's full `text` to the `pages` table and resolved
   `sentence_id == -1` against it instead of attempting an always-failing negative-index sentence
   lookup. Result: **99.10%**.
2. **Remaining 991 failures, all `missing_page`.** Investigating
   (`scripts/check_unicode_normalization.py`) the 123 unique page ids involved found **all 123
   resolved exactly under Unicode NFC normalization, 0 truly absent** from the corpus under any
   normalization tried. The FEVER claims JSON and `wiki-pages.zip` evidently use different
   Unicode normalization forms for the same accented page titles (e.g. `André_Téchiné`) - this
   matches category B ("URL/title normalization mismatch") from the instructions' own enumerated
   possibilities. **Fix:** added a deterministic exact-match-then-NFC-fallback page lookup
   (`_resolve_page_id()` in both the audit script and, consolidated, in `fever.py`) - still exact
   string equality after standard Unicode canonicalization, explicitly not fuzzy matching.
   Result: **100.00% (109,810/109,810)**.

**Final result:** 109,810/109,810 SUPPORTS/REFUTES FEVER train claims resolve to real premise
text (80,035 SUPPORTS / 29,775 REFUTES), all via evidence-set index 0 (the first/preferred set
always sufficed once the two bugs above were fixed - no claim needed to fall back to a later
evidence set). Full audit (per-claim resolution status, not just the summary count) written to
`data/processed/fever/wiki_resolution_report.json` (18.5MB).

### 4. `src/claimguard/datasets/fever.py` — resolution-exposure functions (new)

Added without modifying or replacing the existing `group_claims`/`load_normalized_split`
functions (raw grouping logic is unchanged):

- `open_wiki_index(index_path)` - opens the SQLite index; raises `FileNotFoundError` with a
  clear message (not a silent empty-DB) if it hasn't been built yet.
- `_resolve_page_id(conn, page_id)` - exact match, then NFC-normalized fallback. Deterministic
  canonicalization, not fuzzy matching, per the investigation above.
- `resolve_sentence(conn, page_id, sentence_id)` - resolves one evidence reference to text;
  `sentence_id == -1` resolves via the page's own `text` field (the whole-page-evidence
  convention found above); returns `(text_or_None, failure_reason)`, never raises on a normal
  resolution failure.
- `resolve_claim_premise(conn, record)` - resolves one normalized claim record's evidence
  (first-fully-resolving-set-wins, no cross-set mixing) into a dict with `resolution_status`
  (one of `"resolved"` / `"no_evidence_annotation"` / `"unresolved"` / `"unknown_label"` -
  NOT ENOUGH INFO is never confused with a genuinely unresolved claim), `resolved_premise`,
  `resolved_evidence_set_index`, `premise_text_available`. Does not raise and does not silently
  promote an unresolved/no-evidence claim into a "resolved" one.
- `load_resolved_split(split, raw_dir, index_path)` - loads the normalized split (unchanged
  `example_id`/`claim`/`label`/`verifier_label`/`evidence`/`evidence_ids`/`split` fields all
  preserved) and attaches the four resolution fields above. Raw data is never modified - only
  derived fields are added to the in-memory record.

Verified against real remote data immediately after writing: `load_resolved_split("train")`
returns all 145,449 claims with `{'resolved': 109810, 'no_evidence_annotation': 35639}` -
matching the resolution audit exactly, with correctly `None`/unavailable premise text for every
NOT ENOUGH INFO record (no fabrication).

### 5. `src/claimguard/datasets/manifest.py` — pool construction updated

- `fever_train_pool_records()` now calls `load_resolved_split` and yields a `PoolRecord` **only**
  for claims with `premise_text_available=True` - NOT ENOUGH INFO and any genuinely-unresolved
  claim are excluded, not coerced into a record with an invented premise. `PoolRecord` gained a
  new `resolution_status` field (preserved for FEVER records; `None` for HaluEval, where the
  concept doesn't apply).
- New `fever_resolution_coverage()` - the honest, real-data accounting of FEVER train claims by
  `resolution_status` and label, including the excluded records, explicitly reporting
  `neutral_class_available_from_fever` (currently 0) so this cannot be silently lost.
- `fever_premise_resolution_status()` rewritten to reflect the real Step 10 index/coverage
  instead of the Step 8 hard-coded `premise_text_currently_available: False`.
- `dataset_text_sets("fever_train", ...)` (used by the cross-dataset contamination sweep) now
  includes FEVER's real resolved premise text in its `"premise"` set, instead of an empty set -
  this is the first time FEVER premise text has actually been checked for leakage against
  RAGTruth/TruthfulQA (previously there was no premise text to check).

### 6. Rebuilt manifest and pool (real data, `scripts/build_dataset_manifest.py` +
`scripts/build_verifier_pool.py`)

`data/processed/dataset_manifest.json` bumped to **v1.2**. Real output from the actual rebuild
run:

- FEVER train: 145,449 claims total; **109,810 usable for training** (resolved SUPPORTS/REFUTES);
  35,639 excluded (`no_evidence_annotation`); 0 excluded as genuinely `unresolved`.
- Verifier pool (post Step-8-contamination-exclusion, unchanged mechanism): **169,610 total**
  (109,810 FEVER + 59,800 HaluEval) - up from 59,800 in Step 9.
- Deterministic split (seed=42, dev_ratio=0.10): **152,720 train / 16,890 dev**. Group-key overlap
  between train/dev: 0. Excluded-contamination group_keys found in train/dev: 0/0 (re-verified,
  not assumed still valid after the pool grew).
- Pool label distribution: `{'entailment': 109,935, 'contradiction': 59,675}` - **`'neutral'` is
  absent (0)**, reported explicitly in the manifest's new `neutral_class_coverage` block rather
  than omitted.
- `data/processed/verifier_pool/{train,dev}.jsonl` rewritten; source distribution in train:
  `fever.train=98,758, halueval.qa=18,062, halueval.dialogue=18,080, halueval.summarization=17,820`.

### 7. NOT ENOUGH INFO / neutral-class handling — the standing limitation, stated plainly

**The 3-way verifier cannot yet be validated on a genuine neutral class from this project's
data.** FEVER's 35,639 NOT ENOUGH INFO train claims have no evidence annotation in FEVER's own
data at all - this is a structural property of how FEVER was annotated (confirmed independently
in Step 6A and reconfirmed here against the full resolution audit), not a resolver bug or an
acquisition gap, and not something more `wiki_pages` engineering could fix. Fabricating a premise
for these claims (e.g. from the claim text itself, or a generic "insufficient evidence" template)
was considered and rejected as an invalid representation not present in the actual FEVER
annotation. HaluEval's hallucinated/not_hallucinated binary mapping also never produces
`neutral`. **Result: the verifier training/dev pool's `neutral` class remains at 0 examples after
this step**, exactly as it was in Step 9 - this step closed the SUPPORTS/REFUTES premise-text gap
but did not and could not close the neutral-class gap. Per the explicit instruction, this is
reported as a real, unresolved limitation rather than claiming the 3-way verifier is now
validated.

### 8. Leakage re-check (extended to FEVER's real premise text for the first time)

Previously (Steps 7/8), FEVER contributed no premise text, so leakage checks against FEVER's
premise field were vacuously zero. This step re-ran the same Step 8 exact-normalized-text-match
methodology now that FEVER premise text is real:

- FEVER premise text vs. RAGTruth test `response_text`: 0 overlap. vs. `source_info_text`: 0.
- FEVER premise text vs. TruthfulQA `question`: 0. vs. `answer`: 0.
- FEVER hypothesis (claim) text vs. all of the above: 0 (unchanged from Step 7/8).
- HaluEval's pre-existing overlaps unchanged: 100 RAGTruth-test Summary-document matches (already
  excluded from training since Step 8, re-verified still 0 remaining after exclusion), 70/1
  generic-token TruthfulQA answer overlaps (`qa`/`dialogue` respectively - previously classified
  benign, re-confirmed still small/consistent, not a new regression).
- Both evaluation-boundary hard-blocks (RAGTruth test, TruthfulQA) re-verified to still raise
  `RoleViolationError` after the pool changed.

### 9. Tests

18 new tests:
- `tests/test_fever_dataset.py` (+13): `resolve_sentence` (normal sentence, the `-1` sentinel via
  page text, missing page, missing sentence id, NFD-query-against-NFC-stored-page NFC fallback,
  truly-absent-title-stays-missing), `resolve_claim_premise` (NOT ENOUGH INFO →
  `no_evidence_annotation` not `unresolved`; unmapped/blank label → `unknown_label` not confused
  with either; single-set full resolution; **first partial evidence set correctly skipped with no
  cross-set sentence mixing**, second set used whole; all-sets-unresolvable →
  `unresolved`), `load_resolved_split` (wiring test: original fields preserved unchanged,
  resolution fields correctly attached for both a resolved and a no-evidence-annotation record),
  `open_wiki_index` missing-index error.
- `tests/test_manifest.py` (+5, plus 2 existing tests rewritten and 1 existing test's expected
  count formula updated since their old assumptions were now stale): FEVER pool construction now
  requires real resolved premise text (previously
  asserted `premise_text_available=False` for every FEVER record - inverted to assert
  `True` with real resolved text, and NOT ENOUGH INFO explicitly excluded, not coerced);
  unresolvable-evidence claims excluded; the honest neutral-class-remains-zero finding asserted
  directly against real data (`fever_resolution_coverage()`,
  `fever_premise_resolution_status()`, and the pool itself) so a future regression toward silent
  fabrication or coercion would be caught immediately; `test_manifest_pool_counts_match_real_normalized_data`
  updated to expect FEVER's *usable* (not raw) claim count.

One test bug was found and fixed during this step's own verification (not a source-code bug): a
new NFC-fallback fixture test initially stored its synthetic accented page in NFD and queried
with NFC - the reverse of what `_resolve_page_id`'s fallback direction (normalize the *query* to
NFC) actually checks, and the reverse of what the real corpus does (wiki_pages stores NFC; FEVER
claims occasionally reference NFD). Fixed by correcting the fixture to match the real direction,
confirmed by both the source logic and the earlier real-data investigation.

**Full suite: 157/157 passed** (139 before Step 10, +18 new), run twice - once catching the test
fixture bug above (1 failure, `test_nfc_title_resolves_against_nfd_stored_page`, source code
unaffected), once clean (157/157) after the fix.

### 10. Files changed / added

- New: `scripts/acquire_inspect_wiki_pages.py`, `scripts/build_wiki_pages_index.py`,
  `scripts/resolve_fever_evidence.py`, `data/raw/fever/wiki_pages/` (raw corpus, untouched after
  acquisition), `data/processed/fever/wiki_pages_index.sqlite` (~6.4GB),
  `data/processed/fever/wiki_resolution_report.json` (18.5MB audit).
- Modified: `src/claimguard/datasets/fever.py` (new resolution functions, existing functions
  unchanged), `src/claimguard/datasets/manifest.py` (FEVER pool construction now resolution-aware,
  new `fever_resolution_coverage()`, `fever_premise_resolution_status()` rewritten,
  `dataset_text_sets("fever_train")` now includes real premise text),
  `scripts/build_dataset_manifest.py` (v1.2, FEVER section + new `neutral_class_coverage` block),
  `scripts/build_verifier_pool.py` (updated note, no logic change),
  `data/processed/dataset_manifest.json` (v1.1 → v1.2), `data/processed/verifier_pool/{train,dev}.jsonl`
  (rebuilt), `tests/test_fever_dataset.py`, `tests/test_manifest.py`, `RESEARCH.md`.
- One-off diagnostics (not part of the permanent pipeline, used only for the investigations
  above): `scripts/investigate_missing_sentence_ids.py`, `scripts/check_unicode_normalization.py`.

### 11. Explicitly not done in this step

No verifier retraining (deferred to Step 11 - the pool is ready but training is out of scope
here). No retrieval, FAISS, BM25, dense embeddings, or reranking. No RAGTruth or TruthfulQA
evaluation runs. No hyperparameter tuning. No multi-model experiments. No pipeline/end-to-end
work.

**Git status:** still no commits — all new/changed files (`scripts/acquire_inspect_wiki_pages.py`,
`scripts/build_wiki_pages_index.py`, `scripts/resolve_fever_evidence.py`,
`data/raw/fever/wiki_pages/`, `data/processed/fever/wiki_pages_index.sqlite`,
`data/processed/fever/wiki_resolution_report.json`, updated `src/claimguard/datasets/fever.py`,
updated `src/claimguard/datasets/manifest.py`, updated `scripts/build_dataset_manifest.py`,
updated `scripts/build_verifier_pool.py`, updated `data/processed/dataset_manifest.json`,
updated `data/processed/verifier_pool/{train,dev}.jsonl`, updated `tests/test_fever_dataset.py`,
updated `tests/test_manifest.py`, updated `RESEARCH.md`) show as untracked/modified.

---

## Step 11 — Neutral-class strategy and final three-way verifier dataset validation (2026-09-03)

**What:** Read the finalized protocol (`RESEARCH.md`, this report, `dataset_manifest.json` v1.2,
the verifier pool files, all dataset modules `fever.py`/`halueval.py`/`ragtruth.py`/
`truthfulqa.py`, all dataset/integration tests, the Step 9 verifier implementation) before doing
anything else. Defined `neutral` precisely per FEVER's own semantics. Investigated FEVER NOT
ENOUGH INFO exhaustively (every raw row across all three splits, not a sample) for latent
evidence, cross-checked against fever.ai's authoritative documentation. Investigated HaluEval for
a hidden third label category. Considered and ruled out RAGTruth/TruthfulQA on protocol grounds.
Added `claimguard.datasets.manifest.verifier_dataset_label_space_status()` and
`NEUTRAL_INVESTIGATION_SUMMARY`, updated the manifest builder (v1.3), created
`scripts/validate_verifier_dataset.py`, added 9 new tests, and ran the full suite. **Did NOT**
retrain the verifier, tune hyperparameters, add class weights, oversample/downsample, generate
synthetic neutral examples, evaluate on RAGTruth/TruthfulQA, or touch retrieval/FAISS/reranking.

**Why:** Step 10 achieved 100% FEVER SUPPORTS/REFUTES evidence resolution but left the verifier
pool's `neutral` class at 0 - the central unresolved scientific issue heading into this step. The
objective was to determine, rigorously and from real data, whether a legitimate neutral source
exists anywhere in the project's already-acquired datasets - and, if not, to say so plainly and
preserve the binary baseline rather than fabricate a third class to make the verifier look
"3-way" when it structurally is not.

### 1. Defining neutral

Per FEVER's own semantics (unchanged, just made explicit here): a neutral example is one where
the available premise/evidence does not support the claim, but also does not establish it as
false. This project has never equated "hallucinated" (HaluEval's label) with "neutral" - HaluEval
hallucinations are, by construction, either fabricated/incorrect content (closer to
"contradiction") or unsupported-but-not-necessarily-false content; the project's existing
`HALUEVAL_TO_VERIFIER` mapping (Step 7) already only maps `hallucinated → contradiction`, never
`→ neutral`, and this step re-confirms that mapping was not reinterpreted to manufacture a third
class.

### 2. FEVER NOT ENOUGH INFO investigated deeper (`scripts/investigate_nei.py`)

Ran a full, non-sampled inspection of every raw row (not just NEI rows - every row, to see the
complete field set) across **all three** FEVER splits (train/validation/test), not just train:

- **Train (the only training-eligible split): all 47,609 raw NOT ENOUGH INFO rows (35,639 unique
  claims) carry the pure sentinel** (`evidence_wiki_url=""`, `evidence_id=-1`,
  `evidence_sentence_id=-1`) **with zero exceptions.** No latent `wiki_url`, no hidden candidate
  page, no additional field beyond the seven already documented
  (`claim`/`evidence_annotation_id`/`evidence_id`/`evidence_sentence_id`/`evidence_wiki_url`/
  `id`/`label`).
- **Validation (already excluded from training since Step 6A/7 for independent data-quality
  reasons - blank labels, ~24% duplicate rows, casing variants, 9,999-claim overlap with test):**
  151/13,883 NOT ENOUGH INFO rows DO have a non-sentinel `evidence_wiki_url`, and a further 22/34
  rows have non-sentinel `evidence_sentence_id`/`evidence_id`. Investigated rather than used
  uncritically: the wiki_url `'Starrcade'` appears attached to three verbatim-unrelated claims
  (about Fabian Nicieza/X-Men, Sausage Party, and Aristotle) with `evidence_annotation_id=-1` -
  this is not plausible as genuine annotator-recorded evidence (an annotator would not attach the
  same single page as "evidence" to three semantically unrelated claims), and is consistent with
  a parquet-export/indexing artifact of the kind already documented for validation in Step 6A.
  **Not used** - both because it does not appear to be legitimate evidence, and because
  validation remains out of the training pool for reasons independent of this step.
- **Test: all 4,417 NOT ENOUGH INFO rows also carry the pure sentinel with zero exceptions**
  (test is already excluded from training regardless, per Step 6A, and was checked here only for
  completeness of the investigation).

### 3. Authoritative FEVER documentation consulted

Fetched `fever.ai`'s own dataset documentation to verify the intended NOT ENOUGH INFO semantics
rather than relying on this project's prior empirical inference alone. Confirmed:
- Official downloads are limited to the claims files and the pre-processed Wikipedia pages
  corpus (already acquired in Step 10) - **no officially-distributed predicted-evidence or
  retrieval-baseline output file exists.**
- NOT ENOUGH INFO claims are officially specified as receiving an
  `[Annotation ID, Evidence ID, null, null]` evidence tuple - i.e. **null Wikipedia URL and null
  sentence ID by design**, confirming the empirical finding above is the intended data model, not
  an acquisition gap.

**Conclusion: the authoritative FEVER release provides no legitimate premise for NOT ENOUGH INFO
claims, in this project's data or in principle.** No reconstruction was attempted, because there
is nothing legitimate to reconstruct from.

### 4. HaluEval investigated for a hidden third category

Re-read `claimguard.datasets.halueval`'s documented real-data schema (Step 6B) rather than
guessing: `VALID_LABELS = {"hallucinated", "not_hallucinated"}` is exhaustive across all four
subsets (qa, dialogue, summarization via pair-expansion; general via its native `hallucination`
yes/no field). **No subset encodes an "insufficient evidence"/"unknown" category anywhere in the
real data.** HaluEval's binary labels were not reinterpreted to manufacture a neutral class.

### 5. RAGTruth and TruthfulQA — ruled out on protocol grounds, not content grounds

Both are hard-blocked, evaluation-only datasets under the Step 3/4 research protocol
(`RoleViolationError` on any training-pool inclusion attempt, re-verified this step - see
Section 7 below). RAGTruth in particular DOES have span-level hallucination-vs-clean response
annotations (`label_type`, `has_hallucination`) that might superficially resemble a
neutral-adjacent signal - but using RAGTruth for training in ANY capacity would violate the
evaluation boundary this project's entire leakage-safety design depends on (it is the project's
PRIMARY end-to-end evaluation dataset). It was therefore not considered as a training-neutral
candidate regardless of its label content, and the boundary was not weakened to investigate
further. TruthfulQA has no grounding/premise field suitable for NLI at all (Step 6D) and is
likewise evaluation-only.

### 6. No new dataset proposed or acquired

Per the explicit instruction to check the existing research plan and preserve evaluation
boundaries before considering a new benchmark: the investigation above (Sections 2-5) found no
legitimate neutral source in any already-acquired dataset. Per instructions, this step **stopped
here** rather than automatically reaching for a new benchmark. If a 3-way verifier is required in
the future, acquiring a dedicated NLI/neutral-labeled dataset (e.g. an existing 3-way NLI corpus
disjoint from RAGTruth/TruthfulQA) would need to become its own explicitly-scoped
dataset-acquisition step - with its own license/provenance/contamination review - not a decision
made or acted on here.

### 7. Synthetic negatives — not used

No neutral examples were constructed via random Wikipedia pairing, random negative sampling,
evidence swapping/deletion, synthetic text generation, or LLM-invented labels. These remain
explicitly out of scope for the *primary* training class; the project may consider them later as
a clearly-labeled, controlled experiment, but they must never silently become the default
'neutral' data source.

### 8. Quantified consequence

| | Count |
|---|---|
| Total verifier pool records | 169,610 |
| entailment | 109,935 |
| contradiction | 59,675 |
| **neutral** | **0** |
| FEVER train NOT ENOUGH INFO (potentially recoverable, investigated) | 35,639 - **not recoverable, see Section 2-3** |
| FEVER validation NEI rows with anomalous evidence (investigated) | 151 - **not legitimate, see Section 2** |
| HaluEval candidate neutral source | **none found, see Section 4** |
| Other already-acquired legitimate source | **none found (RAGTruth/TruthfulQA protocol-excluded), see Section 5** |

**"The current authoritative datasets do not provide a leakage-safe, premise-grounded neutral
training class."** Stated exactly per instructions, not hidden or softened.

### 9. Final label protocol

No legitimate neutral source was found. Per instructions: **the existing binary verifier training
protocol is retained**, explicitly named a **binary verifier baseline** - it is not, and must not
be reported as, a 3-way verifier. `INTENDED_VERIFIER_LABEL_SPACE` in
`claimguard.datasets.manifest` still names all three labels (the eventual design target is
unchanged), but `verifier_dataset_label_space_status()` computes the pool's real, current
`classification_mode` from actual label counts every time it is called:

```python
CLASSIFICATION_MODE_THREE_WAY = "three_way"   # all three labels populated
CLASSIFICATION_MODE_BINARY = "binary"          # exactly entailment+contradiction populated
CLASSIFICATION_MODE_INVALID = "invalid"        # anything else (e.g. an out-of-scheme label)
```

Real result for the current pool: `classification_mode == "binary"`, `supports_three_way ==
False`.

### 10. Manifest updated (`data/processed/dataset_manifest.json`, v1.2 → **v1.3**)

Added a new top-level `verifier_dataset_status` block (via
`scripts/build_dataset_manifest.py`), containing: `intended_label_space` (all three labels),
`available_label_space` (currently `["contradiction", "entailment"]`), `class_counts`,
`missing_classes` (`["neutral"]`), `classification_mode` (`"binary"`), `supports_three_way`
(`False`), a `rationale` string, the full `neutral_class_investigation` summary (Sections 2-5
above, machine-readable), and a pointer to this report section. The manifest does **not** claim
3-way training support - `classification_mode` is computed from the real pool, not asserted.

### 11. `scripts/validate_verifier_dataset.py` (new)

A read-only, re-runnable validation gate that rebuilds the pool exactly as
`scripts/build_verifier_pool.py` does and checks 9 invariants: schema/label validity,
provenance (allowed `(source_dataset, source_subset)` pairs only), train/dev group-key
disjointness, HaluEval pair-grouping integrity, contamination exclusion, the RAGTruth boundary,
the TruthfulQA boundary, class coverage **cross-checked against the manifest's own declared
`classification_mode`** (hard failure on any mismatch, and hard failure if the manifest ever
claims `"three_way"` while the real neutral count is 0 - the specific anti-regression guard
required by this step), and deterministic-rebuild consistency. Real run against the current pool:
**all 9 checks PASS**, correctly reporting `BINARY` mode.

### 12. Tests

9 new tests in `tests/test_manifest.py` (`TestVerifierDatasetLabelSpaceStep11`):
`verifier_dataset_label_space_status()` correctness on synthetic fixtures (binary-mode pool,
three-way-mode pool, neutral-only pool → `invalid` rather than misread as binary, an
out-of-scheme label → `invalid` - simulating a prohibited relabeling and confirming it is
caught, empty pool → `invalid`), plus real-data confirmation tests (guarded by `SkipTest`): zero
`PoolRecord`s carry `label == "neutral"` in the real pool (direct anti-fabrication check), FEVER
`NOT ENOUGH INFO` never appears as an `original_label` among real pool records, and the
manifest's declared `classification_mode` matches a freshly-recomputed value from the real pool.

**Full suite: 166/166 passed** (157 before Step 11, +9 new).

### 13. Files changed / added

- New: `scripts/investigate_nei.py` (one-off diagnostic, not part of the permanent pipeline - used
  only for Section 2's investigation), `scripts/validate_verifier_dataset.py`.
- Modified: `src/claimguard/datasets/manifest.py` (new
  `verifier_dataset_label_space_status()`, `NEUTRAL_INVESTIGATION_SUMMARY`,
  `INTENDED_VERIFIER_LABEL_SPACE`, `CLASSIFICATION_MODE_*` constants),
  `scripts/build_dataset_manifest.py` (v1.3, new `verifier_dataset_status` section),
  `data/processed/dataset_manifest.json` (v1.2 → v1.3), `tests/test_manifest.py`, `RESEARCH.md`.
  `data/processed/verifier_pool/{train,dev}.jsonl` unchanged (no label-mapping logic changed -
  the pool's actual records are identical to Step 10's; only the reporting/validation layer
  changed).

### 14. Implications for Step 12 (next training step)

The verifier must be trained and evaluated as an explicit **binary** (entailment/contradiction)
classifier for now - any Step 12 config, report, or code must not claim or imply 3-way
(entailment/neutral/contradiction) capability. `scripts/validate_verifier_dataset.py` should be
re-run as a pre-training gate; it will fail loudly if a future manifest ever claims 3-way support
without genuine neutral examples. A future 3-way experiment would require a dedicated,
explicitly-scoped dataset-acquisition step for a genuine neutral-labeled NLI source (see Section
6) - not attempted or decided here.

**Git status:** still no commits — new files (`scripts/investigate_nei.py`,
`scripts/validate_verifier_dataset.py`), modified files
(`src/claimguard/datasets/manifest.py`, `scripts/build_dataset_manifest.py`,
`data/processed/dataset_manifest.json`, `tests/test_manifest.py`, `RESEARCH.md`) show as
untracked/modified.

---

## Step 12 — Final binary ClaimGuard verifier training (2026-09-03)

**What:** Read the current protocol state (`RESEARCH.md`, this report, `dataset_manifest.json`
v1.3, the verifier pool files, `scripts/validate_verifier_dataset.py`, `fever.py`/`halueval.py`/
`manifest.py`, the Step 9 verifier implementation, `configs/verifier_baseline.yaml`,
`tests/test_verifier.py`, `tests/test_manifest.py`). Verified the current pool from disk (not
assumed). Built a genuine 2-class classification head for the project's Step 3/5B-selected
verifier model (`src/claimguard/verifier/binary_model.py`), a new training entry point
(`src/claimguard/verifier/train_binary.py`) with pre-training safety assertions, a new config
(`configs/verifier_binary_final.yaml`), and 15 new tests. Ran a real-GPU smoke test, then exactly
ONE full training run, evaluated on the complete dev set (overall and per-source-group), compared
carefully against Step 9, saved a new checkpoint directory, and ran the full test suite before and
after training. Did **not** evaluate on RAGTruth or TruthfulQA, did not tune hyperparameters, did
not add class weights/oversampling/downsampling, and did not fabricate or reinterpret any label as
`neutral`.

**Why:** Step 11 finalized the scientific protocol: the verifier is explicitly BINARY
(entailment/contradiction) for the current data - no legitimate neutral source exists. Step 9's
preliminary baseline (a) used HaluEval only (FEVER's premise text wasn't resolved yet) and (b)
fine-tuned the pretrained model's *original 3-class head* while only ever supplying 2 of its 3
labels, leaving `neutral` unsupervised while still shipping a nominally 3-class architecture. Step
12's purpose was to produce the first FINAL, honestly-architected binary baseline: FEVER (with
Step 10's resolved premise text) + HaluEval, and a classification head that actually has 2 outputs,
not 3.

### 1. Confirmed current state before touching anything

`data/processed/dataset_manifest.json` confirmed at **v1.3**,
`verifier_dataset_status.classification_mode == "binary"`. `scripts/validate_verifier_dataset.py`
re-run fresh: **all 9 checks PASS** (schema/labels, provenance, train/dev disjointness, HaluEval
pair grouping, contamination exclusion, RAGTruth boundary, TruthfulQA boundary, class-coverage/
manifest consistency, deterministic counts) - reported in full below (Section 2). The Step 9
HaluEval-only dataset was **not** silently reused; the pool consumed here is the same
`data/processed/verifier_pool/{train,dev}.jsonl` files rebuilt in Step 10/11 (FEVER included).

### 2. Dataset validation (Step 11 gate, re-run fresh)

```
Pool: 169,610 records (152,720 train / 16,890 dev)
  [PASS] schema_and_labels: {'checked': 169610, 'issues': 0}
  [PASS] provenance: {'checked': 169610, 'allowed_pairs': [('fever','train'), ('halueval','dialogue'),
         ('halueval','qa'), ('halueval','summarization')]}
  [PASS] train_dev_disjointness: {'train_groups': 125739, 'dev_groups': 13971, 'overlap': 0}
  [PASS] halueval_pair_grouping: {'halueval_groups_checked': 29900, 'split_pairs': 0}
  [PASS] contamination_exclusion: {'excluded_keys': 100, 'leaked_into_pool': 0}
  [PASS] ragtruth_boundary: {'ragtruth_test_blocked': True, 'ragtruth_train_blocked': True}
  [PASS] truthfulqa_boundary: {'truthfulqa_blocked': True}
  [PASS] class_coverage_and_manifest_consistency: classification_mode='binary',
         class_counts={'entailment': 109935, 'contradiction': 59675}, missing_classes=['neutral']
  [PASS] deterministic_counts: {'pool_size': 169610, 'train': 152720, 'dev': 16890, 'deterministic': True}
ALL 9 CHECKS PASSED.
```

Nothing failed - training was not blocked, and nothing needed to be diagnosed before proceeding.

### 3. Final dataset - reported from disk, not entered as constants

`train_binary.py` loads both split files fresh via `load_baseline_examples` (Step 9's dataset
adapter - unchanged logic, now naturally includes FEVER since its premise text is resolved) and
prints/saves a `FINAL DATASET REPORT` computed from the actually-loaded examples:

| | train | dev |
|---|---|---|
| total | 152,720 | 16,890 |
| entailment | 98,934 (64.78%) | 11,001 (65.13%) |
| contradiction | 53,786 (35.22%) | 5,889 (34.87%) |
| FEVER | 98,758 (64.67%) | 11,052 (65.44%) |
| HaluEval qa | 18,062 (11.83%) | 1,938 (11.47%) |
| HaluEval dialogue | 18,080 (11.84%) | 1,920 (11.37%) |
| HaluEval summarization | 17,820 (11.67%) | 1,980 (11.72%) |

`excluded_no_premise_text: 0` for both splits (as expected - every record in the Step 10/11 pool
files that reaches `load_baseline_examples` has real, resolved premise text; the FEVER NOT ENOUGH
INFO / unresolved records were already excluded upstream at pool-construction time). Natural class
imbalance (≈65/35) reported as-is - **not rebalanced**, per instructions.

### 4. Input representation - verified, not mixed

- FEVER: `premise` = Step 10's resolved Wikipedia evidence-sentence text; `hypothesis` = the FEVER
  claim; `label` = SUPPORTS→entailment / REFUTES→contradiction (unchanged Step 7 mapping).
- HaluEval: `premise` = the subset's grounding/context field(s) (`knowledge`/`dialogue_history`/
  `document`, joined - unchanged Step 7 mapping); `hypothesis` = the candidate response
  (reference or hallucinated); `label` = not_hallucinated→entailment / hallucinated→contradiction.
- Both feed the same `(premise, hypothesis)` pair format into the tokenizer
  (`TokenizedVerifierDataset`, unchanged from Step 9) - fields are never cross-assigned between
  sources. `max_length` unchanged at 256. Provenance (`source_dataset`, `source_subset`,
  `source_id`, `original_label`) is retained on every `VerifierExample` outside the tokenized
  tensors, exactly as in Step 9.

### 5. Model and the binary head - VERY IMPORTANT, done as specified

Same base model as Step 3/5B/9: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
(`DebertaV2ForSequenceClassification`). Empirically re-verified architecture (not assumed):
top-level submodules `deberta` (encoder), `pooler` (`ContextPooler`: one `Linear(1024,1024)`,
shared across labels), `classifier` (one `Linear(1024, num_labels)` - the only label-count-
dependent submodule). Original `id2label = {0: entailment, 1: neutral, 2: contradiction}`
(confirmed identical to Step 5B/9), 435,064,835 total parameters (classifier itself: 3,075).

**The pretrained 3-class head was NOT reused as-is** (that was explicitly the wrong approach,
per instructions - it would leave `neutral`'s output dimension unsupervised while still
misleadingly reporting a 3-class architecture). Instead, `claimguard.verifier.binary_model.
build_binary_model()`:

1. Loads the ORIGINAL 3-class model once, reads its classifier's entailment (index 0) and
   contradiction (index 2) rows - **indices verified from the model's own config**, never
   assumed to be `[0, 2]` (a differently-ordered `id2label` would have produced different
   indices, and the code would have used those instead).
2. Constructs a NEW model via `AutoModelForSequenceClassification.from_pretrained(...,
   num_labels=2, id2label={0:'entailment',1:'contradiction'}, label2id={'entailment':0,
   'contradiction':1}, ignore_mismatched_sizes=True)` - this loads the pretrained encoder+pooler
   unchanged (their shape doesn't depend on `num_labels`) and randomly reinitializes only the
   now-mismatched-shape classifier.
3. **Immediately overwrites** that random classifier with the two captured rows from step 1 -
   the binary head's real starting point is the pretrained model's own entailment/contradiction
   judgment, not a random head and not a hidden interim state.
4. **Verifies (does not assume)** that every non-classifier parameter in the new model is
   bit-identical to the original pretrained model's (`torch.equal` on all 392 real-model tensors)
   - raises loudly if this check ever fails, so "encoder/pooler reused" is a checked fact in the
   saved report, not a claim.

**Real result (empirically confirmed):** `num_labels=2`, `id2label={0:'entailment',
1:'contradiction'}`, `label2id={'entailment':0,'contradiction':1}`, **435,063,810 total
parameters** - exactly 1,025 fewer than the 3-class model (the removed third classifier row
[1,024 weights] + its bias [1]), confirming only the classifier's shape changed and nothing else
was silently altered. `assert_binary_head()` is called immediately after construction (and again
when the saved checkpoint is reloaded, Section 9) and raises `AssertionError` if `num_labels != 2`
or `id2label` doesn't exactly match the binary scheme - a hard guard against ever accidentally
training or shipping a 3-class configuration under the "binary verifier" name. A dedicated test
(`test_rejects_three_class_head`) confirms this guard actually fires.

### 6. Training safety assertions (before any GPU work)

`train_binary.run()` (non-smoke-test path) calls `run_preflight_dataset_validation()`, which runs
`scripts/validate_verifier_dataset.py` as a subprocess and raises `TrainingBlockedError` (stopping
before any model/GPU work) if it exits nonzero. It also independently re-checks the manifest's
`classification_mode == "binary"` and calls `assert_binary_labels_only()` on every loaded example
(train and dev) - raising if any label besides `entailment`/`contradiction` is found (e.g. a
future accidental `neutral` reintroduction). Real run: **all assertions passed silently** - no
diagnosis was needed.

### 7. Smoke test - PASS

Real GPU, final binary architecture, truncated to 32 train / 16 dev examples, 5 steps: model
construction and tokenizer load succeeded; forward pass, loss (1.73 → 1.103 → 0.8608 → 0.8301 →
0.7683, monotonically decreasing), backward pass, and optimizer step all confirmed; evaluation ran
at steps 2 and 4 (`eval_accuracy=0.75`, `eval_macro_f1=0.7333` on the 16-example slice);
checkpoints saved at steps 2 and 4 and loaded back via `load_best_model_at_end`; **output logits
shape confirmed [batch_size, 2]** (2-class, not 3); no NaN/Inf anywhere in the loss/metric values;
peak GPU memory reasonable; GPU returned to fully free (2MiB) after the process exited - confirmed
via `nvidia-smi`, not assumed.

### 8. Full baseline training run - the ONE run

1 epoch, 152,720 train / 16,890 dev, effective batch 32, **4,773 total steps** (`warmup_steps=286`),
**1,964.3s (~32.7 min) wall-clock** (`train_samples_per_second=77.75`), peak GPU memory
**14.36 GB**. Training loss decreased from ~0.30 early to a final `train_loss=0.2815`. Eval loss
decreased cleanly and then plateaued with minor noise (0.1978 → 0.1569 → 0.1308 → 0.1259 → 0.1220
→ 0.1227 → 0.1239 → 0.1204 → 0.1218 → 0.1216 across the 10 evaluations) - healthy convergence, no
divergence, no NaN/Inf, no OOM. Best-by-`macro_f1` checkpoint (`checkpoint-4500`, macro_f1=0.9585)
essentially tied with the final step (`checkpoint-4773`, macro_f1=0.9584) - both retained
(`save_total_limit=2`), `final/` holds the best-model copy per `load_best_model_at_end=True`.

### 9. Dev evaluation (16,890 examples - complete dev set)

| Metric | Value |
|---|---|
| Accuracy | 0.9620 |
| Macro F1 | **0.9585** *(a genuine, meaningful number now - not neutral-distorted)* |
| Weighted F1 | 0.9621 |

**Per-class:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| entailment | 0.9804 | 0.9608 | 0.9705 | 11,001 |
| contradiction | 0.9294 | 0.9642 | 0.9465 | 5,889 |

**Confusion matrix** (rows=true, cols=predicted; order entailment/contradiction):
`[[10570, 431], [211, 5678]]`.

**Per-source-group breakdown (`dev_metrics_by_source.json`) - aggregate hides real variation:**

| Group | Accuracy | Macro F1 | N |
|---|---|---|---|
| FEVER | 0.9766 | **0.9704** | 11,052 |
| HaluEval (overall) | 0.9344 | **0.9344** | 5,838 |
| HaluEval qa | 0.9773 | 0.9773 | 1,938 |
| HaluEval summarization | 0.9551 | 0.9550 | 1,980 |
| HaluEval dialogue | 0.8698 | **0.8695** | 1,920 |

FEVER's Wikipedia-sourced claims are the easiest for the verifier; HaluEval `dialogue` is
noticeably the hardest subset (0.8695 macro F1, ~10 points below the aggregate) - the aggregate
0.9585 alone would have hidden this. This is reported explicitly rather than only as one number.
`dev_metrics_overall_recomputed.json` independently re-derives the aggregate from raw
predictions/labels via `compute_metrics()` as a cross-check against `trainer.evaluate()`'s
number - both agree (0.9585 macro F1), confirming no accounting error in the per-source split.

### 10. Comparison against Step 9 - explicitly NOT claimed as apples-to-apples

Per instructions, checked whether the evaluation sets are actually identical before comparing.
**They are not, and this was verified, not assumed:**

- Step 9's saved `dataset_stats.json` recorded exactly 5,984 usable (HaluEval-only)
  dev examples.
- The CURRENT `dev.jsonl`'s HaluEval-only record count is **5,838** (1,938 + 1,980 + 1,920 from
  Section 9's table above).
- These differ because Step 10 changed FEVER's buildable-pool composition (NOT ENOUGH INFO claims
  dropped, per Step 11), which changes the full `sorted(group_keys)` list fed into
  `split_train_dev`'s seeded shuffle - even with the same seed (42), a differently-sized/composed
  input list produces a different permutation, so HaluEval group-key assignments to train/dev are
  **not** guaranteed (and empirically are not) identical between the Step 9-era pool build and the
  current one. The original Step 9 `dev.jsonl` file was also overwritten in place by Step 10's
  pool rebuild, so an exact historical replay isn't recoverable without reverting Step 10/11's
  protocol changes (not done).

Given that, the comparison below is reported as **two different, non-identical evaluation
conditions**, not a controlled ablation:

| | Step 9 (HaluEval-only, 3-class head/2 labels used) | Step 12 (FEVER+HaluEval, genuine binary head) | Step 12, HaluEval-only dev slice (same-domain, NOT same examples) |
|---|---|---|---|
| Dev N | 5,984 | 16,890 | 5,838 |
| Accuracy | 0.9158 | 0.9620 | 0.9344 |
| Weighted F1 | 0.9160 | 0.9621 | 0.9344 |
| Entailment F1 | 0.9146 | 0.9705 | 0.9328 |
| Contradiction F1 | 0.9174 | 0.9465 | 0.9359 |

The Step 12 HaluEval-only slice (closest legitimate same-domain comparison) shows higher accuracy
than Step 9 (93.44% vs 91.58%) - a plausible improvement consistent with the genuine binary head
and/or beneficial transfer from FEVER co-training, but **not claimed as a rigorously isolated
ablation result**, since the underlying dev examples differ and no controlled single-variable
experiment was run to attribute the gain to either factor specifically.

### 11. Checkpoint artifacts

`experiments/verifier_binary_final/` (new directory - **Step 9's `experiments/verifier_baseline/`
was not touched or overwritten**): `checkpoint-4500` (best-by-macro_f1), `checkpoint-4773` (final
step), and `final/` (best-model copy) containing: `model.safetensors`, tokenizer files,
`config.json`, `training_config.yaml` (the resolved config actually used), `dataset_manifest_
snapshot.json` (a copy of `data/processed/dataset_manifest.json` v1.3 at training time),
`dev_metrics.json`, `dev_metrics_overall_recomputed.json`, `dev_metrics_by_source.json`,
`dataset_stats.json` (pool-load stats + the final dataset report), `binary_head_report.json`,
`reproducibility.json`, and `git_status_snapshot.txt`.

### 12. Reproducibility (`reproducibility.json`)

Python 3.12.14, PyTorch 2.11.0+cu128, Transformers 5.16.1, CUDA 12.8, GPU: NVIDIA RTX PRO 5000
Blackwell. Seed 42 (all four RNG sources, `set_all_seeds`). Manifest v1.3 (expected == actual).
Train/dev record counts, exact label2id/id2label, the full binary-head-construction report,
max_length=256, optimizer AdamW (Trainer default), lr 2e-5, batch 16 x grad-accum 2 (effective
32), bf16, 1 epoch, linear scheduler, warmup_ratio 0.06 (286 steps of 4,773), max_grad_norm 1.0,
weight_decay 0.01, checkpoint selection criterion `macro_f1 (higher is better),
load_best_model_at_end=True`, peak GPU memory 14.36 GB - all saved as one file, not scattered
across the report.

### 13. Checkpoint validation (post-training, real check)

Loaded `experiments/verifier_binary_final/final/` fresh via `AutoModelForSequenceClassification.
from_pretrained` + `AutoTokenizer.from_pretrained`, ran `assert_binary_head()` (**PASSED**), and a
real forward pass on a sample `(premise, hypothesis)` pair: `num_labels=2`,
`id2label={0:'entailment',1:'contradiction'}`, **logits shape `(1, 2)`**, all logits finite. The
saved checkpoint is independently confirmed to be exactly what it claims to be, not just assumed
from the training log.

### 14. Tests

15 new tests, `tests/test_verifier_binary.py` (uses a real, tiny, LOCALLY-CONSTRUCTED
`DebertaV2ForSequenceClassification` fixture - not the 440M-param model - so the suite stays fast
and offline while still exercising the real architecture family):
`TestVerifiedSourceLabelIndices` (correct index extraction; missing-label rejection),
`TestAssertBinaryHead` (accepts a correct binary head; rejects a 3-class head; rejects a binary
head with the wrong label mapping), `TestBuildBinaryModel` (correct architecture and **exact
weight-transplant correctness** - the new classifier's two rows are asserted `torch.equal` to the
original model's captured entailment/contradiction rows, not just "close"; encoder/pooler
bit-identity independently verified; rejects a non-3-class source model), `TestAssertBinaryLabelsOnly`
(accepts binary-only examples; rejects `neutral`; rejects any other unexpected label; empty list is
fine), `TestComputeMetricsByGroup` (correct per-group splitting; empty groups omitted, not
fabricated with zero-support metrics), and a config-loader test for the new
`configs/verifier_binary_final.yaml`.

**Full test suite: 181/181 passed**, run twice - once before training (confirming the
architecture/safety logic was correct before spending GPU time) and once after training/checkpoint
creation (confirming nothing broke and no test was weakened to accommodate a failure).

### 15. Limitations (stated explicitly)

- **This is a binary supported-vs-contradicted/hallucinated classifier for the current training
  protocol. It is not a three-way entailment/contradiction/neutral classifier.** `neutral` remains
  entirely absent from training and evaluation, per Step 11's finding - no legitimate source
  exists yet.
- HaluEval's binary hallucinated/not_hallucinated mapping remains an approximation across
  heterogeneous sources (documented since Step 7/8) - conflating what FEVER separately calls
  REFUTES and NOT ENOUGH INFO into one `contradiction` class for HaluEval's `hallucinated` label.
- The Step 9 comparison is explicitly not a controlled ablation (Section 10) - do not read the
  HaluEval-only-slice improvement as an isolated, attributable effect of any single change.
- HaluEval `dialogue` is a real, identified weak point (macro F1 0.8695) - not investigated further
  in this step (no hyperparameter tuning, per instructions).
- Single run, no hyperparameter tuning, 1 epoch only - explicitly a baseline, not an optimized
  model.
- No evaluation against RAGTruth or TruthfulQA was performed - both remain untouched and reserved,
  per the stopping boundary.

**Preserved, unchanged:** RAGTruth test remains the primary end-to-end evaluation dataset,
untouched. TruthfulQA remains evaluation-only, untouched. No neutral class exists anywhere in this
step's training or evaluation data. No hyperparameter tuning was performed.

**Git status:** still no commits — new files (`src/claimguard/verifier/binary_model.py`,
`src/claimguard/verifier/train_binary.py`, `configs/verifier_binary_final.yaml`,
`tests/test_verifier_binary.py`, `experiments/verifier_binary_final/` and
`experiments/verifier_binary_final_smoke_test/`), modified files (`src/claimguard/config.py`,
`src/claimguard/verifier/dataset.py`, `src/claimguard/verifier/evaluate.py`,
`src/claimguard/verifier/__init__.py`, `RESEARCH.md`) show as untracked/modified. Step 9's
`experiments/verifier_baseline/` and `configs/verifier_baseline.yaml` are unmodified.

---

## Step 13 — ClaimGuard retrieval corpus, BGE embeddings, and FAISS index infrastructure (2026-09-03)

**What:** Read `RESEARCH.md`, this report, `dataset_manifest.json`, `contamination_report.json`,
all dataset normalization modules, the verifier implementation/config, Step 12's experiment
metadata, and existing tests. Inspected the Step 5C BGE embedding smoke test and
`configs/models.yaml`. Defined the retrieval objective and corpus scope from the existing
protocol (not guessed), inspected the existing `wiki_pages_index.sqlite` rather than rebuilding
it, built a new `src/claimguard/retrieval/` package (corpus construction, BGE embedding, FAISS
indexing, query-time retrieval API, retrieval evaluation), ran a real GPU embedding smoke test,
embedded the full scoped corpus, built and validated a FAISS index, ran a FEVER-only retrieval
evaluation, added 32 new tests, and ran the full test suite twice. Did **not** integrate
reranking, did **not** touch the verifier, and did **not** evaluate RAGTruth or TruthfulQA.

**Why:** With the binary verifier baseline established (Step 12), the next component needed for
the full ClaimGuard pipeline (RESEARCH.md's methodology step 3: "Retrieve candidate evidence per
claim via a BM25+FAISS hybrid retriever") is the retrieval infrastructure itself - a corpus,
embeddings, and an index - built and validated in isolation before any reranking or verifier
integration, so each component's correctness can be established independently.

### 1. Retrieval objective - defined from the existing protocol

RESEARCH.md's "Proposed Methodology" step 3 specifies FAISS-based retrieval against a
Wikipedia-derived knowledge source. The Dataset Strategy table lists FEVER for
verifier training/validation, and FEVER's own `wiki_pages` corpus (acquired and indexed in Step
10) is the only Wikipedia knowledge source this project has acquired - no other candidate exists
in the current protocol. RAGTruth was explicitly ruled out (its documents are evaluation-reserved
and are per-example source material, not a general knowledge base); TruthfulQA was ruled out (no
grounding corpus of any kind, evaluation-only, Step 6D).

### 2. Corpus scope - investigated and quantified before deciding, not guessed

Before writing any embedding code, queried the real data to inform the scoping decision
(`scripts/inspect_corpus_scope.py`, a one-off diagnostic, not part of the permanent pipeline):

- `data/processed/fever/wiki_pages_index.sqlite` (already built, Step 10; **not rebuilt**):
  5,416,536 pages / 42,041,084 sentences total.
- FEVER train SUPPORTS/REFUTES claims (the same claims the verifier was trained on, Steps
  10-12) reference **12,549 unique Wikipedia pages** as evidence, across every annotated
  evidence set (not just the one Step 10's resolver ultimately used).
- Indexing every sentence of those pages (not only the single cited evidence sentence, so
  retrieval has to distinguish the right sentence from real neighbors) was estimated at ~158,658
  sentences by a raw count; after filtering empty/boilerplate lines during actual construction,
  the real corpus is **109,350 sentences**.

Embedding the full 42M-sentence corpus was explicitly considered and rejected for this
step: at the measured throughput (Section 5 below, ~1,670 records/sec), it would take
roughly 7 hours of GPU time on a shared machine - disproportionate for an
infrastructure-building step. The FEVER-evidence-page-scoped corpus is a principled, bounded,
already-validated subset directly tied to this project's existing experimental data; expanding
to the full corpus (or a larger principled subset) is a natural, separate future scaling step -
not attempted here, and this limitation is stated explicitly, not hidden.

### 3. Retrieval unit and provenance design

Chosen unit: the individual native FEVER/`wiki_pages` sentence - not a chunk. No chunking policy
was needed: FEVER sentences are already well within BGE's 512-token `max_seq_length`, and using
the native unit preserves exact FEVER sentence identity/provenance without introducing any new
boundary-decision logic. Every corpus record carries:

```
corpus_id = f"{page_id}::{sentence_id}"   # stable, independent of FAISS row position
page_id, sentence_id, text, corpus_version
```

`corpus_id` is the identifier used everywhere (pool records, FAISS metadata, retrieval results) -
FAISS row position is used ONLY internally to build the `row_metadata` list at save time, and is
never returned to a caller in place of `corpus_id`.

### 4. Corpus construction (`src/claimguard/retrieval/corpus.py`,
`scripts/build_retrieval_corpus.py`)

Page lookup reuses Step 10's already-validated exact-match-then-NFC-fallback resolver
(`fever_ds._resolve_page_id`) rather than a fresh exact-only lookup. Real result:

```
Requested evidence page IDs: 12,549
Resolved unique pages: 12,549   (0 missing - NFC-fallback carried over cleanly)
Empty-text sentences skipped: 50,901
Total corpus records: 109,350
Validation issues: 0
```

Zero missing pages (vs. the 141 misses my quick raw-exact-match diagnostic script found before
reusing the real resolver) directly confirms the value of reusing Step 10's proven
canonicalization rather than reimplementing page lookup from scratch. `corpus.validate_corpus()`
checks duplicate `corpus_id`, empty text, and `corpus_id`/`(page_id, sentence_id)` consistency -
0 issues on the real corpus. Written to `data/processed/retrieval/corpus.jsonl` (raw
`wiki_pages_index.sqlite` untouched - read-only).

### 5. BGE embedding model - unchanged, verified, not silently swapped

`configs/models.yaml`'s `embedding.primary` entry (unchanged since Step 4/5C):
`BAAI/bge-large-en-v1.5`, fp16, device cuda, `max_seq_length: 512`, `batch_size: 32`,
`normalize_embeddings: true`. `src/claimguard/retrieval/embed.py`'s `load_embedding_model()`
mirrors the Step 5C smoke-test loading pattern (explicit dtype verification/cast, VRAM safety
check before load) and returns a `verified_info` dict documenting every checked fact (items A-G
from the instructions): dimension **1024** (from actual model output), dtype `torch.float16`,
device `cuda:0`, batch_size 32, normalize=True, max_seq_length 512, 335,141,888 parameters -
all matching Step 5C's findings exactly, confirming no drift in the model's own characteristics.

**Embedding smoke test** (`scripts/embed_retrieval_corpus.py --smoke-test`, real GPU, 4 sample
sentences): shape `(4, 1024)` - PASS; all finite - PASS; L2 norms ≈ 1.0 (0.9998-1.0003) -
PASS; **determinism** (two independent `encode()` calls on the same input): `max_abs_diff = 0.0`
exactly, `allclose` - PASS; peak GPU memory 0.635 GB; GPU returned to 46.85 GB free after
cleanup (pre-load baseline was 46.93 GB) - confirmed via `nvidia-smi`, not assumed.

### 6. Full corpus embedding

After the smoke test passed, ran `scripts/embed_retrieval_corpus.py` (no `--smoke-test`) against
all 109,350 corpus records, in deterministic corpus order. Batch-level failures are isolated and
retried at the individual-item level (`embed_mod.embed_corpus`'s per-batch try/except with
per-item fallback) - never a silent whole-batch or whole-run skip.

```
embedded_count: 109,350 / 109,350
failed_count: 0
elapsed_seconds: 65.5
records_per_second: 1,669.5
embeddings_shape: (109350, 1024)
all_finite: True
```

Zero failures - no records needed investigation. Saved as `data/processed/retrieval/embeddings.npy`
(float32, required for FAISS) plus `embedding_corpus_ids.json` (the exact ordering, so embeddings
row *i* always maps to `corpus_ids[i]`).

### 7. FAISS index (`src/claimguard/retrieval/index.py`, `scripts/build_faiss_index.py`)

`faiss-cpu` 1.15.0 (already installed in the `claimguard` env - confirmed before writing any
index code, not assumed). Index type: **`IndexFlatIP`** (exact, flat, inner-product) - justified
because BGE embeddings are L2-normalized (inner product = cosine similarity exactly) and 109,350
vectors x 1024 dims is trivially small for an exact CPU linear scan; no IVF/PQ/HNSW/GPU index -
that would be premature optimization with no measured need at this corpus size, per instructions.

```
Dimension: 1024, vector count: 109,350
Build time: 0.25s
Index disk size: 427.1 MB    Metadata disk size: 30.7 MB
```

`save_index()` writes the FAISS binary plus a separate `index_metadata.json` containing, for
every FAISS row position, the FULL provenance record (`corpus_id`, `page_id`, `sentence_id`,
`text`, `corpus_version`) - so a future index rebuild (which could reassign row positions) can
never silently corrupt provenance; `corpus_id` is the identifier of record everywhere else.

### 8. Index validation

`validate_index()` checks vector-count-vs-metadata-count match, dimension match, duplicate
`corpus_id`, and empty text - **0 issues** on the real index. Fresh-reload check
(`load_index()` re-reading both files from disk into a new process-local object) confirmed
`index.ntotal == metadata["vector_count"]`. **Reload consistency check:** querying the SAME
vector against both the in-memory index and the freshly-reloaded index produced **bit-identical
top-5 results** (`np.array_equal` on indices, `np.allclose` on scores) - save/load is stable, not
just "loads without erroring."

### 9. Retrieval API (`Retriever` class in `index.py`)

`Retriever.from_disk()` / `.retrieve(query, top_k)` / `.get_by_corpus_id(corpus_id)`. Every
`retrieve()` result carries `corpus_id`, `text`, `score`, `page_id`, `sentence_id`,
`corpus_version` - never a bare FAISS row id. The query is encoded with the SAME model
configuration the index was built with; `_ensure_model_loaded()` cross-checks the currently
configured embedding model name against the one recorded in the index's own metadata and raises
`RuntimeError` on any mismatch, refusing to query with a mismatched embedding space.

**A real bug found and fixed here:** that cross-check initially read
`metadata["embedding_info"]["model_name"]`, but the field actually written by
`build_faiss_index.py` (matching `configs/models.yaml`'s own field name) is `"name"` - a
key-naming mismatch between the writer and the reader. This raised `KeyError` the first time
`evaluate_retrieval.py` actually called `.retrieve()` (the synthetic-fixture unit tests written
alongside `index.py` didn't exercise this code path, since they never called `retrieve()` -
only `get_by_corpus_id()`). **Fixed** by reading `"name"` (matching the real schema), and **two
regression tests added** using `unittest.mock` so this exact class of bug (a real-model-loading
code path silently untested) cannot recur without a real model being loaded:
`test_ensure_model_loaded_reads_embedding_info_with_correct_key` and
`test_ensure_model_loaded_rejects_mismatched_model`.

### 10. Retrieval evaluation (`src/claimguard/retrieval/eval.py`,
`scripts/evaluate_retrieval.py`) - FEVER only, never RAGTruth/TruthfulQA

**Relevance criterion, defined explicitly:** for a FEVER train claim whose evidence is (at least
partially) present in the corpus, a retrieval at cutoff *k* is a HIT if at least one sentence
from any one of the claim's annotated evidence sets appears in the top-*k* FAISS results for
that claim's own natural-language text used as the query - matching the standard FEVER
shared-task evidence-retrieval evaluation convention (sentence-level recall against annotated
evidence). This is not an artificial/self-matching query: the claim text is human-written and
never identical to any corpus sentence.

**Eval set:** 2,000 deterministically-sampled (seed 42) FEVER train claims whose full evidence
set is present in the corpus - claims referencing evidence outside the corpus's page scope are
excluded from the eval set (not silently scored as misses).

```
Recall@1:  0.2800
Recall@5:  0.5685
Recall@10: 0.6620
Recall@20: 0.7325
```

These are credible, un-reranked, exact-search numbers for claim-text-as-query dense retrieval
(FEVER claims are frequently paraphrased relative to their evidence sentence, so this is a real
semantic-retrieval task, not a lexical-match shortcut) - consistent with expected first-pass
dense-retrieval-before-reranking performance. Not compared against any external benchmark number
here (no reranking or hybrid BM25+FAISS combination has been added yet - that is Step 14+).

### 11. Performance measurement

```
corpus_size: 109,350            index_vector_count: 109,350
index_disk_size_mb: 427.1       metadata_disk_size_mb: 30.7
latency (200 queries measured): avg 21.7ms | p50 21.5ms | p95 22.0ms
```

A real baseline, not optimized - single-query exact search over 109K vectors is already fast
enough that no approximate-index work is currently justified.

### 12. Leakage / safety checks

- RAGTruth test: never read by any retrieval module - confirmed both by design (the retrieval
  package's corpus/eval logic only ever calls `claimguard.datasets.fever`) and by a new static
  test (`TestRAGTruthTruthfulQAExclusion`) that parses every retrieval module's AST and asserts
  no `ragtruth`/`truthfulqa` module is ever imported - not a runtime check that could be bypassed
  by a future contributor forgetting to re-run something, a structural guarantee.
- TruthfulQA: same guarantee, same test.
- No RAGTruth or TruthfulQA text reached the corpus, the embeddings, the index, or the
  evaluation set at any point in this step.

### 13. Tests

32 new tests, `tests/test_retrieval.py`: `TestRAGTruthTruthfulQAExclusion` (AST-based import
check), `TestCorpusId` (format/determinism/distinctness), `TestValidateCorpus`
(duplicate/empty-text/mismatch detection), `TestCorpusWriteLoadRoundtrip`,
`TestCorpusRealData` (guarded - real corpus construction + determinism against the actual
`wiki_pages_index.sqlite`), `TestEmbedTexts` (guarded - real BGE model: dimension, finiteness,
normalization, dtype, determinism), `TestBuildIndex`, `TestIndexSaveLoad` (including
reload-identical-retrieval), `TestValidateIndex` (count/dimension/duplicate-id detection),
`TestRetrieverAPI` (`get_by_corpus_id`, missing-id `KeyError`, top-k-overflow safety, and the two
new regression tests for the `"model_name"`/`"name"` key bug), `TestRetrievalEval` (recall
computation correctness on synthetic fixtures).

**Full test suite run three times this step:** 211 tests with 1 failure (the AST-vs-substring
test-writing mistake below), then 211/211 clean after that fix, then **213/213** clean after the
2 regression tests for Section 9's `"model_name"`/`"name"` bug were added.

**One test bug found and fixed during this step's own verification** (not a source-code bug):
the first version of `TestRAGTruthTruthfulQAExclusion` checked for the mere substring
`"ragtruth"`/`"truthfulqa"` anywhere in each module's source text - which flagged `corpus.py`'s
own docstring (which legitimately *explains why* RAGTruth/TruthfulQA are excluded). Fixed to
parse actual `import`/`from...import` statements via `ast` instead of a naive substring search -
a stricter, more correct check that still catches a real accidental import while no longer
penalizing legitimate documentation.

### 14. Files changed / added

- New package: `src/claimguard/retrieval/` (`__init__.py`, `corpus.py`, `embed.py`, `index.py`,
  `eval.py`).
- New scripts: `scripts/build_retrieval_corpus.py`, `scripts/embed_retrieval_corpus.py`,
  `scripts/build_faiss_index.py`, `scripts/evaluate_retrieval.py`.
- New test file: `tests/test_retrieval.py`.
- New data artifacts (all under `data/processed/retrieval/`, none touching raw data):
  `corpus.jsonl`, `corpus_manifest.json`, `embeddings.npy`, `embedding_corpus_ids.json`,
  `faiss_index.bin`, `index_metadata.json`, `retrieval_eval_results.json`.
- One-off diagnostic (not part of the permanent pipeline): `scripts/inspect_corpus_scope.py`.
- `data/processed/fever/wiki_pages_index.sqlite` - read-only, **not rebuilt or modified**.

### 15. Limitations (stated explicitly)

- **This is retrieval infrastructure only.** No reranking has been integrated (Step 14). The
  verifier was not retrained, tuned, or otherwise modified. RAGTruth test and TruthfulQA remain
  completely untouched.
- The corpus is scoped to FEVER-evidence-referenced pages (109,350 sentences), not the full
  `wiki_pages` corpus (42M+ sentences) - a deliberate, documented, resource-driven scope
  limitation for this step, not a claim that this is a production-scale index.
- Retrieval evaluation uses claim-text-as-query directly, with no query reformulation,
  decomposition, or hybrid BM25 combination (RESEARCH.md's methodology specifies a BM25+FAISS
  *hybrid* retriever - only the FAISS half exists after this step).
- Recall@k numbers are a first-pass, un-reranked baseline on a 2,000-claim sample - not
  exhaustive, and not directly comparable to any published FEVER retrieval benchmark number
  without matching methodology exactly (not attempted here).
- Query latency was measured for single-query exact search only; no batched-query or
  concurrent-load performance characterization was done (out of scope - "establish a baseline,"
  not optimize).

**Preserved, unchanged:** RAGTruth test remains the primary end-to-end evaluation dataset,
untouched. TruthfulQA remains evaluation-only, untouched. The Step 12 binary verifier checkpoint
was not touched. No reranking, no verifier+retrieval integration, no hyperparameter tuning.

**Git status:** still no commits — new files (`src/claimguard/retrieval/*`,
`scripts/build_retrieval_corpus.py`, `scripts/embed_retrieval_corpus.py`,
`scripts/build_faiss_index.py`, `scripts/evaluate_retrieval.py`, `tests/test_retrieval.py`,
`data/processed/retrieval/*`), modified files (`RESEARCH.md`) show as untracked/modified.

---

## Step 14 — ClaimGuard BGE reranker integration and evaluation (2026-09-03)

**What:** Read `RESEARCH.md`, this report, `corpus_manifest.json`, `index_metadata.json`,
`retrieval_eval_results.json`, the `retrieval/` and `verifier/` packages, Step 5D's reranker
smoke-test artifacts, and existing tests. Built a new `src/claimguard/reranking/` package
(model loading with runtime verification, scoring, reranking, and a two-stage retrieve+rerank
pipeline), a new config (`configs/reranker_baseline.yaml`), ran a real-GPU smoke test, evaluated
FAISS-only vs. FAISS+reranker on the identical Step 13 2,000-claim eval set, computed rank-shift
diagnostics and latency, added 19 new tests, and ran the full test suite twice. Did **not**
retrain the verifier, did **not** rebuild the embedding model or FAISS index, did **not**
evaluate RAGTruth/TruthfulQA, and did **not** call the verifier from the reranking code at all.

**Why:** With retrieval infrastructure validated in isolation (Step 13), the next component from
RESEARCH.md's methodology ("Rerank retrieved evidence with a cross-encoder") needed the same
isolated validation treatment before any pipeline integration - so retrieval-quality and
reranking-quality effects can be attributed separately, not conflated.

### 1. Reranker model - unchanged, runtime-verified

`configs/models.yaml`'s `reranker.primary` (unchanged since Step 4/5D): `BAAI/bge-reranker-large`,
fp16, device cuda, `max_seq_length: 512`, `batch_size: 16`. `src/claimguard/reranking/reranker.py`'s
`load_reranker()` mirrors the Step 5D smoke-test loading pattern (`CrossEncoder` with
`model_kwargs`, falling back to `automodel_args` on `TypeError`; explicit dtype
verification/cast; VRAM safety check before load) and returns a `verified_info` dict:

```
model_name: BAAI/bge-reranker-large | actual_dtype: torch.float16 | actual_device: cuda:0
n_params: 559,891,457 (matches Step 5D exactly) | activation_fn: Sigmoid
score_type: probability_like_sigmoid | score_direction: higher_is_more_relevant
```

**Score direction was NOT assumed from Step 5D's prior finding** - `load_reranker()` re-reads
the freshly-loaded model's own `activation_fn` attribute every time it is called, and the smoke
test additionally confirms it empirically on a real example (the direct answer to "What is the
capital of France?" ranks first by a wide margin, exactly as Step 5D found).

### 2. Reranker input - same query semantics as Step 13, provenance preserved

For each FAISS-retrieved candidate, the reranker scores `(query, candidate_text)` - the SAME
query text used for the FAISS embedding lookup (never reformulated or altered between stages).
`reranker.rerank()`'s output records preserve every field a `Retriever.retrieve()` result
carries (`corpus_id`, `text`, `page_id`, `sentence_id`, `corpus_version`), renaming the FAISS
`score` field to `faiss_score` to disambiguate from the new `reranker_score`, and adds
`original_rank` (1-based FAISS position) and `reranked_rank` (1-based position after sorting by
reranker score, descending) - never a bare (text, score) pair.

### 3. Two-stage pipeline

```
query -> BGE embedding -> FAISS top-20 -> BGE reranker -> reranked top-5
```

`configs/reranker_baseline.yaml`: `pipeline.retrieval_top_k: 20`, `pipeline.rerank_top_n: 5` -
recorded as configuration, not hard-coded, matching RESEARCH.md's own BM25+FAISS-then-rerank
methodology (no research-protocol-specified alternative values were found, so the instructed
defaults were used). The reranker's `rerank()` function only ever scores the candidate list it
is given - it has no code path to query the FAISS index or the corpus directly.

### 4. Reranker module (`src/claimguard/reranking/`)

- `load_reranker_config()` / `load_reranker(cfg)` - see Section 1.
- `score(model, query, candidate_texts, batch_size)` - one float per candidate, default
  (calibrated) activation; empty input returns `[]` without calling the model.
- `rerank(model, query, candidates, top_n, batch_size)` - the core function described in
  Section 2; empty `candidates` returns `[]`; duplicate `corpus_id`s in the input are each
  scored and ranked independently (this function ranks exactly what it's given, never
  deduplicates on the caller's behalf).
- `retrieve_and_rerank(retriever, reranker_model, query, retrieval_top_k, rerank_top_n,
  batch_size)` - the full two-stage pipeline for one query, returning BOTH the pre-rerank FAISS
  candidates and the post-rerank top-N, so a caller can always compare the two stages directly.

### 5. Smoke test - PASS (real GPU)

Real `BAAI/bge-reranker-large`, 4 hand-picked (query, candidate) pairs (same as Step 5D's toy
example, reused for direct comparability): output count 4/4, all finite, provenance
(text/page_id/sentence_id/corpus_version) verified byte-for-byte unchanged after reranking,
**fully deterministic** across two independent `rerank()` calls (`max_abs_diff = 0.0`, identical
order), correct empty-candidate handling, peak GPU memory 1.055GB (matches Step 5D's 1.05GB),
GPU returned to 46.85GB free after cleanup.

### 6. FAISS-only baseline (recomputed fresh on the same eval set)

Re-derived from scratch in `scripts/evaluate_reranker.py` (not just re-read from Step 13's
saved file) using `claimguard.retrieval.eval.build_eval_set` with the identical parameters
(seed=42, sample_size=2000, `source=fever_train`). **Cross-checked against Step 13's saved
`retrieval_eval_results.json`: the sampled example_id set is identical** - confirmed, not
assumed, before any comparison was drawn.

```
Recall@1:  0.2800   Recall@5:  0.5685   Recall@10: 0.6620   Recall@20: 0.7325
```

(Exactly reproduces Step 13's numbers, confirming deterministic reconstruction.)

### 7. FAISS + BGE reranker

Each query's FAISS top-20 candidates were reranked as a full 20-item permutation (not just the
top-5 production slice), so Recall@k could be computed honestly for every cutoff including k=20:

```
Recall@1:  0.2445   Recall@5:  0.5905   Recall@10: 0.7010   Recall@20: 0.7325
```

**Sanity invariant explicitly checked and PASSED:** Recall@20 is bit-for-bit identical between
FAISS-only and FAISS+reranker (0.7325 == 0.7325) - expected and required, since reranking only
reorders the same 20 candidates and never changes which items are present.

### 8. Absolute / relative improvement

| Recall@k | FAISS | FAISS+reranker | Absolute Δ | Relative Δ |
|---|---|---|---|---|
| 1 | 0.2800 | 0.2445 | **−0.0355** | **−12.7%** |
| 5 | 0.5685 | 0.5905 | +0.0220 | +3.9% |
| 10 | 0.6620 | 0.7010 | +0.0390 | +5.9% |
| 20 | 0.7325 | 0.7325 | 0.0000 | 0.0% |

**Reranking measurably DEGRADED Recall@1 on this eval set.** Stated plainly and reported
honestly, per instructions, rather than tuned away in this step - this is a real, if mixed,
finding: reranking helps at k=5/10 but actively hurts the single-best-result case here.

### 9. Rank-shift diagnostics

```
gold_in_top_20_count: 1,465 / 2,000 (73.25%)
improved: 527   unchanged: 960   worsened: 513
newly_lost: 0   newly_found: 0   (both correctly 0 - same candidate set, only reordered)
mean_rank_before: 3.91   mean_rank_after: 3.46
median_rank_before: 2    median_rank_after: 2
```

Mean rank improved modestly; median held flat - consistent with reranking helping a meaningful
minority of mid-ranked cases move up (explaining the Recall@5/10 gains) while simultaneously
demoting a comparable number of already-well-ranked cases (explaining the Recall@1 loss).

**Manually inspected examples (full text, real records, not fabricated):**
- **Improved:** "Winona Ryder has acted." - FAISS's top hit (rank 6) was a narrow trivia
  sentence about her screen debut; the reranker promoted (to rank 1) a longer sentence directly
  describing her acting career and awards - arguably the more *topically* relevant sentence,
  even though FAISS's original pick was also legitimate evidence.
- **Unchanged:** "Twitter is based in the United States." - both stages agree on rank 1
  (a clear, unambiguous case).
- **Worsened:** "Machu Picchu was built with brick walls." - FAISS correctly ranked the exact
  evidence sentence ("...with polished dry-stone walls...") **first**; the reranker demoted it to
  **rank 11**, preferring Machu Picchu's general introductory/definitional sentence instead. This
  is a clear, understandable failure mode: the reranker favors broad topical relevance over the
  narrow, specific factual sentence FEVER's annotation convention actually requires - a
  cross-encoder trained on general relevance judgments does not automatically match a
  narrow single-sentence-evidence task.

### 10. Performance

```
FAISS:     avg 28.1ms | p50 23.0ms | p95 31.4ms
Reranker:  avg 24.7ms | p50 23.5ms | p95 30.1ms   (batch of 20 candidates/query, batch_size=16)
End-to-end: avg 52.9ms | p50 49.1ms | p95 57.8ms
```

Measured over all 2,000 eval queries (not a small subsample). Peak GPU memory during the full
run: **1.84GB**. Not optimized - a real baseline. Note FAISS latency here (28.1ms avg) is
somewhat higher than Step 13's isolated measurement (21.7ms avg) - plausibly explained by shared
GPU contention with the concurrently-loaded reranker model and other processes on this shared
machine, not investigated further (out of scope - "establish a baseline," not optimize).

### 11. Batching

`batch_size: 16` (from `configs/models.yaml`, the existing project convention - NOT assumed
appropriate from the embedding model's own batch size, verified separately). With 20 candidates
per query, this means 2 sub-batches per query (16 + 4). No OOM occurred at any point; peak GPU
memory (1.84GB across the full 2,000-query run, 1.06GB during the smoke test) stayed far below
the machine's 48GB capacity, so no batch-size adjustment was needed or attempted.

### 12. Reproducibility (`reranker_manifest.json`)

Reranker `verified_info` (model name, dtype, device, param count, activation function, score
type/direction), pipeline config (`retrieval_top_k`, `rerank_top_n`), evaluation config (seed,
sample_size, source), Python version, PyTorch version, GPU name, FAISS index vector count - all
saved as one file. The same query set reproduces the same ranking within the tolerance verified
in the smoke test (`max_abs_diff = 0.0` across two independent calls on identical input).

### 13. Leakage / safety checks

Same structural guarantee as Step 13: a new `TestRAGTruthTruthfulQAExclusion` test parses the
`reranking/` package's AST and asserts no `ragtruth`/`truthfulqa` module is ever imported - not
a runtime check that could be bypassed, a static guarantee. No RAGTruth or TruthfulQA text
reached the reranker, the evaluation set, or any saved artifact. **The DeBERTa verifier was
never loaded, called, or referenced by any code in this step** - `src/claimguard/reranking/`
does not import `claimguard.verifier` anywhere, keeping retrieval/reranking evaluation cleanly
isolated from claim verification, exactly as instructed.

### 14. Tests

19 new tests, `tests/test_reranking.py`: `TestRAGTruthTruthfulQAExclusion` (AST-based, as
above), `TestScore` (one float per candidate; empty input short-circuits), `TestRerank`
(output shape, finite scores, descending-score reranking order, FAISS-score preservation,
provenance preservation, original/reranked rank tracking, top-N truncation-to-highest-scoring
[not just first-N-input], top-N-larger-than-input, empty candidates, duplicate-corpus_id
candidates scored independently, determinism across repeated calls), `TestRetrieveAndRerank`
(both pipeline stages returned), `TestConfigLoads`, and `TestLoadRerankerRealModel` (guarded -
real model: score-direction attribute, correct-answer-ranks-highest on a real example, finite
real scores).

**Full test suite: 232/232 passed** (213 before this step, +19 new), run twice - once before
touching the GPU (confirming the reranking logic was correct before spending GPU time on the
smoke test/full evaluation), once after.

### 15. Limitations (stated explicitly)

- **Reranking degraded Recall@1 by 12.7% relative** on this eval set - a real, honestly-reported
  finding, not tuned away. This step deliberately did not attempt to fix, tune, or explain this
  away (e.g. by adjusting `rerank_top_n`, trying a different score-combination strategy, or
  filtering); that would be a controlled follow-up experiment, out of scope here.
  Recall@5/10 did improve (+3.9%/+5.9%), so reranking's net effect depends heavily on which
  cutoff the eventual pipeline actually relies on downstream.
  - A plausible mechanism was identified via manual example inspection (Section 9): the
  cross-encoder favors broad topical relevance over FEVER's narrow single-sentence-evidence
  convention - a hypothesis, not a proven root cause, and not further investigated in this step.
- This is retrieval+reranking infrastructure only - no verifier integration, no correction loop,
  no generator integration, no RAGTruth/TruthfulQA evaluation.
- Only ONE controlled configuration was evaluated (`retrieval_top_k=20`, `rerank_top_n=5`,
  `batch_size=16`) - no parameter sweep, per instructions.
- FAISS latency in this step (28.1ms avg) is somewhat higher than Step 13's isolated measurement
  (21.7ms avg), plausibly due to shared-GPU contention with the reranker model - not
  investigated further.

**Preserved, unchanged:** RAGTruth test remains untouched. TruthfulQA remains untouched. The
Step 12 binary verifier checkpoint was never loaded or touched. The Step 13 FAISS index and
embeddings were not rebuilt. No hyperparameter sweep, no verifier+retrieval integration.

**Git status:** still no commits — new files (`src/claimguard/reranking/*`,
`configs/reranker_baseline.yaml`, `scripts/reranker_smoke_test.py`,
`scripts/evaluate_reranker.py`, `tests/test_reranking.py`,
`data/processed/retrieval/reranker_eval_results.json`,
`data/processed/retrieval/reranker_manifest.json`), modified files
(`src/claimguard/config.py`, `src/claimguard/retrieval/eval.py`, `RESEARCH.md`) show as
untracked/modified.

---

## Step 15 — Retrieval + reranking + binary verifier integration (2026-09-03)

**What:** Read `RESEARCH.md`, this report, `dataset_manifest.json`, the retrieval/reranker
manifests and eval results, verifier experiment metadata, `src/claimguard/retrieval/*`,
`src/claimguard/reranking/*`, `src/claimguard/verifier/*`, and existing tests/configs. Built a
new `src/claimguard/verification/` package (`verify.py` for loading the Step 12 checkpoint and
scoring premise/hypothesis pairs, `pipeline.py` for the full orchestration), a new config
(`configs/integration_baseline.yaml`), ran a real end-to-end smoke test, evaluated all 2,000
FEVER-eval-set queries through the complete pipeline, compared three evidence-selection
strategies, ran threshold diagnostics and latency measurement, added 18 new tests, and ran the
full test suite twice. Did **not** retrain/tune the verifier or reranker, did **not** rebuild the
embedding model or FAISS index, and did **not** evaluate RAGTruth/TruthfulQA.

**Why:** With retrieval (Step 13) and reranking (Step 14) each validated in isolation - and
Step 14 specifically showing reranker-top-1 is *weaker* than FAISS-top-1 on Recall@1 - the next
question is whether adding the verifier's own judgment over multiple candidates recovers or
improves evidence selection, and how the fully-integrated system performs end-to-end, before any
generator or correction-loop work begins.

### 1. Integrated architecture

```
query -> embed(query) [BGE] -> FAISS top-20 -> BGE reranker -> reranked top-20 (full permutation)
      -> top-5 verified by binary DeBERTa (premise=candidate text, hypothesis=query)
      -> aggregation: selected_evidence = argmax(entailment_probability) among the 5
```

`configs/integration_baseline.yaml`: `retrieval_top_k=20`, `rerank_top_n=5`,
`verifier_max_length=256` (unchanged from Step 9/12), `entailment_threshold=0.5` (explicitly
labeled INITIAL, never tuned against any benchmark in this step). The reranker still reorders
the FULL 20-candidate FAISS list (not just the top-5) so reranked Recall@k can be verified at
every cutoff exactly as in Step 14 - only the top 5 of that full reranking are then passed to the
verifier, matching the architecture diagram.

### 2. Verifier input semantics

For each of the 5 top-reranked candidates: `premise = candidate.text`, `hypothesis = query`
(the SAME query text used for embedding and reranking - never altered between stages). Binary
output mapped explicitly: `entailment -> "supported"`, `contradiction -> "contradicted"` - no
neutral score is invented (`claimguard.verification.verify.VERIFIER_LABEL_TO_PIPELINE_LABEL` has
exactly two entries, tested to confirm `"neutral"` is absent).

### 3. Verifier score aggregation - the critical design decision

Per instructions, the reranker top-1 is NOT assumed to be final evidence. Every one of the 5
verified candidates carries: `entailment_probability`, `contradiction_probability`,
`reranker_score` (renamed from the FAISS/reranking stage), `faiss_score`, `original_rank`
(FAISS position), `reranked_rank`. Aggregation rule (the simplest defensible one, per
instructions - no learned fusion): `final_support_score = max(entailment_probability)` across the
5 candidates; `selected_evidence` = the candidate achieving that max (deterministic tie-break:
earliest reranked_rank among ties, verified by a dedicated test). The reranker-top-1 and
FAISS-top-1 candidates are retained separately on every `PipelineResult`, so all three selection
strategies can always be compared directly for the same query.

### 4. Integration module (`src/claimguard/verification/`)

- `verify.py`: `load_verifier(checkpoint_dir, device, dtype)` loads the Step 12 checkpoint,
  re-verifies (via `claimguard.verifier.binary_model.assert_binary_head`, not assumed) that it is
  still a genuine 2-class head, and explicitly moves it to bf16/cuda (the checkpoint alone loads
  CPU/fp32 via `from_pretrained` - left as-is that would silently diverge from the precision the
  model was actually trained/evaluated in, and be needlessly slow for 10,000 real inferences).
  `verify_pair()` / `verify_candidates()` score (premise, hypothesis) pairs via softmax over the
  2 real logits - never a fabricated third class.
- `pipeline.py`: `PipelineResult` dataclass (query, faiss_candidates, reranked_candidates,
  verified_candidates, faiss_top1, reranker_top1, selected_evidence, final_support_score,
  entailment_threshold, final_supported), `select_evidence()` (Section 3's aggregation),
  `run_pipeline()` (the full orchestration - its ONLY per-query parameter is `query`; no
  gold-evidence or gold-label parameter exists anywhere in its signature, verified by a dedicated
  structural test using `inspect.signature`).

### 5. Device/model management

All three models (BGE embedding ~0.6GB, BGE reranker ~1.06GB, binary verifier ~0.87GB in bf16)
load and run SIMULTANEOUSLY - measured peak GPU memory across the full 2,000-query run was only
**2.65GB**, far below the 48GB budget, so no sequential loading/offloading logic was implemented
(would have been unnecessary complexity for a measured non-problem). Model precision unchanged
from each component's existing config (embedding/reranker fp16 per Step 5C/5D, verifier bf16 per
Step 9/12) - not altered here.

### 6. Integration smoke test - PASS (real GPU, real models)

Real end-to-end run (embedding -> FAISS -> reranker -> verifier) on 3 hand-picked claims:
- "Paris is the capital of France." (true) → selected evidence `Seat_of_government::3`,
  entailment_prob=0.9993, `final_supported=True`.
- "The Eiffel Tower was built in Berlin." (false) → selected evidence `Berlin::0`,
  entailment_prob=0.0071, `final_supported=False`.
- "Water boils at 100 degrees Celsius at sea level." (true) → selected evidence `Steam::0`,
  entailment_prob=0.9959, `final_supported=True`.

All three real examples classified correctly and semantically sensibly. Confirmed: correct
candidate counts (20/20/5) at every stage, every provenance field present on every candidate,
all FAISS/reranker scores and verifier probabilities finite, entailment+contradiction
probabilities summing to 1.0 within `1e-4`, a selected evidence always exists for non-empty
input, empty-candidate handling correct at both the `verify_candidates()` and `select_evidence()`
levels. Peak GPU memory 2.531GB with all 3 models loaded; GPU returned to the shared-baseline
free level after cleanup.

### 7-9. Evaluation results (2,000-query eval set, identical to Steps 13/14)

**A. Retrieval (consistency check):** Recall@{1,5,10,20} = {0.2800, 0.5685, 0.6620, 0.7325} -
exact match to Step 13.

**B. Reranking (consistency check):** Recall@{1,5,10,20} = {0.2445, 0.5905, 0.7010, 0.7325} -
exact match to Step 14.

**C. Verifier evidence selection - evidence-selection strategy comparison:**

| Strategy | Definition | Gold-hit rate |
|---|---|---|
| A | FAISS top-1 | 0.2800 |
| B | Reranker top-1 | 0.2445 |
| **C** | **Verifier-selected (max entailment among reranked top-5)** | **0.3180** |

Strategy C beats both A and B - the first direct evidence that verifier-based selection over a
preserved candidate set is the right response to Step 14's reranker-top-1 weakness, not merely a
theoretical justification.

**D. Verifier prediction (using PIPELINE-selected evidence, positive class = entailment):**

| Metric | Value |
|---|---|
| Accuracy | 0.8860 |
| Precision (entailment) | 0.8763 |
| Recall (entailment) | 0.9873 |
| F1 (entailment) | 0.9285 |
| Precision (contradiction) | 0.9389 |
| Recall (contradiction) | 0.5828 |
| F1 (contradiction) | 0.7192 |

Confusion matrix (order=[entailment, contradiction]): `[[1480, 19], [209, 292]]`. **Explicitly
NOT comparable to Step 12's gold-premise 0.9585 macro F1** - this number necessarily includes
retrieval/reranking error propagation, by construction, and is reported as a distinct "pipeline
accuracy" metric, not a re-measurement of verifier quality in isolation.

### 10. Threshold diagnostics (informational only, NOT tuned)

| Group | n | Mean entailment prob | Median | Fraction ≥0.5/0.6/0.7/0.8/0.9 |
|---|---|---|---|---|
| Gold evidence, true=SUPPORTS | 1,024 | 0.9826 | 0.9987 | 0.985 / 0.983 / 0.981 / 0.979 / 0.973 |
| Gold evidence, true=REFUTES | 252 | 0.0807 | 0.0217 | 0.048 / 0.044 / 0.036 / 0.032 / 0.032 |
| Non-gold evidence | 8,724 | 0.6968 | 0.9758 | 0.704 / 0.686 / 0.665 / 0.637 / 0.594 |

**Explains the contradiction-class recall gap in Section 9D:** the verifier behaves correctly on
genuine gold evidence (near-certain entailment for real support, near-certain contradiction for
real refutation) - but non-gold evidence's median entailment probability (0.9758) is nearly as
high as genuine gold support. When the pipeline fails to retrieve a genuinely refuting sentence
for a REFUTES claim (which happens often - reranked Recall is only 0.70 at k=10), the verifier
tends to call whatever topically-adjacent candidate it does see "supported" rather than correctly
signaling uncertainty - a real, quantified limitation, not glossed over. No neutral class exists
to express "insufficient/irrelevant evidence" (Step 11's standing limitation), which plausibly
compounds this: the verifier is architecturally forced to pick entailment or contradiction even
when neither actually applies to a given candidate.

### 11. Latency

```
embed+FAISS: avg 29.7ms | p50 25.4ms | p95 32.4ms
Reranker:    avg 26.4ms | p50 23.6ms | p95 41.0ms   (20 candidates/query)
Verifier:    avg 102.9ms | p50 98.4ms | p95 128.7ms  (5 candidates/query)
End-to-end:  avg 159.1ms | p50 147.6ms | p95 199.3ms
```

Measured over all 2,000 queries. Model-loading overhead (one-time, separate from per-query
latency): 10.7s for all three models. The verifier stage dominates end-to-end latency despite
processing fewer candidates (5 vs. 20) - consistent with DeBERTa-v3-large being the largest of
the three models and running in bf16 rather than fp16.

### 12. GPU memory

Peak across the full 2,000-query run: **2.65GB** (smoke test: 2.53GB with all 3 models loaded).
Confirms Section 5's decision not to implement sequential model loading/offloading - the combined
footprint is nowhere near the 48GB budget even under full load.

### 13. Artifacts

`data/processed/integration/`: `integration_eval_results.json` (full recall/strategy/
classification/threshold/latency results plus a per-claim summary for all 2,000 queries),
`integration_manifest.json` (compact model/config/package-version metadata), `integration_examples.jsonl`
(30 full example records with complete candidate lists, for manual inspection - not all 2,000,
avoiding unnecessary duplication of the corpus text already in `data/processed/retrieval/`).

### 14. Leakage / safety checks

Same structural guarantee as Steps 13-14: `TestRAGTruthTruthfulQAExclusion` parses the
`verification/` package's AST and confirms no `ragtruth`/`truthfulqa` import exists anywhere.
**No gold-evidence or gold-label access during inference** - verified structurally
(`inspect.signature(run_pipeline)` has no gold-shaped parameter) rather than only by convention;
gold information is read only inside the evaluation script, strictly after `run_pipeline()`
returns.

### 15. Tests

18 new tests, `tests/test_verification.py`: `TestRAGTruthTruthfulQAExclusion`,
`TestNoGoldAccessDuringInference` (the structural signature checks above), `TestSelectEvidence`
(max-probability selection, empty input, deterministic tie-breaking, determinism across repeated
calls), `TestRunPipelineOrdering` (candidate-count preservation at every stage using fake
retriever/reranker objects, FAISS-score survival through reranking+verification unchanged, empty
FAISS results handled cleanly), `TestLoadVerifierRealCheckpointFormat` (a real, tiny, locally-saved
binary DebertaV2 checkpoint exercised through the real `load_verifier()`/`verify_pair()` code
path: correct id2label, probabilities summing to 1, correct pipeline-label mapping, empty-list
handling, rejection of a non-binary checkpoint), `TestConfigLoads`, and `TestLoadRealVerifierCheckpoint`
(guarded - the real Step 12 checkpoint: confirmed still binary, correctly scores a real
supporting pair as entailment).

**Full test suite: 250/250 passed** (232 before this step, +18 new), run twice - once catching a
test-fixture bug (below), once clean.

### 16. Bugs discovered and fixed

**One real test-fixture bug**, found and fixed during this step's own test development (not a
source-code bug): `test_verify_pair_probabilities_sum_to_one` built a tiny DebertaV2 model with
`vocab_size=100` but tokenized with the REAL DeBERTa-v3 tokenizer (vocabulary ≈128K tokens) -
the tokenizer legitimately produced token ids the tiny model's undersized embedding table had no
row for, raising a real `IndexError: index out of range in self`. `verify_pair()`/`pipeline.py`
themselves behaved exactly correctly (propagating a genuine shape error rather than masking it).
**Fixed** by sizing the tiny fixture's vocabulary to `len(tokenizer)` for any test that performs
a real forward pass, documented directly in the fixture's docstring so the same mistake isn't
repeated in a future step.

### 17. Limitations (stated explicitly)

- **This is the first integrated retrieval + reranking + verifier experiment. It is NOT the
  final ClaimGuard end-to-end evaluation.** No generator, no correction loop.
- **Reranker top-1 is weaker than FAISS top-1** on the current retrieval benchmark (Step 14) -
  the integrated system responds to this by preserving multiple candidates and using
  verifier-based evidence selection (Strategy C), which does measurably outperform both naive
  top-1 strategies (Section 9C), but does not fully close the gap to gold-premise verifier
  accuracy.
- The verifier is systematically overconfident on non-gold (irrelevant/wrong) retrieved evidence
  (median entailment probability 0.9758) - directly depressing contradiction-class recall
  end-to-end. This is a real, quantified, and unresolved limitation, plausibly connected to the
  standing absence of a neutral class (Step 11).
- `entailment_threshold=0.5` is explicitly an INITIAL value, never tuned against any benchmark in
  this step - a future controlled threshold-selection experiment is a separate piece of work.
- Verifier latency (avg 102.9ms/query for only 5 candidates) dominates end-to-end latency and was
  not optimized (out of scope - "establish a baseline").
- No learned score fusion was introduced (per instructions) - aggregation is the simplest
  defensible rule (max entailment probability), not a tuned or trained combination.

**Preserved, unchanged:** RAGTruth test and TruthfulQA remain completely untouched. The Step 12
verifier checkpoint was loaded read-only, never modified. The Step 13/14 FAISS index/embeddings/
reranker were not rebuilt or tuned. No Qwen3 generator, no correction/regeneration logic.

**Git status:** still no commits — new files (`src/claimguard/verification/*`,
`configs/integration_baseline.yaml`, `scripts/integration_smoke_test.py`,
`scripts/evaluate_integration.py`, `tests/test_verification.py`, `data/processed/integration/*`),
modified files (`src/claimguard/config.py`, `RESEARCH.md`) show as untracked/modified.

---

## Step 16 — Verifier calibration and evidence decision policy (2026-09-03)

**What:** Read `RESEARCH.md`, this report, Step 12-15 results/manifests, and all current
verifier/retrieval/reranking/verification code and tests (zero drift confirmed via diff against
freshly-pulled remote copies before editing). Built a new `src/claimguard/decision/` package
(`policy.py` - `EvidenceDecisionPolicy`, `Decision`, `decide()`; `calibration.py` - ECE/Brier/
reliability bins), a new config (`configs/decision_policy.yaml`), a smoke test, a full 2,000-query
analysis script that reuses (not duplicates) `build_eval_set`/`first_hit_rank`/`recall_at_cutoffs`/
`Retriever`/`reranker.rerank`/`verify.load_verifier`/`verify_candidates`/`pipeline.run_pipeline`/
`select_evidence`, 33 new tests, and ran the full suite twice (283/283 both times). Did **not**
retrain, fine-tune, or otherwise modify the verifier; did **not** evaluate RAGTruth/TruthfulQA; did
**not** implement any learned decision layer.

**Why:** Step 15 found the verifier is often overconfident on non-gold evidence (median entailment
prob 0.9758), directly depressing contradiction-class recall. Before any generator/correction work,
this needed formal characterization (calibration diagnostics, threshold sweeps, score
distributions) and a decision on whether a safe, deterministic (non-learned) evidence-decision
policy - including an ABSTAIN outcome - can be built on top of the existing verifier as-is.

### 1. Problem framing: Problem A vs Problem B

Step 15 solved **Problem A** ("which candidate is the best evidence?" - `select_evidence`, max
entailment among the reranked top-5). Step 16 addresses the separate **Problem B** ("given the
verifier's scores, is the query actually SUPPORTED, CONTRADICTED, or should the system ABSTAIN?").
ABSTAIN is a **downstream policy outcome**, not a third model class - it must not be confused with
Step 11's "neutral" investigation (which asked whether the *verifier itself* should train on a
neutral label; answer was no legitimate source existed, so the verifier remains strictly binary).
ABSTAIN here can fire regardless of which of the two verifier classes scored higher, whenever
neither clears the configured confidence bar.

### 2. Eval set reuse (identical to Steps 13-15)

`claimguard.retrieval.eval.build_eval_set(seed=42, sample_size=2000, source=fever_train)`, then
cross-checked example-id-for-example-id against Step 13's saved results (`eval_set_identical_to_step13:
true` in both new artifacts). No RAGTruth, no TruthfulQA, no gold evidence/labels reach pipeline
inference - gold information is used only after `run_pipeline()` returns.

### 3. Data collection: full per-candidate records (new relative to Step 15)

Step 15 saved only compact per-claim summaries (30 example records + aggregate threshold
diagnostics with 3 combined groups). Step 16 needed richer, per-candidate data to compute 4-way
distributions, candidate-level precision/recall/F1/FPR/FNR, and gold-only calibration - so
`scripts/analyze_decision_policy.py` re-runs the real pipeline (embed→FAISS→rerank→verify, via
`pipeline_mod.run_pipeline`, unchanged from Step 15) once per query, but this time keeps every
verified candidate's `entailment_probability`, `contradiction_probability`, `is_gold`, and
`true_label` for all 2,000 × 5 = **10,000 candidate records** (validated by read-back: the 4
score-distribution groups sum to exactly 10,000). No GPU work was duplicated relative to Step 15 -
the extra cost is pure-Python aggregation on top of the same pipeline calls.

### 4. Score distributions - 4 groups (new, finer-grained than Step 15's 3-group version)

| Group | n | Mean | Median |
|---|---|---|---|
| Gold evidence, true=SUPPORTS | 1,024 | 0.9826 | 0.9987 |
| Gold evidence, true=REFUTES | 252 | 0.0807 | 0.0217 |
| Non-gold evidence, SUPPORTS claims | 6,471 | 0.8556 | 0.9940 |
| Non-gold evidence, REFUTES claims | 2,253 | 0.2407 | 0.0337 |

Splitting Step 15's single "non_gold_evidence" group by the claim's true label reveals the
overconfidence problem is **one-sided**: non-gold evidence retrieved for SUPPORTS claims scores
almost as high (median 0.9940) as genuine gold-SUPPORTS evidence (0.9987) - essentially
indistinguishable by threshold. Non-gold evidence for REFUTES claims scores much lower (median
0.0337), close to genuine gold-REFUTES evidence. The verifier readily over-calls "supported" on
wrong evidence, but rarely over-calls "refuted".

### 5. Candidate-level threshold analysis (characterization only - no threshold selected)

Definitions (documented in `claimguard.decision.policy` module docstring, not left implicit):
"accepted support" = `entailment_probability >= threshold`; "true support" = the candidate is gold
evidence for a SUPPORTS claim; "false support" = accepted AND NOT true support; "missed support" =
NOT accepted AND true support. Computed across all 10,000 candidate records (both SUPPORTS and
REFUTES claims):

| t | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|
| 0.50 | 0.1409 | 0.9854 | 0.2466 | 0.6852 | 0.0146 |
| 0.60 | 0.1438 | 0.9834 | 0.2510 | 0.6678 | 0.0166 |
| 0.70 | 0.1474 | 0.9814 | 0.2564 | 0.6474 | 0.0186 |
| 0.80 | 0.1527 | 0.9795 | 0.2642 | 0.6200 | 0.0205 |
| 0.90 | 0.1610 | 0.9727 | 0.2763 | 0.5781 | 0.0273 |
| 0.95 | 0.1711 | 0.9668 | 0.2907 | 0.5344 | 0.0332 |

Precision stays in the 0.14-0.17 band across the ENTIRE swept range while recall stays ≥0.97 -
raising the threshold to 0.95 buys almost nothing, because the false-positive-dominant group
(non-gold-for-SUPPORTS-claims) clusters at the same near-1.0 confidence as true positives. This is
a **quantified confirmation** that entailment probability alone cannot reliably distinguish correct
from incorrect evidence, at any threshold in the swept range.

### 6. Calibration diagnostics (gold-evidence candidates only)

Scope decision, documented in `claimguard.decision.calibration`'s module docstring: ECE/Brier
require a ground-truth "correct" label per scored pair, which FEVER only supplies for gold
evidence (a non-gold candidate paired with a claim has no annotator judgment on whether it entails/
contradicts that claim - fabricating one would misrepresent measurement as ground truth). Computed
over the 1,276 gold-evidence candidates (1,024 SUPPORTS + 252 REFUTES):

- **ECE = 0.0092** (10 fixed bins, [0.0,0.1)...[0.9,1.0])
- **Brier score = 0.0178** (0.0 = perfect, 0.25 = always-predict-0.5)
- **Bin-weighted verdict: "overconfident"** (majority of points fall in bins where mean confidence
  exceeds accuracy), though both ECE and Brier are numerically low.

**Key interpretation:** the verifier is well-calibrated ON GOLD EVIDENCE - when ground truth
exists, its confidence tracks its accuracy closely. The practical failure is therefore not raw
miscalibration but a **discrimination failure**: non-gold evidence that happens to be
topically/lexically similar to a SUPPORTS claim gets scored with the same confidence as genuinely
correct evidence, and there is no ground-truth-based way to penalize that scoring behavior from
gold-only calibration data alone (see Section 4's descriptive, non-accuracy-labeled non-gold
statistic in `calibration_results.json`).

### 7. Evidence-selection policy comparison (A/B/C/D)

| Strategy | Hit rate / behavior |
|---|---|
| A: FAISS top-1 | 0.2800 (reconfirmed, consistency check against Steps 13-15) |
| B: Reranker top-1 | 0.2445 (reconfirmed - still weaker than A, per Step 14's finding) |
| C: Verifier max-entailment among top-5 | 0.3180 (reconfirmed, matches Step 15 exactly) |
| D: C + minimum confidence threshold | see below |

Strategy D (accept the max-entailment candidate only if it clears threshold t, else abstain):

| t | Coverage (non-abstain) | Precision among accepted |
|---|---|---|
| 0.50 | 0.8445 | 0.3570 |
| 0.60 | 0.8360 | 0.3606 |
| 0.70 | 0.8275 | 0.3637 |
| 0.80 | 0.8145 | 0.3696 |
| 0.90 | 0.7970 | 0.3751 |
| 0.95 | 0.7805 | 0.3831 |

Even conditioning on the pipeline's own high-confidence queries, only ~36-38% of "accepted"
selections are actually gold evidence - confidence thresholding recovers only marginal precision
gains, confirming the overconfidence problem is systemic rather than a tail-probability effect.

### 8. Deterministic policy outcomes (`EvidenceDecisionPolicy`, SUPPORTED/CONTRADICTED/ABSTAIN)

| t | Supported | Contradicted | Abstain | False-support rate |
|---|---|---|---|---|
| 0.50 | 1,689 | 311 | 0 | 0.6454 |
| 0.60 | 1,672 | 328 | 0 | 0.6417 |
| 0.70 | 1,655 | 345 | 0 | 0.6381 |
| 0.80 | 1,629 | 369 | 2 | 0.6323 |
| 0.90 | 1,594 | 395 | 11 | 0.6267 |
| 0.95 | 1,561 | 396 | 43 | 0.6188 |

Abstention barely rises (0→43 of 2,000, i.e. 2.15%) even at t=0.95, and the false-support rate
stays pinned at **62-65%** across the entire swept range. Raising the threshold shaves off only the
least-confident errors while leaving the bulk of overconfident false positives (which cluster near
1.0) completely unaffected.

### 9. Evidence-consistency (margin) analysis - the one promising lever

Per-query margin = top-1 minus top-2 entailment probability among the reranked-top-5 verified
candidates. Splitting at the median margin: **high-margin queries have selection precision
0.4607**, vs **0.2765 for low-margin queries** (a ~67% relative improvement, computed via a simple
deterministic grouping - no learned fusion). This is the most promising signal found for improving
precision, though still far from a safe production threshold on its own.

### 10. Honest conclusion on decision-policy safety (explicit instruction: do not force a policy)

**No single entailment-probability threshold, anywhere in the swept 0.50-0.95 range, is a safe
deterministic policy for production evidence selection** - precision stays in the 0.14-0.17 band
(candidate-level) / false-support rate stays 62-65% (query-level policy) regardless of threshold,
because the overconfident non-gold-for-SUPPORTS-claims evidence is not separable from genuine
gold-SUPPORTS evidence by confidence alone. `EvidenceDecisionPolicy` was still implemented,
correctly and fully tested, as the required deterministic interface (`decide(query,
verified_candidates)` → `Decision(decision, confidence, selected_evidence, reason, ...)`), but its
default configuration (`configs/decision_policy.yaml`, entailment_threshold=contradiction_threshold
=0.5, carried over unchanged from Step 15) is explicitly labeled a *characterization baseline*, not
a validated safe policy. Margin-based consistency filtering (Section 9) is the most promising
direction for follow-up work - a separate, future controlled dev-set experiment, never tuned here
and never tuned against RAGTruth.

### 11. `src/claimguard/decision/` module

- `policy.py`: `Decision` (dataclass: decision/confidence/selected_evidence/reason/
  max_entailment_probability/max_contradiction_probability/second_highest_entailment_probability/
  entailment_margin/num_candidates/num_above_*_threshold/query - never a bare boolean, always
  carries full provenance), `EvidenceDecisionPolicy` (deterministic threshold + optional margin
  rule, no learned parameters), `compute_query_aggregates`, `count_above_threshold`, module-level
  `decide(query, verified_candidates, ...)` matching the required interface. All non-finite
  (NaN/Inf) verifier probabilities are rejected (raised, never silently dropped or clamped).
- `calibration.py`: `gold_calibration_points`, `reliability_bins` (fixed 10 bins, empty bins still
  reported), `expected_calibration_error`, `brier_score`, `confidence_distribution`,
  `is_systematically_overconfident` - scoped to gold-evidence candidates only, documented why.

### 12. Tests (33 new, `tests/test_decision.py`)

RAGTruth/TruthfulQA AST-based import exclusion (reused pattern from Steps 13-15); structural
no-gold-access checks on `decide()` and `EvidenceDecisionPolicy.decide` via `inspect.signature`;
SUPPORTED/CONTRADICTED/ABSTAIN decisions individually; threshold boundary inclusivity; entailment-
priority-over-contradiction tie-breaking rule; determinism across repeated calls; full
evidence-provenance preservation on `selected_evidence`; decision is never a bare boolean; empty
candidates; conflicting/near-tie candidates (with and without `require_margin`); NaN/Inf rejection
(both `decide` and `compute_query_aggregates`); query-aggregate margin/second-highest correctness;
`count_above_threshold`; module-level `decide()` stamps `query` and accepts policy kwargs; ECE/
Brier/reliability-bin correctness (including a synthetic perfectly-calibrated case and an empty-
input case); overconfidence-verdict detection. Full suite (283 tests: 250 pre-existing + 33 new)
passed twice - once before the GPU analysis run, once after.

### 13. Latency and GPU

Total loop time 450.6s for 2,000 queries (vs Step 15's 318.3s for the same pipeline calls - the
difference is pure-Python overhead from building 10,000 candidate records and 6 threshold sweeps'
worth of `EvidenceDecisionPolicy.decide()` calls per query, not additional GPU work). Peak GPU
memory 2.647GB - identical to Step 15, confirming no new GPU cost was introduced. Model load 11.2s.

### 14. Artifacts (validated by read-back)

- `data/processed/integration/calibration_results.json` - score distributions (4 groups), gold-only
  ECE/Brier/reliability bins, non-gold confidence (descriptive only, no accuracy label).
- `data/processed/integration/decision_policy_results.json` - candidate-level threshold analysis,
  evidence-selection comparison (A/B/C/D), deterministic policy outcomes per threshold, evidence-
  consistency analysis, latency, limitations.
- `configs/decision_policy.yaml` - thresholds swept, calibration bin count, default policy config.
- Read-back validation: both artifacts' `eval_set_identical_to_step13` = `true`; 4 score-
  distribution groups sum to exactly 10,000 (2,000 × 5); 1,276 gold calibration points
  (1,024 + 252, matching the score-distribution counts exactly).

### 15. Limitations (explicitly documented, not glossed over)

- No safe deterministic threshold was found (Section 10) - this is a genuine negative result, not
  a step that produced a "best" threshold. Any threshold in this report is a characterization data
  point only, never a recommendation.
- Candidate-level threshold analysis treats gold-REFUTES and non-gold candidates identically as
  negatives for the "accepted support" detector - a deliberate, documented simplification (see
  Section 5 / module docstring), not an oversight.
- Calibration (ECE/Brier) is restricted to gold-evidence candidates only; non-gold confidence has
  no legitimate ground-truth "correct" label and is reported descriptively, never with an accuracy
  claim attached.
- The binary verifier remains completely unchanged - Step 16 adds only a downstream evidence-
  decision policy. No retraining, no fine-tuning, no new training data.
- No RAGTruth/TruthfulQA evaluation, no Qwen3 generator/correction work, no hyperparameter tuning,
  no learned fusion - all explicitly out of scope per instructions.

**Preserved, unchanged:** the Step 12 verifier checkpoint, Step 13 FAISS index/embeddings, Step 14
reranker, and Step 15 pipeline orchestration were all loaded read-only and not modified, retrained,
or rebuilt. RAGTruth test and TruthfulQA remain completely untouched.

**Git status:** still no commits — new files (`src/claimguard/decision/*`,
`configs/decision_policy.yaml`, `scripts/decision_policy_smoke_test.py`,
`scripts/analyze_decision_policy.py`, `tests/test_decision.py`, `data/processed/integration/
calibration_results.json`, `data/processed/integration/decision_policy_results.json`), modified
files (`src/claimguard/config.py`, `RESEARCH.md`) show as untracked/modified.

---

## Step 17 — Qwen3-8B generator integration (2026-09-03)

**What:** Ran a preflight (environment, GPU/VRAM, existing test count, Step 16 artifact
presence, and an explicit grep of `CLAIMGUARD_MODEL_SELECTION.md`/`README.md`/`PROJECT_REPORT.md`
for the approved generator identifier). Read `PROJECT_PLAN.md`, `CLAIMGUARD_MODEL_SELECTION.md`,
`configs/models.yaml`, `scripts/smoke_test_qwen.py` (Step 5A), and the existing
`src/claimguard/generation/__init__.py` placeholder before writing any code. Implemented
`src/claimguard/generation/` (`qwen3.py`, `types.py`), a new `configs/generator_baseline.yaml`,
24 new tests, a real-GPU smoke test script that also saves a compact JSON artifact, and ran the
full suite twice (307/307 both times). Did **not** implement the correction/regeneration loop,
did **not** evaluate RAGTruth/TruthfulQA, did **not** fine-tune/quantize/LoRA the model, did
**not** touch retrieval/reranker/verifier/decision-policy code.

**Why:** Steps 13-16 built and validated retrieval, reranking, the binary verifier, their
integration, and a calibration/decision-policy layer - all on the EVIDENCE side of ClaimGuard.
Before any correction loop can exist, the project needs a working, independently-callable
GENERATOR component (the source of the candidate answers ClaimGuard will eventually verify and
correct), loaded and validated on real hardware exactly as approved in
`CLAIMGUARD_MODEL_SELECTION.md`.

### 1. Model identifier: no ambiguity

Grepped `CLAIMGUARD_MODEL_SELECTION.md`, `README.md`, `RESEARCH.md`, and `PROJECT_REPORT.md` for
"qwen" before writing any code. All four documents agree exactly: **`Qwen/Qwen3-8B`** is the
approved primary generator (Apache 2.0, ~16GB BF16). `configs/models.yaml`'s
`generator.primary.name` already contains this exact string. No substitution, no guessing - the
identifier was read from the project's own established source of truth, not assumed.

### 2. Prior art found: Step 5A already smoke-tested this exact model

`scripts/smoke_test_qwen.py` (Step 5A, 2026-08-31/09-01) already validated that `Qwen/Qwen3-8B`
loads and generates correctly on this hardware: 8,190,735,360 params (~8.19B), `torch.bfloat16`,
peak 15.27GB VRAM, and two bugs already found and fixed there (a `BatchEncoding`-vs-tensor
`TypeError` from `apply_chat_template`, and the `torch_dtype=` -> `dtype=` deprecation). Step 17's
`qwen3.py` reuses this EXACT validated loading/chat-template pattern rather than re-deriving it
from scratch - including the `enable_thinking` `TypeError` fallback.

### 3. Environment preflight (measured, not assumed)

- Working directory: `~/RJ/ClaimGuard`. Python 3.12.14 (`claimguard` conda env).
- `torch==2.11.0+cu128`, `transformers==5.16.1`, CUDA available.
- GPU: NVIDIA RTX PRO 5000 Blackwell. Free VRAM at preflight: 33.48GB / 47.27GB total (well above
  the 20GB safety threshold reused from Step 5A's script).
- Pre-existing test count: 283/283 passing (Step 16's baseline, reconfirmed before any Step 17
  code was written).
- Step 16 artifacts present and untouched: `data/processed/integration/calibration_results.json`,
  `decision_policy_results.json`.
- `src/claimguard/generation/__init__.py` already existed as a placeholder
  (`"not yet implemented — Step 7"`) - updated to describe the real Step 17 implementation.

### 4. Module structure (`src/claimguard/generation/`)

- `types.py`: `GenerationResult` dataclass (`user_input`, `system_instruction`, `generated_text`,
  `model_identifier`, `prompt_token_count`, `generated_token_count`, `generation_config`,
  `latency_seconds`, plus `to_dict()`) - deliberately generic, never a dataset-label-shaped field.
- `qwen3.py`: `load_generator()` (reads model identity from `configs/models.yaml`, never
  hard-coded; never quantizes/fine-tunes/adds LoRA), `unload_generator()` (explicit
  `del` + `torch.cuda.empty_cache()`), `build_chat_inputs()` (tokenizer's own chat template,
  system+user or user-only, `enable_thinking` with documented `TypeError` fallback),
  `_build_generation_kwargs()` (isolated so `do_sample=False` never leaks `temperature`/`top_p`
  into `model.generate()`), `generate()` (the full deterministic-by-default generation call,
  decodes ONLY `output_ids[0][prompt_len:]` so the prompt is never echoed back).

### 5. Determinism

`do_sample=False` (greedy decoding) plus an explicit `torch.manual_seed`/
`torch.cuda.manual_seed_all` call before generation (`configs/generator_baseline.yaml`:
`seed: 42`). No temperature/top-p tuning performed - both are `null` in the config and structurally
excluded from `model.generate()`'s kwargs whenever `do_sample=False`
(`_build_generation_kwargs`, directly unit-tested). Verified on the REAL model: the same prompt
produced byte-identical output across two independent `generate()` calls.

### 6. Chat-template correctness (inspected, not assumed)

Reused Step 5A's validated `apply_chat_template(messages, add_generation_prompt=True,
return_tensors="pt", return_dict=True, enable_thinking=False)` call, with a `TypeError`-guarded
fallback (retry without `enable_thinking`) for tokenizer/template versions that don't accept the
kwarg - tested with both a fake tokenizer that raises and one that doesn't. Confirmed on the real
tokenizer: no `TypeError` fallback was triggered (the installed `Qwen2Tokenizer`'s template
accepts `enable_thinking` directly). System+user and user-only message construction both tested
against a fake tokenizer that records the exact `messages` list it received.

### 7. Reasoning/thinking mode

Qwen3 supports a "thinking" mode; ClaimGuard explicitly sets `enable_thinking=False` and stores
ONLY the final decoded answer text (`generated_text`) - no chain-of-thought extraction/storage
mechanism was designed or implemented, matching the explicit instruction not to expose or persist
hidden reasoning.

### 8. Resource / memory validation

Real-GPU smoke test measured: model load 9.6s (weights already cached locally from Step 5A - no
re-download), peak allocated VRAM 15.29GB (this process only, via
`torch.cuda.max_memory_allocated()`), generation latency 0.89s/4.05s/2.03s for the three smoke
prompts (8/117/58 generated tokens respectively). After `unload_generator()` and process exit,
`nvidia-smi --query-compute-apps` was checked explicitly and showed **zero** GPU memory
attributed to the generator process - only the unrelated, concurrently-running `exp42` training
job (13.77GB) remained. (Note: the script's own "VRAM after unload" print reflects the SHARED
GPU's total free memory, not a per-process figure, and dipped due to `exp42`'s own fluctuating
usage during the test window - `nvidia-smi --query-compute-apps` is the correct, unambiguous
per-process check, and was used to confirm no leak.) No duplicate generator instances were ever
loaded simultaneously.

### 9-10. Smoke-test data and output validation

Three hand-written prompts only (short factual, longer explanatory, structured list) - explicitly
NOT RAGTruth, NOT TruthfulQA, NOT FEVER, NOT HaluEval, no gold evidence, no project
training/dev/eval dataset. Engineering validation only; factual correctness was NOT scored as a
research metric (though all three outputs were, incidentally, factually sound - not claimed as a
result). Validated per-prompt: non-empty string output, no prompt-text duplication in the
generated text, `generated_token_count > 0`, deterministic repeatability (byte-identical across
two runs of the same prompt), all metadata JSON-serializable. All checks passed - `test_result:
"PASS"`.

### 11. Configuration (`configs/generator_baseline.yaml`)

`model.name`/`dtype`/`device` MIRROR `configs/models.yaml`'s `generator.primary` section exactly
(kept consistent by `tests/test_generation.py::TestConfigLoads::
test_generator_baseline_model_identity_matches_models_yaml` - fails loudly on drift rather than
silently diverging). `generation.max_new_tokens=256`, `do_sample=false`, `temperature=null`,
`top_p=null`, `seed=42`. `chat_template.enable_thinking=false`,
`fallback_on_unsupported_kwarg=true`. No unexplained hyperparameters - every field is used and
documented in the config's own comments.

### 12. Tests (24 new, `tests/test_generation.py`)

RAGTruth/TruthfulQA/FEVER/HaluEval AST-based import exclusion (extended the established pattern to
cover all four dataset names, not just RAGTruth/TruthfulQA); structural no-dataset-label-access
checks on `generate()`/`load_generator()` via `inspect.signature`; config parsing and the
models.yaml-consistency guard; chat-input construction (user-only, system+user, `enable_thinking`
fallback triggered/not-triggered); `_build_generation_kwargs` (`do_sample` True/False x
temperature/top_p presence); output type/structure, prompt-exclusion, deterministic
repeatability, generation-config capture, JSON-serializability, and system-instruction handling
via a fake model/tokenizer harness (no GPU needed for these); one guarded real-Qwen3-8B test class
(model identity, param count, dtype, and a real deterministic-repeatability generation check) -
skipped automatically if CUDA/VRAM/the model are unavailable, matching the established
`TestLoadRealVerifierCheckpoint` pattern from Step 15.

### 13. Full test-suite result

307/307 passed, run twice (once before the GPU smoke test, once after) - 283 pre-existing + 24 new.
No existing test was weakened or skipped to make Step 17 pass.

### 14. Scientific-boundary verification

Explicitly confirmed NOT done in this step: no RAGTruth test data accessed, no TruthfulQA
evaluation, no verifier/reranker/retrieval-corpus/FAISS-index weights or code changed, no
decision-policy changes, no training/dev dataset changes, no threshold tuning, no correction logic,
no fine-tuning/LoRA/PEFT/quantization of Qwen3-8B. The generator module is demonstrably
dataset-agnostic (Section 12's AST/signature checks).

**Preserved, unchanged:** all of Steps 9-16's verifier/retrieval/reranker/decision-policy code,
checkpoints, and configs were not touched. RAGTruth test and TruthfulQA remain completely
untouched.

**Cleanup:** removed the leftover `step17_preflight.py` (from an earlier interrupted preflight
check) and all `step17_*.sh` temp remote scripts; no duplicate model processes or leaked GPU
memory (Section 8); nothing committed to git.

**Git status:** still no commits — new files (`src/claimguard/generation/qwen3.py`,
`src/claimguard/generation/types.py`, `configs/generator_baseline.yaml`,
`scripts/generator_smoke_test.py`, `tests/test_generation.py`,
`data/processed/generation/generator_smoke_results.json`), modified files
(`src/claimguard/generation/__init__.py`, `src/claimguard/config.py`, `RESEARCH.md`) show as
untracked/modified.

---

## Step 18 — ClaimGuard correction/regeneration loop (2026-09-03)

**What:** Read `PROJECT_PLAN.md`, `PROJECT_REPORT.md`, `RESEARCH.md`, `CLAIMGUARD_MODEL_SELECTION.md`,
`README.md`, `configs/models.yaml`, `configs/generator_baseline.yaml`, Step 16's decision-policy
config/artifacts, and `src/claimguard/{generation,retrieval,reranking,verifier,decision}/` before
writing any code. Found and reused the existing `src/claimguard/correction/` placeholder package
(created in early scaffolding, docstring: `"not yet implemented — Step 12: correction"`).
Implemented `correction.py`/`types.py`, a new `configs/correction_baseline.yaml`, 25 new tests, a
real-GPU smoke test loading all four models simultaneously, and ran the full suite twice
(332/332 both times). Did **not** proceed to Step 19, did **not** evaluate RAGTruth/TruthfulQA, did
**not** fine-tune Qwen3/DeBERTa, did **not** touch the reranker/retrieval corpus/FAISS index, did
**not** retune the Step 16 decision policy.

**Why:** Steps 13-16 built and validated the evidence side (retrieval, reranking, binary verifier,
their integration, and a calibration/decision-policy layer); Step 17 built the generator. Step 18
is the FIRST step that connects them into a single bounded pipeline - the actual "ClaimGuard" loop
the whole project has been building toward, still strictly an engineering/integration validation,
not a research performance claim.

### 1. Existing architecture inspected

Confirmed via preflight: 307/307 tests passing (Step 17's baseline), Step 16's
`calibration_results.json`/`decision_policy_results.json` present and untouched,
`src/claimguard/correction/__init__.py` already existed as a docstring-only placeholder (updated,
not replaced) - the natural, already-established home for this work, used in preference to
creating a new `pipeline/` package. Public interfaces reused UNCHANGED: `Retriever.retrieve`,
`reranker.rerank`, `verify.verify_candidates`, `decision.policy.EvidenceDecisionPolicy`,
`generation.qwen3.generate` - no existing API was modified.

### 2. Correction-loop architecture

```
user query -> Qwen3-8B -> candidate answer -> retrieve -> rerank -> verify
    -> Step 16 decision policy ->
        ACCEPT  -> return the answer
        CORRECT -> evidence-grounded regeneration, re-verify (bounded)
        ABSTAIN -> return explicit abstention
```

Implemented as `run_correction_loop()` in `src/claimguard/correction/correction.py`. A plain
bounded `while True` loop with explicit `return` statements at every termination point (ACCEPT,
ABSTAIN, max-attempts, or any failure) - never an unbounded loop, never a hidden retry path that
bypasses the attempt limit. All model objects (retriever, reranker, verifier, generator) are
passed in ALREADY LOADED by the caller; the function never loads a model itself, so it can never
create a duplicate resident GPU copy.

### 3. ACCEPT/CORRECT/ABSTAIN semantics

Step 16's `EvidenceDecisionPolicy` still outputs SUPPORTED/CONTRADICTED/ABSTAIN, completely
unchanged. `STEP16_TO_STEP18_DECISION` is a pure integration-layer relabeling:
`SUPPORTED->ACCEPT`, `CONTRADICTED->CORRECT`, `ABSTAIN->ABSTAIN`. No new scoring formula was
invented; Step 16's actual thresholds/margin logic/tie-breaking are used exactly as-is. Every
`AttemptRecord` preserves BOTH the raw `step16_decision` and the mapped `mapped_decision`, so the
relabeling is always auditable, never opaque.

- **ACCEPT**: the decision policy considers the current candidate sufficiently supported by the
  freshly retrieved/reranked/verified evidence.
- **CORRECT**: the evidence contradicts the candidate; another evidence-grounded generation
  attempt is permitted (if budget remains).
- **ABSTAIN**: neither sufficiently supported nor contradicted (Step 16's confidence bar not met),
  OR the correction budget is exhausted while still CORRECT. **A budget-exhausted CORRECT is
  ALWAYS converted to a terminal ABSTAIN, never silently treated as ACCEPT** - `_finish("ABSTAIN",
  candidate_answer, "max_attempts_reached", True)` is the only return path for that case.

### 4. Maximum correction attempts

`max_correction_attempts: 2` (`configs/correction_baseline.yaml`) - a conservative engineering
baseline, NOT tuned against RAGTruth/TruthfulQA/any evaluation data (Step 16 did not specify a
different value, so this is a fresh, explicitly-labeled engineering choice per the instructions).
Bounds the loop to at most `1 + max_correction_attempts = 3` total generations. Verified by a
dedicated test (`TestMaxAttempts`) with a fixture that always returns CORRECT: the loop performs
exactly 3 generations (never a 4th) and terminates with `final_decision=ABSTAIN`,
`termination_reason="max_attempts_reached"`.

### 5. Evidence-grounded correction prompt

`build_correction_prompt(original_query, previous_answer, evidence_texts)` constructs a prompt
containing the original question, the previous answer, and a bulleted block of the ACTUAL
evidence text the pipeline retrieved for that candidate (never fabricated, never gold evidence) -
then explicitly instructs the generator to: answer the original question directly; use only the
supplied evidence; correct unsupported portions while preserving supported ones; never invent
facts; and return ONLY the final answer (no reasoning trace requested or stored). Sent via
`qwen3.generate()` with a fixed `CORRECTION_SYSTEM_INSTRUCTION` reinforcing the same constraints -
`enable_thinking` stays `False` (Step 17's default), so no hidden reasoning is ever generated or
exposed.

### 6. Retrieval/reranking/verifier re-execution behavior

Every attempt - including every correction - runs a COMPLETELY FRESH
`retriever.retrieve() -> reranker.rerank() -> verify.verify_candidates()` sequence over the
CURRENT candidate answer text (never the raw user question, never the previous attempt's cached
evidence). These three primitives are called individually (not via Step 15's `run_pipeline`
wrapper) specifically so per-stage latency can be recorded separately, as required - this reuses
the exact same underlying functions, it does not reimplement their logic. `TestCorrectPath`
verifies this directly: after a correction, `retriever.call_count` increments, the new evidence's
`corpus_id`s differ from the previous attempt's, and the retrieval query recorded is the
corrected candidate text, never the original question.

### 7. Provenance handling

Every `EvidenceItem` in every `AttemptRecord.evidence` list retains `corpus_id`, `page_id`,
`sentence_id`, `original_rank` (FAISS position), `reranked_rank`, and both verifier probabilities -
the full chain back to the real corpus record. Nothing is fabricated; `TestProvenanceAndSerialization`
checks these fields match the raw candidate dicts exactly.

### 8. State/result structure (`src/claimguard/correction/types.py`)

- `EvidenceItem`: one piece of real, provenance-complete evidence.
- `AttemptLatency`: generation/retrieval/reranking/verification seconds per attempt (+
  `total_seconds` property).
- `AttemptRecord`: attempt number, candidate answer, full evidence list, selected evidence, raw
  Step 16 decision + confidence + reason, mapped ACCEPT/CORRECT/ABSTAIN, per-attempt latency.
- `FailureInfo`: stage, error type, message - structured, never a bare exception swallowed silently.
- `CorrectionResult`: original query, full attempt list, final decision (ACCEPT/ABSTAIN only -
  CORRECT is never a terminal state), final answer (the LAST candidate text even on ABSTAIN, so
  nothing is silently discarded - callers MUST check `final_decision`), termination reason,
  whether correction occurred, total attempts, the configured max, `status`
  (`"completed"`/`"failed"`), failure info, total latency. All dataclasses have `to_dict()` and
  round-trip through `json.dumps`/`json.loads` cleanly (tested).

### 9. Failure handling

Every stage is wrapped and returns a structured `CorrectionResult(status="failed", failure=...)`
rather than letting an exception propagate or silently swallowing it: initial-generation failure,
empty generated answer, retrieval/reranking/verification failure (caught as one combined stage
since these three run consecutively inside one helper), decision-policy failure (including an
"invalid decision" case - tested by mocking the policy to return a bogus label not in
`STEP16_TO_STEP18_DECISION`, confirming the `KeyError` is caught and reported, not raised to the
caller), and correction-generation failure. `final_decision` is ALWAYS `"ABSTAIN"` on any failure
path - never `"ACCEPT"` - directly enforcing "never convert correction-failed into answer-accepted."

### 10. Real Qwen3 smoke-test result (engineering validation only)

Three hand-written questions (not RAGTruth/TruthfulQA/FEVER/HaluEval, no gold evidence): all three
reached **ACCEPT on the first attempt** - CORRECT/ABSTAIN were not naturally triggered by these
simple, well-covered factual questions against the FEVER-wiki-derived retrieval corpus. This is an
honest observation, not a forced or cherry-picked result: the smoke fixture was written before
running it, and no query was discarded or replaced based on its outcome. CORRECT-path,
ABSTAIN-path, and max-attempts-termination behavior were validated instead via 10 dedicated
synthetic/mocked fixtures in `tests/test_correction.py` (`TestCorrectPath`, `TestAbstainPath`,
`TestMaxAttempts`), per the instructions' explicit allowance to mock expensive models for those
cases. `test_result: "PASS"` in the saved artifact.

### 11. GPU memory/resource result

All four models (BGE embedder, BGE reranker, DeBERTa verifier, Qwen3-8B) loaded ONCE,
simultaneously, in 17.8s (mostly cache hits). Peak allocated VRAM across the entire smoke test
(model loads + 3 queries + 2 determinism-check repeats): **17.81GB** - comfortably within
`CLAIMGUARD_MODEL_SELECTION.md`'s ~23-25GB estimate for the full inference stack, and well below
the 33.48GB that was free at preflight. Allocated-VRAM snapshots taken after each of the 3 smoke
queries were IDENTICAL (17.74GB each) - **zero measured GPU memory growth** across repeated
correction-loop calls, confirming no accidental duplicate model instances or per-call leaks.

### 12. Latency observations (engineering only, not a performance claim)

Per-query wall-clock latency: 9.35s / 1.97s / 3.49s for the three smoke questions (the first
included one-time lazy embedding-model load inside `Retriever._ensure_model_loaded`). All three
completed in a single attempt (no correction rounds), so per-stage
generation/retrieval/reranking/verification latency breakdown - while recorded per attempt in
every `AttemptRecord.latency` - was only exercised for the "attempt 0" path in this particular real
run; the CORRECT-path re-verification latency was exercised only in the mocked test fixtures.

### 13. Artifacts created

- `src/claimguard/correction/{correction.py, types.py}` (+ updated `__init__.py`).
- `configs/correction_baseline.yaml` (`max_correction_attempts`, `pipeline`, `decision_policy`,
  `generation`, `verifier` sections - values consistency-tested against
  `configs/decision_policy.yaml` and `configs/generator_baseline.yaml`, never independently
  duplicated with new values).
- `scripts/correction_smoke_test.py`.
- `data/processed/correction/correction_smoke_results.json` (validated by read-back: `test_result`,
  `max_correction_attempts`, 3 query results, deterministic-repeatability result, peak/growth GPU
  memory, model identifiers all present and correct).

### 14. Tests added (25 new, `tests/test_correction.py`)

RAGTruth/TruthfulQA/FEVER/HaluEval AST-based import exclusion; structural no-gold-access checks on
`run_correction_loop`/`build_correction_prompt` via `inspect.signature`; config parsing plus
consistency guards against `decision_policy.yaml` and `generator_baseline.yaml`; ACCEPT
termination (1 attempt, no correction); ABSTAIN termination (1 attempt, no unnecessary
generation) including a genuinely-empty-evidence-set case (graceful ABSTAIN, not a failure);
CORRECT transition (exactly one regeneration, fresh retrieval confirmed via call-count and
differing `corpus_id`s, retrieval query confirmed to be the candidate answer not the original
question); hard max-attempts termination; five failure-handling cases (generator failure, empty
output, verifier failure, correction-generation failure, invalid/unrecognized policy decision);
evidence provenance preservation; JSON state serialization; deterministic repeatability across two
full runs of an identical scripted fixture; and correction-prompt content construction (with and
without evidence).

### 15. Full test-suite result

**332/332 passed**, run twice (once before the GPU smoke test, once after) - 307 pre-existing + 25
new. 2 of the pre-existing real-Qwen3-8B-guarded tests (`test_generation.TestLoadGeneratorRealCheckpoint`)
were gracefully SKIPPED during the FIRST run only, with an explicit logged reason: free VRAM at that
moment was 15.2GB, below the class's own 20GB safety threshold, because the shared GPU's concurrent
`exp42` training job's memory usage was transiently higher than usual. This is the designed
SkipTest behavior working correctly, not a regression - free VRAM was rechecked immediately after
(34.6GB) and the real-model correction-loop smoke test itself ran and passed cleanly with margin to
spare. No existing test was weakened or skipped intentionally to make Step 18 pass.

### 16. Scientific-boundary verification

Confirmed NOT done in this step: no RAGTruth test/labels/spans accessed, no TruthfulQA evaluation,
no FEVER gold evidence used during inference, no HaluEval evaluation labels accessed, no Qwen3
fine-tuning, no verifier fine-tuning, no reranker tuning, no retrieval-corpus/FAISS-index changes,
no decision-policy tuning on evaluation data, no learned correction classifier, no learned fusion,
no fabricated/reinterpreted FEVER neutral examples, no raw dataset modifications, no new research
split. The only newly implemented behavior is the bounded correction/regeneration orchestration
itself (Sections 2-9 above).

**Preserved, unchanged:** Qwen3-8B (Step 17), the DeBERTa verifier (Step 12), the BGE reranker
(Step 14), and the FAISS retrieval corpus/index (Step 13) are all FROZEN - loaded read-only and
never modified. Step 16's decision policy is used exactly as configured, never retuned. RAGTruth
test and TruthfulQA remain completely untouched.

**Cleanup:** removed all `step18_*.sh` temp remote scripts and the temporary artifact-validation
script; no duplicate model processes or leaked GPU memory (Section 11); nothing committed to git.

**Git status:** still no commits — new files (`src/claimguard/correction/correction.py`,
`src/claimguard/correction/types.py`, `configs/correction_baseline.yaml`,
`scripts/correction_smoke_test.py`, `tests/test_correction.py`,
`data/processed/correction/correction_smoke_results.json`), modified files
(`src/claimguard/correction/__init__.py`, `src/claimguard/config.py`, `RESEARCH.md`) show as
untracked/modified.

---

## Step 19 — RAGTruth end-to-end evaluation (2026-09-04/05)

**What:** Read `PROJECT_PLAN.md`, `PROJECT_REPORT.md`, `RESEARCH.md`, `CLAIMGUARD_MODEL_SELECTION.md`,
`README.md`, `data/processed/dataset_manifest.json`, `src/claimguard/datasets/ragtruth.py` (full
source, not just its docstring), Step 16/17/18 configs, and the retrieval/reranking/verifier/
correction code, before writing any evaluation code. Empirically verified (not assumed) the
reserved RAGTruth test-set dimensions against the live dataset module. Extended
`run_correction_loop` with one backward-compatible parameter (`initial_candidate`). Built
`src/claimguard/evaluation/ragtruth_eval.py` (pure metric/label-mapping logic),
`scripts/evaluate_ragtruth.py` (inference orchestration) and `scripts/analyze_ragtruth_results.py`
(aggregation, separated from the expensive GPU pass). Added 36 new tests. Ran a 20-response pilot,
then - after an explicit, disclosed timing/scope check-in with the user given a measured ~24-hour
extrapolated runtime - ran the complete FROZEN system against all 2,700 reserved RAGTruth test
responses across 3 conditions (8,100 total correction-loop calls), completing in 15.27 hours with
zero inference failures. Did **not** change any frozen parameter, did **not** tune anything based
on results, did **not** touch TruthfulQA, did **not** proceed to Step 20.

**Why:** Steps 13-18 built and validated every ClaimGuard component in isolation or on small
engineering smoke tests. RAGTruth is the project's designated primary end-to-end evaluation
dataset (`CLAIMGUARD_MODEL_SELECTION.md`) - this is the first time the complete, frozen system is
measured against a real, independently-annotated hallucination benchmark, on a reserved split it
has never touched.

### 1. Environment

Python 3.12.14 (`claimguard` conda env), `torch==2.11.0+cu128`, `transformers==5.16.1`, NVIDIA RTX
PRO 5000 Blackwell. 368/368 tests passing (332 pre-existing + 36 new) before the evaluation run.

### 2. RAGTruth test-set verification (empirical, not assumed)

Loaded `claimguard.datasets.ragtruth.load_normalized()` + `get_eval_set()` and counted directly:
**2,700 test responses**, **450 distinct source items**, **every one of the 450 sources has
exactly 6 responses** (one per model: gpt-4-0613, gpt-3.5-turbo-0613, mistral-7B-instruct,
llama-2-7b-chat, llama-2-13b-chat, llama-2-70b-chat). `assert_no_train_test_leakage()` confirms
source-level train/test separation holds (0 leaking source_ids). Gold label semantics read
directly from `join_records()`'s source: `has_hallucination = bool(labels)` - a response is
gold-positive iff it has at least one RAGTruth-annotated hallucination span. This is RAGTruth's
own independent label, never reinterpreted as FEVER entailment/contradiction anywhere in this
step. Gold prevalence in the test set: **34.9% hallucinated** (943/2,700).

### 3. Frozen configuration

Nothing changed from Steps 12-18: `Qwen/Qwen3-8B` (bf16, `do_sample=False`, `seed=42`,
`max_new_tokens=256`), the DeBERTa binary verifier checkpoint, `BAAI/bge-large-en-v1.5` embedder +
FAISS index (unchanged, unrebuilt), `BAAI/bge-reranker-large`, Step 16's decision policy
(`entailment_threshold=contradiction_threshold=0.5`), Step 18's correction prompt, and
`max_correction_attempts=2`. All read from `configs/correction_baseline.yaml`/
`configs/decision_policy.yaml`/`configs/generator_baseline.yaml` exactly as Steps 16-18 left them -
no new independent config values were introduced for this step.

### 4. Evaluation conditions

- **B (generated):** fresh Qwen3-8B generation from RAGTruth's own `prompt` field, then the full
  bounded correction loop.
- **C (verify-only):** the ORIGINAL RAGTruth response as the starting candidate
  (`initial_candidate`), a single verify pass (`max_correction_attempts=0`) - isolates raw
  detection with no regeneration.
- **D (original + full loop):** the original response as the starting candidate, full bounded
  correction loop - isolates whether correction improves an already-generated answer.
- **A**, as literally specified ("original response, run verification/decision/correction as
  appropriate"), is procedurally IDENTICAL to D - both mean "run the original response through
  the full correction loop." Rather than silently merging them or wastefully duplicating ~2,700
  identical GPU calls, A's numbers are reported as equal to D's, with this equivalence stated
  explicitly here and in the saved manifest.

**Interface note (compatibility-preserving extension, not a Step 18 API change):**
`run_correction_loop` gained one optional parameter, `initial_candidate` (default `None` preserves
every Step 18 behavior/test exactly - reconfirmed by the full regression suite). When provided,
attempt 0 skips generation and starts from the supplied text; combined with
`max_correction_attempts=0` it yields Condition C's single verify-only pass. No change to the
decision rule, the correction prompt, or any threshold.

### 5. Label semantics

`has_hallucination` (bool) is the sole gold signal used; RAGTruth's finer-grained `label_type`
(Evident Conflict / Evident Baseless Info / Subtle Baseless Info / Subtle Conflict) is preserved in
the joined records but not used for a separate metric in this step (out of scope - response-level
binary detection is the primary ask). Predicted label derivation is defined in
`claimguard.evaluation.ragtruth_eval`: `attempt0_detection_prediction` (ACCEPT->not_hallucinated,
CORRECT/ABSTAIN->hallucinated - the correction-uncontaminated PRIMARY signal) and
`final_outcome_prediction` (post-correction end state, ACCEPT->not_hallucinated,
ABSTAIN->hallucinated; CORRECT is never terminal per Step 18's guarantee).

### 6. Primary response-level metrics (Condition B, n=2,700)

| Metric | attempt-0 (pre-correction) | final (post-correction) |
|---|---|---|
| Accuracy | 0.5589 | 0.5804 |
| Hallucinated precision/recall/F1 | 0.4297 / 0.8038 / 0.5600 | 0.4426 / 0.7773 / 0.5641 |
| Not-hallucinated precision/recall/F1 | 0.8024 / 0.4274 / 0.5577 | 0.7989 / 0.4747 / 0.5955 |
| Macro F1 | 0.5589 | 0.5798 |
| Confusion (TP/FP/FN/TN) | 758/1006/185/751 | 733/923/210/834 |

**The trivial always-predict-"not-hallucinated" baseline scores 65.1% accuracy** (the gold
not-hallucinated rate) - ClaimGuard's 55.9%/58.0% are BOTH below this trivial baseline. Reported
factually, not minimized: this is a poor detection result, not a successful one, on this dataset
with this frozen configuration.

Conditions C and D share an IDENTICAL attempt-0 confusion matrix (807/1248/136/509, accuracy
0.4874, macro F1 0.4811) - a confirmed-correct consistency check (C and D differ only AFTER
attempt 0, since both start from the same original response). D's final-outcome metrics (accuracy
0.5511, macro F1 0.5507) show the correction loop measurably helps relative to verify-only, but
not enough to reach B's numbers or the trivial baseline.

### 7. ACCEPT/CORRECT/ABSTAIN statistics

| | B attempt-0 | B final | C attempt-0/final | D attempt-0 | D final |
|---|---|---|---|---|---|
| ACCEPT rate | 34.67% | 38.67% | 23.89% | 23.89% | 37.89% |
| CORRECT rate | 65.33% | - | 76.11% | 76.11% | - |
| ABSTAIN rate | **0.00%** | 61.33% | **0.00%** / 76.11% | **0.00%** | 62.11% |

**ABSTAIN never fires at attempt 0 in ANY condition, across all 2,700 responses.** Every attempt-0
decision is ACCEPT or CORRECT. This directly generalizes Step 16's own finding (verifier confidence
clusters at the extremes; the "neither confident enough" zone was already rare in-domain on FEVER)
out-of-domain: on RAGTruth, the verifier is never uncertain, just frequently confidently wrong
against irrelevant evidence (Section 10 below).

### 8. Correction effectiveness

`correction_success_rate_among_attempted`: **6.12%** (B) / **18.39%** (D). `max_attempts_reached_rate`:
**61.33%** (B) / **62.11%** (D) - the large majority of correction attempts exhaust the 2-attempt
budget without reaching ACCEPT. Outcome categories (B, n=2,700):

| Category | Count | % |
|---|---|---|
| hallucinated_still_flagged | 733 | 27.1% |
| not_hallucinated_preserved | 834 | 30.9% |
| hallucinated_corrected_successfully | 25 | 0.9% |
| not_hallucinated_degraded | 923 | 34.2% |
| hallucinated_missed | 185 | 6.9% |

`not_hallucinated_degraded` (923, 34.2%) - a genuinely fine original answer ending in ABSTAIN - is
the single largest category after `not_hallucinated_preserved`. `unnecessary_correction_count`
(1,006 for B) equals B's false-positive count exactly, consistent with the 0% ABSTAIN-at-attempt-0
finding: every false positive in this dataset is a CORRECT verdict, never an ABSTAIN. Only 25
responses (0.9%) were successfully corrected from hallucinated to accepted - correction does not
meaningfully fix the underlying evidence-mismatch problem (Section 10).

### 9. Span-level metrics - not computed (documented limitation)

Step 16's decision policy operates at the whole-candidate-answer level; it has no mechanism to
attribute its verdict to a specific substring of the answer. Computing span precision/recall/F1
would require inventing an arbitrary span-alignment heuristic ClaimGuard's actual architecture does
not support - reported as a limitation per the explicit instruction, not fabricated.

### 10. Task-type breakdown (Condition B)

| Task | n | Gold prevalence | Accuracy | Macro F1 | Not-hallucinated recall |
|---|---|---|---|---|---|
| Data2txt | 900 | 64.3% | 63.7% | **0.398** | **0.9%** |
| QA | 900 | 17.8% | 66.7% | **0.496** | 75.9% |
| Summary | 900 | 22.7% | 37.3% | **0.372** | 26.7% |

Data2txt (Yelp-style business listings) and Summary (news articles) have essentially zero
topical overlap with the FEVER-Wikipedia retrieval corpus - Data2txt's not-hallucinated recall of
0.9% means ClaimGuard flags 98.7% of Data2txt responses as needing correction regardless of actual
quality. Summary's 37.3% accuracy is BELOW that task's own trivial 77.3%-accuracy
always-predict-not-hallucinated baseline. QA performs best (still weak) - plausibly because some
QA content brushes against the general-knowledge territory FEVER-wiki covers.

### 11. Source-model breakdown (Condition B)

| Model | n | Gold prevalence | Macro F1 |
|---|---|---|---|
| gpt-4-0613 | 450 | 9.3% | 0.392 |
| gpt-3.5-turbo-0613 | 450 | 10.2% | 0.386 |
| llama-2-13b-chat | 450 | 46.0% | 0.629 |
| llama-2-70b-chat | 450 | 38.0% | 0.588 |
| llama-2-7b-chat | 450 | 50.2% | 0.599 |
| mistral-7B-instruct | 450 | 55.8% | 0.653 |

The strongest, lowest-hallucination-rate models (gpt-4, gpt-3.5) get ClaimGuard's WORST scores -
gpt-4's hallucinated-recall is 95.2% but precision only 13.6% (massive over-flagging of genuinely
good responses). The weaker, higher-hallucination-rate open models score better, not because
ClaimGuard discriminates their errors more accurately, but because its systematic
over-flagging bias happens to numerically align with truth more often when there is genuinely more
to flag. Model metadata was fully consistent (6 clean model names, 450 responses each, no parsing
issues).

### 12. Original vs. final answer analysis

Preserved per response, per condition: `total_attempts`, `correction_occurred`,
`termination_reason`, `final_answer_preview` (150-char truncation - full text not duplicated to
avoid unnecessarily reproducing copyrighted source material), `gold_has_hallucination` (evaluation-
layer only). Gold outcomes are never fed back into inference - confirmed structurally (Section 17).

### 13. Error analysis (Condition B, deterministic selection - confidence-sorted or response-id-
sorted, never manually cherry-picked)

| Category | Count (of 2,700) |
|---|---|
| False positives (gold=no, predicted=hallucinated) | 1,006 |
| False negatives (gold=yes, predicted=not) | 185 |
| Harmful correction (`not_hallucinated_degraded`) | 923 |
| Unnecessary correction (attempted on gold=no) | 1,006 |
| Max-attempts-reached | 1,656 |
| Retrieval failures (0 evidence) | 0 |
| High-confidence wrong (confidence >= 0.9) | 1,010 |
| Inference failures | 0 |

The top-5-by-confidence examples in each category were saved to `ragtruth_error_analysis.json`
with short answer previews and full provenance (response_id/task_type/model/decision/confidence) -
sufficient to trace back to the raw data without reproducing large passages. Generator-degradation
(text-quality decline, as opposed to verifier-judged support decline) was not independently
measured - it would require a dedicated quality-comparison mechanism this evaluation does not
implement; `not_hallucinated_degraded` is the closest available proxy.

### 14. Retrieval diagnostics

0 zero-evidence responses (always retrieves exactly `rerank_top_n`=5 candidates); mean evidence
count 5.0. This confirms retrieval is functioning correctly as designed - the problem is not
retrieval FAILING, it's that the FEVER-Wikipedia corpus has no genuinely relevant content for most
RAGTruth source material, so retrieval succeeds at returning topically-adjacent-but-wrong evidence
every time (Section 10's primary interpretive lens).

### 15. Latency / resource results

| Condition | Mean | Median | P95 | Total |
|---|---|---|---|---|
| B (generated) | 12.56s | 12.04s | 23.98s | 33,924s (9.4h) |
| C (verify-only) | 0.26s | 0.26s | 0.30s | 689s (0.19h) |
| D (original+loop) | 7.54s | 8.15s | 17.97s | 20,359s (5.7h) |

Total: 54,973s = **15.27 hours** for the complete 2,700 x 3-condition run (faster than the ~24h
extrapolated from the 20-response pilot, whose sorted-by-response_id slice happened to include
longer-than-average prompts). Peak GPU memory 18.39GB - essentially unchanged from Step 18's
17.81GB smoke test, confirming no memory growth across a 15+ hour run. Latency is NOT compared
against Step 15's ~159ms/query figure without accounting for the fundamentally different workload
(Step 15 had no generation at all; RAGTruth prompts/contexts are far longer than Step 15's FEVER
claim texts).

### 16. Reproducibility check

A 10-response deterministic subset (first 10 test responses by `response_id`, stable-sorted) was
run through Condition B TWICE, independently, using the already-loaded models: identical
`final_decision`/`final_answer`/`total_attempts` across both runs. Confirmed on real RAGTruth data,
not just synthetic fixtures (which Step 18 already covered).

### 17. Leakage / isolation audit

Verified explicitly: RAGTruth test not used in training/tuning/threshold-selection; TruthfulQA
untouched; no gold RAGTruth labels/spans entered inference; no FEVER gold evidence supplied at
inference; no test labels influenced generation or correction. Structurally enforced by (a)
`claimguard.evaluation`'s AST-checked absence of any inference-path import, and (b) a
mock-based behavioral test (`TestGoldLabelIsolationInOrchestration`) that runs the REAL
`run_one_response` orchestration function with a sentinel gold-label value and confirms it never
appears in any of the 3 `run_correction_loop` calls' positional or keyword arguments - not just a
signature check, an actual behavioral confirmation on the real evaluation code path.

### 18. Artifacts created

`data/processed/evaluation/ragtruth/{ragtruth_eval_results.json, ragtruth_response_results.jsonl
(2,700 lines), ragtruth_error_analysis.json, ragtruth_eval_manifest.json}`. The JSONL stores IDs,
decisions, confidences, latencies, and 150-char answer previews per condition per response -
enough to reproduce every aggregate metric above without rerunning inference, without duplicating
full RAGTruth response/prompt/evidence text.

### 19. Tests added (36 new, `tests/test_ragtruth_evaluation.py`)

Pure-function tests for `ragtruth_eval`'s detection-prediction mapping, confusion-matrix arithmetic
(including the "recall undefined when zero actual positives" edge case - correctly returns `None`,
never a fabricated 0.0), full detection-metrics computation, decision-rate breakdown, all 5
correction-outcome-category branches, and the unnecessary-correction flag; a structural AST guard
confirming `claimguard.evaluation` imports no inference-path module; the behavioral gold-isolation
test described in Section 17; 3 tests for the new `initial_candidate` extension (None preserves
Step 18 generation behavior, a supplied candidate skips generation, `max_correction_attempts=0`
yields a single verify-only pass with the budget-exhausted-CORRECT-becomes-ABSTAIN guarantee
intact); and 6 tests verifying the real RAGTruth test-set dimensions (2,700/450/6, no leakage, raw
files unmutated by loading - confirmed via SHA-256 hash before/after, and that the dataset module
exposes no write-capable function at all).

### 20. Full test-suite result

**368/368 passed**, run twice - 332 pre-existing + 36 new. One genuine test-authoring bug was
found and fixed during this step's own development (a test asserted F1=0.0 for a case where recall
was actually mathematically undefined given zero actual positives in the synthetic gold labels -
the METRIC CODE was correct; the TEST's own expectation was wrong. Fixed by correcting the test
to use a case where recall is well-defined-but-zero, and adding a new test explicitly confirming
the undefined-recall-returns-None behavior is intentional).

### 21. Scientific-boundary audit

Confirmed: no configuration parameter changed after seeing pilot or full results; no threshold/
prompt/max-attempts/generation-parameter tuning; RAGTruth test used only for this evaluation, never
for training/tuning; TruthfulQA completely untouched; no gold labels/spans reached any inference
call (Section 17). The ~24-hour-vs-~15-hour timing discrepancy between the pilot extrapolation and
the actual full run was NOT used as a reason to change anything about the frozen system - it is
purely a compute-time observation, reported honestly in Section 15.

**Scope/timing disclosure:** the 20-response pilot measured 32.06s/response and extrapolated to
~24 hours for the full run. This was surfaced explicitly to the user (not silently decided) before
committing to the multi-hour background job; the user confirmed proceeding with the full 2,700-
response evaluation as specified, which then completed in 15.27 hours (faster than extrapolated,
plausibly because the response-id-sorted pilot slice happened to include disproportionately long
prompts).

**Preserved, unchanged:** Qwen3-8B, DeBERTa verifier, BGE embedder, FAISS index, BGE reranker, and
Step 16's decision policy were all loaded read-only and never modified. `data/raw/ragtruth/*` was
never written to (confirmed by SHA-256 hash before/after loading - Section 19).

**Cleanup:** removed all `step19_*.sh` temp remote scripts and the pilot-only intermediate
artifacts (overwritten by the full run's identically-named outputs); no duplicate model processes;
GPU returned to baseline (only the unrelated concurrent job's memory remained) after the process
exited; nothing committed to git.

**Git status:** still no commits — new files (`src/claimguard/evaluation/ragtruth_eval.py`,
`tests/test_ragtruth_evaluation.py`, `scripts/evaluate_ragtruth.py`,
`scripts/analyze_ragtruth_results.py`, `data/processed/evaluation/ragtruth/*`), modified files
(`src/claimguard/correction/correction.py`, `src/claimguard/evaluation/__init__.py`, `RESEARCH.md`)
show as untracked/modified.

---

## Step 20 — TruthfulQA evaluation (2026-09-05)

**What:** Read `PROJECT_PLAN.md`, `PROJECT_REPORT.md`, `RESEARCH.md`, `CLAIMGUARD_MODEL_SELECTION.md`,
`README.md`, `src/claimguard/datasets/truthfulqa.py` (full source), Step 16/17/18/19 configs and
code, before writing anything. Confirmed empirically (790 questions, 37 categories, no split,
0 duplicates). Added ONE new function to the frozen generator wrapper
(`score_choice_log_likelihood` - a teacher-forced forward pass, not generation, not a weight
change), a new `src/claimguard/evaluation/truthfulqa_eval.py` (pure MC1/MC2/MC0 arithmetic,
re-exporting Step 19's dataset-agnostic `decision_rate_breakdown` rather than duplicating it), 27
new tests, and ran a 20-question pilot before the complete 790-question evaluation (completed in
1.41 hours). Did **not** change any frozen parameter, did **not** tune anything based on results,
did **not** touch RAGTruth, did **not** proceed to Step 21.

**Why:** RAGTruth (Step 19) is the project's primary end-to-end effectiveness benchmark; TruthfulQA
was explicitly reserved as the SECOND, adversarial stress-test evaluation
(`CLAIMGUARD_MODEL_SELECTION.md`), with a standing caveat on record since Step 3 that its age/
ubiquity risks measuring generator memorization rather than pipeline effect. Step 20 runs this
final held-out evaluation of the frozen system exactly as specified.

### 1. Environment

Python 3.12.14 (`claimguard` conda env), `torch==2.11.0+cu128`, `transformers==5.16.1`, NVIDIA RTX
PRO 5000 Blackwell. 368/368 tests passing before this step's code was written.

### 2. TruthfulQA dataset integrity (empirical, not assumed)

`claimguard.datasets.truthfulqa.load_normalized()` returns exactly **790** question-level records
across exactly **37** categories (both counts read live from the module, matching Step 6D's
documented finding that the real data differs from the commonly-cited 817/38 figures). 0 duplicate
questions. No `split` field exists in any record, and the module deliberately exposes no
`get_eval_set()`/`get_training_pool()` pair (unlike `claimguard.datasets.ragtruth`) - confirmed by
`hasattr` checks, not just reading the docstring. `mc0_targets` has exactly 2 keys (1 correct, 1
incorrect) whenever present; `mc2_targets` genuinely contains multiple correct options for some
questions (confirmed, not assumed). All 790 questions were treated as the evaluation set - no
subset, no invented split.

### 3. Number of questions/categories

790 questions total, 37 categories, category sizes ranging from n=3 (Misconceptions: Topical) to
n=100 (Misconceptions) - small categories (<15) are explicitly flagged in the breakdown, not
overinterpreted.

### 4. Frozen system configuration

Identical to Step 19: `Qwen/Qwen3-8B` (bf16, `do_sample=False`, `seed=42`, `max_new_tokens=256`),
the DeBERTa binary verifier checkpoint, `BAAI/bge-large-en-v1.5` embedder + FAISS index (unrebuilt),
`BAAI/bge-reranker-large`, Step 16's decision policy (thresholds=0.5), Step 18's correction prompt,
`max_correction_attempts=2` - all read from `configs/correction_baseline.yaml` unchanged. No
discrepancy from Step 19's configuration was found (same config file, same values) - no STOP
condition triggered here.

### 5. Evaluation conditions

- **A (baseline):** Qwen3-8B's raw free-form answer, no ClaimGuard intervention - extracted as
  `run_correction_loop`'s attempt-0 `candidate_answer`.
- **B (ClaimGuard initial):** the SAME attempt-0 answer's verification/decision outcome (Step 16's
  policy applied once, no correction) - extracted from the same call's `attempts[0]` fields.
- **C (ClaimGuard final):** the loop's full post-correction outcome (`final_decision`/
  `final_answer`).

A, B, and C are all derived from a SINGLE `run_correction_loop` call per question (no redundant
generation) - efficient and exactly matches Step 18's existing interface; no new correction-loop
code was needed for TruthfulQA, unlike Step 19's `initial_candidate` extension (RAGTruth needed to
seed the loop with an existing model's answer; TruthfulQA has no such existing answer to seed with,
since Qwen3 IS the only generator here).

### 6. TruthfulQA metric definitions

Two independent tracks, kept deliberately separate:

- **MC track** (rigorous, ground-truth-backed, published-standard definitions): `score_choice_log_
  likelihood(model, tokenizer, question, choice)` computes the SUMMED log-probability of `choice`'s
  tokens conditioned on `question` via one teacher-forced forward pass (verified against a
  hand-computed expected value in `tests/test_truthfulqa_evaluation.py` using a fake uniform-logit
  model). MC1 = 1 if the single highest-log-likelihood choice among `mc1_targets` is the (exactly
  one) correct option, else 0, averaged over questions. MC0 = the identical rule applied to the
  2-choice `mc0_targets` set. MC2 = softmax-normalize log-likelihoods across ALL choices in
  `mc2_targets`, then sum the probability mass on correct (possibly multiple) choices - continuous
  in [0,1], averaged over questions. This track never invokes retrieval/reranking/verification/
  correction - it measures Qwen3's OWN parametric calibration.
- **Free-form pipeline track**: NOT a truthfulness metric. Reports `decision_rate_breakdown`
  (re-exported unchanged from `claimguard.evaluation.ragtruth_eval` - fully dataset-agnostic, not
  duplicated) over Conditions A/B/C's ACCEPT/CORRECT/ABSTAIN outcomes, plus retrieval diagnostics
  and latency. **No approved automatic judge for free-form TruthfulQA truthfulness exists in this
  project** (documented protocol gap, Section 21 below) - this track measures ClaimGuard's PROCESS
  behavior only.

### 7. Overall baseline results (Condition A)

Condition A's answer text is the SAME text scored implicitly by the free-form track's attempt-0
decision (Section 8) - no separate accuracy number exists for "baseline truthfulness" since no
approved judge exists (Section 6). Qwen3's OWN calibration is instead captured by the MC track
(Section 11): MC1 34.05%, MC2 55.05%, MC0 44.18% (n=790 for all three).

### 8. Overall ClaimGuard initial results (Condition B, attempt-0 process behavior)

ACCEPT rate **80.0%**, CORRECT rate **20.0%**, ABSTAIN rate **0.0%** (n=790). As in Steps 16 and 19,
the decision policy is never uncertain at attempt 0 - it always confidently accepts or flags for
correction.

### 9. Overall ClaimGuard final results (Condition C, post-correction process behavior)

Final ACCEPT rate **95.3%**, final ABSTAIN rate **4.7%** (n=790). Compare to Step 19's RAGTruth
final ACCEPT rate of 38.7% (Condition B) - a striking difference explained in Section 22, not
interpreted as "ClaimGuard is more truthful here."

### 10. Correction effectiveness

Of the 20.0% of questions where correction was attempted (158/790), **76.6% reached ACCEPT within
the 2-attempt budget** - dramatically higher than RAGTruth's 6.1-18.4% correction-success rate.
Max-attempts-reached: **4.7%** (37/790) vs RAGTruth's 61.3%. **This is explicitly NOT interpreted as
"ClaimGuard successfully corrected untruthful answers more often on TruthfulQA"** - no truthfulness
ground truth was computed for the free-form track (Section 6), so "correction reached ACCEPT" means
only that the decision policy's own criteria were satisfied on retry, not that the answer became
more accurate. The more plausible explanation (Section 22): TruthfulQA's short, well-known-topic
questions land nearer SOME plausible Wikipedia sentence in the frozen FEVER corpus than RAGTruth's
long domain-specific documents do, regardless of factual correctness.

### 11. MC0/MC1/MC2 results

| Metric | Accuracy | n |
|---|---|---|
| MC1 | 34.05% | 790 |
| MC2 | 55.05% | 790 |
| MC0 | 44.18% | 790 |

A modest result for Qwen3-8B's own calibration - well above trivial single-choice-among-several
random guessing, far from strong. **Standing caveat (on record since Step 3,
`CLAIMGUARD_MODEL_SELECTION.md`): TruthfulQA is an old, ubiquitous public benchmark plausibly
present in Qwen3's pretraining data - this score may partly reflect memorization, not measured or
distinguishable here.** These three numbers are kept STRICTLY SEPARATE from the free-form
pipeline's process-behavior numbers (Section 8-10) - never collapsed into one unsupported
"TruthfulQA score."

### 12. Category breakdown

37 categories, MC1 ranging from 0.0 (Confusion: Other n=8, Confusion: People n=23, Finance n=9 -
Qwen3 gets every question wrong) to 0.875 (Stereotypes n=24). Small-sample categories (<15
questions) are explicitly flagged with a `"small sample - interpret with caution"` note in the
saved artifact, not silently treated as equally reliable as larger categories. Full per-category
MC1/MC2/decision-rate table saved in `truthfulqa_eval_results.json`.

### 13. Error analysis

Deterministic selection only (confidence/response-id sorted, never manually chosen):

| Category | Count (of 790) |
|---|---|
| MC1 incorrect (Qwen3 ranked a wrong choice highest) | 521 (65.9%) |
| Max-attempts-reached | 37 (4.7%) |
| Zero-evidence retrieved | 0 |
| Inference failures | 0 |

**Categories 1-7 from the original instructions (baseline-truthful/untruthful transitions,
harmful/unnecessary correction, verifier accepted an untruthful answer) are explicitly NOT
computed** - they require a ground-truth truthfulness judgment for free-form generated text that
has no approved implementation in this project (Section 6). This gap is documented, not silently
filled with an invented heuristic - see `claimguard.evaluation.truthfulqa_eval`'s module docstring
and `truthfulqa_error_analysis.json`'s `not_computed_categories` field.

### 14. Retrieval diagnostics

0 zero-evidence questions; mean 5.0 evidence candidates/query (identical pattern to Step 19).
TruthfulQA's 37 categories (law, fiction, proverbs, paranormal, indexical errors, etc.) have no
guaranteed FEVER-Wikipedia coverage - successful retrieval is not interpreted as proof of relevant
evidence, consistent with Step 19's established corpus-mismatch finding (not re-litigated
claim-by-claim here).

### 15. Latency/resource results

Pilot (20 questions): 6.51s/question. Full run (790 questions): **5,092.5s = 1.41 hours** total
loop time (5.9-6.5s/question throughout, consistent with the pilot). Peak GPU memory **17.93GB**,
consistent with Steps 18-19 (no growth). Substantially cheaper per-question than RAGTruth's Step
19 run because (a) most questions ACCEPT on the first pipeline pass (only 20% need correction, vs
RAGTruth's 65-76%) and (b) MC scoring uses short teacher-forced forward passes, not autoregressive
generation.

### 16. Deterministic reproducibility result

A 10-question deterministic subset (first 10 by `question_id`, stable-sorted) was run through the
COMPLETE pipeline (free-form track AND MC scoring) TWICE: identical `condition_c_final_decision`,
`condition_a_baseline_answer_preview`, `condition_c_final_answer_preview`, and
`mc1_choice_log_likelihoods` across both runs. Confirmed on real TruthfulQA data.

### 17. Leakage/isolation audit

Verified explicitly and structurally: TruthfulQA not used in training/tuning/threshold-selection/
model-selection; RAGTruth untouched during Step 20; no gold TruthfulQA labels (the 0/1 values
inside mc0/mc1/mc2_targets, or correct_answers/incorrect_answers/best_answer/
best_incorrect_answer) entered any inference call; no external web retrieval used anywhere.
Enforced by (a) an AST-based structural guard (`claimguard.evaluation` imports no inference-path
module - reused from Step 19's general package-wide check), and (b) a NEW mock-based behavioral
test (`TestGoldLabelIsolationInOrchestration`) that runs the REAL `run_one_question` orchestration
function with a sentinel gold-answer string standing in for the correct answer, and confirms that
sentinel never appears in any `run_correction_loop` call's arguments, and that no raw
label-carrying dict ever reaches either `run_correction_loop` or `score_choice_log_likelihood` -
not just a signature check, an actual behavioral confirmation on the real evaluation code path.

### 18. Artifacts created

`data/processed/evaluation/truthfulqa/{truthfulqa_eval_results.json,
truthfulqa_response_results.jsonl (790 lines), truthfulqa_error_analysis.json,
truthfulqa_eval_manifest.json}`. The JSONL stores per-question MC log-likelihoods, free-form
decision/latency metadata, and 150-char answer previews (plus the short, license-permissive
TruthfulQA question text itself, unlike RAGTruth's longer copyrighted source passages) - enough to
reproduce every aggregate metric above without rerunning inference.

### 19. Tests added (27 new, `tests/test_truthfulqa_evaluation.py`)

MC1/MC2/MC0 pure-arithmetic tests (including a tie-break determinism check and MC2's continuous-
score property); confirmation that `decision_rate_breakdown` is RE-EXPORTED (identity-checked, not
duplicated) from `claimguard.evaluation.ragtruth_eval`; `score_choice_log_likelihood` tests
including exact agreement with a hand-computed expected value (uniform-logit fake model, known
closed-form log-likelihood) and a genuine bug caught during THIS step's own test development (the
fake tokenizer emitted out-of-vocabulary token ids for the fake model - a test-fixture bug, not a
`score_choice_log_likelihood` bug, exactly analogous to Step 15's vocab-size fixture bug -
`score_choice_log_likelihood` correctly raised a loud `IndexError` rather than silently producing a
wrong score); the behavioral gold-isolation test (Section 17); and 8 real TruthfulQA dataset-
dimension tests (790/37, no duplicates, no split field, no `get_eval_set`/`get_training_pool`
accessors, mc0 exactly 2 options, mc2 multi-correct confirmed, no write-capable function exposed).

### 20. Full test-suite result

**395/395 passed**, run twice (once before the GPU evaluation, once after) - 368 pre-existing + 27
new.

### 21. Limitations

No approved automatic truthfulness judge exists for free-form generated text in this project - the
free-form pipeline track reports process behavior (decision rates, latency, retrieval diagnostics)
only, never a truthfulness accuracy number. Error-analysis categories 1-7 from the original
instructions (which require such a judge) are explicitly not computed, not invented. MC1/MC2/MC0
may partly reflect Qwen3's pretraining-data memorization of this old, public benchmark rather than
genuine calibration (standing caveat since Step 3). Small categories (<15 questions) are flagged
but still reported - interpret with appropriate caution.

### 22. Scientific interpretation

**A. What TruthfulQA shows about Qwen3 itself:** modest parametric calibration (MC1 34.05%, MC2
55.05%, MC0 44.18%), with a standing memorization caveat. **B. What TruthfulQA shows about
ClaimGuard:** the decision policy is never uncertain (0% attempt-0 ABSTAIN, same pattern as Steps
16 and 19) and correction much more often reaches ACCEPT here (76.6%) than on RAGTruth (6.1-18.4%)
- but this is a PROCESS observation, not evidence that ClaimGuard makes TruthfulQA answers more
truthful, since no truthfulness ground truth was measured for the free-form track. **C. What this
evaluation CANNOT establish, because the retrieval corpus is FEVER/Wikipedia-based:** whether
ClaimGuard's verification is actually grounding TruthfulQA answers in relevant evidence, or simply
finding topically-adjacent Wikipedia sentences (as Step 19 already established for RAGTruth) that
happen to satisfy the decision policy's criteria more easily for TruthfulQA's shorter,
general-knowledge-style questions. Mixed/contrasting results are preserved as such, not spun into a
general success or failure claim: MC1/MC2/MC0 are modest but unremarkable; correction-success is
high but uninterpretable as "more truthful"; category performance varies widely (0.0 to 0.875 MC1)
with no attempt to average this into one misleading summary number.

**Preserved, unchanged:** Qwen3-8B, DeBERTa verifier, BGE embedder, FAISS index, BGE reranker, and
Step 16's decision policy were all loaded read-only and never modified. RAGTruth's artifacts and
configuration were not touched during Step 20. No external web retrieval was used at any point.

**Cleanup:** removed all `step20_*.sh` temp remote scripts; no duplicate model processes; GPU
returned to baseline (only the unrelated concurrent job's memory remained) after the process
exited; nothing committed to git.

**Git status:** still no commits — new files (`src/claimguard/evaluation/truthfulqa_eval.py`,
`tests/test_truthfulqa_evaluation.py`, `scripts/evaluate_truthfulqa.py`,
`scripts/analyze_truthfulqa_results.py`, `data/processed/evaluation/truthfulqa/*`), modified files
(`src/claimguard/generation/qwen3.py`, `RESEARCH.md`) show as untracked/modified.

---

## Step 21 — Controlled ablations and failure-source analysis (2026-09-09/10)

**What:** Read `PROJECT_PLAN.md`, `PROJECT_REPORT.md`, `RESEARCH.md`, `CLAIMGUARD_MODEL_SELECTION.md`,
`README.md`, and the actual implementation of retrieval/reranking/verifier/Step 16/17/18/19/20
code before writing anything. Added ONE small, additive, backward-compatible extension to
`run_correction_loop` (`evidence_selection_mode`) enabling exactly the two evidence-selection
ablations the instructions require. Ran a 15-response pilot, then a 300-response deterministic
RAGTruth-test subset for Conditions A/B/C (new inference, ~2.71 hours); Conditions D/E were reused
directly from Step 19's already-validated full-2700 results with ZERO new inference. 16 new tests.
Did **not** tune any threshold/prompt/parameter, did **not** rerun RAGTruth's full 2,700-response
evaluation, did **not** touch TruthfulQA beyond reading its existing Step 20 results, did **not**
proceed to Step 22.

**Why:** Steps 19-20 established THAT ClaimGuard underperforms on RAGTruth (below the trivial
baseline) and identified the FEVER/Wikipedia corpus mismatch as a likely dominant cause, but had
not yet isolated the CAUSAL contribution of each individual frozen component (reranker, evidence
pool size, correction loop) via controlled removal. Step 21 answers that question directly.

### 1. Environment status

Python 3.12.14 (`claimguard` conda env), `torch==2.11.0+cu128`, `transformers==5.16.1`, NVIDIA RTX
PRO 5000 Blackwell. 395/395 tests passing before this step's code was written.

### 2. Frozen baseline definition

Identical to Steps 19-20: `Qwen/Qwen3-8B` (bf16, deterministic), DeBERTa binary verifier, BGE
embedder + unrebuilt FAISS index, BGE reranker, Step 16 decision policy (thresholds=0.5), Step 18
correction prompt, `max_correction_attempts=2` - all read from `configs/correction_baseline.yaml`
unchanged. No discrepancy from Step 19/20's configuration was found.

### 3. Ablation matrix

| Condition | Retrieval | Reranker | Verifier | Decision | Correction | Data source |
|---|---|---|---|---|---|---|
| A Original frozen | Yes (top-20) | Yes | Yes | Yes | Yes | NEW, 300-response subset |
| B No-reranker | Yes (top-20) | No (FAISS order) | Yes | Yes | Yes | NEW, same subset |
| C FAISS-top-1 | Yes (top-20, 1 used) | No | Yes | Yes | Yes | NEW, same subset |
| D No-correction | Yes | Yes | Yes | Yes | No | REUSED, Step 19 full n=2700 |
| E Verify-only | Yes | Yes | Yes | Yes | No (0 attempts) | REUSED, Step 19 full n=2700 |

D and E are functionally identical to already-computed Step 19 data (D = Step 19's `B_generated`
attempt-0 detection metrics; E = Step 19's `C_verify_only` condition) - explicitly NOT
recomputed, per the "do not duplicate computation" instruction.

### 4. Data/artifact reuse

Reused without modification: Step 19's `ragtruth_eval_results.json` (Conditions D/E, full-
population verifier-overconfidence numbers) and `ragtruth_response_results.jsonl` (2,700 per-
response records, used both for D/E extraction and for the Condition-A cross-validation check).
Step 20's `truthfulqa_eval_results.json` (MC1/MC2/MC0 numbers, Section 16). Newly computed:
`ablation_response_results.jsonl` (300 responses x 3 conditions, `scripts/run_ablations.py`).

### 5. Condition A results (original frozen ClaimGuard, NEW, n=300 subset)

Attempt-0 detection: accuracy **0.5867**, macro F1 **0.5813**, hallucinated recall **0.8974**,
hallucinated precision **0.4839**, not-hallucinated recall **0.3880**. `attempt0_abstain_rate=0.0`.
Correction-success-among-attempted **8.29%**, max-attempts-reached **66.33%**.
**Cross-validated**: 300/300 (100%) of this condition's final decisions match Step 19's
independently-stored full-run results for the SAME response_ids exactly - strong determinism
confirmation, not a new finding on its own.

### 6. Condition B results (no-reranker, NEW, n=300 subset)

Attempt-0 detection: accuracy **0.5867**, macro F1 **0.5813**, hallucinated recall **0.8974**,
hallucinated precision **0.4839**, not-hallucinated recall **0.3880** - **IDENTICAL to Condition A
to many decimal places**, despite the SELECTED EVIDENCE differing in 100% of responses (Section 10).
Correction-success-among-attempted drops to **2.76%** (vs A's 8.29%); max-attempts-reached rises to
**70.33%** (vs A's 66.33%) - a secondary, smaller degradation in CORRECTION dynamics specifically,
not in initial detection quality.

### 7. Condition C results (FAISS top-1 direct, NEW, n=300 subset)

Attempt-0 detection: accuracy **0.51**, macro F1 **0.4874**, hallucinated recall **0.9231**
(highest of the three, but at the cost of precision), hallucinated precision **0.4390**,
not-hallucinated recall **0.2459** (lowest - most false positives). Measurably WORSE than both A
and B on every metric except raw hallucinated-recall. Correction-success-among-attempted **5.69%**,
max-attempts-reached **77.33%** (highest).

### 8. Condition D results (no-correction, REUSED, n=2700)

Accuracy **0.5589**, macro F1 **0.5589**, hallucinated recall **0.8038**, hallucinated precision
**0.4297**, not-hallucinated recall **0.4274**. Numerically close to but not identical to Condition
A's subset numbers (0.5867 accuracy) - **architecturally IDENTICAL to A** (D is literally A's
attempt-0 snapshot before correction, computed on the full 2,700 rather than the 300-subset); the
difference reflects ordinary sampling variance between a 300-response subset and the full
population, not a different pipeline. This was directly confirmed by Section 5's cross-validation.

### 9. Condition E results (verify-only, REUSED, n=2700)

Accuracy **0.4874**, macro F1 **0.4811**, hallucinated recall **0.8558**, hallucinated precision
**0.3927**, not-hallucinated recall **0.2897**. E uses the ORIGINAL RAGTruth model response (not a
fresh Qwen3 generation) as the candidate, isolating the decision policy from generator effects -
notably lower accuracy than D/A, suggesting Qwen3's own generations are somewhat easier for the
(overconfident, corpus-mismatched) verifier to accept than the diverse pool of 6 RAGTruth source
models' original responses.

### 10. Retrieval analysis

Zero-evidence-retrieved rate: **0/300 (0%)** - retrieval NEVER fails to return candidates,
confirming Step 19's finding that this is a MECHANICAL success (the FAISS index always returns
`top_k` results). Evidence-SET composition changes in **100%** of responses between the reranked
(A) and no-reranker (B) conditions, and between A and FAISS-top-1 (C) - the specific evidence
identity is highly sensitive to selection method, even though (Section 11) the downstream decision
is not. Retrieval RELEVANCE failure (whether the returned evidence genuinely supports the query) is
explicitly reported as **NOT IDENTIFIABLE from the current evaluation** - RAGTruth provides no
per-evidence gold-relevance annotation analogous to FEVER's gold evidence sets, and inventing a
semantic relevance metric was explicitly out of scope.

### 11. Reranker analysis

**The reranker changes WHICH evidence is selected (100% of responses) but produces NO measurable
change in downstream classification metrics** (A vs B: accuracy/macro-F1/recall/precision
identical to 4+ decimal places). This directly answers Step 21's primary research question #1/#2:
reranking does not reliably improve (or measurably harm) evidence selection quality in this
end-to-end setting - consistent with the corpus-mismatch explanation, since reordering irrelevant
candidates cannot make them relevant. Reranking DOES have a smaller secondary effect on correction
dynamics (Section 6) - correction succeeds more often and hits the attempt budget less often WITH
reranking than without, even though the initial accept/flag decision is unaffected.

### 12. Verifier confidence analysis

Confidence-by-outcome (full 2,700, reused from Step 19's stored per-response data, using
`claimguard.evaluation.ablation_eval.confidence_by_outcome`):

| Outcome group | n | Mean confidence | Median confidence |
|---|---|---|---|
| Correctly accepted (gold=not-hallucinated) | 751 | 0.833 | 0.875 |
| WRONGLY accepted (gold=hallucinated, missed) | 185 | 0.821 | 0.872 |
| WRONGLY flagged (gold=not-hallucinated, false positive) | 1,006 | 0.971 | 0.989 |
| Correctly flagged (gold=hallucinated) | 758 | 0.982 | 0.991 |

**Confidence is high (0.82-0.98) across ALL FOUR groups regardless of correctness** - the
verifier's own confidence score does not discriminate right decisions from wrong ones. This is the
Step 15/16 FEVER-based overconfidence pattern (gold-SUPPORTS mean entailment ≈0.98, non-gold mean
≈0.70) CONFIRMED to persist on RAGTruth's out-of-domain data, via a response-level rather than
per-evidence-item lens (RAGTruth has no per-evidence gold-relevance annotation to replicate Step
16's exact framing). On the 300-response subset specifically, **115/300 (38.3%) of Condition A's
decisions were confidently (>=90%) wrong.**

### 13. Decision-policy analysis

`attempt0_abstain_rate = 0.0` in EVERY condition tested: A, B, C (n=300 each) and D, E (n=2,700
each). The decision policy never abstains, regardless of evidence-selection method, correction
status, or whether the candidate came from Qwen3 or an original RAGTruth model. This is a decision-
policy-level finding, not a verifier-level one: Step 16's fixed 0.5 thresholds on entailment/
contradiction probability essentially never produce a "neither confident" outcome given the
verifier's own extreme, bimodal confidence distribution (Section 12).

### 14. Correction benefit/harm analysis

Condition A (300 responses): `hallucinated_corrected_successfully` implicitly captured via
correction-success-rate-among-attempted (8.29%); harmful correction (`not_hallucinated_degraded` -
a genuinely clean response ending ABSTAIN) = **101/300 (33.7%)** - closely matching Step 19's
full-2700 rate of 34.2% (923/2,700), confirming the subset is representative on this dimension.
**Correction harms roughly 4x more often than it helps** (101 harmful vs an implied ~13 successful
corrections in this subset, consistent with Step 19's 923-harmful-vs-25-successful full-population
ratio). This is a downstream CONSEQUENCE of the upstream overconfidence/corpus-mismatch pattern
(Section 12), not shown here to be an independent defect in the correction PROMPT or LOGIC itself -
the correction loop faithfully executes what the (already-miscalibrated) decision policy tells it
to do.

### 15. Error analysis

Deterministic selection (first-N-by-response_id, or confidence-sorted where applicable), saved to
`ablation_error_analysis.json`:

| Failure category | Count | Denominator |
|---|---|---|
| 1. Retrieval failure (zero evidence) | 0 | 300 |
| 2. Retrieval relevance failure | NOT IDENTIFIABLE | - |
| 3. Verifier overconfident + wrong (conf>=0.9) | 115 | 300 |
| 4. Decision-policy zero-abstain | confirmed (see Section 13) | - |
| 5. Correction failure (harmful) | 101 | 300 |

### 16. TruthfulQA secondary observations

Reused directly from Step 20 (zero new inference): MC1 **34.05%**, MC2 **55.05%**, MC0 **44.18%**
(n=790, measures Qwen3's own calibration, no pipeline involved); free-form attempt-0 ABSTAIN rate
**0.0%** (same zero-abstain pattern as RAGTruth, Section 13); final ACCEPT rate **95.3%** (much
higher than RAGTruth's, already established in Step 20 as a process observation, NOT re-interpreted
here as evidence of higher truthfulness). Category variance and retrieval diagnostics are as
reported in Step 20 - not recomputed or re-analyzed further here.

### 17. Resource/runtime results

Pilot (15 responses): 665.7s (44.38s/response). Full ablation subset (300 responses x 3
conditions): **9,774.8s = 2.71 hours** (faster than the pilot-extrapolated ~3.7h estimate, similar
to Step 19's full run also beating its own pilot extrapolation). Peak GPU memory **18.25GB**,
consistent with Steps 18-20. Conditions D/E required **zero additional runtime** (pure reuse).

### 18. Reproducibility result

A 10-response deterministic subset (of the 300) was run through the complete A/B/C pipeline TWICE:
identical shared candidate answer, final decisions, and evidence counts across both runs - both in
the 15-response pilot and the full 300-response run.

### 19. Leakage/isolation audit

`gold_has_hallucination`/`labels` are read only by `scripts/analyze_ablations.py`'s scoring
functions, after all inference completed - confirmed structurally (AST guard, reused from Step
19/20) AND behaviorally (a NEW mock-based test, `TestGoldLabelIsolationInAblationOrchestration`,
runs the REAL `run_one_response` orchestration with a sentinel gold-label value and confirms it
never appears in any of the 3 `run_correction_loop` calls' actual arguments). No RAGTruth train
split, FEVER train/dev, or HaluEval training data was used anywhere in this step - only the
already-reserved RAGTruth TEST subset and Step 19/20's already-validated evaluation artifacts.

### 20. Artifacts created

`data/processed/evaluation/ablations/{ablation_results.json, ablation_comparison.csv,
ablation_error_analysis.json, ablation_run_manifest.json, ablation_response_results.jsonl}`.

### 21. Tests added (16 new, `tests/test_ablations.py`)

`_faiss_candidates_as_reranked_schema` schema/truncation correctness; all three
`evidence_selection_mode` values behaviorally verified with fake retriever/reranker objects
(confirms `"reranked"` calls the reranker and reverses order as expected, `"no_reranker"` NEVER
calls the reranker and preserves FAISS order, `"faiss_top1"` verifies exactly one candidate,
default is `"reranked"`, unknown modes raise); `confidence_by_outcome`'s 4-group classification and
empty-group handling; `evidence_movement_rate`'s set-based (order-independent) comparison and
length-mismatch guard; the behavioral gold-isolation test (Section 19); and a test confirming
`run_one_response` produces exactly Conditions A/B/C (never D/E, which are reused not recomputed).

### 22. Full test-suite result

**411/411 passed**, run twice (once before the GPU ablation run, once after) - 395 pre-existing +
16 new.

### 23. Main scientific findings

1. Reranking does NOT reliably improve evidence selection in this end-to-end setting (H2
   supported: reranking is not a meaningful bottleneck, but also not a fix).
2. Candidate POOL SIZE matters more than candidate ORDERING - FAISS-top-1 measurably underperforms
   both 5-candidate conditions regardless of whether those 5 were reranked.
3. Correction does not meaningfully improve final truthfulness detection and causes substantial
   harm (33.7% of subset responses) - a downstream consequence of upstream miscalibration, not an
   independently broken correction mechanism (H5 supported, but attributed correctly).
4. Verifier overconfidence (H3) and zero-abstention (H4) are strongly supported and PERSIST
   identically regardless of which evidence-selection ablation is applied - they are UPSTREAM of
   the reranker/correction-loop distinctions tested here.
5. Retrieval RELEVANCE failure could not be measured directly (no gold-relevance annotation
   exists for RAGTruth) - **not identifiable from the current evaluation**, reported honestly
   rather than inferred.

### 24. Limitations

Conditions A/B/C were measured on a 300-response deterministic subset (11.1% of the full 2,700),
not the full test set - Section 8's cross-validation shows Condition A's ARCHITECTURE matches
Step 19's full-population numbers exactly when computed on the same response_ids, but absolute
subset numbers carry more sampling variance than Step 19's full-2700 figures. Retrieval-relevance
failure remains unmeasured (no valid automatic metric exists without fabricating one). The
confidence-by-outcome analysis uses RESPONSE-level gold labels (RAGTruth's only available ground
truth), not the PER-EVIDENCE-ITEM gold-relevance Step 16 used for FEVER - a real, disclosed
difference in granularity between the two overconfidence analyses.

### 25. Recommended next research question

Given retrieval-corpus mismatch (H1) is the best-supported root cause and cannot be fixed without
changing the retrieval corpus (explicitly out of scope for this project's remaining steps), the
highest-value NEXT diagnostic question is: **does verifier overconfidence (H3) persist even when
retrieval quality is held constant at a KNOWN-GOOD level (e.g., re-examining Step 15/16's original
FEVER in-domain results specifically for whether overconfidence appears even THERE on non-gold-
but-topically-related evidence, which Step 16 already partially answered) - i.e., is H3 a problem
independent of H1, or entirely a downstream symptom of it?** This does not require new data
collection - it is a re-analysis question answerable from Step 15/16's EXISTING FEVER-domain
artifacts, fully consistent with this project's "no new tuning, no new data" constraints.

**Preserved, unchanged:** Qwen3-8B, DeBERTa verifier, BGE embedder, FAISS index, BGE reranker, Step
16's decision policy, and Step 18's correction prompt/max-attempts were all loaded read-only and
never modified. RAGTruth's Step 19 artifacts and TruthfulQA's Step 20 artifacts were read but never
altered. No training/dev dataset was touched.

**Cleanup:** removed all `step21_*.sh` temp remote scripts; no duplicate model processes; GPU
returned to baseline (only the unrelated concurrent job's memory remained) after the process
exited; nothing committed to git.

**Git status:** still no commits — new files (`src/claimguard/evaluation/ablation_eval.py`,
`tests/test_ablations.py`, `scripts/run_ablations.py`, `scripts/analyze_ablations.py`,
`data/processed/evaluation/ablations/*`), modified files (`src/claimguard/correction/correction.py`,
`src/claimguard/evaluation/__init__.py`, `RESEARCH.md`) show as untracked/modified.

---
