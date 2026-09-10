# ClaimGuard — Model & Dataset Selection

Status: research complete, nothing downloaded, no environment changes.
Hardware: NVIDIA RTX PRO 5000 Blackwell, ~48GB VRAM (shared with SAM-RNet `exp42`, ~12GB in use).
Environment: `claimguard` conda env — Python 3.12.14, torch 2.11.0+cu128, transformers 5.16.1.

## Recommended Stack

| Role | Model / Approach | Why |
|---|---|---|
| **Generator (primary)** | `Qwen/Qwen3-8B` | Apache 2.0, strong instruction-following, ~16GB BF16 weights — smallest footprint of the comparable candidates, leaves headroom for the rest of the pipeline |
| **Generator (secondary, robustness)** | `meta-llama/Llama-3.1-8B-Instruct` | Different lab/pretraining/alignment → tests whether hallucination reduction generalizes across model families, not just Qwen |
| **Generator to avoid** | `google/gemma-4-12B-it` (Unified, multimodal) | Unneeded multimodal complexity for a text-only task; very new, unverified QLoRA/tooling maturity; VRAM cost on par with 12B text-only models while doing more work per pass |
| **Claim extractor** | LLM-prompted atomic-fact decomposition (FActScore-style), via the generator itself | Simplest to implement first; revisit with a dedicated model only if extraction quality is poor |
| **Embedding** | `BAAI/bge-large-en-v1.5` (primary), `BAAI/bge-base-en-v1.5` (fast alternative) | English-only task doesn't need bge-m3's multilingual/sparse machinery; small (<1.5GB), native sentence-transformers, established retrieval quality |
| **Retriever** | BM25 (sparse) + FAISS over bge-large embeddings (hybrid) | Standard hybrid retrieval; faiss-cpu already installed, sufficient at this evidence-corpus scale |
| **Reranker** | `BAAI/bge-reranker-large` (primary), `BAAI/bge-reranker-v2-m3` (alternative) | English-focused, proven, small, native `CrossEncoder`; avoided the 2.5B Gemma2-backbone BGE variant as disproportionate VRAM cost + unverified license |
| **Verifier** | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (primary), `...-v3-base-mnli-fever-anli` (backup) | Already trained on FEVER-NLI + MNLI + ANLI + LingNLI + WANLI (885K pairs), MIT license, 3-way labels map directly to SUPPORTS/REFUTES/NEI, <2GB VRAM |
| **Correction model** | Qwen3-8B in "correction mode" (prompted with claim + verifier verdict + retrieved evidence) | Reuses the generator rather than adding a second large model; revisit as a dedicated QLoRA adapter once enough (claim, evidence, correction) training pairs exist |

## Datasets

| Dataset | Recommended use |
|---|---|
| FEVER (185K claims, SUPPORTS/REFUTES/NEI, CC BY-SA 3.0) | Verifier training + validation |
| HaluEval (~35K, MIT) | Verifier + correction-trigger training/validation |
| RAGTruth (~18K span-annotated, MIT) | Verifier training/validation **and primary held-out TEST** for pipeline effectiveness |
| TruthfulQA (817 Q, Apache 2.0) | Evaluation-only — adversarial stress test |
| FActScore / FActScore-Bio (183 prompts, MIT) | Evaluation-only + design template for claim extraction |
| SelfCheckGPT WikiBio (`potsawee/wiki_bio_gpt3_hallucination`) | Evaluation / baseline comparison (verify license before use) |
| FEVEROUS (87K claims, optional) | Skip for initial scope — no tabular evidence planned |

**Data leakage note:** FEVER and TruthfulQA are old, ubiquitous benchmarks very likely present in the pretraining corpora of all candidate generator LLMs. Using them to measure *the generator's own* hallucination rate risks measuring memorization rather than the pipeline's effect — fine for verifier training (in-distribution, their intended use), not as the primary end-to-end effectiveness claim. **RAGTruth is the cleanest test signal** — naturalistic LLM hallucinations, not synthetic/memorizable benchmark claims.

## GPU Feasibility Summary

Current free VRAM: ~35GB (of 48GB, ~12GB held by SAM-RNet `exp42`).

- Full inference pipeline (Qwen3-8B BF16 + verifier + embedder + reranker) concurrently: **~23–25GB** → fits safely, ~10GB buffer remains alongside SAM-RNet.
- QLoRA fine-tuning of the 8B generator: **~10–14GB** additional — run in isolation (other pipeline components unloaded), not concurrently with the full inference pipeline, to stay clear of the shared ceiling.

## Research Protocol (experiment outline)

1. **Verifier validation** — FEVER dev + held-out RAGTruth-derived triples.
2. **Baseline hallucination rate** — Qwen3-8B / Llama-3.1-8B-Instruct on HaluEval + TruthfulQA, no pipeline, FActScore-style scoring.
3. **Pipeline intervention** — full loop (extract → retrieve → rerank → verify → correct) on the same outputs; primary effectiveness claim on RAGTruth.
4. **Cross-model + ablation** — repeat with secondary generator; ablate retrieval-only / verify-only / full loop.

## Verification status

Model names, params, licenses, context lengths, and dataset sizes/labels/licenses above were checked against current Hugging Face model/dataset cards and official sources as of 2026-08-31. VRAM figures are engineering estimates (param-count heuristics), not benchmarked. Exact `transformers==5.16.1` per-model compatibility was not independently verifiable and should be confirmed with a lightweight `from_pretrained(..., torch_dtype="bfloat16")` config/weight-metadata load (no generation) before implementation begins.
