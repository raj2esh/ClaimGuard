"""ClaimGuard correction/regeneration loop (Step 18).

    user query -> Qwen3-8B -> candidate answer -> retrieve -> rerank ->
        verify -> Step 16 decision policy ->
            ACCEPT   -> return the answer
            CORRECT  -> evidence-grounded regeneration, re-verify (bounded)
            ABSTAIN  -> return explicit abstention

This is an ENGINEERING/INTEGRATION step. It is NOT the final
hallucination-detection performance evaluation - RAGTruth remains
reserved for Step 19, TruthfulQA for Step 20.

(Step 19 note: `run_correction_loop` gained one optional,
backward-compatible parameter, `initial_candidate` - default `None`
preserves every Step 18 behavior/test exactly. See its docstring below.)

Reuses, UNCHANGED, every underlying component:
    - `claimguard.retrieval.index.Retriever.retrieve` (Step 13)
    - `claimguard.reranking.reranker.rerank` (Step 14)
    - `claimguard.verification.verify.verify_candidates` (Step 15)
    - `claimguard.decision.policy.EvidenceDecisionPolicy` (Step 16's
      decision rule - NOT reimplemented or retuned here)
    - `claimguard.generation.qwen3.generate` (Step 17's frozen generator)

Qwen3-8B, the DeBERTa verifier, the BGE reranker, and the FAISS
retrieval corpus/index are all FROZEN in this step - no fine-tuning, no
retraining, no corpus/index changes. All model objects are passed in
already-loaded by the caller; this module never loads or instantiates a
model itself, so it can never create a duplicate resident copy.

## Terminology mapping (documented, not silently invented)

Step 16's `EvidenceDecisionPolicy` produces one of SUPPORTED / CONTRADICTED
/ ABSTAIN (see `claimguard.decision.policy`). Step 18 needs ACCEPT /
CORRECT / ABSTAIN semantics for the correction loop. The mapping used
throughout this module is:

    SUPPORTED    -> ACCEPT   (evidence sufficiently supports the candidate)
    CONTRADICTED -> CORRECT  (evidence contradicts the candidate - retry)
    ABSTAIN      -> ABSTAIN  (neither confident enough - stop)

This is an INTEGRATION-LAYER renaming only. Step 16's actual decision
rule (thresholds, margin logic, tie-breaking) is never modified, retuned,
or reimplemented here - `STEP16_TO_STEP18_DECISION` is a pure relabeling
of `Decision.decision`, applied AFTER `EvidenceDecisionPolicy.decide()`
returns.

## What "query" means to the pipeline at each attempt

The retrieval/reranking/verification `query` used at every attempt is the
CURRENT CANDIDATE ANSWER TEXT being fact-checked - never the raw user
question, and never anything derived from hidden model reasoning. The
user's original question (`original_query`) is preserved separately and
used only for (a) the very first generation call and (b) building later
correction prompts.

## Multi-claim limitation (documented, not silently assumed away)

The Step 16 decision policy operates at the answer/evidence level (its
input is a list of verified CANDIDATES for one query, and its output is
one decision for that query) - it does not decompose a multi-sentence
answer into independently-verified atomic claims. This module preserves
that same abstraction: a candidate answer is treated as a single unit
checked against the retrieved evidence, not as N independently-verified
claims. No new claim-extraction/decomposition model is introduced here.
"""

from __future__ import annotations

import time
from typing import Any

from ..decision import policy as policy_mod
from ..generation import qwen3 as qwen3_mod
from ..reranking import reranker as reranker_mod
from ..verification import verify as verify_mod
from .types import AttemptLatency, AttemptRecord, CorrectionResult, EvidenceItem, FailureInfo

STEP16_TO_STEP18_DECISION: dict[str, str] = {
    "SUPPORTED": "ACCEPT",
    "CONTRADICTED": "CORRECT",
    "ABSTAIN": "ABSTAIN",
}

DEFAULT_MAX_CORRECTION_ATTEMPTS = 2  # conservative engineering baseline - never tuned against
                                      # RAGTruth/TruthfulQA/any evaluation data

CORRECTION_SYSTEM_INSTRUCTION = (
    "You are a careful assistant. Answer using only the evidence you are given. "
    "Do not state anything the evidence does not support. Respond with only your final "
    "answer - no explanation of your reasoning process."
)


def _evidence_item(candidate: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        corpus_id=candidate["corpus_id"], text=candidate["text"], page_id=candidate["page_id"],
        sentence_id=candidate["sentence_id"], original_rank=candidate["original_rank"],
        reranked_rank=candidate["reranked_rank"],
        entailment_probability=candidate["entailment_probability"],
        contradiction_probability=candidate["contradiction_probability"],
    )


def build_correction_prompt(original_query: str, previous_answer: str, evidence_texts: list[str]) -> str:
    """Evidence-grounded correction prompt. Built ONLY from evidence text
    the real retrieval/reranking/verification pipeline actually produced
    for the previous candidate - never fabricated, never gold evidence.
    No chain-of-thought/reasoning-trace request; the model is explicitly
    asked to return only its final answer."""
    if evidence_texts:
        evidence_block = "\n".join(f"- {t}" for t in evidence_texts)
    else:
        evidence_block = "(no supporting evidence was found)"
    return (
        f"Original question: {original_query}\n\n"
        f"Previous answer: {previous_answer}\n\n"
        f"Relevant evidence:\n{evidence_block}\n\n"
        "Using ONLY the evidence above, provide a corrected answer to the original question. "
        "Answer the original question directly. Correct any part of the previous answer that "
        "the evidence does not support, and preserve any part the evidence does support. "
        "Do not invent facts that are not present in the evidence above. "
        "Respond with only the final corrected answer."
    )


EVIDENCE_SELECTION_MODES = ("reranked", "no_reranker", "faiss_top1")


def _faiss_candidates_as_reranked_schema(faiss_candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    """Repackages raw FAISS candidates (key `score`, FAISS rank order)
    into the SAME schema `reranker.rerank()` produces (`faiss_score`,
    `reranker_score`, `original_rank`, `reranked_rank`) WITHOUT calling
    the reranker at all - used by the Step 21 `no_reranker`/`faiss_top1`
    ablation conditions so `verify.verify_candidates()` downstream sees
    an identical record shape regardless of evidence-selection mode.
    `reranker_score` is set to `None` (never fabricated) since no
    reranking happened; `reranked_rank` equals `original_rank` (FAISS
    order is left untouched, not re-sorted by anything)."""
    selected = faiss_candidates[:top_n]
    return [
        {
            "corpus_id": c["corpus_id"], "text": c["text"], "faiss_score": c["score"],
            "reranker_score": None, "original_rank": i + 1, "reranked_rank": i + 1,
            "page_id": c["page_id"], "sentence_id": c["sentence_id"], "corpus_version": c.get("corpus_version"),
        }
        for i, c in enumerate(selected)
    ]


def _retrieve_rerank_verify(
    retriever: Any, reranker_model: Any, reranker_batch_size: int,
    verifier_model: Any, verifier_tokenizer: Any, candidate_text: str,
    retrieval_top_k: int, rerank_top_n: int, verifier_max_length: int,
    evidence_selection_mode: str = "reranked",
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Runs the SAME three primitives Step 15's `run_pipeline` uses
    (`retriever.retrieve`, `reranker.rerank`, `verify.verify_candidates`),
    called individually rather than through `run_pipeline` so latency can
    be recorded per-stage as Step 18 requires. Reuses the existing
    functions exactly - does not reimplement their logic.

    `evidence_selection_mode` (Step 21 ablation infrastructure, default
    `"reranked"` preserves Step 15/18/19 behavior EXACTLY):
        "reranked"     - unchanged: FAISS top-k, then BGE-reranked top-n.
        "no_reranker"  - FAISS top-k, then the first `rerank_top_n`
                          candidates taken DIRECTLY in FAISS order - the
                          reranker model/call is skipped entirely.
        "faiss_top1"   - only the single highest-ranked FAISS candidate
                          is verified (retrieval_top_k/rerank_top_n still
                          govern retrieval depth, but exactly one
                          candidate reaches the verifier).
    Retrieval itself (`retriever.retrieve`, `retrieval_top_k`) is
    UNCHANGED across all three modes - only what happens to the FAISS
    result afterward differs, so this isolates evidence-SELECTION, not
    retrieval depth.
    """
    if evidence_selection_mode not in EVIDENCE_SELECTION_MODES:
        raise ValueError(f"Unknown evidence_selection_mode: {evidence_selection_mode!r}")

    t0 = time.time()
    faiss_candidates = retriever.retrieve(candidate_text, top_k=retrieval_top_k)
    t1 = time.time()

    if evidence_selection_mode == "reranked":
        reranked_candidates = reranker_mod.rerank(
            reranker_model, candidate_text, faiss_candidates, top_n=rerank_top_n, batch_size=reranker_batch_size,
        )
    elif evidence_selection_mode == "no_reranker":
        reranked_candidates = _faiss_candidates_as_reranked_schema(faiss_candidates, top_n=rerank_top_n)
    else:  # "faiss_top1"
        reranked_candidates = _faiss_candidates_as_reranked_schema(faiss_candidates, top_n=1)
    t2 = time.time()

    verified_candidates = verify_mod.verify_candidates(
        verifier_model, verifier_tokenizer, candidate_text, reranked_candidates, max_length=verifier_max_length,
    )
    t3 = time.time()
    latencies = {
        "retrieval_seconds": t1 - t0, "reranking_seconds": t2 - t1, "verification_seconds": t3 - t2,
    }
    return verified_candidates, latencies


def run_correction_loop(
    retriever: Any, reranker_model: Any, reranker_batch_size: int,
    verifier_model: Any, verifier_tokenizer: Any, generator_model: Any, generator_tokenizer: Any,
    query: str, *, initial_candidate: str | None = None, generator_model_identifier: str = "unknown",
    retrieval_top_k: int = 20, rerank_top_n: int = 5, verifier_max_length: int = 256,
    max_new_tokens: int = 256, seed: int = 42,
    entailment_threshold: float = 0.5, contradiction_threshold: float = 0.5,
    max_correction_attempts: int = DEFAULT_MAX_CORRECTION_ATTEMPTS,
    evidence_selection_mode: str = "reranked",
) -> CorrectionResult:
    """The bounded ClaimGuard correction/regeneration loop for one user
    query. Terminates after at most `1 + max_correction_attempts`
    generations - never an unbounded while-loop. Every corrected answer
    is independently re-retrieved, re-reranked, and re-verified; stale
    verifier results are never reused across attempts.

    All model objects (`retriever`, `reranker_model`, `verifier_model`/
    `verifier_tokenizer`, `generator_model`/`generator_tokenizer`) must
    already be loaded by the caller - this function never loads a model
    itself, so it can never create a duplicate resident GPU copy.

    `initial_candidate` (added for Step 19, default `None` preserves all
    Step 18 behavior exactly): when provided, attempt 0 SKIPS the initial
    generation call and uses this text as the starting candidate answer
    instead (`AttemptLatency.generation_seconds` recorded as 0.0 for that
    attempt). This lets a caller evaluate ClaimGuard's verification/
    correction behavior starting from an ALREADY-EXISTING answer (e.g. a
    RAGTruth reference response) rather than one Qwen3 just generated,
    without changing anything about how verification, decision, or
    correction themselves work. Combined with `max_correction_attempts=0`,
    this also yields a single verify-only pass (no regeneration even if
    the policy says CORRECT) - a compatibility-preserving extension, not
    a change to the decision rule or the correction prompt.

    `evidence_selection_mode` (added for Step 21, default `"reranked"`
    preserves all Step 15/18/19 behavior exactly): see
    `_retrieve_rerank_verify`'s docstring - `"no_reranker"` and
    `"faiss_top1"` are controlled ABLATION conditions (Step 21) that
    change ONLY how retrieved evidence is selected for verification,
    never retrieval depth, the decision policy, or the correction
    prompt.

    Knows nothing about FEVER/RAGTruth/TruthfulQA/HaluEval labels or gold
    evidence - its only inputs are already-loaded models, the user's
    query, an optional plain-text starting candidate, and generic
    generation/pipeline configuration values.
    """
    decision_policy = policy_mod.EvidenceDecisionPolicy(
        entailment_threshold=entailment_threshold, contradiction_threshold=contradiction_threshold,
    )
    attempts: list[AttemptRecord] = []
    loop_start = time.time()

    def _fail(
        stage: str, exc: Exception, candidate_answer: str | None, correction_occurred: bool,
        termination_reason: str | None = None,
    ) -> CorrectionResult:
        return CorrectionResult(
            original_query=query, attempts=attempts, final_decision="ABSTAIN", final_answer=candidate_answer,
            termination_reason=termination_reason or f"{stage}_failed", correction_occurred=correction_occurred,
            total_attempts=len(attempts), max_correction_attempts=max_correction_attempts, status="failed",
            failure=FailureInfo(stage=stage, error_type=type(exc).__name__, message=str(exc)),
            total_latency_seconds=time.time() - loop_start,
        )

    def _finish(final_decision: str, final_answer: str | None, termination_reason: str, correction_occurred: bool) -> CorrectionResult:
        return CorrectionResult(
            original_query=query, attempts=attempts, final_decision=final_decision, final_answer=final_answer,
            termination_reason=termination_reason, correction_occurred=correction_occurred,
            total_attempts=len(attempts), max_correction_attempts=max_correction_attempts, status="completed",
            failure=None, total_latency_seconds=time.time() - loop_start,
        )

    # --- Attempt 0: initial generation from the user's original query,
    # UNLESS a pre-supplied starting candidate was given (Step 19 extension) ---
    if initial_candidate is not None:
        candidate_answer = initial_candidate
        generation_seconds = 0.0
    else:
        try:
            gen_result = qwen3_mod.generate(
                generator_model, generator_tokenizer, query, model_identifier=generator_model_identifier,
                max_new_tokens=max_new_tokens, do_sample=False, seed=seed,
            )
            candidate_answer = gen_result.generated_text
            generation_seconds = gen_result.latency_seconds
        except Exception as exc:  # noqa: BLE001
            return _fail("generation", exc, None, False)

    attempt_number = 0
    while True:
        if not candidate_answer or not candidate_answer.strip():
            return _fail(
                "generation", ValueError("Generator returned an empty answer."), candidate_answer,
                attempt_number > 0, termination_reason="empty_generated_answer",
            )

        try:
            verified_candidates, stage_latencies = _retrieve_rerank_verify(
                retriever, reranker_model, reranker_batch_size, verifier_model, verifier_tokenizer,
                candidate_answer, retrieval_top_k, rerank_top_n, verifier_max_length,
                evidence_selection_mode=evidence_selection_mode,
            )
        except Exception as exc:  # noqa: BLE001
            return _fail("retrieval_or_reranking_or_verification", exc, candidate_answer, attempt_number > 0)

        try:
            decision = decision_policy.decide(verified_candidates)
            mapped = STEP16_TO_STEP18_DECISION[decision.decision]
        except Exception as exc:  # noqa: BLE001
            return _fail("decision_policy", exc, candidate_answer, attempt_number > 0)

        attempts.append(AttemptRecord(
            attempt_number=attempt_number, candidate_answer=candidate_answer,
            evidence=[_evidence_item(c) for c in verified_candidates],
            selected_evidence=_evidence_item(decision.selected_evidence) if decision.selected_evidence else None,
            step16_decision=decision.decision, step16_confidence=decision.confidence, step16_reason=decision.reason,
            mapped_decision=mapped,
            latency=AttemptLatency(
                generation_seconds=generation_seconds, retrieval_seconds=stage_latencies["retrieval_seconds"],
                reranking_seconds=stage_latencies["reranking_seconds"],
                verification_seconds=stage_latencies["verification_seconds"],
            ),
        ))

        if mapped == "ACCEPT":
            return _finish("ACCEPT", candidate_answer, "accepted", attempt_number > 0)
        if mapped == "ABSTAIN":
            return _finish("ABSTAIN", candidate_answer, "policy_abstained", attempt_number > 0)

        # mapped == "CORRECT"
        if attempt_number >= max_correction_attempts:
            return _finish("ABSTAIN", candidate_answer, "max_attempts_reached", True)

        evidence_texts = [c["text"] for c in verified_candidates]
        correction_prompt = build_correction_prompt(query, candidate_answer, evidence_texts)
        try:
            gen_result = qwen3_mod.generate(
                generator_model, generator_tokenizer, correction_prompt,
                system_instruction=CORRECTION_SYSTEM_INSTRUCTION, model_identifier=generator_model_identifier,
                max_new_tokens=max_new_tokens, do_sample=False, seed=seed,
            )
            candidate_answer = gen_result.generated_text
            generation_seconds = gen_result.latency_seconds
        except Exception as exc:  # noqa: BLE001
            return _fail("correction_generation", exc, candidate_answer, True)

        attempt_number += 1
