"""Structured state/result types for the ClaimGuard correction/regeneration
loop (Step 18). Deliberately generic - no FEVER/RAGTruth/TruthfulQA/
HaluEval label field, no gold-evidence field, anywhere in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

AttemptDecision = Literal["ACCEPT", "CORRECT", "ABSTAIN"]
FinalDecision = Literal["ACCEPT", "ABSTAIN"]


@dataclass
class EvidenceItem:
    """One piece of evidence actually produced by the real retrieval ->
    reranking -> verification pipeline, with full provenance preserved -
    never fabricated, never substituted with gold evidence."""

    corpus_id: str
    text: str
    page_id: str
    sentence_id: int
    original_rank: int
    reranked_rank: int
    entailment_probability: float
    contradiction_probability: float


@dataclass
class AttemptLatency:
    generation_seconds: float
    retrieval_seconds: float
    reranking_seconds: float
    verification_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            self.generation_seconds + self.retrieval_seconds
            + self.reranking_seconds + self.verification_seconds
        )


@dataclass
class AttemptRecord:
    """One full round of the loop: a candidate answer plus everything the
    pipeline produced while checking it. `mapped_decision` is the Step 18
    ACCEPT/CORRECT/ABSTAIN label; `step16_decision` preserves the RAW
    Step 16 SUPPORTED/CONTRADICTED/ABSTAIN label unchanged, so the mapping
    is always auditable rather than silently applied."""

    attempt_number: int
    candidate_answer: str
    evidence: list[EvidenceItem]
    selected_evidence: EvidenceItem | None
    step16_decision: str
    step16_confidence: float
    step16_reason: str
    mapped_decision: AttemptDecision
    latency: AttemptLatency

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureInfo:
    stage: str
    error_type: str
    message: str


@dataclass
class CorrectionResult:
    """The full, auditable output of `run_correction_loop`. `final_answer`
    is the LAST candidate text produced even when `final_decision` is
    ABSTAIN (so nothing is silently discarded) - callers MUST check
    `final_decision`, never assume a returned answer is verified."""

    original_query: str
    attempts: list[AttemptRecord]
    final_decision: FinalDecision
    final_answer: str | None
    termination_reason: str
    correction_occurred: bool
    total_attempts: int
    max_correction_attempts: int
    status: Literal["completed", "failed"]
    failure: FailureInfo | None
    total_latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
