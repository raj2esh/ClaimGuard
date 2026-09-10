"""ClaimGuard evidence decision policy (Step 16).

This module answers Problem B - "given the verifier's scores on the
reranked top-N candidates for a query, is the query actually
SUPPORTED, CONTRADICTED, or should ClaimGuard ABSTAIN?" - which is
distinct from Problem A ("which single candidate is the best evidence?",
already solved by `claimguard.verification.pipeline.select_evidence` in
Step 15).

The binary verifier trained in Step 12 is NOT modified, retrained, or
fine-tuned anywhere in this module. Step 16 adds only a downstream,
deterministic decision layer on top of the verifier's existing
entailment/contradiction probabilities. There is no machine-learned
component here - every threshold is a plain float comparison.

## ABSTAIN is not a model class

ABSTAIN is a policy-level *outcome*, produced by this module when the
verifier's own probabilities do not clear a configured confidence bar.
It is NOT a third label the verifier itself predicts, and it must not be
confused with Step 11's "neutral" class investigation: Step 11 asked
whether a neutral *training* label should exist for the verifier itself
(answer: no legitimate source existed, so the verifier remains strictly
binary - entailment/contradiction). ABSTAIN here is a downstream
decision-policy concept that can fire regardless of which of the two
verifier classes scored higher, whenever neither score is confident
enough to act on.

## Ground truth conventions used in this module (documented explicitly)

For a candidate `c` on query `q` (a FEVER SUPPORTS/REFUTES claim):
    is_gold(c)     = c["corpus_id"] in relevant_corpus_ids(q)
    true_label(q)  = "SUPPORTS" or "REFUTES" (FEVER-annotated)

"Accepted support" / "false support" / "missed support" (used by the
threshold-analysis script, not by `EvidenceDecisionPolicy.decide` itself)
are defined per-candidate, across ALL candidates from ALL queries
(SUPPORTS and REFUTES claims combined), as a binary detection task:
    predicted positive ("accepted support")  = entailment_probability >= threshold
    actual positive    ("true support")      = is_gold(c) AND true_label(q) == "SUPPORTS"
    false support  (false positive) = accepted support AND NOT true support
    missed support (false negative) = NOT accepted support AND true support
This flags exactly the candidates that genuinely are a SUPPORTS claim's
annotated evidence - a non-gold candidate or a gold-REFUTES candidate
that the verifier scores above threshold both count as false support,
because in both cases the pipeline would be treating an entailment score
as if it identified genuine supporting evidence when it did not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

Decision_T = Literal["SUPPORTED", "CONTRADICTED", "ABSTAIN"]

DEFAULT_ENTAILMENT_THRESHOLD = 0.5    # carried over unchanged from Step 15 - NOT re-tuned here
DEFAULT_CONTRADICTION_THRESHOLD = 0.5  # carried over unchanged from Step 15 - NOT re-tuned here


@dataclass
class Decision:
    """The full, auditable output of `EvidenceDecisionPolicy.decide`.
    Never a bare boolean - always carries the underlying scores and the
    selected evidence's full provenance (corpus_id, page_id,
    sentence_id, corpus_version, faiss/reranker/verifier scores)."""

    decision: Decision_T
    confidence: float
    selected_evidence: dict[str, Any] | None
    reason: str
    max_entailment_probability: float | None
    max_contradiction_probability: float | None
    second_highest_entailment_probability: float | None
    entailment_margin: float | None
    num_candidates: int
    num_above_entailment_threshold: int
    num_above_contradiction_threshold: int
    query: str | None = None


def _assert_finite_probabilities(candidates: list[dict[str, Any]]) -> None:
    """Reject (raise, never silently drop or clamp) any candidate whose
    verifier probabilities are non-finite (NaN/Inf) - a corrupted score
    must never silently influence a SUPPORTED/CONTRADICTED/ABSTAIN
    decision."""
    for c in candidates:
        for key in ("entailment_probability", "contradiction_probability"):
            value = c.get(key)
            if value is None or not math.isfinite(value):
                raise ValueError(
                    f"Candidate {c.get('corpus_id')!r} has non-finite {key}={value!r} - "
                    "refusing to make a decision from a corrupted verifier score."
                )


def compute_query_aggregates(verified_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic, non-learned per-query aggregate statistics over the
    verifier-scored candidates: max/second-highest entailment probability,
    the margin between them, and the max contradiction probability (which
    may come from a DIFFERENT candidate than the max-entailment one - the
    two questions "best entailment evidence" and "best contradiction
    evidence" are independent and both reported)."""
    _assert_finite_probabilities(verified_candidates)
    if not verified_candidates:
        return {
            "max_entailment_probability": None,
            "max_entailment_candidate": None,
            "second_highest_entailment_probability": None,
            "entailment_margin": None,
            "max_contradiction_probability": None,
            "max_contradiction_candidate": None,
            "num_candidates": 0,
        }

    by_entailment = sorted(verified_candidates, key=lambda c: c["entailment_probability"], reverse=True)
    max_entailment_candidate = by_entailment[0]
    max_entailment = max_entailment_candidate["entailment_probability"]
    second_entailment = by_entailment[1]["entailment_probability"] if len(by_entailment) > 1 else None
    margin = (max_entailment - second_entailment) if second_entailment is not None else None

    max_contradiction_candidate = max(verified_candidates, key=lambda c: c["contradiction_probability"])
    max_contradiction = max_contradiction_candidate["contradiction_probability"]

    return {
        "max_entailment_probability": max_entailment,
        "max_entailment_candidate": max_entailment_candidate,
        "second_highest_entailment_probability": second_entailment,
        "entailment_margin": margin,
        "max_contradiction_probability": max_contradiction,
        "max_contradiction_candidate": max_contradiction_candidate,
        "num_candidates": len(verified_candidates),
    }


def count_above_threshold(verified_candidates: list[dict[str, Any]], key: str, threshold: float) -> int:
    """Count candidates whose `key` probability is >= threshold. Shared by
    the policy and the standalone evidence-consistency analysis so both use
    identical semantics."""
    return sum(1 for c in verified_candidates if c[key] >= threshold)


@dataclass
class EvidenceDecisionPolicy:
    """A deterministic (no learned parameters) evidence-decision policy.

    Decision rule, applied in this fixed order:
      1. No candidates                                -> ABSTAIN
      2. max_entailment_probability >= entailment_threshold
             (and, if `require_margin` is set, entailment_margin >=
             require_margin)                           -> SUPPORTED
      3. max_contradiction_probability >= contradiction_threshold
             (same optional margin requirement, computed w.r.t. the
             contradiction-probability ranking)         -> CONTRADICTED
      4. otherwise                                      -> ABSTAIN

    `entailment_threshold`/`contradiction_threshold` default to Step 15's
    original 0.5 (unchanged, not re-tuned by this class). `require_margin`
    is optional (default None = no margin requirement) and implements the
    "require stronger agreement" evidence-consistency variant.
    """

    entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD
    contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD
    require_margin: float | None = None

    def decide(self, verified_candidates: list[dict[str, Any]]) -> Decision:
        _assert_finite_probabilities(verified_candidates)

        if not verified_candidates:
            return Decision(
                decision="ABSTAIN", confidence=0.0, selected_evidence=None,
                reason="no candidates available",
                max_entailment_probability=None, max_contradiction_probability=None,
                second_highest_entailment_probability=None, entailment_margin=None,
                num_candidates=0, num_above_entailment_threshold=0, num_above_contradiction_threshold=0,
            )

        agg = compute_query_aggregates(verified_candidates)
        n_above_ent = count_above_threshold(verified_candidates, "entailment_probability", self.entailment_threshold)
        n_above_contra = count_above_threshold(
            verified_candidates, "contradiction_probability", self.contradiction_threshold
        )
        common_kwargs = dict(
            max_entailment_probability=agg["max_entailment_probability"],
            max_contradiction_probability=agg["max_contradiction_probability"],
            second_highest_entailment_probability=agg["second_highest_entailment_probability"],
            entailment_margin=agg["entailment_margin"],
            num_candidates=agg["num_candidates"],
            num_above_entailment_threshold=n_above_ent,
            num_above_contradiction_threshold=n_above_contra,
        )

        margin_ok = self.require_margin is None or (
            agg["entailment_margin"] is not None and agg["entailment_margin"] >= self.require_margin
        )
        if agg["max_entailment_probability"] >= self.entailment_threshold and margin_ok:
            return Decision(
                decision="SUPPORTED",
                confidence=agg["max_entailment_probability"],
                selected_evidence=agg["max_entailment_candidate"],
                reason=(
                    f"max_entailment_probability={agg['max_entailment_probability']:.4f} "
                    f">= entailment_threshold={self.entailment_threshold}"
                    + (f" and margin={agg['entailment_margin']:.4f}>={self.require_margin}" if self.require_margin else "")
                ),
                **common_kwargs,
            )

        if agg["max_contradiction_probability"] >= self.contradiction_threshold:
            return Decision(
                decision="CONTRADICTED",
                confidence=agg["max_contradiction_probability"],
                selected_evidence=agg["max_contradiction_candidate"],
                reason=(
                    f"max_contradiction_probability={agg['max_contradiction_probability']:.4f} "
                    f">= contradiction_threshold={self.contradiction_threshold}"
                ),
                **common_kwargs,
            )

        reason = "no candidate exceeds the required confidence threshold"
        if self.require_margin is not None and not margin_ok:
            reason = (
                f"max_entailment_probability={agg['max_entailment_probability']:.4f} cleared the threshold "
                f"but entailment_margin={agg['entailment_margin']} did not meet require_margin={self.require_margin} "
                "(candidates are mutually ambiguous)"
            )
        return Decision(
            decision="ABSTAIN",
            confidence=max(agg["max_entailment_probability"], agg["max_contradiction_probability"]),
            selected_evidence=None,
            reason=reason,
            **common_kwargs,
        )


def decide(
    query: str, verified_candidates: list[dict[str, Any]], policy: "EvidenceDecisionPolicy | None" = None,
    **policy_kwargs: Any,
) -> Decision:
    """Module-level `decide(query, verified_candidates)` entry point (the
    interface required by Step 16). `query` is not used by the decision
    rule itself (only the already-scored candidates are) - it is
    stamped onto the returned `Decision.query` purely for provenance/
    logging, mirroring `run_pipeline`'s "query is the only per-call input"
    convention. Pass a pre-built `policy` for repeated calls, or
    `**policy_kwargs` to build a one-off `EvidenceDecisionPolicy`."""
    policy = policy or EvidenceDecisionPolicy(**policy_kwargs)
    result = policy.decide(verified_candidates)
    result.query = query
    return result
