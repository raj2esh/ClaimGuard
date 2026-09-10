"""ClaimGuard retrieval: retrieval-only evaluation (Step 13).

Uses ONLY FEVER train claims - never RAGTruth test, never TruthfulQA - as
the safe development retrieval benchmark. This is entirely separate from
the verifier's own dev evaluation (Step 12): retrieval evaluation measures
whether the FAISS index surfaces a claim's own known FEVER evidence
sentence(s) in its top-k results, using the claim TEXT as the query - it
does not involve the verifier model at all.

## Relevance criterion (defined explicitly, not left implicit)

For a FEVER train SUPPORTS/REFUTES claim whose evidence pages are all
present in the retrieval corpus (Step 13's corpus is built from exactly
these claims' evidence pages, so this holds for effectively all of them -
see PROJECT_REPORT.md Step 13), a retrieval at cutoff k is a HIT if AT
LEAST ONE sentence from ANY ONE of the claim's annotated evidence sets
appears among the top-k FAISS results for that claim's raw claim text used
as the query. This matches the standard FEVER shared-task
evidence-retrieval evaluation convention (sentence-level recall against
annotated evidence, not requiring every evidence sentence to be retrieved
simultaneously). Recall@k = (# claims with a hit at k) / (# claims
evaluated).

This is NOT a synthetic/artificial-query evaluation: the query is the
FEVER claim's own natural-language text (human-written, never identical to
any corpus sentence), and the corpus sentence used to determine relevance
is the SAME sentence FEVER's own annotators cited as evidence for that
claim - a legitimate, standard retrieval-evaluation design, not a
leakage shortcut.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .. import config as cg_config
from ..datasets import fever as fever_ds
from . import corpus as corpus_mod

RECALL_CUTOFFS = (1, 5, 10, 20)
DEFAULT_SEED = 42
DEFAULT_SAMPLE_SIZE = 2000


def build_eval_set(
    corpus_ids_in_corpus: set[str],
    seed: int = DEFAULT_SEED,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    raw_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Deterministically sample FEVER train SUPPORTS/REFUTES claims whose
    evidence is (at least partially) present in the retrieval corpus, and
    compute each sampled claim's relevant corpus_id set.

    Only claims with at least one evidence set FULLY present in the corpus
    are included - a claim referencing evidence outside the corpus's page
    scope would have no achievable ground truth here and is excluded, not
    silently scored as a miss.
    """
    kwargs = {"raw_dir": raw_dir} if raw_dir else {}
    claims = fever_ds.load_normalized_split("train", **kwargs)

    eligible: list[dict[str, Any]] = []
    for c in claims:
        if c["label"] not in fever_ds.RESOLVABLE_LABELS:
            continue
        relevant_ids: set[str] = set()
        for evidence_set in c["evidence"]:
            if not evidence_set:
                continue
            set_ids = {
                corpus_mod.corpus_id(s["wiki_url"], s["sentence_id"])
                for s in evidence_set if s["sentence_id"] != -1
            }
            if set_ids and set_ids.issubset(corpus_ids_in_corpus):
                relevant_ids |= set_ids
        if relevant_ids:
            eligible.append({
                "example_id": c["example_id"],
                "claim": c["claim"],
                "label": c["label"],
                "relevant_corpus_ids": sorted(relevant_ids),
            })

    rng = random.Random(seed)
    rng.shuffle(eligible)
    sampled = eligible[:sample_size]
    return sampled


def first_hit_rank(ranked_corpus_ids: list[str], relevant_ids: set[str]) -> int | None:
    """1-based rank of the first relevant corpus_id in an already-ordered
    list, or None if no relevant id appears anywhere in the list. Shared by
    both plain-FAISS and reranked evaluation (Step 14), so both use
    identically-defined relevance/rank semantics - not two subtly
    different implementations."""
    for rank, cid in enumerate(ranked_corpus_ids, start=1):
        if cid in relevant_ids:
            return rank
    return None


def recall_at_cutoffs(hit_ranks: list[int | None], cutoffs: tuple[int, ...] = RECALL_CUTOFFS) -> dict[str, Any]:
    """Recall@k for each cutoff, given one first_hit_rank per evaluated
    claim (None = no hit within whatever list length was searched)."""
    n = len(hit_ranks)
    hits_at_k = {k: sum(1 for r in hit_ranks if r is not None and r <= k) for k in cutoffs}
    recall = {f"recall_at_{k}": (hits_at_k[k] / n if n else 0.0) for k in cutoffs}
    return {"n_eval_claims": n, **recall, "hits_at_k": hits_at_k}


def evaluate_retrieval(retriever, eval_set: list[dict[str, Any]], max_k: int = max(RECALL_CUTOFFS)) -> dict[str, Any]:
    """Run retrieval for every eval-set claim and compute Recall@k for each
    cutoff in RECALL_CUTOFFS. Returns per-cutoff recall plus per-claim hit
    detail for auditability (not just the aggregate number)."""
    per_claim: list[dict[str, Any]] = []
    hit_ranks: list[int | None] = []

    for item in eval_set:
        results = retriever.retrieve(item["claim"], top_k=max_k)
        retrieved_ids = [r["corpus_id"] for r in results]
        relevant = set(item["relevant_corpus_ids"])

        rank = first_hit_rank(retrieved_ids, relevant)
        hit_ranks.append(rank)
        per_claim.append({
            "example_id": item["example_id"],
            "first_hit_rank": rank,
            "num_relevant": len(relevant),
        })

    summary = recall_at_cutoffs(hit_ranks)
    return {**summary, "per_claim": per_claim}
