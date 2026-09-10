"""ClaimGuard retrieval: FAISS index construction and the retrieval API.

Index type: `faiss.IndexFlatIP` (exact, flat, inner-product) - the
simplest scientifically defensible choice, per instructions. Justified by
the embedding model's own normalization: `configs/models.yaml`'s embedding
entry sets `normalize_embeddings: true`, so BGE embeddings are L2-normalized
and inner product is exactly cosine similarity. No IVF/PQ/HNSW/GPU-index -
158,658 vectors x 1024 dims is trivially small for an exact CPU index
(a full linear scan is fast at this scale), so approximate indexing would
be premature optimization with no measured need.

FAISS row position is NEVER exposed as the primary identifier - every
result carries the corpus's own `corpus_id` plus full provenance
(page_id, sentence_id, text), via a separate metadata mapping
(`faiss row index -> corpus_id -> record`) that is saved and reloaded
alongside the index itself, so a future index rebuild (which could
reassign row positions) can never silently corrupt provenance.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .. import config as cg_config
from . import embed as embed_mod

INDEX_DIR = cg_config.resolve_path("data/processed/retrieval")
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
INDEX_METADATA_PATH = INDEX_DIR / "index_metadata.json"

METRIC = "inner_product"
INDEX_TYPE = "IndexFlatIP"


def build_index(embeddings: np.ndarray) -> Any:
    import faiss

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    if not embeddings.flags["C_CONTIGUOUS"]:
        embeddings = np.ascontiguousarray(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(
    index: Any,
    corpus_records: list[dict[str, Any]],
    corpus_ids_in_index_order: list[str],
    embedding_info: dict[str, Any],
    out_dir: Path = INDEX_DIR,
) -> None:
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    by_id = {r["corpus_id"]: r for r in corpus_records}
    # row_metadata[i] is the full provenance record for FAISS row i - the
    # ONLY place row position is used, and it is never handed back to a
    # caller in place of corpus_id.
    row_metadata = [by_id[cid] for cid in corpus_ids_in_index_order]

    metadata = {
        "index_type": INDEX_TYPE,
        "metric": METRIC,
        "dimension": index.d,
        "vector_count": index.ntotal,
        "embedding_info": embedding_info,
        "python_version": platform.python_version(),
        "faiss_version": __import__("faiss").__version__,
        "row_metadata": row_metadata,
    }
    with INDEX_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def load_index(out_dir: Path = INDEX_DIR) -> tuple[Any, dict[str, Any]]:
    import faiss

    if not FAISS_INDEX_PATH.exists() or not INDEX_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found under {out_dir}. Run scripts/build_faiss_index.py first."
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with INDEX_METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if index.ntotal != metadata["vector_count"]:
        raise RuntimeError(
            f"Loaded FAISS index vector count ({index.ntotal}) does not match saved metadata "
            f"({metadata['vector_count']}) - the index and metadata files are out of sync."
        )
    return index, metadata


def validate_index(index: Any, metadata: dict[str, Any]) -> list[str]:
    """Return a list of integrity issue codes - never raises, callers
    decide (matches the project's validate_record convention)."""
    issues: list[str] = []
    row_metadata = metadata["row_metadata"]
    if index.ntotal != len(row_metadata):
        issues.append(f"vector_metadata_count_mismatch:{index.ntotal}!={len(row_metadata)}")
    if index.d != metadata["dimension"]:
        issues.append(f"dimension_mismatch:{index.d}!={metadata['dimension']}")
    seen_ids = set()
    for r in row_metadata:
        cid = r["corpus_id"]
        if cid in seen_ids:
            issues.append(f"duplicate_corpus_id_in_index:{cid}")
        seen_ids.add(cid)
        if not r.get("text") or not r["text"].strip():
            issues.append(f"empty_text_in_index:{cid}")
    return issues


class Retriever:
    """Query-time retrieval API. Encodes the query with the SAME embedding
    model/config used to build the index (loaded fresh from
    `configs/models.yaml`, not hard-coded), searches the FAISS index, and
    returns full provenance per hit - never a bare FAISS row id.
    """

    def __init__(self, index: Any, metadata: dict[str, Any]):
        self.index = index
        self.metadata = metadata
        self.row_metadata = metadata["row_metadata"]
        self._by_corpus_id = {r["corpus_id"]: r for r in self.row_metadata}
        self._model = None

    def get_by_corpus_id(self, corpus_id: str) -> dict[str, Any]:
        """Direct provenance lookup by the stable corpus_id (never by raw
        FAISS row position). Raises KeyError - loudly, not None - if the
        id isn't in this index."""
        if corpus_id not in self._by_corpus_id:
            raise KeyError(f"corpus_id {corpus_id!r} not found in this index's metadata.")
        return self._by_corpus_id[corpus_id]

    @classmethod
    def from_disk(cls, out_dir: Path = INDEX_DIR) -> "Retriever":
        index, metadata = load_index(out_dir)
        return cls(index, metadata)

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            embedding_cfg = embed_mod.load_embedding_config()
            expected_name = self.metadata["embedding_info"]["name"]
            if embedding_cfg["name"] != expected_name:
                raise RuntimeError(
                    f"configs/models.yaml's embedding model ({embedding_cfg['name']!r}) does "
                    f"not match the model this index was built with ({expected_name!r}) - "
                    "refusing to query with a mismatched embedding space."
                )
            self._model, _ = embed_mod.load_embedding_model(embedding_cfg)
            self._normalize = embedding_cfg["normalize_embeddings"]

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        self._ensure_model_loaded()
        query_emb = embed_mod.embed_texts(self._model, [query], batch_size=1, normalize=self._normalize)
        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, row_idx in zip(scores[0], indices[0]):
            if row_idx < 0:  # FAISS pads with -1 if fewer than top_k results exist
                continue
            record = self.row_metadata[row_idx]
            results.append({
                "corpus_id": record["corpus_id"],
                "text": record["text"],
                "score": float(score),
                "page_id": record["page_id"],
                "sentence_id": record["sentence_id"],
                "corpus_version": record.get("corpus_version"),
            })
        return results
