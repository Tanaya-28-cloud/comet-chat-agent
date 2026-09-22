"""
Retrieval layer for the Aster & Row support agent.

Two-stage design:

  1. `search()` — plain embedding similarity search over every indexed
     chunk. No filtering. This alone would happily retrieve the
     superseded returns policy or the internal migration scratchpad if
     they're semantically close to the query — which is exactly what
     we don't want to hand to the model as "the answer".

  2. `retrieve_authoritative()` — takes the search results and
     deterministically partitions them into:
       - `authoritative`: chunks whose document is active, official,
         and customer-facing. These are the only chunks that may be
         cited as the basis for a policy answer.
       - `excluded`: everything else, each tagged with *why* it was
         excluded (superseded, internal audience, draft status, etc).
         These are still returned (not silently dropped) because the
         agent orchestrator needs to know when a customer is pointing
         at a non-authoritative document (e.g. "the migration note
         says...") so it can explicitly say "that's not an authoritative
         source" rather than just drawing a blank.

The authority classification is intentionally simple and independent of
embedding scores — it's a pure function of front-matter metadata, so a
new document can't accidentally become "authoritative" just because it
scores well on a particular query.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.kb_indexer import Chunk, EMBEDDING_MODEL_NAME, INDEX_DIR, load_index


# ---------------------------------------------------------------------------
# Authority classification
# ---------------------------------------------------------------------------

def classify_authority(metadata: dict) -> tuple[bool, Optional[str]]:
    """Returns (is_authoritative, exclusion_reason).

    A chunk is authoritative only if ALL of:
      - status == "active"            (not superseded, not draft)
      - audience == "customer"        (not an internal-only doc)
      - policy_authority == "official" (not "none" — e.g. the migration scratchpad)

    Checks run in this order so the exclusion_reason is the single most
    relevant reason, not a merged list — useful for debug logs and for
    the eval suite's assertions.
    """
    status = metadata.get("status")
    if status != "active":
        return False, f"status={status!r} (not active)"

    audience = metadata.get("audience")
    if audience != "customer":
        return False, f"audience={audience!r} (not customer-facing)"

    policy_authority = metadata.get("policy_authority")
    if policy_authority != "official":
        return False, f"policy_authority={policy_authority!r} (not official)"

    return True, None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class RetrievalResult:
    query: str
    authoritative: list[RetrievedChunk]
    excluded: list[tuple[RetrievedChunk, str]]  # (chunk, exclusion_reason)

    def citation_for(self, retrieved: RetrievedChunk) -> str:
        """Formats a chunk as a citation: filename + heading, per the
        README requirement that a source identify 'at least the filename
        and relevant heading'."""
        c = retrieved.chunk
        if c.heading:
            return f"{c.source_file} — {c.heading}"
        return c.source_file


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray, model_name: str = EMBEDDING_MODEL_NAME):
        assert len(chunks) == embeddings.shape[0], "chunks/embeddings length mismatch"
        self.chunks = chunks
        self.embeddings = embeddings  # assumed L2-normalized (see kb_indexer.embed_chunks)
        self.model_name = model_name
        self._model = None  # lazy-loaded sentence-transformers model

    @classmethod
    def from_index(cls, index_dir: Path = INDEX_DIR) -> "Retriever":
        chunks, embeddings = load_index(index_dir)
        return cls(chunks, embeddings)

    def _embed_query(self, query: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy import, mirrors kb_indexer

            self._model = SentenceTransformer(self.model_name)
        vec = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec[0], dtype=np.float32)

    def search(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        """Plain top-k similarity search, unfiltered. Embeddings are
        pre-normalized, so a dot product IS the cosine similarity —
        no need for an extra division step here."""
        query_vec = self._embed_query(query)
        scores = self.embeddings @ query_vec
        top_indices = np.argsort(-scores)[:k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(scores[i])) for i in top_indices]

    def retrieve_authoritative(self, query: str, k: int = 8) -> RetrievalResult:
        results = self.search(query, k=k)
        authoritative: list[RetrievedChunk] = []
        excluded: list[tuple[RetrievedChunk, str]] = []
        for r in results:
            is_authoritative, reason = classify_authority(r.chunk.metadata)
            if is_authoritative:
                authoritative.append(r)
            else:
                excluded.append((r, reason))
        return RetrievalResult(query=query, authoritative=authoritative, excluded=excluded)

    def retrieve_applicable(self, query: str, k: int = 8, context: Optional[dict] = None):
        """retrieve_authoritative() plus the applicability filter (see
        src/applicability.py): among authoritative results, excludes
        chunks whose stated scope (e.g. membership tier) contradicts the
        customer's context, and flags when the authoritative set can't
        be narrowed because the customer's context wasn't given. Kept as
        a thin wrapper (rather than folding into retrieve_authoritative)
        so retrieval and applicability stay independently testable."""
        from src.applicability import apply_applicability_filter  # local import avoids a hard dependency for callers that only need authority filtering

        authoritative_result = self.retrieve_authoritative(query, k=k)
        return apply_applicability_filter(authoritative_result.authoritative, query, context=context)