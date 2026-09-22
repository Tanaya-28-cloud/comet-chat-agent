"""
Applicability layer for the Aster & Row support agent.

Retrieval (retriever.py) answers "which documents are topically close to
this query, and are they authoritative?" That's necessary but not
sufficient — a document can be topically close AND authoritative AND
still not apply to this customer's actual situation. The TrailPlus
return-window policy is a real, active, official document about return
windows; it just doesn't apply to a customer who isn't a TrailPlus
member. Semantic similarity has no way to know that, because "TrailPlus
member" and "regular customer" are semantically close (both describe a
customer's membership status) even though they pick out disjoint groups
of people. That's a logical/eligibility relationship, not a distance
relationship, so it needs its own explicit layer.

Design: an "applicability dimension" is something a policy document can
be scoped to (e.g. membership tier). For each dimension we define two
small, independent extractors:

  - extract_from_chunk(chunk) -> the scope value a CHUNK commits to, or
    None if the chunk doesn't restrict itself on this dimension (i.e.
    it's general — applies regardless of the customer's value).
  - extract_from_query(query) -> the scope value the CUSTOMER's message
    asserts about themselves, or None if the message doesn't say.

Deliberately NOT here: any lookup keyed on the literal wording of a
specific evaluation question. Both extractors work off document
structure (headings, doc titles) and general phrasing patterns
(membership-tier vocabulary), so the same mechanism should hold up
against paraphrases and against dimensions/documents this file didn't
have in mind, provided a new dimension is registered the same way.

This module intentionally does NOT try to decide whether two
applicable-and-conflicting sources (e.g. the tumbler dishwasher
question) "really" disagree — both are general/unscoped on every
dimension we track, so neither gets filtered out. Recognizing that they
disagree is left for the answer-generation step, which can read the
actual text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from src.kb_indexer import Chunk
from src.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Dimension framework
# ---------------------------------------------------------------------------

@dataclass
class ApplicabilityDimension:
    name: str
    extract_from_chunk: Callable[[Chunk], Optional[str]]
    extract_from_query: Callable[[str], Optional[str]]


@dataclass
class ClarificationNeed:
    dimension: str
    candidate_values: list[str]
    reason: str


@dataclass
class ApplicabilityResult:
    query: str
    applicable: list[RetrievedChunk]
    inapplicable: list[tuple[RetrievedChunk, str]]         # (chunk, why it was excluded)
    clarification_needed: list[ClarificationNeed]            # context the agent should ask for


# ---------------------------------------------------------------------------
# Membership-tier dimension
# ---------------------------------------------------------------------------
# Fixed by real content inspection, not guessed:
#   - 01-returns-policy-current.md's tier-specific section has "Standard"
#     literally in its heading ("Standard return window"), but its BODY
#     TEXT also mentions "TrailPlus" as a cross-reference to the other
#     policy — so body text is NOT a safe signal, only the heading is.
#   - 09-trailplus-membership.md's relevant chunk heading is just "Return
#     window" (no tier word), but the DOCUMENT TITLE is "TrailPlus
#     Membership Benefits" — reliable at the document level.
#   - Every other chunk (damaged items, refunds, exclusions, etc.) has no
#     tier word in either heading or title -> correctly falls through to
#     None (general / applies regardless of tier).

_STANDARD_HEADING_RE = re.compile(r"\bstandard\b", re.IGNORECASE)
_TRAILPLUS_RE = re.compile(r"\btrail\s*plus\b", re.IGNORECASE)


def _infer_chunk_membership_tier(chunk: Chunk) -> Optional[str]:
    if _STANDARD_HEADING_RE.search(chunk.heading):
        return "standard"
    if _TRAILPLUS_RE.search(chunk.heading):
        return "trailplus"
    if _TRAILPLUS_RE.search(chunk.doc_title):
        return "trailplus"
    return None


# Query-side extraction needs to handle two things naive keyword matching
# gets wrong: (1) "regular"/"standard" customer phrasing that never says
# the word "standard" the doc uses, and (2) negation — "I don't have
# TrailPlus" mentions the word "trailplus" but means the opposite.
_QUERY_STANDARD_PHRASES = re.compile(
    r"\b(regular|standard|non[\s-]?member|not a member|no membership)\b", re.IGNORECASE
)
_QUERY_TRAILPLUS_NEGATED = re.compile(
    r"\b(not|n't|without|no|non|isn't|wasn't|don't have|didn't have|never had)\b"
    r"(?:\s+\w+){0,4}\s+trail\s*plus"
    r"|trail\s*plus(?:\s+\w+){0,4}\s+\b(not|n't|inactive|expired|isn't|wasn't|lapsed)\b",
    re.IGNORECASE,
)
_QUERY_TRAILPLUS_RE = re.compile(r"\btrail\s*plus\b", re.IGNORECASE)


def _infer_query_membership_tier(query: str) -> Optional[str]:
    if _QUERY_TRAILPLUS_RE.search(query):
        if _QUERY_TRAILPLUS_NEGATED.search(query):
            return "standard"
        return "trailplus"
    if _QUERY_STANDARD_PHRASES.search(query):
        return "standard"
    return None


MEMBERSHIP_TIER_DIMENSION = ApplicabilityDimension(
    name="membership_tier",
    extract_from_chunk=_infer_chunk_membership_tier,
    extract_from_query=_infer_query_membership_tier,
)

# Extension point: additional dimensions (e.g. shipping destination,
# final-sale status) can be added here the same way, each as an
# independent ApplicabilityDimension, without touching the filtering
# logic below.
DEFAULT_DIMENSIONS: list[ApplicabilityDimension] = [MEMBERSHIP_TIER_DIMENSION]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_applicability_filter(
    authoritative: list[RetrievedChunk],
    query: str,
    dimensions: list[ApplicabilityDimension] = None,
    context: Optional[dict[str, str]] = None,
) -> ApplicabilityResult:
    """Partitions an already-authoritative result set by whether each
    chunk's stated scope (if any) matches the customer's context.

    `context` lets a future multi-turn agent pass in a value it already
    learned earlier in the conversation (e.g. the customer confirmed
    TrailPlus membership two turns ago) — that takes priority over
    re-parsing the current message, so we don't lose context on a
    follow-up like "what about Canada?" that doesn't restate it.
    """
    dimensions = dimensions or DEFAULT_DIMENSIONS
    context = context or {}

    # chunk_scopes[dim.name][chunk_id] = the dimension value that chunk commits to (or None)
    chunk_scopes: dict[str, dict[str, Optional[str]]] = {dim.name: {} for dim in dimensions}
    for r in authoritative:
        for dim in dimensions:
            chunk_scopes[dim.name][r.chunk.chunk_id] = dim.extract_from_chunk(r.chunk)

    query_values: dict[str, Optional[str]] = {}
    for dim in dimensions:
        query_values[dim.name] = context.get(dim.name) or dim.extract_from_query(query)

    clarification_needed: list[ClarificationNeed] = []
    for dim in dimensions:
        distinct_values = {v for v in chunk_scopes[dim.name].values() if v is not None}
        if query_values[dim.name] is None and len(distinct_values) > 1:
            clarification_needed.append(
                ClarificationNeed(
                    dimension=dim.name,
                    candidate_values=sorted(distinct_values),
                    reason=(
                        f"Authoritative sources differ by {dim.name} "
                        f"({', '.join(sorted(distinct_values))}) and the customer's "
                        f"{dim.name} was not stated."
                    ),
                )
            )

    applicable: list[RetrievedChunk] = []
    inapplicable: list[tuple[RetrievedChunk, str]] = []
    for r in authoritative:
        mismatch_reason = None
        for dim in dimensions:
            chunk_value = chunk_scopes[dim.name][r.chunk.chunk_id]
            query_value = query_values[dim.name]
            if chunk_value is not None and query_value is not None and chunk_value != query_value:
                mismatch_reason = (
                    f"{dim.name}={chunk_value!r} does not match customer's "
                    f"{dim.name}={query_value!r}"
                )
                break
        if mismatch_reason:
            inapplicable.append((r, mismatch_reason))
        else:
            applicable.append(r)

    return ApplicabilityResult(
        query=query,
        applicable=applicable,
        inapplicable=inapplicable,
        clarification_needed=clarification_needed,
    )