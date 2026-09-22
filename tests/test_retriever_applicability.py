"""
Unit tests for src/retriever.py's authority filter and src/applicability.py.

Two kinds of test here, deliberately kept separate:

1. Deterministic filter-logic tests (test_classify_authority_*,
   test_*_applicability_filter*) use hand-engineered RetrievedChunk
   objects with fixed scores. These need no embedding model or
   internet access — they test the FILTERING LOGIC itself, independent
   of embedding quality, so they run anywhere (including CI with no
   network) and can never fail because of a retrieval-quality drift.

2. Real end-to-end tests (test_live_*) call Retriever.from_index() and
   actually embed a real query against the real index. These need the
   embedding model to be downloaded (first run only — sentence-
   transformers caches it locally after that, no ongoing API cost) and
   the index already built (`python -m src.kb_indexer`). These validate
   real retrieval quality, not just the filter logic — this is the one
   thing that could not be verified during initial development in an
   environment without internet access to the embedding model.

If you're running in an environment without internet access, the
test_live_* tests will fail on the first model download — that's an
environment limitation, not a code bug. The deterministic tests above
them do not depend on this.
"""

from src.kb_indexer import load_all_chunks
from src.retriever import classify_authority, Retriever, RetrievedChunk
from src.applicability import apply_applicability_filter


CHUNKS = load_all_chunks()
BY_ID = {c.chunk_id: c for c in CHUNKS}


# ---------------------------------------------------------------------------
# Deterministic — classify_authority against real front matter, no embeddings
# ---------------------------------------------------------------------------

def test_legacy_and_internal_docs_are_non_authoritative():
    non_authoritative_files = {
        "02-returns-policy-legacy.md",
        "13-support-escalation.md",
        "14-internal-content-migration-notes.md",
    }
    for c in CHUNKS:
        is_auth, reason = classify_authority(c.metadata)
        if c.source_file in non_authoritative_files:
            assert is_auth is False, f"{c.chunk_id} should be non-authoritative but wasn't"
            assert reason is not None


def test_current_policy_docs_are_authoritative():
    authoritative_files = {
        "01-returns-policy-current.md",
        "03-final-sale-and-promotions.md",
        "09-trailplus-membership.md",
        "11-product-care.md",
        "12-breeze-tumbler-product-card.md",
    }
    for c in CHUNKS:
        if c.source_file in authoritative_files:
            is_auth, _ = classify_authority(c.metadata)
            assert is_auth is True, f"{c.chunk_id} should be authoritative but wasn't"


# ---------------------------------------------------------------------------
# Deterministic — applicability filter logic, hand-engineered scores
# ---------------------------------------------------------------------------

def _rc(chunk_id, score):
    return RetrievedChunk(chunk=BY_ID[chunk_id], score=score)


def test_regular_customer_query_excludes_trailplus_even_when_it_scores_higher():
    """Regression test for Bug Diary #5, using the exact real scores
    reported from a live Colab run: the TrailPlus chunk scored highest
    (0.578) but must still be excluded as inapplicable to a query about
    a 'regular customer'."""
    authoritative = [
        _rc("09-trailplus-membership.md::Return window", 0.578),
        _rc("01-returns-policy-current.md::Standard return window", 0.508),
        _rc("04-damaged-or-wrong-items.md::Reporting window", 0.478),
    ]
    result = apply_applicability_filter(authoritative, "What is the return window for a regular customer?")

    applicable_ids = [r.chunk.chunk_id for r in result.applicable]
    inapplicable_ids = [r.chunk.chunk_id for r, _reason in result.inapplicable]

    assert applicable_ids[0] == "01-returns-policy-current.md::Standard return window"
    assert "09-trailplus-membership.md::Return window" in inapplicable_ids


def test_trailplus_query_excludes_standard_policy():
    authoritative = [
        _rc("09-trailplus-membership.md::Return window", 0.60),
        _rc("01-returns-policy-current.md::Standard return window", 0.55),
    ]
    result = apply_applicability_filter(authoritative, "What is the return period for a TrailPlus member?")

    applicable_ids = [r.chunk.chunk_id for r in result.applicable]
    inapplicable_ids = [r.chunk.chunk_id for r, _reason in result.inapplicable]

    assert applicable_ids[0] == "09-trailplus-membership.md::Return window"
    assert "01-returns-policy-current.md::Standard return window" in inapplicable_ids


def test_negated_trailplus_phrasing_resolves_to_standard():
    """'I don't have a TrailPlus membership' mentions the word 'trailplus'
    but means the opposite -- must resolve to the standard policy, not
    the TrailPlus one."""
    authoritative = [
        _rc("09-trailplus-membership.md::Return window", 0.55),
        _rc("01-returns-policy-current.md::Standard return window", 0.50),
    ]
    result = apply_applicability_filter(
        authoritative, "I don't have a TrailPlus membership -- how many days do I get to send something back?"
    )
    applicable_ids = [r.chunk.chunk_id for r in result.applicable]
    assert "01-returns-policy-current.md::Standard return window" in applicable_ids
    assert "09-trailplus-membership.md::Return window" not in applicable_ids


def test_dishwasher_conflict_both_sides_survive_applicability_filter():
    """Neither side of the genuine Breeze Tumbler care conflict has a
    membership-tier scope, so neither should be filtered out by the
    applicability layer -- resolving the conflict is a job for the
    agent/answer-generation layer, not retrieval."""
    authoritative = [
        _rc("12-breeze-tumbler-product-card.md::Cleaning", 0.71),
        _rc("11-product-care.md::Breeze Tumbler", 0.69),
    ]
    result = apply_applicability_filter(authoritative, "Can I put the entire Breeze Tumbler in the dishwasher?")
    assert len(result.applicable) == 2
    assert result.inapplicable == []


def test_ambiguous_query_flags_clarification_instead_of_guessing():
    authoritative = [
        _rc("09-trailplus-membership.md::Return window", 0.60),
        _rc("01-returns-policy-current.md::Standard return window", 0.58),
    ]
    result = apply_applicability_filter(authoritative, "What is your return window?")
    assert len(result.clarification_needed) == 1
    assert result.clarification_needed[0].dimension == "membership_tier"
    assert set(result.clarification_needed[0].candidate_values) == {"standard", "trailplus"}
    assert result.inapplicable == []


# ---------------------------------------------------------------------------
# Live — real embedding model + real index. Needs internet on first run
# (to download the model; cached locally after) and a built index/.
# ---------------------------------------------------------------------------

def test_live_authority_filter_excludes_legacy_and_internal_docs():
    retriever = Retriever.from_index()
    result = retriever.retrieve_authoritative("What is the return policy?")

    authoritative_files = {r.chunk.source_file for r in result.authoritative}
    assert "02-returns-policy-legacy.md" not in authoritative_files
    assert "14-internal-content-migration-notes.md" not in authoritative_files
    assert "13-support-escalation.md" not in authoritative_files


def test_live_regular_customer_does_not_use_trailplus_policy():
    query = "I am a regular customer. What is my return window?"
    retriever = Retriever.from_index()

    authoritative = retriever.retrieve_authoritative(query).authoritative
    filtered = apply_applicability_filter(authoritative, query)

    applicable_files = {r.chunk.source_file for r in filtered.applicable}

    # The real embedding model may or may not even surface the TrailPlus
    # chunk in the top-k for this query -- either outcome is fine. What
    # must NOT happen is the TrailPlus policy ending up in the
    # *applicable* set for a self-identified regular customer.
    assert "09-trailplus-membership.md" not in applicable_files


def test_live_retrieval_returns_citable_chunks():
    retriever = Retriever.from_index()
    result = retriever.retrieve_authoritative("Can I put the Breeze Tumbler in the dishwasher?")

    assert len(result.authoritative) > 0
    for r in result.authoritative:
        assert r.chunk.source_file.endswith(".md")
        assert isinstance(r.chunk.heading, str)