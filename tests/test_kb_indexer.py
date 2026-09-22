from pathlib import Path

from src.kb_indexer import load_all_chunks, parse_front_matter


KB_DIR = Path(__file__).resolve().parents[1] / "knowledge-base"


def test_kb_contains_chunks():
    chunks = load_all_chunks(KB_DIR)

    assert len(chunks) > 0


def test_chunks_have_metadata_and_text():
    chunks = load_all_chunks(KB_DIR)

    for chunk in chunks:
        assert chunk.source_file
        assert chunk.doc_title
        assert chunk.text.strip()
        assert isinstance(chunk.metadata, dict)


def test_front_matter_dates_are_serializable():
    sample = """---
document_id: test-doc
status: active
published_at: 2026-01-01
policy_authority: authoritative
---

# Test

Hello.
"""

    metadata, body = parse_front_matter(sample)

    assert metadata["document_id"] == "test-doc"
    assert metadata["status"] == "active"
    assert isinstance(metadata["published_at"], str)
    assert body.strip()