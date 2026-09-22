"""
Knowledge-base indexer for the Aster & Row support agent.

What this does, in order:
  1. Reads every .md file in knowledge-base/.
  2. Parses the YAML front matter (document_id, status, policy_authority,
     supersedes, etc.) — this metadata is what the retrieval/conflict logic
     will lean on later, so we keep every field, not just the ones we
     happen to use today.
  3. Splits the body into chunks along "## " headings. Each chunk carries
     a copy of its document's front matter plus the heading it fell under.
     Chunking by heading (rather than by fixed token count) keeps each
     chunk topically coherent, which matters for citation quality — a
     citation like "01-returns-policy-current.md, Standard return window"
     is only possible if a chunk boundary lines up with a heading.
  4. Embeds every chunk with a local sentence-transformers model and caches
     the result to disk, so we don't recompute embeddings on every run.

Nothing here talks to an LLM. This module's only job is turning markdown
files into a searchable, metadata-rich index.
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge-base"
INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str          # e.g. "01-returns-policy-current.md::Standard return window"
    source_file: str       # e.g. "01-returns-policy-current.md"
    heading: str            # e.g. "Standard return window" (empty string for pre-heading intro text)
    doc_title: str          # H1 title of the document
    text: str                # the chunk's actual content (heading text NOT duplicated in here)
    metadata: dict[str, Any]  # full YAML front matter, unmodified


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """Split a markdown file into (front_matter_dict, body_text).

    Raises ValueError if the file has no front matter block — every
    knowledge-base document is expected to have one, so a missing block
    is treated as a data problem worth surfacing loudly rather than
    silently indexing a document with no metadata.
    """
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        raise ValueError("Document is missing a YAML front matter block")
    front_matter_raw, body = match.groups()
    metadata = yaml.safe_load(front_matter_raw) or {}
    metadata = _stringify_dates(metadata)
    return metadata, body


def _stringify_dates(value: Any) -> Any:
    """Recursively convert date/datetime objects to ISO strings.

    PyYAML auto-parses unquoted dates like `effective_date: 2026-04-01`
    into datetime.date objects. That's convenient for comparisons but not
    JSON-serializable, and it also means the *type* of a field silently
    depends on formatting choices in the source markdown. Normalizing to
    plain ISO strings here keeps the on-disk index format simple and
    keeps every downstream consumer dealing with one type (str) instead
    of guessing whether a given field came through as a date or a string.
    """
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _stringify_dates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_dates(v) for v in value]
    return value


def extract_title(body: str) -> str:
    """Pull the H1 (# Title) line out of the body, if present."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            return stripped[2:].strip()
    return ""


def split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split body text into (heading, section_text) pairs on '## ' boundaries.

    Text before the first '## ' heading (typically the H1 title and any
    lead-in / blockquote, e.g. the "superseded" notice in the legacy
    returns doc) is kept as its own section with an empty heading, since
    that lead-in text can itself be meaningful (status warnings, etc.).
    """
    headings = list(HEADING_RE.finditer(body))
    sections: list[tuple[str, str]] = []

    first_start = headings[0].start() if headings else len(body)
    intro_raw = body[:first_start]
    # Drop the bare H1 title line — it's already captured in doc_title and
    # carries no content of its own. Anything else before the first '##'
    # (e.g. a supersession blockquote) is kept as a real intro chunk.
    intro_lines = [
        line for line in intro_raw.splitlines()
        if not (line.strip().startswith("# ") and not line.strip().startswith("##"))
    ]
    intro = "\n".join(intro_lines).strip()
    if intro:
        sections.append(("", intro))

    for i, h in enumerate(headings):
        heading_text = h.group(1).strip()
        section_start = h.end()
        section_end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        section_text = body[section_start:section_end].strip()
        sections.append((heading_text, section_text))

    return sections


def chunk_file(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    metadata, body = parse_front_matter(raw)
    doc_title = extract_title(body)
    sections = split_into_sections(body)

    chunks: list[Chunk] = []
    for heading, text in sections:
        if not text:
            continue
        chunk_id = f"{path.name}::{heading or '(intro)'}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_file=path.name,
                heading=heading,
                doc_title=doc_title,
                text=text,
                metadata=metadata,
            )
        )
    return chunks


def load_all_chunks(kb_dir: Path = KB_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(chunk_file(path))
    return chunks


# ---------------------------------------------------------------------------
# Embedding + persistence
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[Chunk], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Embed each chunk as '<heading>\\n<text>' so the heading contributes
    to the vector, not just the body — useful since some chunk bodies are
    short and heading-light on their own (e.g. a single sentence)."""
    from sentence_transformers import SentenceTransformer  # imported lazily: slow import, only needed here

    model = SentenceTransformer(model_name)
    texts = [f"{c.heading}\n{c.text}" if c.heading else c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def build_index(kb_dir: Path = KB_DIR, index_dir: Path = INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_all_chunks(kb_dir)
    if not chunks:
        raise RuntimeError(f"No chunks produced from {kb_dir} — is the directory populated?")

    embeddings = embed_chunks(chunks)

    chunks_path = index_dir / "chunks.json"
    embeddings_path = index_dir / "embeddings.npy"

    chunks_path.write_text(
        json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(embeddings_path, embeddings)

    print(f"Indexed {len(chunks)} chunks from {len(list(kb_dir.glob('*.md')))} documents")
    print(f"  -> {chunks_path}")
    print(f"  -> {embeddings_path}  (shape={embeddings.shape})")


def load_index(index_dir: Path = INDEX_DIR) -> tuple[list[Chunk], np.ndarray]:
    chunks_path = index_dir / "chunks.json"
    embeddings_path = index_dir / "embeddings.npy"
    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"No index found in {index_dir}. Run `python -m src.kb_indexer` to build it."
        )
    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in raw_chunks]
    embeddings = np.load(embeddings_path)
    return chunks, embeddings


if __name__ == "__main__":
    build_index()
