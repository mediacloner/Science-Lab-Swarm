"""Semantic text chunker for document ingestion."""

import logging

logger = logging.getLogger(__name__)


def chunk_document(document: dict, chunk_size: int = 512, chunk_overlap: int = 64) -> list[dict]:
    """Split a parsed document into overlapping chunks for embedding.

    Args:
        document: Parsed document dict from parser.py
        chunk_size: Target tokens per chunk (approximate using words / 0.75)
        chunk_overlap: Token overlap between consecutive chunks

    Returns:
        List of chunk dicts with: text, chunk_id, source, section, metadata
    """
    chunks = []

    # Prefer section-aware chunking if sections are available
    sections = document.get("sections", {})
    if sections and len(sections) > 1:
        chunks = _chunk_by_sections(document, sections, chunk_size, chunk_overlap)
    else:
        chunks = _chunk_flat(document, chunk_size, chunk_overlap)

    logger.info(f"Chunked {document['filename']} into {len(chunks)} chunks")
    return chunks


def _chunk_by_sections(document: dict, sections: dict, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Chunk text respecting section boundaries."""
    chunks = []
    chunk_idx = 0
    approx_word_limit = int(chunk_size * 0.75)  # rough tokens-to-words
    overlap_words = int(chunk_overlap * 0.75)

    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue

        words = section_text.split()
        start = 0

        while start < len(words):
            end = min(start + approx_word_limit, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "text": chunk_text,
                "chunk_id": f"{document['filename']}::chunk_{chunk_idx}",
                "source": document["filename"],
                "source_path": document["path"],
                "section": section_name,
                "metadata": {
                    "format": document["format"],
                    "chunk_index": chunk_idx,
                    "word_count": end - start,
                },
            })
            chunk_idx += 1
            start = end - overlap_words if end < len(words) else len(words)

    return chunks


def _chunk_flat(document: dict, chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Chunk text without section awareness (fallback)."""
    chunks = []
    text = document.get("text", "")
    words = text.split()
    approx_word_limit = int(chunk_size * 0.75)
    overlap_words = int(chunk_overlap * 0.75)

    chunk_idx = 0
    start = 0

    while start < len(words):
        end = min(start + approx_word_limit, len(words))
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "text": chunk_text,
            "chunk_id": f"{document['filename']}::chunk_{chunk_idx}",
            "source": document["filename"],
            "source_path": document["path"],
            "section": "unknown",
            "metadata": {
                "format": document["format"],
                "chunk_index": chunk_idx,
                "word_count": end - start,
            },
        })
        chunk_idx += 1
        start = end - overlap_words if end < len(words) else len(words)

    return chunks
