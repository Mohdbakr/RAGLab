"""Word-window text chunking — the mechanics a RAG framework usually hides.

Splits on whitespace and groups words into fixed-size, overlapping
windows. Simple to reason about and test; it can split a sentence in
half, which a semantic/sentence-aware chunker wouldn't.
"""

from __future__ import annotations


def chunk_text(text: str, *, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """Split text into overlapping word-window chunks.

    Args:
        text: The text to split. Whitespace (including newlines/tabs) is
            collapsed to single spaces.
        chunk_size: Words per chunk. Must be positive.
        overlap: Words shared between consecutive chunks. Must be
            non-negative and strictly less than ``chunk_size``.

    Returns:
        The chunks, in order. Empty input yields an empty list.

    Raises:
        ValueError: If ``chunk_size`` isn't positive, ``overlap`` is
            negative, or ``overlap`` isn't smaller than ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks
