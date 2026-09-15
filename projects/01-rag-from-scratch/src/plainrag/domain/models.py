"""Shared data types used across the domain, services, and CLI layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """One chunk of source text, embedded and ready to be searched.

    Attributes:
        text: The chunk's raw text.
        source: Where it came from (e.g. a file path), surfaced as a
            citation when the chunk is retrieved.
        embedding: Its embedding vector.
    """

    text: str
    source: str
    embedding: list[float]


@dataclass
class ScoredChunk:
    """A retrieved chunk paired with its similarity to the query."""

    chunk: DocumentChunk
    score: float


@dataclass
class AnswerResult:
    """An answer paired with the chunks it was grounded in.

    Attributes:
        answer: The generated answer text.
        sources: The retrieved chunks used to ground it, most relevant
            first — empty if the index had nothing to retrieve.
    """

    answer: str
    sources: list[ScoredChunk]
