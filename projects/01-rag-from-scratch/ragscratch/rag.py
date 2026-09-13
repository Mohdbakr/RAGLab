"""Ties chunking, embeddings, the index, and the LLM together.

The whole point of building this "from scratch": ingest and ask are just
a few calls to the other modules, no framework's pipeline abstraction
needed to see the shape of the flow.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragscratch.chunking import chunk_text
from ragscratch.embeddings import embed_texts
from ragscratch.index import CosineSimilarityIndex, DocumentChunk, ScoredChunk
from ragscratch.llm import answer_question


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


async def ingest_text(
    index: CosineSimilarityIndex,
    text: str,
    source: str,
    *,
    chunk_size: int = 200,
    overlap: int = 20,
    embedding_model: str = "text-embedding-3-small",
) -> int:
    """Chunk, embed, and add a document's text to the index.

    Args:
        index: The index to add chunks to.
        text: The document's raw text.
        source: A label for where it came from (e.g. a file path),
            attached to every chunk for later citation.
        chunk_size: Words per chunk, see :func:`ragscratch.chunking.chunk_text`.
        overlap: Words shared between consecutive chunks.
        embedding_model: Any litellm-supported embedding model string.

    Returns:
        The number of chunks added (0 if the text was empty).
    """
    texts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not texts:
        return 0
    vectors = await embed_texts(texts, model=embedding_model)
    index.add(
        [
            DocumentChunk(text=chunk, source=source, embedding=vector)
            for chunk, vector in zip(texts, vectors, strict=True)
        ]
    )
    return len(texts)


async def ask(
    index: CosineSimilarityIndex,
    question: str,
    *,
    k: int = 4,
    embedding_model: str = "text-embedding-3-small",
    chat_model: str = "gpt-4o-mini",
) -> AnswerResult:
    """Answer a question, grounded in whatever's been ingested so far.

    Args:
        index: The index to search.
        question: The question to answer.
        k: Maximum number of chunks to retrieve as context.
        embedding_model: Any litellm-supported embedding model string.
        chat_model: Any litellm-supported chat model string.

    Returns:
        The answer and the chunks it was grounded in.
    """
    (query_vector,) = await embed_texts([question], model=embedding_model)
    sources = index.search(query_vector, k=k)
    answer = await answer_question(question, sources, model=chat_model)
    return AnswerResult(answer=answer, sources=sources)
