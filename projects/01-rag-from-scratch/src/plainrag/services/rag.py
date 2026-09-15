"""Ties chunking, embeddings, the index, and the LLM together.

The whole point of building this "from scratch": ingest and ask are just
a few calls to the other modules, no framework's pipeline abstraction
needed to see the shape of the flow.
"""

from __future__ import annotations

from collections.abc import Callable

from plainrag.core.exceptions import EmptyIndexError
from plainrag.domain.chunking import chunk_text
from plainrag.domain.models import AnswerResult, DocumentChunk, ScoredChunk
from plainrag.services.embeddings import embed_texts
from plainrag.services.index import CosineSimilarityIndex
from plainrag.services.llm import answer_question


async def ingest_text(
    index: CosineSimilarityIndex,
    text: str,
    source: str,
    *,
    chunk_size: int = 200,
    overlap: int = 20,
    embedding_model: str = "text-embedding-3-small",
    on_step: Callable[[str], None] | None = None,
) -> int:
    """Chunk, embed, and add a document's text to the index.

    Args:
        index: The index to add chunks to.
        text: The document's raw text.
        source: A label for where it came from (e.g. a file path),
            attached to every chunk for later citation.
        chunk_size: Words per chunk, see :func:`plainrag.domain.chunking.chunk_text`.
        overlap: Words shared between consecutive chunks.
        embedding_model: Any litellm-supported embedding model string.
        on_step: Optional callback invoked with a short progress message
            after each pipeline step (chunk, embed, index).

    Returns:
        The number of chunks added (0 if the text was empty).
    """
    texts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not texts:
        return 0
    if on_step:
        on_step(f"chunked into {len(texts)} piece(s)")

    vectors = await embed_texts(texts, model=embedding_model)
    if on_step:
        on_step(f"embedded {len(texts)} chunk(s) via {embedding_model}")

    index.add(
        [
            DocumentChunk(text=chunk, source=source, embedding=vector)
            for chunk, vector in zip(texts, vectors, strict=True)
        ]
    )
    if on_step:
        on_step(f"added to index ({len(index)} chunk(s) total)")
    return len(texts)


async def retrieve(
    index: CosineSimilarityIndex,
    question: str,
    *,
    k: int = 4,
    embedding_model: str = "text-embedding-3-small",
    on_step: Callable[[str], None] | None = None,
) -> list[ScoredChunk]:
    """Embed a query and retrieve its ``k`` most similar chunks — no LLM call.

    Args:
        index: The index to search.
        question: The query text.
        k: Maximum number of chunks to retrieve.
        embedding_model: Any litellm-supported embedding model string.
        on_step: Optional callback invoked with a short progress message
            after each pipeline step (embed, retrieve).

    Returns:
        Up to ``k`` chunks, most similar first.

    Raises:
        EmptyIndexError: If the index has nothing in it yet.
    """
    if len(index) == 0:
        raise EmptyIndexError("The index is empty — ingest something before searching.")

    (query_vector,) = await embed_texts([question], model=embedding_model)
    if on_step:
        on_step(f"embedded question via {embedding_model}")

    sources = index.search(query_vector, k=k)
    if on_step:
        scores = ", ".join(f"{s.score:.2f}" for s in sources)
        on_step(f"retrieved {len(sources)} of {len(index)} chunk(s) (scores: {scores})")
    return sources


async def ask(
    index: CosineSimilarityIndex,
    question: str,
    *,
    k: int = 4,
    embedding_model: str = "text-embedding-3-small",
    chat_model: str = "gpt-4o-mini",
    on_step: Callable[[str], None] | None = None,
) -> AnswerResult:
    """Answer a question, grounded in whatever's been ingested so far.

    Args:
        index: The index to search.
        question: The question to answer.
        k: Maximum number of chunks to retrieve as context.
        embedding_model: Any litellm-supported embedding model string.
        chat_model: Any litellm-supported chat model string.
        on_step: Optional callback invoked with a short progress message
            after each pipeline step (embed, retrieve, generate).

    Returns:
        The answer and the chunks it was grounded in.

    Raises:
        EmptyIndexError: If the index has nothing in it yet.
    """
    sources = await retrieve(index, question, k=k, embedding_model=embedding_model, on_step=on_step)
    if on_step:
        on_step(f"generating answer via {chat_model}")
    answer = await answer_question(question, sources, model=chat_model)
    return AnswerResult(answer=answer, sources=sources)
