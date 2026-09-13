"""Prompt assembly (with citations) and a direct completion call.

Goes straight through `litellm` for the same reason `embeddings.py` does:
it's already multi-provider, so no shared client abstraction is needed
here yet.
"""

from __future__ import annotations

import litellm

from ragscratch.index import ScoredChunk

_SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the question using ONLY "
    "the numbered context passages below. Cite the passages you used with "
    "their bracketed number, e.g. [1]. If the passages don't contain the "
    "answer, say so plainly instead of guessing."
)


def build_context_block(chunks: list[ScoredChunk]) -> str:
    """Render retrieved chunks as a numbered, source-labeled context block.

    Args:
        chunks: The retrieved chunks, most relevant first.

    Returns:
        A block like ``"[1] (source: a.txt)\\n...\\n\\n[2] ..."``, or an
        empty string if there are no chunks.
    """
    blocks = [
        f"[{i}] (source: {scored.chunk.source})\n{scored.chunk.text}"
        for i, scored in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[ScoredChunk]) -> list[dict[str, str]]:
    """Assemble the chat messages for answering a question from retrieved context.

    Args:
        question: The user's question.
        chunks: The retrieved chunks to ground the answer in.

    Returns:
        A system + user message pair ready for a chat completion call.
    """
    context = build_context_block(chunks)
    user_content = (
        f"Context:\n{context}\n\nQuestion: {question}"
        if chunks
        else f"Question: {question}\n\n(No context passages were retrieved.)"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def answer_question(
    question: str, chunks: list[ScoredChunk], *, model: str = "gpt-4o-mini"
) -> str:
    """Answer a question, grounded in retrieved context, with citations.

    Args:
        question: The user's question.
        chunks: The retrieved chunks to ground the answer in.
        model: Any litellm-supported chat model string.

    Returns:
        The model's answer text, or an empty string if the model returned
        no text content.
    """
    messages = build_messages(question, chunks)
    response = await litellm.acompletion(model=model, messages=messages, temperature=0.0)
    return response.choices[0].message.content or ""
