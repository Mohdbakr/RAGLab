"""Direct embedding calls — no shared client abstraction, no vector-DB service.

Goes straight through `litellm`, which is itself multi-provider, so this
stays model-agnostic without needing a Protocol/adapter layer on top —
that's worth introducing once a second project actually needs to share
one, not before.
"""

from __future__ import annotations

import litellm


async def embed_texts(
    texts: list[str], *, model: str = "text-embedding-3-small"
) -> list[list[float]]:
    """Embed a batch of texts.

    Args:
        texts: The texts to embed, in order.
        model: Any litellm-supported embedding model string.

    Returns:
        One vector per input text, same order. Empty list for empty input
        (no API call made).
    """
    if not texts:
        return []
    response = await litellm.aembedding(model=model, input=texts)
    return [item.embedding for item in response.data]
