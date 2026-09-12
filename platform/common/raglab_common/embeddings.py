"""Swappable embedding access, kept separate from :mod:`raglab_common.llm`.

Retrieval (embeddings) and generation (chat completion) are different
concerns with different providers worth comparing independently, so they
get their own client and their own config string rather than being
bundled into one "model" setting. A config string picks the
implementation:

- ``"local/<model-name>"`` loads a local sentence-transformers model —
  no API key, no network call per request, useful as a baseline and for
  fully offline runs.
- Anything else is treated as a litellm embedding model string (e.g.
  ``"openai/text-embedding-3-small"``, ``"cohere/embed-english-v3.0"``)
  and routes through litellm the same way :mod:`raglab_common.llm` does.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import litellm

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@runtime_checkable
class EmbeddingClient(Protocol):
    """The only interface RAGLab projects should depend on for embeddings."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: The texts to embed, in order.

        Returns:
            One vector per input text, same order.
        """
        ...


class LiteLLMEmbeddingClient:
    """Default API-backed :class:`EmbeddingClient`, backed by litellm."""

    def __init__(self, model: str, **default_params: Any) -> None:
        """Create a client for one embedding provider/model.

        Args:
            model: A litellm embedding model string, e.g.
                ``"openai/text-embedding-3-small"``.
            **default_params: Extra parameters applied to every call,
                overridable per call.
        """
        self._model = model
        self._default_params = default_params

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """See :meth:`EmbeddingClient.embed`."""
        response = await litellm.aembedding(model=self._model, input=texts, **self._default_params)
        return [item.embedding for item in response.data]


class SentenceTransformerEmbeddingClient:
    """Local, offline :class:`EmbeddingClient` — no API key, no network per call.

    Requires the ``local-embeddings`` extra (``sentence-transformers``);
    the import is deferred to construction time so the base package stays
    light for projects that only use API-backed embeddings.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2") -> None:
        """Load a local sentence-transformers model.

        Args:
            model_name: Any model name `sentence_transformers` accepts.
        """
        from sentence_transformers import SentenceTransformer

        self._model: SentenceTransformer = SentenceTransformer(model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """See :meth:`EmbeddingClient.embed`.

        Runs the (synchronous, CPU/GPU-bound) local model in a worker
        thread so it doesn't block the event loop.
        """
        vectors = await asyncio.to_thread(self._model.encode, texts)
        return [list(vector) for vector in vectors]


def build_embedding_client(spec: str, **kwargs: Any) -> EmbeddingClient:
    """Build an :class:`EmbeddingClient` from a config string.

    Args:
        spec: ``"local/<model-name>"`` for a local sentence-transformers
            model, or any litellm embedding model string otherwise.
        **kwargs: Forwarded to the underlying client's constructor.

    Returns:
        A ready-to-use `EmbeddingClient`.
    """
    if spec.startswith("local/"):
        return SentenceTransformerEmbeddingClient(spec.removeprefix("local/"), **kwargs)
    return LiteLLMEmbeddingClient(spec, **kwargs)
