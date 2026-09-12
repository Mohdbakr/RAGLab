"""Swappable embedding access, kept separate from :mod:`raglab_common.llm`.

Retrieval (embeddings) and generation (chat completion) are different
concerns with different providers worth comparing independently, so they
get their own client and their own config string rather than being
bundled into one "model" setting. A config string picks the
implementation:

- ``"local/<model-name>"`` loads a local sentence-transformers model —
  no API key, no network call per request, useful as a baseline and for
  fully offline runs.
- ``"http://..."``/``"https://..."`` calls a standalone embedding
  microservice (see ``projects/00-embedding-service``) instead of
  embedding in-process — the production-microservice pattern, opt-in per
  project rather than a shared dependency everything requires.
- Anything else is treated as a litellm embedding model string (e.g.
  ``"openai/text-embedding-3-small"``, ``"cohere/embed-english-v3.0"``)
  and routes through litellm the same way :mod:`raglab_common.llm` does.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx
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


class HTTPEmbeddingClient:
    """:class:`EmbeddingClient` backed by a standalone embedding microservice.

    Calls ``POST {base_url}/embed`` (see ``projects/00-embedding-service``)
    instead of embedding in-process — useful for demonstrating/benchmarking
    the production pattern of a shared, independently-scaled embedding
    service, without making every project depend on it to function.
    """

    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Create a client for one embedding-service instance.

        Args:
            base_url: Base URL of the embedding service, e.g.
                ``"http://localhost:9100"``.
            model: Model to request from the service. Omit to use
                whatever default the service itself is configured with.
            client: An `httpx.AsyncClient` to reuse. If omitted, a
                short-lived client is created per call.
            timeout: Seconds to wait for a response, used only when no
                `client` is supplied.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = client
        self._timeout = timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """See :meth:`EmbeddingClient.embed`."""
        body: dict[str, Any] = {"texts": texts}
        if self._model is not None:
            body["model"] = self._model

        client = self._client or httpx.AsyncClient()
        owns_client = self._client is None
        try:
            response = await client.post(
                f"{self._base_url}/embed", json=body, timeout=self._timeout
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
            return list(embeddings)
        finally:
            if owns_client:
                await client.aclose()


def build_embedding_client(spec: str, **kwargs: Any) -> EmbeddingClient:
    """Build an :class:`EmbeddingClient` from a config string.

    Args:
        spec: ``"local/<model-name>"`` for a local sentence-transformers
            model, ``"http://..."``/``"https://..."`` for a standalone
            embedding microservice, or any litellm embedding model string
            otherwise.
        **kwargs: Forwarded to the underlying client's constructor.

    Returns:
        A ready-to-use `EmbeddingClient`.
    """
    if spec.startswith("local/"):
        return SentenceTransformerEmbeddingClient(spec.removeprefix("local/"), **kwargs)
    if spec.startswith("http://") or spec.startswith("https://"):
        return HTTPEmbeddingClient(spec, **kwargs)
    return LiteLLMEmbeddingClient(spec, **kwargs)
