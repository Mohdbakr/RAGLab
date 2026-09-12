"""Caches built EmbeddingClient instances per model spec.

Building a client for a local sentence-transformers model loads real
weights into memory; rebuilding it on every request would be wasteful and
slow. API-backed clients are cheap to rebuild too, but caching them
uniformly keeps this one code path simple.
"""

from __future__ import annotations

from raglab_common import EmbeddingClient, build_embedding_client


class EmbeddingClientCache:
    """Builds an :class:`EmbeddingClient` per model spec, once."""

    def __init__(self) -> None:
        """Create an empty cache."""
        self._clients: dict[str, EmbeddingClient] = {}

    def get(self, model: str) -> EmbeddingClient:
        """Return the client for ``model``, building and caching it on first use.

        Args:
            model: A `build_embedding_client`-compatible model spec.

        Returns:
            The (possibly newly built) client for that spec.
        """
        if model not in self._clients:
            self._clients[model] = build_embedding_client(model)
        return self._clients[model]
