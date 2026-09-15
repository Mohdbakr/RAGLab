"""A flat, brute-force cosine-similarity index — no vector-database service.

O(n) per query: fine for a few thousand chunks, and exactly the tradeoff
that makes an ANN-backed vector database (see other RAGLab projects) a
separate concern once scale actually demands it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from plainrag.core.exceptions import IndexPersistenceError
from plainrag.domain.models import DocumentChunk, ScoredChunk


class CosineSimilarityIndex:
    """An in-memory, JSON-persistable store of embedded chunks."""

    def __init__(self) -> None:
        """Create an empty index."""
        self._chunks: list[DocumentChunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def __repr__(self) -> str:
        dim = len(self._chunks[0].embedding) if self._chunks else None
        return f"CosineSimilarityIndex(chunks={len(self._chunks)}, dim={dim})"

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to the index.

        Args:
            chunks: The chunks to add, already embedded.
        """
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], *, k: int = 4) -> list[ScoredChunk]:
        """Find the ``k`` most similar chunks to a query embedding.

        Args:
            query_embedding: The query's embedding vector.
            k: Maximum number of results.

        Returns:
            Up to ``k`` chunks, most similar first. Empty if the index has
            no chunks.

        Raises:
            ValueError: If ``k`` isn't positive, or ``query_embedding``'s
                dimension doesn't match the chunks already in the index.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not self._chunks:
            return []

        expected_dim = len(self._chunks[0].embedding)
        if len(query_embedding) != expected_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {expected_dim}, "
                f"got {len(query_embedding)}"
            )

        matrix = np.array([c.embedding for c in self._chunks], dtype=np.float64)
        query = np.array(query_embedding, dtype=np.float64)

        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query)
        similarities = (matrix @ query) / (matrix_norms * query_norm + 1e-10)

        top_indices = np.argsort(-similarities)[:k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(similarities[i])) for i in top_indices
        ]

    def save(self, path: Path) -> None:
        """Persist the index to a JSON file.

        Args:
            path: Where to write it.

        Raises:
            IndexPersistenceError: If the file can't be written.
        """
        try:
            path.write_text(json.dumps([asdict(c) for c in self._chunks]))
        except OSError as e:
            raise IndexPersistenceError(f"Could not save index to {path}: {e}") from e

    @classmethod
    def load(cls, path: Path) -> CosineSimilarityIndex:
        """Load an index previously written by :meth:`save`.

        Args:
            path: The file to load.

        Returns:
            The loaded index, or an empty one if ``path`` doesn't exist.

        Raises:
            IndexPersistenceError: If ``path`` exists but can't be read
                or contains malformed data.
        """
        index = cls()
        if not path.exists():
            return index
        try:
            data = json.loads(path.read_text())
            index._chunks = [DocumentChunk(**item) for item in data]
        except (OSError, json.JSONDecodeError, TypeError, KeyError) as e:
            raise IndexPersistenceError(f"Could not load index from {path}: {e}") from e
        return index
