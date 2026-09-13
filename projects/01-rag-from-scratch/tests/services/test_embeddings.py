"""Tests for plainrag.embeddings, written before the implementation.

litellm itself is never called — `litellm.aembedding` is monkeypatched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from plainrag.core.exceptions import EmbeddingError
from plainrag.services.embeddings import embed_texts


@dataclass
class FakeEmbeddingItem:
    embedding: list[float]


@dataclass
class FakeEmbeddingResponse:
    data: list[FakeEmbeddingItem]


class TestEmbedTexts:
    @pytest.mark.asyncio
    async def test_returns_one_vector_per_input_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_aembedding(**kwargs: Any) -> FakeEmbeddingResponse:
            captured.update(kwargs)
            return FakeEmbeddingResponse(
                data=[
                    FakeEmbeddingItem(embedding=[0.1, 0.2]),
                    FakeEmbeddingItem(embedding=[0.3, 0.4]),
                ]
            )

        monkeypatch.setattr("plainrag.services.embeddings.litellm.aembedding", fake_aembedding)

        vectors = await embed_texts(["hello", "world"], model="text-embedding-3-small")

        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert captured["model"] == "text-embedding-3-small"
        assert captured["input"] == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits_without_calling_litellm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_aembedding(**kwargs: Any) -> FakeEmbeddingResponse:
            raise AssertionError("should not be called for empty input")

        monkeypatch.setattr("plainrag.services.embeddings.litellm.aembedding", fake_aembedding)

        assert await embed_texts([]) == []

    @pytest.mark.asyncio
    async def test_wraps_a_litellm_failure_in_embedding_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def failing_aembedding(**kwargs: Any) -> FakeEmbeddingResponse:
            raise RuntimeError("connection refused")

        monkeypatch.setattr("plainrag.services.embeddings.litellm.aembedding", failing_aembedding)

        with pytest.raises(EmbeddingError, match="connection refused"):
            await embed_texts(["hello"])
