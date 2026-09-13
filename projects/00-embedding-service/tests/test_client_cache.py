"""Tests for app.core.client_cache, written before the implementation."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.client_cache import EmbeddingClientCache


class FakeEmbeddingClient:
    def __init__(self, model: str) -> None:
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def test_builds_a_client_on_first_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_build(spec: str, **kwargs: Any) -> FakeEmbeddingClient:
        calls.append(spec)
        return FakeEmbeddingClient(spec)

    monkeypatch.setattr("app.core.client_cache.build_embedding_client", fake_build)
    cache = EmbeddingClientCache()

    client = cache.get("local/all-mpnet-base-v2")

    assert isinstance(client, FakeEmbeddingClient)
    assert calls == ["local/all-mpnet-base-v2"]


def test_reuses_the_same_client_for_the_same_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_build(spec: str, **kwargs: Any) -> FakeEmbeddingClient:
        calls.append(spec)
        return FakeEmbeddingClient(spec)

    monkeypatch.setattr("app.core.client_cache.build_embedding_client", fake_build)
    cache = EmbeddingClientCache()

    first = cache.get("local/all-mpnet-base-v2")
    second = cache.get("local/all-mpnet-base-v2")

    assert first is second
    assert calls == ["local/all-mpnet-base-v2"]  # built only once


def test_builds_a_separate_client_per_distinct_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build(spec: str, **kwargs: Any) -> FakeEmbeddingClient:
        return FakeEmbeddingClient(spec)

    monkeypatch.setattr("app.core.client_cache.build_embedding_client", fake_build)
    cache = EmbeddingClientCache()

    local_client = cache.get("local/all-mpnet-base-v2")
    openai_client = cache.get("openai/text-embedding-3-small")

    assert local_client is not openai_client
    assert local_client.model == "local/all-mpnet-base-v2"  # type: ignore[attr-defined]
    assert openai_client.model == "openai/text-embedding-3-small"  # type: ignore[attr-defined]
