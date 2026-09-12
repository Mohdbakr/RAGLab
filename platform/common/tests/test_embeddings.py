"""Tests for raglab_common.embeddings, written before the implementation.

Two things are faked so these stay fast, free, and offline:
- `litellm.aembedding`, for the API-backed client.
- the `sentence_transformers` package itself (injected into `sys.modules`
  before import), so the local client's logic is tested without needing
  the real (heavy) dependency installed or downloading a model.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from raglab_common.embeddings import (
    EmbeddingClient,
    HTTPEmbeddingClient,
    LiteLLMEmbeddingClient,
    SentenceTransformerEmbeddingClient,
    build_embedding_client,
)


@dataclass
class FakeEmbeddingItem:
    embedding: list[float]


@dataclass
class FakeEmbeddingResponse:
    data: list[FakeEmbeddingItem]


@dataclass
class FakeSentenceTransformer:
    """Stand-in injected as sentence_transformers.SentenceTransformer."""

    model_name: str
    encode_calls: list[list[str]] = field(default_factory=list)

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.encode_calls.append(list(texts))
        return [[float(len(text)), 0.0] for text in texts]


@pytest.fixture
def fake_sentence_transformers_module(monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a fake `sentence_transformers` module and return the fake class."""
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return FakeSentenceTransformer


class TestLiteLLMEmbeddingClient:
    def test_satisfies_the_protocol(self) -> None:
        client = LiteLLMEmbeddingClient(model="openai/text-embedding-3-small")
        assert isinstance(client, EmbeddingClient)

    @pytest.mark.asyncio
    async def test_embed_returns_one_vector_per_input(
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

        monkeypatch.setattr("raglab_common.embeddings.litellm.aembedding", fake_aembedding)
        client = LiteLLMEmbeddingClient(model="openai/text-embedding-3-small")

        vectors = await client.embed(["hello", "world"])

        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert captured["model"] == "openai/text-embedding-3-small"
        assert captured["input"] == ["hello", "world"]


class TestSentenceTransformerEmbeddingClient:
    def test_lazily_constructs_the_underlying_model(
        self, fake_sentence_transformers_module: type
    ) -> None:
        client = SentenceTransformerEmbeddingClient(model_name="all-mpnet-base-v2")

        assert isinstance(client, EmbeddingClient)
        assert client._model.model_name == "all-mpnet-base-v2"

    @pytest.mark.asyncio
    async def test_embed_delegates_to_the_local_models_encode(
        self, fake_sentence_transformers_module: type
    ) -> None:
        client = SentenceTransformerEmbeddingClient(model_name="all-mpnet-base-v2")

        vectors = await client.embed(["hi", "there!"])

        assert vectors == [[2.0, 0.0], [6.0, 0.0]]
        assert client._model.encode_calls == [["hi", "there!"]]


class TestHTTPEmbeddingClient:
    def test_satisfies_the_protocol(self) -> None:
        client = HTTPEmbeddingClient(base_url="http://localhost:9100")
        assert isinstance(client, EmbeddingClient)

    @pytest.mark.asyncio
    async def test_embed_posts_texts_and_returns_the_vectors(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"model": "local/all-mpnet-base-v2", "embeddings": [[0.1, 0.2]]},
            )

        client = HTTPEmbeddingClient(
            base_url="http://localhost:9100",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        vectors = await client.embed(["hello"])

        assert vectors == [[0.1, 0.2]]
        assert captured["url"] == "http://localhost:9100/embed"
        assert captured["body"] == {"texts": ["hello"]}

    @pytest.mark.asyncio
    async def test_embed_includes_the_model_override_when_set(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"model": "openai/text-embedding-3-small", "embeddings": [[0.1]]},
            )

        client = HTTPEmbeddingClient(
            base_url="http://localhost:9100",
            model="openai/text-embedding-3-small",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        await client.embed(["hi"])

        assert captured["body"] == {"texts": ["hi"], "model": "openai/text-embedding-3-small"}

    @pytest.mark.asyncio
    async def test_raises_on_a_server_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        client = HTTPEmbeddingClient(
            base_url="http://localhost:9100",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.embed(["hi"])


class TestBuildEmbeddingClient:
    def test_local_prefix_builds_a_sentence_transformer_client(
        self, fake_sentence_transformers_module: type
    ) -> None:
        client = build_embedding_client("local/all-mpnet-base-v2")

        assert isinstance(client, SentenceTransformerEmbeddingClient)
        assert client._model.model_name == "all-mpnet-base-v2"

    def test_anything_else_builds_a_lite_llm_client(self) -> None:
        client = build_embedding_client("openai/text-embedding-3-small")

        assert isinstance(client, LiteLLMEmbeddingClient)
        assert client._model == "openai/text-embedding-3-small"

    def test_http_prefix_builds_an_http_client(self) -> None:
        client = build_embedding_client("http://localhost:9100")

        assert isinstance(client, HTTPEmbeddingClient)
        assert client._base_url == "http://localhost:9100"

    def test_https_prefix_also_builds_an_http_client(self) -> None:
        client = build_embedding_client("https://embeddings.example.com")

        assert isinstance(client, HTTPEmbeddingClient)
