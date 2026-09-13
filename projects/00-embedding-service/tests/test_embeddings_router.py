"""Tests for the /embed router, written before the implementation.

Dependencies are overridden with fakes so no real model, litellm call, or
Docker daemon is touched.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from raglab_common import BenchmarkLogger

from app.core.client_cache import EmbeddingClientCache
from app.dependencies import get_benchmark_logger, get_client_cache, get_settings
from app.main import app


class FakeSettings:
    default_embedding_model = "local/all-mpnet-base-v2"


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), 0.0] for text in texts]


class FakeClientCache(EmbeddingClientCache):
    def __init__(self, client: FakeEmbeddingClient) -> None:
        super().__init__()
        self._fake_client = client
        self.requested_models: list[str] = []

    def get(self, model: str) -> FakeEmbeddingClient:
        self.requested_models.append(model)
        return self._fake_client


@pytest.fixture
def fake_client() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def fake_cache(fake_client: FakeEmbeddingClient) -> FakeClientCache:
    return FakeClientCache(fake_client)


@pytest.fixture
def client(fake_cache: FakeClientCache, tmp_path: Path) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_settings] = lambda: FakeSettings()
    app.dependency_overrides[get_client_cache] = lambda: fake_cache
    app.dependency_overrides[get_benchmark_logger] = lambda: BenchmarkLogger(
        project_id="00-embedding-service", output_dir=tmp_path
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestEmbedEndpoint:
    def test_uses_the_default_model_when_none_given(
        self, client: TestClient, fake_cache: FakeClientCache
    ) -> None:
        response = client.post("/embed", json={"texts": ["hello"]})

        assert response.status_code == 200
        assert response.json() == {
            "model": "local/all-mpnet-base-v2",
            "embeddings": [[5.0, 0.0]],
        }
        assert fake_cache.requested_models == ["local/all-mpnet-base-v2"]

    def test_uses_the_requested_model_when_given(
        self, client: TestClient, fake_cache: FakeClientCache
    ) -> None:
        response = client.post(
            "/embed", json={"texts": ["hi"], "model": "openai/text-embedding-3-small"}
        )

        assert response.status_code == 200
        assert response.json()["model"] == "openai/text-embedding-3-small"
        assert fake_cache.requested_models == ["openai/text-embedding-3-small"]

    def test_embeds_every_text_in_order(
        self, client: TestClient, fake_client: FakeEmbeddingClient
    ) -> None:
        client.post("/embed", json={"texts": ["a", "bb", "ccc"]})

        assert fake_client.calls == [["a", "bb", "ccc"]]

    def test_rejects_an_empty_text_list(self, client: TestClient) -> None:
        response = client.post("/embed", json={"texts": []})

        assert response.status_code == 422

    def test_logs_a_benchmark_event(self, client: TestClient, tmp_path: Path) -> None:
        client.post("/embed", json={"texts": ["a", "b"]})

        log_file = tmp_path / "00-embedding-service.jsonl"
        assert log_file.exists()
        (recorded,) = (json.loads(line) for line in log_file.read_text().splitlines())
        assert recorded["operation"] == "embed"
        assert recorded["retrieved_k"] == 2
        assert recorded["latency_ms"] >= 0.0


class TestHealthz:
    def test_returns_ok(self, client: TestClient) -> None:
        response = client.get("/healthz")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
