"""Tests for the backends/vectorstores routers, written before the implementation.

Dependencies are overridden with fakes so no real catalog file, Docker
daemon, or health endpoint is touched.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.catalog.models import BackendSpec, StandaloneServiceSpec
from app.dependencies import (
    get_backends,
    get_embedding_services,
    get_lifecycle_service,
    get_vectorstores,
)
from app.main import app
from app.models.status import ComponentState
from app.runtime.docker_compose_runtime import DockerComposeCommandError


def backend(**overrides: object) -> BackendSpec:
    fields: dict[str, object] = {
        "id": "03-production-rag-reference",
        "name": "Production RAG Reference",
        "level": "intermediate",
        "summary": "x",
        "path": "projects/03-production-rag-reference",
        "needs_container": True,
        "compose_file": "docker-compose.yml",
        "base_url": "http://localhost:9003",
        "health_path": "/healthz",
        "compatible_vector_stores": ["chroma"],
        "compatible_embedding_services": ["00-embedding-service"],
    }
    fields.update(overrides)
    return BackendSpec(**fields)  # type: ignore[arg-type]


def vector_store(**overrides: object) -> StandaloneServiceSpec:
    fields: dict[str, object] = {
        "id": "chroma",
        "name": "Chroma",
        "compose_path": "platform/vectorstores/chroma/docker-compose.yml",
        "host": "localhost",
        "port": 8000,
        "health_path": "/api/v1/heartbeat",
    }
    fields.update(overrides)
    return StandaloneServiceSpec(**fields)  # type: ignore[arg-type]


def embedding_service(**overrides: object) -> StandaloneServiceSpec:
    fields: dict[str, object] = {
        "id": "00-embedding-service",
        "name": "Embedding Service",
        "compose_path": "projects/00-embedding-service/docker-compose.yml",
        "host": "localhost",
        "port": 9100,
        "health_path": "/healthz",
    }
    fields.update(overrides)
    return StandaloneServiceSpec(**fields)  # type: ignore[arg-type]


@dataclass
class FakeLifecycleService:
    """One generic set of standalone-service call lists, used by both the
    vector-store and embedding-service router tests (they now go through
    the same LifecycleService methods)."""

    state: ComponentState = ComponentState.STOPPED
    start_backend_calls: list[tuple[BackendSpec, dict[str, str] | None]] = field(
        default_factory=list
    )
    stop_backend_calls: list[BackendSpec] = field(default_factory=list)
    reset_backend_calls: list[tuple[BackendSpec, dict[str, str] | None]] = field(
        default_factory=list
    )
    start_standalone_service_calls: list[StandaloneServiceSpec] = field(default_factory=list)
    stop_standalone_service_calls: list[StandaloneServiceSpec] = field(default_factory=list)
    reset_standalone_service_calls: list[StandaloneServiceSpec] = field(default_factory=list)
    raise_on_start_backend: Exception | None = None

    def start_backend(self, spec: BackendSpec, env: dict[str, str] | None = None) -> None:
        if self.raise_on_start_backend is not None:
            raise self.raise_on_start_backend
        self.start_backend_calls.append((spec, env))

    def stop_backend(self, spec: BackendSpec) -> None:
        self.stop_backend_calls.append(spec)

    def reset_backend(self, spec: BackendSpec, env: dict[str, str] | None = None) -> None:
        self.reset_backend_calls.append((spec, env))

    async def get_backend_status(self, spec: BackendSpec) -> ComponentState:
        return self.state

    def start_standalone_service(self, spec: StandaloneServiceSpec) -> None:
        self.start_standalone_service_calls.append(spec)

    def stop_standalone_service(self, spec: StandaloneServiceSpec) -> None:
        self.stop_standalone_service_calls.append(spec)

    def reset_standalone_service(self, spec: StandaloneServiceSpec) -> None:
        self.reset_standalone_service_calls.append(spec)

    async def get_standalone_service_status(self, spec: StandaloneServiceSpec) -> ComponentState:
        return self.state


@pytest.fixture
def lifecycle() -> FakeLifecycleService:
    return FakeLifecycleService()


@pytest.fixture
def client(lifecycle: FakeLifecycleService) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_backends] = lambda: [backend()]
    app.dependency_overrides[get_vectorstores] = lambda: [vector_store()]
    app.dependency_overrides[get_embedding_services] = lambda: [embedding_service()]
    app.dependency_overrides[get_lifecycle_service] = lambda: lifecycle
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestBackendsRouter:
    def test_list_returns_every_catalog_backend_with_its_state(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        lifecycle.state = ComponentState.HEALTHY

        response = client.get("/backends")

        assert response.status_code == 200
        [entry] = response.json()
        assert entry["spec"]["id"] == "03-production-rag-reference"
        assert entry["state"] == "healthy"

    def test_get_unknown_backend_is_404(self, client: TestClient) -> None:
        assert client.get("/backends/does-not-exist").status_code == 404

    def test_start_without_vector_store_passes_no_env(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post("/backends/03-production-rag-reference/start", json={})

        assert response.status_code == 202
        (spec, env) = lifecycle.start_backend_calls[0]
        assert spec.id == "03-production-rag-reference"
        assert env is None

    def test_start_with_a_compatible_vector_store_forwards_its_connection_env(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post(
            "/backends/03-production-rag-reference/start",
            json={"vector_store_id": "chroma"},
        )

        assert response.status_code == 202
        (_, env) = lifecycle.start_backend_calls[0]
        assert env == {
            "VECTOR_STORE_ID": "chroma",
            "VECTOR_STORE_HOST": "localhost",
            "VECTOR_STORE_PORT": "8000",
        }

    def test_start_with_an_incompatible_vector_store_is_rejected(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post(
            "/backends/03-production-rag-reference/start",
            json={"vector_store_id": "qdrant"},
        )

        assert response.status_code == 400
        assert lifecycle.start_backend_calls == []

    def test_start_with_a_compatible_embedding_service_forwards_its_connection_env(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post(
            "/backends/03-production-rag-reference/start",
            json={"embedding_service_id": "00-embedding-service"},
        )

        assert response.status_code == 202
        (_, env) = lifecycle.start_backend_calls[0]
        assert env == {
            "EMBEDDING_SERVICE_ID": "00-embedding-service",
            "EMBEDDING_SERVICE_HOST": "localhost",
            "EMBEDDING_SERVICE_PORT": "9100",
            "EMBEDDING_SERVICE_URL": "http://localhost:9100",
        }

    def test_start_with_an_incompatible_embedding_service_is_rejected(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post(
            "/backends/03-production-rag-reference/start",
            json={"embedding_service_id": "nope"},
        )

        assert response.status_code == 400
        assert lifecycle.start_backend_calls == []

    def test_start_with_both_a_vector_store_and_an_embedding_service_merges_their_env(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post(
            "/backends/03-production-rag-reference/start",
            json={"vector_store_id": "chroma", "embedding_service_id": "00-embedding-service"},
        )

        assert response.status_code == 202
        (_, env) = lifecycle.start_backend_calls[0]
        assert env == {
            "VECTOR_STORE_ID": "chroma",
            "VECTOR_STORE_HOST": "localhost",
            "VECTOR_STORE_PORT": "8000",
            "EMBEDDING_SERVICE_ID": "00-embedding-service",
            "EMBEDDING_SERVICE_HOST": "localhost",
            "EMBEDDING_SERVICE_PORT": "9100",
            "EMBEDDING_SERVICE_URL": "http://localhost:9100",
        }

    def test_a_failed_compose_command_is_surfaced_as_502_with_its_message(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        lifecycle.raise_on_start_backend = DockerComposeCommandError(
            ["docker", "compose", "up", "-d"], 1, "env file /repo/.env not found"
        )

        response = client.post("/backends/03-production-rag-reference/start", json={})

        assert response.status_code == 502
        assert "env file /repo/.env not found" in response.json()["detail"]

    def test_stop_delegates_to_the_lifecycle_service(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post("/backends/03-production-rag-reference/stop")

        assert response.status_code == 202
        assert len(lifecycle.stop_backend_calls) == 1

    def test_reset_delegates_to_the_lifecycle_service(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post("/backends/03-production-rag-reference/reset", json={})

        assert response.status_code == 202
        assert len(lifecycle.reset_backend_calls) == 1


class TestVectorStoresRouter:
    def test_list_returns_every_catalog_vector_store_with_its_state(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        lifecycle.state = ComponentState.STARTING

        response = client.get("/vectorstores")

        assert response.status_code == 200
        [entry] = response.json()
        assert entry["spec"]["id"] == "chroma"
        assert entry["state"] == "starting"

    def test_start_delegates_to_the_lifecycle_service(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post("/vectorstores/chroma/start")

        assert response.status_code == 202
        [spec] = lifecycle.start_standalone_service_calls
        assert spec.id == "chroma"

    def test_reset_delegates_to_the_lifecycle_service(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post("/vectorstores/chroma/reset")

        assert response.status_code == 202
        [spec] = lifecycle.reset_standalone_service_calls
        assert spec.id == "chroma"

    def test_get_unknown_vector_store_is_404(self, client: TestClient) -> None:
        assert client.get("/vectorstores/does-not-exist").status_code == 404


class TestEmbeddingServicesRouter:
    def test_list_returns_every_catalog_embedding_service_with_its_state(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        lifecycle.state = ComponentState.STARTING

        response = client.get("/embedding-services")

        assert response.status_code == 200
        [entry] = response.json()
        assert entry["spec"]["id"] == "00-embedding-service"
        assert entry["state"] == "starting"

    def test_start_delegates_to_the_lifecycle_service(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post("/embedding-services/00-embedding-service/start")

        assert response.status_code == 202
        [spec] = lifecycle.start_standalone_service_calls
        assert spec.id == "00-embedding-service"

    def test_reset_delegates_to_the_lifecycle_service(
        self, client: TestClient, lifecycle: FakeLifecycleService
    ) -> None:
        response = client.post("/embedding-services/00-embedding-service/reset")

        assert response.status_code == 202
        [spec] = lifecycle.reset_standalone_service_calls
        assert spec.id == "00-embedding-service"

    def test_get_unknown_embedding_service_is_404(self, client: TestClient) -> None:
        assert client.get("/embedding-services/does-not-exist").status_code == 404
