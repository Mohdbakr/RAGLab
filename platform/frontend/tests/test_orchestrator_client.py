"""Tests for raglab_frontend.orchestrator_client, written before the implementation.

No real orchestrator process is involved: httpx's sync client is given a
MockTransport so requests are answered in-process.
"""

from __future__ import annotations

import httpx
import pytest

from raglab_frontend.orchestrator_client import (
    BackendStatus,
    EmbeddingServiceStatus,
    OrchestratorClient,
    VectorStoreStatus,
)

BACKEND_PAYLOAD = {
    "spec": {
        "id": "03-production-rag-reference",
        "name": "Production RAG Reference",
        "level": "intermediate",
        "summary": "Adapter-pattern vector store, retries, caching.",
        "path": "projects/03-production-rag-reference",
        "needs_container": True,
        "compose_file": "docker-compose.yml",
        "base_url": "http://localhost:9003",
        "health_path": "/healthz",
        "compatible_vector_stores": ["chroma"],
        "stateful": True,
        "status": "planned",
    },
    "state": "healthy",
}

VECTOR_STORE_PAYLOAD = {
    "spec": {
        "id": "chroma",
        "name": "Chroma",
        "compose_path": "platform/vectorstores/chroma/docker-compose.yml",
        "host": "localhost",
        "port": 8000,
        "health_path": "/api/v1/heartbeat",
        "status": "planned",
    },
    "state": "stopped",
}

EMBEDDING_SERVICE_PAYLOAD = {
    "spec": {
        "id": "00-embedding-service",
        "name": "Embedding Service",
        "compose_path": "projects/00-embedding-service/docker-compose.yml",
        "host": "localhost",
        "port": 9100,
        "health_path": "/healthz",
        "status": "shipped",
    },
    "state": "stopped",
}


def client_with(handler: httpx.WSGITransport | httpx.MockTransport) -> OrchestratorClient:
    return OrchestratorClient(
        base_url="http://orchestrator:8100",
        client=httpx.Client(transport=handler),
    )


def recording_handler(
    calls: list[httpx.Request], response: httpx.Response
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return response

    return httpx.MockTransport(handle)


class TestListBackends:
    def test_parses_the_list_response(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(calls, httpx.Response(200, json=[BACKEND_PAYLOAD]))
        client = client_with(transport)

        [status] = client.list_backends()

        assert isinstance(status, BackendStatus)
        assert status.spec.id == "03-production-rag-reference"
        assert status.state == "healthy"
        assert calls[0].url == httpx.URL("http://orchestrator:8100/backends")

    def test_raises_on_a_server_error(self) -> None:
        transport = recording_handler([], httpx.Response(500))
        client = client_with(transport)

        with pytest.raises(httpx.HTTPStatusError):
            client.list_backends()


class TestListVectorStores:
    def test_parses_the_list_response(self) -> None:
        transport = recording_handler([], httpx.Response(200, json=[VECTOR_STORE_PAYLOAD]))
        client = client_with(transport)

        [status] = client.list_vector_stores()

        assert isinstance(status, VectorStoreStatus)
        assert status.spec.id == "chroma"
        assert status.state == "stopped"


class TestListEmbeddingServices:
    def test_parses_the_list_response(self) -> None:
        transport = recording_handler(
            [], httpx.Response(200, json=[EMBEDDING_SERVICE_PAYLOAD])
        )
        client = client_with(transport)

        [status] = client.list_embedding_services()

        assert isinstance(status, EmbeddingServiceStatus)
        assert status.spec.id == "00-embedding-service"
        assert status.state == "stopped"


class TestGetOneStatus:
    def test_get_backend_status_parses_the_response(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(calls, httpx.Response(200, json=BACKEND_PAYLOAD))
        client = client_with(transport)

        status = client.get_backend_status("03-production-rag-reference")

        assert status.state == "healthy"
        assert calls[0].url.path == "/backends/03-production-rag-reference"

    def test_get_vector_store_status_parses_the_response(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(calls, httpx.Response(200, json=VECTOR_STORE_PAYLOAD))
        client = client_with(transport)

        status = client.get_vector_store_status("chroma")

        assert status.state == "stopped"
        assert calls[0].url.path == "/vectorstores/chroma"

    def test_get_embedding_service_status_parses_the_response(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(
            calls, httpx.Response(200, json=EMBEDDING_SERVICE_PAYLOAD)
        )
        client = client_with(transport)

        status = client.get_embedding_service_status("00-embedding-service")

        assert status.state == "stopped"
        assert calls[0].url.path == "/embedding-services/00-embedding-service"


class TestStartBackend:
    def test_posts_start_with_no_body_when_no_vector_store_chosen(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(calls, httpx.Response(202, json={"state": "starting"}))
        client = client_with(transport)

        client.start_backend("01-rag-from-scratch")

        request = calls[0]
        assert request.method == "POST"
        assert request.url == httpx.URL(
            "http://orchestrator:8100/backends/01-rag-from-scratch/start"
        )
        assert request.content == b"{}"

    def test_posts_the_vector_store_id_when_chosen(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(calls, httpx.Response(202, json={"state": "starting"}))
        client = client_with(transport)

        client.start_backend("03-production-rag-reference", vector_store_id="chroma")

        assert calls[0].content == b'{"vector_store_id":"chroma"}'

    def test_posts_both_ids_when_both_are_chosen(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(calls, httpx.Response(202, json={"state": "starting"}))
        client = client_with(transport)

        client.start_backend(
            "03-production-rag-reference",
            vector_store_id="chroma",
            embedding_service_id="00-embedding-service",
        )

        assert calls[0].content == (
            b'{"vector_store_id":"chroma","embedding_service_id":"00-embedding-service"}'
        )


class TestStopAndResetBackend:
    def test_stop_posts_to_the_stop_endpoint(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(calls, httpx.Response(202, json={"state": "stopped"}))
        client = client_with(transport)

        client.stop_backend("01-rag-from-scratch")

        assert calls[0].url == httpx.URL(
            "http://orchestrator:8100/backends/01-rag-from-scratch/stop"
        )

    def test_reset_posts_the_vector_store_id_when_chosen(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(calls, httpx.Response(202, json={"state": "starting"}))
        client = client_with(transport)

        client.reset_backend("03-production-rag-reference", vector_store_id="chroma")

        assert calls[0].url.path == "/backends/03-production-rag-reference/reset"
        assert calls[0].content == b'{"vector_store_id":"chroma"}'


class TestVectorStoreLifecycle:
    def test_start_stop_reset_hit_the_expected_paths(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(calls, httpx.Response(202, json={"state": "starting"}))
        client = client_with(transport)

        client.start_vector_store("chroma")
        client.stop_vector_store("chroma")
        client.reset_vector_store("chroma")

        assert [c.url.path for c in calls] == [
            "/vectorstores/chroma/start",
            "/vectorstores/chroma/stop",
            "/vectorstores/chroma/reset",
        ]


class TestEmbeddingServiceLifecycle:
    def test_start_stop_reset_hit_the_expected_paths(self) -> None:
        calls: list[httpx.Request] = []
        transport = recording_handler(calls, httpx.Response(202, json={"state": "starting"}))
        client = client_with(transport)

        client.start_embedding_service("00-embedding-service")
        client.stop_embedding_service("00-embedding-service")
        client.reset_embedding_service("00-embedding-service")

        assert [c.url.path for c in calls] == [
            "/embedding-services/00-embedding-service/start",
            "/embedding-services/00-embedding-service/stop",
            "/embedding-services/00-embedding-service/reset",
        ]
