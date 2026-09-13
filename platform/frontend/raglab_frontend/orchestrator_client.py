"""Typed HTTP client for the orchestrator's REST API.

Kept separate from the Streamlit UI so it's unit-testable without a
browser: every request/response shape is validated through Pydantic
models mirroring the orchestrator's own catalog schemas.
"""

from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel

Level = Literal["basic", "intermediate", "advanced"]
CatalogStatus = Literal["planned", "in_progress", "shipped"]


class BackendSpec(BaseModel):
    """Mirrors ``app.catalog.models.BackendSpec`` on the orchestrator."""

    id: str
    name: str
    level: Level
    summary: str
    path: str
    needs_container: bool
    compose_file: str | None = None
    base_url: str
    health_path: str
    compatible_vector_stores: list[str] = []
    compatible_embedding_services: list[str] = []
    stateful: bool = False
    status: CatalogStatus = "planned"


class StandaloneServiceSpec(BaseModel):
    """Mirrors ``app.catalog.models.StandaloneServiceSpec`` on the orchestrator.

    Used for every standalone-service kind (vector stores, embedding
    services, ...) — they all share this exact shape there too.
    """

    id: str
    name: str
    compose_path: str
    host: str
    port: int
    health_path: str = ""
    status: CatalogStatus = "planned"


class BackendStatus(BaseModel):
    """A backend paired with its live lifecycle state."""

    spec: BackendSpec
    state: str


class StandaloneServiceStatus(BaseModel):
    """A standalone service paired with its live lifecycle state."""

    spec: StandaloneServiceSpec
    state: str


class OrchestratorClient:
    """Thin wrapper around the orchestrator's backend/vector-store/embedding-service API."""

    def __init__(
        self,
        base_url: str,
        client: httpx.Client | None = None,
        timeout: float = 5.0,
    ) -> None:
        """Create a client.

        Args:
            base_url: The orchestrator's base URL, e.g.
                ``"http://localhost:8100"``.
            client: An `httpx.Client` to reuse (tests inject one backed by
                a `MockTransport`). A default client is created if
                omitted.
            timeout: Per-request timeout in seconds, used only when no
                `client` is supplied.
        """
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    # -- standalone services (vector stores, embedding services, ...) ---
    #
    # Both kinds hit the same shaped REST surface on the orchestrator, so
    # the request/response handling lives once here; list_vector_stores,
    # start_embedding_service, etc. below are thin, stably-named wrappers
    # over these for callers.

    def _list_standalone_services(self, segment: str) -> list[StandaloneServiceStatus]:
        response = self._client.get(f"{self._base_url}/{segment}")
        response.raise_for_status()
        return [StandaloneServiceStatus.model_validate(item) for item in response.json()]

    def _get_standalone_service_status(
        self, segment: str, service_id: str
    ) -> StandaloneServiceStatus:
        response = self._client.get(f"{self._base_url}/{segment}/{service_id}")
        response.raise_for_status()
        return StandaloneServiceStatus.model_validate(response.json())

    def _post_standalone_service_action(self, segment: str, service_id: str, action: str) -> None:
        response = self._client.post(f"{self._base_url}/{segment}/{service_id}/{action}")
        response.raise_for_status()

    def list_vector_stores(self) -> list[StandaloneServiceStatus]:
        """Fetch every catalog vector store with its current state."""
        return self._list_standalone_services("vectorstores")

    def list_embedding_services(self) -> list[StandaloneServiceStatus]:
        """Fetch every catalog embedding service with its current state."""
        return self._list_standalone_services("embedding-services")

    def get_vector_store_status(self, vector_store_id: str) -> StandaloneServiceStatus:
        """Fetch one vector store's current state."""
        return self._get_standalone_service_status("vectorstores", vector_store_id)

    def get_embedding_service_status(self, embedding_service_id: str) -> StandaloneServiceStatus:
        """Fetch one embedding service's current state."""
        return self._get_standalone_service_status("embedding-services", embedding_service_id)

    def start_vector_store(self, vector_store_id: str) -> None:
        """Start a vector store."""
        self._post_standalone_service_action("vectorstores", vector_store_id, "start")

    def stop_vector_store(self, vector_store_id: str) -> None:
        """Stop a vector store."""
        self._post_standalone_service_action("vectorstores", vector_store_id, "stop")

    def reset_vector_store(self, vector_store_id: str) -> None:
        """Wipe a vector store's data and bring it back up clean."""
        self._post_standalone_service_action("vectorstores", vector_store_id, "reset")

    def start_embedding_service(self, embedding_service_id: str) -> None:
        """Start an embedding service."""
        self._post_standalone_service_action("embedding-services", embedding_service_id, "start")

    def stop_embedding_service(self, embedding_service_id: str) -> None:
        """Stop an embedding service."""
        self._post_standalone_service_action("embedding-services", embedding_service_id, "stop")

    def reset_embedding_service(self, embedding_service_id: str) -> None:
        """Wipe an embedding service's data and bring it back up clean."""
        self._post_standalone_service_action("embedding-services", embedding_service_id, "reset")

    # -- backends ---------------------------------------------------------

    def list_backends(self) -> list[BackendStatus]:
        """Fetch every catalog backend with its current state."""
        response = self._client.get(f"{self._base_url}/backends")
        response.raise_for_status()
        return [BackendStatus.model_validate(item) for item in response.json()]

    def get_backend_status(self, backend_id: str) -> BackendStatus:
        """Fetch one backend's current state."""
        response = self._client.get(f"{self._base_url}/backends/{backend_id}")
        response.raise_for_status()
        return BackendStatus.model_validate(response.json())

    def start_backend(
        self,
        backend_id: str,
        vector_store_id: str | None = None,
        embedding_service_id: str | None = None,
    ) -> None:
        """Start a backend, optionally pointed at a vector store and/or embedding service."""
        body = {}
        if vector_store_id:
            body["vector_store_id"] = vector_store_id
        if embedding_service_id:
            body["embedding_service_id"] = embedding_service_id
        response = self._client.post(f"{self._base_url}/backends/{backend_id}/start", json=body)
        response.raise_for_status()

    def stop_backend(self, backend_id: str) -> None:
        """Stop a backend."""
        response = self._client.post(f"{self._base_url}/backends/{backend_id}/stop")
        response.raise_for_status()

    def reset_backend(
        self,
        backend_id: str,
        vector_store_id: str | None = None,
        embedding_service_id: str | None = None,
    ) -> None:
        """Wipe a backend's persisted state and bring it back up clean."""
        body = {}
        if vector_store_id:
            body["vector_store_id"] = vector_store_id
        if embedding_service_id:
            body["embedding_service_id"] = embedding_service_id
        response = self._client.post(f"{self._base_url}/backends/{backend_id}/reset", json=body)
        response.raise_for_status()
