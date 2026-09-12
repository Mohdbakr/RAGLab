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
    stateful: bool = False
    status: CatalogStatus = "planned"


class VectorStoreSpec(BaseModel):
    """Mirrors ``app.catalog.models.VectorStoreSpec`` on the orchestrator."""

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


class VectorStoreStatus(BaseModel):
    """A vector store paired with its live lifecycle state."""

    spec: VectorStoreSpec
    state: str


class OrchestratorClient:
    """Thin wrapper around the orchestrator's `/backends` and `/vectorstores` API."""

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

    def list_backends(self) -> list[BackendStatus]:
        """Fetch every catalog backend with its current state."""
        response = self._client.get(f"{self._base_url}/backends")
        response.raise_for_status()
        return [BackendStatus.model_validate(item) for item in response.json()]

    def list_vector_stores(self) -> list[VectorStoreStatus]:
        """Fetch every catalog vector store with its current state."""
        response = self._client.get(f"{self._base_url}/vectorstores")
        response.raise_for_status()
        return [VectorStoreStatus.model_validate(item) for item in response.json()]

    def get_backend_status(self, backend_id: str) -> BackendStatus:
        """Fetch one backend's current state."""
        response = self._client.get(f"{self._base_url}/backends/{backend_id}")
        response.raise_for_status()
        return BackendStatus.model_validate(response.json())

    def get_vector_store_status(self, vector_store_id: str) -> VectorStoreStatus:
        """Fetch one vector store's current state."""
        response = self._client.get(f"{self._base_url}/vectorstores/{vector_store_id}")
        response.raise_for_status()
        return VectorStoreStatus.model_validate(response.json())

    def start_backend(self, backend_id: str, vector_store_id: str | None = None) -> None:
        """Start a backend, optionally pointed at a vector store."""
        body = {"vector_store_id": vector_store_id} if vector_store_id else {}
        response = self._client.post(f"{self._base_url}/backends/{backend_id}/start", json=body)
        response.raise_for_status()

    def stop_backend(self, backend_id: str) -> None:
        """Stop a backend."""
        response = self._client.post(f"{self._base_url}/backends/{backend_id}/stop")
        response.raise_for_status()

    def reset_backend(self, backend_id: str, vector_store_id: str | None = None) -> None:
        """Wipe a backend's persisted state and bring it back up clean."""
        body = {"vector_store_id": vector_store_id} if vector_store_id else {}
        response = self._client.post(f"{self._base_url}/backends/{backend_id}/reset", json=body)
        response.raise_for_status()

    def start_vector_store(self, vector_store_id: str) -> None:
        """Start a vector store."""
        response = self._client.post(f"{self._base_url}/vectorstores/{vector_store_id}/start")
        response.raise_for_status()

    def stop_vector_store(self, vector_store_id: str) -> None:
        """Stop a vector store."""
        response = self._client.post(f"{self._base_url}/vectorstores/{vector_store_id}/stop")
        response.raise_for_status()

    def reset_vector_store(self, vector_store_id: str) -> None:
        """Wipe a vector store's data and bring it back up clean."""
        response = self._client.post(f"{self._base_url}/vectorstores/{vector_store_id}/reset")
        response.raise_for_status()
