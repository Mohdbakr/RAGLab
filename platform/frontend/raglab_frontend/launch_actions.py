"""Call sequencing for the Launch/Stop/Reset buttons.

Kept separate from the UI so the ordering (vector store before backend on
launch, backend before vector store on stop) is unit-testable without
Streamlit or a real orchestrator.
"""

from __future__ import annotations

from typing import Protocol


class LifecycleClient(Protocol):
    """The subset of OrchestratorClient these actions depend on."""

    def start_backend(
        self,
        backend_id: str,
        vector_store_id: str | None = None,
        embedding_service_id: str | None = None,
    ) -> None: ...
    def stop_backend(self, backend_id: str) -> None: ...
    def reset_backend(
        self,
        backend_id: str,
        vector_store_id: str | None = None,
        embedding_service_id: str | None = None,
    ) -> None: ...
    def start_vector_store(self, vector_store_id: str) -> None: ...
    def stop_vector_store(self, vector_store_id: str) -> None: ...
    def reset_vector_store(self, vector_store_id: str) -> None: ...
    def start_embedding_service(self, embedding_service_id: str) -> None: ...
    def stop_embedding_service(self, embedding_service_id: str) -> None: ...
    def reset_embedding_service(self, embedding_service_id: str) -> None: ...


def launch(
    client: LifecycleClient,
    backend_id: str,
    vector_store_id: str | None,
    embedding_service_id: str | None = None,
) -> None:
    """Start the chosen vector store and/or embedding service (if any), then the backend.

    Docker Compose's ``up -d`` is idempotent, so starting an
    already-running vector store or embedding service here is harmless —
    this always runs them first rather than checking current state.

    Args:
        client: The orchestrator client to issue calls through.
        backend_id: Catalog id of the backend to start.
        vector_store_id: Catalog id of the vector store to start first, or
            None if the backend isn't using the shared vector-store slot.
        embedding_service_id: Catalog id of the embedding service to start
            first, or None if the backend isn't using the shared
            embedding-service slot.
    """
    if vector_store_id:
        client.start_vector_store(vector_store_id)
    if embedding_service_id:
        client.start_embedding_service(embedding_service_id)
    client.start_backend(backend_id, vector_store_id, embedding_service_id)


def stop(
    client: LifecycleClient,
    backend_id: str,
    vector_store_id: str | None,
    embedding_service_id: str | None = None,
) -> None:
    """Stop the backend, then the vector store and/or embedding service it was using.

    Args:
        client: The orchestrator client to issue calls through.
        backend_id: Catalog id of the backend to stop.
        vector_store_id: Catalog id of the vector store to stop afterward,
            or None to leave it running.
        embedding_service_id: Catalog id of the embedding service to stop
            afterward, or None to leave it running.
    """
    client.stop_backend(backend_id)
    if vector_store_id:
        client.stop_vector_store(vector_store_id)
    if embedding_service_id:
        client.stop_embedding_service(embedding_service_id)


def reset(
    client: LifecycleClient,
    backend_id: str,
    vector_store_id: str | None,
    embedding_service_id: str | None = None,
) -> None:
    """Wipe the backend's state, then the vector store's/embedding service's, for a clean restart.

    Args:
        client: The orchestrator client to issue calls through.
        backend_id: Catalog id of the backend to reset.
        vector_store_id: Catalog id of the vector store to reset
            afterward, or None if the backend isn't using the shared
            vector-store slot.
        embedding_service_id: Catalog id of the embedding service to reset
            afterward, or None if the backend isn't using the shared
            embedding-service slot.
    """
    client.reset_backend(backend_id, vector_store_id, embedding_service_id)
    if vector_store_id:
        client.reset_vector_store(vector_store_id)
    if embedding_service_id:
        client.reset_embedding_service(embedding_service_id)
