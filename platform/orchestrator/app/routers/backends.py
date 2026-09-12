"""Endpoints for listing and controlling RAG backend implementations."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.catalog.models import BackendSpec, VectorStoreSpec
from app.dependencies import get_backends, get_lifecycle_service, get_vectorstores
from app.models.status import ComponentState
from app.services.lifecycle import LifecycleService

router = APIRouter(prefix="/backends", tags=["Backends"])


class BackendStatus(BaseModel):
    """A catalog backend paired with its live lifecycle state."""

    spec: BackendSpec
    state: ComponentState


class StartOrResetBackendRequest(BaseModel):
    """Optional vector store selection for a start/reset call."""

    vector_store_id: str | None = None


def _find_backend(backends: list[BackendSpec], backend_id: str) -> BackendSpec:
    for spec in backends:
        if spec.id == backend_id:
            return spec
    raise HTTPException(status_code=404, detail=f"unknown backend: {backend_id!r}")


def _find_vector_store(
    vectorstores: list[VectorStoreSpec], vector_store_id: str
) -> VectorStoreSpec:
    for spec in vectorstores:
        if spec.id == vector_store_id:
            return spec
    raise HTTPException(status_code=404, detail=f"unknown vector store: {vector_store_id!r}")


def _connection_env(spec: VectorStoreSpec) -> dict[str, str]:
    return {
        "VECTOR_STORE_ID": spec.id,
        "VECTOR_STORE_HOST": spec.host,
        "VECTOR_STORE_PORT": str(spec.port),
    }


def _resolve_env(
    backend: BackendSpec,
    vector_store_id: str | None,
    vectorstores: list[VectorStoreSpec],
) -> dict[str, str] | None:
    if vector_store_id is None:
        return None
    if vector_store_id not in backend.compatible_vector_stores:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{backend.id!r} is not compatible with vector store {vector_store_id!r}; "
                f"choose one of {backend.compatible_vector_stores}"
            ),
        )
    return _connection_env(_find_vector_store(vectorstores, vector_store_id))


@router.get("", response_model=list[BackendStatus])
async def list_backends(
    backends: list[BackendSpec] = Depends(get_backends),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> list[BackendStatus]:
    """List every catalog backend alongside its current lifecycle state."""
    return [
        BackendStatus(spec=spec, state=await lifecycle.get_backend_status(spec))
        for spec in backends
    ]


@router.get("/{backend_id}", response_model=BackendStatus)
async def get_backend(
    backend_id: str,
    backends: list[BackendSpec] = Depends(get_backends),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> BackendStatus:
    """Get one backend's catalog entry and current lifecycle state."""
    spec = _find_backend(backends, backend_id)
    return BackendStatus(spec=spec, state=await lifecycle.get_backend_status(spec))


@router.post("/{backend_id}/start", status_code=202)
async def start_backend(
    backend_id: str,
    body: StartOrResetBackendRequest | None = None,
    backends: list[BackendSpec] = Depends(get_backends),
    vectorstores: list[VectorStoreSpec] = Depends(get_vectorstores),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> dict[str, str]:
    """Start a backend, optionally pointed at a compatible vector store."""
    spec = _find_backend(backends, backend_id)
    vector_store_id = body.vector_store_id if body else None
    env = _resolve_env(spec, vector_store_id, vectorstores)
    lifecycle.start_backend(spec, env=env)
    return {"state": "starting"}


@router.post("/{backend_id}/stop", status_code=202)
async def stop_backend(
    backend_id: str,
    backends: list[BackendSpec] = Depends(get_backends),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> dict[str, str]:
    """Stop a backend's stack."""
    spec = _find_backend(backends, backend_id)
    lifecycle.stop_backend(spec)
    return {"state": "stopped"}


@router.post("/{backend_id}/reset", status_code=202)
async def reset_backend(
    backend_id: str,
    body: StartOrResetBackendRequest | None = None,
    backends: list[BackendSpec] = Depends(get_backends),
    vectorstores: list[VectorStoreSpec] = Depends(get_vectorstores),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> dict[str, str]:
    """Wipe a backend's persisted state and bring it back up clean."""
    spec = _find_backend(backends, backend_id)
    vector_store_id = body.vector_store_id if body else None
    env = _resolve_env(spec, vector_store_id, vectorstores)
    lifecycle.reset_backend(spec, env=env)
    return {"state": "starting"}
