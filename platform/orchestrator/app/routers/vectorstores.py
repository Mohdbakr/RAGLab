"""Endpoints for listing and controlling standalone vector stores."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.catalog.models import VectorStoreSpec
from app.dependencies import get_lifecycle_service, get_vectorstores
from app.models.status import ComponentState
from app.services.lifecycle import LifecycleService

router = APIRouter(prefix="/vectorstores", tags=["Vector Stores"])


class VectorStoreStatus(BaseModel):
    """A catalog vector store paired with its live lifecycle state."""

    spec: VectorStoreSpec
    state: ComponentState


def _find_vector_store(
    vectorstores: list[VectorStoreSpec], vector_store_id: str
) -> VectorStoreSpec:
    for spec in vectorstores:
        if spec.id == vector_store_id:
            return spec
    raise HTTPException(status_code=404, detail=f"unknown vector store: {vector_store_id!r}")


@router.get("", response_model=list[VectorStoreStatus])
async def list_vectorstores(
    vectorstores: list[VectorStoreSpec] = Depends(get_vectorstores),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> list[VectorStoreStatus]:
    """List every catalog vector store alongside its current lifecycle state."""
    return [
        VectorStoreStatus(spec=spec, state=await lifecycle.get_vector_store_status(spec))
        for spec in vectorstores
    ]


@router.get("/{vector_store_id}", response_model=VectorStoreStatus)
async def get_vector_store(
    vector_store_id: str,
    vectorstores: list[VectorStoreSpec] = Depends(get_vectorstores),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> VectorStoreStatus:
    """Get one vector store's catalog entry and current lifecycle state."""
    spec = _find_vector_store(vectorstores, vector_store_id)
    return VectorStoreStatus(spec=spec, state=await lifecycle.get_vector_store_status(spec))


@router.post("/{vector_store_id}/start", status_code=202)
async def start_vector_store(
    vector_store_id: str,
    vectorstores: list[VectorStoreSpec] = Depends(get_vectorstores),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> dict[str, str]:
    """Start a vector store's stack."""
    spec = _find_vector_store(vectorstores, vector_store_id)
    lifecycle.start_vector_store(spec)
    return {"state": "starting"}


@router.post("/{vector_store_id}/stop", status_code=202)
async def stop_vector_store(
    vector_store_id: str,
    vectorstores: list[VectorStoreSpec] = Depends(get_vectorstores),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> dict[str, str]:
    """Stop a vector store's stack."""
    spec = _find_vector_store(vectorstores, vector_store_id)
    lifecycle.stop_vector_store(spec)
    return {"state": "stopped"}


@router.post("/{vector_store_id}/reset", status_code=202)
async def reset_vector_store(
    vector_store_id: str,
    vectorstores: list[VectorStoreSpec] = Depends(get_vectorstores),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> dict[str, str]:
    """Wipe a vector store's data and bring it back up clean."""
    spec = _find_vector_store(vectorstores, vector_store_id)
    lifecycle.reset_vector_store(spec)
    return {"state": "starting"}
