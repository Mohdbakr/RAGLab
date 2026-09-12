"""Endpoints for listing and controlling standalone embedding services."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.catalog.models import EmbeddingServiceSpec
from app.dependencies import get_embedding_services, get_lifecycle_service
from app.models.status import ComponentState
from app.services.lifecycle import LifecycleService

router = APIRouter(prefix="/embedding-services", tags=["Embedding Services"])


class EmbeddingServiceStatus(BaseModel):
    """A catalog embedding service paired with its live lifecycle state."""

    spec: EmbeddingServiceSpec
    state: ComponentState


def _find_embedding_service(
    embedding_services: list[EmbeddingServiceSpec], embedding_service_id: str
) -> EmbeddingServiceSpec:
    for spec in embedding_services:
        if spec.id == embedding_service_id:
            return spec
    raise HTTPException(
        status_code=404, detail=f"unknown embedding service: {embedding_service_id!r}"
    )


@router.get("", response_model=list[EmbeddingServiceStatus])
async def list_embedding_services(
    embedding_services: list[EmbeddingServiceSpec] = Depends(get_embedding_services),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> list[EmbeddingServiceStatus]:
    """List every catalog embedding service alongside its current lifecycle state."""
    statuses = []
    for spec in embedding_services:
        state = await lifecycle.get_embedding_service_status(spec)
        statuses.append(EmbeddingServiceStatus(spec=spec, state=state))
    return statuses


@router.get("/{embedding_service_id}", response_model=EmbeddingServiceStatus)
async def get_embedding_service(
    embedding_service_id: str,
    embedding_services: list[EmbeddingServiceSpec] = Depends(get_embedding_services),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> EmbeddingServiceStatus:
    """Get one embedding service's catalog entry and current lifecycle state."""
    spec = _find_embedding_service(embedding_services, embedding_service_id)
    state = await lifecycle.get_embedding_service_status(spec)
    return EmbeddingServiceStatus(spec=spec, state=state)


@router.post("/{embedding_service_id}/start", status_code=202)
async def start_embedding_service(
    embedding_service_id: str,
    embedding_services: list[EmbeddingServiceSpec] = Depends(get_embedding_services),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> dict[str, str]:
    """Start an embedding service's stack."""
    spec = _find_embedding_service(embedding_services, embedding_service_id)
    lifecycle.start_embedding_service(spec)
    return {"state": "starting"}


@router.post("/{embedding_service_id}/stop", status_code=202)
async def stop_embedding_service(
    embedding_service_id: str,
    embedding_services: list[EmbeddingServiceSpec] = Depends(get_embedding_services),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> dict[str, str]:
    """Stop an embedding service's stack."""
    spec = _find_embedding_service(embedding_services, embedding_service_id)
    lifecycle.stop_embedding_service(spec)
    return {"state": "stopped"}


@router.post("/{embedding_service_id}/reset", status_code=202)
async def reset_embedding_service(
    embedding_service_id: str,
    embedding_services: list[EmbeddingServiceSpec] = Depends(get_embedding_services),
    lifecycle: LifecycleService = Depends(get_lifecycle_service),
) -> dict[str, str]:
    """Wipe an embedding service's data and bring it back up clean."""
    spec = _find_embedding_service(embedding_services, embedding_service_id)
    lifecycle.reset_embedding_service(spec)
    return {"state": "starting"}
