"""Factory for a standalone-service router (vector stores, embedding services, ...).

Every standalone-service catalog kind shares one shape and one lifecycle
(list/get/start/stop/reset, independent of any one backend), so the REST
surface is built once here and instantiated per kind in `app.main`,
rather than maintaining a near-identical router module per kind.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.catalog.models import StandaloneServiceSpec
from app.dependencies import get_lifecycle_service
from app.models.status import ComponentState
from app.services.lifecycle import LifecycleService


class StandaloneServiceStatus(BaseModel):
    """A catalog standalone service paired with its live lifecycle state."""

    spec: StandaloneServiceSpec
    state: ComponentState


def _find(
    specs: list[StandaloneServiceSpec], service_id: str, kind_label: str
) -> StandaloneServiceSpec:
    for spec in specs:
        if spec.id == service_id:
            return spec
    raise HTTPException(status_code=404, detail=f"unknown {kind_label}: {service_id!r}")


def build_standalone_service_router(
    *,
    prefix: str,
    tag: str,
    kind_label: str,
    get_specs: Callable[[Request], list[StandaloneServiceSpec]],
) -> APIRouter:
    """Build a list/get/start/stop/reset router for one standalone-service kind.

    Args:
        prefix: URL prefix, e.g. ``"/vectorstores"``.
        tag: OpenAPI tag, e.g. ``"Vector Stores"``.
        kind_label: Used in 404 messages, e.g. ``"vector store"``.
        get_specs: The catalog dependency for this kind, e.g.
            ``get_vectorstores``.

    Returns:
        A ready-to-include ``APIRouter``.
    """
    router = APIRouter(prefix=prefix, tags=[tag])

    @router.get("", response_model=list[StandaloneServiceStatus])
    async def list_services(
        specs: list[StandaloneServiceSpec] = Depends(get_specs),
        lifecycle: LifecycleService = Depends(get_lifecycle_service),
    ) -> list[StandaloneServiceStatus]:
        """List every catalog entry of this kind alongside its current lifecycle state."""
        statuses = []
        for spec in specs:
            state = await lifecycle.get_standalone_service_status(spec)
            statuses.append(StandaloneServiceStatus(spec=spec, state=state))
        return statuses

    @router.get("/{service_id}", response_model=StandaloneServiceStatus)
    async def get_service(
        service_id: str,
        specs: list[StandaloneServiceSpec] = Depends(get_specs),
        lifecycle: LifecycleService = Depends(get_lifecycle_service),
    ) -> StandaloneServiceStatus:
        """Get one catalog entry and its current lifecycle state."""
        spec = _find(specs, service_id, kind_label)
        state = await lifecycle.get_standalone_service_status(spec)
        return StandaloneServiceStatus(spec=spec, state=state)

    @router.post("/{service_id}/start", status_code=202)
    async def start_service(
        service_id: str,
        specs: list[StandaloneServiceSpec] = Depends(get_specs),
        lifecycle: LifecycleService = Depends(get_lifecycle_service),
    ) -> dict[str, str]:
        """Start this entry's stack."""
        spec = _find(specs, service_id, kind_label)
        lifecycle.start_standalone_service(spec)
        return {"state": "starting"}

    @router.post("/{service_id}/stop", status_code=202)
    async def stop_service(
        service_id: str,
        specs: list[StandaloneServiceSpec] = Depends(get_specs),
        lifecycle: LifecycleService = Depends(get_lifecycle_service),
    ) -> dict[str, str]:
        """Stop this entry's stack."""
        spec = _find(specs, service_id, kind_label)
        lifecycle.stop_standalone_service(spec)
        return {"state": "stopped"}

    @router.post("/{service_id}/reset", status_code=202)
    async def reset_service(
        service_id: str,
        specs: list[StandaloneServiceSpec] = Depends(get_specs),
        lifecycle: LifecycleService = Depends(get_lifecycle_service),
    ) -> dict[str, str]:
        """Wipe this entry's data and bring it back up clean."""
        spec = _find(specs, service_id, kind_label)
        lifecycle.reset_standalone_service(spec)
        return {"state": "starting"}

    return router
