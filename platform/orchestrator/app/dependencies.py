"""FastAPI dependency accessors for state set up in the app's lifespan.

Routers depend on these functions rather than reaching into
``request.app.state`` directly, so tests can override them with fakes
without needing a real catalog file or Docker daemon.
"""

from __future__ import annotations

from fastapi import Request

from app.catalog.models import BackendSpec, VectorStoreSpec
from app.services.lifecycle import LifecycleService


def get_backends(request: Request) -> list[BackendSpec]:
    """Return the backend catalog loaded at startup."""
    return request.app.state.backends  # type: ignore[no-any-return]


def get_vectorstores(request: Request) -> list[VectorStoreSpec]:
    """Return the vector-store catalog loaded at startup."""
    return request.app.state.vectorstores  # type: ignore[no-any-return]


def get_lifecycle_service(request: Request) -> LifecycleService:
    """Return the process-wide LifecycleService."""
    return request.app.state.lifecycle_service  # type: ignore[no-any-return]
