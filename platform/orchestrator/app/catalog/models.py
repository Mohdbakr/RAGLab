"""Typed schemas mirroring catalog/backends.yaml and catalog/vectorstores.yaml."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

Level = Literal["basic", "intermediate", "advanced"]
CatalogStatus = Literal["planned", "in_progress", "shipped"]


class BackendSpec(BaseModel):
    """One entry from ``catalog/backends.yaml``.

    Attributes:
        id: Unique slug, matches the ``projects/<id>`` folder name.
        name: Human-readable display name.
        level: Drives basic-to-advanced ordering in the launcher UI.
        summary: One-line description shown in the dropdown.
        path: Project folder path, relative to the repo root.
        needs_container: If false, the backend runs in-process and has no
            Docker lifecycle.
        compose_file: Path to the project's compose file, relative to
            ``path``. Required when ``needs_container`` is true.
        base_url: Base URL used to reach the running backend.
        health_path: HTTP path polled to tell "starting" from "healthy".
        compatible_vector_stores: ids from ``vectorstores.yaml`` this
            backend can be pointed at. Empty means not swappable.
        stateful: If true, Reset wipes this backend's persisted data.
        status: Delivery status, tracked as each weekend milestone lands.
    """

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

    @model_validator(mode="after")
    def _require_compose_file_when_containerized(self) -> BackendSpec:
        if self.needs_container and not self.compose_file:
            raise ValueError(
                f"backend {self.id!r} has needs_container=true but no compose_file set"
            )
        return self


class VectorStoreSpec(BaseModel):
    """One entry from ``catalog/vectorstores.yaml``.

    Attributes:
        id: Unique slug.
        name: Human-readable display name.
        compose_path: Path (from the repo root) to a compose file that
            stands up just this store.
        host: Hostname the store is reachable at once its stack is up.
        port: Primary client port.
        health_path: HTTP path polled for health, empty for a TCP-only
            check.
        status: Delivery status, tracked as each weekend milestone lands.
    """

    id: str
    name: str
    compose_path: str
    host: str
    port: int
    health_path: str = ""
    status: CatalogStatus = "planned"
