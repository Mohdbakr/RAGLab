"""The states the launcher UI renders for a backend or vector store."""

from __future__ import annotations

from enum import StrEnum


class ComponentState(StrEnum):
    """Lifecycle state of one backend or vector store."""

    STOPPED = "stopped"
    STARTING = "starting"
    HEALTHY = "healthy"
