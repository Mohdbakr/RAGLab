"""Sorts catalog entries basic-to-advanced for the dropdown."""

from __future__ import annotations

from raglab_frontend.orchestrator_client import BackendStatus

_LEVEL_RANK = {"basic": 0, "intermediate": 1, "advanced": 2}


def sort_basic_to_advanced(backends: list[BackendStatus]) -> list[BackendStatus]:
    """Sort backends by level (basic first), then by id within a level.

    Args:
        backends: The backends to sort.

    Returns:
        A new list, sorted basic -> intermediate -> advanced.
    """
    return sorted(backends, key=lambda item: (_LEVEL_RANK[item.spec.level], item.spec.id))
