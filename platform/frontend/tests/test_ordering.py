"""Tests for raglab_frontend.ordering, written before the implementation."""

from __future__ import annotations

from raglab_frontend.orchestrator_client import BackendSpec, BackendStatus
from raglab_frontend.ordering import sort_basic_to_advanced


def status_for(backend_id: str, level: str) -> BackendStatus:
    return BackendStatus(
        spec=BackendSpec(
            id=backend_id,
            name=backend_id,
            level=level,  # type: ignore[arg-type]
            summary="x",
            path=f"projects/{backend_id}",
            needs_container=False,
            base_url="http://localhost:1",
            health_path="/",
        ),
        state="stopped",
    )


def test_orders_basic_before_intermediate_before_advanced() -> None:
    items = [
        status_for("advanced-one", "advanced"),
        status_for("basic-one", "basic"),
        status_for("intermediate-one", "intermediate"),
    ]

    ordered = sort_basic_to_advanced(items)

    assert [item.spec.id for item in ordered] == [
        "basic-one",
        "intermediate-one",
        "advanced-one",
    ]


def test_breaks_ties_within_a_level_by_id() -> None:
    items = [
        status_for("02-document-qa-foundation", "basic"),
        status_for("01-rag-from-scratch", "basic"),
    ]

    ordered = sort_basic_to_advanced(items)

    assert [item.spec.id for item in ordered] == [
        "01-rag-from-scratch",
        "02-document-qa-foundation",
    ]
