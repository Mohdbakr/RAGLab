"""Tests for raglab_frontend.launch_actions, written before the implementation.

Uses a hand-rolled fake OrchestratorClient so these stay pure unit tests of
call sequencing, independent of HTTP entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from raglab_frontend.launch_actions import launch, reset, stop


@dataclass
class FakeClient:
    """Records calls as (action, id, vector_store_id, embedding_service_id)."""

    calls: list[tuple[str, str, str | None, str | None]] = field(default_factory=list)

    def start_backend(
        self,
        backend_id: str,
        vector_store_id: str | None = None,
        embedding_service_id: str | None = None,
    ) -> None:
        self.calls.append(("start_backend", backend_id, vector_store_id, embedding_service_id))

    def stop_backend(self, backend_id: str) -> None:
        self.calls.append(("stop_backend", backend_id, None, None))

    def reset_backend(
        self,
        backend_id: str,
        vector_store_id: str | None = None,
        embedding_service_id: str | None = None,
    ) -> None:
        self.calls.append(("reset_backend", backend_id, vector_store_id, embedding_service_id))

    def start_vector_store(self, vector_store_id: str) -> None:
        self.calls.append(("start_vector_store", vector_store_id, None, None))

    def stop_vector_store(self, vector_store_id: str) -> None:
        self.calls.append(("stop_vector_store", vector_store_id, None, None))

    def reset_vector_store(self, vector_store_id: str) -> None:
        self.calls.append(("reset_vector_store", vector_store_id, None, None))

    def start_embedding_service(self, embedding_service_id: str) -> None:
        self.calls.append(("start_embedding_service", embedding_service_id, None, None))

    def stop_embedding_service(self, embedding_service_id: str) -> None:
        self.calls.append(("stop_embedding_service", embedding_service_id, None, None))

    def reset_embedding_service(self, embedding_service_id: str) -> None:
        self.calls.append(("reset_embedding_service", embedding_service_id, None, None))


class TestLaunch:
    def test_starts_only_the_backend_when_nothing_else_chosen(self) -> None:
        client = FakeClient()

        launch(client, "01-rag-from-scratch", vector_store_id=None)

        assert client.calls == [("start_backend", "01-rag-from-scratch", None, None)]

    def test_starts_the_vector_store_before_the_backend(self) -> None:
        client = FakeClient()

        launch(client, "03-production-rag-reference", vector_store_id="chroma")

        assert client.calls == [
            ("start_vector_store", "chroma", None, None),
            ("start_backend", "03-production-rag-reference", "chroma", None),
        ]

    def test_starts_the_embedding_service_before_the_backend(self) -> None:
        client = FakeClient()

        launch(
            client,
            "03-production-rag-reference",
            vector_store_id=None,
            embedding_service_id="00-embedding-service",
        )

        assert client.calls == [
            ("start_embedding_service", "00-embedding-service", None, None),
            ("start_backend", "03-production-rag-reference", None, "00-embedding-service"),
        ]

    def test_starts_both_auxiliary_services_before_the_backend(self) -> None:
        client = FakeClient()

        launch(
            client,
            "03-production-rag-reference",
            vector_store_id="chroma",
            embedding_service_id="00-embedding-service",
        )

        assert client.calls == [
            ("start_vector_store", "chroma", None, None),
            ("start_embedding_service", "00-embedding-service", None, None),
            (
                "start_backend",
                "03-production-rag-reference",
                "chroma",
                "00-embedding-service",
            ),
        ]


class TestStop:
    def test_stops_only_the_backend_when_nothing_else_chosen(self) -> None:
        client = FakeClient()

        stop(client, "01-rag-from-scratch", vector_store_id=None)

        assert client.calls == [("stop_backend", "01-rag-from-scratch", None, None)]

    def test_stops_the_backend_before_the_vector_store(self) -> None:
        client = FakeClient()

        stop(client, "03-production-rag-reference", vector_store_id="chroma")

        assert client.calls == [
            ("stop_backend", "03-production-rag-reference", None, None),
            ("stop_vector_store", "chroma", None, None),
        ]

    def test_stops_the_backend_before_both_auxiliary_services(self) -> None:
        client = FakeClient()

        stop(
            client,
            "03-production-rag-reference",
            vector_store_id="chroma",
            embedding_service_id="00-embedding-service",
        )

        assert client.calls == [
            ("stop_backend", "03-production-rag-reference", None, None),
            ("stop_vector_store", "chroma", None, None),
            ("stop_embedding_service", "00-embedding-service", None, None),
        ]


class TestReset:
    def test_resets_backend_and_vector_store_when_one_is_chosen(self) -> None:
        client = FakeClient()

        reset(client, "03-production-rag-reference", vector_store_id="chroma")

        assert client.calls == [
            ("reset_backend", "03-production-rag-reference", "chroma", None),
            ("reset_vector_store", "chroma", None, None),
        ]

    def test_resets_only_the_backend_when_nothing_else_chosen(self) -> None:
        client = FakeClient()

        reset(client, "01-rag-from-scratch", vector_store_id=None)

        assert client.calls == [("reset_backend", "01-rag-from-scratch", None, None)]

    def test_resets_backend_then_both_auxiliary_services(self) -> None:
        client = FakeClient()

        reset(
            client,
            "03-production-rag-reference",
            vector_store_id="chroma",
            embedding_service_id="00-embedding-service",
        )

        assert client.calls == [
            (
                "reset_backend",
                "03-production-rag-reference",
                "chroma",
                "00-embedding-service",
            ),
            ("reset_vector_store", "chroma", None, None),
            ("reset_embedding_service", "00-embedding-service", None, None),
        ]
