"""Tests for raglab_frontend.launch_actions, written before the implementation.

Uses a hand-rolled fake OrchestratorClient so these stay pure unit tests of
call sequencing, independent of HTTP entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from raglab_frontend.launch_actions import launch, reset, stop


@dataclass
class FakeClient:
    calls: list[tuple[str, str, str | None]] = field(default_factory=list)

    def start_backend(self, backend_id: str, vector_store_id: str | None = None) -> None:
        self.calls.append(("start_backend", backend_id, vector_store_id))

    def stop_backend(self, backend_id: str) -> None:
        self.calls.append(("stop_backend", backend_id, None))

    def reset_backend(self, backend_id: str, vector_store_id: str | None = None) -> None:
        self.calls.append(("reset_backend", backend_id, vector_store_id))

    def start_vector_store(self, vector_store_id: str) -> None:
        self.calls.append(("start_vector_store", vector_store_id, None))

    def stop_vector_store(self, vector_store_id: str) -> None:
        self.calls.append(("stop_vector_store", vector_store_id, None))

    def reset_vector_store(self, vector_store_id: str) -> None:
        self.calls.append(("reset_vector_store", vector_store_id, None))


class TestLaunch:
    def test_starts_only_the_backend_when_no_vector_store_chosen(self) -> None:
        client = FakeClient()

        launch(client, "01-rag-from-scratch", vector_store_id=None)

        assert client.calls == [("start_backend", "01-rag-from-scratch", None)]

    def test_starts_the_vector_store_before_the_backend(self) -> None:
        client = FakeClient()

        launch(client, "03-production-rag-reference", vector_store_id="chroma")

        assert client.calls == [
            ("start_vector_store", "chroma", None),
            ("start_backend", "03-production-rag-reference", "chroma"),
        ]


class TestStop:
    def test_stops_only_the_backend_when_no_vector_store_chosen(self) -> None:
        client = FakeClient()

        stop(client, "01-rag-from-scratch", vector_store_id=None)

        assert client.calls == [("stop_backend", "01-rag-from-scratch", None)]

    def test_stops_the_backend_before_the_vector_store(self) -> None:
        client = FakeClient()

        stop(client, "03-production-rag-reference", vector_store_id="chroma")

        assert client.calls == [
            ("stop_backend", "03-production-rag-reference", None),
            ("stop_vector_store", "chroma", None),
        ]


class TestReset:
    def test_resets_backend_and_vector_store_when_one_is_chosen(self) -> None:
        client = FakeClient()

        reset(client, "03-production-rag-reference", vector_store_id="chroma")

        assert client.calls == [
            ("reset_backend", "03-production-rag-reference", "chroma"),
            ("reset_vector_store", "chroma", None),
        ]

    def test_resets_only_the_backend_when_no_vector_store_chosen(self) -> None:
        client = FakeClient()

        reset(client, "01-rag-from-scratch", vector_store_id=None)

        assert client.calls == [("reset_backend", "01-rag-from-scratch", None)]
