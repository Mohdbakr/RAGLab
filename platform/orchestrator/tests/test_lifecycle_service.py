"""Tests for app.services.lifecycle.LifecycleService, written before the implementation."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.catalog.models import BackendSpec, StandaloneServiceSpec
from app.models.status import ComponentState
from app.services.lifecycle import LifecycleService

REPO_ROOT = Path("/repo")


@dataclass
class FakeRuntime:
    """Records calls instead of touching Docker; `running` scripts is_running.

    Each recorded call includes the project_name LifecycleService passed,
    so tests can assert it's always the catalog spec's own id, and the
    thread it ran on, so tests can assert blocking calls are offloaded
    off the event loop rather than run inline.
    """

    running: bool = False
    up_calls: list[tuple[Path, str, Mapping[str, str] | None]] = field(default_factory=list)
    down_calls: list[tuple[Path, str, bool]] = field(default_factory=list)
    call_threads: list[threading.Thread] = field(default_factory=list)

    def up(
        self, compose_file: Path, project_name: str, env: Mapping[str, str] | None = None
    ) -> None:
        self.call_threads.append(threading.current_thread())
        self.up_calls.append((compose_file, project_name, env))
        self.running = True

    def down(
        self, compose_file: Path, project_name: str, *, remove_volumes: bool = False
    ) -> None:
        self.call_threads.append(threading.current_thread())
        self.down_calls.append((compose_file, project_name, remove_volumes))
        self.running = False

    def is_running(self, compose_file: Path, project_name: str) -> bool:
        self.call_threads.append(threading.current_thread())
        return self.running


@dataclass
class FakeHealthChecker:
    healthy: bool = True
    probed_urls: list[str] = field(default_factory=list)

    async def is_healthy(self, url: str) -> bool:
        self.probed_urls.append(url)
        return self.healthy


def containerized_backend(**overrides: object) -> BackendSpec:
    fields: dict[str, object] = {
        "id": "03-production-rag-reference",
        "name": "Production RAG Reference",
        "level": "intermediate",
        "summary": "x",
        "path": "projects/03-production-rag-reference",
        "needs_container": True,
        "compose_file": "docker-compose.yml",
        "base_url": "http://localhost:9003",
        "health_path": "/healthz",
        "compatible_vector_stores": ["chroma"],
    }
    fields.update(overrides)
    return BackendSpec(**fields)  # type: ignore[arg-type]


def in_process_backend(**overrides: object) -> BackendSpec:
    fields: dict[str, object] = {
        "id": "01-rag-from-scratch",
        "name": "RAG From Scratch",
        "level": "basic",
        "summary": "x",
        "path": "projects/01-rag-from-scratch",
        "needs_container": False,
        "base_url": "http://localhost:9001",
        "health_path": "/",
    }
    fields.update(overrides)
    return BackendSpec(**fields)  # type: ignore[arg-type]


def chroma_store(**overrides: object) -> StandaloneServiceSpec:
    fields: dict[str, object] = {
        "id": "chroma",
        "name": "Chroma",
        "compose_path": "platform/vectorstores/chroma/docker-compose.yml",
        "host": "localhost",
        "port": 8000,
        "health_path": "/api/v1/heartbeat",
    }
    fields.update(overrides)
    return StandaloneServiceSpec(**fields)  # type: ignore[arg-type]


def embedding_service(**overrides: object) -> StandaloneServiceSpec:
    fields: dict[str, object] = {
        "id": "00-embedding-service",
        "name": "Embedding Service",
        "compose_path": "projects/00-embedding-service/docker-compose.yml",
        "host": "localhost",
        "port": 9100,
        "health_path": "/healthz",
    }
    fields.update(overrides)
    return StandaloneServiceSpec(**fields)  # type: ignore[arg-type]


class TestBackendLifecycle:
    @pytest.mark.asyncio
    async def test_start_resolves_path_and_forwards_env(self) -> None:
        runtime = FakeRuntime()
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.start_backend(containerized_backend(), env={"VECTOR_STORE_HOST": "chroma"})

        (call,) = runtime.up_calls
        assert call[0] == REPO_ROOT / "projects/03-production-rag-reference" / "docker-compose.yml"
        assert call[1] == "03-production-rag-reference"
        assert call[2] == {"VECTOR_STORE_HOST": "chroma"}

    @pytest.mark.asyncio
    async def test_start_is_a_no_op_for_an_in_process_backend(self) -> None:
        runtime = FakeRuntime()
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.start_backend(in_process_backend())

        assert runtime.up_calls == []

    @pytest.mark.asyncio
    async def test_stop_uses_the_backends_own_id_as_project_name(self) -> None:
        runtime = FakeRuntime(running=True)
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.stop_backend(containerized_backend())

        (call,) = runtime.down_calls
        assert call[1] == "03-production-rag-reference"
        assert call[2] is False

    @pytest.mark.asyncio
    async def test_reset_tears_down_with_volumes_then_brings_back_up(self) -> None:
        runtime = FakeRuntime(running=True)
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.reset_backend(containerized_backend(), env={"X": "1"})

        assert runtime.down_calls == [
            (
                REPO_ROOT / "projects/03-production-rag-reference" / "docker-compose.yml",
                "03-production-rag-reference",
                True,
            )
        ]
        assert runtime.up_calls == [
            (
                REPO_ROOT / "projects/03-production-rag-reference" / "docker-compose.yml",
                "03-production-rag-reference",
                {"X": "1"},
            )
        ]

    @pytest.mark.asyncio
    async def test_status_is_healthy_for_an_in_process_backend_without_any_runtime_call(
        self,
    ) -> None:
        runtime = FakeRuntime()
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        state = await service.get_backend_status(in_process_backend())

        assert state is ComponentState.HEALTHY
        assert runtime.up_calls == []
        assert runtime.down_calls == []

    @pytest.mark.asyncio
    async def test_status_is_stopped_when_not_running(self) -> None:
        runtime = FakeRuntime(running=False)
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        assert await service.get_backend_status(containerized_backend()) is ComponentState.STOPPED

    @pytest.mark.asyncio
    async def test_status_is_starting_when_running_but_not_yet_healthy(self) -> None:
        runtime = FakeRuntime(running=True)
        health = FakeHealthChecker(healthy=False)
        service = LifecycleService(runtime, health, repo_root=REPO_ROOT)

        spec = containerized_backend()
        assert await service.get_backend_status(spec) is ComponentState.STARTING
        assert health.probed_urls == [spec.base_url + spec.health_path]

    @pytest.mark.asyncio
    async def test_status_is_healthy_when_running_and_health_check_passes(self) -> None:
        runtime = FakeRuntime(running=True)
        service = LifecycleService(runtime, FakeHealthChecker(healthy=True), repo_root=REPO_ROOT)

        assert await service.get_backend_status(containerized_backend()) is ComponentState.HEALTHY


class TestVectorStoreLifecycle:
    @pytest.mark.asyncio
    async def test_start_resolves_compose_path_and_uses_its_own_id_as_project_name(self) -> None:
        runtime = FakeRuntime()
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.start_standalone_service(chroma_store())

        (call,) = runtime.up_calls
        assert call[0] == REPO_ROOT / "platform/vectorstores/chroma/docker-compose.yml"
        assert call[1] == "chroma"

    @pytest.mark.asyncio
    async def test_reset_tears_down_with_volumes_then_brings_back_up(self) -> None:
        runtime = FakeRuntime(running=True)
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.reset_standalone_service(chroma_store())

        assert runtime.down_calls[0][1] == "chroma"
        assert runtime.down_calls[0][2] is True
        assert len(runtime.up_calls) == 1

    @pytest.mark.asyncio
    async def test_status_is_healthy_once_running_when_store_has_no_health_path(
        self,
    ) -> None:
        runtime = FakeRuntime(running=True)
        health = FakeHealthChecker()
        service = LifecycleService(runtime, health, repo_root=REPO_ROOT)

        state = await service.get_standalone_service_status(chroma_store(health_path=""))

        assert state is ComponentState.HEALTHY
        assert health.probed_urls == []  # no HTTP probe needed when there's no health path

    @pytest.mark.asyncio
    async def test_status_probes_health_url_built_from_host_and_port(self) -> None:
        runtime = FakeRuntime(running=True)
        health = FakeHealthChecker(healthy=True)
        service = LifecycleService(runtime, health, repo_root=REPO_ROOT)

        spec = chroma_store()
        await service.get_standalone_service_status(spec)

        assert health.probed_urls == [f"http://{spec.host}:{spec.port}{spec.health_path}"]

    @pytest.mark.asyncio
    async def test_status_is_stopped_when_not_running(self) -> None:
        runtime = FakeRuntime(running=False)
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        assert (
            await service.get_standalone_service_status(chroma_store()) is ComponentState.STOPPED
        )


class TestEmbeddingServiceLifecycle:
    @pytest.mark.asyncio
    async def test_start_resolves_compose_path_and_uses_its_own_id_as_project_name(self) -> None:
        runtime = FakeRuntime()
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.start_standalone_service(embedding_service())

        (call,) = runtime.up_calls
        assert call[0] == REPO_ROOT / "projects/00-embedding-service/docker-compose.yml"
        assert call[1] == "00-embedding-service"

    @pytest.mark.asyncio
    async def test_reset_tears_down_with_volumes_then_brings_back_up(self) -> None:
        runtime = FakeRuntime(running=True)
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.reset_standalone_service(embedding_service())

        assert runtime.down_calls[0][1] == "00-embedding-service"
        assert runtime.down_calls[0][2] is True
        assert len(runtime.up_calls) == 1

    @pytest.mark.asyncio
    async def test_status_probes_health_url_built_from_host_and_port(self) -> None:
        runtime = FakeRuntime(running=True)
        health = FakeHealthChecker(healthy=True)
        service = LifecycleService(runtime, health, repo_root=REPO_ROOT)

        spec = embedding_service()
        await service.get_standalone_service_status(spec)

        assert health.probed_urls == [f"http://{spec.host}:{spec.port}{spec.health_path}"]

    @pytest.mark.asyncio
    async def test_status_is_stopped_when_not_running(self) -> None:
        runtime = FakeRuntime(running=False)
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        assert (
            await service.get_standalone_service_status(embedding_service())
            is ComponentState.STOPPED
        )


class TestDoesNotBlockTheEventLoop:
    """Guards against docker-compose subprocess calls stalling the orchestrator.

    Each runtime call must happen off the calling (main/event-loop) thread,
    so one slow `docker compose` invocation can't starve every other
    concurrent request the orchestrator is serving.
    """

    @pytest.mark.asyncio
    async def test_start_runs_off_the_calling_thread(self) -> None:
        runtime = FakeRuntime()
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.start_backend(containerized_backend())

        (thread,) = runtime.call_threads
        assert thread is not threading.current_thread()

    @pytest.mark.asyncio
    async def test_stop_runs_off_the_calling_thread(self) -> None:
        runtime = FakeRuntime(running=True)
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.stop_backend(containerized_backend())

        (thread,) = runtime.call_threads
        assert thread is not threading.current_thread()

    @pytest.mark.asyncio
    async def test_status_check_runs_off_the_calling_thread(self) -> None:
        runtime = FakeRuntime(running=True)
        service = LifecycleService(runtime, FakeHealthChecker(), repo_root=REPO_ROOT)

        await service.get_backend_status(containerized_backend())

        (thread,) = runtime.call_threads
        assert thread is not threading.current_thread()
