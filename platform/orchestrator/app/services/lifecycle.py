"""Ties the catalog, a ContainerRuntime, and a HealthChecker together.

This is the one place that knows how to turn a catalog entry into an
actual running (or stopped, or reset) stack — routers stay thin and just
call through here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from app.catalog.models import BackendSpec, EmbeddingServiceSpec, VectorStoreSpec
from app.models.status import ComponentState
from app.runtime.protocol import ContainerRuntime, HealthCheck


class LifecycleService:
    """Starts, stops, resets, and reports status for catalog components."""

    def __init__(
        self,
        runtime: ContainerRuntime,
        health_checker: HealthCheck,
        repo_root: Path,
    ) -> None:
        """Create a lifecycle service.

        Args:
            runtime: Drives the actual container stack.
            health_checker: Probes a running stack's HTTP health endpoint.
            repo_root: Absolute path to the repo root, used to resolve the
                catalog's repo-relative paths.
        """
        self._runtime = runtime
        self._health = health_checker
        self._repo_root = repo_root

    # -- path resolution -------------------------------------------------

    def _backend_compose_path(self, spec: BackendSpec) -> Path:
        if spec.compose_file is None:
            raise ValueError(f"backend {spec.id!r} has no compose_file to resolve")
        return self._repo_root / spec.path / spec.compose_file

    def _vector_store_compose_path(self, spec: VectorStoreSpec) -> Path:
        return self._repo_root / spec.compose_path

    def _embedding_service_compose_path(self, spec: EmbeddingServiceSpec) -> Path:
        return self._repo_root / spec.compose_path

    # -- generic start/stop/reset/status, shared by every catalog kind --
    #
    # project_name is always the catalog spec's own id, passed explicitly
    # as Compose's `-p` rather than trusting each compose file's own
    # `name:` key — two files can otherwise declare (or default to) the
    # same project name and get treated as one Compose project, so
    # starting/stopping/checking one leaks into the other's containers.

    def _start(self, compose_path: Path, project_name: str, env: Mapping[str, str] | None) -> None:
        self._runtime.up(compose_path, project_name, env=env)

    def _stop(self, compose_path: Path, project_name: str) -> None:
        self._runtime.down(compose_path, project_name)

    def _reset(self, compose_path: Path, project_name: str, env: Mapping[str, str] | None) -> None:
        self._runtime.down(compose_path, project_name, remove_volumes=True)
        self._runtime.up(compose_path, project_name, env=env)

    async def _status(
        self, compose_path: Path, project_name: str, health_url: str | None
    ) -> ComponentState:
        if not self._runtime.is_running(compose_path, project_name):
            return ComponentState.STOPPED
        if health_url is None:
            return ComponentState.HEALTHY
        healthy = await self._health.is_healthy(health_url)
        return ComponentState.HEALTHY if healthy else ComponentState.STARTING

    # -- backends ----------------------------------------------------------

    def start_backend(self, spec: BackendSpec, env: Mapping[str, str] | None = None) -> None:
        """Start a backend's stack, unless it runs in-process.

        Args:
            spec: The backend to start.
            env: Extra environment (e.g. the selected vector store's
                host/port) made available to the compose file.
        """
        if not spec.needs_container:
            return
        self._start(self._backend_compose_path(spec), spec.id, env)

    def stop_backend(self, spec: BackendSpec) -> None:
        """Stop a backend's stack, unless it runs in-process.

        Args:
            spec: The backend to stop.
        """
        if not spec.needs_container:
            return
        self._stop(self._backend_compose_path(spec), spec.id)

    def reset_backend(self, spec: BackendSpec, env: Mapping[str, str] | None = None) -> None:
        """Wipe a backend's persisted state and bring it back up clean.

        Args:
            spec: The backend to reset.
            env: Extra environment to apply when it comes back up.
        """
        if not spec.needs_container:
            return
        self._reset(self._backend_compose_path(spec), spec.id, env)

    async def get_backend_status(self, spec: BackendSpec) -> ComponentState:
        """Report a backend's current lifecycle state.

        Args:
            spec: The backend to check.

        Returns:
            ``HEALTHY`` immediately for an in-process backend; otherwise
            ``STOPPED``/``STARTING``/``HEALTHY`` based on container and
            health-endpoint state.
        """
        if not spec.needs_container:
            return ComponentState.HEALTHY
        health_url = spec.base_url + spec.health_path
        return await self._status(self._backend_compose_path(spec), spec.id, health_url)

    # -- vector stores -------------------------------------------------

    def start_vector_store(self, spec: VectorStoreSpec) -> None:
        """Start a vector store's stack.

        Args:
            spec: The vector store to start.
        """
        self._start(self._vector_store_compose_path(spec), spec.id, None)

    def stop_vector_store(self, spec: VectorStoreSpec) -> None:
        """Stop a vector store's stack.

        Args:
            spec: The vector store to stop.
        """
        self._stop(self._vector_store_compose_path(spec), spec.id)

    def reset_vector_store(self, spec: VectorStoreSpec) -> None:
        """Wipe a vector store's data and bring it back up clean.

        Args:
            spec: The vector store to reset.
        """
        self._reset(self._vector_store_compose_path(spec), spec.id, None)

    async def get_vector_store_status(self, spec: VectorStoreSpec) -> ComponentState:
        """Report a vector store's current lifecycle state.

        Args:
            spec: The vector store to check.

        Returns:
            ``STOPPED`` if no container is running; ``HEALTHY`` once
            running if the store declares no ``health_path`` to probe;
            otherwise ``STARTING``/``HEALTHY`` based on that probe.
        """
        health_url = (
            f"http://{spec.host}:{spec.port}{spec.health_path}" if spec.health_path else None
        )
        return await self._status(self._vector_store_compose_path(spec), spec.id, health_url)

    # -- embedding services ----------------------------------------------

    def start_embedding_service(self, spec: EmbeddingServiceSpec) -> None:
        """Start an embedding service's stack.

        Args:
            spec: The embedding service to start.
        """
        self._start(self._embedding_service_compose_path(spec), spec.id, None)

    def stop_embedding_service(self, spec: EmbeddingServiceSpec) -> None:
        """Stop an embedding service's stack.

        Args:
            spec: The embedding service to stop.
        """
        self._stop(self._embedding_service_compose_path(spec), spec.id)

    def reset_embedding_service(self, spec: EmbeddingServiceSpec) -> None:
        """Wipe an embedding service's data and bring it back up clean.

        Args:
            spec: The embedding service to reset.
        """
        self._reset(self._embedding_service_compose_path(spec), spec.id, None)

    async def get_embedding_service_status(self, spec: EmbeddingServiceSpec) -> ComponentState:
        """Report an embedding service's current lifecycle state.

        Args:
            spec: The embedding service to check.

        Returns:
            ``STOPPED`` if no container is running; ``HEALTHY`` once
            running if the service declares no ``health_path`` to probe;
            otherwise ``STARTING``/``HEALTHY`` based on that probe.
        """
        health_url = (
            f"http://{spec.host}:{spec.port}{spec.health_path}" if spec.health_path else None
        )
        return await self._status(
            self._embedding_service_compose_path(spec), spec.id, health_url
        )
