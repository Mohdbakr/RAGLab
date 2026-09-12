"""Ties the catalog, a ContainerRuntime, and a HealthChecker together.

This is the one place that knows how to turn a catalog entry into an
actual running (or stopped, or reset) stack — routers stay thin and just
call through here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from app.catalog.models import BackendSpec, VectorStoreSpec
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

    # -- generic start/stop/reset/status, shared by both catalog kinds --

    def _start(self, compose_path: Path, env: Mapping[str, str] | None) -> None:
        self._runtime.up(compose_path, env=env)

    def _stop(self, compose_path: Path) -> None:
        self._runtime.down(compose_path)

    def _reset(self, compose_path: Path, env: Mapping[str, str] | None) -> None:
        self._runtime.down(compose_path, remove_volumes=True)
        self._runtime.up(compose_path, env=env)

    async def _status(self, compose_path: Path, health_url: str | None) -> ComponentState:
        if not self._runtime.is_running(compose_path):
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
        self._start(self._backend_compose_path(spec), env)

    def stop_backend(self, spec: BackendSpec) -> None:
        """Stop a backend's stack, unless it runs in-process.

        Args:
            spec: The backend to stop.
        """
        if not spec.needs_container:
            return
        self._stop(self._backend_compose_path(spec))

    def reset_backend(self, spec: BackendSpec, env: Mapping[str, str] | None = None) -> None:
        """Wipe a backend's persisted state and bring it back up clean.

        Args:
            spec: The backend to reset.
            env: Extra environment to apply when it comes back up.
        """
        if not spec.needs_container:
            return
        self._reset(self._backend_compose_path(spec), env)

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
        return await self._status(self._backend_compose_path(spec), health_url)

    # -- vector stores -------------------------------------------------

    def start_vector_store(self, spec: VectorStoreSpec) -> None:
        """Start a vector store's stack.

        Args:
            spec: The vector store to start.
        """
        self._start(self._vector_store_compose_path(spec), None)

    def stop_vector_store(self, spec: VectorStoreSpec) -> None:
        """Stop a vector store's stack.

        Args:
            spec: The vector store to stop.
        """
        self._stop(self._vector_store_compose_path(spec))

    def reset_vector_store(self, spec: VectorStoreSpec) -> None:
        """Wipe a vector store's data and bring it back up clean.

        Args:
            spec: The vector store to reset.
        """
        self._reset(self._vector_store_compose_path(spec), None)

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
        return await self._status(self._vector_store_compose_path(spec), health_url)
