"""Ties the catalog, a ContainerRuntime, and a HealthChecker together.

This is the one place that knows how to turn a catalog entry into an
actual running (or stopped, or reset) stack — routers stay thin and just
call through here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

from app.catalog.models import BackendSpec, StandaloneServiceSpec
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

    def _standalone_service_compose_path(self, spec: StandaloneServiceSpec) -> Path:
        return self._repo_root / spec.compose_path

    # -- generic start/stop/reset/status, shared by every catalog kind --
    #
    # project_name is always the catalog spec's own id, passed explicitly
    # as Compose's `-p` rather than trusting each compose file's own
    # `name:` key — two files can otherwise declare (or default to) the
    # same project name and get treated as one Compose project, so
    # starting/stopping/checking one leaks into the other's containers.
    #
    # Every runtime call runs `docker compose` as a blocking subprocess,
    # so each is offloaded via asyncio.to_thread rather than called
    # inline — otherwise one slow compose invocation (an image pull, a
    # teardown, a status probe) would stall the orchestrator's single
    # event loop and every other request it's serving.

    async def _start(
        self, compose_path: Path, project_name: str, env: Mapping[str, str] | None
    ) -> None:
        await asyncio.to_thread(self._runtime.up, compose_path, project_name, env=env)

    async def _stop(self, compose_path: Path, project_name: str) -> None:
        await asyncio.to_thread(self._runtime.down, compose_path, project_name)

    async def _reset(
        self, compose_path: Path, project_name: str, env: Mapping[str, str] | None
    ) -> None:
        await asyncio.to_thread(
            self._runtime.down, compose_path, project_name, remove_volumes=True
        )
        await asyncio.to_thread(self._runtime.up, compose_path, project_name, env=env)

    async def _status(
        self, compose_path: Path, project_name: str, health_url: str | None
    ) -> ComponentState:
        is_running = await asyncio.to_thread(self._runtime.is_running, compose_path, project_name)
        if not is_running:
            return ComponentState.STOPPED
        if health_url is None:
            return ComponentState.HEALTHY
        healthy = await self._health.is_healthy(health_url)
        return ComponentState.HEALTHY if healthy else ComponentState.STARTING

    # -- backends ----------------------------------------------------------

    async def start_backend(self, spec: BackendSpec, env: Mapping[str, str] | None = None) -> None:
        """Start a backend's stack, unless it runs in-process.

        Args:
            spec: The backend to start.
            env: Extra environment (e.g. the selected vector store's
                host/port) made available to the compose file.
        """
        if not spec.needs_container:
            return
        await self._start(self._backend_compose_path(spec), spec.id, env)

    async def stop_backend(self, spec: BackendSpec) -> None:
        """Stop a backend's stack, unless it runs in-process.

        Args:
            spec: The backend to stop.
        """
        if not spec.needs_container:
            return
        await self._stop(self._backend_compose_path(spec), spec.id)

    async def reset_backend(self, spec: BackendSpec, env: Mapping[str, str] | None = None) -> None:
        """Wipe a backend's persisted state and bring it back up clean.

        Args:
            spec: The backend to reset.
            env: Extra environment to apply when it comes back up.
        """
        if not spec.needs_container:
            return
        await self._reset(self._backend_compose_path(spec), spec.id, env)

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

    # -- standalone services (vector stores, embedding services, ...) ---
    #
    # These catalog kinds share one shape (StandaloneServiceSpec) and one
    # lifecycle, independent of any one backend, so there's a single set
    # of methods rather than a near-identical copy per kind.

    async def start_standalone_service(self, spec: StandaloneServiceSpec) -> None:
        """Start a standalone service's stack.

        Args:
            spec: The service to start.
        """
        await self._start(self._standalone_service_compose_path(spec), spec.id, None)

    async def stop_standalone_service(self, spec: StandaloneServiceSpec) -> None:
        """Stop a standalone service's stack.

        Args:
            spec: The service to stop.
        """
        await self._stop(self._standalone_service_compose_path(spec), spec.id)

    async def reset_standalone_service(self, spec: StandaloneServiceSpec) -> None:
        """Wipe a standalone service's data and bring it back up clean.

        Args:
            spec: The service to reset.
        """
        await self._reset(self._standalone_service_compose_path(spec), spec.id, None)

    async def get_standalone_service_status(self, spec: StandaloneServiceSpec) -> ComponentState:
        """Report a standalone service's current lifecycle state.

        Args:
            spec: The service to check.

        Returns:
            ``STOPPED`` if no container is running; ``HEALTHY`` once
            running if the service declares no ``health_path`` to probe;
            otherwise ``STARTING``/``HEALTHY`` based on that probe.
        """
        health_url = (
            f"http://{spec.host}:{spec.port}{spec.health_path}" if spec.health_path else None
        )
        return await self._status(
            self._standalone_service_compose_path(spec), spec.id, health_url
        )
