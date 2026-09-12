"""The lifecycle contract any container backend must satisfy.

`DockerComposeRuntime` is the only implementation today; the Protocol
exists so the lifecycle service depends on this interface rather than on
`docker compose` specifically (swappable for a Kubernetes- or docker-py
SDK-based runtime later without touching callers).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol


class ContainerRuntime(Protocol):
    """Starts, stops, and reports on a docker-compose-defined stack."""

    def up(self, compose_file: Path, env: Mapping[str, str] | None = None) -> None:
        """Bring every service in ``compose_file`` up, detached.

        Args:
            compose_file: Path to the compose file to run.
            env: Extra environment variables made available for the
                compose file's ``${VAR}`` interpolation, merged on top of
                the current process environment.

        Raises:
            Exception: If the underlying command fails.
        """
        ...

    def down(self, compose_file: Path, *, remove_volumes: bool = False) -> None:
        """Stop every service in ``compose_file``.

        Args:
            compose_file: Path to the compose file to stop.
            remove_volumes: If true, also delete the stack's volumes — this
                is what a demo Reset uses to guarantee a clean slate.

        Raises:
            Exception: If the underlying command fails.
        """
        ...

    def is_running(self, compose_file: Path) -> bool:
        """Report whether at least one service in the stack is running.

        Args:
            compose_file: Path to the compose file to inspect.

        Returns:
            True if any service is currently running; false if the stack
            is stopped or the check itself fails.
        """
        ...


class HealthCheck(Protocol):
    """The narrow interface LifecycleService needs from a health prober."""

    async def is_healthy(self, url: str) -> bool:
        """Report whether ``url`` responds without a server error.

        Args:
            url: The health-check URL to probe.

        Returns:
            True if the target looks healthy, false otherwise.
        """
        ...
