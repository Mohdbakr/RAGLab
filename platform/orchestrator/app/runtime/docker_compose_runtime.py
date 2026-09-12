"""`docker compose` CLI-backed implementation of ``ContainerRuntime``.

Shells out to the `docker compose` binary rather than the Docker SDK: a
compose file is the natural unit for a project's whole stack (vector DB +
backend, or a multi-service vector store), and the CLI is what's already
installed and what a contributor would run by hand.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class DockerComposeCommandError(RuntimeError):
    """A `docker compose` invocation exited non-zero."""

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        """Capture enough context to debug the failure.

        Args:
            command: The full argv that was run.
            returncode: The process's exit code.
            stderr: Captured standard error output.
        """
        super().__init__(f"`{' '.join(command)}` exited {returncode}: {stderr}")
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class DockerComposeRuntime:
    """Runs `docker compose` for whichever compose file it's given."""

    def __init__(self, runner: CommandRunner = subprocess.run) -> None:
        """Create a runtime.

        Args:
            runner: The function used to execute commands. Defaults to
                :func:`subprocess.run`; tests inject a fake to avoid
                touching a real Docker daemon.
        """
        self._runner = runner

    def up(self, compose_file: Path, env: Mapping[str, str] | None = None) -> None:
        """See :meth:`app.runtime.protocol.ContainerRuntime.up`."""
        merged_env = {**os.environ, **env} if env else None
        self._run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d"],
            compose_file,
            env=merged_env,
        )

    def down(self, compose_file: Path, *, remove_volumes: bool = False) -> None:
        """See :meth:`app.runtime.protocol.ContainerRuntime.down`."""
        command = ["docker", "compose", "-f", str(compose_file), "down"]
        if remove_volumes:
            command.append("-v")
        self._run(command, compose_file, env=None)

    def is_running(self, compose_file: Path) -> bool:
        """See :meth:`app.runtime.protocol.ContainerRuntime.is_running`."""
        result = self._runner(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file),
                "ps",
                "--status",
                "running",
                "--services",
            ],
            cwd=compose_file.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        return bool(result.stdout.strip())

    def _run(
        self, command: list[str], compose_file: Path, *, env: dict[str, str] | None
    ) -> None:
        kwargs: dict[str, Any] = {
            "cwd": compose_file.parent,
            "capture_output": True,
            "text": True,
        }
        if env is not None:
            kwargs["env"] = env
        result = self._runner(command, **kwargs)
        if result.returncode != 0:
            raise DockerComposeCommandError(command, result.returncode, result.stderr)
