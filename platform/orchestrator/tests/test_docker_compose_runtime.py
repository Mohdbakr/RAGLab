"""Tests for app.runtime.docker_compose_runtime, written before the implementation.

No real Docker daemon is involved: `subprocess.run` is replaced with a fake
that records every invocation and returns a scripted result.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from app.runtime.docker_compose_runtime import (
    DockerComposeCommandError,
    DockerComposeRuntime,
)


class FakeRunner:
    """Stand-in for subprocess.run that records calls and returns a script."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            args=command, returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
        )


@pytest.fixture
def compose_file(tmp_path: Path) -> Path:
    path = tmp_path / "project" / "docker-compose.yml"
    path.parent.mkdir(parents=True)
    path.write_text("services: {}\n")
    return path


class TestUp:
    def test_runs_docker_compose_up_dash_d_in_the_compose_files_directory(
        self, compose_file: Path
    ) -> None:
        runner = FakeRunner()
        runtime = DockerComposeRuntime(runner=runner)

        runtime.up(compose_file)

        (call,) = runner.calls
        assert call["command"] == [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "up",
            "-d",
        ]
        assert call["cwd"] == compose_file.parent

    def test_merges_extra_env_on_top_of_the_process_environment(
        self, compose_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXISTING_VAR", "kept")
        runner = FakeRunner()
        runtime = DockerComposeRuntime(runner=runner)

        runtime.up(compose_file, env={"VECTOR_STORE_HOST": "chroma"})

        (call,) = runner.calls
        assert call["env"]["VECTOR_STORE_HOST"] == "chroma"
        assert call["env"]["EXISTING_VAR"] == "kept"

    def test_raises_on_nonzero_exit(self, compose_file: Path) -> None:
        runner = FakeRunner(returncode=1, stderr="boom")
        runtime = DockerComposeRuntime(runner=runner)

        with pytest.raises(DockerComposeCommandError, match="boom"):
            runtime.up(compose_file)


class TestDown:
    def test_without_remove_volumes_omits_dash_v(self, compose_file: Path) -> None:
        runner = FakeRunner()
        runtime = DockerComposeRuntime(runner=runner)

        runtime.down(compose_file)

        (call,) = runner.calls
        assert "-v" not in call["command"]

    def test_with_remove_volumes_appends_dash_v(self, compose_file: Path) -> None:
        runner = FakeRunner()
        runtime = DockerComposeRuntime(runner=runner)

        runtime.down(compose_file, remove_volumes=True)

        (call,) = runner.calls
        assert call["command"][-1] == "-v"

    def test_raises_on_nonzero_exit(self, compose_file: Path) -> None:
        runner = FakeRunner(returncode=1, stderr="down failed")
        runtime = DockerComposeRuntime(runner=runner)

        with pytest.raises(DockerComposeCommandError, match="down failed"):
            runtime.down(compose_file)


class TestIsRunning:
    def test_true_when_ps_lists_at_least_one_service(self, compose_file: Path) -> None:
        runner = FakeRunner(returncode=0, stdout="backend\n")
        runtime = DockerComposeRuntime(runner=runner)

        assert runtime.is_running(compose_file) is True

    def test_false_when_ps_lists_no_services(self, compose_file: Path) -> None:
        runner = FakeRunner(returncode=0, stdout="")
        runtime = DockerComposeRuntime(runner=runner)

        assert runtime.is_running(compose_file) is False

    def test_false_on_nonzero_exit_rather_than_raising(self, compose_file: Path) -> None:
        runner = FakeRunner(returncode=1, stderr="no such project")
        runtime = DockerComposeRuntime(runner=runner)

        assert runtime.is_running(compose_file) is False
