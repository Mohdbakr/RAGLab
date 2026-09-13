"""Orchestrator settings: mainly where the repo (and its catalog) live."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_repo_root() -> Path:
    """platform/orchestrator/app/core/config.py -> repo root is four parents up.

    Only valid when this file sits at its real-repo depth, which isn't true
    in the Docker image (COPY flattens it under /app). Kept lazy so it's
    never evaluated there -- docker-compose.yml always sets RAGLAB_REPO_ROOT,
    which pydantic-settings picks up ahead of this default.
    """
    return Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Runtime configuration for the orchestrator service."""

    model_config = SettingsConfigDict(
        env_prefix="RAGLAB_", env_file=".env", extra="ignore"
    )

    repo_root: Path = Field(default_factory=_default_repo_root)

    @property
    def catalog_dir(self) -> Path:
        """Directory holding backends.yaml and vectorstores.yaml."""
        return self.repo_root / "catalog"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()
