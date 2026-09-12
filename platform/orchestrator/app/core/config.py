"""Orchestrator settings: mainly where the repo (and its catalog) live."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# platform/orchestrator/app/core/config.py -> repo root is four parents up.
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Runtime configuration for the orchestrator service."""

    model_config = SettingsConfigDict(env_prefix="RAGLAB_", env_file=".env", extra="ignore")

    repo_root: Path = _DEFAULT_REPO_ROOT

    @property
    def catalog_dir(self) -> Path:
        """Directory holding backends.yaml and vectorstores.yaml."""
        return self.repo_root / "catalog"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()
