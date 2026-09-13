"""Settings for the embedding service."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the embedding service."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    default_embedding_model: str = "local/all-mpnet-base-v2"


@lru_cache
def get_settings_singleton() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()
