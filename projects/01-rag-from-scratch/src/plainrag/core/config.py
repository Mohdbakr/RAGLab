"""Centralized settings — the only place environment/config gets read
directly; everything else receives plain values as arguments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, resolved from environment variables and .env.

    All fields can be overridden via a `PLAINRAG_<FIELD_NAME>` environment
    variable or a `.env` file in the working directory.
    """

    model_config = SettingsConfigDict(env_prefix="PLAINRAG_", env_file=".env", extra="ignore")

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    chunk_size: int = 200
    chunk_overlap: int = 20
    retrieval_k: int = 4
    index_path: Path = Path(".plainrag_index.json")
    log_level: str = "INFO"
    log_file: Path | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance, built once and cached."""
    return Settings()
