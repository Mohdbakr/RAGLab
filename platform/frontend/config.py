"""Settings for the unified launcher frontend."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Streamlit app."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    orchestrator_base_url: str = "http://localhost:8100"


settings = Settings()
