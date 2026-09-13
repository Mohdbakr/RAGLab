"""FastAPI dependency accessors for state set up in the app's lifespan.

The router depends on these functions rather than reaching into
``request.app.state`` directly, so tests can override them with fakes.
"""

from __future__ import annotations

from fastapi import Request
from raglab_common import BenchmarkLogger

from app.core.client_cache import EmbeddingClientCache
from app.core.config import Settings


def get_settings(request: Request) -> Settings:
    """Return the settings loaded at startup."""
    return request.app.state.settings  # type: ignore[no-any-return]


def get_client_cache(request: Request) -> EmbeddingClientCache:
    """Return the process-wide embedding client cache."""
    return request.app.state.client_cache  # type: ignore[no-any-return]


def get_benchmark_logger(request: Request) -> BenchmarkLogger:
    """Return the process-wide benchmark logger."""
    return request.app.state.benchmark_logger  # type: ignore[no-any-return]
