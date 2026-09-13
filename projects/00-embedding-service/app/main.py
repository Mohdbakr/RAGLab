"""Embedding Service: a standalone, model-swappable embedding microservice."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from raglab_common import BenchmarkLogger, configure_logging

from app.core.client_cache import EmbeddingClientCache
from app.core.config import get_settings_singleton
from app.routers.embeddings import router

log = configure_logging(project_id="00-embedding-service")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Wire up settings, the client cache, and the benchmark logger once at startup."""
    settings = get_settings_singleton()
    app.state.settings = settings
    app.state.client_cache = EmbeddingClientCache()
    app.state.benchmark_logger = BenchmarkLogger(project_id="00-embedding-service")
    log.info("embedding service ready, default model {}", settings.default_embedding_model)
    yield


app = FastAPI(
    title="RAGLab Embedding Service",
    description=(
        "Standalone, model-swappable embedding microservice — a production "
        "pattern other RAGLab projects can optionally call over HTTP."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)
