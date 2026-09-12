"""RAGLab Orchestrator: the FastAPI control plane behind the unified launcher."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from raglab_common import configure_logging

from app.catalog.loader import load_backends, load_vectorstores
from app.core.config import get_settings
from app.routers import backends as backends_router
from app.routers import vectorstores as vectorstores_router
from app.runtime.docker_compose_runtime import DockerComposeRuntime
from app.runtime.health import HealthChecker
from app.services.lifecycle import LifecycleService

log = configure_logging(project_id="orchestrator")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the catalog and wire up the lifecycle service once at startup."""
    settings = get_settings()
    app.state.backends = load_backends(settings.catalog_dir / "backends.yaml")
    app.state.vectorstores = load_vectorstores(settings.catalog_dir / "vectorstores.yaml")
    app.state.lifecycle_service = LifecycleService(
        runtime=DockerComposeRuntime(),
        health_checker=HealthChecker(),
        repo_root=settings.repo_root,
    )
    log.info(
        "orchestrator ready: {} backend(s), {} vector store(s) in the catalog",
        len(app.state.backends),
        len(app.state.vectorstores),
    )
    yield


app = FastAPI(
    title="RAGLab Orchestrator",
    description="Docker lifecycle control plane for the RAGLab unified launcher.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(backends_router.router)
app.include_router(vectorstores_router.router)


@app.get("/healthz", tags=["Health"])
async def healthz() -> dict[str, str]:
    """Liveness check for the orchestrator itself."""
    return {"status": "ok"}
