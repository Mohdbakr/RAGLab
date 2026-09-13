"""RAGLab Orchestrator: the FastAPI control plane behind the unified launcher."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from raglab_common import configure_logging

from app.catalog.loader import load_backends, load_embedding_services, load_vectorstores
from app.core.config import get_settings
from app.dependencies import get_embedding_services, get_vectorstores
from app.routers import backends as backends_router
from app.routers.standalone_services import build_standalone_service_router
from app.runtime.docker_compose_runtime import DockerComposeCommandError, DockerComposeRuntime
from app.runtime.health import HealthChecker
from app.services.lifecycle import LifecycleService

log = configure_logging(project_id="orchestrator")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the catalog and wire up the lifecycle service once at startup."""
    settings = get_settings()
    app.state.backends = load_backends(settings.catalog_dir / "backends.yaml")
    app.state.vectorstores = load_vectorstores(settings.catalog_dir / "vectorstores.yaml")
    app.state.embedding_services = load_embedding_services(
        settings.catalog_dir / "embedding_services.yaml"
    )
    app.state.lifecycle_service = LifecycleService(
        runtime=DockerComposeRuntime(),
        health_checker=HealthChecker(),
        repo_root=settings.repo_root,
    )
    log.info(
        "orchestrator ready: {} backend(s), {} vector store(s), {} embedding service(s) "
        "in the catalog",
        len(app.state.backends),
        len(app.state.vectorstores),
        len(app.state.embedding_services),
    )
    yield


app = FastAPI(
    title="RAGLab Orchestrator",
    description="Docker lifecycle control plane for the RAGLab unified launcher.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(backends_router.router)
app.include_router(
    build_standalone_service_router(
        prefix="/vectorstores",
        tag="Vector Stores",
        kind_label="vector store",
        get_specs=get_vectorstores,
    )
)
app.include_router(
    build_standalone_service_router(
        prefix="/embedding-services",
        tag="Embedding Services",
        kind_label="embedding service",
        get_specs=get_embedding_services,
    )
)


@app.exception_handler(DockerComposeCommandError)
async def handle_docker_compose_command_error(
    request: Request, exc: DockerComposeCommandError
) -> JSONResponse:
    """Surface a failed `docker compose` invocation as a clear error.

    Without this, a failure (a missing .env file, a bad Dockerfile, ...)
    propagates as a bare, undetailed 500 that the frontend can only show
    as a raw traceback. 502 signals "the orchestrator's downstream
    command failed", with the actual command/stderr in the body so the
    frontend can show the real reason.
    """
    log.warning("docker compose command failed: {}", exc)
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/healthz", tags=["Health"])
async def healthz() -> dict[str, str]:
    """Liveness check for the orchestrator itself."""
    return {"status": "ok"}
