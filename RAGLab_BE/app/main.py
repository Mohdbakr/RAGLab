from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from .routers.files import router as files_router
from .core.config import Settings
from .core.logging import setup_logging

settings = Settings()
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Performing startup tasks...")
    # startup logic here

    yield

    logger.info("Performing shutdown tasks...")
    # shutdown logic here


app = FastAPI(
    lifespan=lifespan,
    title=settings.PROJECT_NAME,
    description="This is the backend API for the RAG Lab project.",
    version="0.1.0",
    docs_url=f"{settings.API_V1_STR}/docs",
)

app.include_router(
    files_router, prefix=settings.API_V1_STR, tags=["Documents"]
)


@app.get("/", tags=["Health Check"])
async def root() -> Response:
    return {"status": "ok", "message": "Server is Runnings!"}


if __name__ == "__main__":
    root()
