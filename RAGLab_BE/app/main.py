from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from .routers.files import router as files_router
from .core.config import Settings
from .core.logging import SingletonLogger

settings = Settings()
logger = SingletonLogger.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Performing startup tasks...")
    # startup logic here
    try:
        logger.info("Loading embedding model...")
        app.state.embedding_model = SentenceTransformer(settings.MODEL_NAME)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        raise
    yield

    logger.info("Performing shutdown tasks...")
    # shutdown logic here


app = FastAPI(
    lifespan=lifespan,
    title=settings.PROJECT_NAME,
    description="This is the backend API for the RAG Lab project.",
    version="0.1.0",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

app.include_router(
    files_router, prefix=settings.API_V1_STR, tags=["Documents"]
)

origins = [
    "http://localhost:8501",
    "http://frontend:8501",
    "https://yourfrontenddomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins from the list
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/", tags=["Health Check"])
async def root() -> Response:
    return {"status": "ok", "message": "Server is Runnings!"}


if __name__ == "__main__":
    root()
