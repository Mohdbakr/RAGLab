from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "RAG Lab Backend"

    # Milvus Configuration
    MILVUS_URI: str
    MILVUS_TOKEN: str
    MILVUS_DB_NAME: str
    MILVUS_COLLECTION_NAME: str
    MILVUS_VECTOR_DIM: int

    # Embedding Model Configuration
    MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"

    # CORS Configuration
    BACKEND_CORS_ORIGINS: list = ["http://localhost:8501"]

    # OPENAI API Configuration
    OPENAI_API_KEY: str

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = "logs/app.log"

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()

MILVUS_CONFIG = {}
MILVUS_CONFIG["URI"] = settings.MILVUS_URI
MILVUS_CONFIG["TOKEN"] = settings.MILVUS_TOKEN
MILVUS_CONFIG["DB_NAME"] = settings.MILVUS_DB_NAME
MILVUS_CONFIG["COLLECTION_NAME"] = settings.MILVUS_COLLECTION_NAME
MILVUS_CONFIG["VECTOR_DIM"] = settings.MILVUS_VECTOR_DIM
