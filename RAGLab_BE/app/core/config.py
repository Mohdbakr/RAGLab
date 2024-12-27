from pydantic_settings import BaseSettings, SettingsConfigDict

LOGGING_CONFIG = {"level": "INFO", "format": "%(asctime)s [%(levelname)s] %(message)s"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    URI: str
    TOKEN: str
    DB_NAME: str
    COLLECTION_NAME: str
    VECTOR_DIM: int

    OPENAI_API_KEY: str


settings = Settings()

MILVUS_CONFIG = {}
MILVUS_CONFIG["URI"] = settings.URI
MILVUS_CONFIG["TOKEN"] = settings.TOKEN
MILVUS_CONFIG["DB_NAME"] = settings.DB_NAME
MILVUS_CONFIG["COLLECTION_NAME"] = settings.COLLECTION_NAME
MILVUS_CONFIG["VECTOR_DIM"] = settings.VECTOR_DIM
