from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    API_BASE_URL: str
    UPLOAD_API_URL: str
    SEARCH_API_URL: str
    CHAT_API_URL: str


settings = Settings()
