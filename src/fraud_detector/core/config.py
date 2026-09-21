from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_url: str = "postgresql+asyncpg://fraud:fraud@localhost:5432/fraud"
    health_timeout_seconds: float = 2.0
    git_sha: str = "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
