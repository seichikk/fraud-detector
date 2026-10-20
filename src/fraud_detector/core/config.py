import math
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_url: str = "postgresql+asyncpg://fraud:fraud@localhost:5432/fraud"
    health_timeout_seconds: float = 2.0
    git_sha: str = "local"

    rabbit_url: str = "amqp://guest:guest@localhost:5672/"
    tasks_exchange: str = "tasks"
    tasks_queue: str = "tasks"
    queue_max_length: int = Field(default=100, ge=0)
    max_deliveries: int = Field(default=3, ge=1)
    retry_base_seconds: float = Field(default=0.5, ge=0)
    publish_timeout_seconds: float = Field(default=1.0, gt=0)

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_model_name: str = "FraudDetectionModel"
    mlflow_model_alias: str = "champion"
    model_load_attempts: int = Field(default=10, ge=1)
    model_load_retry_seconds: float = Field(default=2.0, ge=0)
    prediction_threshold: float = Field(default=0.5, ge=0, le=1)

    worker_host: str = "127.0.0.1"
    worker_port: int = 8001
    worker_concurrency: int = Field(default=4, ge=1)
    prefetch_count: int | None = Field(default=None, ge=0)
    inference_seconds: float = Field(default=0.2, ge=0)
    worker_graceful_timeout: float = Field(default=30.0, gt=0)

    source_min_backoff_seconds: float = Field(default=0.01, gt=0)
    source_max_backoff_seconds: float = Field(default=0.5, gt=0)

    @property
    def bounded_queue(self) -> bool:
        return self.queue_max_length > 0

    @property
    def effective_prefetch(self) -> int:
        if self.prefetch_count is None:
            return self.worker_concurrency + 1
        return self.prefetch_count

    @property
    def retry_after_seconds(self) -> int:
        drain_time = self.queue_max_length * self.inference_seconds / self.worker_concurrency
        return max(1, math.ceil(drain_time))


@lru_cache
def get_settings() -> Settings:
    return Settings()
