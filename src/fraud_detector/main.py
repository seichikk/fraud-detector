import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import uvicorn
from fastapi import FastAPI
from mlflow import MlflowClient

from fraud_detector.api import system
from fraud_detector.api.process import router as process_router
from fraud_detector.api.v1 import router as api_v1_router
from fraud_detector.core.config import Settings, get_settings
from fraud_detector.core.logs import setup_logging
from fraud_detector.core.meta import DISTRIBUTION_NAME, get_app_version
from fraud_detector.core.middleware import log_requests
from fraud_detector.db.engine import create_engine

logger = logging.getLogger(__name__)

MODEL_NAME = "FraudDetectionModel"
MODEL_ALIAS = "champion"


def load_champion_model(tracking_uri: str) -> tuple[object, str]:
    """Загружает текущую версию модели из MLflow Registry."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    last_error: Exception | None = None

    for attempt in range(1, 11):
        try:
            model_version = client.get_model_version_by_alias(
                name=MODEL_NAME,
                alias=MODEL_ALIAS,
            )

            version = str(model_version.version)
            model_uri = f"models:/{MODEL_NAME}/{version}"

            logger.info(
                "Загружаем модель %s, версия %s",
                MODEL_NAME,
                version,
            )

            model = mlflow.sklearn.load_model(model_uri)

            logger.info(
                "Модель %s версии %s успешно загружена",
                MODEL_NAME,
                version,
            )

            return model, version

        except Exception as exc:
            last_error = exc

            logger.warning(
                "Попытка загрузки модели %s из MLflow не удалась (%s/10): %s",
                MODEL_NAME,
                attempt,
                exc,
            )

            if attempt < 10:
                time.sleep(2)

    raise RuntimeError(
        f"Не удалось загрузить модель {MODEL_NAME} с alias {MODEL_ALIAS}"
    ) from last_error


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    setup_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = create_engine(app_settings.database_url)
        app.state.fraud_model = None
        app.state.fraud_model_version = None

        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

            if tracking_uri:
                # Модель загружается один раз при старте приложения.
                model, version = load_champion_model(tracking_uri)

                app.state.fraud_model = model
                app.state.fraud_model_version = version
            else:
                logger.warning(
                    "MLFLOW_TRACKING_URI не задан. "
                    "Предсказания будут недоступны."
                )

            logger.info(
                "Сервис запущен, версия %s",
                app.version,
            )

            yield

        finally:
            await app.state.engine.dispose()
            logger.info("Сервис остановлен, соединения с базой закрыты")

    app = FastAPI(
        title=DISTRIBUTION_NAME,
        version=get_app_version(),
        lifespan=lifespan,
    )

    app.state.settings = app_settings

    app.middleware("http")(log_requests)

    app.include_router(system.router)
    app.include_router(api_v1_router)
    app.include_router(process_router)

    return app


app = create_app()


def run() -> None:
    settings = get_settings()

    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        log_config=None,
        access_log=False,
    )
