import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from faststream.rabbit import RabbitBroker

from fraud_detector.api import system
from fraud_detector.api.v1 import router as api_v1_router
from fraud_detector.core.config import Settings, get_settings
from fraud_detector.core.logs import setup_logging
from fraud_detector.core.meta import DISTRIBUTION_NAME, get_app_version
from fraud_detector.core.middleware import log_requests
from fraud_detector.db.engine import create_engine
from fraud_detector.db.tasks import ensure_schema
from fraud_detector.messaging import declare_topology

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    setup_logging(app_settings.log_level)
    broker = RabbitBroker(app_settings.rabbit_url, log_level=logging.DEBUG)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = create_engine(app_settings.database_url)
        try:
            await ensure_schema(app.state.engine)
        except Exception as exc:
            logger.warning("не удалось подготовить таблицу задач: %r", exc)

        await broker.connect()
        await declare_topology(broker, app_settings)
        logger.info("сервис запущен, версия %s", app.version)
        try:
            yield
        finally:
            await broker.stop()
            await app.state.engine.dispose()
            logger.info("сервис остановлен, соединения закрыты")

    app = FastAPI(title=DISTRIBUTION_NAME, version=get_app_version(), lifespan=lifespan)
    app.state.settings = app_settings
    app.state.broker = broker

    app.middleware("http")(log_requests)
    app.include_router(system.router)
    app.include_router(api_v1_router)

    return app


app = create_app()


def run() -> None:
    settings = get_settings()

    uvicorn.run(
        app, host=settings.app_host, port=settings.app_port, log_config=None, access_log=False
    )
