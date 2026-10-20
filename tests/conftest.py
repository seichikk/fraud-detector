import os
from collections.abc import AsyncIterator, Iterator

import pytest
from fakes import DEAD_DATABASE, DEAD_RABBIT
from fastapi import FastAPI
from faststream.rabbit import TestRabbitBroker
from httpx import ASGITransport, AsyncClient

from fraud_detector.core.config import Settings
from fraud_detector.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url=DEAD_DATABASE,
        rabbit_url=DEAD_RABBIT,
        health_timeout_seconds=0.5,
    )


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    application = create_app(settings)
    async with (
        TestRabbitBroker(application.state.broker),
        application.router.lifespan_context(application),
    ):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest.fixture
def database_url() -> str:
    url = os.getenv("TEST_DATABASE_URL")
    if url is None:
        pytest.skip("не задан TEST_DATABASE_URL, тест с живой базой пропущен")
    return url


@pytest.fixture(scope="session")
def rabbit_url() -> Iterator[str]:
    url = os.getenv("TEST_RABBIT_URL")
    if url is not None:
        yield url
        return

    try:
        from testcontainers.core.container import DockerContainer
        from testcontainers.core.wait_strategies import LogMessageWaitStrategy

        ready = LogMessageWaitStrategy("Server startup complete").with_startup_timeout(120)
        container = (
            DockerContainer("rabbitmq:4.1-management").with_exposed_ports(5672).waiting_for(ready)
        )
        container.start()
    except Exception as exc:
        pytest.skip(f"не удалось поднять rabbitmq через testcontainers: {exc}")

    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5672)
        yield f"amqp://guest:guest@{host}:{port}/"
    finally:
        container.stop()
