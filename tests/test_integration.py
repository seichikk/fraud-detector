import pytest
from httpx import AsyncClient

from fraud_detector.core.config import Settings


@pytest.fixture
def settings(database_url: str) -> Settings:
    return Settings(database_url=database_url, rabbit_url="amqp://nobody:nothing@127.0.0.1:1/")


async def test_health_sees_real_postgres(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")

    postgres = response.json()["components"][0]
    assert postgres["status"] == "ok"
    assert postgres["version"]
