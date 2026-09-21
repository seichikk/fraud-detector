import os

import pytest
from httpx import AsyncClient

from fraud_detector.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    database_url = os.getenv("TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("не задан TEST_DATABASE_URL, тест с живой базой пропущен")
    return Settings(database_url=database_url)


async def test_health_sees_real_postgres(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    postgres = response.json()["components"][0]
    assert postgres["status"] == "ok"
    assert postgres["version"]
