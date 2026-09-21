import logging
from importlib.metadata import version

import pytest
from httpx import AsyncClient


async def test_healthz_answers_ok(client: AsyncClient) -> None:
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_healthz_lives_outside_api_prefix(client: AsyncClient) -> None:
    response = await client.get("/api/v1/healthz")

    assert response.status_code == 404


async def test_version_is_taken_from_package_metadata(client: AsyncClient) -> None:
    response = await client.get("/api/v1/version")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "fraud-detector"
    assert body["version"] == version("fraud-detector")
    assert body["version"] != "v1"


async def test_request_id_is_echoed_back(client: AsyncClient) -> None:
    response = await client.get("/healthz", headers={"x-request-id": "abc123"})

    assert response.headers["x-request-id"] == "abc123"


async def test_request_id_is_generated_when_missing(client: AsyncClient) -> None:
    response = await client.get("/healthz")

    assert len(response.headers["x-request-id"]) == 32


async def test_every_request_is_logged(
    client: AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    await client.get("/api/v1/version")

    messages = [record.getMessage() for record in caplog.records]
    assert any("GET /api/v1/version -> 200" in message for message in messages)
