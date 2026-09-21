import asyncio
import time
from dataclasses import dataclass

from fastapi import FastAPI
from httpx import AsyncClient

from fraud_detector.api.deps import get_health_checks
from fraud_detector.services.health import HealthCheck, run_checks


@dataclass
class FakeCheck:
    name: str
    version: str = "1.0"
    delay: float = 0.0
    error: Exception | None = None

    async def fetch_version(self) -> str:
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.version


def use_checks(app: FastAPI, *checks: HealthCheck) -> None:
    app.dependency_overrides[get_health_checks] = lambda: list(checks)


async def test_all_components_are_healthy(app: FastAPI, client: AsyncClient) -> None:
    use_checks(app, FakeCheck("postgres", "17.2"), FakeCheck("redis", "7.4"))

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    versions = {component["name"]: component["version"] for component in body["components"]}
    assert versions == {"postgres": "17.2", "redis": "7.4"}
    assert all(component["latency_ms"] >= 0 for component in body["components"])


async def test_broken_component_turns_service_degraded(app: FastAPI, client: AsyncClient) -> None:
    use_checks(app, FakeCheck("postgres"), FakeCheck("redis", error=ConnectionError("нет связи")))

    response = await client.get("/api/v1/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    statuses = {component["name"]: component["status"] for component in body["components"]}
    assert statuses == {"postgres": "ok", "redis": "error"}
    assert body["components"][1]["error"] == "ConnectionError"


async def test_slow_component_is_cut_by_timeout(app: FastAPI, client: AsyncClient) -> None:
    use_checks(app, FakeCheck("postgres", delay=5))

    response = await client.get("/api/v1/health")

    assert response.status_code == 503
    component = response.json()["components"][0]
    assert component["error"] == "превышено время ожидания"
    assert component["latency_ms"] < 2000


async def test_checks_run_concurrently() -> None:
    checks = [FakeCheck(f"service-{index}", delay=0.4) for index in range(3)]

    started = time.perf_counter()
    reports = await run_checks(checks, limit_seconds=2)
    elapsed = time.perf_counter() - started

    assert [report.status for report in reports] == ["ok", "ok", "ok"]
    assert elapsed < 0.9


async def test_unreachable_database_is_reported(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")

    assert response.status_code == 503
    postgres = response.json()["components"][0]
    assert postgres["name"] == "postgres"
    assert postgres["status"] == "error"
    assert postgres["version"] is None
